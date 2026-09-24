import os
import functools
from typing import Tuple

import jax
import mujoco
import numpy as np
import jax.numpy as jnp
from mujoco_playground import registry

import embodied

try:
    from embodied.envs.custom_envs.manipulation import pick_cartesian_env
    from embodied.envs.custom_envs.manipulation_franka import panda_env
    from embodied.envs.custom_envs.locomotion import ant_maze_env
    from embodied.envs.custom_envs.locomotion import point_maze_env
    from embodied.envs.custom_envs.locomotion import swimmer_maze_env
    TASK_TO_ENV = {
        "pickcartesian": pick_cartesian_env.PandaPickCubeCartesian,
        "panda": panda_env.PandaEnv,
        "antmaze": ant_maze_env.AntMaze,
        "swimmermaze": swimmer_maze_env.SwimmerMaze,
        "pointmaze": point_maze_env.PointMaze,
        "randompointmaze": point_maze_env.RandomPointMaze,
    }
except ImportError:
    TASK_TO_ENV = {}
    print(f"[ERROR] Unable to load custom mujoco playground environments...")


class BatchedMujocoPlayground(embodied.Env):
    DEFAULT_CAMERAS = {
        "antmaze": 0,  # track camera defined in ant.xml
        "swimmermaze": 0,  # track camera defined in swimmer.xml
        "pointmaze": 0,
        "randompointmaze": 0,
    }

    def __init__(
        self,
        env,
        num_envs: int,
        repeat: int = 1,
        size: Tuple[int, int] = (64, 64),
        log_size: Tuple[int, int] = None,
        image: bool = False,
        log_image: bool = False,
        greyscale: bool = False,
        camera: int = -1,
        act_key: str = "action",
        obs_key: str = "state",
        length: int = 1_000,
        seed: int = 0,
        reward_scale: float = 1.0,
        action_scale: float = 0.0,
    ):
        if "MUJOCO_GL" not in os.environ:
            os.environ["MUJOCO_GL"] = "egl"

        if isinstance(env, str) and "_" in env:
            env, subtask = env.split("_", 1)
        else:
            subtask = None

        self.num_envs = num_envs
        self._repeat = repeat
        self._size = tuple(size)
        # log_size only affects the log_image rendering path (image=False,
        # log_image=True); falls back to `size` if unset.
        self._log_size = tuple(log_size) if log_size else self._size
        self._image = image
        self._log_image = log_image
        self._greyscale = greyscale
        env_name = env if isinstance(env, str) else None
        if camera == -1 and env_name in self.DEFAULT_CAMERAS:
            camera = self.DEFAULT_CAMERAS[env_name]
        self._camera = camera
        self._act_key = act_key
        self._obs_key = obs_key
        self._episode_length = length
        self._reward_scale = np.float32(reward_scale)

        # get the env
        if isinstance(env, str):
            if env in TASK_TO_ENV:
                # Only panda exposes action_scale; pass it through when the
                # user set a non-zero override (0 == use the env's default).
                custom_overrides = {}
                if env == "panda" and action_scale != 0.0:
                    custom_overrides["action_scale"] = float(action_scale)
                if custom_overrides:
                    self._env = TASK_TO_ENV[env](
                        subtask, config_overrides=custom_overrides)
                else:
                    self._env = TASK_TO_ENV[env](subtask)
            elif env in registry.ALL_ENVS:
                env_cfg = registry.get_default_config(env)
                config_overrides = {'action_repeat': repeat, 'episode_length': length}
                self._env = registry.load(env, config=env_cfg, config_overrides=config_overrides)
            else:
                raise ValueError(f"Environment {env} not found. Available custom environments: {TASK_TO_ENV.keys()}. Available mujoco playground environments: {registry.ALL_ENVS}")

        # random seed
        self._seed = seed
        self._key = jax.random.PRNGKey(seed)

        # setup the renderer (if using vision)
        self._mj_model = self._env.mj_model
        if isinstance(self._camera, (int, np.integer)) and self._camera >= self._mj_model.ncam:
            self._camera = -1
        self._mj_data = mujoco.MjData(self._mj_model)
        self._renderer = None
        if self._image or self._log_image:
            self._img_key = "image" if self._image else "log_image"
            # Agent input uses `size`; log-only rendering uses `log_size`.
            self._render_size = self._size if self._image else self._log_size
            h, w = self._render_size
            self._renderer = mujoco.Renderer(self._mj_model, height=h, width=w)

        # single-env jitted functions
        self._reset_one = jax.jit(self._env.reset)
        self._step_one_dyn = jax.jit(self._env.step)

        # batched-env step function
        self._jit_step_batched = jax.jit(jax.vmap(self._step_one, in_axes=(0, 0, 0, 0, 0)))

        # initial state + number of steps per env
        self._key, rng = jax.random.split(self._key)
        rngs = jax.random.split(rng, self.num_envs)
        self._states = jax.jit(jax.vmap(self._reset_one))(rngs)
        self._env_steps = jnp.zeros((self.num_envs,), dtype=jnp.int32)
        self._done = jnp.ones((self.num_envs, ), dtype=jnp.bool)

        # Discover per-step log metrics emitted by the inner env. Any
        # metric whose key starts with ``log_`` is forwarded to obs so the
        # training loop can aggregate it (e.g. ``log_keys_sum: '^log_rew_'``
        # produces per-episode returns for each reward component).
        metrics = getattr(self._states, 'metrics', None) or {}
        self._log_metric_keys = sorted(
            k for k in metrics if isinstance(k, str) and k.startswith('log_'))

        # warmup step function
        print("Warming up the jax step function...")
        dummy_action = jnp.zeros((self.num_envs, self._env.action_size), dtype=jnp.float32)
        dummy_reset = jnp.zeros((self.num_envs,), dtype=bool)
        _, rng = jax.random.split(self._key)
        rngs = jax.random.split(rng, self.num_envs)
        states, _, _ = self._jit_step_batched(
            dummy_reset, rngs, self._states, dummy_action, self._env_steps
        )
        jax.block_until_ready(states.reward)
        print("Done warming up the jax step function.")

    @functools.cached_property
    def obs_space(self):

        # infer state size
        if isinstance(self._env.observation_size, int):
            obs_space = embodied.Space(dtype=np.float32, shape=(self._env.observation_size,))
        elif isinstance(self._env.observation_size, tuple):
            obs_space = embodied.Space(dtype=np.float32, shape=self._env.observation_size)
        elif self._env.observation_size.get(self._obs_key):
            size = self._env.observation_size.get(self._obs_key)
            obs_space = embodied.Space(dtype=np.float32, shape=size)
        else:
            raise NotImplementedError("Unknown observation size type for mujoco playground env")

        # define the spaces
        spaces = {
            "reward": embodied.Space(np.float32),
            "is_first": embodied.Space(bool),
            "is_last": embodied.Space(bool),
            "is_terminal": embodied.Space(bool),
            self._obs_key: obs_space,
        }

        # add image if we're doing vision
        if self._image or self._log_image:
            nc = 1 if self._greyscale else 3
            spaces[self._img_key] = embodied.Space(np.uint8, self._render_size + (nc,))

        # Forward per-step ``log_*`` metrics from the inner env. The shape is
        # ``(1,)`` (not ``()``) so ``ExpandScalars`` leaves the value alone --
        # we already produce a batched ``(num_envs, 1)`` array. A scalar
        # declaration would be expanded along the wrong axis for batched envs.
        for key in self._log_metric_keys:
            spaces[key] = embodied.Space(np.float32, shape=(1,))

        return spaces

    @functools.cached_property
    def act_space(self):
        return {
            "reset": embodied.Space(bool),
            self._act_key: embodied.Space(np.float32, shape=(self._env.action_size,)),
        }

    def _step_one(self, has_to_reset, rng, state, act, step):

        def reset_fn(rng, state, act, step):
            reset_state = self._reset_one(rng)
            reset_step = jnp.int32(0)
            reset_done = jnp.bool(False)
            return reset_state, reset_step, reset_done

        def step_fn(rng, state, act, step):
            next_state = self._step_one_dyn(state, act)
            next_step = step + jnp.int32(1)
            next_done = next_state.done.astype(jnp.bool) | (next_step >= self._episode_length)
            return next_state, next_step, next_done

        return jax.lax.cond(has_to_reset, reset_fn, step_fn, rng, state, act, step)

    def step(self, actions):
        # determine which envs need to be reset
        resets = jnp.asarray(actions["reset"], dtype=bool)
        reset_mask = resets | self._done

        # define rngs for each env (if they need to be reset)
        self._key, rng = jax.random.split(self._key)
        rngs = jax.random.split(rng, self.num_envs)

        # batched per-env conditional step
        actions_jnp = jnp.asarray(actions[self._act_key], dtype=jnp.float32)
        self._states, self._env_steps, self._done = self._jit_step_batched(
            reset_mask, rngs, self._states, actions_jnp, self._env_steps
        )

        # build numpy observation
        obs = self._obs(reset_mask, self._done)
        return obs

    def _obs(self, is_first, is_last):
        # load observations, rewards, dones, infos
        observations, rewards, dones, _ = self._decompose_state()
        # convert to numpy array
        obs = {
            self._obs_key: np.array(observations),
            "reward": np.array(rewards) * self._reward_scale,
            "is_first": np.array(is_first),
            "is_last": np.array(is_last),
            "is_terminal": np.array(dones)
        }
        if self._image or self._log_image:
            obs[self._img_key] = self._render(self._states)
        if self._log_metric_keys:
            metrics = self._states.metrics
            for key in self._log_metric_keys:
                obs[key] = np.asarray(
                    metrics[key], dtype=np.float32).reshape(self.num_envs, 1)
        return obs

    def _decompose_state(self):
        observations, rewards, dones, infos = \
            self._states.obs, self._states.reward, self._states.done, self._states.info
        if not isinstance(observations, jax.Array):
            observations = observations.get(self._obs_key)
        rewards = rewards.squeeze().astype(jnp.float32)
        dones = dones.squeeze().astype(bool)
        return observations, rewards, dones, infos

    def _render(self, states):
        phys_data = jax.device_get(states.data)

        all_qpos = phys_data.qpos
        all_qvel = phys_data.qvel

        batch_images = []

        # We loop serially. This is the bottleneck, but it is robust.
        for i in range(self.num_envs):
            # 1. Teleport state into the single MjData buffer
            self._mj_data.qpos[:] = all_qpos[i]
            self._mj_data.qvel[:] = all_qvel[i]

            # 2. Forward Kinematics (cheap)
            mujoco.mj_forward(self._mj_model, self._mj_data)

            # 3. Update Scene
            if self._camera is None:
                self._renderer.update_scene(self._mj_data)
            else:
                self._renderer.update_scene(self._mj_data, camera=self._camera)

            # 4. Render
            rgb = self._renderer.render()
            if self._greyscale:
                gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2])
                batch_images.append(gray[:, :, None])
            else:
                batch_images.append(rgb)

        return np.stack(batch_images)

    def close(self):
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None

    # Optional probe methods. All return None when the inner env doesn't
    # implement them, so duck-typed callers can use `is None` uniformly
    # (matches get_coverage_geometry).
    def get_state_table(self, *args, **kwargs):
        if not hasattr(self._env, 'get_state_table'):
            return None
        return self._env.get_state_table(*args, **kwargs)

    def get_action_table(self, *args, **kwargs):
        if not hasattr(self._env, 'get_action_table'):
            return None
        return self._env.get_action_table(*args, **kwargs)

    def get_coverage_geometry(self, *args, **kwargs):
        if not hasattr(self._env, 'get_coverage_geometry'):
            return None
        return self._env.get_coverage_geometry(*args, **kwargs)

    def get_action_grid(self, *args, **kwargs):
        if not hasattr(self._env, 'get_action_grid'):
            return None
        return self._env.get_action_grid(*args, **kwargs)

    def dynamics_complexity_scale(self, obs):
        if not hasattr(self._env, 'dynamics_complexity_scale'):
            return np.float32(1.0)
        return self._env.dynamics_complexity_scale(obs, obs_key=self._obs_key)
