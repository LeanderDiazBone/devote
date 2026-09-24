import os
import copy
import functools
from functools import partial
from typing import List, Tuple, Callable

import jax
import mujoco
import numpy as np
import jax.numpy as jnp
from mujoco import mjx
from mujoco_playground import registry
from mujoco_playground import wrapper

import embodied

try:
    from embodied.envs.custom_envs.manipulation import pick_cartesian_env
    TASK_TO_ENV = {
        "pick_cartesian": pick_cartesian_env.PandaPickCubeCartesian
    }
    TASK_TO_DR_FN = {
        "pick_cartesian": pick_cartesian_env.domain_randomize
    }
except ImportError:
    TASK_TO_ENV = {}
    TASK_TO_DR_FN = {}
    print(f"[ERROR] Unable to load custom mujoco playground environments...")


def tree_where(mask, a, b):
    """Select PyTree elements from `a` or `b` depending on `mask`."""
    def _where(l, r):
        m = mask
        while m.ndim < l.ndim:
            m = m[..., None]  # expand mask dimension to match the shape
        return jnp.where(m, l, r)
    return jax.tree.map(_where, a, b)


def create_renderers(
    base_mj_model: mujoco.MjModel,
    mjx_model_v: mjx.Model,
    in_axes: mjx.Model,
    num_renderers: int,
    height: int,
    width: int
) -> List[mujoco.Renderer]:
    """Create renderers with for the various domain-randomized environments.
        Works by converting `mjx_models` produced by the domain randomization
        function into `mj_models` that can be used by the mujoco renderer."""
    renderers = []
    for i in range(num_renderers):
        # deepcopy of the base mj_model
        mj_model = copy.deepcopy(base_mj_model)

        # transfer fields from the `mjx_model` to the `mj_model` based on `in_axes`
        for field_name in dir(in_axes):
            if field_name.startswith('_'):
                continue
            axis = getattr(in_axes, field_name, None)

            # check if this field was vmapped (it should hold: axis == 0)
            is_vmapped = False
            if axis is not None:
                try:
                    is_vmapped = (axis == 0) if np.isscalar(axis) else False
                except:
                    is_vmapped = False

            if is_vmapped:
                mjx_value = getattr(mjx_model_v, field_name)

                # ensure the value is an array, and get the i-th element which
                #   corresponds to the i-th environment
                try:
                    env_value = np.array(mjx_value[i])
                except (TypeError, IndexError):
                    # Field is not batched (scalar or incompatible), skip it
                    continue

                # set the value in the mj_model if the field exists
                if hasattr(mj_model, field_name):
                    mj_field = getattr(mj_model, field_name)
                    if isinstance(mj_field, np.ndarray):
                        try:
                            # reshape env_value to match mj_field's shape
                            mj_field[:] = env_value.reshape(mj_field.shape)
                        except (ValueError, AttributeError):
                            # shape mismatch or other issue, skip this field
                            continue
                    else:
                        setattr(mj_model, field_name, env_value)

        # create a renderer with this model
        renderer = mujoco.Renderer(mj_model, height=height, width=width)
        renderers.append(renderer)

    return renderers


class DomainRandomizedBatchedMujocoPlayground(embodied.Env):

    def __init__(
        self,
        env,
        num_envs: int,
        dr_fn: Callable = None,
        repeat: int = 1,
        size: Tuple[int, int] = (64, 64),
        image: bool = False,
        log_image: bool = False,
        greyscale: bool = False,
        camera: int = -1,
        act_key: str = "action",
        obs_key: str = "state",
        length: int = 1_000,
        seed: int = 0
    ):
        if "MUJOCO_GL" not in os.environ:
            os.environ["MUJOCO_GL"] = "egl"

        self.num_envs = num_envs
        self._repeat = repeat
        self._size = size
        self._image = image
        self._log_image = log_image
        self._greyscale = greyscale
        self._camera = camera
        self._act_key = act_key
        self._obs_key = obs_key
        self._episode_length = length
        self._img_key = "image" if image else ("log_image" if log_image else None)

        # get the env and the (partial) domain randomization function
        self._dr_fn = dr_fn
        self._partial_dr_fn = dr_fn
        if isinstance(env, str):
            if env in TASK_TO_ENV:
                self._env = TASK_TO_ENV[env]()
                self._dr_fn = TASK_TO_DR_FN.get(env, None)
                # default kwarg "num_worlds"
                self._partial_dr_fn = partial(self._dr_fn, num_worlds=num_envs)
            elif env in registry.ALL_ENVS:
                env_cfg = registry.get_default_config(env)
                config_overrides = {'action_repeat': repeat, 'episode_length': length}  # TODO: check if `length` can be used only here
                self._env = registry.load(env, config=env_cfg, config_overrides=config_overrides)
                self._dr_fn = registry.get_domain_randomizer(env)
                rng = jax.random.split(jax.random.PRNGKey(seed), num_envs)
                # default kwarg "rng"
                self._partial_dr_fn = partial(self._dr_fn, rng=rng)
            else:
                raise ValueError(f"Environment {env} not found. Available custom environments: {TASK_TO_ENV.keys()}. "
                                 f"Available mujoco playground environments: {registry.ALL_ENVS}")
        if self._dr_fn is None:
            raise ValueError(f"No domain randomization function found for environment {env}")

        # store the observation space _before_ we wrap/jit the env
        self._obs_space = self._get_obs_space(self._env)

        # create the environments with domain randomization
        self._env = wrapper.wrap_for_brax_training(
            self._env,
            vision=False,  # `True` will use Madrona; we use a Mujoco EGL renderer
            num_vision_envs=num_envs,
            episode_length=length,
            action_repeat=repeat,
            randomization_fn=self._partial_dr_fn,
        )

        # random seed
        self._seed = seed
        self._key = jax.random.PRNGKey(seed)

        # setup the renderer (if using vision)
        self._renderers = None
        if self._image or self._log_image:
            h, w = self._size
            num_renderers = num_envs if image else min(4, num_envs)
            self._renderers = create_renderers(
                base_mj_model=self._env.unwrapped._mj_model,
                mjx_model_v=self._env._mjx_model_v,
                in_axes=self._env._in_axes,
                num_renderers=num_renderers,
                height=h,
                width=w
            )
            print(f"Created {len(self._renderers)} renderers.")

        # batched-env jitted functions
        self._jit_reset = jax.jit(self._env.reset)
        self._jit_step = jax.jit(self._env.step)

        # batched tree_select function
        self._tree_where = jax.jit(tree_where)

        # initial state + number of steps per env
        self._key, rng = jax.random.split(self._key)
        rngs = jax.random.split(rng, self.num_envs)
        self._states = self._jit_reset(rngs)
        self._env_steps = jnp.zeros((self.num_envs,), dtype=jnp.int32)
        self._done = jnp.zeros((self.num_envs,), dtype=jnp.bool)

        # warmup step function
        print("Warming up the jax step function...")
        _, rng = jax.random.split(self._key)
        dummy_action = jax.random.uniform(rng, (self.num_envs, self._env.action_size), minval=-1.0, maxval=1.0)
        states = self._jit_step(self._states, dummy_action)
        # run again to warum up with "stepped" states (instead of "resetted" states)
        states = self._jit_step(states, dummy_action)
        jax.block_until_ready(states.reward)
        print("Done warming up the jax step function.")

    def _get_obs_space(self, env):
        # infer state size
        if isinstance(env.observation_size, int):
            obs_space = embodied.Space(dtype=np.float32, shape=(env.observation_size,))
        elif isinstance(env.observation_size, tuple):
            obs_space = embodied.Space(dtype=np.float32, shape=env.observation_size)
        elif env.observation_size.get(self._obs_key):
            size = env.observation_size.get(self._obs_key)
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
            spaces[self._img_key] = embodied.Space(np.uint8, self._size + (nc,))

        return spaces

    @functools.cached_property
    def obs_space(self):
        return self._obs_space

    @functools.cached_property
    def act_space(self):
        env = self._env.unwrapped
        return {
            "reset": embodied.Space(bool),
            self._act_key: embodied.Space(np.float32, shape=(env.action_size,)),
        }

    def step(self, actions):
        # determine which envs need to be reset
        resets = jnp.asarray(actions["reset"], dtype=bool)
        reset_mask = resets | self._done

        # define rngs for each env (if they need to be reset)
        self._key, rng = jax.random.split(self._key)
        rngs = jax.random.split(rng, self.num_envs)

        # reset all envs
        reset_states = self._jit_reset(rngs)
        reset_steps = jnp.zeros((self.num_envs,), dtype=jnp.int32)
        self._states = self._tree_where(reset_mask, reset_states, self._states)

        # step all envs
        actions_jnp = jnp.asarray(actions[self._act_key], dtype=jnp.float32)
        self._states = self._jit_step(self._states, actions_jnp)
        step_steps = self._env_steps + 1

        # reset back the ones that were done
        self._states = self._tree_where(reset_mask, reset_states, self._states)
        self._env_steps = jnp.where(reset_mask, reset_steps, step_steps)

        # get the done flags
        self._done = self._states.done.astype(jnp.bool) | (self._env_steps >= self._episode_length)

        # build numpy observation
        obs = self._obs(reset_mask, self._done)
        return obs

    def _obs(self, is_first, is_last):
        # load observations, rewards, dones, infos
        observations, rewards, dones, _ = self._decompose_state()
        # convert to numpy array
        obs = {
            self._obs_key: np.array(observations),
            "reward": np.array(rewards),
            "is_first": np.array(is_first),
            "is_last": np.array(is_last),
            "is_terminal": np.array(dones)
        }
        if self._image or self._log_image:
            obs[self._img_key] = self._render(self._states)
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

        # loop through env/renderer pairs
        batch_images = []
        num_renders = min(self.num_envs, len(self._renderers))
        for i in range(num_renders):
            renderer = self._renderers[i]
            mj_model = renderer.model

            # create or reuse MjData for this model
            if not hasattr(self, '_mj_datas'):
                self._mj_datas = [mujoco.MjData(mj_model) for mj_model in [r.model for r in self._renderers]]
            mj_data = self._mj_datas[i]

            # 1. bring state into this environment's MjData buffer
            mj_data.qpos[:] = all_qpos[i]
            mj_data.qvel[:] = all_qvel[i]

            # 2. FK
            mujoco.mj_forward(mj_model, mj_data)

            # 3. update scene
            if self._camera is None:
                renderer.update_scene(mj_data)
            else:
                renderer.update_scene(mj_data, camera=self._camera)

            # 4. render
            rgb = renderer.render()
            if self._greyscale:
                gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2])
                batch_images.append(gray[:, :, None])
            else:
                batch_images.append(rgb)

        # ugly but necessary: repeat the last image to force the batch shape to 
        #   for cases where we use state-based envs but fewer renderers (e.g., logging)
        while len(batch_images) < self.num_envs:
            batch_images.append(batch_images[-1])

        return np.stack(batch_images)

    def close(self):
        if self._renderers is not None:
            try:
                [renderer.close() for renderer in self._renderers]
            except Exception:
                pass
            self._renderers = None
