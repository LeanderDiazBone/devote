"""State-setting adapters for policy evaluation, sharing production dynamics."""
from functools import lru_cache

import numpy as np


def make_eval_env(task, action_repeat=1, seed=0, max_step_limit=0, render=True,
                  config=None):
    if task.startswith('dmc_'):
        from experiments.dmc_anchors_helper import make_eval_dmc
        return make_eval_dmc(task, action_repeat, seed, max_step_limit, render)
    if task == 'mjp_pointmaze_corridors':
        settings = config.env.mjp if config is not None else {}
        if config is not None and (config.wrapper.action_cost or config.wrapper.discretize
                                   or config.wrapper.dynamics_complexity.dims or settings.get('image')):
            raise ValueError('Point-maze evaluation requires unmodified state inputs and continuous actions, with action_cost=0.')
        return PointMazeEval(
            seed=seed, length=max_step_limit or int(settings.get('length', 1000)),
            reward_scale=float(settings.get('reward_scale', 0.0)))
    raise ValueError(f'No policy-evaluation adapter for {task!r}')


def reset_env(env):
    return env.step({k: np.zeros(s.shape, s.dtype) if k != 'reset' else np.bool_(True)
                     for k, s in env.act_space.items()})


def obs_at(env, qpos, qvel, natural_reset=False):
    if isinstance(env, PointMazeEval):
        obs = reset_env(env)
        return obs if natural_reset else env.set_state(qpos, qvel)
    from experiments.dmc_anchors_helper import obs_at as dmc_obs_at
    return dmc_obs_at(env, qpos, qvel, natural_reset)


def physics_state(env):
    if isinstance(env, PointMazeEval):
        return np.asarray(env.state.data.qpos).copy(), np.asarray(env.state.data.qvel).copy()
    return env._dmenv.physics.data.qpos.copy(), env._dmenv.physics.data.qvel.copy()


def render_frame(env, size):
    if isinstance(env, PointMazeEval):
        return env.render(size)
    return env._dmenv.physics.render(height=size, width=size, camera_id=0)


@lru_cache(maxsize=4)
def _point_dynamics(length):
    # Share the compiled dynamics among MC workers; states and RNGs stay local.
    import jax
    from embodied.envs.custom_envs.locomotion.point_maze_env import PointMaze
    env = PointMaze('corridors', config_overrides={'episode_length': length})
    return env, jax.jit(env.reset), jax.jit(env.step)


class PointMazeEval:
    """Unbatched adapter to the exact MJX point-maze transition function.

    Production custom point-maze step applies one displacement per agent step;
    BatchedMujocoPlayground does not repeat this custom transition with `repeat`.
    Corridor walls block motion, success does not terminate, and only the time
    limit ends an episode. The goal remains the production layout's fixed goal.
    """

    def __init__(self, seed=0, length=1000, reward_scale=0.0):
        import jax
        import embodied
        self.inner, self._reset, self._step = _point_dynamics(int(length))
        self.key = jax.random.PRNGKey(seed)
        self.reward_scale = reward_scale
        self.state = None
        self.renderer = None
        self.obs_space = dict(state=embodied.Space(np.float32, (6,)),
                              reward=embodied.Space(np.float32),
                              **{k: embodied.Space(bool) for k in ('is_first', 'is_last', 'is_terminal')})
        self.act_space = dict(action=embodied.Space(np.float32, (2,), -1, 1),
                              reset=embodied.Space(bool))

    def _obs(self, first=False):
        return dict(state=np.asarray(self.state.obs, np.float32),
                    reward=np.float32(float(self.state.reward) * self.reward_scale),
                    is_first=np.bool_(first), is_last=np.bool_(self.state.done),
                    is_terminal=np.bool_(False))

    def step(self, action):
        import jax
        import jax.numpy as jnp
        if self.state is None or bool(action['reset']) or bool(self.state.done):
            self.key, key = jax.random.split(self.key)
            self.state = self._reset(key)
            return self._obs(first=True)
        self.state = self._step(self.state, jnp.asarray(action['action'], jnp.float32))
        return self._obs()

    def set_state(self, qpos, qvel):
        import jax.numpy as jnp
        if self.state is None:
            reset_env(self)
        qpos, qvel = np.asarray(qpos), np.asarray(qvel)
        if qpos.shape != (2,) or qvel.shape != (2,) or not np.isfinite([qpos, qvel]).all():
            raise ValueError('Point-maze states require finite xy and velocity pairs.')
        if np.any(qvel != 0):
            raise ValueError('Point-maze velocity is always zero; velocity anchors are not reachable.')
        import mujoco
        model = self.inner.mj_model
        for i in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ''
            if name.startswith('maze_wall_'):
                if np.all(np.abs(qpos - model.geom_pos[i, :2]) <= model.geom_size[i, :2] + .35):
                    raise ValueError(f'Point-maze anchor intersects a wall: {qpos}')
        data = self.state.data.replace(qpos=jnp.asarray(qpos, jnp.float32),
                                       qvel=jnp.asarray(qvel, jnp.float32))
        self.state = self.state.replace(data=data, obs=self.inner._get_obs(data, self.key))
        return self._obs(first=True)

    def get_coverage_geometry(self, **kwargs):
        return self.inner.get_coverage_geometry(**kwargs)

    def render(self, size):
        import mujoco
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.inner.mj_model, height=size, width=size)
            self.render_data = mujoco.MjData(self.inner.mj_model)
        self.render_data.qpos[:] = np.asarray(self.state.data.qpos)
        self.render_data.qvel[:] = np.asarray(self.state.data.qvel)
        mujoco.mj_forward(self.inner.mj_model, self.render_data)
        self.renderer.update_scene(self.render_data, camera=0)
        return self.renderer.render().copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
