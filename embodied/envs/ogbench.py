import functools
import os

import embodied
import numpy as np


class OGBench(embodied.Env):

  def __init__(
      self, env, obs_key=None, act_key='action', seed=None,
      reward_scale=1.0):
    # OGBench imports MuJoCo while constructing the environment. Respect an
    # explicitly configured backend but default headless cluster jobs to EGL.
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import ogbench

    from embodied.envs.ogbench_mjp_ant import make_mjp_antmaze
    from embodied.envs.ogbench_mjp_manip import make_mjp_pick_and_place
    from embodied.envs.ogbench_maze import make_named_maze
    from embodied.envs.ogbench_manip import make_named_manipulation
    self._env = make_mjp_antmaze(env)
    if self._env is None:
      self._env = make_mjp_pick_and_place(env)
    if self._env is None:
      self._env = make_named_maze(env, ogbench)
    if self._env is None:
      self._env = make_named_manipulation(env)
    if self._env is None:
      if 'singletask' not in env:
        raise ValueError(
            'OGBench integration currently supports single-task environments '
            f'only; got {env!r}. Choose a single-task or compact named task.')
      self._env = ogbench.make_env_and_datasets(env, env_only=True)
    inner = getattr(self._env, 'unwrapped', self._env)
    self._maze_env = inner if all(hasattr(inner, name) for name in (
        'maze_map', 'ij_to_xy', 'get_xy')) else None
    is_cube = any(
        base.__module__ == 'ogbench.manipspace.envs.cube_env' and
        base.__name__ == 'CubeEnv'
        for base in type(inner).__mro__)
    self._cube_env = inner if (
        is_cube and getattr(inner, '_num_cubes', None) == 1) else None
    self._info_keys = tuple(getattr(inner, '_embodied_info_keys', ()))
    obs_space = self._env.observation_space
    if not all(hasattr(obs_space, name) for name in ('dtype', 'shape', 'low', 'high')):
      raise TypeError(
          'OGBench integration expects a flat Box observation space, got '
          f'{obs_space!r}.')
    if obs_key is None:
      obs_key = 'image' if len(obs_space.shape) == 3 else 'state'
    self._obs_key = obs_key
    self._act_key = act_key
    self._reward_scale = float(reward_scale)
    self._done = True
    self._info = None
    self._random = np.random.RandomState(seed) if seed is not None else None

  @property
  def env(self):
    return self._env

  @property
  def info(self):
    return self._info

  @functools.cached_property
  def obs_space(self):
    spaces = {
        self._obs_key: self._convert(self._env.observation_space),
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
        'log_success': embodied.Space(np.float32),
    }
    if self._maze_env is not None:
      spaces['log_coverage_xy'] = embodied.Space(np.float32, (2,))
    if self._cube_env is not None:
      spaces['log_coverage_xyz'] = embodied.Space(np.float32, (3,))
      spaces['log_d_pinch_cube'] = embodied.Space(np.float32)
    for key in self._info_keys:
      spaces[key] = embodied.Space(np.float32)
    return spaces

  @functools.cached_property
  def act_space(self):
    return {
        self._act_key: self._convert(self._env.action_space),
        'reset': embodied.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self._done:
      kwargs = {}
      if self._random is not None:
        kwargs['seed'] = int(self._random.randint(0, 2 ** 31 - 1))
      obs, self._info = self._env.reset(**kwargs)
      self._done = False
      return self._obs(obs, 0.0, self._info, is_first=True)

    obs, reward, terminated, truncated, self._info = self._env.step(
        action[self._act_key])
    self._done = bool(terminated or truncated)
    return self._obs(
        obs, reward, self._info,
        is_last=self._done,
        is_terminal=bool(terminated))

  def render(self):
    image = self._env.render()
    assert image is not None
    return np.asarray(image)

  def close(self):
    try:
      self._env.close()
    except Exception:
      pass

  def get_coverage_geometry(self, bins_per_cell=1):
    if self._maze_env is None and self._cube_env is None:
      return None
    count = int(bins_per_cell)
    sub = int(round(np.sqrt(count)))
    if sub < 1 or sub * sub != count:
      raise ValueError(
          'coverage_bins_per_cell must be a positive perfect square; '
          f'got {count}.')
    if self._cube_env is not None:
      bounds = np.asarray(
          self._cube_env._workspace_bounds, dtype=np.float64).T

      def project(tran):
        positions = np.asarray(
            tran['log_coverage_xyz']).reshape(-1, 3)
        held = np.asarray(
            tran['log_d_pinch_cube']).reshape(-1) < 0.06
        return positions[held]

      return dict(
          bounds=bounds,
          bins=(12 * sub, 12 * sub, 12 * sub),
          axis_names=('cube_x', 'cube_y', 'cube_z'),
          project=project,
      )
    maze_map = np.asarray(self._maze_env.maze_map)
    rows, cols = maze_map.shape
    first = np.asarray(self._maze_env.ij_to_xy((0, 0)), np.float64)
    last = np.asarray(
        self._maze_env.ij_to_xy((rows - 1, cols - 1)), np.float64)
    maze_unit = float(self._maze_env._maze_unit)
    bounds = np.stack((
        np.minimum(first, last) - maze_unit / 2,
        np.maximum(first, last) + maze_unit / 2,
    ), axis=-1)
    valid_mask = (maze_map == 0).T
    if sub > 1:
      valid_mask = np.kron(
          valid_mask, np.ones((sub, sub), dtype=bool))
    return dict(
        bounds=bounds,
        bins=(cols * sub, rows * sub),
        axis_names=('x', 'y'),
        valid_mask=valid_mask,
        project=lambda tran: np.asarray(tran['log_coverage_xy']),
    )

  def _obs(
      self, obs, reward, info, is_first=False, is_last=False,
      is_terminal=False):
    success = np.asarray(info.get('success', 0.0))
    result = {
        self._obs_key: np.asarray(obs),
        'reward': np.float32(reward * self._reward_scale),
        'is_first': bool(is_first),
        'is_last': bool(is_last),
        'is_terminal': bool(is_terminal),
        'log_success': np.float32(success.max()),
    }
    if self._maze_env is not None:
      result['log_coverage_xy'] = np.asarray(
          self._maze_env.get_xy(), dtype=np.float32)
    if self._cube_env is not None:
      cube_pos = np.asarray(
          self._cube_env.data.joint('object_joint_0').qpos[:3],
          dtype=np.float32)
      pinch_pos = np.asarray(
          self._cube_env.data.site_xpos[self._cube_env._pinch_site_id],
          dtype=np.float32)
      result['log_coverage_xyz'] = cube_pos
      result['log_d_pinch_cube'] = np.float32(
          np.linalg.norm(pinch_pos - cube_pos))
    for key in self._info_keys:
      result[key] = np.float32(np.asarray(info.get(key, 0.0)).max())
    return result

  def _convert(self, space):
    if hasattr(space, 'n'):
      return embodied.Space(np.int32, (), 0, space.n)
    return embodied.Space(space.dtype, space.shape, space.low, space.high)
