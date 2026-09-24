import time
from typing import Any, Dict

import embodied
import numpy as np


class BSuite(embodied.Env):

  def __init__(self, task, reward_scale: float = 1.0):
    print(
        'Warning: BSuite result logging is stateful and therefore training ' +
        'runs cannot be interrupted or restarted.')
    np.int = int  # Patch deprecated Numpy alias used inside BSuite.
    import bsuite
    from . import from_dm
    self._task = task
    if '/' not in task:
      task = f'{task}/0'
    self._task = task
    env = bsuite.load_from_id(task)
    self._raw_obs_shape = tuple(env.observation_spec().shape)
    self.num_episodes = 0
    self.max_episodes = env.bsuite_num_episodes
    self.exit_after = None
    env = from_dm.FromDM(env)
    env = embodied.wrappers.ForceDtypes(env)
    env = embodied.wrappers.FlattenTwoDimObs(env)
    self.env = env
    self._reward_scale = float(reward_scale)

  @property
  def obs_space(self):
    return self.env.obs_space

  @property
  def act_space(self):
    return self.env.act_space

  def step(self, action):
    obs = self.env.step(action)
    if self._reward_scale != 1.0:
      obs['reward'] = (obs['reward'] * self._reward_scale).astype(obs['reward'].dtype)
    if obs['is_last']:
      self.num_episodes += 1
    if self.num_episodes >= self.max_episodes:
      # After reaching the target number of episodes, continue running for 10
      # minutes to make sure logs are flushed and then raise an exception to
      # terminate the program.
      if not self.exit_after:
        self.exit_after = time.time() + 600
      if time.time() > self.exit_after:
        raise RuntimeError('BSuite run complete')
    return obs

  def get_coverage_geometry(self, bins_per_cell: int = 1) -> Dict[str, Any]:
    """Coverage geometry over the deep_sea triangular grid.

    Reachable cells satisfy ``col <= row`` on the NxN grid. ``bins_per_cell``
    is accepted for interface compatibility but ignored: deep_sea cells are
    already discrete.
    """
    del bins_per_cell
    if not str(self._task).startswith('deep_sea/'):
      raise NotImplementedError(
          'BSuite coverage geometry is only implemented for deep_sea tasks.')
    if len(self._raw_obs_shape) != 2 or self._raw_obs_shape[0] != self._raw_obs_shape[1]:
      raise ValueError(
          'DeepSea coverage expects a square 2D observation, got '
          f'{self._raw_obs_shape}.')
    size = int(self._raw_obs_shape[0])
    valid_mask = np.tri(size, dtype=bool)

    def project(tran, _size=size):
      obs = np.asarray(tran['observation'])
      flat = obs.reshape(-1, _size * _size)
      idx = flat.argmax(axis=-1)
      return np.stack([idx // _size, idx % _size], axis=-1)

    return dict(
        bounds=np.array([[0.0, float(size)], [0.0, float(size)]], dtype=np.float64),
        bins=(size, size),
        axis_names=('row', 'col'),
        valid_mask=valid_mask,
        project=project,
    )

  def get_state_table(self) -> Dict[str, Any]:
    if not str(self._task).startswith('deep_sea/'):
      raise NotImplementedError(
          'BSuite state tables are only implemented for deep_sea tasks.')
    if len(self._raw_obs_shape) != 2 or self._raw_obs_shape[0] != self._raw_obs_shape[1]:
      raise ValueError(
          'DeepSea state table expects a square 2D observation, got '
          f'{self._raw_obs_shape}.')

    size = int(self._raw_obs_shape[0])
    states = np.eye(size * size, dtype=np.float32)
    return dict(
        obs={'observation': states},
        obs_col_names={
            'observation': [
                f'cell_{row}_{col}'
                for row in range(size)
                for col in range(size)
            ],
        },
        grid_shape=np.array([size, size], dtype=np.int32),
    )
