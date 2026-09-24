import time

import numpy as np

from . import path as pathlib


def state_obs_keys(obs_space):
  """Compute the 1-D floating-point observation keys (matches _state_obs_keys)."""
  skip = {'is_first', 'is_last', 'is_terminal', 'reward', 'cont', 'stepid'}
  return sorted([
      k for k, v in obs_space.items()
      if k not in skip
      and not k.startswith('log_')
      and len(v.shape) == 1
      and np.issubdtype(v.dtype, np.floating)
  ])


class TrajectoryRecorder:
  """Records eval trajectories to disk as .npz files.

  Registers as a driver.on_step() callback. Maintains per-worker episode
  buffers and saves completed episodes to the specified directory.
  """

  def __init__(self, directory, num_envs, obs_keys, act_keys=(), step_counter=None):
    self.directory = pathlib.Path(directory)
    self.directory.mkdir()
    self.num_envs = num_envs
    self.obs_keys = list(obs_keys)
    self.record_keys = self.obs_keys + list(act_keys) + ['reward', 'is_first', 'is_last', 'is_terminal']
    self._buffers = {i: {} for i in range(num_envs)}
    self._step_counter = step_counter  # callable or object with .value / int

  def _get_train_step(self):
    if self._step_counter is None:
      return 0
    if callable(self._step_counter):
      return int(self._step_counter())
    return int(self._step_counter)

  def __call__(self, tran, worker, **kwargs):
    buf = self._buffers[worker]
    if tran['is_first']:
      buf.clear()
      for key in self.record_keys:
        if key in tran:
          buf[key] = [tran[key]]
      buf['train_step'] = [self._get_train_step()]
    else:
      for key in buf:
        if key == 'train_step':
          buf[key].append(self._get_train_step())
        else:
          buf[key].append(tran[key])
    if tran['is_last'] and buf:
      self._save(buf, worker)

  def reset(self, worker=None):
    if worker is None:
      for buf in self._buffers.values():
        buf.clear()
    else:
      self._buffers[worker].clear()

  def _save(self, buf, worker):
    arrays = {k: np.stack(v) for k, v in buf.items()}
    ts = int(time.time() * 1000)
    filename = self.directory / f'ep_{ts}_{worker}.npz'
    np.savez(str(filename), **arrays)
