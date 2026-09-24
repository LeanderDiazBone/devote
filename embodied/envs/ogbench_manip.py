import functools
import re

import numpy as np


_NAME = re.compile(
    r'^(?P<visual>visual-)?cube-single_pick_and_place'
    r'(?:_(?P<difficulty>easy|medium|mediumhard|hard))?$')
_GOAL_X = {
    'easy': 0.425,
    'medium': 0.500,
    'mediumhard': 0.550,
    'hard': 0.575,
}


def make_named_manipulation(name):
  """Create an OGBench Cube task from an MJP-style compact name."""
  match = _NAME.fullmatch(name)
  if not match:
    return None

  import gymnasium

  visual = bool(match.group('visual'))
  difficulty = match.group('difficulty') or 'medium'
  env_class = _pick_and_place_class(difficulty)
  kwargs = dict(
      env_type='single',
      permute_blocks=False,
      reward_task_id=1,
      success_timing='pre')
  if visual:
    kwargs.update(
        ob_type='pixels',
        width=64,
        height=64,
        visualize_info=False)
  env = env_class(**kwargs)
  return gymnasium.wrappers.TimeLimit(env, max_episode_steps=200)


@functools.lru_cache(None)
def _pick_and_place_class(difficulty):
  from ogbench.manipspace.envs.cube_env import CubeEnv

  init_xyzs = np.asarray([[0.35, 0.0, 0.02]], dtype=np.float64)
  goal_xyzs = np.asarray(
      [[_GOAL_X[difficulty], 0.0, 0.14]], dtype=np.float64)

  class PickAndPlace(CubeEnv):

    def set_tasks(self):
      self.task_infos = [dict(
          task_name=f'pick_and_place_{difficulty}',
          init_xyzs=init_xyzs.copy(),
          goal_xyzs=goal_xyzs.copy(),
      )]
      if self._reward_task_id == 0:
        self._reward_task_id = 1

  PickAndPlace.__name__ = f'OGBenchPickAndPlace{difficulty.title()}'
  return PickAndPlace
