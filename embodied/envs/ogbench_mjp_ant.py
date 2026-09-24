import functools
import re

import numpy as np


_NAME = re.compile(
    r'^(?P<visual>visual-)?mjp-antmaze_'
    r'(?P<layout>avoid_obstacle_(?:3|5)|u_trap_(?:3|5))'
    r'(?P<term>_term)?$')

_EPISODE_LENGTH = 500
_MAZE_UNIT = 5.0
_MAZE_HEIGHT = 0.9  # OGBench multiplier: 0.9 * 5.0 = 4.5 m.
_SUCCESS_THRESHOLD = 1.0
_HEALTHY_Z_RANGE = (0.15, 1.2)


def make_mjp_antmaze(name):
  """Build an OGBench AntMaze with the active MJP task semantics.

  The ``mjp-`` marker deliberately keeps this task family separate from the
  native-reward compact tasks implemented in ``ogbench_maze.py``.
  """
  match = _NAME.fullmatch(name)
  if not match:
    return None

  import gymnasium

  visual = bool(match.group('visual'))
  layout_name = match.group('layout')
  terminate_when_unhealthy = bool(match.group('term'))
  env_class = _mjp_antmaze_class(layout_name, terminate_when_unhealthy)
  kwargs = dict(
      maze_type='arena',
      maze_unit=_MAZE_UNIT,
      maze_height=_MAZE_HEIGHT,
      terminate_at_goal=True,
      success_timing='post',
      reward_task_id=1,
      add_noise_to_goal=False,
  )
  if visual:
    kwargs.update(
        ob_type='pixels',
        render_mode='rgb_array',
        width=64,
        height=64,
        camera_name='back',
    )
  env = env_class(**kwargs)
  return gymnasium.wrappers.TimeLimit(
      env, max_episode_steps=_EPISODE_LENGTH)


@functools.lru_cache(None)
def _mjp_antmaze_class(layout_name, terminate_when_unhealthy):
  from embodied.envs.ogbench_maze import (
      _base_maze_class,
      _convert_layout,
      _resolve_layout,
  )

  base_class = _base_maze_class('ant')
  layout = _resolve_layout('antmaze', layout_name)
  maze_map, init_ij, goal_ij = _convert_layout(layout)

  class MJPCompatibleAntMaze(base_class):

    _mjp_compatible = True
    _embodied_info_keys = (
        'log_dist',
        'log_rew_speed',
        'log_rew_healthy',
        'log_rew_ctrl',
        'log_rew_contact',
        'log_rew_success',
    )

    def update_tree(self, tree):
      # Use the exact MJP topology and positive-coordinate convention. This is
      # isolated to the compatibility task; native compact OGBench mazes keep
      # their original 4 m blocks and coordinate offset.
      self.maze_map = maze_map.copy()
      self._teleport_info = None
      self._offset_x = 0.0
      self._offset_y = 0.0
      super().update_tree(tree)

      # MJP overlaps adjacent 5 m wall cells slightly to eliminate collision
      # seams. OGBench otherwise creates exactly half-cell wall extents.
      wall_half = 0.55 * self._maze_unit
      for i, row in enumerate(self.maze_map):
        for j, occupied in enumerate(row):
          if occupied:
            wall = tree.find(f'.//geom[@name="block_{i}_{j}"]')
            if wall is not None:
              wall.set(
                  'size',
                  f'{wall_half} {wall_half} '
                  f'{self._maze_height / 2 * self._maze_unit}')

      rows, cols = self.maze_map.shape
      floor = tree.find('.//geom[@name="floor"]')
      if floor is not None:
        center_x = (cols - 1) * self._maze_unit / 2
        center_y = (rows - 1) * self._maze_unit / 2
        half_x = center_x + 15.0
        half_y = center_y + 15.0
        floor.set('pos', f'{center_x} {center_y} 0')
        floor.set('size', f'{half_x} {half_y} 0.2')
      return tree

    def set_tasks(self):
      self.task_infos = [dict(
          task_name=layout_name,
          init_ij=init_ij,
          init_xy=self.ij_to_xy(init_ij),
          goal_ij=goal_ij,
          goal_xy=self.ij_to_xy(goal_ij),
      )]
      if self._reward_task_id == 0:
        self._reward_task_id = 1

    def add_noise(self, xy):
      # The MJP task fixes the root XY at the S cell while retaining the
      # native ant reset noise for the remaining qpos and qvel entries.
      return tuple(xy)

    def compute_success(self):
      return bool(
          np.linalg.norm(self.get_xy() - self.cur_goal_xy)
          < _SUCCESS_THRESHOLD)

    def reset(self, *args, **kwargs):
      ob, info = super().reset(*args, **kwargs)
      info.update({
          'success': 0.0,
          'log_dist': np.float32(
              np.linalg.norm(self.get_xy() - self.cur_goal_xy)),
          'log_rew_speed': np.float32(0.0),
          'log_rew_healthy': np.float32(0.0),
          'log_rew_ctrl': np.float32(0.0),
          'log_rew_contact': np.float32(0.0),
          'log_rew_success': np.float32(0.0),
      })
      return ob, info

    def step(self, action):
      action = np.clip(np.asarray(action), -1.0, 1.0)
      ob, _, _, truncated, info = super().step(action)

      qpos = self.data.qpos.copy()
      qvel = self.data.qvel.copy()
      xy = qpos[:2]
      v_xy = qvel[:2]
      to_goal = np.asarray(self.cur_goal_xy) - xy
      distance = float(np.linalg.norm(to_goal))
      goal_direction = to_goal / max(distance, 1e-6)
      speed_reward = float(np.dot(v_xy, goal_direction))

      healthy = _HEALTHY_Z_RANGE[0] <= qpos[2] <= _HEALTHY_Z_RANGE[1]
      healthy_reward = -0.5 * float(not healthy)
      control_reward = -0.1 * float(np.sum(np.square(action)))
      contact_reward = -5e-4 * float(np.sum(np.square(
          np.clip(self.data.cfrc_ext, -1.0, 1.0))))
      success = distance < _SUCCESS_THRESHOLD
      success_reward = float(5 * _EPISODE_LENGTH) * float(success)
      reward = (
          speed_reward + healthy_reward + control_reward + contact_reward
          + success_reward)

      finite = np.all(np.isfinite(qpos)) and np.all(np.isfinite(qvel))
      if not finite:
        ob = np.nan_to_num(np.asarray(ob), copy=False)
        reward = 0.0
        distance = float(np.nan_to_num(distance))
        speed_reward = float(np.nan_to_num(speed_reward))
        contact_reward = float(np.nan_to_num(contact_reward))
      terminated = bool(
          success or not finite
          or (terminate_when_unhealthy and not healthy))
      info.update({
          'success': float(success),
          'log_dist': np.float32(distance),
          'log_rew_speed': np.float32(speed_reward),
          'log_rew_healthy': np.float32(healthy_reward),
          'log_rew_ctrl': np.float32(control_reward),
          'log_rew_contact': np.float32(contact_reward),
          'log_rew_success': np.float32(success_reward),
      })
      return ob, np.float32(reward), terminated, truncated, info

  suffix = 'Term' if terminate_when_unhealthy else ''
  MJPCompatibleAntMaze.__name__ = (
      f'OGBenchMJP{layout_name.title()}{suffix}')
  return MJPCompatibleAntMaze
