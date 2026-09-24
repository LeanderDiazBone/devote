import functools
import re
import xml.etree.ElementTree as ET

import numpy as np


_OFFICIAL_LAYOUTS = frozenset(('medium', 'large', 'giant', 'teleport'))
_FULL_NAME = re.compile(
    r'^(?P<visual>visual-)?(?P<family>pointmaze|antmaze)-'
    r'(?P<layout>[a-z0-9_]+)-navigate-singletask'
    r'(?:-task(?P<task_id>[1-5]))?-v0$')
_COMPACT_NAME = re.compile(
    r'^(?P<visual>visual-)?(?P<family>pointmaze|antmaze)_'
    r'(?P<layout>[a-z0-9_]+)$')


def make_named_maze(name, ogbench):
  """Create an OGBench maze from a full name or an MJP-style compact name.

  Returns None for non-maze OGBench names so the caller can use the regular
  OGBench loader. Compact names always select the single S-to-G task (task 1).
  """
  match = _FULL_NAME.fullmatch(name)
  if not match:
    match = _COMPACT_NAME.fullmatch(name)
  if not match:
    return None

  visual = bool(match.group('visual'))
  family = match.group('family')
  layout_name = match.group('layout')
  task_id = int(match.groupdict().get('task_id') or 1)
  if layout_name in _OFFICIAL_LAYOUTS:
    return _make_official_maze(
        ogbench, family, layout_name, visual, task_id)
  if task_id != 1:
    raise ValueError(
        f'Custom OGBench maze layout {layout_name!r} exposes only task1; '
        f'got task{task_id}.')
  return _make_custom_maze(family, layout_name, visual)


def _make_official_maze(ogbench, family, layout_name, visual, task_id):
  prefix = 'visual-' if visual else ''
  dataset_name = (
      f'{prefix}{family}-{layout_name}-navigate-'
      f'singletask-task{task_id}-v0')
  if visual and family == 'pointmaze':
    # OGBench does not register visual PointMaze IDs, but its PointMaze uses
    # the same pixel observation path as the other locomaze environments.
    # Construct it directly so we can add and select the rear tracking camera
    # that OGBench's Ant has but its Point XML lacks.
    import gymnasium

    env = _visual_point_maze_class()(
        maze_type=layout_name,
        reward_task_id=task_id,
        add_noise_to_goal=False,
        success_timing='pre',
        ob_type='pixels',
        render_mode='rgb_array',
        width=64,
        height=64,
        camera_name='back')
    return gymnasium.wrappers.TimeLimit(env, max_episode_steps=1000)
  return ogbench.make_env_and_datasets(dataset_name, env_only=True)


def _make_custom_maze(family, layout_name, visual):
  import gymnasium

  maze_class = _custom_maze_class(family, layout_name)
  kwargs = dict(
      maze_type='arena',
      reward_task_id=1,
      add_noise_to_goal=False,
      success_timing='pre')
  if visual:
    kwargs.update(
        ob_type='pixels',
        render_mode='rgb_array',
        width=64,
        height=64,
        camera_name='back')
  env = maze_class(**kwargs)
  return gymnasium.wrappers.TimeLimit(env, max_episode_steps=1000)


@functools.lru_cache(None)
def _custom_maze_class(family, layout_name):
  loco_type = {'pointmaze': 'point', 'antmaze': 'ant'}[family]
  base_class = _base_maze_class(loco_type)
  layout = _resolve_layout(family, layout_name)
  maze_map, init_ij, goal_ij = _convert_layout(layout)

  class CustomMaze(base_class):

    def update_tree(self, tree):
      # The parent selects its temporary arena map before calling this method.
      # Replace only that map; all XML, physics, and rendering logic remains
      # OGBench's implementation.
      self.maze_map = maze_map.copy()
      self._teleport_info = None
      super().update_tree(tree)
      if family == 'pointmaze':
        _add_point_back_camera(tree)
      return tree

    def set_tasks(self):
      self.task_infos = [dict(
          task_name='task1',
          init_ij=init_ij,
          init_xy=self.ij_to_xy(init_ij),
          goal_ij=goal_ij,
          goal_xy=self.ij_to_xy(goal_ij),
      )]
      if self._reward_task_id == 0:
        self._reward_task_id = 1

  CustomMaze.__name__ = f'OGBench{family.title()}{layout_name.title()}'
  return CustomMaze


@functools.lru_cache(None)
def _visual_point_maze_class():
  base_class = _base_maze_class('point')

  class VisualPointMaze(base_class):

    def update_tree(self, tree):
      super().update_tree(tree)
      _add_point_back_camera(tree)
      return tree

  VisualPointMaze.__name__ = 'OGBenchVisualPointMaze'
  return VisualPointMaze


def _add_point_back_camera(tree):
  """Give PointMaze the same rear tracking camera as OGBench AntMaze."""
  if tree.find('.//camera[@name="back"]') is not None:
    return
  torso = tree.find('.//body[@name="torso"]')
  if torso is None:
    raise ValueError('OGBench PointMaze XML has no torso body for its camera.')
  ET.SubElement(
      torso,
      'camera',
      name='back',
      pos='0 -2.5 5',
      xyaxes='1 0 0 0 2 1',
      mode='trackcom')


@functools.lru_cache(None)
def _base_maze_class(loco_type):
  # OGBench defines MazeEnv inside this public factory instead of exporting
  # the class. Construct one state-based probe to obtain the generated class,
  # then cache it for all named layouts in this process.
  from ogbench.locomaze.maze import make_maze_env

  probe = make_maze_env(
      loco_env_type=loco_type,
      maze_env_type='maze',
      maze_type='arena',
      reward_task_id=1,
      add_noise_to_goal=False,
      success_timing='pre')
  base_class = type(probe)
  probe.close()
  return base_class


def _resolve_layout(family, name):
  from embodied.envs.custom_envs.locomotion.point_maze_env import (
      BIG_ROOM_MAZE_LAYOUT,
      BIG_SQUARE_MAZE_LAYOUT,
      BIGGEST_ROOM_MAZE_LAYOUT,
      CORRIDORS_MAZE_LAYOUT,
      ESCAPE_MAZE_LAYOUT,
      LARGE_ESCAPE_MAZE_LAYOUT,
      LARGE_SPIRAL_MAZE_LAYOUT,
      MINI_ROOM_MAZE_LAYOUT,
      ROOM_MAZE_LAYOUT,
      SPIRAL_MAZE_LAYOUT,
      SQUARE_MAZE_LAYOUT,
      WALL_ROOM_MAZE_LAYOUT,
  )

  layouts = {
      'room': ROOM_MAZE_LAYOUT,
      'miniroom': MINI_ROOM_MAZE_LAYOUT,
      'bigroom': BIG_ROOM_MAZE_LAYOUT,
      'biggestroom': BIGGEST_ROOM_MAZE_LAYOUT,
      'square': SQUARE_MAZE_LAYOUT,
      'bigsquare': BIG_SQUARE_MAZE_LAYOUT,
      'spiral': SPIRAL_MAZE_LAYOUT,
      'wallroom': WALL_ROOM_MAZE_LAYOUT,
      'escape': ESCAPE_MAZE_LAYOUT,
      'largeescape': LARGE_ESCAPE_MAZE_LAYOUT,
      'largespiral': LARGE_SPIRAL_MAZE_LAYOUT,
      'corridors': CORRIDORS_MAZE_LAYOUT,
  }
  if name in layouts:
    return layouts[name]
  if family == 'antmaze':
    from embodied.envs.custom_envs.locomotion.ant_maze_env import (
        _build_avoid_obstacle_layout,
        _build_u_trap_layout,
    )
    match = re.fullmatch(r'avoid_obstacle(?:_(1|3|5|7))?', name)
    if match:
      return _build_avoid_obstacle_layout(int(match.group(1) or 1))
    match = re.fullmatch(r'avoid_obstacle_small(?:_(1|3|5))?', name)
    if match:
      return _build_avoid_obstacle_layout(
          int(match.group(1) or 1), size=7)
    match = re.fullmatch(r'u_trap(?:_(3|5))?', name)
    if match:
      return _build_u_trap_layout(int(match.group(1) or 3))
  choices = sorted(layouts)
  if family == 'antmaze':
    choices += [
        'avoid_obstacle[_1|_3|_5|_7]',
        'avoid_obstacle_small[_1|_3|_5]',
        'u_trap[_3|_5]',
    ]
  raise ValueError(
      f'Unknown custom OGBench {family} layout {name!r}. Choose from '
      f'{choices}.')


def _convert_layout(layout):
  rows = len(layout)
  if not rows or not layout[0]:
    raise ValueError('Maze layout must not be empty.')
  cols = len(layout[0])
  init_ij = goal_ij = None
  for row, line in enumerate(layout):
    if len(line) != cols:
      raise ValueError('Maze layout must be rectangular.')
    for col, char in enumerate(line):
      if char not in '#.SG':
        raise ValueError(f'Unsupported maze layout character {char!r}.')
      # MJP's row zero is the largest Y value, while OGBench's row zero is
      # the smallest Y value. Flip the row when carrying task markers over.
      ij = (rows - 1 - row, col)
      if char == 'S':
        init_ij = ij
      elif char == 'G':
        goal_ij = ij
  if init_ij is None or goal_ij is None:
    raise ValueError('Maze layout must contain both S and G markers.')
  maze_map = np.asarray([
      [int(char == '#') for char in line]
      for line in reversed(layout)
  ], dtype=np.int32)
  return maze_map, init_ij, goal_ij
