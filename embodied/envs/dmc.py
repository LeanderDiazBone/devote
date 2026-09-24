import functools
import os
from typing import Any, Dict
from typing import Optional

import embodied
import numpy as np


class DMC(embodied.Env):

  DEFAULT_CAMERAS = dict(
      quadruped=2,
      locom_rodent=4,
  )

  def __init__(
      self, env, repeat=1, size=(64, 64), image=True, camera=-1,
      seed: Optional[int] = None, reward_scale: float = 1.0,
      max_step_limit: int = 0, log_image: bool = True,
      distract_difficulty: str = '', distract_dataset_path: str = '',
      distract_dynamic: bool = False, distract_videos: str = 'train'):
    if 'MUJOCO_GL' not in os.environ:
      os.environ['MUJOCO_GL'] = 'egl'
    self._task = env if isinstance(env, str) else None
    self._reward_scale = float(reward_scale)
    if isinstance(env, str):
      domain, task = env.split('_', 1)
      if camera == -1:
        camera = self.DEFAULT_CAMERAS.get(domain, 0)
      if domain == 'cup':  # Only domain with multiple words.
        domain = 'ball_in_cup'
      if domain == 'manip':
        from dm_control import manipulation
        env = manipulation.load(task + '_vision')
      elif domain == 'locom':
        # camera 0: topdown map
        # camera 2: shoulder
        # camera 4: topdown tracking
        # camera 5: eyes
        from dm_control.locomotion.examples import basic_rodent_2020
        env = getattr(basic_rodent_2020, task)()
      elif distract_difficulty:
        from .distracting_control import suite as dc_suite
        env = dc_suite.load(
            domain, task,
            difficulty=distract_difficulty,
            dynamic=bool(distract_dynamic),
            background_dataset_path=distract_dataset_path or None,
            background_dataset_videos=distract_videos or 'train',
            render_kwargs=dict(camera_id=camera),
            add_pixel_wrapper=False,
        )
      else:
        from dm_control import suite
        env = suite.load(domain, task)
    if max_step_limit and hasattr(env, '_step_limit'):
      env._step_limit = int(max_step_limit)
    self._dmenv = env
    from . import from_dm
    self._env = from_dm.FromDM(self._dmenv, seed=seed)
    self._env = embodied.wrappers.ExpandScalars(self._env)
    self._env = embodied.wrappers.ActionRepeat(self._env, repeat)
    self._size = size
    self._image = image
    self._log_image = log_image
    self._camera = camera

  @functools.cached_property
  def obs_space(self):
    spaces = self._env.obs_space.copy()
    if self._image or self._log_image:
      key = 'image' if self._image else 'log_image'
      spaces[key] = embodied.Space(np.uint8, self._size + (3,))
    return spaces

  @functools.cached_property
  def act_space(self):
    return self._env.act_space

  def step(self, action):
    for key, space in self.act_space.items():
      if not space.discrete:
        assert np.isfinite(action[key]).all(), (key, action[key])
    obs = self._env.step(action)
    if self._reward_scale != 1.0:
      obs['reward'] = np.float32(obs['reward'] * self._reward_scale)
    if self._image or self._log_image:
      key = 'image' if self._image else 'log_image'
      obs[key] = self._dmenv.physics.render(*self._size, camera_id=self._camera)
    for key, space in self.obs_space.items():
      if np.issubdtype(space.dtype, np.floating):
        assert np.isfinite(obs[key]).all(), (key, obs[key])
    return obs

  def get_coverage_geometry(self):
    task = self._task or ''
    if task.startswith('cartpole_'):
      # position = [cart_x, cos(theta), sin(theta)]; slider limited to [-1.8, 1.8].
      def project(tran):
        p = np.asarray(tran['position'], dtype=np.float64)
        theta = np.arctan2(p[..., 2], p[..., 1])
        return np.stack([p[..., 0], theta], axis=-1)
      return dict(
          bounds=np.array(
              [[-1.8, 1.8], [-np.pi, np.pi]], dtype=np.float64),
          bins=(25, 25),
          axis_names=('cart_x', 'pole_angle'),
          project=project,
      )
    if task in ('acrobot_swingup', 'acrobot_swingup_sparse'):
      # orientations = [cos(shoulder), sin(shoulder), cos(elbow), sin(elbow)].
      def project(tran):
        o = np.asarray(tran['orientations'], dtype=np.float64)
        theta1 = np.arctan2(o[..., 1], o[..., 0])
        theta2 = np.arctan2(o[..., 3], o[..., 2])
        return np.stack([theta1, theta2], axis=-1)
      return dict(
          bounds=np.array(
              [[-np.pi, np.pi], [-np.pi, np.pi]], dtype=np.float64),
          bins=(25, 25),
          axis_names=('shoulder_angle', 'elbow_angle'),
          project=project,
      )
    if task in ('hopper_stand', 'hopper_hop'):
      # position = qpos[1:] = [torso_z, torso_angle, hip, knee, foot].
      # velocity = qvel        = [vx, vz, wy, vhip, vknee, vfoot].
      def project(tran):
        p = np.asarray(tran['position'], dtype=np.float64)
        v = np.asarray(tran['velocity'], dtype=np.float64)
        return np.stack([p[..., 0], v[..., 0]], axis=-1)
      return dict(
          bounds=np.array(
              [[0.0, 1.4], [-3.0, 3.0]], dtype=np.float64),
          bins=(25, 25),
          axis_names=('torso_z', 'forward_vx'),
          project=project,
      )
    if task in ('walker_walk', 'walker_run', 'walker_stand'):
      # velocity[0] = vx (forward), velocity[1] = vz (vertical). Random walkers
      # fall and flail, which trivially fills (vx, height); vz stays near 0
      # unless the agent actively jumps, so it discriminates exploration better.
      def project(tran):
        v = np.asarray(tran['velocity'], dtype=np.float64)
        return np.stack([v[..., 0], v[..., 1]], axis=-1)
      return dict(
          bounds=np.array(
              [[-3.0, 3.0], [-2.0, 4.0]], dtype=np.float64),
          bins=(25, 25),
          axis_names=('forward_vx', 'vertical_vz'),
          project=project,
      )
    if task == 'cheetah_run':
      # velocity = qvel = [vx, vz, wy, ...]. vz reflects jumping; random ~ 0.
      def project(tran):
        v = np.asarray(tran['velocity'], dtype=np.float64)
        return np.stack([v[..., 0], v[..., 1]], axis=-1)
      return dict(
          bounds=np.array(
              [[-5.0, 10.0], [-2.0, 4.0]], dtype=np.float64),
          bins=(25, 25),
          axis_names=('forward_vx', 'vertical_vz'),
          project=project,
      )
    if task == 'cup_catch':
      # position = [cup_x, cup_z, ball_x, ball_z] (world coords).
      def project(tran):
        p = np.asarray(tran['position'], dtype=np.float64)
        return np.stack([p[..., 2], p[..., 3]], axis=-1)
      return dict(
          bounds=np.array(
              [[-0.4, 0.4], [-0.4, 0.4]], dtype=np.float64),
          bins=(25, 25),
          axis_names=('ball_x', 'ball_z'),
          project=project,
      )
    return None

  def dynamics_complexity_scale(self, obs):
    task = self._task or ''
    domain = task.split('_', 1)[0] if task else None
    if domain != 'cartpole' or 'position' not in obs:
      return np.float32(1.0)
    position = np.asarray(obs['position'], np.float32)
    pole_height = np.clip((position[..., 1] + 1.0) / 2.0, 0.0, 1.0)
    return 5.0 * np.clip(np.exp(-2.0 * np.log(2) * pole_height).astype(np.float32)-0.5, 0.0, 1.0)

  def get_state_table(
      self, points_per_axis: int = 81) -> Dict[str, Any]:
    task = self._task or ''
    domain = task.split('_', 1)[0] if task else None
    if domain != 'cartpole':
      raise NotImplementedError(
          'DMC state tables are only implemented for cartpole tasks.')

    position_space = self._env.obs_space.get('position')
    velocity_space = self._env.obs_space.get('velocity')
    if position_space is None or velocity_space is None:
      raise ValueError(
          'Cartpole state table expects DMC observations with '
          '"position" and "velocity" keys.')
    if position_space.shape != (3,) or velocity_space.shape != (2,):
      raise ValueError(
          'Cartpole state table assumes a single-pole observation layout: '
          f'position={position_space.shape}, velocity={velocity_space.shape}')

    # DM Control cartpole.xml constrains the slider joint to [-1.8, 1.8].
    xs = np.linspace(-1.8, 1.8, points_per_axis, dtype=np.float32)
    thetas = np.linspace(-np.pi, np.pi, points_per_axis, dtype=np.float32)
    grid_x, grid_theta = np.meshgrid(xs, thetas, indexing='xy')
    flat_x = grid_x.reshape(-1)
    flat_theta = grid_theta.reshape(-1)

    position = np.stack(
        [flat_x, np.cos(flat_theta), np.sin(flat_theta)], axis=-1)
    velocity = np.zeros((position.shape[0], 2), np.float32)
    return dict(
        obs={
            'position': position.astype(np.float32),
            'velocity': velocity,
        },
        obs_col_names={
            'position': ['cart_position', 'pole_angle_cos', 'pole_angle_sin'],
            'velocity': ['cart_velocity', 'pole_angular_velocity'],
        },
        grid_shape=np.array(grid_x.shape, dtype=np.int32),
    )

  def get_action_grid(self, points_per_axis: int = 41) -> np.ndarray:
    """Return a flattened normalized action grid of shape [N, action_dim]."""
    task = self._task or ''
    domain = task.split('_', 1)[0] if task else None
    if domain != 'cartpole':
      raise NotImplementedError(
          'DMC action grid only implemented for cartpole tasks.')
    return np.linspace(
        -1.0, 1.0, points_per_axis, dtype=np.float32).reshape(-1, 1)

  def get_action_table(
      self,
      state_points_per_axis: int = 5,
      action_points: int = 41,
      state_points: np.ndarray = None,
  ) -> Dict[str, Any]:
    task = self._task or ''
    domain = task.split('_', 1)[0] if task else None
    if domain != 'cartpole':
      raise NotImplementedError(
          'DMC action tables are only implemented for cartpole tasks.')

    position_space = self._env.obs_space.get('position')
    velocity_space = self._env.obs_space.get('velocity')
    if position_space is None or velocity_space is None:
      raise ValueError(
          'Cartpole action table expects DMC observations with '
          '"position" and "velocity" keys.')
    if position_space.shape != (3,) or velocity_space.shape != (2,):
      raise ValueError(
          'Cartpole action table assumes a single-pole observation layout: '
          f'position={position_space.shape}, velocity={velocity_space.shape}')

    if state_points is not None:
      sp = np.asarray(state_points, dtype=np.float32)
      if sp.ndim != 2 or sp.shape[-1] not in (2, 5):
        raise ValueError(
            f'state_points must be [N, 2] (x, theta) or [N, 5] '
            f'(pos+vel); got {sp.shape}')
      if sp.shape[-1] == 2:
        position = np.stack(
            [sp[:, 0], np.cos(sp[:, 1]), np.sin(sp[:, 1])], axis=-1)
        velocity = np.zeros((sp.shape[0], 2), np.float32)
      else:
        position = sp[:, :3]
        velocity = sp[:, 3:5]
      state_grid_shape = np.array([sp.shape[0]], dtype=np.int32)
    else:
      xs = np.linspace(-1.8, 1.8, state_points_per_axis, dtype=np.float32)
      thetas = np.linspace(
          -np.pi, np.pi, state_points_per_axis, dtype=np.float32)
      grid_x, grid_theta = np.meshgrid(xs, thetas, indexing='xy')
      flat_x = grid_x.reshape(-1)
      flat_theta = grid_theta.reshape(-1)
      position = np.stack(
          [flat_x, np.cos(flat_theta), np.sin(flat_theta)], axis=-1)
      velocity = np.zeros((position.shape[0], 2), np.float32)
      state_grid_shape = np.array(grid_x.shape, dtype=np.int32)

    actions = self.get_action_grid(action_points)
    repeats = actions.shape[0]
    action_key = next(k for k in self.act_space if k != 'reset')
    return dict(
        obs={
            'position': np.repeat(position.astype(np.float32), repeats, axis=0),
            'velocity': np.repeat(velocity, repeats, axis=0),
        },
        actions={
            action_key: np.tile(actions, (position.shape[0], 1)).astype(np.float32),
        },
        obs_col_names={
            'position': ['cart_position', 'pole_angle_cos', 'pole_angle_sin'],
            'velocity': ['cart_velocity', 'pole_angular_velocity'],
        },
        action_col_names={action_key: ['action']},
        state_grid_shape=state_grid_shape,
        action_grid_shape=np.array([action_points], dtype=np.int32),
    )
