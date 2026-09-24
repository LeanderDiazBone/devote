import functools
import re

import numpy as np


_NAME = re.compile(
    r'^(?P<visual>visual-)?mjp-cube-single_'
    r'pick_and_place_init_(?P<difficulty>mediumhard|hard)$')

_EPISODE_LENGTH = 250
_INIT_XYZ = np.asarray([0.35, 0.0, 0.06], dtype=np.float64)
_GOAL_X = {
    # MJP mediumhard covers 0.675 / 0.700 of the hard horizontal travel.
    # Scale that ratio into the reachable OGBench workspace.
    'mediumhard': 0.35 + (0.575 - 0.35) * (0.675 / 0.700),
    'hard': 0.575,
}
_GOAL_Z = 0.29
_CUBE_REST_Z = 0.02
_LIFT_CAP_Z = 0.14
_SUCCESS_THRESHOLD = 0.05
_LOST_GRIP_THRESHOLD = 0.12


def make_mjp_pick_and_place(name):
  """Build the MJP init-grasp task on OGBench's native Cube environment."""
  match = _NAME.fullmatch(name)
  if not match:
    return None

  import gymnasium

  visual = bool(match.group('visual'))
  difficulty = match.group('difficulty')
  env_class = _mjp_pick_and_place_class(difficulty)
  kwargs = dict(
      env_type='single',
      permute_blocks=False,
      reward_task_id=1,
      terminate_at_goal=True,
      success_timing='post',
  )
  if visual:
    kwargs.update(
        ob_type='pixels',
        width=64,
        height=64,
        visualize_info=False,
    )
  env = env_class(**kwargs)
  return gymnasium.wrappers.TimeLimit(
      env, max_episode_steps=_EPISODE_LENGTH)


@functools.lru_cache(None)
def _mjp_pick_and_place_class(difficulty):
  from ogbench.manipspace.envs.cube_env import CubeEnv

  goal_xyz = np.asarray(
      [_GOAL_X[difficulty], 0.0, _GOAL_Z], dtype=np.float64)

  class MJPCompatiblePickAndPlace(CubeEnv):

    _mjp_compatible = True
    _embodied_info_keys = (
        'log_dist',
        'log_rew_success',
        'log_rew_lift',
        'log_rew_healthy',
        'log_rew_ctrl',
    )

    def __init__(self, *args, **kwargs):
      self._last_action = np.zeros(5, dtype=np.float64)
      self._reward_terms = {}
      super().__init__(*args, **kwargs)

    def set_tasks(self):
      self.task_infos = [dict(
          task_name=f'pick_and_place_init_{difficulty}',
          init_xyzs=_INIT_XYZ[None].copy(),
          goal_xyzs=goal_xyz[None].copy(),
      )]
      if self._reward_task_id == 0:
        self._reward_task_id = 1

    def reset(self, *args, **kwargs):
      # Let OGBench select the task, create the native goal observation, and
      # initialize all internal bookkeeping before replacing only the physical
      # initial state with the MJP-style pre-grasp state.
      super().reset(*args, **kwargs)
      self._set_pregrasp_state()
      self._last_action.fill(0.0)
      self._success = False
      ob = self.compute_observation()
      info = self.get_reset_info()
      info['success'] = 0.0
      return ob, info

    def _set_pregrasp_state(self):
      import mujoco
      from ogbench.manipspace import lie

      cube_xyz = self.cur_task_info['init_xyzs'][0].copy()
      cube_xyz[:2] += self.np_random.uniform(-0.001, 0.001, size=2)
      cube_joint = self._data.joint('object_joint_0')
      cube_joint.qpos[:3] = cube_xyz
      cube_joint.qpos[3:] = lie.SO3.identity().wxyz.tolist()
      cube_joint.qvel[:] = 0.0

      # Use OGBench's own IK controller and downward-facing end-effector pose;
      # this retains the UR5 controller and action semantics.
      pinch_pose = lie.SE3.from_rotation_and_translation(
          self._effector_down_rotation, cube_xyz)
      attach_pose = pinch_pose @ self._T_pa
      arm_qpos = self._ik.solve(
          pos=attach_pose.translation(),
          quat=attach_pose.rotation().wxyz,
          curr_qpos=self._home_qpos,
      )
      arm_noise = self.np_random.uniform(-0.002, 0.002, size=arm_qpos.shape)
      arm_qpos = arm_qpos + arm_noise
      self._data.qpos[self._arm_joint_ids] = arm_qpos
      self._data.qvel[:] = 0.0
      self._data.ctrl[self._arm_actuator_ids] = arm_qpos

      # A normalized opening of 0.5 is approximately the 40 mm cube width;
      # the zero actuator target then maintains a closing force like MJP.
      for name in ('right_driver_joint', 'left_driver_joint'):
        joint_id = self._model.joint(f'ur5e/robotiq/{name}').id
        qpos_address = self._model.jnt_qposadr[joint_id]
        self._data.qpos[qpos_address] = 0.4
      self._data.ctrl[self._gripper_actuator_ids] = 0.0

      mujoco.mj_forward(self._model, self._data)
      self.pre_step()
      self.post_step()

    def _compute_successes(self):
      cube_xyz = self._data.joint('object_joint_0').qpos[:3]
      target_xyz = self._data.mocap_pos[self._cube_target_mocap_ids[0]]
      return [
          bool(np.linalg.norm(cube_xyz - target_xyz) < _SUCCESS_THRESHOLD)]

    def step(self, action):
      self._last_action = np.clip(
          np.asarray(action, dtype=np.float64), -1.0, 1.0)
      return super().step(action)

    def compute_reward(self):
      cube_xyz = self._data.joint('object_joint_0').qpos[:3]
      pinch_xyz = self._data.site_xpos[self._pinch_site_id]
      target_xyz = self._data.mocap_pos[self._cube_target_mocap_ids[0]]
      distance = float(np.linalg.norm(cube_xyz - target_xyz))
      pinch_distance = float(np.linalg.norm(pinch_xyz - cube_xyz))

      lift_span = _LIFT_CAP_Z - _CUBE_REST_Z
      lift_height = np.clip(
          cube_xyz[2] - _CUBE_REST_Z, 0.0, lift_span)
      lift_reward = 2.0 * lift_height / lift_span - 1.0
      healthy_reward = -0.5 * float(
          pinch_distance > _LOST_GRIP_THRESHOLD)
      control_reward = -0.1 * float(np.sum(np.square(self._last_action)))
      success_reward = float(5 * _EPISODE_LENGTH) * float(self._success)
      reward = (
          success_reward + lift_reward + healthy_reward + control_reward)
      self._reward_terms = {
          'log_dist': np.float32(distance),
          'log_d_pinch_cube': np.float32(pinch_distance),
          'log_rew_success': np.float32(success_reward),
          'log_rew_lift': np.float32(lift_reward),
          'log_rew_healthy': np.float32(healthy_reward),
          'log_rew_ctrl': np.float32(control_reward),
      }
      return np.float32(reward)

    def get_step_info(self):
      info = super().get_step_info()
      info.update(self._reward_terms)
      return info

    def get_reset_info(self):
      info = super().get_reset_info()
      pinch_xyz = self._data.site_xpos[self._pinch_site_id]
      cube_xyz = self._data.joint('object_joint_0').qpos[:3]
      target_xyz = self._data.mocap_pos[self._cube_target_mocap_ids[0]]
      info.update({
          'log_dist': np.float32(np.linalg.norm(cube_xyz - target_xyz)),
          'log_d_pinch_cube': np.float32(
              np.linalg.norm(pinch_xyz - cube_xyz)),
          'log_rew_success': np.float32(0.0),
          'log_rew_lift': np.float32(0.0),
          'log_rew_healthy': np.float32(0.0),
          'log_rew_ctrl': np.float32(0.0),
      })
      return info

  MJPCompatiblePickAndPlace.__name__ = (
      f'OGBenchMJPPickAndPlaceInit{difficulty.title()}')
  return MJPCompatiblePickAndPlace
