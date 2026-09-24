"""Sparse-reward Franka Panda manipulation env with four sub-tasks.

Sub-tasks:
- ``reach``: move the end-effector to a random goal position
  (``panda_reach.xml`` — 7-DoF arm + free-floating goal sphere, no fingers).
- ``slide``: push the cube to a goal pose on the table surface
  (``panda_push_hard.xml`` — 7-DoF arm, parallel-jaw gripper, cube, goal box,
   two bins).
- ``pick_and_place``: pick the cube up and place it at an elevated goal
  (same XML as ``slide``).
- ``pick_and_place_init``: same task and reward as ``pick_and_place``, but the
  arm starts with the gripper already wrapped around the cube (fingers
  partially closed at the cube position). Skips the picking phase so the
  policy only has to lift and place. The goal sits 1.5x higher than the
  ``pick_and_place`` default (z=0.30 instead of 0.20).
- ``pick_and_place_init_wall``: like ``pick_and_place_init`` but with the
  default z=0.20 goal and a thin wall between cube spawn and goal (top at
  z=0.50, spanning y in [0.50, 0.80]). Carrying the cube over the top is
  near the arm's reach limit at the direct crossing, so the intended
  route is around the wall's near end (y < 0.50).
- ``pick_and_place_init_vertical``: like ``pick_and_place_init`` but the dense
  lift term rewards approach to a point straight ABOVE the cube's spawn at
  z=0.20 (penalizing horizontal drift) rather than raw height, and the goal is
  NOT raised (z=0.20). The sparse goal stays at the displaced location and
  still shifts with the easy/medium/hard x-offset, so the dense (over-spawn)
  and sparse (displaced) targets are distinct.

Reward: ``success_reward`` on success (target within ``success_thresh`` of
the goal), plus task-specific dense shaping that does NOT reveal the goal
location:
- ``reach``: per-step velocity bonus on the end-effector.
- ``slide``: smooth EE→cube proximity bonus + cube velocity bonus.
- ``pick_and_place``: layered shaping with monotonically-growing per-step
  ceiling as the skill advances:
    reach (long-range EE→cube proximity)
    < close (tight proximity AND gripper closure)
    < lift (continuous cube height above table)
    << success (sparse, dominates).
  Per-episode max of every dense term is small relative to the sparse
  success bonus, so sparse always wins. No goal-distance term.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env


_ROOT = Path(__file__).resolve().parent
_REACH_XML = _ROOT / "assets" / "panda_reach.xml"
_PUSH_XML = _ROOT / "assets" / "panda_push_hard.xml"
_PUSH_WALL_XML = _ROOT / "assets" / "panda_push_hard_wall.xml"

SUB_TASKS = (
    "reach",
    "pick_and_place",
    "pick_and_place_init",
    "pick_and_place_init_wall",
    "pick_and_place_init_vertical",
    "slide",
)

# Optional ``_easy`` / ``_medium`` / ``_mediumhard`` / ``_hard`` suffix shifts
# the goal box in +x. Medium keeps the existing default; harder = further to
# the right. ``mediumhard`` sits halfway between medium and hard. Suffix is a
# single token (no second underscore) since the parser uses ``rpartition('_')``.
DIFFICULTIES = ("easy", "medium", "mediumhard", "hard")
_DIFFICULTY_X_OFFSET = {"easy": -0.25, "medium": 0.0, "mediumhard": 0.175, "hard": 0.20}

# "low_home" pose from the upstream mujoco_playground franka panda keyframe.
_HOME_ARM_QPOS = (0.0, 0.6, 0.0, -1.4, 0.0, 2.4, 0.0)
_HOME_FINGER_QPOS = 0.04          # fingers open, meters
_HOME_FINGER_CTRL = 255.0         # remapped open command (ctrlrange 0..255)

# For pick_and_place_init: fingers half-closed around the cube (cube half-extent
# 0.03 m + ~0.5 cm clearance per side) and a closing ctrl command so the
# actuator squeezes the cube rather than springing back open.
_GRASP_FINGER_QPOS = 0.028        # ~0.056 m total opening, cube is 0.06 m wide
_GRASP_FINGER_CTRL = 0.0          # remapped close command


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.05,
        sim_dt=0.005,
        episode_length=250,
        action_repeat=1,
        # Per-step delta applied to arm ctrl, in radians.
        action_scale=0.04, # 0.05

        # Sparse-reward parameters. success_reward = 1.0 sets the unit: one
        # step at the goal pays 1.0. All dense terms are scaled to be much
        # smaller per step (and per episode) than the sparse target.
        success_thresh=0.05,
        # When ``terminate_on_success`` is True the episode ends on the first
        # success step and the reward at that step is overridden to
        # ``5 * episode_length`` (one-shot terminal bonus, five times the
        # horizon, replaces the per-step camping). When False,
        # ``success_reward`` is paid per step at goal.
        success_reward=1.0,
        terminate_on_success=True,
        # Lost-grip threshold on ``d_ee_cube`` (cube has slipped out of the
        # gripper). The init grasp keeps d_ee_cube ~= 0 while held; once the
        # cube falls or is bumped clear of the fingers it grows quickly.
        # Drives the ant-style ``healthy_reward`` penalty (see below) and,
        # optionally via the ``_term`` suffix, also drives termination.
        lost_grip_thresh=0.12,
        # Per-step ant-style healthy penalty: paid while ``d_ee_cube``
        # exceeds ``lost_grip_thresh`` in init-grasp mode. Mirrors ant's
        # ``healthy_reward`` so the policy gets a signed grounding signal
        # without needing to terminate on grip loss.
        healthy_reward=-0.5,
        # Per-step velocity bonus = vel_reward_scale * min(||v||, vel_max) / vel_max.
        # In [0, vel_reward_scale]; linear up to vel_max so slow pushes already pay.
        vel_reward_scale=0.02,
        vel_max=0.3,
        # Quadratic control cost = ctrl_cost_scale * sum(action**2). Matched
        # to ant (``ctrl_cost=0.1``): 8-dim action clipped to [-1, 1] gives a
        # per-step worst case of 0.8, comparable to the lift reward (max 1.0)
        # and the healthy penalty (-0.5).
        ctrl_cost_scale=0.1,
        # Direction-agnostic shaping (slide / pick_and_place).
        # reach_reward: smooth proximity bonus on EE→cube, in (0, reach_reward_scale].
        # contact_thresh / grasp_thresh are logging-only diagnostics; the
        # reward no longer references grasp_thresh.
        reach_reward_scale=0.02,
        reach_scale=0.5,
        contact_thresh=0.05,
        grasp_thresh=0.04,
        # Pick-and-place shaping. The lift term is now SIGNED (mirroring
        # ant's signed speed projection): cube at rest gives
        # ``-lift_reward_scale``, cube at the lift cap gives
        # ``+lift_reward_scale``. The normalization absorbs the dynamic
        # ``max_lift_height`` so ``lift_reward_scale`` directly equals the
        # per-step magnitude.
        #
        # Per-step range (init_grasp pick_and_place):
        #   lift (signed, normalized over [0, max_lift_height])    ±1.000
        #   reach (proximity at cube)                               0.020
        #   close (proximity_grasp * finger_closure)                0.060
        #   healthy penalty (lost grip)                            -0.500
        #   ctrl  (worst-case 8·a², a∈[-1,1])                      -0.800
        #
        # Success is now a one-shot terminal bonus of ``5 * episode_length``
        # (see ``terminate_on_success``), five times the horizon.
        #
        # Cube rests with its center at z = cube_rest_z (table plane at 0,
        # cube half-extent 0.03 → settles at 0.03). Using 0.03 (not 0.07,
        # the spawn z) removes the lift-reward "dead zone" that previously
        # required >4 cm of lift before any signal fired.
        cube_rest_z=0.03,
        lift_reward_scale=1.0,
        # close_reward: proximity_grasp-gated gripper closure bonus. Provides
        # the missing gradient from "hover at cube" to "actually grasp cube".
        # proximity_grasp = 1 - tanh(d_ee_cube / grasp_scale) uses a tight
        # scale so the bonus only fires when fingers are wrapped around the
        # cube; finger_closure = 1 - qpos[7]/0.05 in [0, 1] (0 = open).
        close_reward_scale=0.06,
        grasp_scale=0.05,
        # Goal sampling bounds (m, world frame). Defaults are degenerate
        # (low == high) so the goal is a single fixed point. Widen the box
        # via config_overrides to recover per-episode randomization.
        reach_goal_low=(0.0, 0.5, 0.25),
        reach_goal_high=(0.0, 0.5, 0.25),
        slide_goal_low=(0.25, 0.65, 0.07),
        slide_goal_high=(0.25, 0.65, 0.07),
        pick_goal_low=(0.25, 0.65, 0.20),
        pick_goal_high=(0.25, 0.65, 0.20),
        # Cube spawn bounds (slide / pick_and_place only). Degenerate by
        # default (low == high) so the cube spawns at a fixed point in the
        # middle of the prior randomization box. Widen via config_overrides
        # to recover per-episode randomization.
        cube_low=(-0.25, 0.65, 0.07),
        cube_high=(-0.25, 0.65, 0.07),
        # Small per-episode init noise for diversity. Kept tight so the
        # ``pick_and_place_init`` grasp doesn't break: arm noise is mirrored
        # into the actuator ctrl so the arm doesn't spring back to the
        # noise-free IK pose at step 0, and cube xy noise stays inside the
        # finger squeeze envelope (fingers grip ~2 mm tighter than the cube).
        init_arm_qpos_noise=0.002,
        init_cube_xy_noise=0.001,
        ctrl_leak=0.75,
    )


class PandaEnv(mjx_env.MjxEnv):
    def __init__(
        self,
        sub_task: Optional[str] = None,
        config: Optional[config_dict.ConfigDict] = None,
        config_overrides: Optional[Dict[str, Union[str, int, float, list[Any]]]] = None,
    ):
        if config is None:
            config = default_config()
        mjx_env.MjxEnv.__init__(self, config, config_overrides)

        sub_task = (sub_task or "reach").lower()
        # Optional ``_term`` suffix: end the episode when the gripper loses
        # the cube (d_ee_cube > lost_grip_thresh). Stripped BEFORE the
        # difficulty suffix so ``..._init_hard_term`` and ``..._init_term``
        # both work. Only meaningful for the init variants (the cube starts
        # in the gripper); enforced after difficulty parsing below.
        self._terminate_on_lost_grip = False
        if sub_task.endswith("_term"):
            self._terminate_on_lost_grip = True
            sub_task = sub_task[: -len("_term")]
        # Optional difficulty suffix: `<task>_easy|_medium|_hard`. Medium is
        # the default and reproduces the historical (suffix-less) goal box.
        difficulty = "medium"
        head, _, tail = sub_task.rpartition("_")
        if head and tail in DIFFICULTIES:
            difficulty = tail
            sub_task = head
        if sub_task not in SUB_TASKS:
            raise ValueError(
                f"Unknown sub_task {sub_task!r}; choose from {SUB_TASKS}.")
        self._sub_task = sub_task
        self._difficulty = difficulty
        self._is_reach = sub_task == "reach"
        # pick_and_place_init(_wall/_vertical) share reward + bookkeeping with
        # pick_and_place.
        self._is_pap = sub_task in (
            "pick_and_place",
            "pick_and_place_init",
            "pick_and_place_init_wall",
            "pick_and_place_init_vertical",
        )
        self._init_grasp = sub_task in (
            "pick_and_place_init", "pick_and_place_init_wall",
            "pick_and_place_init_vertical")
        self._is_wall = sub_task == "pick_and_place_init_wall"
        # Vertical-lift variant: dense reward shapes the cube toward a point
        # straight above its spawn (goal-agnostic); the sparse goal stays at
        # the displaced location. See reset()/step()/get_coverage_geometry.
        self._is_vertical = sub_task == "pick_and_place_init_vertical"
        if self._terminate_on_lost_grip and not self._init_grasp:
            raise ValueError(
                f"_term suffix is only valid for the init variants "
                f"('pick_and_place_init', 'pick_and_place_init_wall'); "
                f"got sub_task {sub_task!r}. Non-init variants start with "
                f"the gripper far from the cube, so termination would fire "
                f"on step 1.")

        # Lift-only shaping for the init variants. The arm starts already
        # gripping the cube, so reach/close give no useful gradient and the
        # close term in particular conflicts with maintaining the grip
        # (finger_closure peaks past the cube width).
        if self._init_grasp:
            self._config.reach_reward_scale = 0.0
            self._config.close_reward_scale = 0.0

        # Vertical variant: shrink the lost-grip ``healthy_reward`` penalty to
        # well below the lift reward (the default -0.5 is ~half of
        # lift_reward_scale). At ~5% of the lift magnitude the over-spawn lift
        # signal stays the dominant dense term and dropping is only a minor
        # nudge -- the agent already forfeits the lift reward when the cube
        # leaves the gripper.
        if self._is_vertical:
            self._config.healthy_reward = -0.05 * self._config.lift_reward_scale

        if self._is_reach:
            xml_path = _REACH_XML
        elif self._is_wall:
            xml_path = _PUSH_WALL_XML
        else:
            xml_path = _PUSH_XML
        mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
        mj_model.opt.timestep = self._config.sim_dt

        self._mj_model = mj_model
        self._mjx_model = mjx.put_model(mj_model)
        self._xml_path = str(xml_path)

        # Body and joint address lookups.
        self._hand_body = int(mj_model.body("hand").id)
        self._goal_body = int(mj_model.body("goal_marker").id)
        self._goal_qposadr = self._freejoint_qposadr("goal_marker")
        self._left_finger_body = int(mj_model.body("left_finger").id)
        self._right_finger_body = int(mj_model.body("right_finger").id)
        if self._is_reach:
            self._cube_body = -1
            self._cube_qposadr = -1
            self._cube_qveladr = -1
        else:
            self._cube_body = int(mj_model.body("cube").id)
            self._cube_qposadr = self._freejoint_qposadr("cube")
            self._cube_qveladr = self._freejoint_qveladr("cube")

        # Initial qpos / ctrl. The first 7 qpos entries are the arm joints;
        # the next two are the finger joints.
        init_qpos = jp.array(mj_model.qpos0, dtype=jp.float32)
        init_qpos = init_qpos.at[:7].set(jp.asarray(_HOME_ARM_QPOS))
        init_qpos = init_qpos.at[7:9].set(_HOME_FINGER_QPOS)
        self._init_qpos = init_qpos
        self._init_qvel = jp.zeros(mj_model.nv, dtype=jp.float32)

        init_ctrl = jp.zeros(mj_model.nu, dtype=jp.float32)
        init_ctrl = init_ctrl.at[:7].set(jp.asarray(_HOME_ARM_QPOS))
        init_ctrl = init_ctrl.at[7:9].set(_HOME_FINGER_CTRL)
        self._init_ctrl = init_ctrl

        # Action layout: 7 arm joints + 1 gripper command (mirrored to both
        # finger actuators; the XML enforces equality between the two finger
        # joints). For ``reach`` the gripper command has no effect on reward.
        self._action_size = 8

        # Actuator ctrlrange clip bounds.
        self._arm_lower = jp.asarray(mj_model.actuator_ctrlrange[:7, 0], jp.float32)
        self._arm_upper = jp.asarray(mj_model.actuator_ctrlrange[:7, 1], jp.float32)
        self._finger_lower = float(mj_model.actuator_ctrlrange[7, 0])
        self._finger_upper = float(mj_model.actuator_ctrlrange[7, 1])

        # Goal / cube randomization bounds. The difficulty suffix shifts the
        # whole goal box in +x: easy = current default - 0.25, medium = current
        # default, hard = current default + 0.20.
        if self._is_reach:
            goal_low = np.asarray(self._config.reach_goal_low, np.float32)
            goal_high = np.asarray(self._config.reach_goal_high, np.float32)
        elif sub_task == "slide":
            goal_low = np.asarray(self._config.slide_goal_low, np.float32)
            goal_high = np.asarray(self._config.slide_goal_high, np.float32)
        else:  # pick_and_place(_init)
            goal_low = np.asarray(self._config.pick_goal_low, np.float32)
            goal_high = np.asarray(self._config.pick_goal_high, np.float32)
        x_offset = np.float32(_DIFFICULTY_X_OFFSET[difficulty])
        goal_low = goal_low.copy(); goal_low[0] += x_offset
        goal_high = goal_high.copy(); goal_high[0] += x_offset
        # Lift shaping saturates at the configured goal height, anchored
        # HERE — before the init-variant goal raise below — so raising the
        # goal does not move the lift-reward cap.
        self._lift_cap_z = float(goal_high[2]) - self._config.success_thresh
        # Base init variant: raise the goal 1.5x (0.20 -> 0.30). The wall
        # variant keeps z=0.20. The vertical variant anchors its dense
        # lift-target height at the un-raised goal z (like _lift_cap_z above),
        # then raises only the sparse goal 1.25x (0.20 -> 0.25) so success sits
        # above where the dense lift reward saturates.
        if self._is_vertical:
            self._lift_target_z = float(goal_high[2])
            goal_low[2] *= 1.25
            goal_high[2] *= 1.25
        elif self._init_grasp and not self._is_wall:
            goal_low[2] *= 1.5
            goal_high[2] *= 1.5
        self._goal_low = jp.asarray(goal_low, jp.float32)
        self._goal_high = jp.asarray(goal_high, jp.float32)
        if not self._is_reach:
            self._cube_low = jp.asarray(self._config.cube_low, jp.float32)
            self._cube_high = jp.asarray(self._config.cube_high, jp.float32)
            if self._is_vertical:
                # Dense-reward normalizer: vertical distance from the cube
                # spawn z to the lift-target z (``_lift_target_z``, the
                # un-raised goal height). The cube spawns at cube_high[2]=0.07
                # and the target sits at 0.20, so at reset d_lift == lift_span
                # and the reward starts at -scale (cube on the table) and
                # saturates at +scale (cube lifted to the point over the spawn).
                self._lift_span = (
                    self._lift_target_z - float(self._config.cube_high[2]))

        # For the "init" variant, solve IK once on CPU mujoco to put the
        # gripper midpoint at the cube spawn center, then override the arm
        # init pose and finger init/ctrl so the cube starts grasped.
        if self._init_grasp:
            cube_center = 0.5 * (np.asarray(self._config.cube_low, np.float64)
                                 + np.asarray(self._config.cube_high, np.float64))
            arm_qpos = self._solve_grasp_ik(cube_center)
            init_qpos = self._init_qpos.at[:7].set(jp.asarray(arm_qpos, jp.float32))
            init_qpos = init_qpos.at[7:9].set(_GRASP_FINGER_QPOS)
            self._init_qpos = init_qpos
            init_ctrl = self._init_ctrl.at[:7].set(jp.asarray(arm_qpos, jp.float32))
            init_ctrl = init_ctrl.at[7:9].set(_GRASP_FINGER_CTRL)
            self._init_ctrl = init_ctrl

    # -- helpers -------------------------------------------------------------

    def _freejoint_qposadr(self, body_name: str) -> int:
        jnt = int(self._mj_model.body(body_name).jntadr[0])
        return int(self._mj_model.jnt_qposadr[jnt])

    def _freejoint_qveladr(self, body_name: str) -> int:
        jnt = int(self._mj_model.body(body_name).jntadr[0])
        return int(self._mj_model.jnt_dofadr[jnt])

    def _solve_grasp_ik(self, target_pos: np.ndarray) -> np.ndarray:
        # Damped-least-squares IK on the 7-DoF arm only. Drives the midpoint
        # of the two fingertips to ``target_pos``. Runs once on CPU mujoco at
        # __init__; the result is baked into ``_init_qpos``.
        mj_data = mujoco.MjData(self._mj_model)
        qpos = np.array(self._mj_model.qpos0, copy=True, dtype=np.float64)
        qpos[:7] = np.asarray(_HOME_ARM_QPOS, dtype=np.float64)
        qpos[7:9] = _HOME_FINGER_QPOS
        arm_lo = np.asarray(self._mj_model.actuator_ctrlrange[:7, 0], np.float64)
        arm_hi = np.asarray(self._mj_model.actuator_ctrlrange[:7, 1], np.float64)
        target = np.asarray(target_pos, np.float64)
        nv = self._mj_model.nv
        jac_l = np.zeros((3, nv))
        jac_r = np.zeros((3, nv))
        for _ in range(300):
            mj_data.qpos[:] = qpos
            mujoco.mj_forward(self._mj_model, mj_data)
            ee = 0.5 * (mj_data.xpos[self._left_finger_body]
                        + mj_data.xpos[self._right_finger_body])
            err = target - ee
            if np.linalg.norm(err) < 1e-4:
                break
            mujoco.mj_jacBody(self._mj_model, mj_data, jac_l, None,
                              self._left_finger_body)
            mujoco.mj_jacBody(self._mj_model, mj_data, jac_r, None,
                              self._right_finger_body)
            jac_arm = 0.5 * (jac_l[:, :7] + jac_r[:, :7])
            damping = 0.1
            delta = jac_arm.T @ np.linalg.solve(
                jac_arm @ jac_arm.T + damping ** 2 * np.eye(3), err)
            qpos[:7] = np.clip(qpos[:7] + np.clip(delta, -0.3, 0.3),
                               arm_lo, arm_hi)
        return qpos[:7].astype(np.float32)

    @staticmethod
    def _set_freejoint_pos(qpos: jax.Array, qposadr: int, xyz: jax.Array) -> jax.Array:
        qpos = qpos.at[qposadr:qposadr + 3].set(xyz)
        # Identity quaternion (w, x, y, z).
        return qpos.at[qposadr + 3:qposadr + 7].set(
            jp.array([1.0, 0.0, 0.0, 0.0], dtype=jp.float32))

    # -- properties ----------------------------------------------------------

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def observation_size(self) -> int:
        # arm qpos (7) + arm qvel (7) + ee_pos (3)
        # + [finger qpos (2) + finger qvel (2) + cube_pos (3)] (slide/pick only)
        # + goal_pos (3)
        return 7 + 7 + 3 + (0 if self._is_reach else 2 + 2 + 3) + 3

    # -- coverage geometry --------------------------------------------------

    def get_coverage_geometry(self, bins_per_cell: int = 1) -> dict:
        """Uniform-binning geometry over the object of interest.

        Coverage is tracked over the end-effector for ``reach`` (3D),
        cube ``(x, y)`` for ``slide`` (2D, the cube stays on the table),
        and cube ``(x, y, z)`` for ``pick_and_place(_init)`` (3D). For
        pick_and_place the cube position is only binned while the gripper is
        holding it (grasp-gated), so coverage measures where the arm *placed*
        the cube rather than wherever it happens to rest un-grasped.

        ``bins_per_cell`` must be a positive perfect square; its square
        root multiplies the per-axis bin count (matches the point_maze
        convention).
        """
        k = int(bins_per_cell)
        sub = int(round(np.sqrt(k)))
        if sub * sub != k or sub < 1:
            raise ValueError(
                f'coverage_bins_per_cell must be a positive perfect square; got {k}')

        # ``grasp_gated`` restricts coverage to steps where the gripper is
        # actually holding the cube (see the projection below). Only
        # pick_and_place needs it: there the cube can rest at many positions
        # without the arm (before the pick, after a drop), so gating makes
        # coverage reflect where the arm *placed* the cube. Reach has no cube
        # and slide's cube only moves while pushed, so both stay ungated.
        grasp_gated = False
        if self._is_reach:
            bounds = np.array(
                [[-0.55, 0.55], [0.00, 0.85], [0.00, 0.60]], dtype=np.float64)
            bins = (12 * sub, 12 * sub, 12 * sub)
            axis_names = ('ee_x', 'ee_y', 'ee_z')
            lo, hi = 14, 17                 # ee_pos slice in obs
        elif self._sub_task == 'slide':
            bounds = np.array(
                [[-0.55, 0.55], [0.35, 0.95]], dtype=np.float64)
            bins = (25 * sub, 25 * sub)
            axis_names = ('cube_x', 'cube_y')
            # Obs layout: qpos[7] + qvel[7] + ee_pos[3] + finger_qpos[2] +
            # finger_qvel[2] + cube_pos[3] + goal_pos[3]. cube_pos starts at 21.
            lo, hi = 21, 23                 # cube_pos xy slice in obs
        else:  # pick_and_place
            bounds = np.array(
                [[-0.55, 0.55], [0.35, 0.95], [0.05, 0.45]], dtype=np.float64)
            bins = (12 * sub, 12 * sub, 12 * sub)
            axis_names = ('cube_x', 'cube_y', 'cube_z')
            lo, hi = 21, 24                 # cube_pos slice in obs
            grasp_gated = True

        if grasp_gated:
            # Count the cube position only while it is actually in the gripper,
            # measured by the gripper-midpoint->cube distance ``d_ee_cube``
            # (forwarded as ``log_d_ee_cube``). It is ~0 at the init grasp and
            # only grows once the cube leaves the fingers, so this is far more
            # robust than the brittle per-finger grasp detector. The cutoff is
            # the env's own ``lost_grip_thresh`` ("cube slipped out").
            held_thresh = float(self._config.lost_grip_thresh)
            def project(tran, _lo=lo, _hi=hi, _thr=held_thresh):
                state = np.asarray(tran['state'])
                pos = state[..., _lo:_hi].reshape(-1, _hi - _lo)
                held = np.asarray(tran['log_d_ee_cube']).reshape(-1) < _thr
                return pos[held]
        else:
            def project(tran, _lo=lo, _hi=hi):
                return np.asarray(tran['state'])[..., _lo:_hi]

        return dict(
            bounds=bounds,
            bins=bins,
            axis_names=axis_names,
            project=project,
        )

    def _state_component_names(self) -> list[str]:
        names = (
            [f'arm_qpos_{i}' for i in range(7)]
            + [f'arm_qvel_{i}' for i in range(7)]
            + ['ee_x', 'ee_y', 'ee_z'])
        if not self._is_reach:
            names += (
                [f'finger_qpos_{i}' for i in range(2)]
                + [f'finger_qvel_{i}' for i in range(2)]
                + ['cube_x', 'cube_y', 'cube_z'])
        names += ['goal_x', 'goal_y', 'goal_z']
        return names

    def get_state_table(
        self,
        points_per_axis: int = 7,
    ) -> Dict[str, Any]:
        """2D grid over cube (x, y) with the arm grasping the cube at goal z.

        For each grid point: solve IK so the fingertip midpoint sits at
        ``(cube_x, cube_y, goal_z)`` with fingers closed at the grasp width,
        place the cube there, leave the goal marker at the goal-box center.
        ``goal_z`` is the z of the goal-box center (the height the cube has
        to reach for success), so the table represents "cube held aloft at
        target altitude" rather than "cube resting on the table".
        """
        if self._is_reach:
            raise NotImplementedError(
                'get_state_table is cube-based; not supported for "reach".')

        geom = self.get_coverage_geometry(bins_per_cell=1)
        bounds = np.asarray(geom['bounds'], dtype=np.float64)
        xs = np.linspace(bounds[0, 0], bounds[0, 1],
                         points_per_axis, dtype=np.float64)
        ys = np.linspace(bounds[1, 0], bounds[1, 1],
                         points_per_axis, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(xs, ys, indexing='xy')
        grid_shape = np.asarray(grid_x.shape, dtype=np.int32)
        cube_xy = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)  # [N, 2]
        N = cube_xy.shape[0]

        goal_center = 0.5 * (
            np.asarray(self._goal_low, dtype=np.float64)
            + np.asarray(self._goal_high, dtype=np.float64))
        goal_z = float(goal_center[2])
        identity_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        mj_data = mujoco.MjData(self._mj_model)
        obs_dim = self.observation_size
        state = np.empty((N, obs_dim), dtype=np.float32)

        for i, (cx, cy) in enumerate(cube_xy):
            target = np.array([cx, cy, goal_z], dtype=np.float64)
            arm_qpos = self._solve_grasp_ik(target).astype(np.float64)

            qpos = np.array(self._mj_model.qpos0, copy=True, dtype=np.float64)
            qpos[:7] = arm_qpos
            qpos[7:9] = _GRASP_FINGER_QPOS
            qpos[self._cube_qposadr:self._cube_qposadr + 3] = target
            qpos[self._cube_qposadr + 3:self._cube_qposadr + 7] = identity_quat
            qpos[self._goal_qposadr:self._goal_qposadr + 3] = goal_center
            qpos[self._goal_qposadr + 3:self._goal_qposadr + 7] = identity_quat

            mj_data.qpos[:] = qpos
            mj_data.qvel[:] = 0.0
            mujoco.mj_forward(self._mj_model, mj_data)

            ee_pos = np.asarray(mj_data.xpos[self._hand_body], dtype=np.float32)
            cube_pos = np.asarray(mj_data.xpos[self._cube_body], dtype=np.float32)
            goal_pos = np.asarray(mj_data.xpos[self._goal_body], dtype=np.float32)
            state[i] = np.concatenate([
                arm_qpos.astype(np.float32),
                np.zeros(7, dtype=np.float32),
                ee_pos,
                np.full(2, _GRASP_FINGER_QPOS, dtype=np.float32),
                np.zeros(2, dtype=np.float32),
                cube_pos,
                goal_pos,
            ])

        return dict(
            obs={'state': state},
            obs_col_names={'state': self._state_component_names()},
            grid_shape=grid_shape,
        )

    # -- reset / step --------------------------------------------------------

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, rng_goal, rng_cube, rng_arm, rng_cube_noise = jax.random.split(rng, 5)

        # Small uniform noise on the arm joints, applied to both qpos and
        # ctrl so the actuator target moves with the joints (otherwise the
        # arm would spring back to the noise-free pose at step 0 and shake
        # off the init grasp).
        arm_noise = jax.random.uniform(
            rng_arm, (7,),
            minval=-self._config.init_arm_qpos_noise,
            maxval=self._config.init_arm_qpos_noise,
        )
        qpos = self._init_qpos.at[:7].add(arm_noise)
        ctrl = self._init_ctrl.at[:7].add(arm_noise)

        goal_pos = jax.random.uniform(
            rng_goal, (3,), minval=self._goal_low, maxval=self._goal_high)
        qpos = self._set_freejoint_pos(qpos, self._goal_qposadr, goal_pos)
        if not self._is_reach:
            cube_pos = jax.random.uniform(
                rng_cube, (3,), minval=self._cube_low, maxval=self._cube_high)
            # Additive xy jitter on top of the cube sampling box. Stays
            # inside the finger squeeze envelope so the init grasp holds;
            # z is left alone to avoid clipping the table or finger geom.
            cube_xy_noise = jax.random.uniform(
                rng_cube_noise, (2,),
                minval=-self._config.init_cube_xy_noise,
                maxval=self._config.init_cube_xy_noise,
            )
            cube_pos = cube_pos.at[:2].add(cube_xy_noise)
            qpos = self._set_freejoint_pos(qpos, self._cube_qposadr, cube_pos)

        data = mjx_env.init(
            self._mjx_model,
            qpos=qpos,
            qvel=self._init_qvel,
            ctrl=ctrl,
        )

        info = {
            "rng": rng,
            "_steps": jp.array(0, dtype=jp.int32),
        }
        if self._is_vertical:
            # Lift target: straight above the cube spawn at ``_lift_target_z``
            # (the un-raised goal height, 0.20). Carried per-episode so step()'s
            # dense reward tracks the actual spawn xy (incl. init noise / any
            # cube randomization).
            info["lift_target"] = jp.concatenate(
                [cube_pos[:2], jp.asarray([self._lift_target_z], jp.float32)])
        dist = jp.linalg.norm(self._target_pos(data) - self._goal_pos(data))
        zero = jp.array(0.0, dtype=jp.float32)
        metrics = {
            "dist": dist.astype(jp.float32),
            "success": zero,
            # Per-step reward components; summed by log_keys_sum at episode end.
            "log_rew_success": zero,
            "log_rew_velocity": zero,
            "log_rew_reach": zero,
            "log_rew_close": zero,
            "log_rew_lift": zero,
            "log_rew_ctrl": zero,
            "log_rew_healthy": zero,
            # Target-to-goal distance, aggregated by min/avg at episode end as
            # a diagnostic for "how close did the agent get". Independent of
            # the training reward.
            "log_dist": dist.astype(jp.float32),
            # Per-step diagnostics; also summed -> per-episode totals.
            # log_grasp: number of steps the cube was grasped this episode
            # (binary detector retained for plotting, not used by the reward).
            # log_cube_speed: time-integrated cube speed (m/s * steps), so
            #   ``log_cube_speed * ctrl_dt`` is the cube's total path length.
            "log_grasp": zero,
            "log_cube_speed": zero,
        }
        if not self._is_reach:
            cube_xpos = data.xpos[self._cube_body]
            ee_xpos = 0.5 * (data.xpos[self._left_finger_body]
                             + data.xpos[self._right_finger_body])
            # log_d_ee_cube: gripper-midpoint->cube distance, forwarded into the
            # transition (``log_`` prefix) so coverage can gate on "cube in the
            # gripper". ~0 at the init grasp; grows once the cube leaves the
            # fingers. Reset value is the true distance (not zeroed).
            metrics["log_d_ee_cube"] = jp.linalg.norm(
                ee_xpos - cube_xpos).astype(jp.float32)
            metrics["skill_indicator"] = zero

        return mjx_env.State(
            data,
            self._get_obs(data),
            jp.array(0.0, dtype=jp.float32),
            jp.array(0.0, dtype=jp.float32),
            metrics,
            info,
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        action = jp.clip(jp.asarray(action, dtype=jp.float32), -1.0, 1.0)

        # Arm: delta control around the previous ctrl, with a convex leak
        # of the ctrl target toward the current qpos. ctrl_leak=0 reduces
        # to the pure integrator (target leads qpos by accumulated history);
        # ctrl_leak>0 collapses the ctrl-qpos gap each step so action=0
        # ⇒ ctrl→qpos ⇒ arm holds in place wherever the policy left it.
        # No absolute reference is introduced — qpos itself still drifts
        # under noise, just at a (1+leak)x slower rate. See default_config
        # for the noise-rejection analysis.
        leak = self._config.ctrl_leak
        arm_ctrl = (
            (1.0 - leak) * state.data.ctrl[:7]
            + leak * state.data.qpos[:7]
            + self._config.action_scale * action[:7]
        )
        arm_ctrl = jp.clip(arm_ctrl, self._arm_lower, self._arm_upper)
        ctrl = state.data.ctrl.at[:7].set(arm_ctrl)

        # Gripper: absolute control, [-1, 1] -> [finger_lower, finger_upper].
        # For ``reach`` this has no effect on reward, but the actuator exists
        # so the robot model is identical across sub-tasks.
        mid = 0.5 * (self._finger_lower + self._finger_upper)
        half = 0.5 * (self._finger_upper - self._finger_lower)
        finger_ctrl = jp.clip(mid + half * action[7],
                              self._finger_lower, self._finger_upper)
        ctrl = ctrl.at[7].set(finger_ctrl).at[8].set(finger_ctrl)

        data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)

        goal_pos = self._goal_pos(data)
        target_pos = self._target_pos(data)
        dist = jp.linalg.norm(target_pos - goal_pos)
        success = (dist < self._config.success_thresh).astype(jp.float32)

        vel_norm = jp.linalg.norm(self._target_vel(data))

        # Direction-agnostic shaping. reach / slide use the simple proximity
        # bonus + saturating velocity bonus; pick_and_place layers
        # reach + close + lift (see default_config docstring for budget).
        lift_reward = jp.float32(0.0)
        close_reward = jp.float32(0.0)
        # Diagnostics (zero for sub-tasks without a cube/grasp concept).
        grasped_indicator = jp.float32(0.0)
        cube_speed = jp.float32(0.0)
        if self._is_reach:
            reach_reward = jp.float32(0.0)
            d_ee_cube = jp.float32(0.0)
            skill_indicator = jp.float32(0.0)
        else:
            cube_xpos = data.xpos[self._cube_body]
            ee_xpos = 0.5 * (data.xpos[self._left_finger_body]
                             + data.xpos[self._right_finger_body])
            d_ee_cube = jp.linalg.norm(ee_xpos - cube_xpos)
            reach_reward = self._config.reach_reward_scale * (
                1.0 - jp.tanh(d_ee_cube / self._config.reach_scale)
            )
            if self._sub_task == "slide":
                skill_indicator = (
                    d_ee_cube < self._config.contact_thresh
                ).astype(jp.float32)
                # vel_norm is the cube speed for slide (see _target_vel).
                cube_speed = vel_norm
            else:  # pick_and_place
                # Close reward: proximity_grasp-gated gripper-closure bonus.
                # proximity_grasp uses a tight scale so the bonus only pays
                # when fingers actually wrap the cube (not from hovering
                # nearby). finger_closure ∈ [0, 1] from qpos[7] (joint range
                # [0, 0.05], 0 = closed); the XML equality constraint keeps
                # finger_joint2 in lockstep.
                proximity_grasp = 1.0 - jp.tanh(
                    d_ee_cube / self._config.grasp_scale)
                finger_closure = 1.0 - data.qpos[7] / 0.05
                close_reward = (
                    self._config.close_reward_scale
                    * proximity_grasp * finger_closure
                )
                # Continuous cube height above rest (true rest at z = 0.03,
                # the cube half-extent above the table plane). Any lift
                # earns reward immediately — no dead zone. Capped at
                # ``_lift_cap_z`` (configured goal height minus
                # ``success_thresh``, anchored in __init__ before the
                # init-variant goal raise) so the saturation point stays
                # put when the goal moves up.
                lift_cap_z = self._lift_cap_z
                max_lift_height = jp.maximum(
                    lift_cap_z - self._config.cube_rest_z, 0.0)
                lift_height = jp.clip(
                    cube_xpos[2] - self._config.cube_rest_z,
                    0.0,
                    max_lift_height,
                )
                # Center to [-scale, +scale]: cube at rest → -scale,
                # cube at lift cap → +scale. Mirrors ant's signed speed
                # reward (stationary → 0, away from goal → negative).
                if self._is_vertical:
                    # Vertical-lift reward: signed/normalized closeness to the
                    # lift target straight above the spawn. Penalizes
                    # horizontal drift (any xy deviation grows ``d_lift``).
                    # Same [-scale, +scale] shape as the height reward but on
                    # distance-to-point instead of raw height. Goal-agnostic:
                    # the target is over the spawn, not the displaced goal.
                    d_lift = jp.linalg.norm(
                        cube_xpos - state.info["lift_target"])
                    progress = jp.clip(
                        self._lift_span - d_lift, 0.0, self._lift_span)
                    lift_norm = (
                        2.0 * progress / jp.maximum(self._lift_span, 1e-6) - 1.0
                    )
                else:
                    lift_norm = (
                        2.0 * lift_height / jp.maximum(max_lift_height, 1e-6) - 1.0
                    )
                lift_reward = self._config.lift_reward_scale * lift_norm
                # Binary grasp detector retained as a logging diagnostic
                # only; the reward does not depend on it. grasp_thresh
                # tightened to 0.04 m so the indicator reflects "fingers
                # close to the cube" rather than "hand in the neighborhood".
                d_left = jp.linalg.norm(
                    data.xpos[self._left_finger_body] - cube_xpos)
                d_right = jp.linalg.norm(
                    data.xpos[self._right_finger_body] - cube_xpos)
                grasped_indicator = (
                    (d_left < self._config.grasp_thresh)
                    & (d_right < self._config.grasp_thresh)
                ).astype(jp.float32)
                cube_speed = jp.linalg.norm(self._target_vel(data))
                skill_indicator = grasped_indicator

        # Success: one-shot terminal bonus of ``5 * episode_length`` when
        # ``terminate_on_success`` is True; otherwise per-step camping at
        # ``success_reward``. The terminal-bonus variant mirrors ant.
        if self._config.terminate_on_success:
            rew_success = jp.float32(5 * self._config.episode_length) * success
        else:
            rew_success = self._config.success_reward * success
        # Healthy reward: ant-style penalty paid per step while the gripper
        # has lost the cube (``d_ee_cube > lost_grip_thresh``). Only
        # meaningful in init-grasp mode; zeroed elsewhere because the
        # threshold would fire from initial state.
        if self._init_grasp:
            is_healthy = (d_ee_cube <= self._config.lost_grip_thresh).astype(jp.float32)
            healthy_reward = self._config.healthy_reward * (1.0 - is_healthy)
        else:
            healthy_reward = jp.float32(0.0)
        # Velocity bonus (slide/reach only). For pick_and_place(_init) the lift
        # term replaces it — rewarding cube velocity would just reward
        # shaking the cube.
        if self._is_pap:
            rew_velocity = jp.float32(0.0)
        else:
            rew_velocity = self._config.vel_reward_scale * (
                jp.minimum(vel_norm, self._config.vel_max) / self._config.vel_max
            )
        # Quadratic control cost on the raw (clipped) action. Logged as a
        # negative reward component so episode sums are directly comparable
        # to the positive shaping terms.
        rew_ctrl = -self._config.ctrl_cost_scale * jp.sum(action ** 2)
        reward = (
            rew_success
            + rew_velocity
            + reach_reward
            + close_reward
            + lift_reward
            + healthy_reward
            + rew_ctrl
        )

        steps = state.info["_steps"] + 1
        if self._config.terminate_on_success:
            done = (success > 0.5).astype(jp.float32)
        else:
            done = jp.float32(0.0)
        if self._terminate_on_lost_grip:
            lost_grip = (d_ee_cube > self._config.lost_grip_thresh).astype(jp.float32)
            done = jp.maximum(done, lost_grip)

        state.info["_steps"] = steps
        metric_updates = {
            "dist": dist.astype(jp.float32),
            "success": success,
            "log_rew_success": rew_success.astype(jp.float32),
            "log_rew_velocity": rew_velocity.astype(jp.float32),
            "log_rew_reach": reach_reward.astype(jp.float32),
            "log_rew_close": close_reward.astype(jp.float32),
            "log_rew_lift": lift_reward.astype(jp.float32),
            "log_rew_ctrl": rew_ctrl.astype(jp.float32),
            "log_rew_healthy": healthy_reward.astype(jp.float32),
            "log_grasp": grasped_indicator.astype(jp.float32),
            "log_cube_speed": cube_speed.astype(jp.float32),
            "log_dist": dist.astype(jp.float32),
        }
        if not self._is_reach:
            metric_updates["log_d_ee_cube"] = d_ee_cube.astype(jp.float32)
            metric_updates["skill_indicator"] = skill_indicator
        state.metrics.update(**metric_updates)

        return state.replace(
            data=data,
            obs=self._get_obs(data),
            reward=reward,
            done=done,
        )

    # -- observation / target helpers ---------------------------------------

    def _get_obs(self, data: mjx.Data) -> jax.Array:
        parts = [
            data.qpos[:7].astype(jp.float32),
            data.qvel[:7].astype(jp.float32),
            data.xpos[self._hand_body].astype(jp.float32),
        ]
        if not self._is_reach:
            # Finger qpos + qvel so the policy can observe its own gripper
            # state. Without these, close_reward gates on a quantity the
            # policy can't see.
            parts.append(data.qpos[7:9].astype(jp.float32))
            parts.append(data.qvel[7:9].astype(jp.float32))
            parts.append(data.xpos[self._cube_body].astype(jp.float32))
        parts.append(self._goal_pos(data).astype(jp.float32))
        return jp.concatenate(parts)

    def _goal_pos(self, data: mjx.Data) -> jax.Array:
        return data.xpos[self._goal_body]

    def _target_pos(self, data: mjx.Data) -> jax.Array:
        if self._is_reach:
            return data.xpos[self._hand_body]
        return data.xpos[self._cube_body]

    def _target_vel(self, data: mjx.Data) -> jax.Array:
        """Linear velocity (3-vector) of the object the agent is moving."""
        if self._is_reach:
            # cvel layout in MuJoCo: [angular (3), linear (3)] in world frame.
            return data.cvel[self._hand_body, 3:6]
        return data.qvel[self._cube_qveladr:self._cube_qveladr + 3]
