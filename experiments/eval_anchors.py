"""Anchor (state, action) specs for policy-evaluation experiments.

Each anchor defines a physics state (qpos, qvel) and an action for one of the
DMC tasks {cartpole_swingup, walker_run, cheetah_run} or corridor point maze. We use proprioceptive
(state-based) observations, so the obs is fully determined by qpos/qvel.

Per non-natural anchor we define a 20-point 1D state sweep on the most
informative qpos/qvel component (pole_theta for cartpole, rootz for walker,
vx for cheetah). MC ground-truth and TD readout are evaluated at the anchor
itself plus every sweep point. The natural_reset anchor contributes a single
point and no sweeps.

DMC qpos/qvel layouts (suite domains, dm_control v1.x):
  cartpole_swingup
    qpos[0] = cart_x          velocity[0] = cart_vx
    qpos[1] = pole_theta      velocity[1] = pole_omega
    action[0] in [-1, 1]      (theta=0 upright, theta=pi hanging)
  walker (run)
    qpos[0] = rootz offset    qvel[0] = vertical velocity
    qpos[1] = rootx
    qpos[2] = rooty           (pitch; 0 = upright)
    qpos[3..5] = right hip/knee/ankle
    qpos[6..8] = left hip/knee/ankle
    action[0..2] = right hip/knee/ankle torque
    action[3..5] = left  hip/knee/ankle torque
  cheetah (run)
    qpos[0] = rootx
    qpos[1] = rootz
    qpos[2] = rooty           (pitch)
    qpos[3..8] = bthigh, bshin, bfoot, fthigh, fshin, ffoot
    action[0..5] mirror joint order

NB: rootx never enters the proprio obs (walker uses orientations/height;
cheetah drops it via qpos[1:]). It is
included in qpos here only because dm_control's physics.named.data.qpos has
length 9 for walker/cheetah and we set the full array.
"""

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class Anchor:
    name: str
    tag: str                              # 'ID', 'ID-transient', 'OOD', 'natural'
    qpos: np.ndarray
    qvel: np.ndarray
    action: np.ndarray
    natural_reset: bool = False           # if True, skip set_physics and let
                                          # dm_control's default init drive the
                                          # start state. qpos/qvel ignored.
    state_sweep: dict = field(default_factory=dict)   # {'name', 'kind', 'index', 'values'}
    action_sweep: dict = field(default_factory=dict)  # {'name', 'index', 'values'}


def _arr(xs):
    return np.asarray(xs, dtype=np.float32)


# -------------------------- cartpole_swingup --------------------------
# qpos = [cart_x, pole_theta]; qvel = [cart_vx, pole_omega]; action[0] in [-1, 1].
# Sweep dims: state = pole_omega (qvel[1]); action = cart_force (action[0]).
# Anchors all start at omega=0 so the state sweep is the differentiator.
# Note: dm_control's natural reset is near-hanging, so theta=pi at center is
# IN-distribution; OOD has to come from cart_x at the rail or large cart_vx.

_CARTPOLE_OMEGA_SWEEP = np.linspace(-12.0, 12.0, 20, dtype=np.float32)
_CARTPOLE_ACTION_SWEEP = np.linspace(-1.0, 1.0, 20, dtype=np.float32)

def _cartpole_anchor(name, tag, qpos, qvel, action=(0.0,)):
    return Anchor(
        name=name, tag=tag,
        qpos=_arr(qpos), qvel=_arr(qvel), action=_arr(action),
        state_sweep=dict(name='pole_omega', kind='qvel', index=1,
                         values=_CARTPOLE_OMEGA_SWEEP),
        action_sweep=dict(name='cart_force', index=0,
                          values=_CARTPOLE_ACTION_SWEEP),
    )

CARTPOLE_SWINGUP = [
    Anchor(
        name='natural_reset', tag='natural',
        qpos=_arr([0.0, 0.0]), qvel=_arr([0.0, 0.0]),  # ignored
        action=_arr([0.0]),
        natural_reset=True,
    ),
    # --- 3 ID anchors (distinguished by theta and cart state, not by omega) ---
    _cartpole_anchor('upright_balanced', 'ID',
        qpos=[0.0, 0.0], qvel=[0.0, 0.0]),
    _cartpole_anchor('near_top_offset', 'ID',
        qpos=[0.0, 0.6], qvel=[0.0, 0.0]),
    _cartpole_anchor('midswing_climbing', 'ID',
        qpos=[0.4, 1.5], qvel=[0.5, 0.0]),
    # --- 3 OOD anchors (rail proximity / cart speed far beyond policy data) ---
    _cartpole_anchor('rail_pinned_hanging', 'OOD',
        qpos=[-1.85, np.pi], qvel=[-0.5, 0.0]),
    _cartpole_anchor('racing_upright', 'OOD',
        qpos=[0.0, 0.0], qvel=[6.0, 0.0]),
    _cartpole_anchor('rail_charging_hanging', 'OOD',
        qpos=[1.5, np.pi], qvel=[4.0, 0.0]),
]


# -------------------------- walker_run --------------------------
# qpos[9] = [rootz offset, rootx, rooty, R-hip, R-knee, R-ankle, L-hip, L-knee, L-ankle]
# qvel[9] mirrors, with [vz, vx, omega_y] for the first three.
# action[6] = right-{hip,knee,ankle}, left-{hip,knee,ankle}, in [-1, 1].
# Sweep dims: state = rootz offset (qpos[0]); torso height = 1.3 + rootz; action = R-hip (action[0]).
# OOD anchors are differentiated on pitch (rooty), vx, vz, and omega_y so they
# don't collapse to the ID anchors when rootz is replaced by the sweep.

_WALKER_ROOTZ_SWEEP = np.linspace(0.2, 1.8, 20, dtype=np.float32)
_WALKER_ACTION_SWEEP = np.linspace(-1.0, 1.0, 20, dtype=np.float32)

def _walker_qpos(rootz=1.30, rooty=0.0, r_hip=0.0, r_knee=0.0, r_ankle=0.0,
                 l_hip=0.0, l_knee=0.0, l_ankle=0.0):
    return _arr([rootz - 1.30, 0.0, rooty, r_hip, r_knee, r_ankle, l_hip, l_knee, l_ankle])

def _walker_qvel(vx=0.0, vz=0.0, omega=0.0, r=(0.0, 0.0, 0.0), l=(0.0, 0.0, 0.0)):
    return _arr([vz, vx, omega, r[0], r[1], r[2], l[0], l[1], l[2]])

def _walker_anchor(name, tag, qpos, qvel, action=(0.0,) * 6):
    return Anchor(
        name=name, tag=tag, qpos=qpos, qvel=qvel, action=_arr(action),
        state_sweep=dict(name='rootz_offset', kind='qpos', index=0,
                         values=_WALKER_ROOTZ_SWEEP - 1.30),
        action_sweep=dict(name='R_hip_torque', index=0,
                          values=_WALKER_ACTION_SWEEP),
    )

WALKER_RUN = [
    Anchor(
        name='natural_reset', tag='natural',
        qpos=_walker_qpos(), qvel=_walker_qvel(),       # ignored
        action=_arr([0.0] * 6),
        natural_reset=True,
    ),
    # --- 3 ID anchors (close to on-policy visitation; rootz is the swept dim) ---
    # Ordering matches measured visitation distance (lowest first):
    #   stand_neutral ~3.5, backward_walking ~4.5, stride_R_forward ~5.5.
    _walker_anchor('stand_neutral', 'ID',
        qpos=_walker_qpos(rootz=1.30, rooty=0.0),
        qvel=_walker_qvel()),
    # Was OOD, but the policy's joint config under reverse motion is close
    # enough to its forward gait that the proprio distance is small.
    _walker_anchor('backward_walking', 'ID',
        qpos=_walker_qpos(rootz=1.20, rooty=0.0),
        qvel=_walker_qvel(vx=-2.0)),
    _walker_anchor('stride_R_forward', 'ID',
        qpos=_walker_qpos(rootz=1.25, rooty=0.05,
                          r_hip=0.4, r_knee=-0.3, l_hip=-0.4, l_knee=-0.1),
        qvel=_walker_qvel(vx=3.5),
        action=(0.3, -0.2, 0.0, -0.3, 0.2, 0.0)),
    # --- 3 OOD anchors (each pushed harder along multiple proprio dims since
    # walker has 24-dim proprio and z-scoring compresses single-dim deviations) ---
    # Was 'pushoff_extended' (ID) but distance ~6.5 — closer to OOD than ID.
    # Exaggerated rootz + vz so it lives clearly off-trajectory.
    _walker_anchor('leap_high', 'OOD',
        qpos=_walker_qpos(rootz=1.60, rooty=0.0, r_knee=-0.5, l_knee=-0.5),
        qvel=_walker_qvel(vx=4.5, vz=3.0),
        action=(0.5, 0.5, 0.0, 0.5, 0.5, 0.0)),
    # Was 'fallen_pitch_forward' (rootz=0.30, pitch=1.6, dist ~6.1) —
    # pushed deeper: nearly face-down at very low height.
    _walker_anchor('fallen_pitch_forward', 'OOD',
        qpos=_walker_qpos(rootz=0.20, rooty=2.5),
        qvel=_walker_qvel()),
    # Was (rootz=1.65, vz=3, omega=3, dist ~6.4). Cranked vz + omega so the
    # tumbling state is unambiguously off-distribution.
    _walker_anchor('aerial_tumbling', 'OOD',
        qpos=_walker_qpos(rootz=1.70, rooty=-0.8, r_knee=-0.5, l_knee=-0.5),
        qvel=_walker_qvel(vz=5.0, omega=5.0)),
]


# -------------------------- cheetah_run --------------------------
# qpos[9] = [rootx, rootz, rooty, bthigh, bshin, bfoot, fthigh, fshin, ffoot]
# qvel[9] = [vx, vz, omega_y, *joint_vels]
# action[6] = back-{thigh,shin,foot}, front-{thigh,shin,foot}, in [-1, 1].
# Sweep dims: state = vx (qvel[0]); action = bthigh (action[0]).
# ID anchors are different gait phases; OOD anchors are differentiated on
# pitch (rooty), rootz, and omega_y so they stay OOD across the vx sweep.

_CHEETAH_VX_SWEEP = np.linspace(-5.0, 12.0, 20, dtype=np.float32)
_CHEETAH_ACTION_SWEEP = np.linspace(-1.0, 1.0, 20, dtype=np.float32)

def _cheetah_qpos(rootz=0.0, rooty=0.0, joints=(0.0,) * 6):
    return _arr([0.0, rootz, rooty, *joints])

def _cheetah_qvel(vx=0.0, vz=0.0, omega=0.0, joints=(0.0,) * 6):
    return _arr([vx, vz, omega, *joints])

def _cheetah_anchor(name, tag, qpos, qvel, action=(0.0,) * 6):
    return Anchor(
        name=name, tag=tag, qpos=qpos, qvel=qvel, action=_arr(action),
        state_sweep=dict(name='vx', kind='qvel', index=0,
                         values=_CHEETAH_VX_SWEEP),
        action_sweep=dict(name='bthigh_torque', index=0,
                          values=_CHEETAH_ACTION_SWEEP),
    )

CHEETAH_RUN = [
    Anchor(
        name='natural_reset', tag='natural',
        qpos=_cheetah_qpos(), qvel=_cheetah_qvel(),     # ignored
        action=_arr([0.0] * 6),
        natural_reset=True,
    ),
    # --- 3 ID anchors (close to on-policy visitation; vx is swept) ---
    # Ordering matches measured visitation distance (lowest first):
    #   crouched_low ~8.5, stride_phase ~10.3, tilted_run ~17.
    _cheetah_anchor('crouched_low', 'ID',
        qpos=_cheetah_qpos(rootz=-0.25),
        qvel=_cheetah_qvel()),
    _cheetah_anchor('stride_phase', 'ID',
        qpos=_cheetah_qpos(joints=(0.5, -0.3, 0.0, -0.4, 0.2, 0.0)),
        qvel=_cheetah_qvel(),
        action=(0.5, -0.3, 0.0, -0.3, 0.3, 0.0)),
    # Was 'tumbling_spin' with omega_y=4 + rooty=0.8 — softened to a tilted
    # running pose to push it firmly inside the policy's visitation envelope.
    _cheetah_anchor('tilted_run', 'ID',
        qpos=_cheetah_qpos(rooty=0.3),
        qvel=_cheetah_qvel(omega=2.0)),
    # --- 3 OOD anchors (far from on-policy visitation) ---
    # Ordering matches measured visitation distance (lowest first):
    #   hard_landing ~22 (after exaggeration), pushoff_airborne ~45, flipped ~73.
    # Was 'landing_recovery' with rootz=0.05, vz=-1 — exaggerated rootz and vz
    # so the "violent landing from height" state actually lives outside the
    # visited envelope.
    _cheetah_anchor('hard_landing', 'OOD',
        qpos=_cheetah_qpos(rootz=0.35,
                           joints=(0.5, -0.6, 0.0, 0.5, -0.6, 0.0)),
        qvel=_cheetah_qvel(vz=-3.5)),
    _cheetah_anchor('pushoff_airborne', 'OOD',
        qpos=_cheetah_qpos(rootz=0.2, rooty=-0.1,
                           joints=(0.6, -0.5, 0.0, 0.6, -0.5, 0.0)),
        qvel=_cheetah_qvel(vz=2.0),
        action=(0.6, 0.5, 0.0, 0.6, 0.5, 0.0)),
    _cheetah_anchor('flipped', 'OOD',
        qpos=_cheetah_qpos(rootz=-0.1, rooty=np.pi),
        qvel=_cheetah_qvel()),
]


def pointmaze_corridor_anchors():
    # Use production layout coordinates. All points are in the start-connected
    # component; the isolated G cell is deliberately not an exploration anchor.
    from embodied.envs.custom_envs.locomotion.point_maze_env import (
        CORRIDORS_MAZE_LAYOUT, _MAZE_CELL_SIZE, _cell_center)
    layout = CORRIDORS_MAZE_LAYOUT
    specs = [('start', 1, 7, 0), ('left_turn', 2, 2, 1),
             ('right_turn', 2, 12, 1), ('left_middle', 7, 2, 1),
             ('right_middle', 7, 12, 1), ('left_end', 13, 2, 0),
             ('right_end', 13, 12, 0)]
    anchors = []
    for name, row, col, axis in specs:
        xy = _arr(_cell_center(row, col, len(layout), len(layout[0]),
                               _MAZE_CELL_SIZE, positive_coords=True))
        # Keep sphere radius .35 plus .10 clearance from cell edges, even
        # beside walls. Sweep inside the selected cell, never through walls.
        half = _MAZE_CELL_SIZE / 2 - .45
        anchors.append(Anchor(
            name=name, tag='start' if name == 'start' else 'probe',
            qpos=xy, qvel=_arr([0, 0]), action=_arr([0, 0]),
            state_sweep=dict(name='xy'[axis], kind='qpos', index=axis,
                             values=np.linspace(xy[axis] - half, xy[axis] + half, 20)),
            action_sweep=dict(name='action_' + 'xy'[axis], index=axis,
                              values=np.linspace(-1, 1, 20))))
    start = anchors[0]
    return [Anchor('natural_reset', 'natural', start.qpos.copy(),
                   start.qvel.copy(), start.action.copy(), natural_reset=True)] + anchors


ANCHORS = {
    'dmc_cartpole_swingup': CARTPOLE_SWINGUP,
    'dmc_cartpole_swingup_sparse': CARTPOLE_SWINGUP,
    'dmc_walker_run': WALKER_RUN,
    'dmc_cheetah_run': CHEETAH_RUN,
}


def get_anchors(task: str) -> Sequence[Anchor]:
    if task == 'mjp_pointmaze_corridors':
        return pointmaze_corridor_anchors()
    if task not in ANCHORS:
        raise KeyError(f'No anchors defined for task {task!r}. Known: {list(ANCHORS)}.')
    return ANCHORS[task]
