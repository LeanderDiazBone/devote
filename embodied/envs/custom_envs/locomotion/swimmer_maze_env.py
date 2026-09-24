"""5-link swimmer (4 actuated joints) navigating the same maze layouts as
``ant_maze_env``. Structure mirrors ``ant_maze_env.py`` so the wrapper, the
launcher, and the coverage tracker can be reused without changes.

Why a swimmer:
- 4-D action space (4 inter-link hinges); 16-D state observation
  (nq=7, nv=7, +2 goal). Much smaller than the ant (8-D action, ~30-D
  obs) while keeping a real motor-skill learning problem.
- Locomotion is direction-agnostic: the dense speed-shaping reward
  teaches undulation, the sparse goal reward provides the exploration
  signal. No "alive"/"unhealthy" failure mode (planar joint constrains
  z), so the reward design has one fewer knob to tune.
- The planar torso joint (slider_x, slider_y, rot_z) is unactuated;
  the swimmer moves by coordinated firing of the 4 internal hinges,
  with reaction forces provided by MuJoCo's Stokes-drag fluid model.
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx
from etils import epath
from ml_collections import config_dict

from mujoco_playground._src import mjx_env

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


# 10 m cells: the swimmer uses the canonical gym dimensions (1 m per link,
# ~5 m total length, capsule radius 0.1 m, gear 150). Anything smaller and
# the fluid drag forces fall below the numerical damping from `armature`,
# and the swimmer's body length crowds the maze cells. Walls extend from
# z=0 upward so they collide with the planar swimmer at z=0.
_MAZE_CELL_SIZE = 10.0
_MAZE_WALL_HALF_HEIGHT = 1.5

_MAZE_LAYOUTS = {
    "room": ROOM_MAZE_LAYOUT,
    "miniroom": MINI_ROOM_MAZE_LAYOUT,
    "bigroom": BIG_ROOM_MAZE_LAYOUT,
    "biggestroom": BIGGEST_ROOM_MAZE_LAYOUT,
    "square": SQUARE_MAZE_LAYOUT,
    "bigsquare": BIG_SQUARE_MAZE_LAYOUT,
    "spiral": SPIRAL_MAZE_LAYOUT,
    "wallroom": WALL_ROOM_MAZE_LAYOUT,
    "escape": ESCAPE_MAZE_LAYOUT,
    "largeescape": LARGE_ESCAPE_MAZE_LAYOUT,
    "largespiral": LARGE_SPIRAL_MAZE_LAYOUT,
    "corridors": CORRIDORS_MAZE_LAYOUT,
}


def _cell_center(row: int, col: int, rows: int, cols: int, cell_size: float) -> tuple[float, float]:
    x = (col - (cols - 1) / 2.0) * cell_size
    y = ((rows - 1) / 2.0 - row) * cell_size
    return x, y


def _make_maze_geometry(
    layout: tuple[str, ...],
    cell_size: float = _MAZE_CELL_SIZE,
    wall_half_height: float = _MAZE_WALL_HALF_HEIGHT,
) -> tuple[str, tuple[float, float], tuple[float, float]]:
    rows, cols = len(layout), len(layout[0])
    # Slightly overlap neighboring wall cells to avoid seams/gaps at edges.
    wall_half = cell_size * 0.55
    wall_geoms = []
    start_xy = None
    goal_xy = None

    for r, row in enumerate(layout):
        if len(row) != cols:
            raise ValueError("Maze layout must be rectangular.")
        for c, ch in enumerate(row):
            x, y = _cell_center(r, c, rows, cols, cell_size)
            if ch == "#":
                wall_geoms.append(
                    f'<geom name="maze_wall_{r}_{c}" type="box" '
                    f'pos="{x:.3f} {y:.3f} {wall_half_height:.3f}" '
                    f'size="{wall_half:.3f} {wall_half:.3f} {wall_half_height:.3f}" '
                    f'rgba="0.55 0.4 0.3 1"/>'
                )
            elif ch == "S":
                start_xy = (x, y)
            elif ch == "G":
                goal_xy = (x, y)

    if start_xy is None or goal_xy is None:
        raise ValueError("Maze layout must contain both 'S' and 'G'.")

    goal_marker = (
        f'<geom name="goal_marker" type="cylinder" contype="0" conaffinity="0" '
        f'pos="{goal_xy[0]:.3f} {goal_xy[1]:.3f} 0.03" size="0.35 0.03" '
        f'rgba="0.15 0.85 0.25 0.9"/>'
    )
    maze_xml = "\n    ".join([*wall_geoms, goal_marker])
    return maze_xml, start_xy, goal_xy


# ----------------------------
# Config
# ----------------------------
def default_config():
    _, _, maze_goal = _make_maze_geometry(ROOM_MAZE_LAYOUT)
    return config_dict.create(
        ctrl_dt=0.05,
        sim_dt=0.01,
        episode_length=500,
        action_repeat=1,
        goal=jp.array(maze_goal),        # (x,y) in world coords
        # 2 m threshold matched to the larger 10 m cell + 5 m swimmer body:
        # the COM has to be within 2 m of the goal, which means most of the
        # body is overlapping the goal cylinder.
        success_thresh=2.0,
        # Speed shaping = lin*|v_xy| + quad*|v_xy|^2. Calibrated for the
        # swimmer's ~1 m/s top speed: at v=1 m/s the per-step bonus is
        # 0.05 + 0.20 = 0.25. The linear term keeps gradient at v=0; the
        # quadratic term takes over above v = lin/quad = 0.25 m/s and
        # pushes toward fast swimming. Per-episode max shaping ~125,
        # leaving the sparse return dominant.
        speed_lin_scale=0.05,
        speed_quad_scale=0.20,
        ctrl_cost=0.0,                   # coefficient on sum(action^2); off by default
        contact_cost=5e-4,               # coefficient on sum(clip(cfrc_ext, -1, 1)^2)
        # Sparse goal: +1 per step while within success_thresh of the goal.
        # Episode does NOT terminate on success, so the optimal policy is to
        # reach the goal and camp there for the rest of the episode.
        success_reward=1.0,
        dense_goal_reward=0.0,           # scales -dist_to_goal
    )


# ----------------------------
# Assets / XML
# ----------------------------
_ROOT = epath.Path(__file__).parent

def get_assets():
    return {}


def _build_swimmer_maze_xml(maze_geoms_xml: str) -> str:
    # The swimmer.xml fragment is a 5-link / 4-actuator planar swimmer
    # adapted from the OpenAI Gym MuJoCo swimmer (the de facto reference
    # for this morphology, originally 3-link), extended from 3 to 5 links
    # so we get 4 actuated joints. Scaled to ~2.5 m total length so the
    # agent fits comfortably inside the 6 m maze cells used by the ant
    # maze infrastructure. The dm_control suite uses the same template:
    #   gym:        github.com/openai/gym/blob/master/gym/envs/mujoco/assets/swimmer.xml
    #   dm_control: github.com/google-deepmind/dm_control/blob/main/dm_control/suite/swimmer.xml
    # The 3-DOF planar torso joint (slider_x, slider_y, rot_z) is
    # unactuated; locomotion arises from coordinated firing of the 4
    # inter-link hinges, with reaction forces provided by Stokes drag
    # (configured via the wrapper's option viscosity below).
    swimmer_fragment = (_ROOT / "assets" / "xmls" / "swimmer.xml").read_text()
    # The fragment ends with </mujoco>; strip it so we can splice in the
    # maze walls + the outer <mujoco> wrapper. Same convention as ant.xml.
    swimmer_fragment = swimmer_fragment.split("</mujoco>", 1)[0].rstrip()
    swimmer_fragment = swimmer_fragment.replace(
        "</worldbody>",
        f"""
    {maze_geoms_xml}
  </worldbody>""",
        1,
    )
    # Outer wrapper. Matches the OpenAI Gym swimmer's <option> verbatim
    # (viscosity + density activate the passive fluid-drag model -- the
    # swimmer's only source of propulsion). Differences from ant wrapper:
    #   - viscosity / density set to gym defaults (= dm_control swimmer).
    #   - <default> block specifies geom contype/conaffinity/condim and
    #     joint armature for all bodies in the swimmer fragment.
    return f"""
<mujoco model="swimmer_maze">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option timestep="0.01" integrator="RK4" viscosity="0.1" density="4000"/>
  <size njmax="2000" nconmax="200"/>
  <default>
    <geom rgba="0.8 0.6 0.4 1" contype="1" conaffinity="1" condim="1"/>
    <joint armature="0.1"/>
  </default>
  <asset>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="1 1" rgba="0.93 0.87 0.76 1"/>
  </asset>
  {swimmer_fragment}
</mujoco>
""".strip()


class SwimmerMaze(mjx_env.MjxEnv):
    def __init__(
        self,
        subtask=None,
        config=default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, float, list[Any]]]] = None,
    ):
        mjx_env.MjxEnv.__init__(self, config, config_overrides)

        maze_variant = subtask if subtask else "room"
        if maze_variant not in _MAZE_LAYOUTS:
            raise ValueError(
                f"Unknown subtask: {subtask!r}. "
                f"Choose from {list(_MAZE_LAYOUTS.keys())}"
            )
        self._maze_layout = _MAZE_LAYOUTS[maze_variant]
        maze_geoms_xml, start_xy, maze_goal = _make_maze_geometry(self._maze_layout)
        self._start_xy = jp.array(start_xy, dtype=jp.float32)
        self._goal_xy = jp.array(maze_goal, dtype=jp.float32)

        xml = _build_swimmer_maze_xml(maze_geoms_xml)
        mj_model = mujoco.MjModel.from_xml_string(xml, assets=get_assets())
        mj_model.opt.timestep = self._config.sim_dt

        self._mj_model = mj_model
        self._mjx_model = mjx.put_model(mj_model)

        # qpos layout: [slider_x, slider_y, rot_z, j2, j3, j4, j5] -- all
        # scalar DOFs, no quaternion. qpos0 is all zeros (swimmer at origin,
        # untilted, fully extended); reset overrides [:2] with start_xy.
        self._init_qpos = jp.asarray(mj_model.qpos0, dtype=jp.float32)
        self._init_qvel = jp.zeros(mj_model.nv, dtype=jp.float32)

        self._action_size = int(self._mj_model.nu)

    # ---- required properties used by your wrapper ----
    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property
    def xml_path(self) -> str:
        return "<in-memory-swimmer-maze-xml>"

    def get_coverage_geometry(self, bins_per_cell: int = 1) -> dict:
        """Uniform-binning geometry over the maze's (x, y) cells.

        Identical to AntMaze's coverage tracker: the swimmer's COM xy
        (qpos[:2]) is the first slice of the observation, so the project
        callback is the same.
        """
        k = int(bins_per_cell)
        sub = int(round(np.sqrt(k)))
        if sub * sub != k or sub < 1:
            raise ValueError(
                f'coverage_bins_per_cell must be a positive perfect square; got {k}')
        layout = tuple(self._maze_layout)
        rows, cols = len(layout), len(layout[0])
        cs = float(_MAZE_CELL_SIZE)
        x_extent = cols * cs / 2.0
        y_extent = rows * cs / 2.0
        open_layout = np.array(
            [[ch != '#' for ch in row] for row in layout], dtype=bool)
        # valid_mask[x_bin, y_bin]; y_bin increases upward (row 0 = top => max y_bin).
        valid_mask = np.flipud(open_layout).T.copy()
        if sub > 1:
            valid_mask = np.kron(valid_mask, np.ones((sub, sub), dtype=bool))
        return dict(
            bounds=np.array(
                [[-x_extent, x_extent],
                 [-y_extent, y_extent]], dtype=np.float64),
            bins=(cols * sub, rows * sub),
            axis_names=('x', 'y'),
            valid_mask=valid_mask,
            project=lambda tran: np.asarray(tran['state'])[..., :2],
        )

    # ----------------------------
    # RL API
    # ----------------------------
    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, k1, k2 = jax.random.split(rng, 3)
        qpos = self._init_qpos + 0.01 * jax.random.normal(k1, self._init_qpos.shape)
        qvel = self._init_qvel + 0.01 * jax.random.normal(k2, self._init_qvel.shape)
        qpos = qpos.at[:2].set(self._start_xy)
        data = mjx_env.init(
            self._mjx_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=jp.zeros(self._mj_model.nu, dtype=jp.float32),
        )

        info = {
            "rng": rng,
            "_steps": jp.array(0, dtype=jp.int32),
        }
        zero = jp.array(0.0, dtype=jp.float32)
        metrics = {
            "dist": jp.linalg.norm(self._start_xy - self._goal_xy).astype(jp.float32),
            "success": zero,
            # Per-step reward components; summed by log_keys_sum at episode end.
            "log_rew_speed": zero,
            "log_rew_ctrl": zero,
            "log_rew_contact": zero,
            "log_rew_success": zero,
            "log_rew_dense_goal": zero,
        }

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
        data0 = state.data
        data = mjx_env.step(self._mjx_model, data0, action, self.n_substeps)

        # Linear + squared torso xy speed. Linear term keeps gradient at v=0;
        # quadratic term dominates above v = lin/quad and rewards fast
        # swimming. qvel[0:2] are slider_x, slider_y -- the planar COM
        # velocity in world frame.
        torso_speed = jp.linalg.norm(data.qvel[:2])
        speed_reward = (
            self._config.speed_lin_scale * torso_speed
            + self._config.speed_quad_scale * jp.square(torso_speed)
        )

        ctrl_cost = self._config.ctrl_cost * jp.sum(jp.square(action))
        contact_cost = self._config.contact_cost * jp.sum(
            jp.square(jp.clip(data.cfrc_ext, -1.0, 1.0))
        )

        # qpos[0:2] are slider_x, slider_y -- the planar COM position.
        xy = data.qpos[:2]
        dist = jp.linalg.norm(xy - self._goal_xy)
        success = (dist < self._config.success_thresh).astype(jp.float32)

        reward = (
            speed_reward
            - ctrl_cost
            - contact_cost
            + self._config.success_reward * success
            - self._config.dense_goal_reward * dist
        ).astype(jp.float32)

        steps = state.info["_steps"] + 1
        # No "unhealthy" termination: the planar joint constrains the
        # swimmer to z=0 and there's no fall mode. Always run to timeout.
        done = (steps >= self._config.episode_length).astype(jp.float32)

        state.info["_steps"] = steps
        state.metrics.update(
            dist=dist.astype(jp.float32),
            success=success,
            log_rew_speed=speed_reward.astype(jp.float32),
            log_rew_ctrl=(-ctrl_cost).astype(jp.float32),
            log_rew_contact=(-contact_cost).astype(jp.float32),
            log_rew_success=(self._config.success_reward * success).astype(jp.float32),
            log_rew_dense_goal=(-self._config.dense_goal_reward * dist).astype(jp.float32),
        )

        return state.replace(
            data=data,
            obs=self._get_obs(data),
            reward=reward,
            done=done,
        )

    def _get_obs(self, data: mjx.Data) -> jax.Array:
        # 16-dim obs: qpos (7) + qvel (7) + goal_xy (2). qpos[:2] is the
        # COM position in world frame; downstream code (coverage tracker)
        # relies on this slice being first.
        return jp.concatenate([
            jp.asarray(data.qpos, dtype=jp.float32),
            jp.asarray(data.qvel, dtype=jp.float32),
            self._goal_xy,
        ])

    @property
    def observation_size(self) -> int:
        return int(self._mj_model.nq + self._mj_model.nv + 2)
