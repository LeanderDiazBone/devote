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


# Sized so the ant's ~1.2m leg span has plenty of room in corridors and the
# walls visibly tower over it at the trackcom camera angle. 5.0 puts the
# start->goal straight line at 8*5 = 40m, which an optimal Brax-ant policy
# (~3 m/s) covers in ~270 steps, leaving headroom for the avoid_obstacle
# detour to fit inside a 300-step optimal trajectory.
_MAZE_CELL_SIZE = 5.0
_MAZE_WALL_HALF_HEIGHT = 2.25

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

# avoid_obstacle family: programmatically generated NxN room with a
# centered vertical wall ``n_blocks`` cells tall between S (left) and G
# (right). Selected via ``avoid_obstacle`` (= 1 block, the default for
# testing) or ``avoid_obstacle_<n>`` for n in {1, 3, 5, 7}. Always uses
# the goal-directed speed reward (greedy goal-following stalls against
# the wall; reaching G requires detouring above or below).
# NOTE: with rows=9 the inner column is only 7 cells tall, so n_blocks=7
# fills it completely and leaves no detour (unsolvable).
#
# A ``_simple`` variant — ``avoid_obstacle_simple`` /
# ``avoid_obstacle_simple_<n>`` — keeps the same geometry and speed/goal
# rewards but zeros out ``healthy_reward`` and ``ctrl_cost``. Use it to
# remove the "stand still and stay healthy" local attractor when probing
# whether the agent can learn locomotion at all under Brax-style shaping.
#
# A ``_small`` variant — ``avoid_obstacle_small`` /
# ``avoid_obstacle_small_<n>`` — uses a 7x7 room (inner column 5 cells)
# instead of 9x9 for a shorter optimal trajectory. Valid n for the small
# variant is {1, 3, 5}; n=5 fills the inner column (unsolvable), kept
# for parity with the 9x9 case where n=7 is also unsolvable.
_AVOID_OBSTACLE_PREFIX = "avoid_obstacle"
_AVOID_OBSTACLE_SIMPLE_PREFIX = "avoid_obstacle_simple"
_AVOID_OBSTACLE_SMALL_PREFIX = "avoid_obstacle_small"
_AVOID_OBSTACLE_VALID_N = (1, 3, 5, 7)
_AVOID_OBSTACLE_SMALL_VALID_N = (1, 3, 5)

# u_trap family: a U-shaped wall centered in an 11x11 room with its mouth
# opening toward S on the left. A straight-line walk from S to G passes
# through the mouth and hits the closed right side of the U — the naive
# "go straight at the goal" policy gets caught inside the cup. Escape
# requires backing out through the mouth and detouring above or below
# the U. ``side_length`` is the number of cells along each arm of the U
# (top bar, closed right side, bottom bar). Selected via ``u_trap``
# (= side 3, the default) or ``u_trap_<n>`` for n in {3, 5}.
_U_TRAP_PREFIX = "u_trap"
_U_TRAP_VALID_N = (3, 5)
_U_TRAP_SIZE = 11


def _build_u_trap_layout(side_length: int) -> tuple[str, ...]:
    if side_length not in _U_TRAP_VALID_N:
        raise ValueError(
            f"u_trap side_length must be one of {_U_TRAP_VALID_N}; "
            f"got {side_length}.")
    rows = cols = _U_TRAP_SIZE
    u_top = (rows - side_length) // 2
    u_bot = u_top + side_length - 1
    u_left = (cols - side_length) // 2
    u_right = u_left + side_length - 1
    inside_row = (u_top + u_bot) // 2
    layout = []
    for r in range(rows):
        if r in (0, rows - 1):
            layout.append("#" * cols)
            continue
        chars = ["#"] + ["."] * (cols - 2) + ["#"]
        if r == u_top or r == u_bot:
            for c in range(u_left, u_right + 1):
                chars[c] = "#"
        elif u_top < r < u_bot:
            # Closed (right) side of the U; left column stays open as the mouth.
            chars[u_right] = "#"
        if r == inside_row:
            chars[1] = "S"
            chars[cols - 2] = "G"
        layout.append("".join(chars))
    return tuple(layout)


def _build_avoid_obstacle_layout(n_blocks: int, size: int = 9) -> tuple[str, ...]:
    valid_n = (
        _AVOID_OBSTACLE_SMALL_VALID_N if size == 7 else _AVOID_OBSTACLE_VALID_N)
    if n_blocks not in valid_n:
        raise ValueError(
            f"avoid_obstacle n_blocks for size {size} must be one of "
            f"{valid_n}; got {n_blocks}.")
    rows, cols = size, size
    mid_row, mid_col = rows // 2, cols // 2
    half = n_blocks // 2
    wall_rows = set(range(mid_row - half, mid_row + half + 1))
    layout = []
    for r in range(rows):
        if r in (0, rows - 1):
            layout.append("#" * cols)
            continue
        chars = ["#"] + ["."] * (cols - 2) + ["#"]
        if r == mid_row:
            chars[1] = "S"
            chars[cols - 2] = "G"
        if r in wall_rows:
            chars[mid_col] = "#"
        layout.append("".join(chars))
    return tuple(layout)


def _resolve_maze_layout(variant: str) -> tuple[tuple[str, ...], bool, bool]:
    """Return (layout, goal_directed_speed, simple_reward) for the variant.

    goal_directed_speed: projects torso velocity onto the unit vector to the
    goal (signed) instead of using direction-agnostic torso speed. Used by
    hard-exploration variants where a wall blocks the direct line.

    simple_reward: when True, the env zeros out ``healthy_reward`` and
    ``ctrl_cost`` so only the speed reward, sparse goal bonus, and small
    contact cost contribute. Removes the stand-still attractor for
    debugging locomotion learning.
    """
    if variant in _MAZE_LAYOUTS:
        return _MAZE_LAYOUTS[variant], False, False
    # Match the more-specific _simple / _small prefixes BEFORE the broader
    # ``avoid_obstacle`` prefix so that ``avoid_obstacle_simple_3`` /
    # ``avoid_obstacle_small_3`` don't get routed to the regular branch
    # (where ``simple`` / ``small`` would fail the int-parse).
    if variant == _AVOID_OBSTACLE_SIMPLE_PREFIX:
        return _build_avoid_obstacle_layout(1), True, True
    if variant.startswith(_AVOID_OBSTACLE_SIMPLE_PREFIX + "_"):
        suffix = variant[len(_AVOID_OBSTACLE_SIMPLE_PREFIX) + 1:]
        try:
            n = int(suffix)
        except ValueError:
            n = None
        if n is not None:
            return _build_avoid_obstacle_layout(n), True, True
    if variant == _AVOID_OBSTACLE_SMALL_PREFIX:
        return _build_avoid_obstacle_layout(1, size=7), True, False
    if variant.startswith(_AVOID_OBSTACLE_SMALL_PREFIX + "_"):
        suffix = variant[len(_AVOID_OBSTACLE_SMALL_PREFIX) + 1:]
        try:
            n = int(suffix)
        except ValueError:
            n = None
        if n is not None:
            return _build_avoid_obstacle_layout(n, size=7), True, False
    if variant == _AVOID_OBSTACLE_PREFIX:
        return _build_avoid_obstacle_layout(1), True, False
    if variant.startswith(_AVOID_OBSTACLE_PREFIX + "_"):
        suffix = variant[len(_AVOID_OBSTACLE_PREFIX) + 1:]
        try:
            n = int(suffix)
        except ValueError:
            n = None
        if n is not None:
            return _build_avoid_obstacle_layout(n), True, False
    if variant == _U_TRAP_PREFIX:
        return _build_u_trap_layout(3), True, False
    if variant.startswith(_U_TRAP_PREFIX + "_"):
        suffix = variant[len(_U_TRAP_PREFIX) + 1:]
        try:
            n = int(suffix)
        except ValueError:
            n = None
        if n is not None:
            return _build_u_trap_layout(n), True, False
    raise ValueError(
        f"Unknown subtask: {variant!r}. Choose from {list(_MAZE_LAYOUTS)}, "
        f"'avoid_obstacle[_<n>]' (n in {_AVOID_OBSTACLE_VALID_N}), "
        f"'avoid_obstacle_simple[_<n>]' (n in {_AVOID_OBSTACLE_VALID_N}), "
        f"'avoid_obstacle_small[_<n>]' (n in {_AVOID_OBSTACLE_SMALL_VALID_N}), "
        f"or 'u_trap[_<n>]' (n in {_U_TRAP_VALID_N}).")


def _cell_center(
    row: int,
    col: int,
    rows: int,
    cols: int,
    cell_size: float,
    positive_coords: bool = False,
) -> tuple[float, float]:
    if positive_coords:
        x = col * cell_size
        y = (rows - 1 - row) * cell_size
    else:
        x = (col - (cols - 1) / 2.0) * cell_size
        y = ((rows - 1) / 2.0 - row) * cell_size
    return x, y


def _make_maze_geometry(
    layout: tuple[str, ...],
    cell_size: float = _MAZE_CELL_SIZE,
    wall_half_height: float = _MAZE_WALL_HALF_HEIGHT,
    positive_coords: bool = False,
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
            x, y = _cell_center(r, c, rows, cols, cell_size, positive_coords)
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
        sim_dt=0.005,
        episode_length=500,
        action_repeat=1,
        goal=jp.array(maze_goal),        # (x,y) in world coords
        success_thresh=1.0,
        # Brax ant-v4 reward coefficients verbatim; only structural change
        # is that the forward direction is the unit vector toward the goal
        # rather than world-frame +x (handled in step()). v_eff is the
        # SIGNED projection — moving away from the goal is penalized just
        # like Brax penalizes negative v_x — so SAC gets a one-sided
        # gradient that points the ant at the goal during exploration.
        #   speed_reward    = 1.0 * v_eff                     (Brax: 1.0 * v_x)
        #   healthy_reward  = -0.25 while NOT is_healthy      (Brax: +1.0 while is_healthy)
        #   ctrl_cost       = 0.5 * sum(action^2)             (Brax: 0.5)
        #   contact_cost    = 5e-4 * sum(clip(cfrc,-1,1)^2)   (Brax: 5e-4)
        speed_lin_scale=1.0,
        speed_quad_scale=0.0,
        healthy_reward=-0.5,
        ctrl_cost=0.1,
        contact_cost=5e-4,
        terminate_when_unhealthy=False,
        healthy_z_range=(0.15, 1.2),
        # Sparse goal. Two modes:
        #   terminate_on_success=False — +``success_reward`` per step while
        #     within ``success_thresh`` of the goal; the optimal policy is to
        #     reach the goal and camp there (matches the original ant setup).
        #   terminate_on_success=True — episode ends on the first success
        #     step and the reward at that step is overridden to
        #     ``5 * episode_length`` (one-shot terminal bonus, five times
        #     the horizon). This is the panda-mirror configuration.
        success_reward=5.0,
        terminate_on_success=True,
        dense_goal_reward=0.0,           # scales -dist_to_goal
        positive_coords=True,
    )


# ----------------------------
# Assets / XML
# ----------------------------
_ROOT = epath.Path(__file__).parent

def get_assets():
    return {}


def _build_ant_maze_xml(
    maze_geoms_xml: str,
    floor_center: tuple[float, float] = (0.0, 0.0),
    floor_half: tuple[float, float] = (40.0, 40.0),
) -> str:
    ant_fragment = (_ROOT / "assets" / "xmls" / "ant.xml").read_text()
    ant_fragment = ant_fragment.split("</mujoco>", 1)[0].rstrip()
    # The ant.xml ships the floor centered at (0, 0) with a fixed 40 m
    # rendered half-extent; shift and resize it so any maze sits inside the
    # plane's bounds. (Plane collision is infinite either way; this only
    # affects the visible rectangle.)
    ant_fragment = ant_fragment.replace(
        'name="floor" pos="0 0 0"',
        f'name="floor" pos="{floor_center[0]:.3f} {floor_center[1]:.3f} 0"',
        1,
    )
    ant_fragment = ant_fragment.replace(
        'size="40 40 40" type="plane"',
        f'size="{floor_half[0]:.3f} {floor_half[1]:.3f} 40" type="plane"',
        1,
    )
    ant_fragment = ant_fragment.replace(
        "</worldbody>",
        f"""
    {maze_geoms_xml}
  </worldbody>""",
        1,
    )
    return f"""
<mujoco model="ant_maze">
  <compiler angle="degree" coordinate="local"/>
  <option timestep="0.005" integrator="implicitfast"/>
  <size njmax="2000" nconmax="200"/>
  <default>
    <default class="ant">
      <joint armature="1" damping="1" limited="true"/>
      <geom conaffinity="0" condim="3" density="5.0" friction="1 0.5 0.5" margin="0.01"/>
    </default>
  </default>
  <asset>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="1 1" rgba="0.93 0.87 0.76 1"/>
  </asset>
  {ant_fragment}
</mujoco>
""".strip()


class AntMaze(mjx_env.MjxEnv):
    def __init__(
        self,
        subtask = None,
        config=None,
        config_overrides: Optional[Dict[str, Union[str, int, float, list[Any]]]] = None,
    ):
        maze_variant = subtask if subtask else "room"
        # Optional ``_term`` suffix flips ``terminate_when_unhealthy`` on for
        # any maze variant (e.g. ``avoid_obstacle_3_term`` = same layout as
        # ``avoid_obstacle_3`` but the episode ends as soon as the ant flips).
        terminate_unhealthy = False
        if maze_variant.endswith("_term"):
            terminate_unhealthy = True
            maze_variant = maze_variant[: -len("_term")]
        layout, goal_directed_speed, simple_reward = _resolve_maze_layout(
            maze_variant)
        # ``_simple`` variants disable the survival bonus and ctrl cost. We
        # fold them into config_overrides BEFORE the parent constructor so
        # the rest of the env (and the metrics) see the resolved zeros.
        # `setdefault` lets an explicit caller-provided override win.
        if simple_reward or terminate_unhealthy:
            config_overrides = dict(config_overrides) if config_overrides else {}
            if simple_reward:
                config_overrides.setdefault('healthy_reward', 0.0)
                config_overrides.setdefault('ctrl_cost', 0.0)
            if terminate_unhealthy:
                config_overrides.setdefault('terminate_when_unhealthy', True)
        # OGBench imports this module for maze construction inside each
        # environment worker, so keep JAX initialization instance-local.
        if config is None:
            config = default_config()
        mjx_env.MjxEnv.__init__(self, config, config_overrides)

        self._maze_layout = layout
        self._goal_directed_speed = goal_directed_speed
        self._positive_coords = bool(self._config.positive_coords)
        maze_geoms_xml, start_xy, maze_goal = _make_maze_geometry(
            self._maze_layout, positive_coords=self._positive_coords)
        self._start_xy = jp.array(start_xy, dtype=jp.float32)
        self._goal_xy = jp.array(maze_goal, dtype=jp.float32)

        rows = len(self._maze_layout)
        cols = len(self._maze_layout[0])
        if self._positive_coords:
            floor_center = (
                (cols - 1) * _MAZE_CELL_SIZE / 2.0,
                (rows - 1) * _MAZE_CELL_SIZE / 2.0,
            )
        else:
            floor_center = (0.0, 0.0)
        # +15 m beyond the outermost cell centers covers the wall overhang
        # (2.75 m) plus a visual apron; reproduces the stock 40 m half-extent
        # for 11x11 layouts and scales up for larger ones.
        floor_half = (
            (cols - 1) * _MAZE_CELL_SIZE / 2.0 + 15.0,
            (rows - 1) * _MAZE_CELL_SIZE / 2.0 + 15.0,
        )
        xml = _build_ant_maze_xml(
            maze_geoms_xml, floor_center=floor_center, floor_half=floor_half)
        mj_model = mujoco.MjModel.from_xml_string(xml, assets=get_assets())
        mj_model.opt.timestep = self._config.sim_dt

        self._mj_model = mj_model
        self._mjx_model = mjx.put_model(mj_model)

        # Use the MJCF default pose: torso at z=0.75 with identity orientation
        # quaternion (qpos[3:7] = (1, 0, 0, 0)). Passing all-zero qpos here
        # would leave the free-joint quaternion at (0, 0, 0, 0); after the
        # 0.01 reset noise the ant would spawn in a random (often upside-down)
        # orientation.
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
        return "<in-memory-ant-maze-xml>"

    def get_coverage_geometry(self, bins_per_cell: int = 1) -> dict:
        """Uniform-binning geometry over the maze's (x, y) cells.

        ``bins_per_cell`` is the total number of sub-bins per open cell and
        must be a perfect square (1, 4, 9, ...); the per-axis subdivision is
        sqrt(bins_per_cell).
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
        if self._positive_coords:
            cx, cy = (cols - 1) * cs / 2.0, (rows - 1) * cs / 2.0
        else:
            cx, cy = 0.0, 0.0
        open_layout = np.array(
            [[ch != '#' for ch in row] for row in layout], dtype=bool)
        # valid_mask[x_bin, y_bin]; y_bin increases upward (row 0 = top => max y_bin).
        valid_mask = np.flipud(open_layout).T.copy()
        if sub > 1:
            valid_mask = np.kron(valid_mask, np.ones((sub, sub), dtype=bool))
        return dict(
            bounds=np.array(
                [[cx - x_extent, cx + x_extent],
                 [cy - y_extent, cy + y_extent]], dtype=np.float64),
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
        qpos = self._init_qpos + 0.1 * jax.random.normal(k1, self._init_qpos.shape)
        qvel = self._init_qvel + 0.1 * jax.random.normal(k2, self._init_qvel.shape)
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
        init_dist = jp.linalg.norm(self._start_xy - self._goal_xy).astype(jp.float32)
        metrics = {
            "dist": init_dist,
            "success": zero,
            # Per-step reward components; summed by log_keys_sum at episode end.
            "log_rew_speed": zero,
            "log_rew_healthy": zero,
            "log_rew_ctrl": zero,
            "log_rew_contact": zero,
            "log_rew_success": zero,
            "log_rew_dense_goal": zero,
            # Distance to goal, aggregated by min/avg at episode end as a
            # diagnostic for "how close did the agent get". Independent of the
            # training reward — logged even when ``dense_goal_reward`` is 0.
            "log_dist": init_dist,
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

        # Healthy z check (Brax convention). Used only for termination.
        min_z, max_z = self._config.healthy_z_range
        z = data.qpos[2]
        is_healthy = jp.where(z < min_z, 0.0, 1.0)
        is_healthy = jp.where(z > max_z, 0.0, is_healthy)

        ctrl_cost = self._config.ctrl_cost * jp.sum(jp.square(action))
        contact_cost = self._config.contact_cost * jp.sum(
            jp.square(jp.clip(data.cfrc_ext, -1.0, 1.0))
        )

        xy = data.qpos[:2]
        v_xy = data.qvel[:2]
        dist = jp.linalg.norm(xy - self._goal_xy)
        success = (dist < self._config.success_thresh).astype(jp.float32)

        # Speed reward. Direction-agnostic by default (rewards any running).
        # For ``avoid_obstacle`` (and any other goal-directed variant) the
        # forward direction is the unit vector from ant to goal, recomputed
        # each step. The projection is SIGNED — moving away from the goal
        # is penalized, mirroring how Brax's `v_x` penalizes backward
        # motion — so SAC has a one-sided gradient toward the goal during
        # exploration. (Going around a wall briefly costs reward; the
        # post-detour straight run more than recovers it.)
        if self._goal_directed_speed:
            goal_dir = (self._goal_xy - xy) / jp.maximum(dist, 1e-6)
            effective_speed = jp.dot(v_xy, goal_dir)
        else:
            effective_speed = jp.linalg.norm(v_xy)
        speed_reward = (
            self._config.speed_lin_scale * effective_speed
            + self._config.speed_quad_scale * jp.square(effective_speed)
        )

        healthy_reward = self._config.healthy_reward * (1.0 - is_healthy)

        # Success bonus: one-shot terminal payoff of ``5 * episode_length``
        # when terminating on success; otherwise per-step camping at
        # ``success_reward`` (legacy behavior).
        if self._config.terminate_on_success:
            rew_success = jp.float32(5 * self._config.episode_length) * success
        else:
            rew_success = self._config.success_reward * success

        reward = (
            speed_reward
            + healthy_reward
            - ctrl_cost
            - contact_cost
            + rew_success
            - self._config.dense_goal_reward * dist
        ).astype(jp.float32)

        steps = state.info["_steps"] + 1
        if self._config.terminate_when_unhealthy:
            done = (is_healthy < 0.5).astype(jp.float32)
        else:
            done = jp.float32(0.0)
        if self._config.terminate_on_success:
            done = jp.maximum(done, success)

        # MJX can diverge under extreme contacts (non-finite qpos/qvel) and
        # never recovers: terminate so the driver resets the env, and zero
        # the non-finite outputs so the policy-input finiteness check and
        # the logged episode stats stay clean.
        bad = ~(
            jp.isfinite(data.qpos).all()
            & jp.isfinite(data.qvel).all()
            & jp.isfinite(reward)
        )
        done = jp.maximum(done, bad.astype(jp.float32))
        reward = jp.where(bad, 0.0, reward)

        state.info["_steps"] = steps
        state.metrics.update(
            dist=dist.astype(jp.float32),
            success=success,
            log_rew_speed=speed_reward.astype(jp.float32),
            log_rew_healthy=healthy_reward.astype(jp.float32),
            log_rew_ctrl=(-ctrl_cost).astype(jp.float32),
            log_rew_contact=(-contact_cost).astype(jp.float32),
            log_rew_success=rew_success.astype(jp.float32),
            log_rew_dense_goal=(-self._config.dense_goal_reward * dist).astype(jp.float32),
            log_dist=dist.astype(jp.float32),
        )
        state.metrics.update(
            {k: jp.where(jp.isfinite(v), v, 0.0) for k, v in state.metrics.items()})

        obs = self._get_obs(data)
        return state.replace(
            data=data,
            obs=jp.where(bad, jp.zeros_like(obs), obs),
            reward=reward,
            done=done,
        )

    def _get_obs(self, data: mjx.Data) -> jax.Array:
        return jp.concatenate([
            jp.asarray(data.qpos, dtype=jp.float32),
            jp.asarray(data.qvel, dtype=jp.float32),
            self._goal_xy,
        ])

    @property
    def observation_size(self) -> int:
        return int(self._mj_model.nq + self._mj_model.nv + 2)

    def _state_component_names(self) -> list[str]:
        nq = int(self._mj_model.nq)
        nv = int(self._mj_model.nv)
        return (['x_1', 'x_2']
                + [f'qpos_{i}' for i in range(2, nq)]
                + [f'qvel_{i}' for i in range(nv)]
                + ['goal_1', 'goal_2'])

    def get_state_table(
        self,
        points_per_cell: int = 3,
    ) -> Dict[str, Any]:
        """Cell-aligned grid of raw state observations over the maze's open xy.

        Mirrors ``PointMaze.get_state_table``: sweeps the torso (x, y) over the
        maze's open cells while holding the rest of qpos at the MJCF default
        pose (z=0.75, identity quaternion, joints at home) and qvel at zero.
        """
        rows, cols = len(self._maze_layout), len(self._maze_layout[0])
        open_rs, open_cs = [], []
        for r, layout_row in enumerate(self._maze_layout):
            for c, v in enumerate(layout_row):
                if v != '#':
                    open_rs.append(r); open_cs.append(c)
        if not open_rs:
            raise ValueError('AntMaze layout contains no open cells.')

        r_min, r_max = min(open_rs), max(open_rs)
        c_min, c_max = min(open_cs), max(open_cs)

        if points_per_cell <= 1:
            sub = np.array([0.0], dtype=np.float32)
        else:
            sub = (np.arange(points_per_cell, dtype=np.float32) + 0.5) / points_per_cell - 0.5

        cs = float(_MAZE_CELL_SIZE)
        if self._positive_coords:
            x_centers = np.arange(c_min, c_max + 1, dtype=np.float32) * cs
            y_centers = ((rows - 1) - np.arange(r_min, r_max + 1, dtype=np.float32)) * cs
        else:
            x_centers = (np.arange(c_min, c_max + 1, dtype=np.float32) - (cols - 1) / 2.0) * cs
            y_centers = ((rows - 1) / 2.0 - np.arange(r_min, r_max + 1, dtype=np.float32)) * cs

        xs = (x_centers[:, None] + sub[None, :] * cs).reshape(-1)
        ys = (y_centers[:, None] + sub[None, :] * cs).reshape(-1)
        xs.sort()
        ys.sort()
        grid_x, grid_y = np.meshgrid(xs, ys, indexing='xy')
        xy = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)

        nq = int(self._mj_model.nq)
        nv = int(self._mj_model.nv)
        rest_qpos = np.broadcast_to(
            np.asarray(self._init_qpos[2:], dtype=np.float32), (xy.shape[0], nq - 2))
        qvel = np.zeros((xy.shape[0], nv), dtype=np.float32)
        goal = np.broadcast_to(
            np.asarray(self._goal_xy, dtype=np.float32), (xy.shape[0], 2))
        state = np.concatenate(
            [xy.astype(np.float32), rest_qpos, qvel, goal], axis=-1)
        return dict(
            obs={'state': state},
            obs_col_names={'state': self._state_component_names()},
            grid_shape=np.array(grid_x.shape, dtype=np.int32),
        )
