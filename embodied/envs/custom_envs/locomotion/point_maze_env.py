from __future__ import annotations
import os
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx
from ml_collections import config_dict
from mujoco_playground._src import mjx_env


MINI_ROOM_MAZE_LAYOUT = (
    "#####",
    "#..G#",
    "#.S.#",
    "#...#",
    "#####",
)

ROOM_MAZE_LAYOUT = (
    "#######",
    "#....G#",
    "#.....#",
    "#..S..#",
    "#.....#",
    "#.....#",
    "#######",
)

BIG_ROOM_MAZE_LAYOUT = (
    "#########",
    "#.......#",
    "#.#...#.#",
    "#.......#",
    "#...S...#",
    "#.......#",
    "#.#...#.#",
    "#G......#",
    "#########",
)

BIGGEST_ROOM_MAZE_LAYOUT = (
    "###########",
    "#.........#",
    "#.#.....#.#",
    "#.........#",
    "#.........#",
    "#....S....#",
    "#.........#",
    "#.........#",
    "#.#.....#.#",
    "#G........#",
    "###########",
)

WALL_ROOM_MAZE_LAYOUT = (
    "###########",
    "#....G....#",
    "#.#######.#",
    "#.........#",
    "#.........#",
    "#....S....#",
    "#.........#",
    "#.........#",
    "#.#######.#",
    "#.........#",
    "###########",
)
SQUARE_MAZE_LAYOUT = (
    "#######",
    "#....G#",
    "#.#.#.#",
    "#..S..#",
    "#.#.#.#",
    "#.....#",
    "#######",
)

BIG_SQUARE_MAZE_LAYOUT = (
    "############",
    "#..#.GG..#.#",
    "#....##....#",
    "#..##..##..#",
    "#..#....#..#",
    "#....SS....#",
    "#....SS....#",
    "#..#....#..#",
    "#..##..##..#",
    "#..........#",
    "#..#.....#.#",
    "############",
)

# SPIRAL_MAZE_LAYOUT = (
#     "##############",
#     "#............#",
#     "#.##########.#",
#     "#.#........#.#",
#     "#.#.######.#.#",
#     "#.#.#S...#.#.#",
#     "#.#.####.#.#.#",
#     "#.#......#.#.#",
#     "#.########.#.#",
#     "#..........#.#",
#     "#.##########.#",
#     "#G...........#",
#     "##############",
# )

SPIRAL_MAZE_LAYOUT = (
    "###########",
    "#........G#",
    "#.#########",
    "#.#.......#",
    "#.#.#####.#",
    "#.#.#S..#.#",
    "#.#.###.#.#",
    "#.#.....#.#",
    "#.#######.#",
    "#.........#",
    "###########",
)

LARGE_SPIRAL_MAZE_LAYOUT = (
    "################",
    "#.............G#",
    "#..............#",
    "#..#############",
    "#..#...........#",
    "#..#...........#",
    "#..#..#######..#",
    "#..#..#S....#..#",
    "#..#..#.....#..#",
    "#..#..####..#..#",
    "#..#........#..#",
    "#..#........#..#",
    "#..##########..#",
    "#..............#",
    "#..............#",
    "################",
)

CORRIDORS_MAZE_LAYOUT = (
    "###############",
    "#......S......#",
    "#.............#",
    "#..#########..#",
    "#..#########..#",
    "#..#########..#",
    "#..#########..#",
    "#..#########..#",
    "#..#########..#",
    "#..#########..#",
    "#..#########..#",
    "#..#########..#",
    "#..#########..#",
    "#..####G####..#",
    "###############",
)

ESCAPE_MAZE_LAYOUT = (
    "###############",
    "#.............#",
    "#.#.#########.#",
    "#.#.........#.#",
    "#.#.#######.#.#",
    "#.#.#.......#.#",
    "#.#.#.....#.#.#",
    "#.#.#..S..#.#.#",
    "#.#.#.....#.#.#",
    "#.#.......#.#.#",
    "#.#.#######.#.#",
    "#.#.........#.#",
    "#.#########.#.#",
    "#G............#",
    "###############",
)
LARGE_ESCAPE_MAZE_LAYOUT = (
    "###################",
    "#.................#",
    "#.................#",
    "#..#..##########..#",
    "#..#...........#..#",
    "#..#...........#..#",
    "#..#..#######..#..#",
    "#..#..#........#..#",
    "#..#..#........#..#",
    "#..#..#..S..#..#..#",
    "#..#........#..#..#",
    "#..#........#..#..#",
    "#..#..#######..#..#",
    "#..#...........#..#",
    "#..#...........#..#",
    "#..##########..#..#",
    "#.................#",
    "#G................#",
    "###################",
)


_MAZE_CELL_SIZE = 2.5
_MAZE_WALL_HALF_HEIGHT = 1.0
_LOG_STATE_POINTS_ENV = "MJP_POINT_MAZE_LOG_STATES"
_DEFAULT_LOG_STATE_POINTS = "(0,0);(0,5);(5,0);(-5,0);(0,-5)"
_TERMINATE_ON_WALL_SUFFIXES = (
    "terminate_on_wall_collision",
    "terminate_on_collision",
    "termwall",
    "terminate",
)


def _parse_xy_points(spec: str) -> np.ndarray:
    rows = []
    spec = str(spec).strip()
    if ';' not in spec and '),' in spec:
        spec = spec.replace('),', ');')
    for chunk in str(spec).split(';'):
        chunk = chunk.strip().strip('()[]')
        if not chunk:
            continue
        values = [float(x.strip()) for x in chunk.split(',') if x.strip()]
        if len(values) != 2:
            raise ValueError(
                f'{_LOG_STATE_POINTS_ENV} entries must be x,y pairs; got {chunk!r}')
        rows.append(values)
    if not rows:
        raise ValueError(f'{_LOG_STATE_POINTS_ENV} must contain at least one x,y pair.')
    return np.asarray(rows, dtype=np.float32)


def _log_state_points() -> np.ndarray:
    return _parse_xy_points(
        os.environ.get(_LOG_STATE_POINTS_ENV, _DEFAULT_LOG_STATE_POINTS))

_COLLISION_COST_SUFFIXES = (
    "collision_cost",
    "wallcost",
)


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

    # Slight overlap between neighboring walls to avoid seams.
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
                    f'rgba="0.55 0.4 0.3 1" contype="1" conaffinity="1"/>'
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

    maze_xml = "\n      ".join([*wall_geoms, goal_marker])
    return maze_xml, start_xy, goal_xy


def _open_cell_bounds(
    layout: tuple[str, ...],
    cell_size: float = _MAZE_CELL_SIZE,
    positive_coords: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    open_cells = []
    rows, cols = len(layout), len(layout[0])
    for row, layout_row in enumerate(layout):
        for col, value in enumerate(layout_row):
            if value != '#':
                open_cells.append(
                    _cell_center(row, col, rows, cols, cell_size, positive_coords))
    if not open_cells:
        raise ValueError('PointMaze layout contains no open cells.')
    open_cells = np.asarray(open_cells, dtype=np.float32)
    return open_cells.min(axis=0), open_cells.max(axis=0)


def default_config():
    _, _, maze_goal = _make_maze_geometry(ROOM_MAZE_LAYOUT)
    return config_dict.create(
        ctrl_dt=0.05,
        sim_dt=0.005,
        episode_length=1000,
        action_repeat=1,
        goal=jp.array(maze_goal, dtype=jp.float32),
        success_thresh=1.0,
        ctrl_cost=0.0,
        success_reward=10.0,
        action_scale=0.25, # 0.05, 0.15
        collision_cost=0.1,
        positive_coords=True,
    )


def _parse_wall_suffix(sub_task: Optional[str], default_variant: str) -> tuple[str, bool, bool]:
    """Parse sub_task for terminate-on-wall and collision-cost suffixes.

    Returns (maze_variant, terminate_on_wall_collision, apply_collision_cost).
    The two flags are mutually exclusive; collision-cost suffixes are checked
    first so they take priority when both could match.
    """
    if not sub_task:
        return default_variant, False, False

    # Check collision-cost suffixes first.
    for suffix in _COLLISION_COST_SUFFIXES:
        if sub_task == suffix:
            return default_variant, False, True
        suffix_token = f"_{suffix}"
        if sub_task.endswith(suffix_token):
            variant = sub_task[:-len(suffix_token)] or default_variant
            return variant, False, True

    # Then check terminate-on-wall suffixes (original behaviour).
    for suffix in _TERMINATE_ON_WALL_SUFFIXES:
        if sub_task == suffix:
            return default_variant, True, False
        suffix_token = f"_{suffix}"
        if sub_task.endswith(suffix_token):
            variant = sub_task[:-len(suffix_token)] or default_variant
            return variant, True, False

    return sub_task, False, False


def _build_point_maze_xml(maze_geoms_xml: str) -> str:
    return f"""
    <mujoco model="point_maze">
      <compiler angle="degree" coordinate="local"/>
      <option timestep="0.005" integrator="Euler"/>

      <worldbody>
        <light directional="true" diffuse=".8 .8 .8" specular=".2 .2 .2"
               pos="0 0 5" dir="0 0 -1"/>

        <geom name="floor"
              type="plane"
              size="40 40 .05"
              rgba="0.93 0.87 0.76 1"
              contype="1"
              conaffinity="1"/>

        <body name="torso" pos="0 0 0.35">
            <joint name="x" type="slide" axis="1 0 0" damping="15"/>
            <joint name="y" type="slide" axis="0 1 0" damping="15"/>
            <geom name="ball"
                    type="sphere"
                    size="0.35"
                    mass="0.2"
                    rgba="0 0.7 0.7 1"
                    contype="1"
                    conaffinity="1"/>
            <camera name="track"
                    mode="trackcom"
                    pos="0 -10 15"
                    xyaxes="1 0 0 0 0.8 1"/>
            </body>



        {maze_geoms_xml}
      </worldbody>

      <actuator>
        <velocity name="x_motor"
                    joint="x"
                    kv="20"
                    ctrlrange="-2 2"
                    ctrllimited="true"/>
        <velocity name="y_motor"
                    joint="y"
                    kv="20"
                    ctrlrange="-2 2"
                    ctrllimited="true"/>
        </actuator>
            </mujoco>
    """.strip()


class PointMaze(mjx_env.MjxEnv):
    def __init__(
        self,
        sub_task=None,
        config=None,
        config_overrides: Optional[Dict[str, Union[str, int, float, list[Any]]]] = None,
    ):
        # OGBench imports this module for its layout constants inside each
        # environment worker, so do not initialize JAX merely by importing it.
        if config is None:
            config = default_config()
        mjx_env.MjxEnv.__init__(self, config, config_overrides)

        maze_variant, terminate_on_wall_collision, apply_collision_cost = _parse_wall_suffix(
            sub_task, default_variant="room"
        )
        maze_layouts = {
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
        if maze_variant not in maze_layouts:
            raise ValueError(f"Unknown sub_task: {sub_task}")
        maze_layout = maze_layouts[maze_variant]
        self._maze_layout = maze_layout
        self._positive_coords = bool(self._config.positive_coords)
        self._xy_min, self._xy_max = _open_cell_bounds(
            maze_layout, positive_coords=self._positive_coords)

        maze_geoms_xml, start_xy, maze_goal = _make_maze_geometry(
            maze_layout, positive_coords=self._positive_coords)
        self._start_xy = jp.array(start_xy, dtype=jp.float32)
        self._goal_xy = jp.array(maze_goal, dtype=jp.float32)
        self._terminate_on_wall_collision = jp.array(
            terminate_on_wall_collision, dtype=jp.bool_
        )
        self._apply_collision_cost = jp.array(
            apply_collision_cost, dtype=jp.bool_
        )

        xml = _build_point_maze_xml(maze_geoms_xml)
        mj_model = mujoco.MjModel.from_xml_string(xml)
        mj_model.opt.timestep = self._config.sim_dt

        self._mj_model = mj_model
        self._mjx_model = mjx.put_model(mj_model)

        self._init_qpos = jp.array(mj_model.qpos0, dtype=jp.float32)  # [x, y]
        self._init_qvel = jp.zeros(self._mj_model.nv, dtype=jp.float32)  # [vx, vy]
        self._action_size = int(self._mj_model.nu)
        self.action_scale = self._config.action_scale

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property
    def xml_path(self) -> str:
        return "<in-memory-point-maze-xml>"

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def observation_size(self) -> int:
        # qpos [x, y] + qvel [vx, vy] + goal [gx, gy]
        return self._mj_model.nq + self._mj_model.nv + 2

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

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, rng_obs = jax.random.split(rng)

        qpos = self._init_qpos.at[:2].set(self._start_xy)
        qvel = self._init_qvel

        data = mjx_env.init(
            self._mjx_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=jp.zeros(self._action_size, dtype=jp.float32),
        )

        info = {
            "rng": rng,
            "_steps": jp.array(0, dtype=jp.int32),
        }
        metrics = {
            "dist": jp.linalg.norm(self._start_xy - self._goal_xy),
            "success": jp.array(0.0, dtype=jp.float32),
            "wall_collisions": jp.array(0.0, dtype=jp.float32),
        }

        obs = self._get_obs(data, rng_obs)
        return mjx_env.State(
            data,
            obs,
            jp.array(0.0, dtype=jp.float32),
            jp.array(0.0, dtype=jp.float32),
            metrics,
            info,
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        action = jp.asarray(action, dtype=jp.float32)
        action = jp.clip(action, -1.0, 1.0)
        delta = self.action_scale * action

        cur_xy = jp.asarray(state.data.qpos[:2], dtype=jp.float32)
        cand_xy = cur_xy + delta

        # Ball radius from XML: sphere size="0.35"
        ball_radius = 0.35

        # Check whether candidate position would intersect any wall box.
        # We read wall geoms from the static MuJoCo model and do simple AABB collision.
        collision = False
        for i in range(self._mj_model.ngeom):
            name = mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name is None or not name.startswith("maze_wall_"):
                continue

            wall_pos = jp.array(self._mj_model.geom_pos[i, :2], dtype=jp.float32)
            wall_half = jp.array(self._mj_model.geom_size[i, :2], dtype=jp.float32)

            # Expand wall by ball radius and test point-in-box.
            lower = wall_pos - wall_half - ball_radius
            upper = wall_pos + wall_half + ball_radius

            in_wall = jp.logical_and(cand_xy >= lower, cand_xy <= upper).all()
            collision = jp.logical_or(collision, in_wall)

        next_xy = jp.where(collision, cur_xy, cand_xy)

        # Directly set position, remove velocity carry-over, and keep ctrl shape unchanged.
        new_qpos = state.data.qpos.at[:2].set(next_xy)
        new_qvel = jp.zeros_like(state.data.qvel)
        new_ctrl = jp.zeros_like(state.data.ctrl)

        data = state.data.replace(qpos=new_qpos, qvel=new_qvel, ctrl=new_ctrl)

        xy = data.qpos[:2]
        dist = jp.linalg.norm(xy - self._goal_xy)
        success = (dist < self._config.success_thresh).astype(jp.float32)

        # Reward: base is success signal.
        # - terminate mode: collision gives -1 reward and ends episode
        # - collision_cost mode: collision subtracts a penalty but continues
        terminate_on_wall_collision = self._terminate_on_wall_collision
        collision_penalty = jp.where(
            self._apply_collision_cost & collision,
            jp.array(self._config.collision_cost, dtype=jp.float32),
            jp.array(0.0, dtype=jp.float32),
        )
        reward = jp.where(
            terminate_on_wall_collision & collision,
            jp.array(-1.0, dtype=jp.float32),
            success - collision_penalty,
        )

        steps = state.info["_steps"] + 1
        timeout = steps >= self._config.episode_length
        done = jp.where(
            timeout | (terminate_on_wall_collision & collision),
            1.0, 0.0,
        )

        state.info["_steps"] = steps
        wall_collisions = state.metrics["wall_collisions"] + collision.astype(jp.float32)
        state.metrics.update(dist=dist, success=success, wall_collisions=wall_collisions)

        rng, rng_obs = jax.random.split(state.info["rng"])
        state.info["rng"] = rng
        return state.replace(
            data=data,
            obs=self._get_obs(data, rng_obs),
            reward=reward,
            done=done,
        )

    def _get_obs(self, data: mjx.Data, rng: jax.Array) -> jax.Array:
        base = jp.concatenate([jp.asarray(data.qpos, dtype=jp.float32), jp.asarray(data.qvel, dtype=jp.float32), self._goal_xy,])
        return base

    def _state_component_names(self) -> list[str]:
        return ['x_1', 'x_2', 'v_1', 'v_2', 'goal_1', 'goal_2']

    def dynamics_complexity_scale(self, obs, obs_key='state'):
        state = np.asarray(obs[obs_key], np.float32)
        xy = state[..., :2]
        span = np.maximum(self._xy_max - self._xy_min, 1e-6)
        xy01 = np.clip((xy - self._xy_min) / span, 0.0, 1.0)
        left_lower = 1.0 - 0.5 * (xy01[..., 0] + xy01[..., 1])
        #return np.exp(-3.0 * left_lower).astype(np.float32)
        return 5.0 * np.clip(np.exp(-2.0 * np.log(2) * left_lower).astype(np.float32)-0.5, 0.0, 1.0)


    def get_state_table(
        self,
        points_per_cell: int = 5,
    ) -> Dict[str, Any]:
        """Return a cell-aligned grid of raw state observations over the maze domain.

        Samples ``points_per_cell`` evenly spaced positions per axis inside each
        open cell, so a downstream binning with one bin per cell will tile the
        maze exactly with no straddling of cell boundaries.
        """
        rows, cols = len(self._maze_layout), len(self._maze_layout[0])
        open_rs, open_cs = [], []
        for r, layout_row in enumerate(self._maze_layout):
            for c, v in enumerate(layout_row):
                if v != '#':
                    open_rs.append(r); open_cs.append(c)
        if not open_rs:
            raise ValueError('PointMaze layout contains no open cells.')

        r_min, r_max = min(open_rs), max(open_rs)
        c_min, c_max = min(open_cs), max(open_cs)

        # Sub-cell offsets in (-0.5, +0.5) cell-units, centered.
        if points_per_cell <= 1:
            sub = np.array([0.0], dtype=np.float32)
        else:
            sub = (np.arange(points_per_cell, dtype=np.float32) + 0.5) / points_per_cell - 0.5

        if self._positive_coords:
            x_centers = np.arange(c_min, c_max + 1, dtype=np.float32) * _MAZE_CELL_SIZE
            y_centers = ((rows - 1) - np.arange(r_min, r_max + 1, dtype=np.float32)) * _MAZE_CELL_SIZE
        else:
            x_centers = (np.arange(c_min, c_max + 1, dtype=np.float32) - (cols - 1) / 2.0) * _MAZE_CELL_SIZE
            y_centers = ((rows - 1) / 2.0 - np.arange(r_min, r_max + 1, dtype=np.float32)) * _MAZE_CELL_SIZE

        xs = (x_centers[:, None] + sub[None, :] * _MAZE_CELL_SIZE).reshape(-1)
        ys = (y_centers[:, None] + sub[None, :] * _MAZE_CELL_SIZE).reshape(-1)
        xs.sort()
        ys.sort()
        grid_x, grid_y = np.meshgrid(xs, ys, indexing='xy')
        xy = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)

        qvel = np.zeros((xy.shape[0], self._mj_model.nv), dtype=np.float32)
        goal = np.broadcast_to(
            np.asarray(self._goal_xy, dtype=np.float32), (xy.shape[0], 2))
        blocks = [xy.astype(np.float32), qvel, goal]

        state = np.concatenate(blocks, axis=-1)
        return dict(
            obs={'state': state},
            obs_col_names={'state': self._state_component_names()},
            grid_shape=np.array(grid_x.shape, dtype=np.int32),
        )

    def get_action_grid(self, points_per_axis: int = 21) -> np.ndarray:
        """Return a flattened normalized 2D action grid of shape [N, 2]."""
        if self._action_size != 2:
            raise NotImplementedError(
                'PointMaze action grid expects a 2D action space.')
        axis = np.linspace(-1.0, 1.0, points_per_axis, dtype=np.float32)
        g_ax, g_ay = np.meshgrid(axis, axis, indexing='xy')
        return np.stack([g_ax, g_ay], axis=-1).reshape(-1, 2)

    def get_action_table(
        self,
        state_points_per_axis: int = 5,
        action_points_per_axis: int = 21,
        state_points: np.ndarray = None,
    ) -> Dict[str, Any]:
        """Return a maze-aware state grid crossed with a normalized action grid.
        ``state_points`` can override the default open-cell centers."""
        if self._action_size != 2:
            raise NotImplementedError(
                'PointMaze action tables expect a 2D action space.')

        if state_points is not None:
            sp = np.asarray(state_points, dtype=np.float32)
            full_dim = 2 + self._mj_model.nv + 2
            if sp.ndim != 2 or sp.shape[-1] not in (2, full_dim):
                raise ValueError(
                    f'state_points must be [N, 2] or [N, {full_dim}]; got {sp.shape}')
            if sp.shape[-1] == 2:
                xy = sp
                qvel = np.zeros((xy.shape[0], self._mj_model.nv), dtype=np.float32)
                goal = np.broadcast_to(
                    np.asarray(self._goal_xy, dtype=np.float32), (xy.shape[0], 2))
                state = np.concatenate([xy, qvel, goal], axis=-1)
            else:
                state = sp
                xy = sp[:, :2]
            state_grid_shape = np.array([xy.shape[0]], dtype=np.int32)
        else:
            xy = _log_state_points()
            qvel = np.zeros((xy.shape[0], self._mj_model.nv), dtype=np.float32)
            goal = np.broadcast_to(
                np.asarray(self._goal_xy, dtype=np.float32), (xy.shape[0], 2))
            state = np.concatenate([xy.astype(np.float32), qvel, goal], axis=-1)
            state_grid_shape = np.array([xy.shape[0]], dtype=np.int32)

        actions = self.get_action_grid(action_points_per_axis)
        repeats = actions.shape[0]
        action_grid_shape = np.array(
            [action_points_per_axis, action_points_per_axis], dtype=np.int32)

        return dict(
            obs={'state': np.repeat(state, repeats, axis=0)},
            actions={'action': np.tile(actions, (state.shape[0], 1))},
            obs_col_names={'state': self._state_component_names()},
            action_col_names={'action': ['action_x', 'action_y']},
            state_grid_shape=state_grid_shape,
            action_grid_shape=action_grid_shape,
        )


# ---------------------------------------------------------------------------
# Random Point Maze – walls, start, and goal are regenerated every episode
# ---------------------------------------------------------------------------

_RANDOM_GRID_SIZES = {
    "small": 7,
    "medium": 9,
    "large": 11,
    "big": 13,
    "huge": 15,
    None: 9,
}


def _random_default_config(grid_size: int = 9):
    return config_dict.create(
        ctrl_dt=0.05,
        sim_dt=0.005,
        episode_length=1000,
        action_repeat=1,
        success_thresh=1.0,
        ctrl_cost=0.0,
        success_reward=10.0,
        action_scale=2.0,
        wall_prob=0.2,
        collision_cost=0.1,
        positive_coords=False,
    )


def _make_outer_walls_xml(
    rows: int,
    cols: int,
    cell_size: float = _MAZE_CELL_SIZE,
    wall_half_height: float = _MAZE_WALL_HALF_HEIGHT,
    positive_coords: bool = False,
) -> str:
    wall_half = cell_size * 0.55
    geoms = []
    for r in range(rows):
        for c in range(cols):
            if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                x, y = _cell_center(r, c, rows, cols, cell_size, positive_coords)
                geoms.append(
                    f'<geom name="maze_wall_{r}_{c}" type="box" '
                    f'pos="{x:.3f} {y:.3f} {wall_half_height:.3f}" '
                    f'size="{wall_half:.3f} {wall_half:.3f} {wall_half_height:.3f}" '
                    f'rgba="0.55 0.4 0.3 1" contype="1" conaffinity="1"/>'
                )
    return "\n      ".join(geoms)


class RandomPointMaze(mjx_env.MjxEnv):
    """PointMaze variant that randomizes walls, start, and goal every episode.

    Interior cells are independently toggled as walls with probability
    ``wall_prob`` (default 0.2).  Start and goal are placed on random open
    cells.  The outer boundary is always walled.

    Collision is checked entirely via vectorized JAX AABB tests against
    pre-computed cell positions, so the per-episode wall mask is fully
    compatible with ``jax.jit`` / ``jax.vmap``.

    Subtasks: ``"small"`` (7×7), ``"medium"`` (9×9, default), ``"large"``
    (11×11).  Append ``_terminate`` / ``_termwall`` to terminate on wall
    collision, or ``_collision_cost`` / ``_wallcost`` to apply a per-step
    penalty instead.
    """

    def __init__(
        self,
        sub_task=None,
        config=None,
        config_overrides: Optional[Dict[str, Union[str, int, float, list[Any]]]] = None,
    ):
        maze_variant, terminate_on_wall_collision, apply_collision_cost = _parse_wall_suffix(sub_task, default_variant="medium")
        grid_size = _RANDOM_GRID_SIZES.get(maze_variant)
        if grid_size is None:
            raise ValueError(
                f"Unknown sub_task: {sub_task!r}. "
                f"Choose from {list(_RANDOM_GRID_SIZES.keys())}"
            )

        if config is None:
            config = _random_default_config(grid_size)
        mjx_env.MjxEnv.__init__(self, config, config_overrides)

        self._grid_size = grid_size
        self._terminate_on_wall_collision = jp.array(
            terminate_on_wall_collision, dtype=jp.bool_
        )
        self._apply_collision_cost = jp.array(
            apply_collision_cost, dtype=jp.bool_
        )
        self._wall_prob = self._config.wall_prob
        self._positive_coords = bool(self._config.positive_coords)
        self._cell_size = _MAZE_CELL_SIZE
        wall_half = self._cell_size * 0.55

        rows = cols = grid_size

        # Separate outer (always walls) and interior cell positions.
        outer_positions = []
        interior_positions = []
        for r in range(rows):
            for c in range(cols):
                x, y = _cell_center(
                    r, c, rows, cols, self._cell_size, self._positive_coords)
                if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                    outer_positions.append((x, y))
                else:
                    interior_positions.append((x, y))

        self._outer_wall_pos = jp.array(outer_positions, dtype=jp.float32)
        self._outer_wall_half = jp.full(
            (len(outer_positions), 2), wall_half, dtype=jp.float32
        )
        self._interior_cell_pos = jp.array(interior_positions, dtype=jp.float32)
        self._interior_wall_half = jp.full(
            (len(interior_positions), 2), wall_half, dtype=jp.float32
        )
        self._num_interior = len(interior_positions)
        interior_np = np.asarray(interior_positions, dtype=np.float32)
        self._xy_min = interior_np.min(axis=0)
        self._xy_max = interior_np.max(axis=0)

        # Build MuJoCo model with outer walls only (for rendering).
        outer_xml = _make_outer_walls_xml(
            rows, cols, self._cell_size, positive_coords=self._positive_coords)
        xml = _build_point_maze_xml(outer_xml)
        mj_model = mujoco.MjModel.from_xml_string(xml)
        mj_model.opt.timestep = self._config.sim_dt

        self._mj_model = mj_model
        self._mjx_model = mjx.put_model(mj_model)
        self._init_qpos = jp.array(mj_model.qpos0, dtype=jp.float32)
        self._init_qvel = jp.zeros(self._mj_model.nv, dtype=jp.float32)
        self._action_size = int(self._mj_model.nu)
        self.action_scale = 0.025

    # -- properties ----------------------------------------------------------

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property
    def xml_path(self) -> str:
        return "<in-memory-random-point-maze-xml>"

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def observation_size(self) -> int:
        # qpos [x, y] + qvel [vx, vy] + goal [gx, gy]
        return self._mj_model.nq + self._mj_model.nv + 2

    # -- reset / step --------------------------------------------------------

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, rng_walls, rng_start, rng_goal = jax.random.split(rng, 4)

        # Random interior wall mask.
        wall_probs = jax.random.uniform(rng_walls, (self._num_interior,))
        interior_wall_mask = wall_probs < self._wall_prob

        # Random start and goal among interior cells.
        start_idx = jax.random.randint(rng_start, (), 0, self._num_interior)
        goal_idx = jax.random.randint(rng_goal, (), 0, self._num_interior)
        goal_idx = jp.where(
            goal_idx == start_idx, (goal_idx + 1) % self._num_interior, goal_idx
        )

        # Guarantee start/goal cells are open.
        interior_wall_mask = interior_wall_mask.at[start_idx].set(False)
        interior_wall_mask = interior_wall_mask.at[goal_idx].set(False)

        start_xy = self._interior_cell_pos[start_idx]
        goal_xy = self._interior_cell_pos[goal_idx]

        qpos = self._init_qpos.at[:2].set(start_xy)
        qvel = self._init_qvel

        data = mjx_env.init(
            self._mjx_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=jp.zeros(self._action_size, dtype=jp.float32),
        )

        info = {
            "rng": rng,
            "_steps": jp.array(0, dtype=jp.int32),
            "goal_xy": goal_xy,
            "interior_wall_mask": interior_wall_mask,
        }
        metrics = {
            "dist": jp.linalg.norm(start_xy - goal_xy),
            "success": jp.array(0.0, dtype=jp.float32),
            "wall_collisions": jp.array(0.0, dtype=jp.float32),
        }

        obs = self._get_obs(data, goal_xy)
        return mjx_env.State(
            data,
            obs,
            jp.array(0.0, dtype=jp.float32),
            jp.array(0.0, dtype=jp.float32),
            metrics,
            info,
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        action = jp.clip(jp.asarray(action, dtype=jp.float32), -1.0, 1.0)
        delta = self.action_scale * action

        cur_xy = jp.asarray(state.data.qpos[:2], dtype=jp.float32)
        cand_xy = cur_xy + delta

        ball_radius = 0.35

        # Outer walls – always active.
        outer_collision = _aabb_collision(
            cand_xy, self._outer_wall_pos, self._outer_wall_half, ball_radius
        )

        # Interior walls – masked per episode.
        interior_collision = _aabb_collision_masked(
            cand_xy,
            self._interior_cell_pos,
            self._interior_wall_half,
            state.info["interior_wall_mask"],
            ball_radius,
        )

        collision = outer_collision | interior_collision
        next_xy = jp.where(collision, cur_xy, cand_xy)

        new_qpos = state.data.qpos.at[:2].set(next_xy)
        new_qvel = jp.zeros_like(state.data.qvel)
        new_ctrl = jp.zeros_like(state.data.ctrl)
        data = state.data.replace(qpos=new_qpos, qvel=new_qvel, ctrl=new_ctrl)

        goal_xy = state.info["goal_xy"]
        xy = data.qpos[:2]
        dist = jp.linalg.norm(xy - goal_xy)
        success = (dist < self._config.success_thresh).astype(jp.float32)

        # Reward: base is success signal.
        # - terminate mode: collision gives -1 reward and ends episode
        # - collision_cost mode: collision subtracts a penalty but continues
        terminate_on_wall_collision = self._terminate_on_wall_collision
        collision_penalty = jp.where(
            self._apply_collision_cost & collision,
            jp.array(self._config.collision_cost, dtype=jp.float32),
            jp.array(0.0, dtype=jp.float32),
        )
        reward = jp.where(
            terminate_on_wall_collision & collision,
            jp.array(-1.0, dtype=jp.float32),
            success - collision_penalty,
        )

        steps = state.info["_steps"] + 1
        timeout = steps >= self._config.episode_length
        done = jp.where(
            timeout | (terminate_on_wall_collision & collision),
            1.0, 0.0,
        )

        state.info["_steps"] = steps
        wall_collisions = state.metrics["wall_collisions"] + collision.astype(jp.float32)
        state.metrics.update(dist=dist, success=success, wall_collisions=wall_collisions)

        return state.replace(
            data=data,
            obs=self._get_obs(data, goal_xy),
            reward=reward,
            done=done,
        )

    def _get_obs(self, data: mjx.Data, goal_xy: jax.Array) -> jax.Array:
        return jp.concatenate(
            [
                jp.asarray(data.qpos, dtype=jp.float32),
                jp.asarray(data.qvel, dtype=jp.float32),
                goal_xy,
            ]
        )

    def dynamics_complexity_scale(self, obs, obs_key='state'):
        state = np.asarray(obs[obs_key], np.float32)
        xy = state[..., :2]
        span = np.maximum(self._xy_max - self._xy_min, 1e-6)
        xy01 = np.clip((xy - self._xy_min) / span, 0.0, 1.0)
        left_lower = 1.0 - 0.5 * (xy01[..., 0] + xy01[..., 1])
        #return np.exp(-3.0 * left_lower).astype(np.float32)
        return np.clip(np.exp(-2.0 * np.log(2) * left_lower).astype(np.float32)-0.5, 0.0, 1.0)


# -- Vectorized AABB helpers (pure JAX, jit/vmap safe) ----------------------

def _aabb_collision(
    point: jax.Array,
    wall_pos: jax.Array,
    wall_half: jax.Array,
    ball_radius: float,
) -> jax.Array:
    """True if *point* (with *ball_radius*) overlaps any wall box."""
    lower = wall_pos - wall_half - ball_radius
    upper = wall_pos + wall_half + ball_radius
    inside = (point >= lower) & (point <= upper)  # [N, 2]
    return inside.all(axis=-1).any()


def _aabb_collision_masked(
    point: jax.Array,
    wall_pos: jax.Array,
    wall_half: jax.Array,
    mask: jax.Array,
    ball_radius: float,
) -> jax.Array:
    """Like ``_aabb_collision`` but only considers walls where *mask* is True."""
    lower = wall_pos - wall_half - ball_radius
    upper = wall_pos + wall_half + ball_radius
    inside = (point >= lower) & (point <= upper)  # [N, 2]
    return (inside.all(axis=-1) & mask).any()
