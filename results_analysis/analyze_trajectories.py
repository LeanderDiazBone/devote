"""Offline trajectory analysis: load k equally-spaced checkpoints and compute Q-metrics.

Usage:
    python multimex/optimistic_curiosity/analyze_trajectories.py \
        --logdir /path/to/run/logdir \
        --alg Observer \
        --num_checkpoints 5

Loads config.yaml, either eval_trajectories/ or explore_trajectories/, and up to
--num_checkpoints equally-spaced checkpoint files from <logdir>/. Trajectories
are merged into a single cached file so subsequent runs skip per-file loading.

Produces trajectory_metrics_<step>.npz per checkpoint in a mode-specific
analysis output directory under <logdir>/.
"""

import pathlib
import re
import sys

directory = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(directory.parent))
sys.path.insert(0, str(directory.parent.parent))

import glob
import os

import jax
import numpy as np

import embodied

os.environ.setdefault('MUJOCO_GL', 'egl')

# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def discover_checkpoints(logdir, num_checkpoints):
    """Find all checkpoint files under *logdir* and return *num_checkpoints*
    equally-spaced paths (always including the first and last).

    Supports common naming patterns:
        checkpoint_<step>.ckpt  /  checkpoint-<step>.ckpt
        cp<step>.ckpt
    """
    logdir = pathlib.Path(str(logdir))
    ckpt_files = sorted(
        logdir.glob('**/*.ckpt'),
        key=lambda p: p.name,
    )
    if not ckpt_files:
        raise FileNotFoundError(f'No .ckpt files found under {logdir}')

    def _step(path):
        m = re.search(r'(\d+)', path.stem)
        return int(m.group(1)) if m else None

    ckpt_files = [p for p in ckpt_files if _step(p) is not None]
    if not ckpt_files:
        raise FileNotFoundError(f'No numbered .ckpt files found under {logdir}')

    ckpt_files = sorted(ckpt_files, key=_step)

    if len(ckpt_files) <= num_checkpoints:
        return ckpt_files

    # Pick equally-spaced indices (always include first & last).
    indices = np.round(np.linspace(0, len(ckpt_files) - 1, num_checkpoints)).astype(int)
    indices = sorted(set(indices))
    return [ckpt_files[i] for i in indices]


# ---------------------------------------------------------------------------
# Trajectory caching
# ---------------------------------------------------------------------------

def load_trajectories(traj_dir, cache_path):
    """Load trajectories from *traj_dir*.  If *cache_path* already exists,
    load the pre-combined file instead of reading every individual .npz.

    Returns a list[dict[str, np.ndarray]].
    """
    cache_path = pathlib.Path(str(cache_path))

    if cache_path.exists():
        print(f'Loading cached combined trajectories from {cache_path}')
        data = np.load(str(cache_path), allow_pickle=True)
        trajectories = list(data['trajectories'])
        # Each element is a dict saved via allow_pickle.
        print(f'  → {len(trajectories)} trajectories from cache')
        return trajectories

    # Fall back to loading individual files.
    traj_files = sorted(glob.glob(os.path.join(str(traj_dir), '*.npz')))
    assert traj_files, f'No .npz trajectory files found in {traj_dir}'
    print(f'Loading {len(traj_files)} trajectory files from {traj_dir}')
    trajectories = [dict(np.load(f)) for f in traj_files]

    # Cache for next time.
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(cache_path), trajectories=np.array(trajectories, dtype=object))
    print(f'Cached combined trajectories → {cache_path}')

    return trajectories


# ---------------------------------------------------------------------------
# Environment-backed state/action tables
# ---------------------------------------------------------------------------

def build_state_table(config):
    """Build an environment-backed state table when raw state inputs are used."""
    if str(getattr(config, 'ac_inputs', 'wm')) != 'obs':
        return None

    from experiments import make_env as _make_env

    env = _make_env(config, 0, wrapper_log_images=False)
    try:
        try:
            get_state_table = env.get_state_table
        except AttributeError:
            return None
        try:
            table = get_state_table()
        except (NotImplementedError, AttributeError):
            return None
        if table is None:
            return None
    finally:
        env.close()

    if not isinstance(table, dict) or 'obs' not in table:
        raise ValueError(
            'Environment get_state_table() must return a dict containing "obs".')
    return table


def build_action_table(config):
    """Build an environment-backed state-action table when raw state inputs are used."""
    if str(getattr(config, 'ac_inputs', 'wm')) != 'obs':
        return None

    from experiments import make_env as _make_env

    env = _make_env(config, 0, wrapper_log_images=False)
    try:
        try:
            get_action_table = env.get_action_table
        except AttributeError:
            return None
        except ValueError as e:
            if str(e) == 'get_action_table':
                return None
            raise
        try:
            table = get_action_table()
        except (NotImplementedError, AttributeError):
            return None
        except ValueError as e:
            if str(e) == 'get_action_table':
                return None
            raise
        if table is None:
            return None
    finally:
        env.close()

    if not isinstance(table, dict) or 'obs' not in table or 'actions' not in table:
        raise ValueError(
            'Environment get_action_table() must return a dict containing '
            '"obs" and "actions".')
    return table


# ---------------------------------------------------------------------------
# Analysis for a single checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint(agent, checkpoint_path):
    """Load *checkpoint_path* into *agent* and return the parsed step."""
    ckpt = embodied.Checkpoint()
    ckpt.agent = agent
    ckpt.load(str(checkpoint_path), keys=['agent'])
    step_match = re.findall(r'(\d+)', pathlib.Path(str(checkpoint_path)).stem)
    if not step_match:
        raise ValueError(f'Checkpoint path has no numeric step: {checkpoint_path}')
    return float(step_match[-1])


def _extract_table_metric(mets, needle):
    for key in mets:
        if needle in key:
            return key
    return None


def analyze_checkpoint(agent, trajectories, config, out_path, tag, checkpoint_step):
    """Run report across all trajectories and write trajectory_metrics_<tag>.npz."""
    print(f'\n=== Checkpoint tag={tag}  step={checkpoint_step:.0f} ===')

    B = config.batch_size
    T = config.batch_length_eval
    max_len = max(len(traj[list(traj.keys())[0]]) for traj in trajectories)
    print(f'  Trajectory length: {max_len}, chunk size T={T}, batch size B={B}')

    # Debug: compare keys (once).
    traj_keys = set(trajectories[0].keys())
    agent_keys = set(agent.keys)
    print(f'  Keys in agent but not trajectory: {sorted(agent_keys - traj_keys)}')
    print(f'  Keys in trajectory but not agent: {sorted(traj_keys - agent_keys)}')
    for k in sorted(traj_keys):
        v = trajectories[0][k]
        print(f'    {k}: shape={v.shape}, dtype={v.dtype}, '
              f'min={v.min():.4f}, max={v.max():.4f}')

    results = []
    for batch_start in range(0, len(trajectories), B):
        batch_trajs = trajectories[batch_start:batch_start + B]
        if len(batch_trajs) < B:
            while len(batch_trajs) < B:
                batch_trajs.append(batch_trajs[-1])

        carry = agent.init_report(B)
        chunk_tables = []
        n_real = min(len(trajectories) - batch_start, B)
        col_names = None

        for t_start in range(0, max_len, T):
            chunk_trajs = []
            for traj in batch_trajs:
                chunk = {}
                for k, v in traj.items():
                    chunk[k] = v[t_start:t_start + T]
                chunk_trajs.append(chunk)

            actual_chunk_len = len(
                chunk_trajs[0][list(chunk_trajs[0].keys())[0]]
            )
            data = _make_batch(chunk_trajs, T, agent.spaces, agent.keys)
            if t_start > 0:
                data['is_first'][:, 0] = False
            data = {
                **jax.device_put(data, agent.train_sharded),
                'seed': agent._next_seeds(agent.train_sharded),
            }
            mets, carry = agent.report(data, carry)

            # Extract calibration table from this chunk.
            table_key = _extract_table_metric(mets, 'calibration_table')
            if table_key:
                if col_names is None:
                    col_names = table_key.split('__')[1:]
                    col_names = '__'.join(col_names).split('__')
                table = np.array(mets[table_key])
                rows_per_traj = table.shape[0] // B
                traj_ids = np.repeat(
                    np.arange(batch_start, batch_start + B), rows_per_traj
                ).astype(np.float32)
                timesteps = np.tile(
                    np.arange(rows_per_traj) + t_start, B
                ).astype(np.float32)
                train_steps = []
                for traj in chunk_trajs:
                    traj_train_steps = traj.get('train_step')
                    if traj_train_steps is None:
                        train_steps.append(
                            np.full(rows_per_traj, np.nan, dtype=np.float32))
                        continue
                    traj_train_steps = np.asarray(
                        traj_train_steps[:rows_per_traj], dtype=np.float32)
                    if len(traj_train_steps) < rows_per_traj:
                        padded = np.full(rows_per_traj, np.nan, dtype=np.float32)
                        padded[:len(traj_train_steps)] = traj_train_steps
                        traj_train_steps = padded
                    train_steps.append(traj_train_steps)
                train_steps = np.concatenate(train_steps, axis=0)
                checkpoint_steps = np.full(
                    B * rows_per_traj, checkpoint_step, dtype=np.float32)
                table = np.column_stack(
                    [table, traj_ids, timesteps, train_steps, checkpoint_steps])
                # Trim padding rows.
                if actual_chunk_len < T:
                    actual_rows = max(
                        actual_chunk_len - (T - rows_per_traj), 0
                    )
                    mask = np.zeros(B * rows_per_traj, dtype=bool)
                    for i in range(B):
                        mask[i * rows_per_traj:
                             i * rows_per_traj + actual_rows] = True
                    table = table[mask]
                chunk_tables.append(table)

        if chunk_tables:
            full_table = np.concatenate(chunk_tables, axis=0)
            results.append({
                'table': full_table,
                'col_names': col_names + ['traj_id', 'timestep', 'train_step', 'checkpoint_step'],
                'n_real': n_real,
                'batch_start': batch_start,
            })
            print(f'    Batch {batch_start}: table shape {full_table.shape}')
        else:
            print(f'    Batch {batch_start}: no calibration table in metrics')
            print(f'    Available metric keys: {sorted(mets.keys())[:20]}...')

    # Save per-checkpoint results.
    if results:
        all_tables = np.concatenate([r['table'] for r in results], axis=0)
        col_names = results[0]['col_names']
        out_file = out_path / f'trajectory_metrics_{tag}.npz'
        np.savez(str(out_file), table=all_tables, col_names=col_names)
        print(f'  Saved {out_file} with shape {all_tables.shape}')
        print(f'  Columns: {col_names}')
        _print_metric_summary(all_tables, col_names)
        return all_tables, col_names
    else:
        print('  No results produced for this checkpoint.')
        return None, None


def _rename_state_columns(col_names, state_table):
    renamed = list(col_names)
    for names_key in ('obs_col_names', 'action_col_names'):
        for key, names in state_table.get(names_key, {}).items():
            generic = [key] if len(names) == 1 else [f'{key}_{i}' for i in range(len(names))]
            mapping = dict(zip(generic, names))
            renamed = [mapping.get(name, name) for name in renamed]
    return renamed


def _action_columns(action_table):
    columns = []
    names = []
    named = action_table.get('action_col_names', {})
    source = action_table.get('action_values', action_table['actions'])
    for key, values in source.items():
        values = np.asarray(values).reshape(len(values), -1).astype(np.float32)
        columns.append(values)
        custom = named.get(key)
        if custom is not None and len(custom) == values.shape[-1]:
            names.extend(custom)
        else:
            dim = values.shape[-1]
            names.extend([key] if dim == 1 else [f'{key}_{i}' for i in range(dim)])
    return np.concatenate(columns, axis=-1), names

def _make_table_report_batch(
        batch_obs, batch_actions, batch_size, time_size, spaces, valid_keys,
        table_code):
    """Create a synthetic fixed-length batch so report_q_calibration can score rows."""
    batch = {}
    for key in valid_keys:
        if key not in spaces:
            continue
        space = spaces[key]
        batch[key] = np.zeros((batch_size, time_size, *space.shape), dtype=space.dtype)

    for key, values in batch_obs.items():
        values = np.asarray(values)
        repeated = np.repeat(values[:, None], time_size, axis=1)
        batch[key] = repeated.astype(batch[key].dtype, copy=False)
    for key, values in (batch_actions or {}).items():
        values = np.asarray(values)
        repeated = np.repeat(values[:, None], time_size, axis=1)
        batch[key] = repeated.astype(batch[key].dtype, copy=False)

    batch['reward'].fill(0.0)
    batch['is_first'].fill(True)
    batch['is_last'].fill(False)
    batch['is_terminal'].fill(False)
    if 'stepid' in batch:
        batch['stepid'].fill(0)
        batch['stepid'][..., 0] = table_code
    return batch


def _make_state_table_report_batch(batch_obs, batch_size, time_size, spaces, valid_keys):
    return _make_table_report_batch(
        batch_obs, None, batch_size, time_size, spaces, valid_keys,
        table_code=255)


def analyze_state_table(agent, state_table, config, out_path, tag, checkpoint_step):
    """Run report() on a synthetic state batch and write state_table_metrics_<tag>.npz."""
    B = config.batch_size
    T = config.batch_length_eval
    if T < 2:
        raise ValueError(
            f'State-table analysis requires batch_length_eval >= 2, got {T}.')
    obs_keys = list(state_table['obs'].keys())
    num_states = len(state_table['obs'][obs_keys[0]])
    print(f'  State table rows: {num_states}, synthetic chunk size T={T}, batch size B={B}')

    results = []
    col_names = None
    rows_per_state = max(T - 1, 1)

    for batch_start in range(0, num_states, B):
        n_real = min(num_states - batch_start, B)
        batch_obs = {}
        for key in obs_keys:
            values = np.asarray(state_table['obs'][key])
            chunk = values[batch_start:batch_start + n_real]
            if n_real < B:
                pad = np.repeat(chunk[-1:], B - n_real, axis=0)
                chunk = np.concatenate([chunk, pad], axis=0)
            batch_obs[key] = chunk

        carry = agent.init_report(B)
        data = _make_state_table_report_batch(
            batch_obs, B, T, agent.spaces, agent.keys)
        data = {
            **jax.device_put(data, agent.train_sharded),
            'seed': agent._next_seeds(agent.train_sharded),
        }
        mets, carry = agent.report(data, carry)

        table_key = _extract_table_metric(mets, 'calibration_table')
        if not table_key:
            print('  No state-table calibration table produced for this checkpoint.')
            return None, None
        if col_names is None:
            col_names = table_key.split('__')[1:]
            col_names = '__'.join(col_names).split('__')
            col_names = _rename_state_columns(col_names, state_table)

        raw_table = np.asarray(mets[table_key], dtype=np.float32)
        select = np.arange(B) * rows_per_state
        table = raw_table[select[:n_real]]
        state_ids = np.arange(batch_start, batch_start + n_real, dtype=np.float32)
        checkpoint_steps = np.full(n_real, checkpoint_step, dtype=np.float32)
        table = np.column_stack([table, state_ids, checkpoint_steps])
        results.append(table)

    if not results:
        print('  No state-table results produced for this checkpoint.')
        return None, None

    all_tables = np.concatenate(results, axis=0)
    col_names = col_names + ['state_id', 'checkpoint_step']
    save_kwargs = dict(table=all_tables, col_names=np.array(col_names, dtype=object))
    if 'grid_shape' in state_table:
        save_kwargs['grid_shape'] = np.asarray(state_table['grid_shape'], dtype=np.int32)

    out_file = out_path / f'state_table_metrics_{tag}.npz'
    np.savez(str(out_file), **save_kwargs)
    print(f'  Saved {out_file} with shape {all_tables.shape}')
    print(f'  State-table columns: {col_names}')
    _print_metric_summary(all_tables, col_names)
    return all_tables, col_names


def analyze_action_table(agent, action_table, config, out_path, tag, checkpoint_step):
    """Run report() on synthetic state-action rows and write action_table_metrics_<tag>.npz."""
    B = config.batch_size
    T = config.batch_length_eval
    if T < 2:
        raise ValueError(
            f'Action-table analysis requires batch_length_eval >= 2, got {T}.')
    obs_keys = list(action_table['obs'].keys())
    act_keys = list(action_table['actions'].keys())
    num_rows = len(action_table['obs'][obs_keys[0]])
    print(f'  Action table rows: {num_rows}, synthetic chunk size T={T}, batch size B={B}')

    results = []
    col_names = None
    action_values, action_col_names = _action_columns(action_table)
    action_grid_shape = np.asarray(
        action_table.get('action_grid_shape', [1]), dtype=np.int32)
    actions_per_state = int(np.prod(action_grid_shape))
    rows_per_item = max(T - 1, 1)

    for batch_start in range(0, num_rows, B):
        n_real = min(num_rows - batch_start, B)
        batch_obs = {}
        batch_actions = {}
        for key in obs_keys:
            values = np.asarray(action_table['obs'][key])
            chunk = values[batch_start:batch_start + n_real]
            if n_real < B:
                pad = np.repeat(chunk[-1:], B - n_real, axis=0)
                chunk = np.concatenate([chunk, pad], axis=0)
            batch_obs[key] = chunk
        for key in act_keys:
            values = np.asarray(action_table['actions'][key])
            chunk = values[batch_start:batch_start + n_real]
            if n_real < B:
                pad = np.repeat(chunk[-1:], B - n_real, axis=0)
                chunk = np.concatenate([chunk, pad], axis=0)
            batch_actions[key] = chunk

        carry = agent.init_report(B)
        data = _make_table_report_batch(
            batch_obs, batch_actions, B, T, agent.spaces, agent.keys,
            table_code=254)
        data = {
            **jax.device_put(data, agent.train_sharded),
            'seed': agent._next_seeds(agent.train_sharded),
        }
        mets, carry = agent.report(data, carry)

        table_key = _extract_table_metric(mets, 'q_calibration_table')
        if not table_key:
            print('  No action-table calibration table produced for this checkpoint.')
            return None, None
        if col_names is None:
            col_names = table_key.split('__')[1:]
            col_names = '__'.join(col_names).split('__')
            col_names = _rename_state_columns(col_names, action_table)

        raw_table = np.asarray(mets[table_key], dtype=np.float32)
        select = np.arange(B) * rows_per_item
        table = raw_table[select[:n_real]]
        actions = action_values[batch_start:batch_start + n_real]
        row_ids = np.arange(batch_start, batch_start + n_real, dtype=np.float32)
        state_ids = (row_ids // actions_per_state).astype(np.float32)
        action_ids = (row_ids % actions_per_state).astype(np.float32)
        checkpoint_steps = np.full(n_real, checkpoint_step, dtype=np.float32)
        table = np.column_stack([
            table, actions, state_ids, action_ids, row_ids, checkpoint_steps])
        results.append(table)

    if not results:
        print('  No action-table results produced for this checkpoint.')
        return None, None

    all_tables = np.concatenate(results, axis=0)
    col_names = col_names + action_col_names + [
        'state_id', 'action_id', 'row_id', 'checkpoint_step']
    save_kwargs = dict(table=all_tables, col_names=np.array(col_names, dtype=object))
    for key in ('state_grid_shape', 'action_grid_shape', 'grid_shape'):
        if key in action_table:
            save_kwargs[key] = np.asarray(action_table[key], dtype=np.int32)

    out_file = out_path / f'action_table_metrics_{tag}.npz'
    np.savez(str(out_file), **save_kwargs)
    print(f'  Saved {out_file} with shape {all_tables.shape}')
    print(f'  Action-table columns: {col_names}')
    _print_metric_summary(all_tables, col_names)
    return all_tables, col_names


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

def _make_batch(trajectories, T, spaces, valid_keys):
    """Format trajectories into a (B, T, ...) batch for agent.report().

    Pads or truncates trajectories to length T.  Fills missing keys with zeros.
    """
    B = len(trajectories)
    batch = {}

    for key in valid_keys:
        if key in spaces:
            space = spaces[key]
            arr = np.zeros((B, T, *space.shape), dtype=space.dtype)
            for i, traj in enumerate(trajectories):
                if key in traj:
                    tlen = min(len(traj[key]), T)
                    arr[i, :tlen] = traj[key][:tlen]
            batch[key] = arr

    if 'is_first' in batch:
        batch['is_first'][:, 0] = True

    return batch


def _print_metric_summary(table, col_names):
    """Print compact summary stats for key per-checkpoint metric columns."""
    preferred = [
        'q_mean',
        'q_std',
        'q_head_0',
        'return',
        'error',
        'reward',
        'reward_pred_mean',
        'reward_pred_std',
        'intr_q_mean',
        'intr_reward',
        'weighted_intr_reward',
    ]
    col_idx = {name: i for i, name in enumerate(col_names)}
    present = [name for name in preferred if name in col_idx]
    if not present:
        return

    print('  Metric summary:')
    for name in present:
        values = np.asarray(table[:, col_idx[name]], dtype=np.float32)
        finite = np.isfinite(values)
        finite_count = int(finite.sum())
        if finite_count == 0:
            print(f'    {name:18s} finite=0/{len(values)}')
            continue
        finite_vals = values[finite]
        nonzero_frac = float(np.count_nonzero(np.abs(finite_vals) > 1e-8)) / finite_count
        print(
            f'    {name:18s} '
            f'finite={finite_count}/{len(values)} '
            f'nonzero={nonzero_frac:.3f} '
            f'mean={finite_vals.mean(): .4e} '
            f'std={finite_vals.std(): .4e} '
            f'min={finite_vals.min(): .4e} '
            f'max={finite_vals.max(): .4e}'
        )

    if 'q_mean' in col_idx and 'q_head_0' in col_idx:
        q_mean = np.asarray(table[:, col_idx['q_mean']], dtype=np.float32)
        q_head = np.asarray(table[:, col_idx['q_head_0']], dtype=np.float32)
        finite = np.isfinite(q_mean) & np.isfinite(q_head)
        if finite.any():
            diff = np.abs(q_mean[finite] - q_head[finite])
            print(
                '    q_mean_vs_q_head_0   '
                f'mean_abs_diff={diff.mean(): .4e} '
                f'max_abs_diff={diff.max(): .4e}'
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    from experiments import make_agent as _make_agent

    parsed, other = embodied.Flags(
        logdir='',
        alg='Dreamer',
        num_checkpoints=5,
        trajectory_mode='exploitation',
    ).parse_known(argv)

    logdir = embodied.Path(parsed.logdir)
    assert logdir, 'Must specify --logdir'
    alg = parsed.alg
    num_checkpoints = int(parsed.num_checkpoints)
    trajectory_mode = str(parsed.trajectory_mode).lower()
    assert trajectory_mode in ('exploitation', 'exploration'), (
        f'Unsupported trajectory_mode={trajectory_mode!r}. '
        'Expected exploitation or exploration.'
    )

    # Reload saved config from the training run.
    import ruamel.yaml as yaml
    config_data = yaml.YAML(typ='safe').load((logdir / 'config.yaml').read())
    # Analysis-specific overrides.
    config_data['log_images'] = False
    config_data['use_image'] = False
    config_data["log_metrics_table"] = True
    config = embodied.Config(config_data)
    config = embodied.Flags(config).parse(other)

    # Paths.
    if trajectory_mode == 'exploration':
        traj_dir = str(logdir / 'explore_trajectories')
        out_path = logdir / 'analysis_output_explore'
        cache_name = 'combined_trajectories_explore.npz'
    else:
        traj_dir = str(logdir / 'eval_trajectories')
        out_path = logdir / 'analysis_output'
        cache_name = 'combined_trajectories.npz'
    out_path.mkdir()
    cache_path = out_path / cache_name

    # Load (or cache) trajectories once.
    trajectories = load_trajectories(traj_dir, cache_path)
    state_table = build_state_table(config)
    if state_table is not None:
        print('Built environment-backed state table '
              f'with {len(next(iter(state_table["obs"].values())))} rows.')
    else:
        print('No environment-backed state table available for this run.')
    action_table = build_action_table(config)
    if action_table is not None:
        print('Built environment-backed action table '
              f'with {len(next(iter(action_table["obs"].values())))} rows.')
    else:
        print('No environment-backed action table available for this run.')

    # Discover checkpoints.
    ckpt_paths = discover_checkpoints(logdir, num_checkpoints)
    print(f'\nSelected {len(ckpt_paths)} checkpoints:')
    for p in ckpt_paths:
        print(f'  {p}')

    # Create agent once – weights are overwritten per checkpoint.
    agent = _make_agent(config, alg=alg)
    # Run analysis for each checkpoint.
    summary = []
    state_summary = []
    action_summary = []
    for ckpt_path in ckpt_paths:
        m = re.search(r'(\d+)', pathlib.Path(str(ckpt_path)).stem)
        tag = m.group(1)
        checkpoint_step = load_checkpoint(agent, ckpt_path)
        table, col_names = analyze_checkpoint(
            agent, trajectories, config, out_path, tag, checkpoint_step,
        )
        if table is not None:
            summary.append({'tag': tag, 'path': str(ckpt_path),
                            'rows': table.shape[0]})
        if state_table is not None:
            state_metrics, _ = analyze_state_table(
                agent, state_table, config, out_path, tag, checkpoint_step)
            if state_metrics is not None:
                state_summary.append({
                    'tag': tag,
                    'path': str(ckpt_path),
                    'rows': state_metrics.shape[0],
                })
        if action_table is not None:
            action_metrics, _ = analyze_action_table(
                agent, action_table, config, out_path, tag, checkpoint_step)
            if action_metrics is not None:
                action_summary.append({
                    'tag': tag,
                    'path': str(ckpt_path),
                    'rows': action_metrics.shape[0],
                })

    print('\n=== Summary ===')
    for s in summary:
        print(f'  {s["tag"]}: {s["rows"]} rows  ({s["path"]})')
    if state_summary:
        print('\n=== State Table Summary ===')
        for s in state_summary:
            print(f'  {s["tag"]}: {s["rows"]} rows  ({s["path"]})')
    if action_summary:
        print('\n=== Action Table Summary ===')
        for s in action_summary:
            print(f'  {s["tag"]}: {s["rows"]} rows  ({s["path"]})')
    print('Done.')


if __name__ == '__main__':
    main()
