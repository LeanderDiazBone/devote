"""Policy-evaluation run mode.

Workflow:
  1. Build an ObserverAgent configured for uncertainty-aware Q training (e.g.
     td_target_mode in {q_full, rnd}, policy_mode=exp_actor).
  2. Load the trained `actor_sac` weights from a checkpoint into the agent's
     `exp_actor_sac` slot; restore its prior and optionally its corrector.
  3. Compute MC ground-truth returns by setting DMC physics state at each
     anchor + sweep point and rolling out the frozen exp_actor.
  4. Run the standard Q-training loop (policy frozen via loss_scales) while
     periodically reading TD predictions on the same anchor + sweep points.

Outputs (under args.logdir / 'policy_eval/'):
  * `mc.npz`               — anchor + sweep MC means/stds (computed once).
  * `mc_rollouts/*.npz`    — per-step rewards, states, actions, and masks;
                            frozen-corrector runs also include prior/corrector
                            particles in unit-scale RB target units.
  * `td_<step>.npz`        — TD predictions at each eval step.
  * `anchors.npz`          — anchor metadata for downstream plotting.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial as bind
from pathlib import Path

import embodied
import numpy as np

# Repo-local imports (not under embodied/) — these are loaded lazily inside
# the function so this module stays importable without the experiments tree
# being on the path during pure-embodied use.


# ---------------------- checkpoint remap ----------------------

# ---------------------- anchor / sweep evaluation grid ----------------------

def _flatten_eval_points(anchors):
    """Return a flat list of {anchor_idx, kind, sweep_idx, qpos, qvel, action,
    natural_reset} rows. `kind` is one of {'anchor', 'state', 'action'}. Sweep
    rows hold the perturbed qpos/qvel or action; anchor rows hold the originals.
    Anchors marked `natural_reset=True` contribute only a single row and no
    sweeps."""
    rows = []
    for ai, a in enumerate(anchors):
        natural = bool(getattr(a, 'natural_reset', False))
        rows.append(dict(anchor_idx=ai, anchor_name=a.name, anchor_tag=a.tag, kind='anchor',
                         sweep_idx=0, qpos=a.qpos.copy(), qvel=a.qvel.copy(),
                         action=a.action.copy(), natural_reset=natural))
        if natural:
            continue
        if a.state_sweep:
            for si, val in enumerate(a.state_sweep['values']):
                qpos = a.qpos.copy(); qvel = a.qvel.copy()
                arr = qpos if a.state_sweep['kind'] == 'qpos' else qvel
                arr[a.state_sweep['index']] = val
                rows.append(dict(anchor_idx=ai, anchor_name=a.name, anchor_tag=a.tag, kind='state',
                                 sweep_idx=si, qpos=qpos, qvel=qvel,
                                 action=a.action.copy(), natural_reset=False))
        if a.action_sweep:
            for si, val in enumerate(a.action_sweep['values']):
                action = a.action.copy()
                action[a.action_sweep['index']] = val
                rows.append(dict(anchor_idx=ai, anchor_name=a.name, anchor_tag=a.tag, kind='action',
                                 sweep_idx=si, qpos=a.qpos.copy(), qvel=a.qvel.copy(),
                                 action=action, natural_reset=False))
    return rows


# ---------------------- Anchor-vs-visitation distance ----------------------


_VISIT_SKIP_KEYS = frozenset(
    {'reward', 'is_first', 'is_last', 'is_terminal', 'image', 'log_image'})


def _proprio_keys(env):
    """All proprio obs keys (everything except control flow + image)."""
    return sorted(k for k in env.obs_space if k not in _VISIT_SKIP_KEYS)


def _stack_proprio(obs, keys):
    """Stack the given obs keys into a single flat float32 vector."""
    return np.concatenate(
        [np.atleast_1d(np.asarray(obs[k], dtype=np.float32)).ravel() for k in keys]
    )


def _collect_policy_visitation(task, action_repeat, agent, num_steps, num_envs=8,
                               seed=0, max_step_limit=0):
    """Roll out the frozen explore-actor from the environment's natural reset and
    collect every visited proprio obs vector. Used to score each anchor by how
    close it is to the policy's on-policy visitation.

    Returns (visited [N, D], proprio_keys list) where D is the env-specific
    proprio feature dimension."""
    from experiments.policy_eval_env import make_eval_env, reset_env

    envs = [make_eval_env(task, action_repeat=action_repeat, seed=seed + i,
                          max_step_limit=max_step_limit, render=False, config=getattr(agent, 'agent', agent).config)
            for i in range(num_envs)]
    action_key = next(k for k in envs[0].act_space if k != 'reset')
    proprio_keys = _proprio_keys(envs[0])

    def _step_env(env, action):
        return env.step({action_key: action.astype(np.float32),
                         'reset': np.bool_(False)})

    trans = [reset_env(env) for env in envs]
    carry = agent.init_policy(batch_size=num_envs)
    visited = []
    t0 = time.time()
    executor = ThreadPoolExecutor(max_workers=num_envs)
    try:
        for t in range(num_steps):
            obs_batch = {k: np.stack([trans[i][k] for i in range(num_envs)])
                         for k in trans[0]}
            acts, _, carry = agent.policy(obs_batch, carry, mode='explore')
            actions = np.asarray(acts[action_key])
            # Reset finished envs first (sequential, cheap), then step the rest in parallel.
            active = []
            for i in range(num_envs):
                if trans[i]['is_last']:
                    trans[i] = reset_env(envs[i])
                else:
                    active.append(i)
            futures = {
                i: executor.submit(_step_env, envs[i], actions[i])
                for i in active
            }
            for i, fut in futures.items():
                trans[i] = fut.result()
                visited.append(_stack_proprio(trans[i], proprio_keys))
            if t == 0 or (t + 1) % 500 == 0 or t + 1 == num_steps:
                print(f'[policy_eval][visit] step {t + 1}/{num_steps} '
                      f'collected={len(visited)} elapsed={time.time() - t0:.0f}s')
    finally:
        executor.shutdown(wait=True)
    for env in envs:
        env.close()
    return np.stack(visited), proprio_keys


def _compute_anchor_distances(task, action_repeat, eval_rows, visited, proprio_keys, seed=0, config=None):
    """Per-row min normalized Euclidean distance from each anchor row's
    DMC-encoded proprio obs to any visited proprio obs. Returns
    (distances [N], stats dict)."""
    from experiments.policy_eval_env import make_eval_env, obs_at

    # Build the same proprio vector for each anchor row by setting physics and
    # reading the env's obs (cartpole encodes theta as sin/cos, walker uses
    # orientations + height, etc. — `obs_at` handles each task's encoding).
    env = make_eval_env(task, action_repeat=action_repeat, seed=seed, render=False, config=config)
    anchor_vecs = []
    for row in eval_rows:
        obs = obs_at(env, row['qpos'], row['qvel'],
                     natural_reset=row.get('natural_reset', False))
        anchor_vecs.append(_stack_proprio(obs, proprio_keys))
    anchor_z_raw = np.stack(anchor_vecs).astype(np.float32)

    mean = visited.mean(axis=0)
    std = visited.std(axis=0).clip(min=1e-6)
    visit_z = ((visited - mean) / std).astype(np.float32)
    anchor_z = ((anchor_z_raw - mean) / std).astype(np.float32)

    dists = np.empty(anchor_z.shape[0], dtype=np.float32)
    for i in range(anchor_z.shape[0]):
        d = np.linalg.norm(visit_z - anchor_z[i], axis=1)
        dists[i] = d.min()
    env.close()
    return dists, dict(mean=mean, std=std, proprio_keys=np.array(proprio_keys))


# ---------------------- MC ground truth ----------------------

MC_MAX_PARALLEL = 64


def _make_mc_components_fn(agent, *, include_bootstrap=False):
    """Read current particles in RB target units with a fixed bank of z."""
    import jax
    import jax.numpy as jnp
    from dreamerv3 import jaxutils, ninjax as nj

    inner = getattr(agent, 'agent', agent)
    if inner.ac_inputs != 'obs' or inner.visual_prior:
        raise ValueError('MC component logging requires state-observation inputs.')
    q = inner.q
    samples = q.epistemic_samples if q.epistemic_dim else 1
    contexts = np.random.default_rng(0).uniform(
        -q.epistemic_std, q.epistemic_std, (samples, q.epistemic_dim)).astype(np.float32)

    def read(obs, actions, contexts):
        obs = inner.preprocess(obs)
        actions = jaxutils.onehot_dict(actions, inner.act_space)
        inputs = inner._q_input(inner._augment_ac_inputs({}, obs, headed=False), actions)
        B = next(iter(obs.values())).shape[0]
        result = {}
        for i, head in enumerate(q.heads):
            parts = []
            for context in contexts:
                epistemic = jnp.broadcast_to(context, (B, q.epistemic_dim)) if q.epistemic_dim else None
                components = head.component_means(
                    inputs, bdims=1, epistemic=epistemic,
                    prior_scale=1.0, corrector_scale=1.0)
                keys = ('prior', 'corrector', 'residual_bootstrap') if include_bootstrap else ('prior', 'corrector')
                parts.append({k: inner._select_action_values(components[k], actions)
                              for k in keys})
            prefix = '' if i == 0 else f'q{i + 1}/'
            for key in keys:
                result[prefix + key] = jnp.concatenate(
                    [p[key] for p in parts], axis=0).T.astype(jnp.float32)  # [B, P]
        return result

    pure = nj.pure(read)
    jitted = jax.jit(lambda params, obs, actions, contexts, seed:
                    pure(params, obs, actions, contexts, seed=seed)[1])
    contexts_device, seed = jax.device_put((contexts, np.array([0, 0], np.uint32)))

    def call(obs, actions):
        obs, actions = jax.device_put((obs, actions))
        return jax.device_get(jitted(agent.params, obs, actions, contexts_device, seed))

    metadata = {
        'epistemic_contexts': contexts,
        'particle_context': np.repeat(np.arange(samples), q.num_heads),
        'particle_ensemble': np.tile(np.arange(q.num_heads), samples),
        'component_units': np.array('unit_scale'),
        'component_step': np.int64(0),
    }
    for i, head in enumerate(q.heads):
        prefix = '' if i == 0 else f'q{i + 1}/'
        metadata[prefix + 'prior_scale'] = np.float32(head.prior_scale)
        metadata[prefix + 'corrector_scale'] = np.float32(head.corrector_scale)
    return call, metadata


def _mc_trajectory_components(trajectory, action_key, component_fn, batch_size=1024):
    """Bound forward-pass memory and preserve [episode, time, particle] axes."""
    shape = trajectory['valid'].shape
    ids = np.flatnonzero(trajectory['valid'])
    obs = {k[4:]: v.reshape((-1,) + v.shape[2:])
           for k, v in trajectory.items() if k.startswith('obs/')}
    actions = trajectory['action'].reshape((-1,) + trajectory['action'].shape[2:])
    result = {}
    for start in range(0, len(ids), batch_size):
        selected = ids[start:start + batch_size]
        # Pad the final batch to avoid recompiling for every episode length.
        padded = np.pad(selected, (0, batch_size - len(selected)), mode='edge')
        values = component_fn({k: v[padded] for k, v in obs.items()},
                              {action_key: actions[padded]})
        for key, value in values.items():
            if key not in result:
                result[key] = np.full(shape + (value.shape[-1],), np.nan, np.float32)
            result[key].reshape((-1, value.shape[-1]))[selected] = value[:len(selected)]
    return result


def _mc_parallel_chunk(envs, qpos, qvel, init_action, agent, action_key,
                       discount, max_steps, natural_reset=False,
                       record_video=False, video_size=64, record_trajectory=False):
    """Run len(envs) MC rollouts in parallel, all starting from (qpos, qvel)
    and applying init_action at t=0. Subsequent actions come from a batched
    agent.policy call. Returns (returns[N], frames_or_None, trajectory_or_None).
    A trajectory row pairs obs/action at t with the resulting reward at t+1.
    `valid` excludes padding after episode end; max_steps includes init_action.

    If `record_video` is True, env-0's frames are rendered each step via
    dm_control physics and returned as a [T, H, W, 3] uint8 array.

    If `natural_reset` is True, dm_control's default init drives the start
    state (qpos / qvel are ignored)."""
    from experiments.policy_eval_env import obs_at, physics_state, render_frame

    N = len(envs)
    if max_steps < 1:
        raise ValueError('MC max_steps must include at least the initial action.')
    trans = [obs_at(env, qpos, qvel, natural_reset=natural_reset) for env in envs]
    obs_keys = [k for k in trans[0] if not k.startswith('log_') and k != 'image']
    actions = np.broadcast_to(np.asarray(init_action, dtype=np.float32),
                              (N,) + np.asarray(init_action).shape).copy()

    def _step_env(env, action):
        return env.step({action_key: action.astype(np.float32),
                         'reset': np.bool_(False)})

    returns = np.zeros(N, np.float64)
    done = np.zeros(N, bool)
    frames, history = [], defaultdict(list)
    carry = agent.init_policy(batch_size=N)
    executor = ThreadPoolExecutor(max_workers=N)
    try:
        for t in range(max_steps):
            if done.all():
                break
            obs_batch = {k: np.stack([tran[k] for tran in trans]) for k in obs_keys}
            # mode='explore' is the only mode that dispatches to self.exp_actor —
            # which is where the trained policy weights were loaded.
            if t:
                acts, _, carry = agent.policy(obs_batch, carry, mode='explore')
                actions = np.asarray(acts[action_key])
            valid = ~done
            if record_trajectory:
                for k, v in obs_batch.items():
                    history[f'obs/{k}'].append(v)
                history['action'].append(actions.copy())
                states = [physics_state(env) for env in envs]
                history['qpos'].append(np.stack([x[0] for x in states]))
                history['qvel'].append(np.stack([x[1] for x in states]))
                history['valid'].append(valid.copy())
            active = np.flatnonzero(valid)
            futures = {
                i: executor.submit(_step_env, envs[i], actions[i])
                for i in active
            }
            rewards = np.zeros(N, np.float32)
            last, terminal = np.zeros(N, bool), np.zeros(N, bool)
            for i, fut in futures.items():
                trans[i] = fut.result()
                rewards[i] = trans[i]['reward']
                last[i], terminal[i] = trans[i]['is_last'], trans[i]['is_terminal']
                done[i] = last[i]
            returns += float(discount) ** t * rewards
            if record_trajectory:
                history['reward'].append(rewards)
                history['is_last'].append(last)
                history['is_terminal'].append(terminal)
            if record_video and valid[0]:
                frames.append(render_frame(envs[0], video_size))
    finally:
        executor.shutdown(wait=True)

    video = np.stack(frames, axis=0).astype(np.uint8) if frames else None
    trajectory = None
    if record_trajectory:
        trajectory = {k: np.stack(v, axis=1) for k, v in history.items()}  # [N, T, ...]
        trajectory['length'] = trajectory['valid'].sum(1).astype(np.int32)
        # A time limit or MC cap leaves an unobserved continuation, unlike a terminal.
        trajectory['truncated'] = ~trajectory['is_terminal'].any(1)
    return returns.astype(np.float32), video, trajectory


def _compute_mc(eval_rows, task, action_repeat, discount, num_episodes,
                max_steps, agent, seed=0, logger=None, video_size=0,
                max_step_limit=0, rollout_dir=None):
    """Parallel MC. Builds N = min(num_episodes, MC_MAX_PARALLEL) envs and
    batches the policy call across them. If num_episodes > N, runs ceil()
    chunks of at most N episodes.

    With rollout_dir, saves aligned per-step rewards and source states/actions
    once per chunk. Frozen-corrector runs also save unit-scale prior/corrector
    particles, with the same ensemble/z ordering across all times and episodes.

    If `logger` is provided and `video_size > 0`, env-0's frames from the
    first chunk of each anchor-row rollout are rendered and logged as
    `policy_eval/mc_video/<anchor_name>`.

    `max_step_limit` (env frames) overrides dm_control's episode length for
    the MC envs only; 0 keeps the default. Use this together with a large
    `max_steps` to make MC approximate the infinite-horizon V^π."""
    from experiments.policy_eval_env import make_eval_env

    component_fn, component_meta = None, {}
    if rollout_dir is not None:
        rollout_dir = embodied.Path(rollout_dir)
        rollout_dir.mkdir()
        if getattr(getattr(agent, 'agent', agent).config, 'freeze_corrector', False):
            component_fn, component_meta = _make_mc_components_fn(agent)
        print('[policy_eval][MC] saving reward trajectories' +
              (' and frozen prior/corrector particles' if component_fn is not None else ''))
    N = min(int(num_episodes), MC_MAX_PARALLEL)
    print(f'[policy_eval][MC] building {N} parallel envs (num_episodes={num_episodes}, '
          f'max_step_limit={max_step_limit})')
    # render=False turns off the per-step EGL render in DMC.step. We still get
    # videos because _mc_parallel_chunk renders env-0 directly via
    # `envs[0]._dmenv.physics.render(...)`, which is single-threaded and
    # doesn't fight with the parallel env steps for the EGL context.
    envs = [make_eval_env(task, action_repeat=action_repeat, seed=seed + i,
                          max_step_limit=max_step_limit,
                          render=False, config=getattr(agent, 'agent', agent).config)
            for i in range(N)]
    action_key = next(k for k in envs[0].act_space if k != 'reset')

    n_chunks = (int(num_episodes) + N - 1) // N

    means = np.zeros(len(eval_rows), dtype=np.float32)
    stds = np.zeros(len(eval_rows), dtype=np.float32)
    raw = np.zeros((len(eval_rows), int(num_episodes)), dtype=np.float32)
    log_video = logger is not None and int(video_size) > 0
    t0 = time.time()
    try:
        for i, row in enumerate(eval_rows):
            # One video per anchor: render env-0 of chunk-0 on anchor rows only.
            record = log_video and row['kind'] == 'anchor'
            for c in range(n_chunks):
                count = min(N, int(num_episodes) - c * N)
                chunk_returns, chunk_video, trajectory = _mc_parallel_chunk(
                    envs[:count], row['qpos'], row['qvel'], row['action'], agent,
                    action_key, discount, max_steps,
                    natural_reset=row.get('natural_reset', False),
                    record_video=(record and c == 0),
                    video_size=int(video_size), record_trajectory=rollout_dir is not None)
                raw[i, c * N:c * N + count] = chunk_returns
                if trajectory is not None:
                    components = (_mc_trajectory_components(trajectory, action_key, component_fn)
                                  if component_fn is not None else {})
                    np.savez_compressed(
                        rollout_dir / f'point_{i:04d}_chunk_{c:04d}.npz',
                        **trajectory, **components, **component_meta,
                        eval_point=np.int32(i), episode=np.arange(c * N, c * N + count),
                        anchor_idx=np.int32(row['anchor_idx']), anchor_name=row['anchor_name'],
                        anchor_tag=row['anchor_tag'], kind=row['kind'], sweep_idx=row['sweep_idx'],
                        natural_reset=row.get('natural_reset', False), action_key=action_key,
                        discount=np.float64(discount),
                        discount_weight=float(discount) ** np.arange(trajectory['reward'].shape[1]),
                        action_repeat=np.int32(action_repeat), max_steps=np.int32(max_steps),
                        returns=chunk_returns)
                if chunk_video is not None:
                    logger.add({f'policy_eval/mc_video/{row["anchor_name"]}': chunk_video})
            means[i] = raw[i].mean()
            stds[i] = raw[i].std()
            if i == 0 or (i + 1) % 5 == 0 or i + 1 == len(eval_rows):
                dt = time.time() - t0
                print(f'[policy_eval][MC] point {i + 1}/{len(eval_rows)} '
                      f'({row["anchor_name"]}/{row["kind"]}/{row["sweep_idx"]}) '
                      f'mean={means[i]:.3f} std={stds[i]:.3f} elapsed={dt:.0f}s')
    finally:
        for env in envs:
            env.close()
    if log_video:
        logger.write()
    for env in envs:
        env.close()
    return means, stds, raw


# ---------------------- TD readout ----------------------

def _make_q_values_fn(wrapped_agent):
    """Build a JIT'd callable returning [B, P] particles and [B] value scores
    using the inner ObserverAgent.q_values method. Bypasses jaxagent (which only
    wraps init_policy/policy/train/report) without modifying it."""
    import jax
    from dreamerv3 import ninjax as nj

    inner = getattr(wrapped_agent, 'agent', wrapped_agent)
    if not hasattr(inner, 'q_values'):
        return None

    def fn(params, obs, actions, seed):
        pure = nj.pure(inner.q_values)
        _, out = pure(params, obs, actions, seed=seed)
        return out

    jitted = jax.jit(fn)
    # transfer_guard='disallow' blocks the implicit host→device transfer that
    # `jax.random.PRNGKey(int)` does; mirror jaxagent._next_seeds and build a
    # uint32[2] seed on the host, then explicitly device_put it.
    rng = np.random.default_rng(0)

    def call(obs_dict, action_dict):
        seed_np = rng.integers(
            0, np.iinfo(np.uint32).max, size=(2,), dtype=np.uint32)
        # All host→device transfers must be explicit (transfer_guard='disallow').
        seed, obs_dict, action_dict = jax.device_put(
            (seed_np, obs_dict, action_dict))
        params = wrapped_agent.params
        result = jitted(params, obs_dict, action_dict, seed)
        return {k: np.asarray(v, dtype=np.float32)
                for k, v in jax.device_get(result).items()}

    return call


def _evaluate_q(agent, eval_rows, task, action_repeat, seed):
    """Query the agent's Q ensemble at every eval point. Returns
    dict[str, [num_points, P]] with keys {'q', 'raw', 'prior', 'corrector'}
    and optionally 'residual_bootstrap'. P = num_q_heads * (epistemic_samples
    or 1), plus [num_points] base_value, optimism_bonus, and optimistic_value."""
    from experiments.policy_eval_env import make_eval_env, obs_at
    env = make_eval_env(task, action_repeat=action_repeat, seed=seed, render=False, config=getattr(agent, 'agent', agent).config)
    action_key = next(k for k in env.act_space if k != 'reset')

    # `env.obs_space` always advertises `log_image` (or `image`), but those are
    # only filled inside env.step()'s render call — obs_at skips rendering.
    # Filter to keys that obs_at actually returns and drop control-flow keys.
    obs_sample = obs_at(env, eval_rows[0]['qpos'], eval_rows[0]['qvel'])
    obs_keys = [k for k in env.obs_space
                if k in obs_sample
                and k not in ('reward', 'is_first', 'is_last', 'is_terminal')]

    obs_batch = defaultdict(list)
    act_batch = []
    for row in eval_rows:
        obs = obs_at(env, row['qpos'], row['qvel'],
                     natural_reset=row.get('natural_reset', False))
        for k in obs_keys:
            obs_batch[k].append(np.asarray(obs[k], dtype=np.float32))
        act_batch.append(np.asarray(row['action'], dtype=np.float32))
    obs_batch = {k: np.stack(v, axis=0) for k, v in obs_batch.items()}
    act_batch = {action_key: np.stack(act_batch, axis=0)}

    q_fn = getattr(agent, '_pe_q_values_fn', None)
    if q_fn is None:
        q_fn = _make_q_values_fn(agent)
        agent._pe_q_values_fn = q_fn
    if q_fn is None:
        print('[policy_eval][TD] WARNING: ObserverAgent.q_values not found; '
              'returning NaNs.')
        return {'q': np.full((len(eval_rows), 1), np.nan, dtype=np.float32)}
    env.close()
    return q_fn(obs_batch, act_batch)


# ---------------------- main run entry ----------------------

def _resolve_action_repeat(args):
    """Read action_repeat from the flat `args.action_repeat` slot that
    exp.py copies out of config.env.dmc.repeat. Falls back to 1 if missing
    (e.g. a non-DMC task or older exp.py without the explicit copy)."""
    try:
        return int(args.action_repeat)
    except (AttributeError, KeyError, TypeError):
        try:
            # Older fallback: in case someone wires it through env.dmc.repeat
            # directly on the args object.
            return int(args.env.dmc.repeat)
        except (AttributeError, KeyError, TypeError):
            return 1


def policy_eval(make_agent, make_replay, make_env, make_logger, args, make_reporter=None):
    """`args` is an embodied.Config exposing standard `run.*` flags plus:
      - from_checkpoint: path to .ckpt
      - task:            'dmc_<domain>_<task>' string (from config.task)
      - policy_eval_mc_episodes: int  (per anchor / sweep point)
      - policy_eval_mc_max_steps: int
      - policy_eval_every: int        (TD readout cadence in env steps)
      - policy_eval_discount: float
    """
    assert args.from_checkpoint, 'policy_eval requires --from_checkpoint'

    from experiments.eval_anchors import get_anchors

    logdir = embodied.Path(args.logdir)
    out_dir = logdir / 'policy_eval'
    out_dir.mkdir()

    if args.task.startswith('mjp_') and not getattr(args, 'use_jax_driver', False):
        raise ValueError('MJP policy evaluation requires use_jax_driver=1.')
    print('[policy_eval] building agent + env')
    print(f'[policy_eval] resolved action_repeat = {_resolve_action_repeat(args)}')
    agent = make_agent()
    env_ctor_idx = lambda i: make_env(i)
    logger = make_logger()
    if make_reporter is None:
        from devote.report import Reporter as make_reporter
    reporter = make_reporter(agent, logger, args, policy_eval=True)

    # Restore only the treatment's requested components; RB/optimizers stay fresh.
    from .policy_eval_loading import load_policy_eval
    manifest = load_policy_eval(agent, args.from_checkpoint,
                                args.policy_eval_corrector,
                                args.policy_eval_checkpoint_config)
    (Path(str(out_dir)) / 'initialization.json').write_text(json.dumps(manifest, indent=2))
    diagnostic_every = int(args.policy_eval_diagnostics_every)
    if diagnostic_every > 0 and (
            int(args.policy_eval_mc_episodes) < 2 or int(args.policy_eval_visit_steps) <= 0):
        raise ValueError('MC diagnostics need at least two MC episodes and positive visitation steps.')
    if diagnostic_every > 0:
        horizons = [int(h) for h in args.policy_eval_diagnostics_horizons.split(',')]
        if min(horizons) < 1 or max(horizons) >= int(args.policy_eval_mc_max_steps):
            raise ValueError('Diagnostic horizons must be positive and shorter than the MC step cap.')
    # Startup evaluation must not advance the training policy's random stream.
    training_rng = agent.rng
    agent.rng = np.random.default_rng(int(args.seed) + 1701)

    # ---- 2. Resolve anchors / eval grid ----
    task = args.task
    eval_rows = _flatten_eval_points(get_anchors(task))
    anchor_meta = dict(
        anchor_idx=np.array([r['anchor_idx'] for r in eval_rows]),
        kind=np.array([r['kind'] for r in eval_rows]),
        sweep_idx=np.array([r['sweep_idx'] for r in eval_rows]),
        anchor_name=np.array([r['anchor_name'] for r in eval_rows]),
        anchor_tag=np.array([r['anchor_tag'] for r in eval_rows]),
        natural_reset=np.array([r['natural_reset'] for r in eval_rows]),
        qpos=np.stack([r['qpos'] for r in eval_rows]),
        qvel=np.stack([r['qvel'] for r in eval_rows]),
        action=np.stack([r['action'] for r in eval_rows]),
    )
    np.savez(out_dir / 'anchors.npz', **anchor_meta)

    # ---- 3. MC ground truth (once, expensive) ----
    # Set policy_eval_mc_episodes=0 to skip MC for testing — Q-loop still runs;
    # mse_vs_mc / corr_vs_mc metrics will be NaN.
    if int(args.policy_eval_mc_episodes) > 0:
        # Probe: call agent.policy on a deterministic obs and print action stats.
        # If the action is near-uniform on [-1, 1] we have a random actor; trained
        # policy actions should cluster in a narrower band.
        from experiments.policy_eval_env import make_eval_env, obs_at
        probe_env = make_eval_env(task, action_repeat=_resolve_action_repeat(args), seed=0, render=False, config=getattr(agent, 'agent', agent).config)
        probe_obs = obs_at(probe_env, eval_rows[0]['qpos'], eval_rows[0]['qvel'])
        probe_obs_batch = {k: np.asarray(v)[None] for k, v in probe_obs.items()}
        probe_carry = agent.init_policy(batch_size=1)
        probe_acts_explore, _, _ = agent.policy(probe_obs_batch, probe_carry, mode='explore')
        probe_acts_eval,    _, _ = agent.policy(probe_obs_batch, probe_carry, mode='eval')
        probe_env.close()
        a_explore = np.asarray(probe_acts_explore['action'][0])
        a_eval    = np.asarray(probe_acts_eval['action'][0])
        print(f'[policy_eval] probe action (explore→exp_actor): '
              f'mean={a_explore.mean():.3f} std={a_explore.std():.3f} '
              f'abs_mean={np.abs(a_explore).mean():.3f}  vals={a_explore.round(3).tolist()}')
        print(f'[policy_eval] probe action (eval→actor random): '
              f'mean={a_eval.mean():.3f} std={a_eval.std():.3f} '
              f'abs_mean={np.abs(a_eval).mean():.3f}     vals={a_eval.round(3).tolist()}')

        print(f'[policy_eval] computing MC over {len(eval_rows)} points '
              f'× {args.policy_eval_mc_episodes} episodes')
        # Use the agent's own self.discount so MC return is comparable with the
        # logged training returns. Falls back to args.policy_eval_discount if the
        # attribute isn't reachable (e.g. wrapped agent without inner attr).
        inner = getattr(agent, 'agent', agent)
        agent_discount = getattr(inner, 'discount', None)
        discount = float(agent_discount) if agent_discount is not None else float(args.policy_eval_discount)
        print(f'[policy_eval] MC discount = {discount:.6f} '
              f'(agent.discount={agent_discount}, fallback={args.policy_eval_discount})')

        mc_mean, mc_std, mc_raw = _compute_mc(
            eval_rows, task,
            action_repeat=_resolve_action_repeat(args),
            discount=discount,
            num_episodes=int(args.policy_eval_mc_episodes),
            max_steps=int(args.policy_eval_mc_max_steps),
            agent=agent,
            seed=int(args.seed),
            logger=logger,
            video_size=int(getattr(args, 'policy_eval_video_size', 0)),
            max_step_limit=int(getattr(args, 'policy_eval_mc_max_step_limit', 0)),
            rollout_dir=out_dir / 'mc_rollouts',
        )
        np.savez(out_dir / 'mc.npz', mean=mc_mean, std=mc_std, returns=mc_raw,
                 discount=np.float64(discount))
    else:
        print('[policy_eval] skipping MC (policy_eval_mc_episodes=0)')
        n = len(eval_rows)
        mc_mean = np.full(n, np.nan, np.float32)
        mc_std  = np.full(n, np.nan, np.float32)
        mc_raw  = np.full((n, 0), np.nan, np.float32)

    # ---- 3b. Per-anchor distance to on-policy visitation (ID vs OOD check) ----
    visited, proprio_keys = None, None
    n_visit = int(getattr(args, 'policy_eval_visit_steps', 0))
    if n_visit > 0:
        print(f'[policy_eval] collecting {n_visit} agent steps of on-policy visitation')
        visited, proprio_keys = _collect_policy_visitation(
            task, _resolve_action_repeat(args), agent,
            num_steps=n_visit, num_envs=8,
            seed=int(getattr(args, 'seed', 0)),
            max_step_limit=int(getattr(args, 'policy_eval_mc_max_step_limit', 0)),
        )
        dists, norm_stats = _compute_anchor_distances(
            task, _resolve_action_repeat(args), eval_rows, visited, proprio_keys,
            seed=int(getattr(args, 'seed', 0)), config=getattr(agent, 'agent', agent).config)
        np.savez(out_dir / 'anchor_distances.npz',
                 distances=dists, **norm_stats, visited=visited)
        # Print sorted summary for anchor rows only.
        print('[policy_eval] anchor distances (lower = closer to on-policy visitation):')
        anchor_mask = anchor_meta['kind'] == 'anchor'
        order = np.argsort(dists[anchor_mask])
        anchor_names = anchor_meta['anchor_name'][anchor_mask]
        anchor_dists = dists[anchor_mask]
        for j in order:
            print(f'  {str(anchor_names[j]):>26s}  min_dist = {float(anchor_dists[j]):8.3f}')
    else:
        print('[policy_eval] skipping anchor-distance step (policy_eval_visit_steps=0)')

    # ---- 4. Q-only training loop with periodic TD readout ----
    # Mirrors embodied.run.train_eval: use agent.dataset(replay.dataset(...))
    # rather than the non-existent replay.sample(). Training cadence is driven
    # by the standard train_ratio Ratio scheduler.
    agent.rng = training_rng
    print('[policy_eval] starting Q-only training loop')
    replay = make_replay()
    train_env = None
    if getattr(args, 'use_jax_driver', False):
        train_env = make_env(0, num_envs=args.num_envs)
        driver = embodied.JaxDriver(train_env)
    else:
        fns = [bind(env_ctor_idx, i) for i in range(args.num_envs)]
        driver = embodied.Driver(fns, args.driver_parallel)

    step = logger.step
    policy_fps = embodied.FPS()
    train_fps = embodied.FPS()
    batch_steps = args.batch_size * (args.batch_length - args.replay_context)
    should_train = embodied.when.Ratio(args.train_ratio / batch_steps)
    should_eval = embodied.when.Every(int(args.policy_eval_every))
    should_log = embodied.when.Clock(args.log_every)

    dataset_train = agent.dataset(bind(
        replay.dataset, args.batch_size, args.batch_length, 'ac'))
    reporter.init_replays(train=replay)
    carry = [agent.init_train(args.batch_size)]
    reporter.reset_carry()
    sync_agent_step = lambda: agent.set_global_step(step)

    driver.on_step(lambda tran, _: step.increment())
    driver.on_step(lambda tran, _: policy_fps.step())

    reporter.init_coverage(env_ctor_idx, env=train_env)
    reporter.attach_coverage(driver, count=True)
    # Annotate bins before replay insertion so report sequences retain them.
    driver.on_step(replay.add)

    reporter.init_policy_episodes()
    driver.on_step(reporter.policy_episode)

    train_updates = [0]

    def train_step(tran, worker):
        if len(replay) < args.batch_size or step < args.train_fill:
            return
        repeats = should_train(step)
        if repeats or worker == args.num_envs - 1:
            sync_agent_step()
        for _ in range(repeats):
            batch = next(dataset_train)
            # Q-only update: loss scales for actor/exp_actor/alpha are zeroed.
            outs, carry[0], mets = agent.train(batch, carry[0])
            train_updates[0] += 1
            train_fps.step(batch_steps)
            if 'replay' in outs:
                replay.update(outs['replay'])
            reporter.add_train_metrics(mets)
    driver.on_step(train_step)

    # Q-training driver: data collection runs the trained exp_actor.
    policy = lambda *a: agent.policy(*a, mode='explore')
    driver.reset(agent.init_policy)
    sync_agent_step()

    diagnostics = None
    if diagnostic_every > 0:
        from .policy_eval_diagnostics import PolicyEvalDiagnostics
        coverage = reporter.coverage
        if coverage is None:
            raise ValueError('MC novelty diagnostics require coverage geometry and log_coverage=1.')
        # Reuse precisely the training geometry, including maze sub-cell bins.
        geometry = dict(bounds=coverage.bounds, bins=coverage.bins,
                        axis_names=coverage.axis_names, valid_mask=coverage.valid_mask,
                        project=reporter.project)
        component_reader, _ = _make_mc_components_fn(agent)
        start_reader, _ = _make_mc_components_fn(agent, include_bootstrap=True)
        diagnostics = PolicyEvalDiagnostics(
            agent, args, eval_rows, out_dir, visited, proprio_keys, geometry,
            component_reader, start_reader, _mc_trajectory_components)
        td = _evaluate_q(agent, eval_rows, task,
                         action_repeat=_resolve_action_repeat(args), seed=int(args.seed))
        np.savez(out_dir / f'td_{int(step):08d}.npz', step=int(step), **td)
        logger.add({'policy_eval/mc/train_updates': 0})
        diagnostics.report(int(step), td, reporter.coverage, logger)
    next_diagnostic = diagnostic_every

    while step < args.steps:
        driver(policy, steps=10)
        evaluate = should_eval(step)
        diagnose = diagnostics is not None and int(step) >= next_diagnostic
        if evaluate or diagnose:
            sync_agent_step()
            td = _evaluate_q(agent, eval_rows, task,
                             action_repeat=_resolve_action_repeat(args),
                             seed=int(args.seed) if hasattr(args, 'seed') else 0)
            np.savez(out_dir / f'td_{int(step):08d}.npz',
                     step=int(step), **td)
            reporter.policy_values(td, mc_mean, anchor_meta)
            if evaluate:
                reporter.report_replay('train')
            if diagnose:
                logger.add({'policy_eval/mc/train_updates': train_updates[0]})
                diagnostics.report(int(step), td, reporter.coverage, logger)
                next_diagnostic = (int(step) // diagnostic_every + 1) * diagnostic_every
        if should_log(step):
            logger.add({'fps/policy': policy_fps.result()})
            logger.add({'fps/train': train_fps.result()})
            logger.add(replay.stats(), prefix='replay')
            reporter.log_coverage()
            reporter.flush_policy_episodes()
            logger.write()

    driver.close()
    logger.close()
    print('[policy_eval] done. outputs:', out_dir)
