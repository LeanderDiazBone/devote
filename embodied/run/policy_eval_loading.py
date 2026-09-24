"""Selective initialization for the three policy-evaluation treatments."""

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


MODES = ('load_freeze', 'load_train', 'fresh_train')


def merge_components(live, saved, components, targets):
    """Copy complete named subtrees, then synchronize online/target subtrees.

    Component pairs are (live prefix, checkpoint prefix). Nothing else is
    restored: in particular RB, raw Q and optimizer state remain fresh.
    """
    merged, counts = dict(live), {}
    for destination, source in components:
        dst = {k[len(destination):]: k for k in live if k.startswith(destination + '/')}
        src = {k[len(source):]: k for k in saved if k.startswith(source + '/')}
        if not dst or dst.keys() != src.keys():
            raise ValueError(f'Incomplete checkpoint component {source} -> {destination}: '
                             f'missing={sorted(dst.keys() - src.keys())}, '
                             f'extra={sorted(src.keys() - dst.keys())}')
        for suffix, key in dst.items():
            value = saved[src[suffix]]
            if np.shape(value) != np.shape(live[key]):
                raise ValueError(f'Checkpoint shape mismatch for {key}: '
                                 f'{np.shape(value)} != {np.shape(live[key])}')
            merged[key] = value
        counts[destination] = len(dst)
    for online, target in targets:
        dst = {k[len(target):]: k for k in live if k.startswith(target + '/')}
        src = {k[len(online):]: k for k in merged if k.startswith(online + '/')}
        if not dst or dst.keys() != src.keys():
            raise ValueError(f'Online/target parameter trees differ: {online}, {target}')
        for suffix, key in dst.items():
            if np.shape(live[key]) != np.shape(merged[src[suffix]]):
                raise ValueError(f'Online/target shape mismatch: {key}')
            merged[key] = merged[src[suffix]]
    return merged, counts


def load_policy_eval(agent, checkpoint, mode, config_path=''):
    import embodied
    import jax

    if mode not in MODES:
        raise ValueError(f'Unknown corrector mode: {mode}')
    inner = getattr(agent, 'agent', agent)
    if inner.ac_inputs != 'obs' or inner.visual_prior:
        raise ValueError('Policy-eval component loading currently requires proprio ac_inputs=obs.')
    if len(inner.q.heads) != 1:
        raise ValueError('Matched MC diagnostics currently require single-Q pessimism=-1.')
    if inner.policy_mode != 'exp_actor' or inner.config.freeze_corrector != (mode == 'load_freeze'):
        raise ValueError('Policy-eval mode and agent freeze/policy configuration disagree.')
    if any(float(inner.config.loss_scales[k]) != 0 for k in ('actor', 'exp_actor')):
        raise ValueError('Policy evaluation requires frozen actor and exp_actor losses.')
    if any(inner.config[k] != 'exp_actor' for k in ('pc_target_actor', 'td_target_actor')):
        raise ValueError('Both residual and Q targets must use the loaded exp_actor.')
    path = Path(checkpoint)
    if path.is_dir():
        path = path / 'latest.ckpt'
    candidates = [Path(config_path)] if config_path else [
        path.parent / 'config.yaml', path.parent.parent / 'config.yaml']
    source_path = next((p for p in candidates if p.exists()), None)
    if source_path is None:
        raise ValueError('Checkpoint config.yaml is required to verify prior/corrector '
                         'compatibility; pass --policy_eval_checkpoint_config.')
    source = embodied.Config.load(str(source_path))
    current = inner.config
    for key in ('task', 'ac_inputs', 'visual_prior', 'actor_dist_cont', 'actor_dist_disc'):
        if source.get(key) != current.get(key):
            raise ValueError(f'Checkpoint config mismatch for {key}: '
                             f'{source.get(key)} != {current.get(key)}')
    # These options can change a forward pass without changing parameter shapes.
    old_prior = inner._resolved_prior_config(source.critic, source.critic_prior)
    new_prior = inner._resolved_prior_config(current.critic, current.critic_prior)
    for key in ('use_rff', 'length_scale', 'action_independent', 'act', 'outact',
                'norm', 'epistemic', 'epistemic_dim', 'epistemic_std'):
        # RFF priors do not use the configured hidden activation.
        if key == 'act' and old_prior.get('use_rff') and new_prior.get('use_rff'):
            continue
        # Disabled epistemic conditioning ignores its dimension and scale.
        if (key in ('epistemic_dim', 'epistemic_std')
                and not old_prior.get('epistemic') and not new_prior.get('epistemic')):
            continue
        if old_prior.get(key) != new_prior.get(key):
            raise ValueError(f'Checkpoint prior mismatch for {key}: '
                             f'{old_prior.get(key)} != {new_prior.get(key)}')
    for key in ('act', 'norm', 'minstd', 'maxstd'):
        if source.actor.get(key) != current.actor.get(key):
            raise ValueError(f'Checkpoint actor mismatch for {key}')
    with path.open('rb') as file:
        saved = pickle.load(file)['agent']
    live = jax.device_get(agent.params)
    actor = inner.exp_actor.path
    policy_mode = source.get('policy_mode', 'actor')
    if policy_mode not in ('actor', 'exp_actor'):
        raise ValueError(f'Unsupported checkpoint collection policy: {policy_mode}')
    source_actor = 'exp_actor_sac' if policy_mode == 'exp_actor' else 'actor_sac'
    components = [(actor, actor.replace('/exp_actor_sac', '/' + source_actor)),
                  (inner.q_prior.path, inner.q_prior.path)]
    for head in inner.q.heads:
        if head.prior_corrector is None or head.residual_bootstrap is None:
            raise ValueError('Policy evaluation requires corrector and RB networks.')
        if mode != 'fresh_train':
            components.append((head.prior_corrector.path, head.prior_corrector.path))
    targets = [(q.path, qt.path) for q, qt in zip(inner.q.heads, inner.q_target.heads)]
    merged, counts = merge_components(live, saved, components, targets)
    agent.load(merged)
    digest = hashlib.sha256()
    for dst, _ in components:
        for key in sorted(k for k in merged if k.startswith(dst + '/')):
            digest.update(key.encode())
            digest.update(np.asarray(merged[key]).tobytes())
    manifest = dict(corrector_mode=mode, checkpoint=str(path.resolve()),
                    policy_source=source_actor,
                    checkpoint_config=str(source_path.resolve()),
                    loaded_parameters=counts, loaded_sha256=digest.hexdigest(),
                    rb_initialization='fresh', optimizer_initialization='fresh',
                    targets='copied_from_online')
    print('[policy_eval] initialization:', json.dumps(manifest, indent=2))
    return manifest
