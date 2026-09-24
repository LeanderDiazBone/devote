"""Host-side MC panel lifecycle for frozen-policy evaluation."""

from pathlib import Path

import numpy as np

from optimistic_curiosity.reporting.novelty import _average_ranks, StateDensityReference
from optimistic_curiosity.reporting.residual_mc import trajectory_returns, link_metrics


def panel_indices(rows, limit):
    candidates = [i for i, row in enumerate(rows) if not row['natural_reset']]
    if limit <= 0 or limit >= len(candidates):
        return candidates
    anchors = [i for i in candidates if rows[i]['kind'] == 'anchor'][:limit]
    rest = [i for i in candidates if i not in anchors]
    selected = np.linspace(0, len(rest) - 1, limit - len(anchors), dtype=int)
    return sorted(anchors + [rest[i] for i in selected])


def load_trajectory(paths):
    chunks = []
    for path in paths:
        with np.load(path) as z:
            keys = [k for k in z.files if k.startswith('obs/') or k in (
                'action', 'valid', 'is_last', 'is_terminal')]
            chunks.append({k: z[k] for k in keys})
    length = max(c['valid'].shape[1] for c in chunks)
    return {k: np.concatenate([
        np.pad(c[k], [(0, 0), (0, length - c[k].shape[1])] +
               [(0, 0)] * (c[k].ndim - 2)) for c in chunks], axis=0)
            for k in chunks[0]}


def novelty_percentiles(counts, valid_mask):
    result = np.full(counts.size, np.nan)
    mask = valid_mask.reshape(-1)
    result[mask] = (_average_ranks(1 / (counts.reshape(-1)[mask] + 1)) + .5) / mask.sum()
    return result


class PolicyEvalDiagnostics:
    """Evaluate current residual targets on an unchanged bank of trajectories."""

    def __init__(self, agent, args, rows, out_dir, visited, proprio_keys, geometry,
                 component_reader, start_reader, evaluate_components):
        import embodied

        self.agent, self.args = agent, args
        self.inner = getattr(agent, 'agent', agent)
        self.out_dir = Path(str(out_dir)) / 'mc_diagnostics'
        self.out_dir.mkdir(exist_ok=True)
        self.ids = panel_indices(rows, int(args.policy_eval_diagnostics_points))
        if not self.ids:
            raise ValueError('MC diagnostics require deterministic anchor/sweep points.')
        self.paths = [sorted((Path(str(out_dir)) / 'mc_rollouts').glob(f'point_{i:04d}_chunk_*.npz'))
                      for i in self.ids]
        if any(not paths for paths in self.paths):
            raise ValueError('MC diagnostics require saved rollouts for every panel point.')
        self.horizons = sorted(set(int(h) for h in args.policy_eval_diagnostics_horizons.split(',')))
        if not self.horizons or min(self.horizons) < 1:
            raise ValueError('Diagnostic horizons must be positive.')
        with np.load(self.paths[0][0]) as z:
            self.gamma = float(z['discount'])
            self.action_key = str(z['action_key'])
        self.geometry = geometry
        self.fixed_coverage = embodied.CoverageTracker(
            geometry['bounds'], geometry['bins'], geometry.get('axis_names'), geometry.get('valid_mask'))
        if visited is None or not len(visited):
            raise ValueError('MC diagnostics require policy_eval_visit_steps > 0.')
        trajectory = load_trajectory(self.paths[0])
        obs, offset = {}, 0
        for key in proprio_keys:
            shape = trajectory[f'obs/{key}'].shape[2:]
            width = int(np.prod(shape))
            obs[key] = visited[:, offset:offset + width].reshape((len(visited),) + shape)
            offset += width
        self.fixed_coverage.add(geometry['project'](obs))
        self.proprio_keys = proprio_keys
        self.density = StateDensityReference(visited)
        self.density_cache = {}
        np.savez_compressed(self.out_dir / 'density_reference.npz',
                            states=visited, keys=proprio_keys, mean=self.density.mean,
                            scale=self.density.scale, k=1, metric='standardized_euclidean')
        self.read, self.read_start = component_reader, start_reader
        self.evaluate = evaluate_components
        self.frozen = args.policy_eval_corrector == 'load_freeze'
        self.cache, self.previous = {}, {}
        self.mode = self.inner._td_target_mode
        if self.mode not in ('q_full', 'rnd'):
            raise ValueError('MC residual diagnostics require q_full or rnd.')
        self.shift = float(self.inner.config.residual_transform)
        np.savez(self.out_dir / 'reference.npz', point=self.ids,
                 counts=self.fixed_coverage.counts, bounds=geometry['bounds'], bins=geometry['bins'],
                 valid_mask=self.fixed_coverage.valid_mask, discount=self.gamma,
                 reference='fixed_policy_visitation', horizons=self.horizons)

    def report(self, step, td, coverage, logger):
        all_returns, all_valid, predictions = {}, {}, {}
        novelties = {name: [] for name in ('fixed', 'training', 'density')}
        novelty_valid = {name: [] for name in novelties}
        references = {'fixed': self.fixed_coverage}
        if coverage is not None:
            references['training'] = coverage
        percentiles = {name: novelty_percentiles(c.counts.copy(), c.valid_mask)
                       for name, c in references.items()}
        for point, paths in zip(self.ids, self.paths):
            trajectory = load_trajectory(paths)
            obs = {k[4:]: v for k, v in trajectory.items() if k.startswith('obs/')}
            start = self.read_start({k: v[:, 0] for k, v in obs.items()},
                                   {self.action_key: trajectory['action'][:, 0]})
            if point in self.cache:
                point_returns, point_valid = self.cache[point]
            else:
                parts = self.evaluate(trajectory, self.action_key, self.read)
                point_returns, point_valid = {}, {}
                for i in range(len(self.inner.q.heads)):
                    prefix = '' if i == 0 else f'q{i + 1}/'
                    residual = parts[prefix + 'prior'] + parts[prefix + 'corrector']
                    source = np.abs(residual) if self.mode == 'rnd' else residual
                    point_returns[prefix], point_valid[prefix] = trajectory_returns(
                        source, trajectory, self.gamma, self.horizons, self.shift)
                if self.frozen:
                    self.cache[point] = point_returns, point_valid
            for prefix in point_returns:
                # All episodes start at the same deterministic state/action.
                predictions.setdefault(prefix, []).append(start[prefix + 'residual_bootstrap'][0])
                for h in self.horizons:
                    key = prefix, h
                    all_returns.setdefault(key, []).append(point_returns[prefix][h])
                    all_valid.setdefault(key, []).append(point_valid[prefix][h])
            if point not in self.density_cache:
                shape = trajectory['valid'].shape
                states = np.concatenate([obs[k].reshape(shape + (-1,))
                                         for k in self.proprio_keys], axis=-1)
                distance = self.density.distances(states)
                self.density_cache[point] = trajectory_returns(
                    distance, trajectory, self.gamma, self.horizons)
            density_returns, density_valid = self.density_cache[point]
            novelties['density'].append(density_returns)
            novelty_valid['density'].append(density_valid)
            ids = self.fixed_coverage.bin_ids(self.geometry['project'](obs))
            for name, values in percentiles.items():
                ret, valid = trajectory_returns(values[ids], trajectory, self.gamma, self.horizons)
                novelties[name].append(ret)
                novelty_valid[name].append(valid)
        saved = {'point': np.asarray(self.ids), 'step': np.int64(step),
                 'discount': np.float64(self.gamma), 'residual_shift': self.shift,
                 'training_counts': coverage.counts.copy() if coverage is not None else np.array([])}
        metrics = {}
        for (critic, h), values in all_returns.items():
            predicted = np.asarray(predictions[critic])
            returns = np.asarray(values)
            valid = np.asarray(all_valid[critic, h])
            saved[f'{critic}prediction'] = predicted
            saved[f'{critic}returns_h{h}'] = returns
            for reference in [*references, 'density']:
                novelty = np.asarray([v[h][:, 0] for v in novelties[reference]])
                usable = valid & np.asarray([v[h] for v in novelty_valid[reference]])
                key = critic, h, reference
                stats, target, keep = link_metrics(
                    predicted, returns, novelty, usable, self.mode,
                    previous=self.previous.get(key), actual_bonus=td['optimism_bonus'][self.ids])
                moment = predicted.var(-1) if self.mode == 'q_full' else predicted.mean(-1)
                self.previous[key] = np.where(keep, moment, np.nan), np.where(keep, target, np.nan)
                prefix = f'policy_eval/mc/{reference}/{critic}h{h}/'
                metrics.update({prefix + k: v for k, v in stats.items()})
                metrics[prefix + 'discount_tail_fraction'] = self.gamma ** h
                saved[f'{reference}/novelty_h{h}'] = novelty
                saved[f'{reference}/{critic}valid_h{h}'] = usable
        rb = td['residual_bootstrap'][self.ids]
        if self.shift > 0:
            alpha = (1 - self.gamma) * np.log(self.inner.config.residual_transform_reduction)
            rb = np.exp(alpha * rb)
        metrics['policy_eval/mc/actual_bonus/clipped_fraction'] = (
            float((rb.std(-1) >= 100).mean()) if self.mode == 'q_full' else 0.0)
        metrics['policy_eval/mc/step'] = float(step)
        logger.add(metrics)
        logger.write()
        np.savez_compressed(self.out_dir / f'panel_{int(step):08d}.npz', **saved)
