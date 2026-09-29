"""State density and host-side joins between residual/bootstrap scores and coverage."""

import re
import numpy as np
import jax.numpy as jnp
import jax
from dreamerv3 import jaxutils

f32 = jnp.float32


class NoveltyReports:
    """State density and host-side joins between residual/bootstrap scores and coverage."""

    def report_state_diagnostics(self, data, carry, inputs=None):
        embed, outs_h = self._observe_report(data, carry, inputs)
        outs = self._take_dyn_head(outs_h)
        z = f32(outs['deter'])  # [B, T, D]
        distances = _nearest_neighbor_distances(z.reshape(-1, z.shape[-1]))
        metrics = {
            'state_density/knn_dist_mean': distances.mean(),
            'state_density/knn_dist_std': distances.std(),
            'state_density/knn_dist_median': jnp.median(distances),
        }
        metrics.update(self._report_coverage_residual_samples(data, outs, embed))
        return metrics

    def _report_coverage_residual_samples(self, data, outs, embed):
        """Per-state prior residual for host-side coverage comparison.

        True coverage counts live outside the JAX agent. The host reporter removes
        this private array from the report and joins it to each transition's stable
        coverage-bin ID using the current training coverage histogram.
        """
        run = getattr(self.config, 'run', None)
        if not (
                getattr(run, 'log_coverage', False)
                and (getattr(self, 'visual_prior', False)
                     or getattr(self, 'ac_inputs', 'wm') == 'obs')
                and hasattr(self, 'q')
                and hasattr(self.q, 'heads')
                and getattr(self.config.critic_prior, 'use_rff', False)
                and any(q.prior_corrector is not None for q in self.q.heads)):
            return {}

        states = dict(outs)
        if getattr(self, 'ac_inputs', 'wm') == 'drq':
            states['embed'] = embed
        states = self._augment_ac_inputs(states, data, headed=False)
        actions = jaxutils.onehot_dict(
            {k: data[k] for k in self.act_space}, self.act_space)
        q_inp = self._q_input(states, actions)

        residuals = []
        for q in self.q.heads:
            components = q.component_means(
                q_inp, bdims=2, has_ensemble=False,
                prior_scale=1.0, corrector_scale=1.0)
            residuals.append(self._select_action_values(
                components['prior_corrector'], actions))
        residual = jnp.stack(residuals)                       # [Q, E, B, T]
        residual = jnp.abs(residual).mean(axis=(0, 1))       # [B, T]
        return {'_coverage/residual': residual}


def _nearest_neighbor_distances(points, block_size=256):
    """Exact all-neighbor search with O(block_size * N) temporary distances."""
    points = f32(points)
    size = points.shape[0]
    if not size:
        return jnp.empty((0,), f32)
    block_size = min(block_size, size)
    padding = (-size) % block_size
    padded = jnp.pad(points, ((0, padding), (0, 0)))
    norms = (points ** 2).sum(-1)
    columns = jnp.arange(size)

    def nearest(start):
        rows = start + jnp.arange(block_size)
        block = jax.lax.dynamic_slice_in_dim(padded, start, block_size)
        distances = jnp.maximum(
            (block ** 2).sum(-1)[:, None] + norms[None] - 2 * block @ points.T, 0.0)
        # Retain the historical finite diagonal penalty, including N=1.
        distances = distances + (rows[:, None] == columns[None]) * 1e18
        return jnp.sqrt(distances.min(-1))

    starts = jnp.arange(0, size + padding, block_size, dtype=jnp.int32)
    return jax.lax.map(nearest, starts).reshape(-1)[:size]


def _coverage_residual_metrics(metrics, batch, coverage):
    """Join report residuals to current true-state visitation counts by bin."""
    aligned = {}
    metrics = _coverage_bootstrap_metrics(metrics, batch, coverage, aligned)
    residual = metrics.pop('_coverage/residual', None)
    if residual is None or coverage is None or 'coverage_bin' not in batch:
        return metrics

    residual = np.asarray(residual, np.float64)
    bin_ids, bin_valid, flat_counts, _ = _coverage_inputs(batch, coverage, residual.shape, aligned)
    bin_ids, residual = bin_ids.reshape(-1), residual.reshape(-1)
    valid = bin_valid.reshape(-1) & np.isfinite(residual)
    residual, bin_ids = residual[valid], bin_ids[valid]
    if not len(bin_ids):
        return metrics

    # Give every true-state bin equal weight. Otherwise frequently sampled old
    # bins would dominate both sides of the comparison by construction.
    unique_ids, inverse = np.unique(bin_ids, return_inverse=True)
    occurrences = np.bincount(inverse)
    bin_residual = np.bincount(inverse, weights=residual) / occurrences
    bin_count = flat_counts[unique_ids]

    # Quartiles give an interpretable residual ratio without hard-coding an
    # environment-specific density scale.
    low_cut = np.quantile(bin_count, 0.25)
    high_cut = np.quantile(bin_count, 0.75)
    if low_cut < high_cut:
        low_residual = bin_residual[bin_count <= low_cut]
        high_residual = bin_residual[bin_count >= high_cut]
        low_mean = float(low_residual.mean())
        high_mean = float(high_residual.mean())
        low_high_auc = _pairwise_auc(low_residual, high_residual)
        ratio = low_mean / max(high_mean, 1e-8)
        contrast = (low_mean - high_mean) / max(low_mean + high_mean, 1e-8)
    else:
        low_mean = high_mean = low_high_auc = ratio = contrast = np.nan

    # The strict discriminator follows coverage's existing "visited >= 10"
    # convention: never visited by training is novel; >=10 visits is old.
    novel = bin_count == 0
    old = bin_count >= 10
    prefix = 'coverage_novelty/'
    metrics.update({
            prefix + 'count_rank_auc': np.float32(
                    _visitation_rank_auc(bin_count, bin_residual)),
            prefix + 'novel_vs_old_auc': np.float32(
                    _pairwise_auc(bin_residual[novel], bin_residual[old])),
            prefix + 'low_vs_high_auc': np.float32(low_high_auc),
            prefix + 'low_count_residual': np.float32(low_mean),
            prefix + 'high_count_residual': np.float32(high_mean),
            prefix + 'low_high_residual_ratio': np.float32(ratio),
            prefix + 'low_high_residual_contrast': np.float32(contrast),
            prefix + 'evaluated_bins': np.float32(len(unique_ids)),
            prefix + 'novel_bins': np.float32(novel.sum()),
            prefix + 'old_bins': np.float32(old.sum()),
    })
    return metrics


def _coverage_bootstrap_metrics(metrics, batch, coverage, aligned=None):
    """Rank RB moments against discounted bin-novelty percentiles.

    Counts are frozen at report time. A bin's percentile is the fraction of valid
    bins with a smaller inverse count, plus half the fraction with an equal count.
    Both this return and the agent's fit return start at t+1 with weight gamma;
    true terminals close the return and missing/truncated futures invalidate it.
    """
    samples = {k[4:]: metrics.pop(k) for k in list(metrics) if k.startswith('_rb/')}
    if not samples or coverage is None or 'coverage_bin' not in batch:
        return metrics
    scores = {k: np.asarray(v, np.float64) for k, v in samples.items()
                        if k.rsplit('/', 1)[-1] in ('var', 'mean')}
    if not scores:
        return metrics
    shape = next(iter(scores.values())).shape
    if len(shape) != 2 or any(v.shape != shape for v in scores.values()):
        raise ValueError(f'Expected aligned [B, T] bootstrap scores, got {shape}.')
    aligned = {} if aligned is None else aligned
    bin_ids, bin_valid, counts, geometry_valid = _coverage_inputs(batch, coverage, shape, aligned)
    first, last, terminal = (
            _report_batch_array(batch[k], shape).astype(bool)
            for k in ('is_first', 'is_last', 'is_terminal'))
    percentiles = np.full(counts.shape, np.nan)
    num_bins = int(geometry_valid.sum())
    if num_bins:
        inverse_counts = 1.0 / (counts[geometry_valid].astype(np.float64) + 1.0)
        percentiles[geometry_valid] = (_average_ranks(inverse_counts) + 0.5) / num_bins
    clipped = np.clip(bin_ids, 0, len(counts) - 1)
    novelty = np.where(bin_valid, percentiles[clipped], 0)

    def next_step(x):
        return np.concatenate([x[..., 1:], np.zeros_like(x[..., :1])], -1)

    known_step = ~last & ~terminal & next_step(~first)
    next_terminal = next_step(terminal)
    usable_source = next_step(bin_valid & ~last)
    returns, valid = np.zeros(shape), np.ones(shape, bool)
    horizons = sorted({int(k.rsplit('valid_h', 1)[1]) for k in samples
                                          if re.fullmatch(r'(q\d+/)?valid_h\d+', k)})
    for h in range(1, max(horizons, default=0) + 1):
        returns = float(samples['discount']) * np.where(
                next_terminal, 0, next_step(novelty) + next_step(returns))
        valid = known_step & (next_terminal | (usable_source & next_step(valid)))
        if h not in horizons:
            continue
        for key, score in scores.items():
            critic = key.rsplit('/', 1)[0] + '/' if '/' in key else ''
            moment = key.rsplit('/', 1)[-1]
            fit_valid = np.asarray(samples[f'{critic}valid_h{h}'], bool)
            if fit_valid.shape != shape:
                raise ValueError(f'Bootstrap validity shape differs: {fit_valid.shape} versus {shape}.')
            keep = (valid & fit_valid & bin_valid
                            & np.isfinite(score) & np.isfinite(returns))
            residual_return = samples.get(f'{critic}return_moment_h{h}')
            if residual_return is not None:
                residual_return = np.asarray(residual_return, np.float64)
                if residual_return.shape != shape:
                    raise ValueError('Residual and novelty return shapes differ.')
                keep &= np.isfinite(residual_return)
            prefix = f'rb_novelty/{critic}'
            metrics[f'{prefix}{moment}_spearman_h{h}'] = np.float32(
                    _spearman_rank_corr(score[keep], returns[keep]))
            metrics[f'{prefix}valid_pairs_h{h}'] = np.float32(keep.sum())
            metrics[f'{prefix}distinct_start_bins_h{h}'] = np.float32(
                    len(np.unique(bin_ids[keep])))
            if residual_return is not None:
                metrics[f'residual_return_novelty/{critic}{moment}_spearman_h{h}'] = np.float32(
                    _spearman_rank_corr(residual_return[keep], returns[keep]))
                metrics[f'rb_fit/{critic}{moment}_return_spearman_h{h}'] = np.float32(
                    _spearman_rank_corr(score[keep], residual_return[keep]))
    return metrics


def _coverage_inputs(batch, coverage, shape, aligned):
    shape = tuple(shape)
    if shape not in aligned:
        bins = _report_batch_array(batch['coverage_bin'], shape).astype(np.int64)
        if 'geometry' not in aligned:
            aligned['geometry'] = coverage.counts.reshape(-1).copy(), coverage.valid_mask.reshape(-1)
        counts, geometry = aligned['geometry']
        valid = (bins >= 0) & (bins < len(counts))
        valid &= geometry[np.clip(bins, 0, len(counts) - 1)]
        aligned[shape] = bins, valid, counts, geometry
    return aligned[shape]


def _report_batch_array(value, shape):
    """Match the first report replica to its batch shard, retaining time axes."""
    import jax
    if tuple(value.shape) != tuple(shape) and hasattr(value, 'addressable_shards'):
        value = value.addressable_shards[0].data
    value = np.asarray(jax.device_get(value))
    if value.shape != tuple(shape):
        raise ValueError(
                f'Coverage report and batch shapes differ: {shape} versus {value.shape}.')
    return value


def _pairwise_auc(positive, negative):
    """Probability that a positive item scores above a negative item."""
    positive = np.asarray(positive, np.float64).reshape(-1)
    negative = np.asarray(negative, np.float64).reshape(-1)
    if not len(positive) or not len(negative):
        return np.nan
    ordered = np.sort(negative[~np.isnan(negative)])
    lower = np.searchsorted(ordered, positive, side='left')
    upper = np.searchsorted(ordered, positive, side='right')
    # Preserve subtraction-based comparisons: NaNs never win; inf-inf is not a tie.
    wins = lower + 0.5 * (upper - lower) * np.isfinite(positive)
    wins = np.where(np.isnan(positive), 0, wins)
    return float(wins.sum() / (len(positive) * len(negative)))


def _visitation_rank_auc(counts, residuals):
    """Pairwise ranking accuracy for lower count -> larger residual."""
    counts = np.asarray(counts).reshape(-1)
    residuals = np.asarray(residuals, np.float64).reshape(-1)
    order = np.argsort(counts, kind='stable')
    counts, residuals = counts[order], residuals[order]
    # Count ordered residual ranks using a Fenwick tree. Query an entire
    # visitation-count group before inserting it, so equal counts never compete.
    ranks = np.searchsorted(np.unique(residuals[~np.isnan(residuals)]), residuals)
    tree = np.zeros(len(residuals) + 1, np.int64)

    def prefix(stop):
        total = 0
        while stop:
            total += tree[stop]
            stop -= stop & -stop
        return total

    compared, correct, inserted = 0, 0.0, 0
    start = 0
    while start < len(counts):
        stop = start + 1
        while stop < len(counts) and counts[stop] == counts[start]:
            stop += 1
        for i in range(start, stop):
            if np.isnan(residuals[i]):
                continue
            below, through = prefix(ranks[i]), prefix(ranks[i] + 1)
            correct += inserted - through
            if np.isfinite(residuals[i]):
                correct += 0.5 * (through - below)
        compared += start * (stop - start)
        for i in range(start, stop):
            if np.isnan(residuals[i]):
                continue
            inserted += 1
            index = ranks[i] + 1
            while index < len(tree):
                tree[index] += 1
                index += index & -index
        start = stop
    return correct / compared if compared else np.nan


def _average_ranks(values):
    """Zero-based ranks with the mean rank assigned to every tie."""
    _, inverse, counts = np.unique(
            np.asarray(values), return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    ranks = ends - (counts + 1) / 2
    return ranks[inverse]


def _spearman_rank_corr(x, y):
    x, y = np.asarray(x).reshape(-1), np.asarray(y).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return np.nan
    rx, ry = _average_ranks(x[finite]), _average_ranks(y[finite])
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denominator = np.linalg.norm(rx) * np.linalg.norm(ry)
    return float(np.clip(rx @ ry / denominator, -1, 1)) if denominator > 0 else np.nan


class StateDensityReference:
    """Exact nearest-neighbor novelty against a fixed empirical state sample.

    Host counterpart of `_nearest_neighbor_distances`, using a KD tree for
    repeated out-of-sample queries. Standardization is fitted once on reference
    observations; constant coordinates retain unit scale. No coverage bins or
    changing world-model representation enter this score.
    """

    def __init__(self, points):
        from scipy.spatial import cKDTree
        points = np.asarray(points, np.float64)
        if points.ndim != 2 or len(points) < 2 or not np.isfinite(points).all():
            raise ValueError('State density requires at least two finite reference states.')
        self.mean = points.mean(0)
        std = points.std(0)
        self.scale = np.where(std > 1e-6, std, 1.0)
        self.tree = cKDTree((points - self.mean) / self.scale)

    def distances(self, points):
        points = np.asarray(points, np.float64)
        shape = points.shape[:-1]
        flat = points.reshape(-1, points.shape[-1])
        valid = np.isfinite(flat).all(-1)
        result = np.full(len(flat), np.nan)
        result[valid] = self.tree.query((flat[valid] - self.mean) / self.scale,
                                        k=1, eps=0)[0]
        return result.reshape(shape)
