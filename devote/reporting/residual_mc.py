"""Host reductions for matched MC residual/RB/novelty diagnostics.

Axes: point, episode, time, particle. Episode expectation precedes particle
variance. No model calls or persistent report state live here.
"""

import numpy as np

from .novelty import _pairwise_auc, _spearman_rank_corr


def trajectory_returns(values, trajectory, discount, horizons, shift=0.0):
    """Returns at the initial state, starting at t+1 with weight gamma.

    Trajectory reward/terminal flags describe the outgoing transition, unlike
    replay observation flags. True terminals close returns; timeouts, resets,
    nonfinite sources and missing futures invalidate them.
    """
    values = np.asarray(values, np.float64)
    if values.ndim == 2:
        values = values[..., None]
    episodes, length, particles = values.shape
    horizons = sorted(set(int(h) for h in horizons))
    if not horizons or horizons[0] < 1:
        raise ValueError('MC horizons must be positive.')
    total = np.zeros((episodes, particles), np.float64)
    valid = np.asarray(trajectory['valid'][:, 0], bool).copy()
    closed = np.zeros(episodes, bool)
    results, masks = {}, {}
    for k in range(1, max(horizons) + 1):
        if k - 1 < length:
            terminal = trajectory['is_terminal'][:, k - 1].astype(bool)
            last = trajectory['is_last'][:, k - 1].astype(bool)
            closed |= valid & terminal
            valid &= closed | ~last
        active = valid & ~closed
        if k >= length:
            valid &= closed
        else:
            usable = (trajectory['valid'][:, k]
                      & ~trajectory['obs/is_first'][:, k]
                      & np.isfinite(values[:, k]).all(-1))
            valid &= ~active | usable
            total += discount ** k * np.where(
                (valid & ~closed)[:, None], values[:, k] - shift, 0)
        if k in horizons:
            results[k] = np.where(valid[:, None], total, np.nan).copy()
            masks[k] = valid.copy()
    return results, masks


def _pearson(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def link_metrics(prediction, returns, novelty, valid, mode, previous=None,
                 actual_bonus=None):
    """One horizon/reference on a common set of points for all three links.

    prediction [N,P], returns [N,E,P], novelty/valid [N,E]. ``returns`` must
    already use signed residuals (q_full) or absolute residuals (rnd).
    """
    if mode not in ('q_full', 'rnd'):
        raise ValueError(f'Unsupported residual target mode: {mode}')
    prediction = np.asarray(prediction, np.float64)
    valid = (np.asarray(valid, bool) & np.isfinite(returns).all(-1)
             & np.isfinite(novelty))
    counts = valid.sum(1)
    expected = np.where(valid[..., None], returns, 0).sum(1) / np.maximum(counts[:, None], 1)
    novelty_mean = np.where(valid, novelty, 0).sum(1) / np.maximum(counts, 1)
    moment = (lambda x: x.var(-1)) if mode == 'q_full' else (lambda x: x.mean(-1))
    predicted, target = moment(prediction), moment(expected)
    keep = ((counts >= 2) & np.isfinite(prediction).all(-1)
            & np.isfinite(target) & np.isfinite(novelty_mean))
    if actual_bonus is not None:
        keep &= np.isfinite(actual_bonus)
    b, u, n = predicted[keep], target[keep], novelty_mean[keep]
    result = {'valid_points': float(keep.sum()), 'valid_episodes': float(counts[keep].sum()),
              'particles': float(prediction.shape[-1]), 'moment_is_variance': float(mode == 'q_full')}
    if not keep.any():
        for key in ('fit/predicted', 'fit/target', 'fit/bias', 'fit/mae', 'fit/nmae',
                    'fit/rmse', 'fit/spearman', 'residual_novelty/spearman',
                    'residual_novelty/pearson', 'rb_novelty/spearman', 'rb_novelty/pearson'):
            result[key] = np.nan
        return result, target, keep
    err = b - u
    result.update({'fit/predicted': b.mean(), 'fit/target': u.mean(),
                   'fit/bias': err.mean(), 'fit/mae': np.abs(err).mean(),
                   'fit/nmae': np.abs(err).mean() / max(np.abs(u).mean(), 1e-8),
                   'fit/rmse': np.sqrt(np.mean(err ** 2)),
                   'fit/spearman': _spearman_rank_corr(b, u),
                   'fit/pearson': _pearson(b, u),
                   'fit/head_rmse': np.sqrt(np.mean((prediction[keep] - expected[keep]) ** 2)),
                   'fit/target_std': u.std(), 'fit/predicted_std': b.std(),
                   'novelty/mean': n.mean(), 'novelty/std': n.std()})
    # MC error of each head's conditional return estimate. For variance also
    # expose the reporting-style per-trajectory moment and sampling correction.
    centered = np.where(valid[..., None], returns - expected[:, None], 0)
    episode_var = (centered ** 2).sum(1) / np.maximum(counts[:, None] - 1, 1)
    result['fit/mc_head_mean_se'] = np.sqrt(episode_var[keep] / counts[keep, None]).mean()
    path_moment = np.where(valid, moment(returns), 0).sum(1) / np.maximum(counts, 1)
    result['fit/trajectory_moment'] = path_moment[keep].mean()
    if mode == 'q_full':
        corrected = (counts * target - path_moment) / np.maximum(counts - 1, 1)
        result['fit/sampling_corrected_target'] = corrected[keep].mean()
    for name, score in [('residual_novelty', u), ('rb_novelty', b)]:
        result[f'{name}/spearman'] = _spearman_rank_corr(score, n)
        result[f'{name}/pearson'] = _pearson(score, n)
        lo, hi = np.quantile(n, [.25, .75])
        result[f'{name}/high_low_auc'] = (
            _pairwise_auc(score[n >= hi], score[n <= lo]) if lo < hi else np.nan)
    if actual_bonus is not None:
        bonus = np.asarray(actual_bonus)[keep]
        result['actual_bonus/novelty_spearman'] = _spearman_rank_corr(bonus, n)
        result['actual_bonus/std'] = bonus.std()
    if previous is not None:
        old_prediction, old_target = previous
        paired = keep & np.isfinite(old_prediction) & np.isfinite(old_target)
        result['tracking/valid_points'] = float(paired.sum())
        for name, delta in [('prediction_change', predicted - old_prediction),
                            ('target_change', target - old_target)]:
            result[f'tracking/{name}'] = np.abs(delta[paired]).mean() if paired.any() else np.nan
        result['tracking/change_spearman'] = _spearman_rank_corr(
            (predicted - old_prediction)[paired], (target - old_target)[paired])
    return result, target, keep
