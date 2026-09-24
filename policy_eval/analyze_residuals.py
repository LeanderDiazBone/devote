"""Offline policy-eval analysis; no JAX or environment execution.

Saved anchor diagnostics are distinct from replay reporting metrics. Frozen
trajectory targets exclude t=0 and never bootstrap an unobserved tail.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


HORIZONS = [1, 2, 4, 8, 16, 32, 64, 128, 200, 1000, 2000, 2499]


def metadata(run):
    env, rest = run.name.split('__', 1)
    kv = dict(x.split('=', 1) for x in rest.split('_h=')[0].split('_'))
    return dict(run=run.name, env=env.removeprefix('obs_dmc-'),
                method=kv['tdm'], fc=int(kv['fc']), pls=float(kv['pls']))


def corr(x, y, rank=True):
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[keep], np.asarray(y)[keep]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    if rank:
        x, y = rankdata(x), rankdata(y)
    return float(np.corrcoef(x, y)[0, 1])


def auc(positive, negative):
    positive, negative = positive[np.isfinite(positive)], negative[np.isfinite(negative)]
    if not len(positive) or not len(negative):
        return np.nan
    neg = np.sort(negative)
    return float(np.mean((np.searchsorted(neg, positive, 'left') +
                          np.searchsorted(neg, positive, 'right')) / (2 * len(neg))))


def trajectory_targets(run, out):
    """Per-anchor, per-episode paired residual returns and propagation metrics.

    At t=0 all deterministic anchors match TD inputs. Natural-reset draws do
    not match, so those rows are retained in cache but excluded from fit.
    """
    cache = out / 'targets' / (run.name + '.npz')
    if cache.exists():
        return dict(np.load(cache))
    pe = run / 'policy_eval'
    files = sorted((pe / 'mc_rollouts').glob('*.npz'))
    signed, absolute, valid, propagation, initial = [], [], [], [], []
    for f in files:
        with np.load(f) as z:
            if 'prior' not in z:
                return None
            residual = z['prior'].astype(np.float64) + z['corrector']
            gamma = float(z['discount'])
            first = z['obs/is_first'].astype(bool)
            last = z['obs/is_last'].astype(bool)
            terminal = z['obs/is_terminal'].astype(bool)
            live = z['valid'].astype(bool)
            pscale, cscale = float(z['prior_scale']), float(z['corrector_scale'])
            bonus = pscale * z['prior'].astype(np.float64) + cscale * z['corrector']
            initial.append(residual[:, 0])
            rs, ra, vs, ps = [], [], [], []
            for h in HORIZONS:
                if h >= residual.shape[1]:
                    rs.append(np.full(residual.shape[::2], np.nan))
                    ra.append(np.full(residual.shape[::2], np.nan))
                    vs.append(np.zeros(residual.shape[0], bool))
                    ps.append(np.full((residual.shape[0], 5), np.nan))
                    continue
                # The data inspected here have no natural terminals in these
                # prefixes. Mark any boundary as invalid rather than silently
                # treating timeouts or absent endpoints as terminal closure.
                ok = (live[:, :h + 1].all(1) & ~first[:, 1:h + 1].any(1)
                      & ~last[:, :h + 1].any(1) & ~terminal[:, :h + 1].any(1))
                weights = gamma ** np.arange(1, h + 1)
                s = np.einsum('etp,t->ep', residual[:, 1:h + 1], weights)
                a = np.einsum('etp,t->ep', np.abs(residual[:, 1:h + 1]), weights)
                rs.append(np.where(ok[:, None], s, np.nan))
                ra.append(np.where(ok[:, None], a, np.nan))
                vs.append(ok)
                # Existing bonus_propagation definition starts at t=0.
                w = gamma ** np.arange(h)
                b = bonus[:, :h]
                signed_std = np.einsum('etp,t->ep', b, w).std(-1)
                abs_mean = np.einsum('et,t->e', np.abs(b).mean(-1), w)
                sum_std = np.einsum('et,t->e', b.std(-1), w)
                retention = np.divide(signed_std, sum_std, out=np.full_like(sum_std, np.nan), where=sum_std > 1e-12)
                kappa = np.divide(abs_mean, signed_std, out=np.full_like(sum_std, np.nan), where=signed_std > 1e-12)
                ps.append(np.where(ok[:, None], np.stack([signed_std, abs_mean, sum_std, retention, kappa], -1), np.nan))
            signed.append(rs); absolute.append(ra); valid.append(vs); propagation.append(ps)
    result = dict(signed=np.asarray(signed), absolute=np.asarray(absolute),
                  valid=np.asarray(valid), propagation=np.asarray(propagation),
                  initial=np.asarray(initial), horizons=HORIZONS,
                  prior_scale=pscale, corrector_scale=cscale)
    cache.parent.mkdir(exist_ok=True)
    np.savez_compressed(cache, **result)
    return result


def analyze(root, out):
    out.mkdir(parents=True, exist_ok=True)
    runs = sorted(r for r in root.iterdir() if (r / 'policy_eval/mc.npz').exists())
    summary, fits, points, propagation, checks = [], [], [], [], []
    for run in runs:
        meta = metadata(run)
        print('Analyzing', run.name, flush=True)
        pe = run / 'policy_eval'
        anchors = dict(np.load(pe / 'anchors.npz'))
        mc = dict(np.load(pe / 'mc.npz'))
        distance = np.load(pe / 'anchor_distances.npz')['distances']
        targets = trajectory_targets(run, out) if meta['fc'] else None
        # Frozen files provide the scale. Other runs use matching prior setup;
        # report assumption explicitly and validate frozen initial components.
        scale = float(targets['prior_scale']) if targets else 100.0
        regular = ~anchors['natural_reset']
        masks = {'all': regular, 'state': anchors['kind'] == 'state',
                 'action': anchors['kind'] == 'action',
                 'anchor': (anchors['kind'] == 'anchor') & regular}
        first_td = dict(np.load(sorted(pe.glob('td_*.npz'))[0]))
        if targets:
            initial = targets['initial'].mean(1)
            difference = np.abs((first_td['prior'] + first_td['corrector']) / scale - initial)
            checks.append({**meta, 'scale': scale,
                'initial_cp_mae_excluding_natural': float(difference[regular].mean()),
                'initial_cp_max_error_excluding_natural': float(difference[regular].max()),
                'valid_target_fraction': float(targets['valid'][regular].mean())})
            for hi, h in enumerate(HORIZONS):
                values = targets['propagation'][:, hi]
                for subset, mask in masks.items():
                    row = {**meta, 'subset': subset, 'horizon': h,
                           'valid_sequences': int(targets['valid'][mask, hi].sum())}
                    for j, key in enumerate(['signed_std', 'absolute_mean', 'sum_step_std', 'signed_retention', 'kappa']):
                        row[key] = float(np.nanmean(values[mask, :, j]))
                    propagation.append(row)
        for file in sorted(pe.glob('td_*.npz')):
            td = dict(np.load(file)); step = int(td['step'])
            raw = td['raw'].mean(1)
            error = np.abs(raw - mc['mean'])
            cp = (td['prior'] + td['corrector']).astype(np.float64) / scale
            rb = td['residual_bootstrap'].astype(np.float64) / scale
            scores = dict(cp_abs=np.abs(cp).mean(1), cp_std=cp.std(1),
                          rb_std=rb.std(1), rb_var=rb.var(1), rb_mean=rb.mean(1),
                          bonus=td['optimism_bonus'], raw_std=td['raw'].std(1))
            score = scores['rb_std' if meta['method'] == 'qf' else 'rb_mean']
            for subset, mask in masks.items():
                d, e = distance[mask], error[mask]
                tags = anchors['anchor_tag'][mask]
                row = {**meta, 'step': step, 'subset': subset, 'n': int(mask.sum()),
                       'raw_mae': float(e.mean()), 'raw_rmse': float(np.sqrt(np.mean(e ** 2))),
                       'raw_bias': float((raw - mc['mean'])[mask].mean()),
                       'mc_mean': float(mc['mean'][mask].mean()),
                       'mc_se_mean': float((mc['std'] / np.sqrt(mc['returns'].shape[1]))[mask].mean()),
                       'raw_mean': float(raw[mask].mean()),
                       'q_mae': float(np.abs(td['q'].mean(1) - mc['mean'])[mask].mean()),
                       'corrector_abs_mean': float(np.abs(td['corrector'][mask] / scale).mean()),
                       'prior_abs_mean': float(np.abs(td['prior'][mask] / scale).mean()),
                       'cp_change_from_init': float(np.abs(cp - (first_td['prior'] + first_td['corrector']) / scale)[mask].mean()),
                       'nonfinite_values': int(sum((~np.isfinite(v[mask])).sum() for k,v in td.items() if k != 'step')),
                       'negative_rb_fraction': float((rb[mask] < 0).mean()),
                       'error_distance_spearman': corr(e, d)}
                for name, s in scores.items():
                    s = s[mask]
                    low, high = s[d <= np.quantile(d, .25)], s[d >= np.quantile(d, .75)]
                    row.update({name + '_mean': float(s.mean()),
                        name + '_distance_spearman': corr(s, d),
                        name + '_error_spearman': corr(s, e),
                        name + '_error_pearson': corr(s, e, False),
                        name + '_ood_id_auc': auc(s[tags == 'OOD'], s[tags == 'ID']),
                        name + '_far_near_auc': auc(high, low),
                        name + '_far_near_ratio': float(high.mean() / low.mean()) if abs(low.mean()) > 1e-12 else np.nan})
                for beta in [1, 2, 5]:
                    row[f'raw_error_bonus_coverage_beta{beta}'] = float((e < beta * td['optimism_bonus'][mask] + 1e-6).mean())
                summary.append(row)
            if targets:
                for hi, h in enumerate(HORIZONS):
                    ret = targets['signed' if meta['method'] == 'qf' else 'absolute'][:, hi]
                    # Reporting compares per-trajectory particle moments;
                    # the expectation-first target instead matches conditional
                    # value prediction under a stochastic policy.
                    pred = rb.var(1) if meta['method'] == 'qf' else rb.mean(1)
                    target = np.nanvar(ret, axis=2) if meta['method'] == 'qf' else np.nanmean(ret, axis=2)
                    expected = np.nanmean(ret, axis=1)
                    target_expected = expected.var(1) if meta['method'] == 'qf' else expected.mean(1)
                    # Variance of a sample mean still includes policy sampling
                    # noise. Unbiased correction across independent episodes;
                    # keep negative estimates rather than silently clipping.
                    corrected = ((ret.shape[1] * target_expected - np.nanmean(target, 1)) /
                                 (ret.shape[1] - 1)) if meta['method'] == 'qf' else target_expected
                    for subset in ['all', 'state', 'anchor']:
                        mask = masks[subset]
                        moment = 'var' if meta['method'] == 'qf' else 'mean'
                        fits.append({**meta, 'step': step, 'subset': subset, 'horizon': h, 'moment': moment,
                            'predicted_moment': float(pred[mask].mean()),
                            'return_moment': float(np.nanmean(target[mask])),
                            'moment_abs_error': float(np.nanmean(np.abs(pred[mask,None] - target[mask]))),
                            'expected_return_moment': float(np.nanmean(target_expected[mask])),
                            'sampling_corrected_expected_moment': float(np.nanmean(corrected[mask])),
                            'expected_moment_abs_error': float(np.nanmean(np.abs(pred[mask] - target_expected[mask]))),
                            'target_spearman': corr(pred[mask], np.nanmean(target[mask], 1)),
                            'expected_target_spearman': corr(pred[mask], target_expected[mask]),
                            'valid_starts': int(targets['valid'][mask,hi].sum()),
                            'particle_rmse_vs_expected': float(np.sqrt(np.nanmean((rb[mask] - expected[mask]) ** 2)))})
            if step in [16, 10000, 100000, 500000, 1000000, 2000000]:
                frame = pd.DataFrame({**{k: np.repeat(v,len(raw)) for k,v in meta.items()},
                    'step': step, 'point': np.arange(len(raw)), 'anchor': anchors['anchor_name'],
                    'tag': anchors['anchor_tag'], 'kind': anchors['kind'], 'distance': distance,
                    'mc': mc['mean'], 'mc_std': mc['std'], 'raw': raw, 'error': error, **scores})
                points.append(frame)
    for name, data in [('timeseries',summary), ('frozen_rb_fit',fits),
                       ('frozen_bonus_propagation',propagation), ('validation', checks)]:
        pd.DataFrame(data).to_csv(out / (name + '.csv'), index=False)
    pd.concat(points).to_csv(out / 'point_diagnostics.csv.gz', index=False)
    print('Wrote', out, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    analyze(args.root, args.out)
