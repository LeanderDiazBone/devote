"""Reporting-style novelty statistics using a fixed MC visitation reference.

This reference is NOT the training coverage histogram. Names are explicitly
prefixed proxy_ to keep these outputs distinct from original logged metrics.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from analyze_residuals import HORIZONS, metadata, corr, auc


def bins(obs, env):
    # Matches embodied/envs/dmc.py:get_coverage_geometry, including clipping.
    if env.startswith('cartpole'):
        p = obs['position']
        xy = np.stack([p[..., 0], np.arctan2(p[..., 2], p[..., 1])], -1)
        lo, hi = np.array([-1.8, -np.pi]), np.array([1.8, np.pi])
    else:
        xy = obs['velocity'][..., :2]
        lo = np.array([-5., -2.]) if env.startswith('cheetah') else np.array([-3., -2.])
        hi = np.array([10., 4.]) if env.startswith('cheetah') else np.array([3., 4.])
    ij = np.clip(np.floor((xy - lo) / (hi - lo) * 25).astype(int), 0, 24)
    return ij[..., 0] * 25 + ij[..., 1]


def references(run, out):
    cache = out / 'novelty_targets' / (run.name + '.npz')
    if cache.exists():
        return dict(np.load(cache))
    pe = run / 'policy_eval'; env = metadata(run)['env']
    with np.load(pe / 'mc_rollouts/point_0000_chunk_0000.npz') as z:
        dims = {k[4:]: int(np.prod(z[k].shape[2:])) for k in z.files if k.startswith('obs/')}
    with np.load(pe / 'anchor_distances.npz') as z:
        visited, keys = z['visited'], z['proprio_keys']
    offset, obs = 0, {}
    for key in keys:
        obs[key] = visited[:, offset:offset+dims[key]]
        offset += dims[key]
    counts = np.bincount(bins(obs, env), minlength=625)
    percentile = (rankdata(1. / (counts + 1)) - .5) / len(counts)
    starts, targets, valid = [], [], []
    obs_key = 'position' if env.startswith('cartpole') else 'velocity'
    for f in sorted((pe / 'mc_rollouts').glob('*.npz')):
        with np.load(f) as z:
            ids = bins({obs_key: z['obs/' + obs_key]}, env)
            live = z['valid'] & ~z['obs/is_last'] & ~z['obs/is_terminal']
            live[:, 1:] &= ~z['obs/is_first'][:, 1:]
            gamma = float(z['discount'])
            cs = np.cumsum(percentile[ids[:, 1:]] * gamma ** np.arange(1,ids.shape[1]), axis=1)
            keep = np.logical_and.accumulate(live, axis=1)
            targets.append(np.stack([cs[:, h-1] for h in HORIZONS]))
            valid.append(np.stack([keep[:, h] for h in HORIZONS]))
            starts.append(ids[:, 0])
    result = dict(counts=counts, percentile=percentile, starts=np.array(starts),
                  returns=np.array(targets), valid=np.array(valid))
    cache.parent.mkdir(exist_ok=True)
    np.savez_compressed(cache, **result)
    return result


def analyze(root, out):
    rows, coverage_rows = [], []
    for run in sorted(root.iterdir()):
        if not (run / 'policy_eval/mc.npz').exists(): continue
        print('Novelty', run.name, flush=True)
        meta=metadata(run); pe=run/'policy_eval'; ref=references(run,out)
        anchors=dict(np.load(pe/'anchors.npz'))
        for file in sorted(pe.glob('td_*.npz')):
            with np.load(file) as z:
                rb=z['residual_bootstrap'].astype(float)/100
                residual=np.abs((z['prior'].astype(float)+z['corrector'])/100).mean(1)
                step=int(z['step'])
            moment=rb.var(1) if meta['method']=='qf' else rb.mean(1)
            for subset, mask in [('all',~anchors['natural_reset']),('state',anchors['kind']=='state'),('anchor',(anchors['kind']=='anchor')&~anchors['natural_reset'])]:
                for hi,h in enumerate(HORIZONS):
                    valid=ref['valid'][mask,hi]
                    ret=ref['returns'][mask,hi]
                    pred=np.broadcast_to(moment[mask,None],ret.shape)
                    rows.append({**meta,'step':step,'subset':subset,'horizon':h,
                        'proxy_future_novelty_spearman':corr(pred[valid],ret[valid]),
                        'proxy_expected_future_novelty_spearman':corr(moment[mask],np.where(valid,ret,np.nan).mean(1)),
                        'valid_pairs':int(valid.sum()),'distinct_start_bins':len(np.unique(ref['starts'][mask][valid])),
                        'mean_novelty_return':float(ret[valid].mean())})
                ids=ref['starts'][mask,0]
                unique,inv=np.unique(ids,return_inverse=True)
                r=np.bincount(inv,weights=residual[mask])/np.bincount(inv)
                counts=ref['counts'][unique]
                less=counts[:,None]<counts[None,:]
                wins=(r[:,None]>r[None,:])+.5*(r[:,None]==r[None,:])
                low_cut,high_cut=np.quantile(counts,[.25,.75])
                low=r[counts<=low_cut]
                high=r[counts>=high_cut]
                separated=low_cut<high_cut
                coverage_rows.append({**meta,'step':step,'subset':subset,
                    'proxy_count_rank_auc':float(wins[less].mean()) if less.any() else np.nan,
                    'proxy_novel_vs_old_auc':auc(r[counts==0],r[counts>=10]),
                    'proxy_low_vs_high_auc':auc(low,high) if separated else np.nan,
                    'low_count_residual':float(low.mean()) if separated else np.nan,
                    'high_count_residual':float(high.mean()) if separated else np.nan,
                    'low_high_residual_ratio':float(low.mean()/max(high.mean(),1e-8)) if separated else np.nan,
                    'low_high_residual_contrast':float((low.mean()-high.mean())/max(low.mean()+high.mean(),1e-8)) if separated else np.nan,
                    'evaluated_bins':len(unique),'novel_bins':int((counts==0).sum()),'old_bins':int((counts>=10).sum())})
    pd.DataFrame(rows).to_csv(out/'proxy_rb_novelty.csv',index=False)
    pd.DataFrame(coverage_rows).to_csv(out/'proxy_coverage_novelty.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();analyze(a.root,a.out)
