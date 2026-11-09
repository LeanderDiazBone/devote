"""Compare our evaluation returns with a reference implementation's.

Both sides are on environment (simulator) steps. Ours: `eval_epstats/score/avg`
from each run's metrics.jsonl (the logger already multiplies steps by the action repeat).
Reference: BRO's reported curves (`--ref_npy`, shape [evals, seeds], one eval every
`--ref_every` steps) or MaxInfoRL's `{seed}.txt` files (`--ref_txt_dir`, columns:
environment steps, return).

Per seed, the curve is interpolated on a common grid; we compare the final return
(mean over the last 10% of the horizon) and the mean return over the whole curve.
A metric passes if the bootstrap 95% CI of (ours - reference) contains 0, or the
difference is within `--margin` of the reference mean. Exit code 1 if any fails.
"""

import argparse
import glob
import json
import pathlib
import sys

import numpy as np


def load_ours(logdir):
    points = []
    for line in (logdir / 'metrics.jsonl').read_text().splitlines():
        row = json.loads(line)
        if 'eval_epstats/score/avg' in row:
            points.append((row['step'], row['eval_epstats/score/avg']))
    return np.array(points)


def load_reference(args):
    if args.ref_npy:
        returns = np.load(args.ref_npy)
        steps = args.ref_every * np.arange(1, len(returns) + 1)
        return [np.stack([steps, returns[:, i]], 1) for i in range(returns.shape[1])]
    return [np.loadtxt(f, ndmin=2) for f in sorted(glob.glob(f'{args.ref_txt_dir}/*.txt'))]


def summarize(curves, grid):
    curves = np.stack([np.interp(grid, c[:, 0], c[:, 1]) for c in curves])
    final = curves[:, grid >= 0.9 * grid[-1]].mean(1)
    return {'final': final, 'mean': curves.mean(1)}


def bootstrap_diff(a, b, rng, n=10000):
    diffs = rng.choice(a, (n, len(a))).mean(1) - rng.choice(b, (n, len(b))).mean(1)
    return np.percentile(diffs, [2.5, 97.5])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True, help='Our run logdirs (one per seed).')
    parser.add_argument('--ref_npy', default='')
    parser.add_argument('--ref_every', type=int, default=25000)
    parser.add_argument('--ref_txt_dir', default='')
    parser.add_argument('--margin', type=float, default=0.1)
    args = parser.parse_args()
    assert bool(args.ref_npy) != bool(args.ref_txt_dir), 'Pass exactly one of --ref_npy or --ref_txt_dir.'

    ours = [load_ours(pathlib.Path(d)) for d in args.runs]
    ours = [c for c in ours if len(c) >= 2]
    reference = load_reference(args)
    assert ours and reference, (len(ours), len(reference))
    horizon = min(min(c[-1, 0] for c in ours), min(c[-1, 0] for c in reference))
    start = max(max(c[0, 0] for c in ours), max(c[0, 0] for c in reference))
    grid = np.linspace(start, horizon, 50)
    ours_s, ref_s = summarize(ours, grid), summarize(reference, grid)

    rng = np.random.default_rng(0)
    print(f'seeds: ours={len(ours)} reference={len(reference)}; compared steps {start:.0f}..{horizon:.0f}')
    passed = True
    for name in ('final', 'mean'):
        a, b = ours_s[name], ref_s[name]
        lo, hi = bootstrap_diff(a, b, rng)
        diff = a.mean() - b.mean()
        ok = lo <= 0 <= hi or abs(diff) <= args.margin * abs(b.mean())
        passed &= ok
        print(f'{name:>5}: ours {a.mean():7.1f} ± {a.std():5.1f}  reference {b.mean():7.1f} ± {b.std():5.1f}  '
              f'diff {diff:+7.1f} (95% CI {lo:+.1f}..{hi:+.1f})  {"PASS" if ok else "FAIL"}')
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
