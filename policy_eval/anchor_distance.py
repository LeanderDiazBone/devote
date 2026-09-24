"""Summarize anchor_distances.npz produced by embodied.run.policy_eval.

For each run dir it prints a table of per-anchor min normalized Euclidean
distance to the on-policy visitation collected at the end of MC. Lower =
closer to what the frozen exp_actor actually visits.

Usage:
    python anchor_distance.py --root data/policy_eval_19
    python anchor_distance.py --run  data/policy_eval_19/obs_dmc-cheetah-run__...
"""

import argparse
from pathlib import Path

import numpy as np


def _kind_str(arr):
    return arr.astype(str) if arr.dtype.kind == "S" else arr


def summarize_run(run_dir: Path) -> None:
    pe = run_dir / "policy_eval"
    dist_path = pe / "anchor_distances.npz"
    if not dist_path.exists():
        print(f"[skip] no anchor_distances.npz under {pe}")
        return

    anchors = np.load(pe / "anchors.npz")
    dist = np.load(dist_path)

    kind = _kind_str(anchors["kind"])
    name = _kind_str(anchors["anchor_name"])
    dists = dist["distances"]
    mask = kind == "anchor"

    print(f"\n=== {run_dir.name} ===")
    print(f"  visitation samples : {dist['visited'].shape[0]}")
    print(f"  proprio dim={dist['mean'].shape[0]}  keys={list(dist['proprio_keys'])}")
    print(f"  {'anchor':>26s}  min_dist")
    order = np.argsort(dists[mask])
    a_names = name[mask]
    a_dists = dists[mask]
    for j in order:
        print(f"  {str(a_names[j]):>26s}  {float(a_dists[j]):8.3f}")


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", type=Path, help="single run directory")
    g.add_argument("--root", type=Path, help="parent dir with many run dirs")
    args = p.parse_args()

    if args.run is not None:
        summarize_run(args.run)
        return
    for run_dir in sorted(args.root.iterdir()):
        if run_dir.is_dir() and (run_dir / "policy_eval" / "anchor_distances.npz").exists():
            summarize_run(run_dir)


if __name__ == "__main__":
    main()
