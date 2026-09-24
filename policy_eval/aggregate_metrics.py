"""Aggregate MC and TD metrics across runs into a single long-format table.

For each (run, anchor, training step) it records the MC mean/std (from
`mc.npz`) and the TD mean/std (computed across the ensemble heads in
`td_<step>.npz`). Anchor metadata is joined in from `anchors.npz`.

Run from the policy_eval directory:
    python aggregate_metrics.py --root data/policy_eval_11 --out aggregated.csv
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


RUN_RE = re.compile(
    r"^(?P<exp>[^_]+)_(?P<env>[\w-]+)__"
    r"(?P<kvs>(?:[A-Za-z0-9]+=[^_]+_)*)"
    r"h=(?P<hash>[^_]+)_"
    r"seed_(?P<seed>\d+)_"
    r"(?P<timestamp>\d{8}T\d{6})$"
)


def parse_run_name(name: str) -> dict | None:
    m = RUN_RE.match(name)
    if m is None:
        return None
    kv_str = m["kvs"].rstrip("_")
    kvs: dict[str, str] = {}
    if kv_str:
        for pair in kv_str.split("_"):
            k, _, v = pair.partition("=")
            kvs[k] = v
    method = "_".join(f"{k}={kvs[k]}" for k in sorted(kvs))
    return {
        "exp": m["exp"],
        "env": m["env"],
        "hash": m["hash"],
        "seed": m["seed"],
        "timestamp": m["timestamp"],
        "method": method,
        # Common keys aliased for convenience; absent keys -> None.
        "td_method": kvs.get("tdm"),
        **kvs,
    }


def load_run(run_dir: Path) -> pd.DataFrame:
    pe = run_dir / "policy_eval"
    anchors = np.load(pe / "anchors.npz")
    n = anchors["anchor_idx"].shape[0]

    base = pd.DataFrame({
        "anchor_idx": anchors["anchor_idx"],
        "sweep_idx": anchors["sweep_idx"],
        "anchor_name": anchors["anchor_name"],
        "kind": anchors["kind"],
    })

    mc_path = pe / "mc.npz"
    if mc_path.exists():
        mc = np.load(mc_path)
        base["mc_mean"] = mc["mean"]
        base["mc_std"] = mc["std"]
    else:
        base["mc_mean"] = np.nan
        base["mc_std"] = np.nan

    rows = []
    for td_path in sorted(pe.glob("td_*.npz")):
        td = np.load(td_path)
        q = td["q"]  # (n_anchors, n_heads)
        assert q.shape[0] == n, f"{td_path}: expected {n} anchors, got {q.shape[0]}"
        step = int(td["step"]) if "step" in td.files else int(td_path.stem.split("_")[1])
        df = base.copy()
        df["step"] = step

        def _add(prefix, arr):
            df[f"{prefix}_mean"] = arr.mean(axis=1)
            df[f"{prefix}_std"] = arr.std(axis=1)
            for i in range(arr.shape[1]):
                df[f"{prefix}_h{i}"] = arr[:, i]

        _add("td", q)
        if "raw" in td.files:
            _add("raw", td["raw"])
        if "prior" in td.files and "corrector" in td.files:
            _add("cp", td["prior"] + td["corrector"])
        if "residual_bootstrap" in td.files:
            _add("rb", td["residual_bootstrap"])
        rows.append(df)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def aggregate(root: Path) -> pd.DataFrame:
    frames = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        meta = parse_run_name(run_dir.name)
        if meta is None:
            # Skip placeholder dirs like ..._{timestamp}.
            continue
        if not (run_dir / "policy_eval").is_dir():
            continue
        df = load_run(run_dir)
        if df.empty:
            continue
        for k, v in meta.items():
            df[k] = v
        frames.append(df)

    if not frames:
        raise SystemExit(f"No runs found under {root}")

    out = pd.concat(frames, ignore_index=True)
    fixed = [
        "exp", "env", "method", "td_method", "hash",
        "seed", "timestamp",
        "anchor_idx", "sweep_idx", "anchor_name", "kind", "step",
        "mc_mean", "mc_std",
        "td_mean", "td_std", "raw_mean", "raw_std",
        "cp_mean", "cp_std", "rb_mean", "rb_std",
    ]
    fixed = [c for c in fixed if c in out.columns]
    extra = sorted(c for c in out.columns if c not in fixed)
    return out[fixed + extra]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/policy_eval_11"))
    p.add_argument("--out", type=Path, default=Path("aggregated.csv"))
    args = p.parse_args()

    df = aggregate(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix == ".parquet":
        df.to_parquet(args.out, index=False)
    else:
        df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows -> {args.out}")
    print(df.head())


if __name__ == "__main__":
    main()
