"""Generate train / eval_id / eval_ood splits for each normalized dataset.

Loads from `data/normalized/`. For each dataset and each split index, writes
`train.csv`, `eval_id.csv`, `eval_ood.csv`, and a `stats.csv` (per-column
mean/variance for each partition) under:

    data/splits/<dataset>/split<NN>/{train,eval_id,eval_ood,stats}.csv

The OOD criterion is selected via `--strategy`. To add a new strategy,
register a function `(X, y, seed) -> (train_idx, id_idx, ood_idx)` in
`SPLITTERS` below.

Implemented strategies:
    foong_gap   Foong et al. 2019, "'In-Between' Uncertainty in Bayesian
                Neural Networks" (arXiv:1906.11537). Split s holds out the
                middle 1/3 along feature dim s % D as OOD; from the outer
                2/3, a fraction `id_frac` is randomly held out as eval_id.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
NORM = HERE / "data" / "normalized"
SPLITS = HERE / "data" / "splits"

Splitter = Callable[[np.ndarray, np.ndarray, int],
                    tuple[np.ndarray, np.ndarray, np.ndarray]]


# ============================================================
# Splitters
# ============================================================

def foong_gap(X: np.ndarray, y: np.ndarray, seed: int,
              *, id_frac: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, d = X.shape
    order = np.argsort(X[:, seed % d])
    lo, hi = n // 3, 2 * n // 3
    ood_idx = order[lo:hi]
    pool = np.concatenate([order[:lo], order[hi:]])
    perm = np.random.default_rng(seed).permutation(pool)
    n_id = max(int(round(id_frac * len(pool))), 1)
    return perm[n_id:], perm[:n_id], ood_idx


SPLITTERS: dict[str, Splitter] = {
    "foong_gap": foong_gap,
}


# ============================================================
# I/O
# ============================================================

def write_split(out_dir: Path, df: pd.DataFrame,
                train_idx: np.ndarray, id_idx: np.ndarray, ood_idx: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = {"train": df.iloc[train_idx],
             "eval_id": df.iloc[id_idx],
             "eval_ood": df.iloc[ood_idx]}
    for name, part in parts.items():
        part.to_csv(out_dir / f"{name}.csv", index=False)
    stats = pd.DataFrame({f"{name}_{stat}": getattr(part, stat)()
                          for name, part in parts.items()
                          for stat in ("mean", "var")})
    stats.index.name = "column"
    stats.to_csv(out_dir / "stats.csv")


# ============================================================
# Entry point
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="foong_gap", choices=list(SPLITTERS))
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--datasets", nargs="+", default=None, help="subset (matched against data/normalized/*.csv stems)")
    args = parser.parse_args()

    csvs = sorted(NORM.glob("*.csv"))
    if args.datasets:
        csvs = [c for c in csvs if c.stem in args.datasets]
    if not csvs:
        raise SystemExit(f"No CSVs in {NORM}. Run download_data.py first.")

    splitter = SPLITTERS[args.strategy]
    for csv in csvs:
        df = pd.read_csv(csv)
        X = df[[c for c in df.columns if c.startswith("x")]].to_numpy(np.float32)
        y = df["y"].to_numpy(np.float32)
        for s in range(args.n_splits):
            train_idx, id_idx, ood_idx = splitter(X, y, s)
            write_split(SPLITS / csv.stem / f"split{s:02d}", df,
                        train_idx, id_idx, ood_idx)
        print(f"{csv.stem}: wrote {args.n_splits} {args.strategy} splits")


if __name__ == "__main__":
    main()
