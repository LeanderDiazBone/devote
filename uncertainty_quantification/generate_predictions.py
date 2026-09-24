"""Train a (bootstrapped) MLP ensemble per split and write per-point predictions.

With `--K 1` (default) this is a single MLP point estimator — the baseline.
With `--K > 1` it trains K members in one batched pass, each weighted by an
independent Poisson(1) bootstrap draw of the training data (Osband et al.
2016, Lakshminarayanan et al. 2017). Per-point prediction is the ensemble
mean; ensemble variance is reported alongside.

For each split under `data/splits/<dataset>/split<NN>/`, writes:

    data/predictions/K<K>/<dataset>/split<NN>/{train,eval_id,eval_ood}.csv
        per-point predictions with columns (i, y, f, var)
        (var is identically 0 when K=1)
    data/predictions/K<K>/<dataset>/split<NN>/grid.csv  (only if source has grid.csv)
        ground-truth columns from the source grid (x..., f_true, mu_gp, var_gp),
        the trained ensemble's mean f and variance var, plus per-member
        predictions f_0..f_{K-1} from the RAW (untrained) initialization
        — i.e. the "prior" functions before any data is seen.
    data/predictions/K<K>/<dataset>/split<NN>/stats.csv
        per-partition mse / rmse / mean_var for this split
    data/predictions/K<K>/stats.csv
        all per-(dataset, split, partition) rows concatenated
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from networks import EnsembleMLP

HERE = Path(__file__).parent
SPLITS = HERE / "data" / "splits"
PREDS = HERE / "data" / "predictions"


def _load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    X_cols = [c for c in df.columns if c.startswith("x")]
    return df[X_cols].to_numpy(np.float32), df["y"].to_numpy(np.float32)


def make_mlp(in_dim: int, *, K: int, seed: int, device: str) -> EnsembleMLP:
    """Construct the K-member ensemble at its raw initialization ('prior')."""
    torch.manual_seed(seed)
    return EnsembleMLP(K=K, in_dim=in_dim).to(device)


def train_mlp(model: EnsembleMLP, X: np.ndarray, y: np.ndarray, *, K: int, seed: int,
              epochs: int, batch_size: int, lr: float, weight_decay: float, device: str) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    Xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y).to(device)
    n = len(X)
    # Per-member Poisson(1) bootstrap weights when K > 1; ones for the single-model case.
    if K == 1:
        w = torch.ones(1, n, device=device)
    else:
        w = torch.from_numpy(
            np.random.default_rng(seed).poisson(1.0, size=(K, n)).astype(np.float32)
        ).to(device)
    for _ in tqdm(range(epochs)):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            ((model(Xt[idx]) - yt[idx]).pow(2) * w[:, idx]).mean().backward()
            opt.step()


def predict(model: EnsembleMLP, X: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X).to(device)).cpu().numpy()  # [K, n]


def run_split(split_dir: Path, out_dir: Path, *, K: int, seed: int, **train_kwargs) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    X_tr, y_tr = _load(split_dir / "train.csv")
    device = train_kwargs["device"]
    model = make_mlp(in_dim=X_tr.shape[1], K=K, seed=seed, device=device)

    # Per-member predictions at the raw initialization ("prior" functions) on the
    # grid — captured BEFORE training so they are not influenced by the data.
    grid_path = split_dir / "grid.csv"
    src_grid = None
    prior_grid = None
    if grid_path.exists():
        src_grid = pd.read_csv(grid_path)
        X_g = src_grid[[c for c in src_grid.columns if c.startswith("x")]].to_numpy(np.float32)
        prior_grid = predict(model, X_g, device)  # [K, n_grid]

    train_mlp(model, X_tr, y_tr, K=K, seed=seed, **train_kwargs)

    rows: list[dict] = []
    for part in ("train", "eval_id", "eval_ood"):
        X, y = _load(split_dir / f"{part}.csv")
        preds = predict(model, X, device)  # [K, n]
        f, var = preds.mean(axis=0), preds.var(axis=0)
        pd.DataFrame({"i": np.arange(len(y)), "y": y, "f": f, "var": var}).to_csv(out_dir / f"{part}.csv", index=False)
        mse = float(np.mean((f - y) ** 2))
        rows.append({"partition": part, "n": len(y), "mse": mse,
                     "rmse": float(np.sqrt(mse)), "mean_var": float(var.mean())})

    # Grid (synth 1D only): copy source ground-truth columns, then add the trained
    # ensemble's mean f / variance, and the per-member PRIOR predictions f_0..f_{K-1}.
    if src_grid is not None:
        X_g = src_grid[[c for c in src_grid.columns if c.startswith("x")]].to_numpy(np.float32)
        preds_g = predict(model, X_g, device)  # [K, n_grid]
        out = src_grid.copy()
        out["f"] = preds_g.mean(axis=0)
        out["var"] = preds_g.var(axis=0)
        for k in range(K):
            out[f"f_{k}"] = prior_grid[k]
        out.to_csv(out_dir / "grid.csv", index=False)

    pd.DataFrame(rows).to_csv(out_dir / "stats.csv", index=False)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=1, help="ensemble size; K=1 is the single-MLP baseline, K>1 trains a Poisson-bootstrapped ensemble")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu")) # else "mps" if torch.backends.mps.is_available()
    parser.add_argument("--datasets", nargs="+", default=None, help="subset (matched against dataset folder names under data/splits)")
    args = parser.parse_args()

    splits = sorted(SPLITS.glob("*/split*"))
    if args.datasets:
        splits = [s for s in splits if s.parts[-2] in args.datasets]
    if not splits:
        raise SystemExit(f"No splits in {SPLITS}. Run generate_splits.py first.")

    out_root = PREDS / f"K{args.K}"
    train_kwargs = dict(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay, device=args.device)
    all_rows: list[dict] = []
    for split_dir in splits:
        dataset, split_name = split_dir.parts[-2], split_dir.parts[-1]
        split_idx = int(split_name.removeprefix("split"))
        rows = run_split(split_dir, out_root / dataset / split_name, K=args.K, seed=split_idx, **train_kwargs)
        for r in rows:
            all_rows.append({"dataset": dataset, "split": split_name, **r})
        print(f"{dataset}/{split_name}: " + "  ".join(f"{r['partition']}_rmse={r['rmse']:.3f}/var={r['mean_var']:.4f}" for r in rows))

    out_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out_root / "stats.csv", index=False)
    print(f"\nWrote aggregated stats to {out_root / 'stats.csv'}")


if __name__ == "__main__":
    main()
