"""Train a corrector network to mimic a frozen random prior, in one of three ENN modes.

Common setup: frozen prior f, trainable corrector g, residual r = f - g. Training
drives r -> 0 on the support; off-support r retains its random-init magnitude. The
"variance over an index" of r is reported as uncertainty.

The mode controls how that index is realized:

    ensemble: K independent (f_k, g_k) pairs. Sampled functions: r_k(x), k = 1..K.
              sigma^2(x) = Var_k[r_k(x)].

    input:    Single (f, g) (K = 1) with augmented input [x; z], z ~ N(0, I_{z_dim}).
              Trained with fresh z per example. Sampled functions: r(x; z_s) for S
              MC draws z_s. sigma^2(x) = Var_z[r(x; z)].

    output:   K-member ensemble trained as in `ensemble`; at eval, sampled functions
              are r(x; z) = (1/sqrt(K)) z^T [f(x) - g(x)] for z ~ N(0, I_K). The
              1/sqrt(K) factor keeps the magnitude scale-comparable with ensemble
              mode (Var_z[r(x;z)] = (1/K) sum_k r_k(x)^2).

For each split under `data/splits/<dataset>/split<NN>/`, writes:

    data/uncertainty_predictions/<tag>/<dataset>/split<NN>/{train,eval_id,eval_ood}.csv
        per-point uncertainty with columns (i, y, var)
        y is included for downstream calibration analysis, NOT used in training
    data/uncertainty_predictions/<tag>/<dataset>/split<NN>/grid.csv  (only if source has grid.csv)
        ground-truth columns from the source grid, residual variance var, and a
        bank of PRIOR sample columns f_0..f_{S-1} (frozen random init — never
        sees Y). S = K for ensemble/output; S = n_samples for input.
    data/uncertainty_predictions/<tag>/<dataset>/split<NN>/stats.csv
        per-partition mean variance
    data/uncertainty_predictions/<tag>/stats.csv
        all per-(dataset, split, partition) rows concatenated
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

if __package__:
    from .networks import EnsembleMLP, EnsembleRFN
else:
    from networks import EnsembleMLP, EnsembleRFN

HERE = Path(__file__).parent
SPLITS = HERE / "data" / "splits"
PREDS = HERE / "data" / "uncertainty_predictions"

ARCHS = {"mlp": EnsembleMLP, "rfn": EnsembleRFN}
MODES = ("ensemble", "input", "output")


def _load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    X_cols = [c for c in df.columns if c.startswith("x")]
    return df[X_cols].to_numpy(np.float32), df["y"].to_numpy(np.float32)


def make_pair(arch: str, K: int, in_dim: int, *, mode: str, z_dim: int,
              **kwargs) -> tuple[nn.Module, nn.Module]:
    """Build (corrector, prior). Input mode uses K=1 with [x; z] augmentation."""
    cls = ARCHS[arch]
    if mode == "input":
        K_net, net_in_dim = 1, in_dim + z_dim
    else:
        K_net, net_in_dim = K, in_dim
    corrector = cls(K=K_net, in_dim=net_in_dim, **kwargs)
    prior = cls(K=K_net, in_dim=net_in_dim, **kwargs)
    for p in prior.parameters():
        p.requires_grad = False
    return corrector, prior


def _augment(X: torch.Tensor, z_dim: int) -> torch.Tensor:
    z = torch.randn(X.shape[0], z_dim, device=X.device, dtype=X.dtype)
    return torch.cat([X, z], dim=-1)


def train(corrector: nn.Module, prior: nn.Module, X: np.ndarray, *,
          mode: str, z_dim: int, epochs: int, batch_size: int, lr: float,
          weight_decay: float, device: str) -> None:
    corrector.to(device)
    prior.to(device)
    opt = torch.optim.Adam(corrector.parameters(), lr=lr, weight_decay=weight_decay)
    Xt = torch.from_numpy(X).to(device)
    n = len(X)
    for _ in tqdm(range(epochs), leave=False):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            xb = Xt[perm[i:i + batch_size]]
            if mode == "input":
                xb = _augment(xb, z_dim)
            opt.zero_grad()
            with torch.no_grad():
                target = prior(xb)
            (corrector(xb) - target).pow(2).mean().backward()
            opt.step()


def residuals(corrector: nn.Module, prior: nn.Module, X: np.ndarray, device: str,
              *, mode: str, K: int, z_dim: int, n_samples: int) -> np.ndarray:
    """Sampled residuals r(x; index) of shape [S, n].

    ensemble: S = K, one row per member.
    input:    S = n_samples, one row per MC z-draw.
    output:   S = n_samples, each row is (1/sqrt(K)) z^T r_member(x), z ~ N(0, I_K).
    """
    corrector.eval()
    prior.eval()
    Xt = torch.from_numpy(X).to(device)
    n, d = Xt.shape
    with torch.no_grad():
        if mode == "ensemble":
            return (prior(Xt) - corrector(Xt)).cpu().numpy()  # [K, n]
        if mode == "input":
            S = n_samples
            x_rep = Xt.unsqueeze(0).expand(S, n, d).reshape(S * n, d)
            z = torch.randn(S * n, z_dim, device=device, dtype=Xt.dtype)
            x_aug = torch.cat([x_rep, z], dim=-1)
            r = (prior(x_aug) - corrector(x_aug))            # [1, S*n]
            return r.view(S, n).cpu().numpy()
        if mode == "output":
            r = (prior(Xt) - corrector(Xt))                  # [K, n]
            z = torch.randn(n_samples, K, device=device, dtype=Xt.dtype) / math.sqrt(K)
            return (z @ r).cpu().numpy()                     # [S, n]
        raise ValueError(f"unknown mode: {mode}")


def prior_samples(prior: nn.Module, X: np.ndarray, device: str, *, mode: str,
                  z_dim: int, n_samples: int) -> np.ndarray:
    """Prior sample bank for grid visualization. [K, n] for ensemble/output, [S, n] for input."""
    prior.eval()
    Xt = torch.from_numpy(X).to(device)
    n, d = Xt.shape
    with torch.no_grad():
        if mode in ("ensemble", "output"):
            return prior(Xt).cpu().numpy()
        if mode == "input":
            S = n_samples
            x_rep = Xt.unsqueeze(0).expand(S, n, d).reshape(S * n, d)
            z = torch.randn(S * n, z_dim, device=device, dtype=Xt.dtype)
            x_aug = torch.cat([x_rep, z], dim=-1)
            return prior(x_aug).view(S, n).cpu().numpy()
        raise ValueError(f"unknown mode: {mode}")


def run_split(split_dir: Path, out_dir: Path, *, arch: str, K: int, mode: str,
              z_dim: int, n_samples: int, seed: int, arch_kwargs: dict,
              **train_kwargs) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    X_tr, _ = _load(split_dir / "train.csv")
    torch.manual_seed(seed)
    corrector, prior = make_pair(
        arch, K=K, in_dim=X_tr.shape[1], mode=mode, z_dim=z_dim, **arch_kwargs)
    train(corrector, prior, X_tr, mode=mode, z_dim=z_dim, **train_kwargs)
    device = train_kwargs["device"]
    res_kw = dict(mode=mode, K=K, z_dim=z_dim, n_samples=n_samples)

    rows: list[dict] = []
    for part in ("train", "eval_id", "eval_ood"):
        X, y = _load(split_dir / f"{part}.csv")
        var = residuals(corrector, prior, X, device, **res_kw).var(axis=0)
        pd.DataFrame({"i": np.arange(len(y)), "y": y, "var": var}).to_csv(
            out_dir / f"{part}.csv", index=False)
        rows.append({"partition": part, "n": len(y), "mean_var": float(var.mean())})

    # Grid (synth 1D only): copy source ground-truth columns, add the residual
    # variance, and a bank of PRIOR sample columns f_0..f_{S-1} (frozen random
    # init — never trained on Y). S = K for ensemble/output; S = n_samples for input.
    grid_path = split_dir / "grid.csv"
    if grid_path.exists():
        src = pd.read_csv(grid_path)
        X_g = src[[c for c in src.columns if c.startswith("x")]].to_numpy(np.float32)
        res = residuals(corrector, prior, X_g, device, **res_kw)
        priors = prior_samples(
            prior, X_g, device, mode=mode, z_dim=z_dim, n_samples=n_samples)
        out = src.copy()
        out["var"] = res.var(axis=0)
        for k in range(priors.shape[0]):
            out[f"f_{k}"] = priors[k]
        out.to_csv(out_dir / "grid.csv", index=False)

    pd.DataFrame(rows).to_csv(out_dir / "stats.csv", index=False)
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=list(ARCHS), default="mlp")
    parser.add_argument("--mode", choices=MODES, default="ensemble",
                        help="ENN mode: ensemble (K members), input (z appended to network input, K forced to 1), output (K-member ensemble combined via z ~ N(0, I_K))")
    parser.add_argument("--K", type=int, default=10, help="ensemble size (ignored for input mode)")
    parser.add_argument("--z-dim", type=int, default=4, help="z dimension for input mode")
    parser.add_argument("--n-samples", type=int, default=0, help="MC z samples for input/output (default: K)")
    parser.add_argument("--width", type=int, default=100, help="MLP hidden width (mlp arch only)")
    parser.add_argument("--depth", type=int, default=2, help="MLP number of hidden layers (mlp arch only)")
    parser.add_argument("--length-scale", type=float, default=1.0, help="RBF length scale for RFN priors (ignored for MLP)")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--datasets", nargs="+", default=None, help="subset (matched against dataset folder names under data/splits)")
    args = parser.parse_args(argv)

    z_dim = args.z_dim
    n_samples = args.n_samples or args.K

    if args.arch == "rfn":
        arch_kwargs = {"length_scale": args.length_scale}
    else:
        arch_kwargs = {"width": args.width, "n_hidden": args.depth}
    tag = [args.arch]
    if args.arch == "mlp":
        tag.append(f"w{args.width}d{args.depth}")
    if args.mode != "ensemble":
        tag.append(args.mode)
        if args.mode == "input":
            tag.append(f"zd{z_dim}")
    if args.arch == "rfn":
        tag.append(f"l{args.length_scale:g}")
    arch_tag = "_".join(tag)

    splits = sorted(SPLITS.glob("*/split*"))
    if args.datasets:
        splits = [s for s in splits if s.parts[-2] in args.datasets]
    if not splits:
        raise SystemExit(f"No splits in {SPLITS}. Run generate_splits.py first.")

    out_root = PREDS / arch_tag
    train_kwargs = dict(epochs=args.epochs, batch_size=args.batch_size,
                        lr=args.lr, weight_decay=args.weight_decay, device=args.device)
    all_rows: list[dict] = []
    for split_dir in splits:
        dataset, split_name = split_dir.parts[-2], split_dir.parts[-1]
        split_idx = int(split_name.removeprefix("split"))
        rows = run_split(split_dir, out_root / dataset / split_name,
                         arch=args.arch, K=args.K, mode=args.mode, z_dim=z_dim,
                         n_samples=n_samples, seed=split_idx,
                         arch_kwargs=arch_kwargs, **train_kwargs)
        for r in rows:
            all_rows.append({"dataset": dataset, "split": split_name, **r})
        print(f"{dataset}/{split_name}: " + "  ".join(
            f"{r['partition']}_var={r['mean_var']:.4f}" for r in rows))

    out_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out_root / "stats.csv", index=False)
    print(f"\nWrote aggregated stats to {out_root / 'stats.csv'}")


if __name__ == "__main__":
    main()
