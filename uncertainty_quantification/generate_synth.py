"""Generate a synthetic GP regression dataset (arbitrary dim, configurable kernel)
with clustered train/eval_id + uniform eval_ood splits.

For each split:
  1. Draw anchors uniformly in [domain]^d.
  2. Train and eval_id share the same anchors and form eps-clusters around them.
     eval_ood is uniform over the full domain.
  3. Sample noiseless f at all points jointly from GP(0, k) and add iid
     Gaussian noise to produce y.
  4. Conditioned on (X_tr, y_tr), compute the exact GP posterior (latent mean
     and variance) at the eval points and (in 1D only) on a dense grid.

Layout (mirrors `generate_splits.py`):

    data/splits/<name>/
        config.json                          # kernel, length_scale, noise_std, etc.
        split<NN>/
            train.csv      # x0,...,x{D-1}, y
            eval_id.csv    # x0,...,x{D-1}, y, mu_gp, var_gp
            eval_ood.csv   # x0,...,x{D-1}, y, mu_gp, var_gp
            grid.csv       # 1D only: x0, f_true, mu_gp, var_gp, prior_0,...,prior_{S-1}
            stats.csv

`mu_gp`/`var_gp` are the LATENT posterior moments (Var[f(x) | X_tr, y_tr]);
add `noise_std**2` to `var_gp` to get the predictive variance over y.

Default name is `synth_gp_<dim>_<kernel>_<length_scale>` so multiple
configurations coexist under `data/splits/`.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from networks import EnsembleMLP

HERE = Path(__file__).parent
SPLITS = HERE / "data" / "splits"


# ============================================================
# Kernels
# ============================================================

def _sqdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances, shape (len(a), len(b))."""
    return np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)


def _rbf(a, b, *, length_scale, sigma_f):
    return sigma_f ** 2 * np.exp(-_sqdist(a, b) / (2 * length_scale ** 2))


def _matern(a, b, *, nu, length_scale, sigma_f):
    r = np.sqrt(np.maximum(_sqdist(a, b), 0.0))
    if nu == 0.5:
        return sigma_f ** 2 * np.exp(-r / length_scale)
    if nu == 1.5:
        c = math.sqrt(3.0) * r / length_scale
        return sigma_f ** 2 * (1.0 + c) * np.exp(-c)
    if nu == 2.5:
        c = math.sqrt(5.0) * r / length_scale
        return sigma_f ** 2 * (1.0 + c + c ** 2 / 3.0) * np.exp(-c)
    raise ValueError(f"Unsupported Matern nu={nu}; use 0.5, 1.5, or 2.5")


KERNELS = ("rbf", "matern12", "matern32", "matern52")


def make_kernel(name: str, *, length_scale: float, sigma_f: float):
    """Return k(a, b) closure for the named stationary kernel."""
    if name == "rbf":
        return lambda a, b: _rbf(a, b, length_scale=length_scale, sigma_f=sigma_f)
    nu = {"matern12": 0.5, "matern32": 1.5, "matern52": 2.5}.get(name)
    if nu is None:
        raise ValueError(f"Unknown kernel {name!r}; choose from {KERNELS}")
    return lambda a, b: _matern(a, b, nu=nu, length_scale=length_scale, sigma_f=sigma_f)


# ============================================================
# GP sampling + posterior
# ============================================================

def _chol(K: np.ndarray, jitter: float = 1e-6) -> np.ndarray:
    """Cholesky with adaptive jitter for numerical stability."""
    n = K.shape[0]
    for _ in range(8):
        try:
            return np.linalg.cholesky(K + jitter * np.eye(n))
        except np.linalg.LinAlgError:
            jitter *= 10
    raise np.linalg.LinAlgError(f"Cholesky failed up to jitter {jitter}")


def sample_prior(kernel, X: np.ndarray, seed: int) -> np.ndarray:
    """Draw f at X from GP(0, kernel) — noiseless."""
    L = _chol(kernel(X, X))
    return L @ np.random.default_rng(seed).standard_normal(len(X))


def sample_prior_batch(kernel, X: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    """Draw `n_samples` independent f's at X from GP(0, kernel). Returns (n_samples, len(X))."""
    L = _chol(kernel(X, X))
    Z = np.random.default_rng(seed).standard_normal((len(X), n_samples))
    return (L @ Z).T.astype(np.float32)


def gp_posterior(kernel, X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, *,
                 noise_std: float) -> tuple[np.ndarray, np.ndarray]:
    """Latent posterior (mean, variance) of f at X_te given noisy (X_tr, y_tr)."""
    K_tr   = kernel(X_tr, X_tr) + noise_std ** 2 * np.eye(len(X_tr))
    K_s_tr = kernel(X_te, X_tr)
    k_ss   = np.diag(kernel(X_te, X_te))
    L = _chol(K_tr)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_tr))   # K_tr^{-1} y_tr
    V     = np.linalg.solve(L, K_s_tr.T)                       # L^{-1} K_tr_te
    mu = K_s_tr @ alpha
    var = np.maximum(k_ss - np.sum(V ** 2, axis=0), 0.0)
    return mu, var


# ============================================================
# Point generation
# ============================================================

def make_points(*, d: int, n_anchors: int, n_per_anchor: int, n_ood: int,
                eps: float, domain: tuple[float, float], seed: int
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clustered (train, id) sharing anchors + uniform ood."""
    rng = np.random.default_rng(seed)
    anchors = rng.uniform(domain[0], domain[1], size=(n_anchors, d))

    def cluster(n: int) -> np.ndarray:
        return np.concatenate(
            [a + rng.uniform(-eps, eps, size=(n, d)) for a in anchors]
        ).astype(np.float32)

    return (cluster(n_per_anchor),
            cluster(n_per_anchor),
            rng.uniform(domain[0], domain[1], size=(n_ood, d)).astype(np.float32))


# ============================================================
# Grid MLP baseline (1D only)
# ============================================================

def fit_grid_mlp(X_grid: np.ndarray, f_grid: np.ndarray, *, epochs: int,
                 batch_size: int, lr: float, weight_decay: float,
                 device: str, seed: int) -> torch.nn.Module:
    """Train a single MLP on (X_grid, f_grid) -- the dense noiseless GP draw."""
    torch.manual_seed(seed)
    model = EnsembleMLP(K=1, in_dim=X_grid.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    Xt = torch.from_numpy(X_grid).to(device)
    yt = torch.from_numpy(f_grid).to(device)
    n = len(X_grid)
    for _ in tqdm(range(epochs), leave=False):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            (model(Xt[idx]) - yt[idx]).pow(2).mean().backward()
            opt.step()
    return model


def predict_mlp(model: torch.nn.Module, X: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X).to(device)).squeeze(0).cpu().numpy().astype(np.float32)


# ============================================================
# I/O
# ============================================================

def to_df(X: np.ndarray, **extra) -> pd.DataFrame:
    return pd.DataFrame({**{f"x{i}": X[:, i] for i in range(X.shape[1])}, **extra})


def write_split(out_dir: Path, *, train, id_, ood, grid=None, predictions=None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    X_tr, y_tr = train
    X_id, y_id, mu_id, var_id = id_
    X_oo, y_oo, mu_oo, var_oo = ood
    dfs = {
        "train":    to_df(X_tr, y=y_tr),
        "eval_id":  to_df(X_id, y=y_id, mu_gp=mu_id, var_gp=var_id),
        "eval_ood": to_df(X_oo, y=y_oo, mu_gp=mu_oo, var_gp=var_oo),
    }
    if predictions is not None:
        for name, df in dfs.items():
            df["f"] = predictions[name]
    for name, df in dfs.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    if grid is not None:
        X_g, f_g, mu_g, var_g, prior_samples = grid
        extra = {f"prior_{k}": prior_samples[k] for k in range(len(prior_samples))}
        grid_df = to_df(X_g, f_true=f_g, mu_gp=mu_g, var_gp=var_g, **extra)
        if predictions is not None and "grid" in predictions:
            grid_df["f"] = predictions["grid"]
        grid_df.to_csv(out_dir / "grid.csv", index=False)
    stats = pd.DataFrame({f"{name}_{stat}": getattr(df, stat)()
                          for name, df in dfs.items()
                          for stat in ("mean", "var")})
    stats.index.name = "column"
    stats.to_csv(out_dir / "stats.csv")


# ============================================================
# Entry point
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",          default=None, help="output subdir under data/splits (default: synth_gp_<dim>_<kernel>_<length_scale>)")
    parser.add_argument("--kernel",        default="rbf", choices=KERNELS)
    parser.add_argument("--length-scale",  type=float, default=1.0)
    parser.add_argument("--sigma-f",       type=float, default=1.0, help="kernel output amplitude")
    parser.add_argument("--noise-std",     type=float, default=0.1)
    parser.add_argument("--dim",           type=int,   default=1, dest="d")
    parser.add_argument("--n-splits",      type=int,   default=10)
    parser.add_argument("--n-anchors",     type=int,   default=3)
    parser.add_argument("--n-per-anchor",  type=int,   default=50)
    parser.add_argument("--n-ood",         type=int,   default=250)
    parser.add_argument("--eps",           type=float, default=0.1)
    parser.add_argument("--domain",        type=float, nargs=2, default=[0.0, 10.0], metavar=("LO", "HI"))
    parser.add_argument("--n-grid",        type=int,   default=1000, help="dense 1D grid size (ignored for d > 1)")
    parser.add_argument("--n-prior-samples", type=int, default=10,   help="GP prior function draws added to grid.csv (1D only)")
    parser.add_argument("--base-seed",     type=int,   default=0,   help="seed offset for per-split anchors / GP draw")
    parser.add_argument("--mlp-epochs",      type=int,   default=500, help="MLP baseline: epochs (1D only, fit on grid)")
    parser.add_argument("--mlp-batch-size",  type=int,   default=256)
    parser.add_argument("--mlp-lr",          type=float, default=1e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument("--mlp-device",      default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()

    name = args.name or f"synth_gp_{args.d}_{args.kernel}_{args.length_scale:g}"
    out_root = SPLITS / name
    out_root.mkdir(parents=True, exist_ok=True)
    domain = (float(args.domain[0]), float(args.domain[1]))
    kernel = make_kernel(args.kernel, length_scale=args.length_scale, sigma_f=args.sigma_f)

    (out_root / "config.json").write_text(json.dumps({
        "kernel": args.kernel, "length_scale": args.length_scale, "sigma_f": args.sigma_f,
        "noise_std": args.noise_std, "dim": args.d, "domain": list(domain),
    }, indent=2))

    grid_X = (np.linspace(domain[0], domain[1], args.n_grid, dtype=np.float32)[:, None]
              if args.d == 1 else None)

    for s in range(args.n_splits):
        seed = args.base_seed + s
        X_tr, X_id, X_ood = make_points(
            d=args.d, n_anchors=args.n_anchors, n_per_anchor=args.n_per_anchor,
            n_ood=args.n_ood, eps=args.eps, domain=domain, seed=seed,
        )
        # Joint noiseless GP draw at every point we ever need labels for.
        blocks = [X_tr, X_id, X_ood] + ([grid_X] if grid_X is not None else [])
        offs = np.cumsum([0] + [len(b) for b in blocks])
        f_all = sample_prior(kernel, np.concatenate(blocks), seed=seed)
        f_tr, f_id, f_ood = f_all[offs[0]:offs[1]], f_all[offs[1]:offs[2]], f_all[offs[2]:offs[3]]

        noise = np.random.default_rng(seed + 10_000)
        y_tr  = (f_tr  + noise.normal(0, args.noise_std, len(f_tr))).astype(np.float32)
        y_id  = (f_id  + noise.normal(0, args.noise_std, len(f_id))).astype(np.float32)
        y_ood = (f_ood + noise.normal(0, args.noise_std, len(f_ood))).astype(np.float32)

        mu_id,  var_id  = gp_posterior(kernel, X_tr, y_tr, X_id,  noise_std=args.noise_std)
        mu_ood, var_ood = gp_posterior(kernel, X_tr, y_tr, X_ood, noise_std=args.noise_std)

        grid_pack = None
        predictions = None
        if grid_X is not None:
            f_grid = f_all[offs[3]:offs[4]].astype(np.float32)
            mu_g, var_g = gp_posterior(kernel, X_tr, y_tr, grid_X, noise_std=args.noise_std)
            prior_samples = sample_prior_batch(kernel, grid_X,
                                               n_samples=args.n_prior_samples,
                                               seed=seed + 1_000_000)
            grid_pack = (grid_X, f_grid,
                         mu_g.astype(np.float32), var_g.astype(np.float32),
                         prior_samples)
            mlp = fit_grid_mlp(grid_X, f_grid, epochs=args.mlp_epochs,
                               batch_size=args.mlp_batch_size, lr=args.mlp_lr,
                               weight_decay=args.mlp_weight_decay,
                               device=args.mlp_device, seed=seed)
            predictions = {
                "train":    predict_mlp(mlp, X_tr,    args.mlp_device),
                "eval_id":  predict_mlp(mlp, X_id,    args.mlp_device),
                "eval_ood": predict_mlp(mlp, X_ood,   args.mlp_device),
                "grid":     predict_mlp(mlp, grid_X,  args.mlp_device),
            }

        write_split(
            out_root / f"split{s:02d}",
            train=(X_tr, y_tr),
            id_=(X_id,  y_id,  mu_id.astype(np.float32),  var_id.astype(np.float32)),
            ood=(X_ood, y_ood, mu_ood.astype(np.float32), var_ood.astype(np.float32)),
            grid=grid_pack,
            predictions=predictions,
        )

    print(f"{name}: wrote {args.n_splits} splits to {out_root}")


if __name__ == "__main__":
    main()
