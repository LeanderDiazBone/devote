"""Download UCI/OpenML regression datasets used by the UQ benchmarks.

Writes one CSV per dataset under `data/base/` (raw float32, last column `y`)
and a 1st/99th-percentile-clipped copy under `data/normalized/`. The
normalized file also carries a column `f` with the prediction of a single
MLP fit on the full normalized data; this baseline propagates through
`generate_splits.py` into every split partition.
"""
from __future__ import annotations
import argparse
import io
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

if __package__:
    from .networks import EnsembleMLP
else:
    from networks import EnsembleMLP

HERE = Path(__file__).parent
BASE = HERE / "data" / "base"
NORM = HERE / "data" / "normalized"

URLS = {
    "yacht":    "https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data",
    "energy":   "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
    "concrete": "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls",
    "wine":     "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    "power":    "https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip",
    "naval":    "https://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI%20CBM%20Dataset.zip",
    "protein":  "https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv",
}

DATASETS = [*URLS, "kin8nm"]


def _zip_open(url: str, name_contains: str):
    """Return a file-like handle to the first member of `url` whose name contains `name_contains`."""
    with urllib.request.urlopen(url) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    return z.open(next(n for n in z.namelist() if name_contains in n.lower()))


def fetch(name: str) -> pd.DataFrame:
    """Download `name` and return a DataFrame whose last column is the target."""
    if name == "yacht":
        df = pd.read_csv(URLS[name], sep=r"\s+", header=None).dropna()
    elif name == "energy":
        # Energy efficiency has two targets (Y1 heating, Y2 cooling); keep Y1.
        df = pd.read_excel(URLS[name]).dropna(axis=1, how="all").dropna().iloc[:, :9]
    elif name == "concrete":
        df = pd.read_excel(URLS[name])
    elif name == "wine":
        df = pd.read_csv(URLS[name], sep=";")
    elif name == "power":
        # Combined Cycle Power Plant: 4 features + PE (target). The xlsx has
        # 5 sheets that are independent shuffles of the same data; use sheet 0.
        df = pd.read_excel(_zip_open(URLS[name], "folds5x2"), sheet_name=0)
    elif name == "naval":
        # Naval Propulsion CBM: 16 features (two are constants) + 2 targets.
        # Drop the constant features and keep the first target (GT compressor
        # decay state coefficient), per the standard UQ-benchmark convention.
        df = pd.read_csv(_zip_open(URLS[name], "data.txt"), sep=r"\s+", header=None)
        df = df.loc[:, df.nunique() > 1].iloc[:, :-1]
    elif name == "protein":
        # CASP: target RMSD is the first column; move it to last.
        df = pd.read_csv(URLS[name])
        df = df[[c for c in df.columns if c != "RMSD"] + ["RMSD"]]
    elif name == "kin8nm":
        from sklearn.datasets import fetch_openml
        ds = fetch_openml(name="kin8nm", version=1, as_frame=True, parser="auto")
        df = pd.concat([ds.data, ds.target.rename("y")], axis=1)
    else:
        raise ValueError(f"unknown dataset {name!r}")
    df.columns = [*[f"x{i}" for i in range(df.shape[1] - 1)], "y"]
    return df.astype(np.float32).reset_index(drop=True)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column affine rescale to [0, 1] using 1st/99th percentiles, clipped."""
    p1, p99 = df.quantile(0.01), df.quantile(0.99)
    return ((df - p1) / np.maximum(p99 - p1, 1e-6)).clip(0.0, 1.0).astype(np.float32)


def fit_full_mlp(df: pd.DataFrame, *, epochs: int, batch_size: int, lr: float,
                 weight_decay: float, device: str, seed: int) -> np.ndarray:
    """Train a single MLP on the full (x*, y) and return its predictions."""
    torch.manual_seed(seed)
    X = df[[c for c in df.columns if c.startswith("x")]].to_numpy(np.float32)
    y = df["y"].to_numpy(np.float32)
    model = EnsembleMLP(K=1, in_dim=X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    Xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y).to(device)
    n = len(X)
    for _ in tqdm(range(epochs), leave=False):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            (model(Xt[idx]) - yt[idx]).pow(2).mean().backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return model(Xt).squeeze(0).cpu().numpy().astype(np.float32)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS, help="subset to download (default: all)")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--fit-seed", type=int, default=0)
    args = parser.parse_args(argv)

    BASE.mkdir(parents=True, exist_ok=True)
    NORM.mkdir(parents=True, exist_ok=True)
    fit_kwargs = dict(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                      weight_decay=args.weight_decay, device=args.device,
                      seed=args.fit_seed)
    for name in args.datasets:
        df = fetch(name)
        df.to_csv(BASE / f"{name}.csv", index=False)
        norm = normalize(df)
        norm["f"] = fit_full_mlp(norm, **fit_kwargs)
        norm.to_csv(NORM / f"{name}.csv", index=False)
        rmse = float(np.sqrt(np.mean((norm["f"] - norm["y"]) ** 2)))
        print(f"{name}: {df.shape[0]} rows, {df.shape[1] - 1} features, full-data rmse={rmse:.4f}")


if __name__ == "__main__":
    main()
