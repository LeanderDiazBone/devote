"""Network architectures for the UQ benchmarks.

Two model families, both ensembled by construction (use K=1 for the
single-model case):
    EnsembleMLP        K parallel ReLU MLPs.
    EnsembleRFN        K parallel Randomized Fourier Networks (fixed RBF
                       random features + trainable linear head).

Ensembles store K members' parameters as a single tensor with leading axis K
and use a batched einsum so a forward/backward pass over the whole ensemble
costs roughly one matmul per layer instead of K.
"""
from __future__ import annotations
import math
import torch
from torch import nn


class EnsembleLinear(nn.Module):
    """K parallel `nn.Linear` layers as a single batched op.

    Accepts input of shape `[B, in_dim]` (broadcast across all K members) or
    `[K, B, in_dim]` (per-member inputs). Output is `[K, B, out_dim]`.
    Historical initialization applies Kaiming to [in_dim, out_dim] directly,
    so weight variance is 1/(3*out_dim), not nn.Linear's 1/(3*in_dim).
    This convention is retained for reproducibility of existing UQ runs.
    """
    def __init__(self, K: int, in_dim: int, out_dim: int):
        super().__init__()
        self.K = K
        self.weight = nn.Parameter(torch.empty(K, in_dim, out_dim))
        self.bias = nn.Parameter(torch.empty(K, out_dim))
        for k in range(K):
            nn.init.kaiming_uniform_(self.weight[k], a=math.sqrt(5))
        bound = 1.0 / math.sqrt(in_dim)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            out = torch.einsum("bi,kio->kbo", x, self.weight)
        else:
            out = torch.einsum("kbi,kio->kbo", x, self.weight)
        return out + self.bias.unsqueeze(1)


class EnsembleMLP(nn.Module):
    """K parallel ReLU MLPs, returning `[K, B]` predictions in one batched pass."""
    def __init__(self, K: int, in_dim: int, *, width: int = 100,
                 n_hidden: int = 2, out_dim: int = 1):
        super().__init__()
        self.K = K
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(n_hidden):
            layers += [EnsembleLinear(K, d, width), nn.ReLU()]
            d = width
        layers.append(EnsembleLinear(K, d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class EnsembleRFN(nn.Module):
    """K parallel RFNs. Each member has its own (frozen) random features and
    a trainable linear head; all K are evaluated in one batched pass."""
    def __init__(self, K: int, in_dim: int, *, n_features: int = 1024,
                 length_scale: float = 1.0, out_dim: int = 1):
        super().__init__()
        self.K = K
        self.register_buffer("omega",
                             torch.randn(K, in_dim, n_features) / length_scale)
        self.register_buffer("bias", torch.rand(K, n_features) * 2 * math.pi)
        self.scale = math.sqrt(2.0 / n_features)
        self.linear = EnsembleLinear(K, n_features, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = torch.einsum("bi,kio->kbo", x, self.omega) + self.bias.unsqueeze(1)
        phi = self.scale * torch.cos(proj)
        return self.linear(phi).squeeze(-1)
