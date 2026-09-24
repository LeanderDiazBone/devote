"""Generic state-coverage tracking via uniform binning over a low-dim projection.

Environments expose a `get_coverage_geometry()` returning a dict consumed by
`make_coverage`. The dict describes a D-dim box (`bounds`, `bins`) plus a
`project` callable that maps a transition dict to a D-dim point. An optional
`valid_mask` lets envs exclude bins from the denominator (e.g. maze walls).
"""

import numpy as np


class CoverageTracker:

  def __init__(self, bounds, bins, axis_names=None, valid_mask=None):
    self.bounds = np.asarray(bounds, dtype=np.float64).reshape(-1, 2)
    self.bins = tuple(int(b) for b in bins)
    self.ndim = len(self.bins)
    assert self.bounds.shape == (self.ndim, 2), (self.bounds.shape, self.bins)
    self.axis_names = (
        tuple(axis_names) if axis_names is not None
        else tuple(f'x{i}' for i in range(self.ndim)))
    if valid_mask is None:
      valid_mask = np.ones(self.bins, dtype=bool)
    self.valid_mask = np.asarray(valid_mask, dtype=bool)
    assert self.valid_mask.shape == self.bins, (
        self.valid_mask.shape, self.bins)
    self.counts = np.zeros(self.bins, dtype=np.int64)

  def add(self, point):
    """Add points and return their flattened coverage-bin IDs."""
    idx = self.bin_indices(point)
    flat = idx.reshape(-1, self.ndim)
    np.add.at(self.counts, tuple(flat[:, d] for d in range(self.ndim)), 1)
    return self._flatten_indices(idx)

  def bin_indices(self, point):
    """Map points of shape ``(..., ndim)`` to clipped bin coordinates."""
    point = np.asarray(point, dtype=np.float64)
    leading = point.shape[:-1]
    pt = point.reshape(-1, self.ndim)
    low = self.bounds[:, 0]
    span = np.maximum(self.bounds[:, 1] - low, 1e-9)
    rel = (pt - low) / span
    idx = np.floor(rel * np.array(self.bins)).astype(np.int64)
    for d in range(self.ndim):
      idx[:, d] = np.clip(idx[:, d], 0, self.bins[d] - 1)
    return idx.reshape(leading + (self.ndim,))

  def bin_ids(self, point):
    """Map points of shape ``(..., ndim)`` to flattened bin IDs."""
    return self._flatten_indices(self.bin_indices(point))

  def _flatten_indices(self, idx):
    leading = idx.shape[:-1]
    flat = idx.reshape(-1, self.ndim)
    ids = np.ravel_multi_index(tuple(flat[:, d] for d in range(self.ndim)), self.bins)
    return ids.reshape(leading).astype(np.int32)

  def stats(self):
    valid = self.counts[self.valid_mask]
    n_valid = int(valid.size)
    n_visited = int((valid > 0).sum())
    n_visited_min10 = int((valid >= 10).sum())
    total = float(valid.sum())
    if total > 0 and n_valid > 0:
      p = valid.astype(np.float64) / total
      mask = p > 0
      entropy = float(-(p[mask] * np.log(p[mask])).sum())
    else:
      entropy = 0.0
    max_entropy = float(np.log(n_valid)) if n_valid > 0 else 0.0
    return {
        'visited_cells': float(n_visited),
        'visited_fraction': float(n_visited / max(1, n_valid)),
        'visited_cells_min10': float(n_visited_min10),
        'visited_fraction_min10': float(n_visited_min10 / max(1, n_valid)),
        'open_cells': float(n_valid),
        'entropy': entropy,
        'entropy_normalized':
            float(entropy / max_entropy) if max_entropy > 0 else 0.0,
        'total_visits': total,
    }

  def heatmap(self):
    """2D only: counts as (n_axis1, n_axis0) image with axis 1 top-to-bottom
    so high values of the second axis render at the top."""
    if self.ndim != 2:
      raise NotImplementedError('heatmap only available for 2D coverage')
    h = self.counts.astype(np.float32)
    h[~self.valid_mask] = np.nan
    return np.flip(h.T, axis=0)

  def reset(self):
    self.counts[...] = 0


def make_coverage(geom):
  """Build a CoverageTracker from an env-supplied geometry dict."""
  return CoverageTracker(
      bounds=geom['bounds'],
      bins=geom['bins'],
      axis_names=geom.get('axis_names'),
      valid_mask=geom.get('valid_mask'),
  )
