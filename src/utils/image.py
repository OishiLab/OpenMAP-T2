"""Intensity preprocessing used by the trained OpenMAP-T2 networks."""

from __future__ import annotations

import numpy as np

CLIP_STD = 3.0


def preprocess_intensity(voxel: np.ndarray, clip_std: float = CLIP_STD) -> np.ndarray:
    """Clip positive-foreground outliers, z-score, then map to [-1, 1]."""
    voxel = np.asarray(voxel, dtype=np.float32)
    nonzero = voxel[voxel > 0]
    if nonzero.size:
        upper = float(nonzero.mean() + clip_std * nonzero.std())
        voxel = np.clip(voxel, 0.0, upper)
        mu = float(nonzero.mean())
        sigma = float(nonzero.std())
        sigma = sigma if sigma > 1e-6 else 1.0
        voxel = (voxel - mu) / sigma
    vmin = float(voxel.min())
    vmax = float(voxel.max())
    if vmax > vmin:
        voxel = (voxel - vmin) / (vmax - vmin)
    else:
        voxel = np.zeros_like(voxel)
    return (voxel * 2.0 - 1.0).astype(np.float32)
