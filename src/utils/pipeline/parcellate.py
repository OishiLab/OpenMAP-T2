"""139-class parcellation with one network and a direction channel."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from utils.image import preprocess_intensity
from utils.pipeline.infer import DIRECTION_VALUE, forward_plane

N_CLASSES = 139


def parcellation(voxel: np.ndarray, pnet: nn.Module, device: torch.device, batch_size: int, use_amp: bool) -> np.ndarray:
    """Sum coronal, sagittal, and axial probabilities, then take the argmax label."""
    volume = preprocess_intensity(voxel)
    # One (n_classes, X, Y, Z) accumulator; each plane writes straight into a permuted view of it,
    # so no per-plane full-volume tensor is allocated (peak memory drops from ~3x to ~1x).
    acc = torch.empty((N_CLASSES, *volume.shape), dtype=torch.float32)
    planes = (
        (volume.transpose(1, 2, 0), "coronal", (2, 0, 3, 1)),
        (volume, "sagittal", (1, 0, 2, 3)),
        (volume.transpose(2, 1, 0), "axial", (3, 0, 2, 1)),
    )
    for k, (plane, name, inverse) in enumerate(planes):
        forward_plane(
            plane,
            model=pnet,
            device=device,
            n_classes=N_CLASSES,
            direction=DIRECTION_VALUE[name],
            batch_size=batch_size,
            use_amp=use_amp,
            binary=False,
            out_view=acc.permute(*inverse),
            accumulate=k > 0,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return torch.argmax(acc, dim=0).numpy().astype(np.uint16)
