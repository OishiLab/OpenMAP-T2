"""Left/right hemisphere segmentation on coronal and axial views."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage

from utils.image import preprocess_intensity
from utils.pipeline.infer import DIRECTION_VALUE, forward_plane

N_CLASSES = 3
HEMI_DILATE = 5


def hemisphere(voxel: np.ndarray, hnet: nn.Module, device: torch.device, batch_size: int, use_amp: bool) -> np.ndarray:
    volume = preprocess_intensity(voxel)
    acc = torch.empty((N_CLASSES, *volume.shape), dtype=torch.float32)
    planes = (
        (volume.transpose(1, 2, 0), "coronal", (2, 0, 3, 1)),
        (volume.transpose(2, 1, 0), "axial", (3, 0, 2, 1)),
    )
    for k, (plane, name, inverse) in enumerate(planes):
        forward_plane(
            plane,
            model=hnet,
            device=device,
            n_classes=N_CLASSES,
            direction=DIRECTION_VALUE[name],
            batch_size=batch_size,
            use_amp=use_amp,
            binary=False,
            out_view=acc.permute(*inverse),
            accumulate=k > 0,
        )
    fused = torch.argmax(acc, dim=0).numpy().astype(np.int16)
    del acc
    if device.type == "cuda":
        torch.cuda.empty_cache()

    left = ndimage.binary_dilation(fused == 1, iterations=HEMI_DILATE).astype(np.int16)
    left[fused == 2] = 2
    right = ndimage.binary_dilation(left == 2, iterations=HEMI_DILATE).astype(np.int16) * 2
    right[left == 1] = 1
    return right.astype(np.uint16)
