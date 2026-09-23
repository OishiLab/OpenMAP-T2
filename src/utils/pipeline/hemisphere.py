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
    kwargs = dict(
        model=hnet,
        device=device,
        n_classes=N_CLASSES,
        batch_size=batch_size,
        use_amp=use_amp,
        binary=False,
    )
    out_c = forward_plane(volume.transpose(1, 2, 0), direction=DIRECTION_VALUE["coronal"], **kwargs).permute(1, 3, 0, 2)
    out_a = forward_plane(volume.transpose(2, 1, 0), direction=DIRECTION_VALUE["axial"], **kwargs).permute(1, 3, 2, 0)
    fused = torch.argmax(out_c + out_a, dim=0).numpy().astype(np.int16)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    left = ndimage.binary_dilation(fused == 1, iterations=HEMI_DILATE).astype(np.int16)
    left[fused == 2] = 2
    right = ndimage.binary_dilation(left == 2, iterations=HEMI_DILATE).astype(np.int16) * 2
    right[left == 1] = 1
    return right.astype(np.uint16)
