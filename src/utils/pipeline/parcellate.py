"""139-class parcellation with one network and a direction channel."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from utils.image import preprocess_intensity
from utils.pipeline.infer import DIRECTION_VALUE, forward_plane

N_CLASSES = 139


def _accumulate(volume: np.ndarray, model: nn.Module, device: torch.device, batch_size: int, use_amp: bool) -> np.ndarray:
    voxel = preprocess_intensity(volume)
    kwargs = dict(
        model=model,
        device=device,
        n_classes=N_CLASSES,
        batch_size=batch_size,
        use_amp=use_amp,
        binary=False,
    )
    out_c = forward_plane(voxel.transpose(1, 2, 0), direction=DIRECTION_VALUE["coronal"], **kwargs).permute(1, 3, 0, 2)
    out_s = forward_plane(voxel, direction=DIRECTION_VALUE["sagittal"], **kwargs).permute(1, 0, 2, 3)
    acc = out_c + out_s
    del out_c, out_s
    out_a = forward_plane(voxel.transpose(2, 1, 0), direction=DIRECTION_VALUE["axial"], **kwargs).permute(1, 3, 2, 0)
    acc = acc + out_a
    del out_a
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.argmax(acc, dim=0).numpy().astype(np.uint16)


def parcellation(voxel: np.ndarray, pnet: nn.Module, device: torch.device, batch_size: int, use_amp: bool) -> np.ndarray:
    return _accumulate(voxel, pnet, device, batch_size, use_amp)
