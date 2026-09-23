"""Skull stripping on the 256³ grid, then histogram matching and a 224³ crop."""

from __future__ import annotations

import os

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from skimage.exposure import match_histograms

from utils.image import preprocess_intensity
from utils.output.output_space import save_mask_and_masked
from utils.pipeline.infer import forward_plane

CROP = 16
REFERENCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_intensity.npy")
_REFERENCE: np.ndarray | None = None


def reference_intensities() -> np.ndarray:
    global _REFERENCE
    if _REFERENCE is None:
        if not os.path.isfile(REFERENCE_PATH):
            raise FileNotFoundError(f"Histogram-matching reference is missing: {REFERENCE_PATH}")
        _REFERENCE = np.load(REFERENCE_PATH).astype(np.float32, copy=False)
    return _REFERENCE


def hist_match(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = image.copy()
    foreground = mask > 0
    reference = reference_intensities()
    if not np.any(foreground) or reference.size == 0:
        return out
    out[foreground] = match_histograms(image[foreground], reference, channel_axis=None).astype(np.float32)
    return out


def crop_224(volume: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Center-of-mass shift toward (128, 120, 128), then drop a 16-voxel border."""
    if not np.any(mask):
        return volume[CROP:-CROP, CROP:-CROP, CROP:-CROP].copy(), (0, 0, 0)
    x, y, z = (int(v) for v in ndimage.center_of_mass(mask.astype(np.uint8)))
    shift = (128 - x, 120 - y, 128 - z)
    rolled = np.roll(volume, shift, axis=(0, 1, 2))
    return rolled[CROP:-CROP, CROP:-CROP, CROP:-CROP].copy(), shift


def predict_stripping(
    volume: np.ndarray,
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    use_amp: bool,
) -> np.ndarray:
    voxel = preprocess_intensity(volume)
    kwargs = dict(
        model=model,
        device=device,
        n_classes=1,
        direction=None,
        batch_size=batch_size,
        use_amp=use_amp,
        binary=True,
    )
    out_c = forward_plane(voxel.transpose(1, 2, 0), **kwargs).permute(2, 0, 1)
    out_s = forward_plane(voxel, **kwargs)
    out_a = forward_plane(voxel.transpose(2, 1, 0), **kwargs).permute(2, 1, 0)
    mask = ((out_c + out_s + out_a) / 3.0) > 0.5
    return mask.numpy().astype(np.uint8)


def stripping(
    output_dir: str,
    basename: str,
    odata: nib.Nifti1Image,
    data: nib.Nifti1Image,
    ssnet: nn.Module,
    device: torch.device,
    output_ext: str,
    output_space: str,
    batch_size: int,
    use_amp: bool,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Return the histogram-matched 224³ brain and the shift used to crop it."""
    volume = np.ascontiguousarray(np.asanyarray(data.dataobj).astype(np.float32, copy=False))
    mask = predict_stripping(volume, ssnet, device, batch_size, use_amp)
    stripped = volume * mask.astype(np.float32)
    save_mask_and_masked(output_dir, basename, "stripped", mask, odata, data, output_ext, output_space)
    matched = hist_match(stripped, mask)
    return crop_224(matched, mask)
