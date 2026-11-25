"""Fuse hemisphere and 139-class labels into the 280-region map, then restore the 256³ grid."""

from __future__ import annotations

import os
import pickle

import numpy as np
from scipy import ndimage

from utils.pipeline.stripping import CROP

SPLIT_MAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "split_map.pkl")
_LUT: np.ndarray | None = None


def _lut() -> np.ndarray:
    global _LUT
    if _LUT is None:
        with open(SPLIT_MAP_PATH, "rb") as handle:
            split_map = pickle.load(handle)
        max_h = max(key[0] for key in split_map)
        max_p = max(key[1] for key in split_map)
        lut = np.zeros((max_h + 1, max_p + 1), dtype=np.int16)
        for (hemi, parc), value in split_map.items():
            lut[int(hemi), int(parc)] = int(value)
        _LUT = lut
    return _LUT


def _relabel_table(max_label: int) -> np.ndarray:
    """Lookup table for the sulcal/Sylvian relabeling: shift 251-280 up by 6, then merge in order."""
    lut = np.arange(max(max_label, 286) + 1, dtype=np.int16)
    lut[251:281] += 6
    for old, new in (
        (257, 252), (258, 254), (259, 256), (260, 258), (261, 259), (262, 260), (263, 262),
        (281, 251), (282, 253), (283, 255), (284, 257), (285, 261), (286, 263),
    ):  # fmt: skip
        lut[lut == old] = new
    return lut


def _relabel_segmentation(parc: np.ndarray) -> np.ndarray:
    return _relabel_table(int(parc.max()))[parc]


def _fuse_280(parcellated: np.ndarray, separated: np.ndarray) -> np.ndarray:
    """Map (hemisphere, 139-class label) to 274 regions, then split Sylvian/CSF to 280."""
    hemi = separated.astype(np.int16, copy=False)
    parc = parcellated.astype(np.int16, copy=False)
    lut = _lut()
    h = np.clip(hemi, 0, lut.shape[0] - 1)
    p = np.clip(parc, 0, lut.shape[1] - 1)
    output = lut[h, p]
    output = output * np.logical_or.reduce((hemi > 0, parc == 87, parc == 136))
    output = output.astype(np.int16)

    boundary = (hemi == 1) & ndimage.binary_dilation(hemi == 2, structure=np.ones((3, 3, 3)))
    coords = np.column_stack(np.where(boundary))
    if coords.shape[0] < 32:
        return _relabel_segmentation(output)

    centroid = coords.mean(axis=0)
    centered = coords - centroid
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, int(np.argmin(eigvals))]
    target = np.array([1.0, 0.0, 0.0])
    v = np.cross(normal, target)
    s = np.linalg.norm(v)
    c = float(np.dot(normal, target))
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)
    rot = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s**2 + 1e-8))

    mask_brain = output > 0
    coords_brain = np.column_stack(np.where(mask_brain))
    if coords_brain.shape[0] == 0:
        return _relabel_segmentation(output)
    centroid_brain = coords_brain.mean(axis=0)

    parc_rot = ndimage.affine_transform(output, rot, offset=centroid - rot @ centroid, order=0)
    parc_rot = ndimage.median_filter(parc_rot, size=3)
    cx_b, _, cz_b = rot @ (centroid_brain - centroid) + centroid
    grid_x, _, grid_z = np.meshgrid(
        np.arange(parc_rot.shape[0]),
        np.arange(parc_rot.shape[1]),
        np.arange(parc_rot.shape[2]),
        indexing="ij",
    )
    dx = grid_x - cx_b
    dz = grid_z - cz_b
    radius = np.sqrt(dx**2 + dz**2)
    mask_radius = (radius >= 5) & (radius <= 100)
    theta = np.arctan2(dx, dz)
    wedge = (theta >= np.deg2rad(-30)) & (theta <= np.deg2rad(30)) & mask_radius

    new_parc = parc_rot.copy()
    new_parc[wedge & (parc_rot == 250)] = 275
    new_parc[wedge & (parc_rot == 252)] = 277
    new_parc[wedge & (parc_rot == 256)] = 279
    new_parc[wedge & (parc_rot == 251)] = 276
    new_parc[wedge & (parc_rot == 253)] = 278
    new_parc[wedge & (parc_rot == 257)] = 280

    rot_inv = rot.T
    back = ndimage.affine_transform(new_parc, rot_inv, offset=centroid - rot_inv @ centroid, order=0)
    return _relabel_segmentation(back)


def uncrop_256(volume224: np.ndarray, shift: tuple[int, int, int]) -> np.ndarray:
    padded = np.pad(volume224, [(CROP, CROP)] * 3, mode="constant", constant_values=0)
    return np.roll(padded, (-shift[0], -shift[1], -shift[2]), axis=(0, 1, 2))


def postprocessing(parcellated: np.ndarray, separated: np.ndarray, shift: tuple[int, int, int]) -> np.ndarray:
    """Return the 280-label volume on the 256³ processing grid."""
    labels224 = _fuse_280(parcellated, separated)
    return uncrop_256(labels224, shift).astype(np.uint16)
