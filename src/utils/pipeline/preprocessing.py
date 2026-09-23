"""N4 bias-field correction and resampling onto the 1 mm 256³ grid."""

from __future__ import annotations

import os

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from nibabel import processing

GRID = (256, 256, 256)
VOXEL_MM = (1.0, 1.0, 1.0)


def N4_Bias_Field_Correction(input_path: str, output_path: str) -> None:
    raw = sitk.ReadImage(input_path, sitk.sitkFloat32)
    head_mask = sitk.LiThreshold(sitk.RescaleIntensity(raw, 0, 255), 0, 1)
    shrink = 4
    dim = raw.GetDimension()
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.Execute(sitk.Shrink(raw, [shrink] * dim), sitk.Shrink(head_mask, [shrink] * dim))
    corrected = sitk.Cast(raw / sitk.Exp(corrector.GetLogBiasFieldAsImage(raw)), sitk.sitkFloat32)
    sitk.WriteImage(corrected, output_path)


def _to_processing_grid(odata: nib.Nifti1Image) -> nib.Nifti1Image:
    array = np.asanyarray(odata.dataobj)
    zooms = tuple(float(z) for z in odata.header.get_zooms()[:3])
    already = tuple(array.shape) == GRID and np.allclose(zooms, VOXEL_MM)
    if already:
        data = np.ascontiguousarray(array.astype(np.float32, copy=False))
        image = nib.Nifti1Image(data, odata.affine)
        image.set_data_dtype(np.float32)
        return image
    return processing.conform(odata, out_shape=GRID, voxel_size=VOXEL_MM, order=1)


def preprocessing(ipath: str, output_dir: str, basename: str, output_ext: str = ".nii.gz"):
    """Return the canonical N4 image and the 1 mm 256³ image used by the networks."""
    opath = os.path.join(output_dir, f"original/{basename}_N4{output_ext}")
    try:
        N4_Bias_Field_Correction(ipath, opath)
        odata = nib.squeeze_image(nib.as_closest_canonical(nib.load(opath)))
    except Exception as exc:
        print(f"N4 failed for {basename}: {exc}; using the input image")
        odata = nib.squeeze_image(nib.as_closest_canonical(nib.load(ipath)))
    data = _to_processing_grid(odata)
    return odata, data
