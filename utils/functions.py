import json
import numpy as np
from skimage.exposure import match_histograms

def normalize(voxel):
    nonzero = voxel[voxel > 0]
    voxel = np.clip(voxel, 0, np.mean(nonzero) + np.std(nonzero) * 2)
    voxel = (voxel - np.min(voxel)) / (np.max(voxel) - np.min(voxel))
    voxel = (voxel * 2) - 1
    return voxel.astype("float32")

def hist_matching(img, mask):
    ref_json = 'utils/reference_intensity.json'
    with open(ref_json, 'r') as f:
        ref_dict = json.load(f)
    ref_intensity = np.array(ref_dict["intensity"])

    matched = img.copy()
    matched[mask] = match_histograms(img[mask], ref_intensity, channel_axis=None)
    return matched