import torch
import pickle
import numpy as np
from scipy import ndimage

def insert_label(seg, sn, fn):
    new_seg = seg.copy()
    for num in reversed(range(sn, fn)):
        new_seg[seg == num] = num + 6
    return new_seg

def relabel_segmentation(parc):
    seg = parc.copy()
    seg = insert_label(seg, 251, 281)
    seg[seg == 257] = 252
    seg[seg == 258] = 254
    seg[seg == 259] = 256
    seg[seg == 260] = 258
    seg[seg == 261] = 259
    seg[seg == 262] = 260
    seg[seg == 263] = 262
    
    seg[seg == 281] = 251
    seg[seg == 282] = 253
    seg[seg == 283] = 255
    seg[seg == 284] = 257
    seg[seg == 285] = 261
    seg[seg == 286] = 263
    return seg

def postprocessing(parcellated, separated, shift, device):
    with open("level/split_map.pkl", "rb") as tf:
        dictionary = pickle.load(tf)

    pmap = torch.tensor(parcellated.astype("int16"), requires_grad=False).to(device)
    hmap = torch.tensor(separated.astype("int16"), requires_grad=False).to(device)
    combined = torch.stack((torch.flatten(hmap), torch.flatten(pmap)), axis=-1)
    output = torch.zeros_like(hmap).ravel()
    for key, value in dictionary.items():
        key = torch.tensor(key, requires_grad=False).to(device)
        mask = torch.all(combined == key, axis=1)
        output[mask] = value
    output = output.reshape(hmap.shape)
    output = output.cpu().detach().numpy()
    output = output * (np.logical_or(np.logical_or(separated > 0, parcellated == 87), parcellated == 136))
    parc = output.astype("int16")
    hemi = separated.astype("int16")

    boundary = (hemi == 1) & (ndimage.binary_dilation(hemi == 2, structure=np.ones((3,3,3))))
    coords = np.column_stack(np.where(boundary))
    centroid = coords.mean(axis=0)
    coords_centered = coords - centroid
    cov = np.cov(coords_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, np.argmin(eigvals)]
    target = np.array([1, 0, 0], dtype=float)
    v = np.cross(normal, target)
    s = np.linalg.norm(v)
    c = np.dot(normal, target)
    vx = np.array([[0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]])
    R = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s**2 + 1e-8))

    mask_brain = parc > 0
    coords_brain = np.column_stack(np.where(mask_brain))
    centroid_brain = coords_brain.mean(axis=0)

    parc_rot = ndimage.affine_transform(parc, R, offset=centroid - R @ centroid, order=0)
    parc_rot = ndimage.median_filter(parc_rot, size=3)
    centroid_brain_rot = R @ (centroid_brain - centroid) + centroid
    cx_b, cy_b, cz_b = centroid_brain_rot

    X, Y, Z = np.meshgrid(np.arange(parc_rot.shape[0]), np.arange(parc_rot.shape[1]), np.arange(parc_rot.shape[2]), indexing="ij")
    dx = X - cx_b
    dz = Z - cz_b

    r = np.sqrt(dx**2 + dz**2)
    r_min = 5
    r_max = 100
    mask_radius = (r >= r_min) & (r <= r_max)

    theta = np.arctan2(dx, dz)
    deg = np.deg2rad
    mask_neg30_to_0 = (theta >= -deg(30)) & (theta <= deg(30))
    mask_0_to_pos30 = (theta >= -deg(30)) & (theta <= deg(30))
    mask_neg30_to_0 = mask_neg30_to_0 & mask_radius
    mask_0_to_pos30 = mask_0_to_pos30 & mask_radius

    new_parc_rot = parc_rot.copy()
    # [-30°, 0°]：250 -> 275, 252 -> 277. 256 -> 279
    mask_A = mask_neg30_to_0 & (parc_rot == 250)
    new_parc_rot[mask_A] = 275
    mask_A = mask_neg30_to_0 & (parc_rot == 252)
    new_parc_rot[mask_A] = 277
    mask_A = mask_neg30_to_0 & (parc_rot == 256)
    new_parc_rot[mask_A] = 279
    # [0°, +30°]：251 -> 276, 253 -> 278, 257 -> 280
    mask_B = mask_0_to_pos30 & (parc_rot == 251)
    new_parc_rot[mask_B] = 276
    mask_B = mask_0_to_pos30 & (parc_rot == 253)
    new_parc_rot[mask_B] = 278
    mask_B = mask_0_to_pos30 & (parc_rot == 257)
    new_parc_rot[mask_B] = 280

    R_inv = R.T
    new_parc_back = ndimage.affine_transform(new_parc_rot, R_inv, offset=centroid - R_inv @ centroid, order=0)
    output = relabel_segmentation(new_parc_back)
    output = np.pad(output, [(16, 16), (16, 16), (16, 16)], "constant", constant_values=0)
    output = np.roll(output, (-shift[0], -shift[1], -shift[2]), axis=(0, 1, 2))
    return output
