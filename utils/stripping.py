import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from utils.functions import normalize, hist_matching

def strip(voxel, model, device):
    """
    Applies a given model to a 3D voxel array and returns the processed output.

    Args:
        voxel (numpy.ndarray): A 3D numpy array of shape (224, 224, 224) representing the input voxel data.
        model (torch.nn.Module): A PyTorch model to be used for processing the voxel data.
        device (torch.device): The device (CPU or GPU) on which the model and data should be loaded.

    Returns:
        torch.Tensor: A 3D tensor of shape (224, 224, 224) containing the processed output.
    """
    # Set the model to evaluation mode
    model.eval()
    voxel = np.pad(voxel, [(1, 1), (0, 0), (0, 0)], "constant", constant_values=voxel.min())

    # Disable gradient calculation for inference
    with torch.inference_mode():
        # Initialize an empty tensor to store the output
        output = torch.zeros(224, 224, 224).to(device)

        # Iterate over each slice in the voxel data
        for i in range(1, 224 + 1): # for i, v in enumerate(voxel):
            # Reshape the slice to match the model's input dimensions
            image = np.stack([voxel[i - 1], voxel[i], voxel[i + 1]])

            # Convert the numpy array to a PyTorch tensor and move it to the specified device
            image = torch.tensor(image.reshape(1, 3, 224, 224))
            image = image.to(device)

            # Apply the model to the input image and apply the sigmoid activation function
            x_out = F.sigmoid(model(image)).detach()

            # Store the output in the corresponding slice of the output tensor
            output[i - 1] = x_out
            
        # Reshape the output tensor to the original voxel dimensions and return it
        return output.reshape(224, 224, 224)


def stripping(data, ssnet, device):
    """
    Perform brain stripping on a given voxel using a specified neural network.

    This function normalizes the input voxel, applies brain stripping in three anatomical planes
    (coronal, sagittal, and axial), and combines the results to produce a final stripped brain image.
    The stripped image is then centered and cropped.

    Args:
        voxel (numpy.ndarray): The input 3D voxel data to be stripped.
        data (nibabel.Nifti1Image): The original neuroimaging data.
        ssnet (torch.nn.Module): The neural network model used for brain stripping.
        device (torch.device): The device on which the neural network model is loaded (e.g., CPU or GPU).

    Returns:
        tuple: A tuple containing:
            - stripped (numpy.ndarray): The stripped and processed brain image.
            - (xd, yd, zd) (tuple of int): The shifts applied to center the brain image in the x, y, and z directions.
    """
    # Normalize the input voxel data
    voxel = data.get_fdata().astype(np.float32)
    voxel = normalize(voxel)

    # Prepare the voxel data in three anatomical planes: coronal, sagittal, and axial
    coronal = voxel.transpose(1, 2, 0)
    sagittal = voxel
    axial = voxel.transpose(2, 1, 0)

    # Apply the brain stripping model to each plane
    out_c = strip(coronal, ssnet, device).permute(2, 0, 1)
    out_s = strip(sagittal, ssnet, device)
    out_a = strip(axial, ssnet, device).permute(2, 1, 0)

    # Combine the results from the three planes and threshold the output
    out_e = ((out_c + out_s + out_a) / 3) > 0.5
    out_e = out_e.cpu().numpy().astype(np.uint16)

    # Multiply the original data by the thresholded output to get the stripped brain image
    stripped = data.get_fdata().astype(np.float32) * out_e

    stripped = hist_matching(stripped, out_e.astype(bool))

    # Calculate the center of mass of the stripped brain image
    x, y, z = map(int, ndimage.center_of_mass(out_e))

    # Calculate the shifts needed to center the brain image
    xd = 112 - x
    yd = 105 - y
    zd = 112 - z

    # Apply the shifts to center the brain image
    stripped = np.roll(stripped, (xd, yd, zd), axis=(0, 1, 2))

    # Crop the centered brain image
    # stripped = stripped[10:-10, 10:-10, 10:-10]

    # Return the stripped brain image and the shifts applied
    return normalize(stripped), (xd, yd, zd), out_e