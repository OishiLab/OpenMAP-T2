import os
import torch
from utils.network import UNet

def load_model(opt, device):
    """
    This function loads multiple pre-trained models and sets them to evaluation mode.
    The models loaded are:
    1. SSNet: Another U-Net model for a different task.
    2. PNet coronal: A U-Net model for coronal plane predictions.
    3. PNet sagittal: A U-Net model for sagittal plane predictions.
    4. PNet axial: A U-Net model for axial plane predictions.
    5. HNet coronal: A U-Net model for coronal plane predictions with different input/output channels.
    6. HNet axial: A U-Net model for axial plane predictions with different input/output channels.

    Parameters:
    opt (object): An options object containing model paths.
    device (torch.device): The device on which to load the models (CPU or GPU).

    Returns:
    tuple: A tuple containing all the loaded models.
    """

    # Load SSNet model
    ssnet = UNet(3, 1)
    ssnet.load_state_dict(torch.load(os.path.join(opt.m, "SSNet/SSNet.pth"), weights_only=True))
    ssnet.to(device)
    ssnet.eval()

    # Load PNet coronal model
    pnet_c = UNet(3, 139)
    pnet_c.load_state_dict(torch.load(os.path.join(opt.m, "PNet/coronal.pth"), weights_only=True))
    pnet_c.to(device)
    pnet_c.eval()

    # Load PNet sagittal model
    pnet_s = UNet(3, 139)
    pnet_s.load_state_dict(torch.load(os.path.join(opt.m, "PNet/sagittal.pth"), weights_only=True))
    pnet_s.to(device)
    pnet_s.eval()

    # Load PNet axial model
    pnet_a = UNet(3, 139)
    pnet_a.load_state_dict(torch.load(os.path.join(opt.m, "PNet/axial.pth"), weights_only=True))
    pnet_a.to(device)
    pnet_a.eval()

    # Load HNet coronal model
    hnet_c = UNet(1, 3)
    hnet_c.load_state_dict(torch.load(os.path.join(opt.m, "HNet/coronal.pth"), weights_only=True))
    hnet_c.to(device)
    hnet_c.eval()

    # Load HNet axial model
    hnet_a = UNet(1, 3)
    hnet_a.load_state_dict(torch.load(os.path.join(opt.m, "HNet/axial.pth"), weights_only=True))
    hnet_a.to(device)
    hnet_a.eval()

    # Return all loaded models
    return ssnet, pnet_c, pnet_s, pnet_a, hnet_c, hnet_a