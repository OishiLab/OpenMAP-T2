"""Load the three OpenMAP-T2 weight files."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn

from utils.models.network import UNet

# Public layout:  MODEL_FOLDER/{SSNet,PNet,HNet}/*.pth
# Research layout: MODEL_FOLDER/{stripping,parcellation,hemisphere}/models/model.pth
PHASES = (
    ("stripping", "SSNet", "SSNet.pth", 3, 1),
    ("parcellation", "PNet", "PNet.pth", 4, 139),
    ("hemisphere", "HNet", "HNet.pth", 4, 3),
)


def _resolve_weight(model_dir: Path, phase: str, folder: str, filename: str) -> Path:
    direct = model_dir / folder / filename
    if direct.is_file():
        return direct

    models = model_dir / phase / "models"
    final = models / "model.pth"
    if final.is_file():
        return final
    checkpoint = models / "checkpoint.pt"
    if checkpoint.is_file():
        return checkpoint
    epochs = sorted(models.glob("epoch_*.pth"))
    if epochs:
        return epochs[-1]
    raise FileNotFoundError(
        f"No weights for {phase}. Expected {direct} or {models / 'model.pth'}."
    )


def _read_state_dict(path: Path, device: torch.device) -> dict:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and isinstance(ckpt.get("model"), dict):
        return ckpt["model"]
    return ckpt


def _load_one(model_dir: Path, phase: str, folder: str, filename: str, ch_in: int, ch_out: int, device: torch.device) -> nn.Module:
    path = _resolve_weight(model_dir, phase, folder, filename)
    model = UNet(ch_in, ch_out)
    model.load_state_dict(_read_state_dict(path, device))
    model.to(device)
    model.eval()
    print(f"{phase}: {path}")
    return model


def load_model(opt, device: torch.device) -> tuple[nn.Module, nn.Module, nn.Module]:
    """Return (ssnet, pnet, hnet) on `device` in eval mode."""
    model_dir = Path(os.path.abspath(opt.m))
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model folder does not exist: {model_dir}")
    ssnet, pnet, hnet = (
        _load_one(model_dir, phase, folder, filename, ch_in, ch_out, device)
        for phase, folder, filename, ch_in, ch_out in PHASES
    )
    return ssnet, pnet, hnet
