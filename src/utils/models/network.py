"""2D U-Net used by OpenMAP-T2."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.normal_(self.conv1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.conv2.weight, mean=0.0, std=0.02)
        self.batchnorm1 = nn.BatchNorm2d(ch_out)
        self.batchnorm2 = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.batchnorm1(self.conv1(x)))
        return self.relu(self.batchnorm2(self.conv2(h)))


class EncodeBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int) -> None:
        super().__init__()
        self.conv = ConvBlock(ch_in, ch_out)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)
        return self.pool(skip), skip


class DecodeBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2, padding=0, bias=True)
        self.conv = ConvBlock(ch_out * 2, ch_out)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        h = self.up(x)
        return self.conv(torch.cat([h, skip], dim=1))


class UNet(nn.Module):
    def __init__(self, ch_in: int, ch_out: int) -> None:
        super().__init__()
        self.econv0 = nn.Conv2d(ch_in, 64, kernel_size=1, stride=1, padding=0, bias=True)
        nn.init.normal_(self.econv0.weight, mean=0.0, std=0.02)

        self.econv1 = EncodeBlock(64, 64)
        self.econv2 = EncodeBlock(64, 128)
        self.econv3 = EncodeBlock(128, 256)
        self.econv4 = EncodeBlock(256, 512)
        self.bottle = ConvBlock(512, 1024)
        self.dconv4 = DecodeBlock(1024, 512)
        self.dconv3 = DecodeBlock(512, 256)
        self.dconv2 = DecodeBlock(256, 128)
        self.dconv1 = DecodeBlock(128, 64)

        self.dconv0 = nn.Conv2d(64, ch_out, kernel_size=1, stride=1, padding=0, bias=True)
        nn.init.normal_(self.dconv0.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.econv0(x)
        x, skip1 = self.econv1(x)
        x, skip2 = self.econv2(x)
        x, skip3 = self.econv3(x)
        x, skip4 = self.econv4(x)
        x = self.bottle(x)
        x = self.dconv4(x, skip4)
        x = self.dconv3(x, skip3)
        x = self.dconv2(x, skip2)
        x = self.dconv1(x, skip1)
        return self.dconv0(x)
