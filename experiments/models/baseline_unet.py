"""Baseline U-Net model for spectrogram masking."""

from __future__ import annotations

import torch
import torch.nn as nn


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class BaselineUNet(nn.Module):
    def __init__(self, n_class: int):
        super().__init__()
        self.enc1 = conv_block(1, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = conv_block(32, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = conv_block(64, 32)
        self.upconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec2 = conv_block(32, 16)
        self.outconv = nn.Conv2d(16, n_class, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        b = self.bottleneck(p2)
        u1 = self.upconv1(b)
        e2 = e2[:, :, : u1.size(2), : u1.size(3)]
        d1 = self.dec1(torch.cat([u1, e2], dim=1))
        u2 = self.upconv2(d1)
        e1 = e1[:, :, : u2.size(2), : u2.size(3)]
        d2 = self.dec2(torch.cat([u2, e1], dim=1))
        return torch.sigmoid(self.outconv(d2))

