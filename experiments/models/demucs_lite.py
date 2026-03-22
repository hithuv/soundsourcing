"""Small hybrid Demucs-like model for class-wise separation."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        # k=8, p=3, stride=1 shortens length by 1, breaking residual adds; k=7 preserves length.
        k1, p1 = (7, 3) if stride == 1 else (8, 3)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=k1, stride=stride, padding=p1),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
        )
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride) if (
            in_channels != out_channels or stride != 1
        ) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)


class DemucsLite(nn.Module):
    def __init__(self, num_classes: int, base_channels: int = 32):
        super().__init__()
        self.num_classes = num_classes
        self.enc1 = ConvBlock1d(1, base_channels, stride=2)
        self.enc2 = ConvBlock1d(base_channels, base_channels * 2, stride=2)
        self.enc3 = ConvBlock1d(base_channels * 2, base_channels * 4, stride=2)
        self.bottleneck = ConvBlock1d(base_channels * 4, base_channels * 4, stride=1)
        self.up3 = nn.ConvTranspose1d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.dec3 = ConvBlock1d(base_channels * 4, base_channels * 2, stride=1)
        self.up2 = nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.dec2 = ConvBlock1d(base_channels * 2, base_channels, stride=1)
        self.up1 = nn.ConvTranspose1d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)
        self.out = nn.Conv1d(base_channels, num_classes, kernel_size=1)

    def forward(self, mix_wave: torch.Tensor) -> torch.Tensor:
        x = mix_wave.unsqueeze(1)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)
        u3 = self.up3(b)
        e2 = e2[:, :, : u3.size(-1)]
        d3 = self.dec3(torch.cat([u3, e2], dim=1))
        u2 = self.up2(d3)
        e1 = e1[:, :, : u2.size(-1)]
        d2 = self.dec2(torch.cat([u2, e1], dim=1))
        u1 = self.up1(d2)
        y = self.out(u1)
        return y

