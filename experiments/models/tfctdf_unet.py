"""Light TFC-TDF U-Net style model for spectrogram masking."""

from __future__ import annotations

import torch
import torch.nn as nn


class TfcTdfBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.tfc = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.tdf = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 5), padding=(0, 2)),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=(5, 1), padding=(2, 0)),
            nn.BatchNorm2d(channels),
        )
        self.out = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.tfc(x)
        x = self.tdf(x)
        return self.out(x + residual)


class TFCTDFUNet(nn.Module):
    def __init__(self, n_class: int, base: int = 24):
        super().__init__()
        self.stem = nn.Conv2d(1, base, kernel_size=3, padding=1)
        self.enc1 = nn.Sequential(TfcTdfBlock(base), TfcTdfBlock(base))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(base, base * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            TfcTdfBlock(base * 2),
        )
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            TfcTdfBlock(base * 4),
        )
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base * 4, base * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            TfcTdfBlock(base * 2),
        )
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base * 2, base, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            TfcTdfBlock(base),
        )
        self.head = nn.Conv2d(base, n_class, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        u2 = self.up2(b)
        e2 = e2[:, :, : u2.size(2), : u2.size(3)]
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        e1 = e1[:, :, : u1.size(2), : u1.size(3)]
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return torch.sigmoid(self.head(d1))

