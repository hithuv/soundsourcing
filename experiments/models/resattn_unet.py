"""Residual attention U-Net variant."""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + identity)


class AttentionGate(nn.Module):
    def __init__(self, gate_ch: int, skip_ch: int, inter_ch: int):
        super().__init__()
        self.g_proj = nn.Sequential(nn.Conv2d(gate_ch, inter_ch, kernel_size=1), nn.BatchNorm2d(inter_ch))
        self.x_proj = nn.Sequential(nn.Conv2d(skip_ch, inter_ch, kernel_size=1), nn.BatchNorm2d(inter_ch))
        self.psi = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(inter_ch, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        alpha = self.psi(self.g_proj(gate) + self.x_proj(skip))
        return skip * alpha


class ResidualAttentionUNet(nn.Module):
    def __init__(self, n_class: int, base: int = 32):
        super().__init__()
        self.enc1 = ResidualBlock(1, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResidualBlock(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResidualBlock(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = ResidualBlock(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.att3 = AttentionGate(base * 4, base * 4, base * 2)
        self.dec3 = ResidualBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.att2 = AttentionGate(base * 2, base * 2, base)
        self.dec2 = ResidualBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.att1 = AttentionGate(base, base, base // 2)
        self.dec1 = ResidualBlock(base * 2, base)
        self.out = nn.Conv2d(base, n_class, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        u3 = self.up3(b)
        e3 = e3[:, :, : u3.size(2), : u3.size(3)]
        d3 = self.dec3(torch.cat([u3, self.att3(u3, e3)], dim=1))

        u2 = self.up2(d3)
        e2 = e2[:, :, : u2.size(2), : u2.size(3)]
        d2 = self.dec2(torch.cat([u2, self.att2(u2, e2)], dim=1))

        u1 = self.up1(d2)
        e1 = e1[:, :, : u1.size(2), : u1.size(3)]
        d1 = self.dec1(torch.cat([u1, self.att1(u1, e1)], dim=1))
        return torch.sigmoid(self.out(d1))

