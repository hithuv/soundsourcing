"""Compact Conv-TasNet style separator (multi-class output)."""

from __future__ import annotations

import torch
import torch.nn as nn


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, kernel_size=1),
            nn.PReLU(),
            nn.GroupNorm(1, hidden_channels),
            nn.Conv1d(
                hidden_channels,
                hidden_channels,
                kernel_size=kernel_size,
                groups=hidden_channels,
                dilation=dilation,
                padding=padding,
            ),
            nn.PReLU(),
            nn.GroupNorm(1, hidden_channels),
            nn.Conv1d(hidden_channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ConvTasNetLite(nn.Module):
    def __init__(self, num_classes: int, enc_dim: int = 128, bottleneck_dim: int = 128):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = nn.Conv1d(1, enc_dim, kernel_size=16, stride=8, padding=8)
        self.bottleneck = nn.Conv1d(enc_dim, bottleneck_dim, kernel_size=1)
        blocks = []
        for _repeat in range(2):
            for dilation in [1, 2, 4, 8, 16]:
                blocks.append(DepthwiseSeparableConv1d(bottleneck_dim, bottleneck_dim * 2, kernel_size=3, dilation=dilation))
        self.temporal_stack = nn.Sequential(*blocks)
        self.mask_head = nn.Conv1d(bottleneck_dim, num_classes * enc_dim, kernel_size=1)
        self.decoder = nn.ConvTranspose1d(enc_dim, 1, kernel_size=16, stride=8, padding=8)

    def forward(self, mix_wave: torch.Tensor) -> torch.Tensor:
        x = mix_wave.unsqueeze(1)
        enc = self.encoder(x)
        feat = self.temporal_stack(self.bottleneck(enc))
        bsz, _channels, frames = feat.shape
        masks = self.mask_head(feat).view(bsz, self.num_classes, enc.shape[1], frames)
        masks = torch.sigmoid(masks)
        masked = masks * enc.unsqueeze(1)
        masked = masked.view(bsz * self.num_classes, enc.shape[1], frames)
        decoded = self.decoder(masked).squeeze(1)
        decoded = decoded.view(bsz, self.num_classes, -1)
        return decoded

