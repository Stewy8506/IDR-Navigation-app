"""Moderately sized residual TCN for V5 vehicle-speed regression."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()
        padding = dilation
        self.body = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, 3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
        )
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return F.gelu(self.body(x) + self.skip(x))


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.layers(x).unsqueeze(-1)


class VehicleSpeedNetV5(nn.Module):
    """Input (B, 6, 64), output (B, 1) normalized speed."""

    config = {
        "in_channels": 6,
        "window_size": 64,
        "stem_channels": 64,
        "block_channels": [96, 128, 160, 192],
        "dropout": 0.2,
    }

    def __init__(self, in_channels: int = 6, window_size: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.tcn = nn.Sequential(
            ResidualBlock(64, 96, 1),
            ResidualBlock(96, 128, 2),
            ResidualBlock(128, 160, 4),
            ResidualBlock(160, 192, 8),
        )
        self.se = SEBlock(192)
        self.fc = nn.Sequential(
            nn.Linear(384, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(192, 96),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.speed_head = nn.Sequential(
            nn.Linear(96, 48),
            nn.GELU(),
            nn.Linear(48, 1),
        )

    def forward(self, x):
        features = self.se(self.tcn(self.stem(x)))
        pooled = torch.cat([features.mean(dim=-1), features.amax(dim=-1)], dim=-1)
        return self.speed_head(self.fc(pooled))
