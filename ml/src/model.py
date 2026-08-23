"""
model.py - Dilated Temporal Convolutional Network (TCN) with Squeeze-and-Excitation (SE) Attention
and Dual-Head Output: Forward Speed (mu) and Heteroscedastic Uncertainty (var).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation1D(nn.Module):
    """Channel attention block to dynamically recalibrate informative IMU channels."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, max(4, channels // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(max(4, channels // reduction), channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Channels, Length)
        w = self.fc(x).unsqueeze(-1)  # (Batch, Channels, 1)
        return x * w


class DilatedConvBlock(nn.Module):
    """Dilated convolution block with residual connection."""
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.residual(x)


class SpeedVibrationFilterNet(nn.Module):
    def __init__(self, in_channels: int = 10, window_size: int = 20):
        super().__init__()
        self.in_channels = in_channels
        self.window_size = window_size

        # Multi-scale Temporal Dilated Feature Extractor
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )

        self.block1 = DilatedConvBlock(32, 64, dilation=1)
        self.block2 = DilatedConvBlock(64, 64, dilation=2)  # Expanded temporal receptive field

        self.se = SqueezeExcitation1D(channels=64, reduction=4)

        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),  # (Batch, 128, 1)
        )

        self.shared_fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.15),
        )

        # Dual-Head Outputs
        # 1. Forward Speed Head (mu >= 0)
        self.speed_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # 2. Uncertainty / Variance Head (var > 0)
        self.var_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, in_channels, window_size)
        feat = self.stem(x)
        feat = self.block1(feat)
        feat = self.block2(feat)
        feat = self.se(feat)
        feat = self.block3(feat)
        feat = feat.squeeze(-1)  # (Batch, 128)

        shared = self.shared_fc(feat)

        # Predict non-negative forward speed mu
        mu = F.relu(self.speed_head(shared))

        # Predict strictly positive variance sigma^2 (min variance floor = 0.01)
        var = F.softplus(self.var_head(shared)) + 0.01

        return torch.cat([mu, var], dim=-1)
