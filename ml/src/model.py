"""
model.py - 1D-CNN / Temporal Convolutional Network for forward speed estimation.
Takes (Batch, 6, WindowSize) IMU window -> Outputs (Batch, 1) predicted forward speed (m/s).
"""

import torch
import torch.nn as nn


class SpeedVibrationFilterNet(nn.Module):
    def __init__(self, in_channels: int = 6, window_size: int = 100):
        super().__init__()
        
        # 1D Convolutional feature extractor
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Regressor head
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, 6, WindowSize)
        feat = self.conv_block(x) # (Batch, 128, 1)
        feat = feat.squeeze(-1)   # (Batch, 128)
        out = self.regressor(feat) # (Batch, 1)
        return out
