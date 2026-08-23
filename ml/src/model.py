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


class RecurrentSpeedFilterNet(nn.Module):
    """
    Recurrent Speed & Vibration Filter Network (Conv1D + Multi-Scale SE + 2-layer GRU).
    Conditioned on Last-Known GNSS Speed (v_prior) to anchor high-speed highway velocity tracking.
    Maintains continuous temporal state (inertia/momentum memory) across driving sequences.
    Regularized with recurrent dropout, weight decay compatibility, and dual-head uncertainty.
    """
    def __init__(
        self,
        in_channels: int = 16,
        window_size: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_prior_speed: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_prior_speed = use_prior_speed

        # 1. Multi-scale Dilated Convolutional Feature Extractor
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )

        self.block1 = DilatedConvBlock(32, 64, dilation=1)
        self.block2 = DilatedConvBlock(64, 96, dilation=2)
        self.block3 = DilatedConvBlock(96, 128, dilation=4)  # Multi-dilation pyramid

        self.se = SqueezeExcitation1D(channels=128, reduction=4)

        self.encoder_pool = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),  # (Batch, 128, 1)
        )

        # 2. Prior Speed Conditioning Projection (1 -> 32)
        if use_prior_speed:
            self.prior_proj = nn.Sequential(
                nn.Linear(1, 32),
                nn.GELU(),
            )
            gru_input_dim = 128 + 32
        else:
            self.prior_proj = None
            gru_input_dim = 128

        # 3. Recurrent Memory Core (2-layer GRU with Recurrent Dropout)
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 4. Dense Projection with Regularization
        self.fc_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 5. Dual-Head Outputs
        self.speed_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        self.var_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def extract_window_features(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Encodes a batch of (B, C, W) windows into (B, 128) spatial/spectral embeddings."""
        feat = self.stem(x_flat)
        feat = self.block1(feat)
        feat = self.block2(feat)
        feat = self.block3(feat)
        feat = self.se(feat)
        feat = self.encoder_pool(feat).squeeze(-1)
        return feat

    def forward(
        self,
        x: torch.Tensor,
        v_prior: torch.Tensor = None,
        h_0: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass supporting both:
        - Sequence mode for training:
            x is (Batch, Seq_Len, Channels, Window_Size)
            v_prior is (Batch, Seq_Len, 1) or (Batch, 1)
            -> returns ((Batch, Seq_Len, 2), h_n)
        - Single-window mode for inference:
            x is (Batch, Channels, Window_Size)
            v_prior is (Batch, 1) or scalar
            -> returns ((Batch, 2), h_n)
        """
        if x.dim() == 4:
            # (Batch, Seq_Len, Channels, Window_Size)
            batch_size, seq_len, c, w = x.shape
            x_flat = x.view(batch_size * seq_len, c, w)
            feat_flat = self.extract_window_features(x_flat)  # (Batch*Seq_Len, 128)
            feat_seq = feat_flat.view(batch_size, seq_len, -1)  # (Batch, Seq_Len, 128)

            if self.use_prior_speed:
                if v_prior is None:
                    v_prior = torch.zeros(batch_size, seq_len, 1, device=x.device, dtype=x.dtype)
                elif v_prior.dim() == 2:
                    # (Batch, 1) -> expand across seq_len
                    v_prior = v_prior.unsqueeze(1).expand(-1, seq_len, -1)
                elif v_prior.dim() == 1:
                    v_prior = v_prior.view(batch_size, 1, 1).expand(-1, seq_len, -1)

                v_proj = self.prior_proj(v_prior)  # (Batch, Seq_Len, 32)
                gru_in = torch.cat([feat_seq, v_proj], dim=-1)  # (Batch, Seq_Len, 160)
            else:
                gru_in = feat_seq

            gru_out, h_n = self.gru(gru_in, h_0)  # (Batch, Seq_Len, hidden_dim)

            shared = self.fc_head(gru_out)
            mu = F.relu(self.speed_head(shared))
            var = F.softplus(self.var_head(shared)) + 0.01
            out = torch.cat([mu, var], dim=-1)  # (Batch, Seq_Len, 2)
            return out, h_n

        elif x.dim() == 3:
            # (Batch, Channels, Window_Size) - single step
            batch_size = x.shape[0]
            feat = self.extract_window_features(x)  # (Batch, 128)

            if self.use_prior_speed:
                if v_prior is None:
                    v_prior = torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
                elif v_prior.dim() == 1:
                    v_prior = v_prior.view(batch_size, 1)

                v_proj = self.prior_proj(v_prior)  # (Batch, 32)
                feat_combined = torch.cat([feat, v_proj], dim=-1)  # (Batch, 160)
            else:
                feat_combined = feat

            feat_seq = feat_combined.unsqueeze(1)  # (Batch, 1, 160)
            gru_out, h_n = self.gru(feat_seq, h_0)  # (Batch, 1, hidden_dim)
            gru_out = gru_out.squeeze(1)  # (Batch, hidden_dim)

            shared = self.fc_head(gru_out)
            mu = F.relu(self.speed_head(shared))
            var = F.softplus(self.var_head(shared)) + 0.01
            out = torch.cat([mu, var], dim=-1)  # (Batch, 2)
            return out, h_n
        else:
            raise ValueError(f"Expected input of dim 3 or 4, got {x.dim()}")


