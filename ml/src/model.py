"""
model.py - Temporal Convolutional Networks with Squeeze-and-Excitation
and dual-head speed/uncertainty prediction.

Models:
1. SpeedVibrationFilterNet
   - 16-channel spectral/physics input
   - Multi-scale dilated TCN
   - SE attention
   - Speed + uncertainty heads

2. RecurrentSpeedFilterNet
   - Conv1D + multi-scale dilated TCN
   - SE attention
   - 2-layer GRU
   - Optional prior-speed conditioning
   - Speed + uncertainty heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation1D(nn.Module):
    """Channel attention block for dynamically recalibrating input features."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()

        hidden = max(4, channels // reduction)

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Channels, Length)
        weights = self.fc(x).unsqueeze(-1)
        return x * weights


class DilatedConvBlock(nn.Module):
    """Dilated convolution block with residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int = 1,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

        self.residual = (
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.residual(x)


class SpeedVibrationFilterNet(nn.Module):
    """
    Spectral speed regression model.

    Input:
        (Batch, 16, Window)

    Current V1 architecture:
        16
         ↓
        32 stem
         ↓
        64 dilation=1
         ↓
        64 dilation=2
         ↓
        96 dilation=4
         ↓
        SE attention
         ↓
        128 projection
         ↓
        Global average pooling
         ↓
        64-dimensional shared representation
         ↓
        ┌───────────────┐
        ↓               ↓
      speed           variance
        μ                σ²
    """

    def __init__(
        self,
        in_channels: int = 16,
        window_size: int = 64,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.window_size = window_size

        # ---------------------------------------------------------
        # 1. Input stem
        # ---------------------------------------------------------
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )

        # ---------------------------------------------------------
        # 2. Multi-scale dilated temporal feature extractor
        # ---------------------------------------------------------

        # Local temporal patterns
        self.block1 = DilatedConvBlock(
            32,
            64,
            dilation=1,
        )

        # Medium-range temporal patterns
        self.block2 = DilatedConvBlock(
            64,
            64,
            dilation=2,
        )

        # Larger temporal receptive field
        self.block3 = DilatedConvBlock(
            64,
            96,
            dilation=4,
        )

        # ---------------------------------------------------------
        # 3. Channel attention
        # ---------------------------------------------------------
        self.se = SqueezeExcitation1D(
            channels=96,
            reduction=4,
        )

        # ---------------------------------------------------------
        # 4. Feature projection + temporal aggregation
        # ---------------------------------------------------------
        self.block4 = nn.Sequential(
            nn.Conv1d(
                96,
                128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # ---------------------------------------------------------
        # 5. Shared representation
        # ---------------------------------------------------------
        self.shared_fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.15),
        )

        # ---------------------------------------------------------
        # 6. Speed prediction head
        # ---------------------------------------------------------
        self.speed_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # ---------------------------------------------------------
        # 7. Uncertainty / variance prediction head
        # ---------------------------------------------------------
        self.var_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                Shape = (Batch, 16, Window)

        Returns:
            Shape = (Batch, 2)

            output[:, 0] = predicted speed mu
            output[:, 1] = predicted variance sigma²
        """

        # Input
        feat = self.stem(x)

        # Multi-scale temporal extraction
        feat = self.block1(feat)
        feat = self.block2(feat)
        feat = self.block3(feat)

        # Channel attention
        feat = self.se(feat)

        # Project and aggregate
        feat = self.block4(feat)

        # (B, 128, 1) -> (B, 128)
        feat = feat.squeeze(-1)

        # Shared representation
        shared = self.shared_fc(feat)

        # Speed
        mu = F.relu(
            self.speed_head(shared)
        )

        # Positive variance
        var = (
            F.softplus(
                self.var_head(shared)
            )
            + 0.01
        )

        return torch.cat(
            [mu, var],
            dim=-1,
        )


class RecurrentSpeedFilterNet(nn.Module):
    """
    Recurrent Speed & Vibration Filter Network.

    Architecture:

        Input IMU/spectral window
                ↓
        Conv1D stem
                ↓
        Dilated Conv blocks
          dilation 1
          dilation 2
          dilation 4
                ↓
          SE attention
                ↓
        temporal feature pooling
                ↓
        optional prior-speed conditioning
                ↓
             2-layer GRU
                ↓
        shared FC representation
                ↓
          ┌─────────────┐
          ↓             ↓
        speed        variance

    Supports:

    Single-window:
        x = (B, C, W)

    Sequence:
        x = (B, Seq, C, W)
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

        # ---------------------------------------------------------
        # 1. Input stem
        # ---------------------------------------------------------
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )

        # ---------------------------------------------------------
        # 2. Multi-scale temporal feature extractor
        # ---------------------------------------------------------
        self.block1 = DilatedConvBlock(
            32,
            64,
            dilation=1,
        )

        self.block2 = DilatedConvBlock(
            64,
            96,
            dilation=2,
        )

        self.block3 = DilatedConvBlock(
            96,
            128,
            dilation=4,
        )

        # IMPORTANT:
        # block3 outputs 128 channels, therefore SE must also
        # operate on 128 channels.
        self.se = SqueezeExcitation1D(
            channels=128,
            reduction=4,
        )

        # ---------------------------------------------------------
        # 3. Temporal feature aggregation
        # ---------------------------------------------------------
        self.encoder_pool = nn.Sequential(
            nn.Conv1d(
                128,
                128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # ---------------------------------------------------------
        # 4. Prior-speed conditioning
        # ---------------------------------------------------------
        if use_prior_speed:
            self.prior_proj = nn.Sequential(
                nn.Linear(1, 32),
                nn.GELU(),
            )

            gru_input_dim = 128 + 32

        else:
            self.prior_proj = None
            gru_input_dim = 128

        # ---------------------------------------------------------
        # 5. GRU memory
        # ---------------------------------------------------------
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            ),
        )

        # ---------------------------------------------------------
        # 6. Shared output representation
        # ---------------------------------------------------------
        self.fc_head = nn.Sequential(
            nn.Linear(
                hidden_dim,
                64,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ---------------------------------------------------------
        # 7. Speed head
        # ---------------------------------------------------------
        self.speed_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # ---------------------------------------------------------
        # 8. Variance head
        # ---------------------------------------------------------
        self.var_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def extract_window_features(
        self,
        x_flat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encodes windows into 128-dimensional embeddings.

        Input:
            (B, C, W)

        Output:
            (B, 128)
        """

        feat = self.stem(x_flat)
        feat = self.block1(feat)
        feat = self.block2(feat)
        feat = self.block3(feat)

        feat = self.se(feat)

        feat = self.encoder_pool(feat)

        feat = feat.squeeze(-1)

        return feat

    def forward(
        self,
        x: torch.Tensor,
        v_prior: torch.Tensor = None,
        h_0: torch.Tensor = None,
    ):
        """
        Supports both single-window and sequence operation.

        Single-window:
            x:
                (B, C, W)

            returns:
                output: (B, 2)
                h_n: GRU hidden state

        Sequence:
            x:
                (B, Seq, C, W)

            returns:
                output: (B, Seq, 2)
                h_n: GRU hidden state
        """

        # =========================================================
        # SEQUENCE MODE
        # =========================================================
        if x.dim() == 4:

            batch_size, seq_len, channels, window = x.shape

            x_flat = x.reshape(
                batch_size * seq_len,
                channels,
                window,
            )

            feat_flat = self.extract_window_features(
                x_flat
            )

            feat_seq = feat_flat.view(
                batch_size,
                seq_len,
                -1,
            )

            # -----------------------------------------------------
            # Prior speed
            # -----------------------------------------------------
            if self.use_prior_speed:

                if v_prior is None:

                    v_prior = torch.zeros(
                        batch_size,
                        seq_len,
                        1,
                        device=x.device,
                        dtype=x.dtype,
                    )

                elif v_prior.dim() == 2:

                    v_prior = (
                        v_prior
                        .unsqueeze(1)
                        .expand(
                            -1,
                            seq_len,
                            -1,
                        )
                    )

                elif v_prior.dim() == 1:

                    v_prior = (
                        v_prior
                        .view(
                            batch_size,
                            1,
                            1,
                        )
                        .expand(
                            -1,
                            seq_len,
                            -1,
                        )
                    )

                v_proj = self.prior_proj(
                    v_prior
                )

                gru_in = torch.cat(
                    [
                        feat_seq,
                        v_proj,
                    ],
                    dim=-1,
                )

            else:
                gru_in = feat_seq

            # -----------------------------------------------------
            # GRU
            # -----------------------------------------------------
            gru_out, h_n = self.gru(
                gru_in,
                h_0,
            )

            # -----------------------------------------------------
            # Output heads
            # -----------------------------------------------------
            shared = self.fc_head(
                gru_out
            )

            mu = F.relu(
                self.speed_head(shared)
            )

            var = (
                F.softplus(
                    self.var_head(shared)
                )
                + 0.01
            )

            output = torch.cat(
                [mu, var],
                dim=-1,
            )

            return output, h_n

        # =========================================================
        # SINGLE WINDOW MODE
        # =========================================================
        elif x.dim() == 3:

            batch_size = x.shape[0]

            feat = self.extract_window_features(
                x
            )

            # -----------------------------------------------------
            # Prior speed
            # -----------------------------------------------------
            if self.use_prior_speed:

                if v_prior is None:

                    v_prior = torch.zeros(
                        batch_size,
                        1,
                        device=x.device,
                        dtype=x.dtype,
                    )

                elif v_prior.dim() == 1:

                    v_prior = v_prior.view(
                        batch_size,
                        1,
                    )

                v_proj = self.prior_proj(
                    v_prior
                )

                feat_combined = torch.cat(
                    [
                        feat,
                        v_proj,
                    ],
                    dim=-1,
                )

            else:
                feat_combined = feat

            # -----------------------------------------------------
            # One-step GRU sequence
            # -----------------------------------------------------
            feat_seq = feat_combined.unsqueeze(1)

            gru_out, h_n = self.gru(
                feat_seq,
                h_0,
            )

            gru_out = gru_out.squeeze(1)

            # -----------------------------------------------------
            # Output heads
            # -----------------------------------------------------
            shared = self.fc_head(
                gru_out
            )

            mu = F.relu(
                self.speed_head(shared)
            )

            var = (
                F.softplus(
                    self.var_head(shared)
                )
                + 0.01
            )

            output = torch.cat(
                [mu, var],
                dim=-1,
            )

            return output, h_n

        else:

            raise ValueError(
                f"Expected input of dim 3 or 4, got {x.dim()}"
            )