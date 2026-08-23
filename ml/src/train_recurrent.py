"""
train_recurrent.py - Trains Prior-Conditioned RecurrentSpeedFilterNet (Conv-GRU) on sequential chunks.
Optimizes multi-step Huber loss + Gaussian NLL for calibrated velocity and uncertainty.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .model import RecurrentSpeedFilterNet
from .dataset_recurrent import RecurrentIOVNBDDataset


def train(
    epochs: int = 15,
    batch_size: int = 128,
    lr: float = 1e-3,
    in_channels: int = 16,
    window_size: int = 32,
    seq_len: int = 16,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using compute device: {device}")

    train_ds = RecurrentIOVNBDDataset(
        window_size=window_size,
        seq_len=seq_len,
        step_size=4,
        is_train=True,
        val_split=False,
        augment=True,
    )
    val_ds = RecurrentIOVNBDDataset(
        window_size=window_size,
        seq_len=seq_len,
        step_size=8,
        is_train=False,
        val_split=True,
        augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = RecurrentSpeedFilterNet(
        in_channels=in_channels,
        window_size=window_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        use_prior_speed=True,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    huber_loss_fn = nn.SmoothL1Loss(beta=1.0)
    nll_loss_fn = nn.GaussianNLLLoss(full=False)

    os.makedirs("ml/weights", exist_ok=True)
    best_val_mae = float("inf")
    save_path = "ml/weights/best_recurrent_speed_filter.pt"

    print(f"\nTraining Prior-Conditioned Conv-GRU Model on {len(train_ds)} sequences ({seq_len} steps each), validating on {len(val_ds)} sequences...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_mae = 0.0
        batches = 0

        for x_seq, v_prior, y_seq in train_loader:
            x_seq = x_seq.to(device)      # (B, Seq_Len, 16, 32)
            v_prior = v_prior.to(device)  # (B, 1)
            y_seq = y_seq.to(device)      # (B, Seq_Len)
            optimizer.zero_grad()

            pred, _ = model(x_seq, v_prior=v_prior)  # (B, Seq_Len, 2)
            mu = pred[:, :, 0]
            var = pred[:, :, 1]

            loss_huber = huber_loss_fn(mu, y_seq)
            loss_nll = nll_loss_fn(mu, y_seq, var)
            loss = loss_huber + 0.15 * loss_nll

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_mae += torch.abs(mu - y_seq).mean().item()
            batches += 1

        scheduler.step()

        # Validation across distinct validation sequences
        model.eval()
        val_mae = 0.0
        val_rmse_sq = 0.0
        val_samples = 0

        with torch.no_grad():
            for x_seq, v_prior, y_seq in val_loader:
                x_seq = x_seq.to(device)
                v_prior = v_prior.to(device)
                y_seq = y_seq.to(device)

                pred, _ = model(x_seq, v_prior=v_prior)
                mu = pred[:, :, 0]

                err = torch.abs(mu - y_seq)
                val_mae += err.sum().item()
                val_rmse_sq += (err ** 2).sum().item()
                val_samples += y_seq.numel()

        avg_val_mae_mps = val_mae / val_samples
        avg_val_rmse_mps = (val_rmse_sq / val_samples) ** 0.5
        avg_val_mae_kmh = avg_val_mae_mps * 3.6
        avg_val_rmse_kmh = avg_val_rmse_mps * 3.6

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] - Train MAE: {total_mae/batches*3.6:.2f} km/h | "
            f"Val MAE: {avg_val_mae_kmh:.2f} km/h | Val RMSE: {avg_val_rmse_kmh:.2f} km/h"
        )

        if avg_val_mae_kmh < best_val_mae:
            best_val_mae = avg_val_mae_kmh
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best recurrent model checkpoint (Val MAE: {best_val_mae:.2f} km/h)")

    print(f"\nTraining Complete! Best Checkpoint Saved to {save_path} with Val MAE: {best_val_mae:.2f} km/h.")


if __name__ == "__main__":
    train(epochs=25, batch_size=64, in_channels=16, window_size=32, seq_len=16)
