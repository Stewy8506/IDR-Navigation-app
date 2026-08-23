"""
train_spectral.py - Trains SpeedVibrationFilterNet with Spectral Features + Combined Huber & NLL Loss.
Directly optimizes mu regression accuracy while calibrating dynamic variance sigma^2.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .model import SpeedVibrationFilterNet
from .dataset_spectral import SpectralIOVNBDDataset


def train(epochs: int = 20, batch_size: int = 128, lr: float = 1e-3, in_channels: int = 16, window_size: int = 32):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using compute device: {device}")

    train_ds = SpectralIOVNBDDataset(is_train=True, val_split=False, window_size=window_size)
    val_ds = SpectralIOVNBDDataset(is_train=False, val_split=True, window_size=window_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = SpeedVibrationFilterNet(in_channels=in_channels, window_size=window_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    huber_loss_fn = nn.SmoothL1Loss(beta=1.0)
    nll_loss_fn = nn.GaussianNLLLoss(full=False)

    os.makedirs("ml/weights", exist_ok=True)
    best_val_mae = float("inf")
    save_path = "ml/weights/best_spectral_speed_filter.pt"

    print(f"\nTraining 16-Channel Spectral Model on {len(train_ds)} samples, validating on {len(val_ds)} samples...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_mae = 0.0
        batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            pred = model(x)
            mu = pred[:, 0]
            var = pred[:, 1]

            # Combined loss: direct Huber loss on mu + Gaussian NLL for variance
            loss_huber = huber_loss_fn(mu, y)
            loss_nll = nll_loss_fn(mu, y, var)
            loss = loss_huber + 0.2 * loss_nll

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_mae += torch.abs(mu - y).mean().item()
            batches += 1

        scheduler.step()

        # Validation
        model.eval()
        val_mae = 0.0
        val_rmse_sq = 0.0
        val_samples = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                mu = pred[:, 0]

                err = torch.abs(mu - y)
                val_mae += err.sum().item()
                val_rmse_sq += (err ** 2).sum().item()
                val_samples += len(y)

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
            print(f"  --> Saved new best spectral checkpoint (Val MAE: {best_val_mae:.2f} km/h)")


if __name__ == "__main__":
    train(epochs=15, batch_size=128, in_channels=16, window_size=32)
