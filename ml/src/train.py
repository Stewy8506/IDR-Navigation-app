"""
train.py - Training script for Dual-Head Speed & Uncertainty Estimator with 10 Physics Features.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .dataset import IOVNBDDataset
from .model import SpeedVibrationFilterNet


def evaluate_nll(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs)  # (Batch, 2)
            mu = preds[:, 0]
            var = preds[:, 1]

            loss = criterion(mu, targets, var)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
    return total_loss / total_samples if total_samples > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Train Dual-Head Speed & Uncertainty Estimator")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="ml/external/IO-VNBD_repo/Synchronised V abd S datasets/Categorised IOVNB Dataset",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--in_channels", type=int, default=10, help="Number of input channels")
    parser.add_argument("--window_size", type=int, default=20, help="Window length in samples")
    parser.add_argument("--output_dir", type=str, default="ml/weights")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using compute device: {device}")

    # Datasets with 10 physics features
    train_dataset = IOVNBDDataset(
        data_dir=args.data_dir, window_size=args.window_size, step_size=2, is_train=True
    )
    val_dataset = IOVNBDDataset(
        data_dir=args.data_dir,
        window_size=args.window_size,
        step_size=5,
        is_train=False,
        val_split=True,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = SpeedVibrationFilterNet(in_channels=args.in_channels, window_size=args.window_size).to(device)
    criterion = nn.GaussianNLLLoss(eps=1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    print(
        f"Training 10-Channel Physics Model on {len(train_dataset)} samples, validating on {len(val_dataset)} samples..."
    )

    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_samples = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            preds = model(inputs)  # (Batch, 2)
            mu = preds[:, 0]
            var = preds[:, 1]

            loss = criterion(mu, targets, var)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_samples += inputs.size(0)

        scheduler.step()
        epoch_train_loss = train_loss / train_samples
        val_loss = evaluate_nll(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch+1:02d}/{args.epochs:02d}] - Train NLL: {epoch_train_loss:.4f} | Val NLL: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.output_dir, "best_speed_filter.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best checkpoint to {save_path} (Val NLL: {val_loss:.4f})")


if __name__ == "__main__":
    main()
