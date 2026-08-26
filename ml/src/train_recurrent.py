"""Train the NO-PRIOR RecurrentSpeedFilterNet."""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .dataset_recurrent import RecurrentIOVNBDDataset, SPEED_BINS_KMH
from .model import RecurrentSpeedFilterNet


SEED = 42


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 3e-4,
    data_dir: str = "ml/data/IO-VNBD",
):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    train_ds = RecurrentIOVNBDDataset(
        data_dir=data_dir, window_size=32, seq_len=16, step_size=4,
        is_train=True, val_split=False, augment=True,
    )
    val_ds = RecurrentIOVNBDDataset(
        data_dir=data_dir, window_size=32, seq_len=16, step_size=8,
        is_train=False, val_split=True, augment=False,
    )
    if not train_ds or not val_ds:
        raise RuntimeError("Training and validation datasets must both contain sequences.")

    # Balance sequences by inverse frequency of their mean-speed bin.
    bin_counts = train_ds.speed_bin_counts()
    print("Training sequence counts before balancing:")
    for (low, high), count in zip(SPEED_BINS_KMH, bin_counts):
        print(f"  {low:g}-{high:g} km/h: {count}")
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(SEED),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
    )

    model = RecurrentSpeedFilterNet(
        in_channels=16, window_size=32, hidden_dim=128, num_layers=2,
        dropout=0.2, use_prior_speed=False,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )
    huber_loss_fn = nn.SmoothL1Loss(beta=1.0)
    nll_loss_fn = nn.GaussianNLLLoss(full=False)

    save_path = "ml/weights/best_recurrent_v2_no_prior.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best_val_mae_kmh = float("inf")

    print(
        f"Training {len(train_ds)} sequences and validating on {len(val_ds)} "
        "natural Driver D sequences."
    )
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_abs_error = 0.0
        total_values = 0

        for x_seq, _v_prior, y_seq in train_loader:
            x_seq = x_seq.to(device)
            y_seq = y_seq.to(device)
            optimizer.zero_grad()

            # NO-PRIOR experiment: the compatibility prior is deliberately unused.
            pred, _ = model(x_seq, v_prior=None)
            mu = pred[..., 0]
            variance = pred[..., 1]
            loss = huber_loss_fn(mu, y_seq) + 0.10 * nll_loss_fn(mu, y_seq, variance)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * y_seq.numel()
            total_abs_error += torch.abs(mu.detach() - y_seq).sum().item()
            total_values += y_seq.numel()

        scheduler.step()
        metrics = validate(model, val_loader, device)
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] | "
            f"Loss: {total_loss / total_values:.4f} | "
            f"Train MAE: {total_abs_error / total_values * 3.6:.2f} km/h | "
            f"Val MAE: {metrics['mae_mps']:.3f} m/s ({metrics['mae_kmh']:.2f} km/h) | "
            f"Val RMSE: {metrics['rmse_mps']:.3f} m/s ({metrics['rmse_kmh']:.2f} km/h) | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )
        print_bin_metrics(metrics["bins"])

        # Select and save only the checkpoint with the best Driver D MAE.
        if metrics["mae_kmh"] < best_val_mae_kmh:
            best_val_mae_kmh = metrics["mae_kmh"]
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best checkpoint: {save_path}")

    print(f"Training complete. Best Driver D MAE: {best_val_mae_kmh:.2f} km/h")


def validate(model, loader, device):
    model.eval()
    targets = []
    predictions = []
    with torch.no_grad():
        for x_seq, _v_prior, y_seq in loader:
            pred, _ = model(x_seq.to(device), v_prior=None)
            predictions.append(pred[..., 0].cpu().numpy())
            targets.append(y_seq.numpy())

    target_array = np.concatenate(targets).reshape(-1)
    prediction_array = np.concatenate(predictions).reshape(-1)
    error_array = prediction_array - target_array
    metrics = {
        "mae_mps": float(np.abs(error_array).mean()),
        "rmse_mps": float(np.sqrt(np.mean(error_array ** 2))),
    }
    metrics["mae_kmh"] = metrics["mae_mps"] * 3.6
    metrics["rmse_kmh"] = metrics["rmse_mps"] * 3.6
    bins = []
    target_kmh = target_array * 3.6
    for low, high in SPEED_BINS_KMH:
        mask = (target_kmh >= low) & (target_kmh < high)
        bin_errors = error_array[mask]
        bins.append({
            "count": int(mask.sum()),
            "mae": float(np.abs(bin_errors).mean()) * 3.6 if len(bin_errors) else float("nan"),
            "bias": float(bin_errors.mean()) * 3.6 if len(bin_errors) else float("nan"),
        })
    metrics["bins"] = bins
    return metrics


def print_bin_metrics(bin_metrics):
    for (low, high), values in zip(SPEED_BINS_KMH, bin_metrics):
        print(
            f"  {low:g}-{high:g} km/h: count={values['count']}, "
            f"MAE={values['mae']:.2f} km/h, bias={values['bias']:.2f} km/h"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="ml/data/IO-VNBD")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
