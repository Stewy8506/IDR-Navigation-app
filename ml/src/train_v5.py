"""Train the isolated V5 vehicle-speed regression pipeline."""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset_v5 import (
    STEP_SIZE,
    WINDOW_SIZE,
    V5SpeedDataset,
    describe_targets,
    discover_pairs,
    fit_normalization,
    load_windows,
    save_normalization,
    select_pairs,
)
from .model_v5 import VehicleSpeedNetV5


SEED = 42
NORMALIZATION_PATH = "ml/weights/v5_normalization.npz"
CHECKPOINT_PATH = "ml/weights/best_v5_vehicle_speed.pt"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, target_mean, target_std, device):
    model.eval()
    errors = []
    with torch.no_grad():
        for features, target in loader:
            prediction = model(features.to(device)).squeeze(-1).cpu().numpy()
            target_kmh = target.numpy() * target_std + target_mean
            prediction_kmh = prediction * target_std + target_mean
            errors.append(prediction_kmh - target_kmh)
    error = np.concatenate(errors)
    return float(np.abs(error).mean()), float(np.sqrt(np.mean(error ** 2)))


def train(data_dir="ml/data/IO-VNBD", epochs=30, batch_size=128, learning_rate=3e-4):
    set_seed()
    pairs, number_s_files, skipped_files = discover_pairs(data_dir)
    train_pairs = select_pairs(pairs, "train")
    validation_pairs = select_pairs(pairs, "validation")
    test_pairs = select_pairs(pairs, "test")
    if not train_pairs or not validation_pairs or not test_pairs:
        raise RuntimeError(
            f"Required split is empty: train={len(train_pairs)}, "
            f"validation={len(validation_pairs)}, test={len(test_pairs)}"
        )
    if any("Driver E" in s for s, _ in train_pairs + validation_pairs):
        raise RuntimeError("Driver E leakage detected in training or validation pairs.")

    train_windows, train_targets, _ = load_windows(train_pairs, WINDOW_SIZE, STEP_SIZE)
    validation_windows, validation_targets, _ = load_windows(validation_pairs, WINDOW_SIZE, STEP_SIZE)
    test_windows, test_targets, _ = load_windows(test_pairs, WINDOW_SIZE, STEP_SIZE)
    if len(train_targets) == 0 or len(validation_targets) == 0 or len(test_targets) == 0:
        raise RuntimeError("Training, validation, and Driver E target distributions must be non-empty.")
    stats = fit_normalization(train_windows, train_targets)
    save_normalization(NORMALIZATION_PATH, stats)

    print(f"number of S-files found: {number_s_files}")
    print(f"number of valid S/V pairs: {len(pairs)}")
    print(f"number of skipped files: {skipped_files}")
    print(f"number of training windows: {len(train_targets)}")
    print(f"number of validation windows: {len(validation_targets)}")
    print(f"training target {describe_targets(train_targets)}")
    print(f"validation target {describe_targets(validation_targets)}")
    print(f"Driver E target {describe_targets(test_targets)}")
    print("target: V-file Indicated Vehicle Speed (km/hr), converted only for normalization/reporting")
    print("normalization: fitted on training windows only")

    train_loader = DataLoader(V5SpeedDataset(train_windows, train_targets, stats), batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(V5SpeedDataset(validation_windows, validation_targets, stats), batch_size=batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VehicleSpeedNetV5(in_channels=6, window_size=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    loss_fn = nn.SmoothL1Loss(beta=1.0)
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    best_mae = float("inf")
    patience = 6
    epochs_without_improvement = 0

    print(f"device: {device}")
    for epoch in range(1, epochs + 1):
        model.train()
        for features, target in train_loader:
            features = features.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            prediction = model(features).squeeze(-1)
            loss = loss_fn(prediction, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()
        validation_mae, validation_rmse = evaluate(model, validation_loader, stats["target_mean"], stats["target_std"], device)
        train_mae, _ = evaluate(model, train_loader, stats["target_mean"], stats["target_std"], device)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train MAE km/h: {train_mae:.3f} | Validation MAE km/h: {validation_mae:.3f} | Validation RMSE km/h: {validation_rmse:.3f}")
        if validation_mae < best_mae:
            best_mae = validation_mae
            torch.save({"model_state_dict": model.state_dict(), "model_config": model.config, "normalization_path": NORMALIZATION_PATH, "target_column": "Indicated Vehicle Speed (km/hr)"}, CHECKPOINT_PATH)
            print(f"  saved {CHECKPOINT_PATH}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {epoch} epochs.")
                break
    print(f"Best validation MAE km/h: {best_mae:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="ml/data/IO-VNBD")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    train(args.data_dir, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
