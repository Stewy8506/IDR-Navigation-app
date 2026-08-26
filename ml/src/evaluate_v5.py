"""Evaluate V5 only on held-out Driver E."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from .dataset_v5 import (
    SPEED_BINS_KMH,
    STEP_SIZE,
    WINDOW_SIZE,
    V5SpeedDataset,
    describe_targets,
    discover_pairs,
    load_normalization,
    load_windows,
    select_pairs,
)
from .model_v5 import VehicleSpeedNetV5


def evaluate(data_dir, weights_path, normalization_path="ml/weights/v5_normalization.npz"):
    pairs, number_s_files, skipped_files = discover_pairs(data_dir)
    test_pairs = select_pairs(pairs, "test")
    if not test_pairs:
        raise RuntimeError("No Driver E S/V pairs found.")
    windows, targets_kmh, source_indices = load_windows(test_pairs, WINDOW_SIZE, STEP_SIZE)
    if len(targets_kmh) == 0:
        raise RuntimeError("Driver E target distribution is empty.")
    stats = load_normalization(normalization_path)
    dataset = V5SpeedDataset(windows, targets_kmh, stats)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(weights_path, map_location=device)
    model_config = checkpoint.get("model_config", {}) if isinstance(checkpoint, dict) else {}
    model = VehicleSpeedNetV5(
        in_channels=model_config.get("in_channels", 6),
        window_size=model_config.get("window_size", 64),
    ).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    predictions = []
    with torch.no_grad():
        for features, _ in loader:
            normalized_prediction = model(features.to(device)).squeeze(-1).cpu().numpy()
            predictions.append(normalized_prediction * stats["target_std"] + stats["target_mean"])
    predictions_kmh = np.concatenate(predictions).astype(np.float32)
    error = predictions_kmh - targets_kmh
    print(f"number of S-files found: {number_s_files}")
    print(f"number of valid S/V pairs: {len(pairs)}; skipped files: {skipped_files}")
    print(f"total windows: {len(targets_kmh)}")
    print(f"GT mean/std: {targets_kmh.mean():.3f}/{targets_kmh.std():.3f} km/h")
    print(f"prediction mean/std: {predictions_kmh.mean():.3f}/{predictions_kmh.std():.3f} km/h")
    print(f"MAE: {np.abs(error).mean():.3f} km/h")
    print(f"RMSE: {np.sqrt(np.mean(error ** 2)):.3f} km/h")
    if np.std(targets_kmh) > 0 and np.std(predictions_kmh) > 0:
        correlation = float(np.corrcoef(targets_kmh, predictions_kmh)[0, 1])
    else:
        correlation = float("nan")
    print(f"Prediction correlation: {correlation:.6f}")
    print("Speed-bin statistics:")
    for low, high in SPEED_BINS_KMH:
        mask = (targets_kmh >= low) & (targets_kmh < high)
        if mask.any():
            print(f"  {low}-{high}: count={mask.sum()}, mean GT={targets_kmh[mask].mean():.3f}, mean prediction={predictions_kmh[mask].mean():.3f}, bias={error[mask].mean():+.3f} km/h")
        else:
            print(f"  {low}-{high}: count=0, mean GT=nan, mean prediction=nan, bias=nan")

    results_dir = "ml/results"
    os.makedirs(results_dir, exist_ok=True)
    maximum = max(float(targets_kmh.max()), float(predictions_kmh.max()))
    plt.figure(figsize=(8, 7))
    plt.scatter(targets_kmh, predictions_kmh, s=4, alpha=0.25)
    plt.plot([0, maximum], [0, maximum], "k--", linewidth=1.5)
    plt.xlabel("GT speed (km/h)")
    plt.ylabel("Predicted speed (km/h)")
    plt.title("V5 Driver E: ground truth vs prediction")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "v5_driver_e_scatter.png"), dpi=180)
    plt.close()

    representative = source_indices[0]
    representative_mask = np.asarray(source_indices) == representative
    indices = np.flatnonzero(representative_mask)
    plt.figure(figsize=(11, 5))
    plt.plot(np.arange(len(indices)), targets_kmh[indices], label="GT speed")
    plt.plot(np.arange(len(indices)), predictions_kmh[indices], label="Predicted speed")
    plt.xlabel("Window index")
    plt.ylabel("Speed (km/h)")
    plt.title("V5 Driver E representative recording")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "v5_driver_e_timeseries.png"), dpi=180)
    plt.close()
    print(f"saved plots under {results_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--normalization", default="ml/weights/v5_normalization.npz")
    args = parser.parse_args()
    evaluate(args.data_dir, args.weights, args.normalization)


if __name__ == "__main__":
    main()
