"""
evaluate.py - Benchmark evaluation for Dual-Head 10-Channel Physics Model.
"""

import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import IOVNBDDataset
from .model import SpeedVibrationFilterNet


def main():
    parser = argparse.ArgumentParser(description="Evaluate 10-Channel Model")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="ml/external/IO-VNBD_repo/Synchronised V abd S datasets/Categorised IOVNB Dataset",
    )
    parser.add_argument("--weights", type=str, default="ml/weights/best_speed_filter.pt")
    parser.add_argument("--in_channels", type=int, default=10)
    parser.add_argument("--window_size", type=int, default=20)
    args = parser.parse_args()

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    model = SpeedVibrationFilterNet(in_channels=args.in_channels, window_size=args.window_size).to(device)
    if os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"Loaded weights from {args.weights}")
    else:
        print(f"Weights file not found at {args.weights}.")
        return

    test_dataset = IOVNBDDataset(
        data_dir=args.data_dir,
        window_size=args.window_size,
        step_size=2,
        is_train=False,
        val_split=False,
    )
    if len(test_dataset) == 0:
        print("No test data found.")
        return

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    model.eval()

    all_mus = []
    all_vars = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            preds = model(inputs)
            mu = preds[:, 0].cpu().numpy()
            var = preds[:, 1].cpu().numpy()

            all_mus.extend(mu)
            all_vars.extend(var)
            all_targets.extend(targets.numpy())

    mus = np.array(all_mus)
    vars_ = np.array(all_vars)
    stds = np.sqrt(vars_)
    targets = np.array(all_targets)

    # Metrics
    errors = mus - targets
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(np.abs(errors))
    mean_std = np.mean(stds)

    # Calibration: % of ground truth targets falling within +/- 2 sigma
    in_bounds = np.abs(errors) <= (2.0 * stds)
    coverage_2sigma = np.mean(in_bounds) * 100.0

    print("\n" + "=" * 55)
    print("      10-CHANNEL PHYSICS SPEED + UNCERTAINTY BENCHMARK")
    print("=" * 55)
    print(f"Total Test Samples:        {len(mus)}")
    print(f"Speed RMSE:                {rmse:.4f} m/s ({rmse * 3.6:.2f} km/h)")
    print(f"Speed MAE:                 {mae:.4f} m/s ({mae * 3.6:.2f} km/h)")
    print(f"Average Predicted StdDev:  {mean_std:.4f} m/s ({mean_std * 3.6:.2f} km/h)")
    print(f"2-Sigma (95%) Coverage:    {coverage_2sigma:.1f}%")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
