"""
evaluate_spectral_binned.py - Evaluates the trained Spectral model across discrete speed bins on held-out Driver E.
"""

import glob
import os
import numpy as np
import pandas as pd
import torch

from .model import SpeedVibrationFilterNet
from .dataset_spectral import compute_spectral_physics_features


def load_normalization_stats(path):
    stats = np.load(path)
    mean = stats["mean"].astype(np.float32)
    std = stats["std"].astype(np.float32)
    if mean.shape != (16,) or std.shape != (16,):
        raise ValueError(f"Invalid spectral normalization statistics in '{path}'.")
    return mean, std


def main():
    weights_path = "ml/weights/best_spectral_speed_filter.pt"
    normalization_path = "ml/weights/spectral_normalization.npz"
    device = torch.device("cpu")
    window_size = 64

    if not os.path.exists(normalization_path):
        raise FileNotFoundError(
            f"File not found: {normalization_path}. Train the spectral model first "
            "to create training-only normalization statistics."
        )
    feature_mean, feature_std = load_normalization_stats(normalization_path)

    model = SpeedVibrationFilterNet(in_channels=16, window_size=window_size)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded {weights_path}")
    else:
        print(f"File not found: {weights_path}")
        return
    model.eval()

    test_s_files = glob.glob(
    "ml/data/IO-VNBD/**/*Driver E*/**/S-*.csv",
    recursive=True
    )
    print(f"Evaluating across {len(test_s_files)} Driver E test files...")

    all_gt_kmh = []
    all_pred_kmh = []

    with torch.no_grad():
        for s_file in test_s_files:
            v_file = os.path.join(os.path.dirname(s_file), os.path.basename(s_file).replace("S-", "V-"))
            if not os.path.exists(v_file):
                continue

            df_s = pd.read_csv(s_file, encoding="latin1")
            df_v = pd.read_csv(v_file, encoding="latin1")
            df_s.columns = df_s.columns.str.strip()
            df_v.columns = df_v.columns.str.strip()

            ax = df_s["ACCELEROMETER X (m/s²)"].values
            ay = df_s["ACCELEROMETER Y (m/s²)"].values
            az = df_s["ACCELEROMETER Z (m/s²)"].values
            gy = df_s["GYROSCOPE Yaw (rad/s)"].values
            gp = df_s["GYROSCOPE Pitch (rad/s)"].values
            gr = df_s["GYROSCOPE Roll (rad/s)"].values

            if "Indicated Vehicle Speed (km/hr)" in df_v.columns:
                speed_kmh = df_v["Indicated Vehicle Speed (km/hr)"].values
            elif "Velocity (km/hr)" in df_v.columns:
                speed_kmh = df_v["Velocity (km/hr)"].values
            else:
                continue

            N = min(len(ax), len(speed_kmh))
            if N < window_size:
                continue

            raw_6ch = np.stack([ax[:N], ay[:N], az[:N], gy[:N], gp[:N], gr[:N]], axis=0)

            for i in range(window_size, N, 4):
                w_raw = raw_6ch[:, i - window_size : i]
                feat16 = compute_spectral_physics_features(w_raw)
                feat16 = ((feat16 - feature_mean[:, None]) / feature_std[:, None]).astype(np.float32)
                out = model(torch.from_numpy(feat16).unsqueeze(0).float()).squeeze(0)
                pred_kmh = out[0].item() * 3.6

                all_gt_kmh.append(speed_kmh[i])
                all_pred_kmh.append(pred_kmh)

    all_gt_kmh = np.array(all_gt_kmh)
    all_pred_kmh = np.array(all_pred_kmh)

    rmse = np.sqrt(np.mean((all_pred_kmh - all_gt_kmh) ** 2))
    mae = np.mean(np.abs(all_pred_kmh - all_gt_kmh))

    print("\n" + "=" * 70)
    print("      16-CHANNEL SPECTRAL MODEL EVALUATION (HELD-OUT DRIVER E)")
    print("=" * 70)
    print(f"Total Evaluated Windows:     {len(all_gt_kmh)}")
    print(f"Ground Truth Mean Speed:     {np.mean(all_gt_kmh):.2f} km/h (std: {np.std(all_gt_kmh):.2f})")
    print(f"Predicted Mean Speed:        {np.mean(all_pred_kmh):.2f} km/h (std: {np.std(all_pred_kmh):.2f})")
    print(f"Speed RMSE:                  {rmse:.2f} km/h ({rmse/3.6:.2f} m/s)")
    print(f"Speed MAE:                   {mae:.2f} km/h ({mae/3.6:.2f} m/s)")
    print("-" * 70)
    print(f"{'Speed Bin':<18} | {'Count':<7} | {'Mean GT':<10} | {'Mean Pred':<10} | {'Bias (Pred - GT)':<15}")
    print("-" * 70)

    bins = [(0, 10), (10, 30), (30, 50), (50, 70), (70, 90), (90, 140)]
    for b_low, b_high in bins:
        mask = (all_gt_kmh >= b_low) & (all_gt_kmh < b_high)
        if np.sum(mask) > 0:
            m_gt = np.mean(all_gt_kmh[mask])
            m_pr = np.mean(all_pred_kmh[mask])
            bias = m_pr - m_gt
            print(f"{f'{b_low}-{b_high} km/h':<18} | {np.sum(mask):<7} | {m_gt:<10.1f} | {m_pr:<10.1f} | {bias:<+15.1f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
