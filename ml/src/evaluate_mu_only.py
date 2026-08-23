"""
evaluate_mu_only.py - Evaluates mu-only Speed RMSE and MAE across test sets, decoupled from NLL.
"""

import glob
import os
import numpy as np
import pandas as pd
import torch

from .model import SpeedVibrationFilterNet
from .evaluate_full_pipeline import extract_window_features


def evaluate_checkpoint(weights_path: str, in_channels: int = 10, window_size: int = 20):
    device = torch.device("cpu")
    model = SpeedVibrationFilterNet(in_channels=in_channels, window_size=window_size)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded {weights_path}")
    else:
        print(f"Weights file not found: {weights_path}")
        return
    model.eval()

    test_s_files = glob.glob("ml/external/IO-VNBD_repo/Synchronised V abd S datasets/Categorised IOVNB Dataset/**/*Driver E*/**/S-*.csv", recursive=True)
    print(f"Evaluating across {len(test_s_files)} held-out Driver E drives...")

    all_gt_kmh = []
    all_pred_kmh = []
    all_pred_var = []

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

            for i in range(window_size, N, 2):
                w = raw_6ch[:, i - window_size : i]
                f = extract_window_features(w)
                out = model(torch.from_numpy(f).unsqueeze(0).float()).squeeze(0)
                pred_mps = out[0].item()
                var = out[1].item()

                all_gt_kmh.append(speed_kmh[i])
                all_pred_kmh.append(pred_mps * 3.6)
                all_pred_var.append(var)

    all_gt_kmh = np.array(all_gt_kmh)
    all_pred_kmh = np.array(all_pred_kmh)
    all_pred_var = np.array(all_pred_var)

    rmse = np.sqrt(np.mean((all_pred_kmh - all_gt_kmh) ** 2))
    mae = np.mean(np.abs(all_pred_kmh - all_gt_kmh))
    mean_pred = np.mean(all_pred_kmh)
    std_pred = np.std(all_pred_kmh)
    mean_gt = np.mean(all_gt_kmh)
    std_gt = np.std(all_gt_kmh)

    print("\n" + "=" * 65)
    print(f"      MU-ONLY EVALUATION (Decoupled from NLL Variance)")
    print("=" * 65)
    print(f"Total Evaluated Windows:     {len(all_gt_kmh)}")
    print(f"Ground Truth Mean Speed:     {mean_gt:.2f} km/h (std: {std_gt:.2f} km/h)")
    print(f"Predicted Mean Speed (mu):   {mean_pred:.2f} km/h (std: {std_pred:.2f} km/h)")
    print(f"Speed RMSE:                  {rmse:.2f} km/h ({rmse/3.6:.2f} m/s)")
    print(f"Speed MAE:                   {mae:.2f} km/h ({mae/3.6:.2f} m/s)")
    print(f"Average Predicted StdDev (sigma): {np.mean(np.sqrt(all_pred_var))*3.6:.2f} km/h")
    print("=" * 65)

    # Binned table
    bins = [(0, 10), (10, 30), (30, 50), (50, 70), (70, 90), (90, 140)]
    print(f"\n{'Speed Bin':<18} | {'Count':<6} | {'Mean GT':<10} | {'Mean Pred':<10} | {'Bias (Pred - GT)':<15}")
    print("-" * 65)
    for b_low, b_high in bins:
        mask = (all_gt_kmh >= b_low) & (all_gt_kmh < b_high)
        if np.sum(mask) > 0:
            m_gt = np.mean(all_gt_kmh[mask])
            m_pr = np.mean(all_pred_kmh[mask])
            print(f"{f'{b_low}-{b_high} km/h':<18} | {np.sum(mask):<6} | {m_gt:<10.1f} | {m_pr:<10.1f} | {m_pr - m_gt:<+15.1f}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    evaluate_checkpoint("ml/weights/best_speed_filter.pt")
