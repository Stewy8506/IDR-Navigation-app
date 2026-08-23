"""
task1_uncertainty_calibration.py - Uncertainty Calibration Check across Drivers A, B, D, and E.
Extracts predicted mu and sigma vs actual absolute error, computes distribution per driver,
and generates the calibration scatter plot.
"""

import glob
import math
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .model import SpeedVibrationFilterNet
from .dataset_spectral import compute_spectral_physics_features


def main():
    weights_path = "ml/weights/best_spectral_speed_filter.pt"
    device = torch.device("cpu")
    window_size = 32

    model = SpeedVibrationFilterNet(in_channels=16, window_size=window_size)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded {weights_path}")
    else:
        print(f"Weights not found: {weights_path}")
        return
    model.eval()

    drivers = ["Driver A", "Driver B", "Driver D", "Driver E"]
    results = {}

    for driver in drivers:
        pattern = f"ml/external/IO-VNBD_repo/Synchronised V abd S datasets/Categorised IOVNB Dataset/**/*{driver}*/**/S-*.csv"
        s_files = glob.glob(pattern, recursive=True)

        driver_gt = []
        driver_mu = []
        driver_sigma = []
        driver_abs_err = []

        with torch.no_grad():
            for sf in s_files[:8]:  # Sample across representative drives per driver
                vf = os.path.join(os.path.dirname(sf), os.path.basename(sf).replace("S-", "V-"))
                if not os.path.exists(vf):
                    continue

                dfs = pd.read_csv(sf, encoding="latin1")
                dfv = pd.read_csv(vf, encoding="latin1")
                dfs.columns = dfs.columns.str.strip()
                dfv.columns = dfv.columns.str.strip()

                ax = dfs["ACCELEROMETER X (m/s²)"].values
                ay = dfs["ACCELEROMETER Y (m/s²)"].values
                az = dfs["ACCELEROMETER Z (m/s²)"].values
                gy = dfs["GYROSCOPE Yaw (rad/s)"].values
                gp = dfs["GYROSCOPE Pitch (rad/s)"].values
                gr = dfs["GYROSCOPE Roll (rad/s)"].values

                if "Indicated Vehicle Speed (km/hr)" in dfv.columns:
                    spd = dfv["Indicated Vehicle Speed (km/hr)"].values
                elif "Velocity (km/hr)" in dfv.columns:
                    spd = dfv["Velocity (km/hr)"].values
                else:
                    continue

                N = min(len(ax), len(spd))
                if N < window_size:
                    continue

                raw_6ch = np.stack([ax[:N], ay[:N], az[:N], gy[:N], gp[:N], gr[:N]], axis=0)

                for i in range(window_size, N, 5):
                    w = raw_6ch[:, i - window_size : i]
                    f = compute_spectral_physics_features(w)
                    out = model(torch.from_numpy(f).unsqueeze(0).float()).squeeze(0)

                    mu_kmh = out[0].item() * 3.6
                    var_mps2 = out[1].item()
                    sigma_kmh = math.sqrt(max(0.01, var_mps2)) * 3.6
                    gt_kmh = spd[i]

                    driver_gt.append(gt_kmh)
                    driver_mu.append(mu_kmh)
                    driver_sigma.append(sigma_kmh)
                    driver_abs_err.append(abs(mu_kmh - gt_kmh))

        results[driver] = {
            "gt": np.array(driver_gt),
            "mu": np.array(driver_mu),
            "sigma": np.array(driver_sigma),
            "abs_err": np.array(driver_abs_err),
        }

    print("\n" + "=" * 80)
    print("      TASK 1: UNCERTAINTY CALIBRATION & OOD SENSITIVITY REPORT")
    print("=" * 80)
    print(f"{'Driver':<10} | {'Samples':<8} | {'MAE (km/h)':<11} | {'Mean σ (km/h)':<13} | {'P50 σ':<9} | {'P95 σ':<9} | {'Max σ':<9}")
    print("-" * 80)

    for driver in drivers:
        d = results[driver]
        mae = np.mean(d["abs_err"])
        mean_sig = np.mean(d["sigma"])
        p50_sig = np.percentile(d["sigma"], 50)
        p95_sig = np.percentile(d["sigma"], 95)
        max_sig = np.max(d["sigma"])
        print(f"{driver:<10} | {len(d['abs_err']):<8} | {mae:<11.2f} | {mean_sig:<13.2f} | {p50_sig:<9.2f} | {p95_sig:<9.2f} | {max_sig:<9.2f}")
    print("=" * 80)

    # Plot Scatter of Predicted Sigma vs Actual Absolute Error
    os.makedirs("ml/evaluation_plots", exist_ok=True)
    plot_path = "ml/evaluation_plots/uncertainty_calibration_scatter.png"

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {"Driver A": "blue", "Driver B": "green", "Driver D": "purple", "Driver E": "red"}
    alphas = {"Driver A": 0.25, "Driver B": 0.25, "Driver D": 0.25, "Driver E": 0.4}

    for driver in drivers:
        d = results[driver]
        # Subsample for clear scatter plot if large
        n_pts = len(d["sigma"])
        idx = np.random.choice(n_pts, min(2000, n_pts), replace=False)
        ax.scatter(
            d["sigma"][idx],
            d["abs_err"][idx],
            c=colors[driver],
            alpha=alphas[driver],
            s=18,
            label=f"{driver} (MAE: {np.mean(d['abs_err']):.1f}, Mean σ: {np.mean(d['sigma']):.1f} km/h)",
        )

    # Reference ideal line (y = x, y = 2x)
    max_val = 80
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.6, label="Ideal 1:1 Calibration Line (|Err| = σ)")
    ax.plot([0, max_val/2], [0, max_val], "k:", alpha=0.5, label="2σ Error Bound Line (|Err| = 2σ)")

    ax.set_title("Predicted Uncertainty (σ) vs. Actual Absolute Error Across Drivers", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Standard Deviation σ (km/h)", fontsize=11)
    ax.set_ylabel("Actual Absolute Speed Error |μ - GT| (km/h)", fontsize=11)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    print(f"\nSaved calibration scatter plot to {plot_path}")


if __name__ == "__main__":
    main()
