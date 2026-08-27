"""
audit_exp6a.py - Rigorous Forensic Validation & Generalization Audit for Experiment 6A.
Calculates all statistics, distributions, error breakdowns, regression metrics, and domain shifts.
"""

import glob
import json
import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml.kaggle.standalone_exp6a_kaggle import (
    align_imu_to_vehicle_frame,
    compute_18ch_features,
    compute_physical_pitch_series,
    DeepSpeedKinematicsNet,
    SequencePhysicsDataset,
)

DATA_DIR = "ml/external/IO-VNBD_repo/Synchronised V abd S datasets/Categorised IOVNB Dataset"

def run_audit():
    print("=" * 80)
    print("      FORENSIC AUDIT: EXPERIMENT 6A (INSS NAVIGATION ML PIPELINE)")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # 1 & 2. Speed Distribution & Speed Bin Sample Counts
    # -----------------------------------------------------------------------
    s_files = sorted(glob.glob(os.path.join(DATA_DIR, "**", "S-*.csv"), recursive=True))
    train_files = [f for f in s_files if "Driver E" not in f and "Vw" not in f and "Vta" not in f and "Vtb" not in f and "Vf" not in f and "S3a" not in f]
    val_files = [f for f in s_files if "S3a" in f]

    bins = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 80), (80, 300)]
    bin_names = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-80", "80+"]

    def extract_drive_data(file_list):
        all_speeds_kmh = []
        all_ay = []
        all_wz = []
        all_norm_a = []
        all_pitch = []
        drive_stats = {}

        for sf in file_list:
            vf = os.path.join(os.path.dirname(sf), os.path.basename(sf).replace("S-", "V-"))
            if not os.path.exists(vf):
                continue
            df_s = pd.read_csv(sf, encoding="latin1")
            df_v = pd.read_csv(vf, encoding="latin1")
            df_s.columns = df_s.columns.str.strip()
            df_v.columns = df_v.columns.str.strip()

            ax = df_s["ACCELEROMETER X (m/s²)"].values.astype(np.float32)
            ay = df_s["ACCELEROMETER Y (m/s²)"].values.astype(np.float32)
            az = df_s["ACCELEROMETER Z (m/s²)"].values.astype(np.float32)
            gy = df_s["GYROSCOPE Yaw (rad/s)"].values.astype(np.float32)
            gp = df_s["GYROSCOPE Pitch (rad/s)"].values.astype(np.float32)
            gr = df_s["GYROSCOPE Roll (rad/s)"].values.astype(np.float32)
            raw_imu = np.stack([ax, ay, az, gy, gp, gr], axis=0)

            speed_col = "Indicated Vehicle Speed (km/hr)" if "Indicated Vehicle Speed (km/hr)" in df_v.columns else "Velocity (km/hr)"
            speed_kmh = df_v[speed_col].values.astype(np.float32)
            min_len = min(raw_imu.shape[1], len(speed_kmh))
            raw_imu = raw_imu[:, :min_len]
            speed_kmh = speed_kmh[:min_len]

            aligned = align_imu_to_vehicle_frame(raw_imu)
            pitch = compute_physical_pitch_series(aligned[0], aligned[1], aligned[2], wy=aligned[4], wx=aligned[5], wz=aligned[3], dt=0.1)

            norm_a = np.linalg.norm(aligned[:3], axis=0)

            dname = os.path.basename(os.path.dirname(sf)) + "/" + os.path.basename(sf)
            drive_stats[dname] = {
                "count": len(speed_kmh),
                "min": float(np.min(speed_kmh)),
                "max": float(np.max(speed_kmh)),
                "mean": float(np.mean(speed_kmh)),
                "median": float(np.median(speed_kmh)),
                "std": float(np.std(speed_kmh)),
                "p5": float(np.percentile(speed_kmh, 5)),
                "p25": float(np.percentile(speed_kmh, 25)),
                "p75": float(np.percentile(speed_kmh, 75)),
                "p95": float(np.percentile(speed_kmh, 95)),
                "p99": float(np.percentile(speed_kmh, 99)),
                "speeds": speed_kmh,
                "ay": aligned[1],
                "wz": aligned[3],
                "norm_a": norm_a,
                "pitch": pitch,
            }
            all_speeds_kmh.extend(speed_kmh)
            all_ay.extend(aligned[1])
            all_wz.extend(aligned[3])
            all_norm_a.extend(norm_a)
            all_pitch.extend(pitch)

        return np.array(all_speeds_kmh), np.array(all_ay), np.array(all_wz), np.array(all_norm_a), np.array(all_pitch), drive_stats

    train_speeds, train_ay, train_wz, train_norm_a, train_pitch, train_drive_stats = extract_drive_data(train_files)
    val_speeds, val_ay, val_wz, val_norm_a, val_pitch, val_drive_stats = extract_drive_data(val_files)

    print("\n--- SECTION 1: PER-DRIVE SPEED DISTRIBUTION SUMMARY ---")
    print(f"{'Drive File':<32} | {'Count':<7} | {'Min':<5} | {'Max':<5} | {'Mean':<6} | {'Med':<6} | {'Std':<6} | {'P5':<5} | {'P25':<5} | {'P75':<5} | {'P95':<5} | {'P99':<5}")
    print("-" * 115)
    for dname, st in list(train_drive_stats.items()) + list(val_drive_stats.items()):
        tag = "[VAL] " if "S3a" in dname else "[TRN] "
        print(f"{tag + dname:<32} | {st['count']:<7d} | {st['min']:<5.1f} | {st['max']:<5.1f} | {st['mean']:<6.1f} | {st['median']:<6.1f} | {st['std']:<6.1f} | {st['p5']:<5.1f} | {st['p25']:<5.1f} | {st['p75']:<5.1f} | {st['p95']:<5.1f} | {st['p99']:<5.1f}")

    print("-" * 115)
    print(f"{'ALL TRAIN AGGREGATE':<32} | {len(train_speeds):<7d} | {np.min(train_speeds):<5.1f} | {np.max(train_speeds):<5.1f} | {np.mean(train_speeds):<6.1f} | {np.median(train_speeds):<6.1f} | {np.std(train_speeds):<6.1f} | {np.percentile(train_speeds,5):<5.1f} | {np.percentile(train_speeds,25):<5.1f} | {np.percentile(train_speeds,75):<5.1f} | {np.percentile(train_speeds,95):<5.1f} | {np.percentile(train_speeds,99):<5.1f}")
    print(f"{'ALL VAL (S3a) AGGREGATE':<32} | {len(val_speeds):<7d} | {np.min(val_speeds):<5.1f} | {np.max(val_speeds):<5.1f} | {np.mean(val_speeds):<6.1f} | {np.median(val_speeds):<6.1f} | {np.std(val_speeds):<6.1f} | {np.percentile(val_speeds,5):<5.1f} | {np.percentile(val_speeds,25):<5.1f} | {np.percentile(val_speeds,75):<5.1f} | {np.percentile(val_speeds,95):<5.1f} | {np.percentile(val_speeds,99):<5.1f}")

    print("\n--- SECTION 2: SPEED-BIN SAMPLE COUNTS & REPRESENTATION ---")
    print(f"{'Speed Bin (km/h)':<18} | {'TRAIN Count':<12} | {'TRAIN %':<10} | {'VAL Count':<12} | {'VAL %':<10} | {'Ratio (TRN/VAL %)'}")
    print("-" * 80)
    for (blow, bhigh), bname in zip(bins, bin_names):
        trn_c = int(np.sum((train_speeds >= blow) & (train_speeds < bhigh)))
        trn_pct = (trn_c / len(train_speeds)) * 100.0
        val_c = int(np.sum((val_speeds >= blow) & (val_speeds < bhigh)))
        val_pct = (val_c / len(val_speeds)) * 100.0
        ratio = trn_pct / val_pct if val_pct > 0 else float("nan")
        print(f"{bname:<18} | {trn_c:<12d} | {trn_pct:<9.2f}% | {val_c:<12d} | {val_pct:<9.2f}% | {ratio:<.2f}x")

    # -----------------------------------------------------------------------
    # 3. Model Inference & Validation Evaluation
    # -----------------------------------------------------------------------
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Inference Device: {device}", flush=True)

    val_ds = SequencePhysicsDataset(data_dir=DATA_DIR, split="val", seq_len=32, seq_stride=32)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False)

    ckpt_best_path = "ml/weights/exp6a_best_spectral_speed_filter.pt"
    ckpt_final_path = "ml/weights/exp6a_final_spectral_speed_filter.pt"

    def evaluate_checkpoint(ckpt_path):
        print(f"Evaluating checkpoint: {ckpt_path}...", flush=True)
        model = DeepSpeedKinematicsNet().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()

        val_preds, val_gts = [], []
        with torch.no_grad():
            for x_seq, targets in val_loader:
                x_seq = x_seq.to(device)
                v_gt_seq = targets["v"].to(device)
                B, L, C, W = x_seq.shape
                v_val_state = torch.zeros(B, dtype=torch.float32, device=device)
                b_preds = []
                for t in range(L):
                    out = model(x_seq[:, t], v_anchor=v_val_state)
                    mu_t = out["mu_v"]
                    b_preds.append(mu_t)
                    v_val_state = mu_t.detach()
                b_preds = torch.stack(b_preds, dim=1) * 3.6
                val_preds.extend(b_preds.cpu().numpy().flatten())
                val_gts.extend((v_gt_seq * 3.6).cpu().numpy().flatten())

        val_preds = np.array(val_preds)
        val_gts = np.array(val_gts)
        errors = np.abs(val_preds - val_gts)
        signed_errors = val_preds - val_gts

        overall_mae = float(errors.mean())
        overall_rmse = float(np.sqrt(np.mean((val_preds - val_gts)**2)))
        r = float(np.corrcoef(val_preds, val_gts)[0, 1])

        # Regression slope & intercept
        slope, intercept = np.polyfit(val_gts, val_preds, 1)
        r2 = r ** 2

        bin_stats = {}
        for (blow, bhigh), bn in zip(bins, bin_names):
            mask = (val_gts >= blow) & (val_gts < bhigh)
            n_m = int(mask.sum())
            if n_m > 0:
                b_err = errors[mask]
                b_s_err = signed_errors[mask]
                b_gt = val_gts[mask]
                b_pred = val_preds[mask]
                b_mape = float(np.mean(np.abs(b_pred - b_gt) / np.maximum(b_gt, 1.0))) * 100.0
                b_r = float(np.corrcoef(b_pred, b_gt)[0, 1]) if np.std(b_pred) > 0 and np.std(b_gt) > 0 else 0.0
                bin_stats[bn] = {
                    "count": n_m,
                    "mae": float(b_err.mean()),
                    "rmse": float(np.sqrt(np.mean((b_pred - b_gt)**2))),
                    "mean_signed_err": float(b_s_err.mean()),
                    "median_signed_err": float(np.median(b_s_err)),
                    "mape": b_mape,
                    "r": b_r,
                    "mean_gt": float(b_gt.mean()),
                    "mean_pred": float(b_pred.mean()),
                }
            else:
                bin_stats[bn] = {"count": 0, "mae": 0.0, "rmse": 0.0, "mean_signed_err": 0.0, "median_signed_err": 0.0, "mape": 0.0, "r": 0.0, "mean_gt": 0.0, "mean_pred": 0.0}

        balanced_mae = float(np.mean([bin_stats[bn]["mae"] for bn in bin_names]))

        return {
            "overall_mae": overall_mae,
            "overall_rmse": overall_rmse,
            "balanced_mae": balanced_mae,
            "r": r,
            "r2": r2,
            "slope": slope,
            "intercept": intercept,
            "mean_pred": float(val_preds.mean()),
            "mean_gt": float(val_gts.mean()),
            "std_pred": float(val_preds.std()),
            "std_gt": float(val_gts.std()),
            "bin_stats": bin_stats,
            "val_preds": val_preds,
            "val_gts": val_gts,
        }

    res_best = evaluate_checkpoint(ckpt_best_path)
    res_final = evaluate_checkpoint(ckpt_final_path)

    print("\n--- SECTION 4: HIGH-SPEED & SPEED-BIN SIGNED ERROR ANALYSIS (BEST CHECKPOINT: EPOCH 7) ---")
    print(f"{'Speed Bin':<10} | {'Count':<6} | {'MAE (km/h)':<10} | {'RMSE (km/h)':<11} | {'Mean Signed (Bias)':<18} | {'Median Signed':<14} | {'MAPE (%)':<10} | {'Pearson r':<10}")
    print("-" * 95)
    for bn in bin_names:
        bs = res_best["bin_stats"][bn]
        print(f"{bn:<10} | {bs['count']:<6d} | {bs['mae']:<10.2f} | {bs['rmse']:<11.2f} | {bs['mean_signed_err']:<+18.2f} | {bs['median_signed_err']:<+14.2f} | {bs['mape']:<9.1f}% | {bs['r']:<10.3f}")

    print("\n--- SECTION 5: PREDICTION-VS-TARGET LINEAR COMPRESSION AUDIT ---")
    for name, res in [("Best Checkpoint (Epoch 7)", res_best), ("Final Checkpoint (Epoch 15)", res_final)]:
        print(f"\n{name}:")
        print(f"  Overall Val MAE       : {res['overall_mae']:.2f} km/h")
        print(f"  Overall Val RMSE      : {res['overall_rmse']:.2f} km/h")
        print(f"  Balanced Val MAE      : {res['balanced_mae']:.2f} km/h")
        print(f"  Pearson Correlation r : {res['r']:.4f} (R² = {res['r2']:.4f})")
        print(f"  Linear Fit Equation   : Predicted = {res['slope']:.4f} * True + {res['intercept']:+.2f} km/h")
        print(f"  Mean Speed            : True = {res['mean_gt']:.2f} km/h | Predicted = {res['mean_pred']:.2f} km/h (Bias: {res['mean_pred'] - res['mean_gt']:+.2f} km/h)")
        print(f"  Standard Deviation    : True = {res['std_gt']:.2f} km/h | Predicted = {res['std_pred']:.2f} km/h (Compression Ratio: {res['std_pred']/res['std_gt']:.3f})")

    # -----------------------------------------------------------------------
    # 6. Distribution Shift Analysis Across Physics Dimensions
    # -----------------------------------------------------------------------
    print("\n--- SECTION 6: PHYSICAL DISTRIBUTION SHIFT AUDIT (TRAIN vs S3a VAL) ---")
    metrics_comp = [
        ("Speed (km/h)", train_speeds, val_speeds),
        ("Fwd Accel ay (m/s²)", train_ay, val_ay),
        ("Braking (ay < -0.5 m/s²)", train_ay[train_ay < -0.5], val_ay[val_ay < -0.5]),
        ("Yaw Rate wz (rad/s)", train_wz, val_wz),
        ("IMU Norm |a| (m/s²)", train_norm_a, val_norm_a),
        ("Pitch Theta (rad)", train_pitch, val_pitch),
    ]
    print(f"{'Physical Dimension':<26} | {'TRAIN Mean ± Std':<20} | {'VAL (S3a) Mean ± Std':<20} | {'TRAIN [P5, P95]':<20} | {'VAL [P5, P95]':<20}")
    print("-" * 115)
    for mname, t_arr, v_arr in metrics_comp:
        t_mean, t_std = np.mean(t_arr), np.std(t_arr)
        v_mean, v_std = np.mean(v_arr), np.std(v_arr)
        t_p5, t_p95 = np.percentile(t_arr, 5), np.percentile(t_arr, 95)
        v_p5, v_p95 = np.percentile(v_arr, 5), np.percentile(v_arr, 95)
        print(f"{mname:<26} | {t_mean:+.2f} ± {t_std:.2f}{'':<10} | {v_mean:+.2f} ± {v_std:.2f}{'':<10} | [{t_p5:+.2f}, {t_p95:+.2f}]{'':<7} | [{v_p5:+.2f}, {v_p95:+.2f}]")

if __name__ == "__main__":
    run_audit()
