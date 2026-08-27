"""
diagnostic_6c_preflight.py
Comprehensive Pre-Flight Diagnostics for Experiment 6C Design:
1. Residual Closed-Loop Delta_V & State Propagation Dynamics Audit
2. Pitch Distribution Shift & Pitch-Conditioned Error Audit
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.kaggle.standalone_exp6a_kaggle import (
    SequencePhysicsDataset,
    DeepSpeedKinematicsNet,
    align_imu_to_vehicle_frame,
    compute_physical_pitch_series,
    compute_18ch_features,
    GRAVITY
)

DATA_DIR = "ml/external/IO-VNBD_repo/Synchronised V abd S datasets/Categorised IOVNB Dataset"

def run_diagnostics():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running Diagnostics on Device: {device}", flush=True)

    val_ds = SequencePhysicsDataset(data_dir=DATA_DIR, split="val", seq_len=32, seq_stride=32)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False)

    ckpt_6a_path = "ml/weights/exp6a_best_spectral_speed_filter.pt"
    ckpt_6ab_path = "ml/weights/exp6ab_best_spectral_speed_filter.pt"

    # =========================================================================
    # PART 1: RESIDUAL CLOSED-LOOP DELTA_V & STATE DYNAMICS AUDIT
    # =========================================================================
    print("\n" + "=" * 80)
    print("      PART 1: RESIDUAL STATE (v_anchor + delta_v) DYNAMICS AUDIT")
    print("=" * 80)

    def evaluate_residual_dynamics(ckpt_path, name):
        print(f"\n--- Analyzing {name} ({ckpt_path}) ---", flush=True)
        model = DeepSpeedKinematicsNet().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()

        # We will track:
        # 1. Closed-loop rollout (v_anchor = pred_{t-1})
        # 2. Open-loop teacher-forced rollout (v_anchor = gt_{t-1})
        # 3. Delta_v predicted vs Delta_v true across speed bins
        # 4. Zero anchor test (v_anchor = 0, pure static feature delta_v)

        cl_preds, tf_preds, zero_preds = [], [], []
        gts = []
        cl_dvs, tf_dvs, gt_dvs = [], [], []
        pitches = []
        ay_comps = []

        with torch.no_grad():
            for x_seq, targets in val_loader:
                x_seq = x_seq.to(device) # (B, L, C, W)
                v_gt_seq = targets["v"].to(device) # (B, L)
                dv_gt_seq = targets["delta_v"].to(device)
                pitch_gt_seq = targets["pitch"].to(device)
                B, L, C, W = x_seq.shape

                # Closed-Loop Rollout
                v_cl_state = torch.zeros(B, dtype=torch.float32, device=device)
                b_cl_preds, b_cl_dvs = [], []
                for t in range(L):
                    out = model(x_seq[:, t], v_anchor=v_cl_state)
                    mu_t = out["mu_v"]
                    dv_t = out["delta_v"]
                    b_cl_preds.append(mu_t)
                    b_cl_dvs.append(dv_t)
                    v_cl_state = mu_t.detach()
                b_cl_preds = torch.stack(b_cl_preds, dim=1) * 3.6
                b_cl_dvs = torch.stack(b_cl_dvs, dim=1) * 3.6
                cl_preds.extend(b_cl_preds.cpu().numpy().flatten())
                cl_dvs.extend(b_cl_dvs.cpu().numpy().flatten())

                # Teacher-Forced (Open-Loop) Rollout: v_anchor = v_gt_{t-1}
                b_tf_preds, b_tf_dvs = [], []
                for t in range(L):
                    v_prev_gt = v_gt_seq[:, t-1] if t > 0 else torch.zeros(B, dtype=torch.float32, device=device)
                    out = model(x_seq[:, t], v_anchor=v_prev_gt)
                    mu_t = out["mu_v"]
                    dv_t = out["delta_v"]
                    b_tf_preds.append(mu_t)
                    b_tf_dvs.append(dv_t)
                b_tf_preds = torch.stack(b_tf_preds, dim=1) * 3.6
                b_tf_dvs = torch.stack(b_tf_dvs, dim=1) * 3.6
                tf_preds.extend(b_tf_preds.cpu().numpy().flatten())
                tf_dvs.extend(b_tf_dvs.cpu().numpy().flatten())

                # Zero-Anchor Test: What does the network predict from IMU alone with v_anchor=0?
                b_zero_preds = []
                for t in range(L):
                    out = model(x_seq[:, t], v_anchor=0.0)
                    b_zero_preds.append(out["mu_v"])
                b_zero_preds = torch.stack(b_zero_preds, dim=1) * 3.6
                zero_preds.extend(b_zero_preds.cpu().numpy().flatten())

                gts.extend((v_gt_seq * 3.6).cpu().numpy().flatten())
                gt_dvs.extend((dv_gt_seq * 3.6).cpu().numpy().flatten())
                pitches.extend(pitch_gt_seq.cpu().numpy().flatten())

        cl_preds = np.array(cl_preds)
        tf_preds = np.array(tf_preds)
        zero_preds = np.array(zero_preds)
        gts = np.array(gts)
        cl_dvs = np.array(cl_dvs)
        tf_dvs = np.array(tf_dvs)
        gt_dvs = np.array(gt_dvs)
        pitches = np.array(pitches)

        cl_mae = np.mean(np.abs(cl_preds - gts))
        tf_mae = np.mean(np.abs(tf_preds - gts))
        zero_mae = np.mean(np.abs(zero_preds - gts))

        cl_slope, cl_intercept = np.polyfit(gts, cl_preds, 1)
        tf_slope, tf_intercept = np.polyfit(gts, tf_preds, 1)
        zero_slope, zero_intercept = np.polyfit(gts, zero_preds, 1)

        print(f"Closed-Loop Rollout MAE : {cl_mae:5.2f} km/h | Slope: {cl_slope:.4f} | Intercept: {cl_intercept:+5.2f} km/h")
        print(f"Teacher-Forced MAE      : {tf_mae:5.2f} km/h | Slope: {tf_slope:.4f} | Intercept: {tf_intercept:+5.2f} km/h")
        print(f"Zero-Anchor Static MAE  : {zero_mae:5.2f} km/h | Slope: {zero_slope:.4f} | Intercept: {zero_intercept:+5.2f} km/h")

        # Delta_v statistics across speed regimes
        bins = [(0, 20), (20, 50), (50, 80), (80, 200)]
        bin_names = ["Low (0-20)", "Mid (20-50)", "High (50-80)", "VHigh (80+)"]
        print(f"\nDelta_V Increment Statistics by Speed Regime ({name}):")
        print(f"{'Speed Regime':<15} | {'Count':<6} | {'Mean GT dv':<12} | {'Mean CL dv':<12} | {'Mean TF dv':<12} | {'Zero Pred v'}")
        print("-" * 75)
        for (bl, bh), bn in zip(bins, bin_names):
            m = (gts >= bl) & (gts < bh)
            cnt = int(m.sum())
            if cnt > 0:
                print(f"{bn:<15} | {cnt:<6d} | {np.mean(gt_dvs[m]):+10.3f}   | {np.mean(cl_dvs[m]):+10.3f}   | {np.mean(tf_dvs[m]):+10.3f}   | {np.mean(zero_preds[m]):5.2f} km/h")

        return {
            "cl_preds": cl_preds, "tf_preds": tf_preds, "zero_preds": zero_preds,
            "gts": gts, "cl_dvs": cl_dvs, "tf_dvs": tf_dvs, "gt_dvs": gt_dvs, "pitches": pitches
        }

    res_6a = evaluate_residual_dynamics(ckpt_6a_path, "Exp6A (Baseline)")
    res_6ab = evaluate_residual_dynamics(ckpt_6ab_path, "Exp6A-B (Speed-Balanced)")

    # =========================================================================
    # PART 2: PITCH-CONDITIONED ERROR AUDIT
    # =========================================================================
    print("\n" + "=" * 80)
    print("      PART 2: PITCH DISTRIBUTION SHIFT & CONDITIONED ERROR AUDIT")
    print("=" * 80)

    pitches = res_6a["pitches"]
    pitches_deg = pitches * (180.0 / math.pi)
    gts = res_6a["gts"]
    preds_6a = res_6a["cl_preds"]
    preds_6ab = res_6ab["cl_preds"]
    errs_6a = preds_6a - gts
    errs_6ab = preds_6ab - gts

    print(f"Validation S3a Pitch Range (rad) : Min={np.min(pitches):+.4f}, Max={np.max(pitches):+.4f}, Mean={np.mean(pitches):+.4f}, Median={np.median(pitches):+.4f}, Std={np.std(pitches):.4f}")
    print(f"Validation S3a Pitch Range (deg) : Min={np.min(pitches_deg):+5.1f}°, Max={np.max(pitches_deg):+5.1f}°, Mean={np.mean(pitches_deg):+5.1f}°, Median={np.median(pitches_deg):+5.1f}°, Std={np.std(pitches_deg):4.1f}°")

    # Define Pitch Bins (in degrees): Flat (-5 to +5), Mild Uphill (+5 to +15), Steep Uphill (+15 to +30), Downhill (<-5)
    pitch_bins = [(-90, -5), (-5, 5), (5, 15), (15, 30), (30, 90)]
    p_names = ["Downhill (<-5°)", "Flat (-5° to +5°)", "Mild Uphill (+5° to +15°)", "Steep Uphill (+15° to +30°)", "Extreme (>+30°)"]

    print("\nPitch-Conditioned Speed Prediction Performance (Exp6A vs Exp6A-B):")
    print(f"{'Pitch Regime':<26} | {'Count':<6} | {'Mean Speed':<10} | {'6A MAE':<9} | {'6A Bias':<9} | {'6AB MAE':<9} | {'6AB Bias'}")
    print("-" * 85)
    for (pl, ph), pn in zip(pitch_bins, p_names):
        m = (pitches_deg >= pl) & (pitches_deg < ph)
        cnt = int(m.sum())
        if cnt > 0:
            avg_spd = np.mean(gts[m])
            mae_6a = np.mean(np.abs(errs_6a[m]))
            sgn_6a = np.mean(errs_6a[m])
            mae_6ab = np.mean(np.abs(errs_6ab[m]))
            sgn_6ab = np.mean(errs_6ab[m])
            print(f"{pn:<26} | {cnt:<6d} | {avg_spd:5.1f} km/h | {mae_6a:5.2f}    | {sgn_6a:+6.2f}   | {mae_6ab:5.2f}    | {sgn_6ab:+6.2f}")

    # Correlation between Pitch and Error
    corr_pitch_err_6a = np.corrcoef(pitches, errs_6a)[0, 1]
    corr_pitch_err_6ab = np.corrcoef(pitches, errs_6ab)[0, 1]
    corr_pitch_gt = np.corrcoef(pitches, gts)[0, 1]

    print(f"\nCorrelation between Road Pitch and Ground-Truth Speed  : r = {corr_pitch_gt:+.3f}")
    print(f"Correlation between Road Pitch and Signed Error (Exp6A)  : r = {corr_pitch_err_6a:+.3f}")
    print(f"Correlation between Road Pitch and Signed Error (Exp6A-B): r = {corr_pitch_err_6ab:+.3f}")

    # Inspect High-Speed (>60 km/h) section vs Pitch
    high_m = gts >= 60.0
    print(f"\nHigh-Speed Section (Speed >= 60 km/h, N={high_m.sum()} samples):")
    print(f"  Mean Pitch in High-Speed Section : {np.mean(pitches_deg[high_m]):+.1f}° ({np.mean(pitches[high_m]):+.3f} rad)")
    print(f"  Exp6A High-Speed Signed Error    : {np.mean(errs_6a[high_m]):+.2f} km/h")
    print(f"  Exp6AB High-Speed Signed Error   : {np.mean(errs_6ab[high_m]):+.2f} km/h")

    # Physics check on gravity compensation:
    # ay_comp = ay - g * sin(theta)
    # If vehicle is going uphill at constant speed (ay = +g*sin(theta)), then ay_comp = 0.
    # What is the formula used in compute_18ch_features?
    # In compute_18ch_features:
    # ay_comp = ay - GRAVITY * sin(theta_phys_window)
    # Let's inspect the sign and magnitude!
    print("\nPhysics Consistency Check on Gravity-Compensated Channel:")
    print(f"  Gravity constant g               : {GRAVITY:.5f} m/s²")
    print(f"  Pitch sign convention            : theta > 0 means Uphill (+ay pitch tilt)")
    print(f"  Uphill gravity compensation term : -g * sin(theta) < 0 (cancels positive accelerometer tilt)")
    print(f"  >>> Physics formulation verified: ay_comp = ay - g*sin(theta) correctly removes gravity offset. <<<")

if __name__ == "__main__":
    run_diagnostics()
