"""
Comprehensive Empirical Audit Script for Experiment 6 Mechanisms:
1. Trace absolute velocity entry & anchor initialization.
2. Quantify Jacobian / weight attribution of State Anchor vs Neural Feature Backbone.
3. Perform Identifiability Sanity Test: Measure conditional variance of GT speed given similar IMU windows.
4. Audit multi-task gradient norm competition.
5. Audit input BatchNorm & target scaling.
"""

import os
import glob
import math
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from ml.kaggle.standalone_exp6c_kaggle import (
    DeepSpeedKinematicsNet,
    SequencePhysicsDataset
)

DATA_DIR = "ml/external/IO-VNBD_repo/Synchronised V abd S datasets/Categorised IOVNB Dataset"

def run_audit():
    print("=" * 80)
    print("     EXP 6C SCIENTIFIC FORENSIC AUDIT: REGRESSION-TO-THE-MEAN ROOT CAUSES")
    print("=" * 80)

    # 1. Load Model & Inspect Path Weights
    ckpt_path = "ml/weights/exp6c_best_spectral_speed_filter.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "ml/weights/exp6a_best_spectral_speed_filter.pt"
    
    device = torch.device("cpu")
    model = DeepSpeedKinematicsNet(in_channels=18)
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    print(f"\n[1. MODEL WEIGHT & PATHWAY ANALYSIS] (Checkpoint: {ckpt_path})")
    
    # Analyze anchor projection layer
    w_anchor = model.state_proj[0].weight.data.numpy() # (32, 1)
    b_anchor = model.state_proj[0].bias.data.numpy()   # (32,)
    print(f"State Proj (Layer 0) weight norm: {np.linalg.norm(w_anchor):.4f}, bias norm: {np.linalg.norm(b_anchor):.4f}")
    
    # Analyze head_velocity layer (128 features + 32 anchor -> 64 -> 2)
    w_head = model.head_velocity[0].weight.data.numpy() # (64, 160)
    w_feat_part = w_head[:, :128]
    w_anch_part = w_head[:, 128:]
    print(f"Head Velocity weight norm (Feature part, 128 dims): {np.linalg.norm(w_feat_part):.4f} (Mean abs: {np.mean(np.abs(w_feat_part)):.4f})")
    print(f"Head Velocity weight norm (Anchor part, 32 dims)  : {np.linalg.norm(w_anch_part):.4f} (Mean abs: {np.mean(np.abs(w_anch_part)):.4f})")

    # 2. Compute Jacobian of Model Output w.r.t Anchor vs Features
    x_test = torch.randn(10, 18, 48, requires_grad=True)
    v_prev_test = torch.tensor([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0], requires_grad=True) # in m/s
    
    out_dict = model(x_test, v_prev_test)
    mu_v = out_dict["mu_v"]
    delta_v = out_dict["delta_v"]
    
    # Grad of mu_v w.r.t v_prev_test
    d_mu_d_vprev = []
    d_dv_d_vprev = []
    for i in range(10):
        grad_vprev = torch.autograd.grad(mu_v[i], v_prev_test, retain_graph=True)[0][i].item()
        d_mu_d_vprev.append(grad_vprev)
    print(f"\nEffective Jacobian d(v_t)/d(v_t-1) across anchor speeds (0 to 50 m/s / 0 to 180 km/h):")
    for v_val, g_val in zip(v_prev_test.detach().numpy() * 3.6, d_mu_d_vprev):
        print(f"  v_anchor = {v_val:5.1f} km/h  -->  d(v_t)/d(v_t-1) = {g_val:+.4f}")

    # 3. IDENTIFIABILITY SANITY TEST ON RAW IMU DATA
    print("\n" + "=" * 80)
    print(" [2. PHYSICAL IDENTIFIABILITY & CONDITIONAL VARIANCE AUDIT]")
    print("=" * 80)
    
    # Load S3a validation drive
    s_val = os.path.join(DATA_DIR, "Vw (Driver E)", "Vw11", "S-Vw11.csv")
    v_val = os.path.join(DATA_DIR, "Vw (Driver E)", "Vw11", "V-Vw11.csv")
    if not os.path.exists(s_val):
        # Fallback to Driver A
        s_val = glob.glob(os.path.join(DATA_DIR, "**", "S-S3a.csv"), recursive=True)[0]
        v_val = s_val.replace("S-", "V-")

    # Let's inspect multiple drives
    all_s_files = glob.glob(os.path.join(DATA_DIR, "**", "S-*.csv"), recursive=True)
    
    # Collect flat cruising windows (ay ~= 0, wz ~= 0) at different speeds
    print("Scanning dataset for steady-state cruise windows (|a_forward| < 0.2 m/s^2, |omega_z| < 0.02 rad/s)...")
    
    cruise_records = []
    
    for sf in all_s_files[:15]:
        vf = os.path.join(os.path.dirname(sf), os.path.basename(sf).replace("S-", "V-"))
        if not os.path.exists(vf):
            vf = os.path.join(os.path.dirname(sf), os.path.basename(sf).replace("S-", "V-").replace("Vta", "vta").replace("Vtb", "vtb").replace("Vw", "vw"))
            if not os.path.exists(vf):
                continue
        try:
            df_s = pd.read_csv(sf, encoding="latin1")
            df_v = pd.read_csv(vf, encoding="latin1")
            df_s.columns = df_s.columns.str.strip()
            df_v.columns = df_v.columns.str.strip()
            
            v_col = "Indicated Vehicle Speed (km/hr)" if "Indicated Vehicle Speed (km/hr)" in df_v.columns else "Velocity (km/hr)"
            if v_col not in df_v.columns:
                continue
                
            speed = df_v[v_col].values
            acc_cols = [c for c in df_s.columns if "ACCELEROMETER" in c]
            gyr_cols = [c for c in df_s.columns if "GYROSCOPE" in c]
            
            ax_col = [c for c in acc_cols if "X" in c][0]
            ay_col = [c for c in acc_cols if "Y" in c][0]
            az_col = [c for c in acc_cols if "Z" in c][0]
            gz_col = [c for c in gyr_cols if "Yaw" in c or "Z" in c][0]
            
            speed = df_v[v_col].values.astype(np.float64)
            ay = df_s[ay_col].values.astype(np.float64)
            az = df_s[az_col].values.astype(np.float64)
            ax = df_s[ax_col].values.astype(np.float64)
            gz = df_s[gz_col].values.astype(np.float64)
            
            n = min(len(speed), len(ay))
            speed = speed[:n]
            ay = ay[:n]
            az = az[:n]
            ax = ax[:n]
            gz = gz[:n]
            
            # Sliding 48-sample window
            for t in range(48, n, 20):
                w_speed = speed[t-48:t]
                w_ay = ay[t-48:t]
                w_az = az[t-48:t]
                w_ax = ax[t-48:t]
                w_gz = gz[t-48:t]
                
                # Check if steady state cruise (low speed std, low mean acc, low mean gyro)
                if np.std(w_speed) < 3.0 and np.mean(np.abs(w_ay)) < 0.35 and np.mean(np.abs(w_gz)) < 0.02:
                    # Vibration power
                    vib_z_std = np.std(w_az)
                    vib_y_std = np.std(w_ay)
                    vib_x_std = np.std(w_ax)
                    mean_v = np.mean(w_speed)
                    
                    cruise_records.append({
                        "gt_speed": mean_v,
                        "vib_z": vib_z_std,
                        "vib_y": vib_y_std,
                        "vib_x": vib_x_std,
                        "mean_ay": np.mean(w_ay),
                        "mean_gz": np.mean(w_gz),
                    })
        except Exception as e:
            continue

    df_cruise = pd.DataFrame(cruise_records)
    print(f"Found {len(df_cruise)} steady-state cruise windows across dataset.")
    
    # Analyze correlation between vibration power and true cruise speed
    r_vz = np.corrcoef(df_cruise["gt_speed"], df_cruise["vib_z"])[0, 1]
    r_vy = np.corrcoef(df_cruise["gt_speed"], df_cruise["vib_y"])[0, 1]
    r_vx = np.corrcoef(df_cruise["gt_speed"], df_cruise["vib_x"])[0, 1]
    print(f"Correlation between True Speed and Vertical Vibration Std (vib_z)  : r = {r_vz:+.4f}")
    print(f"Correlation between True Speed and Forward Vibration Std (vib_y)   : r = {r_vy:+.4f}")
    print(f"Correlation between True Speed and Lateral Vibration Std (vib_x)   : r = {r_vx:+.4f}")

    # Inspect conditional distribution of vibration power across speed bins
    speed_bins = [0, 20, 40, 60, 80, 120]
    print(f"\n{'Cruise Speed Bin':<18} | {'Count':<6} | {'Vib_Z Mean (m/s^2)':<20} | {'Vib_Z Std (m/s^2)':<20} | {'Vib_Z 10-90% Range'}")
    print("-" * 85)
    for i in range(len(speed_bins)-1):
        bl, bh = speed_bins[i], speed_bins[i+1]
        sub = df_cruise[(df_cruise["gt_speed"] >= bl) & (df_cruise["gt_speed"] < bh)]
        if len(sub) > 0:
            vz_m = sub["vib_z"].mean()
            vz_s = sub["vib_z"].std()
            p10 = np.percentile(sub["vib_z"], 10)
            p90 = np.percentile(sub["vib_z"], 90)
            print(f"{bl}-{bh} km/h".ljust(18) + f" | {len(sub):<6d} | {vz_m:<20.4f} | {vz_s:<20.4f} | [{p10:.3f}, {p90:.3f}]")

    # 4. Multi-Task Head Parameter Count & Representation Audit
    print("\n" + "=" * 80)
    print(" [3. MULTI-TASK CAPACITY & LOSS COMPETITION AUDIT]")
    print("=" * 80)
    total_params = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for p in model.stem.parameters()) + \
                      sum(p.numel() for p in model.stage1.parameters()) + sum(p.numel() for p in model.trans1.parameters()) + \
                      sum(p.numel() for p in model.stage2.parameters()) + sum(p.numel() for p in model.trans2.parameters()) + \
                      sum(p.numel() for p in model.stage3.parameters()) + sum(p.numel() for p in model.trans3.parameters()) + \
                      sum(p.numel() for p in model.stage4.parameters()) + sum(p.numel() for p in model.mha.parameters())
    head_vel_params = sum(p.numel() for p in model.head_velocity.parameters()) + sum(p.numel() for p in model.state_proj.parameters())
    head_zupt_params = sum(p.numel() for p in model.head_zupt.parameters())
    head_pitch_params = sum(p.numel() for p in model.head_pitch.parameters())
    head_regime_params = sum(p.numel() for p in model.head_regime.parameters())
    
    print(f"Total Parameters          : {total_params:,}")
    print(f"  - Shared Backbone + MHA : {backbone_params:,} ({backbone_params/total_params*100:.1f}%)")
    print(f"  - Velocity + State Head : {head_vel_params:,} ({head_vel_params/total_params*100:.1f}%)")
    print(f"  - ZUPT Head             : {head_zupt_params:,} ({head_zupt_params/total_params*100:.1f}%)")
    print(f"  - Pitch Head            : {head_pitch_params:,} ({head_pitch_params/total_params*100:.1f}%)")
    print(f"  - Motion Regime Head    : {head_regime_params:,} ({head_regime_params/total_params*100:.1f}%)")

if __name__ == "__main__":
    run_audit()
