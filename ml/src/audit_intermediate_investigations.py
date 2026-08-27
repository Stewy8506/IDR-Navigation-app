"""
Intermediate Investigations:
1. Exact sigma(pred) vs sigma(gt) for Cause E isolation test.
2. p_ZUPT calibration curve in 0-15 km/h pull-away transition band on S-S3a.
3. Unfrozen end-to-end training of ConvNeXt backbone with Variance-Preserving Loss on M4 GPU (3 epochs),
   evaluating closed-loop rollout slope, intercept, sigma(pred), sigma(gt), MAE, and 80+ error.
"""

import os
import glob
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from ml.kaggle.standalone_exp6c_kaggle import (
    DeepSpeedKinematicsNet,
    SequencePhysicsDataset
)

DATA_DIR = "ml/external/IO-VNBD_repo/Synchronised V abd S datasets/Categorised IOVNB Dataset"

def run_intermediate_tests():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running Intermediate Investigations on: {device}\n")

    ckpt_path = "ml/weights/exp6c_best_spectral_speed_filter.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "ml/weights/exp6a_best_spectral_speed_filter.pt"

    # =========================================================================
    # PART 1: EVALUATE BASELINE & CAUSE E FROZEN HEAD VARIANCE
    # =========================================================================
    print("=" * 115)
    print(" [PART 1: CAUSE E ISOLATION - DIRECT SIGMA(PRED) vs SIGMA(GT) COMPARISON]")
    print("=" * 115)
    
    model = DeepSpeedKinematicsNet(in_channels=18).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    val_dataset = SequencePhysicsDataset(DATA_DIR, split="val", seq_len=48, seq_stride=48)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)

    val_x_list, val_gts_list = [], []
    with torch.no_grad():
        for x_seq, targets in val_loader:
            val_x_list.append(x_seq)
            val_gts_list.append(targets["v"])

    all_val_x = torch.cat(val_x_list, dim=0) # (511, 48, 18, 48)
    all_val_gts = torch.cat(val_gts_list, dim=0) * 3.6 # (511, 48) in km/h
    N_val_seq = all_val_x.shape[0]

    # Baseline Closed-Loop Rollout
    preds_base_list, p_zupt_all_list = [], []
    v_anc = 0.0
    with torch.no_grad():
        for seq_i in range(N_val_seq):
            x_seq_i = all_val_x[seq_i].to(device) # (48, 18, 48)
            for t in range(48):
                x_t = x_seq_i[t:t+1] # (1, 18, 48)
                # v_anchor must be raw m/s — model normalizes internally via /30.0
                out = model(x_t, v_anchor=torch.tensor([v_anc], device=device, dtype=torch.float32))
                v_pred_mps = out["mu_v"].item()  # already ReLU'd, >= 0
                p_z = out["p_zupt"].item()
                preds_base_list.append(v_pred_mps * 3.6)
                p_zupt_all_list.append(p_z)
                v_anc = v_pred_mps

    p_base = np.array(preds_base_list)
    g_val = all_val_gts.view(-1).numpy()
    p_zupt_arr = np.array(p_zupt_all_list)

    sigma_gt = np.std(g_val)
    sigma_base = np.std(p_base)
    slope_base = np.cov(p_base, g_val)[0, 1] / (np.var(g_val) + 1e-6)
    int_base = np.mean(p_base) - slope_base * np.mean(g_val)
    r_base = np.corrcoef(p_base, g_val)[0, 1]

    print(f"GROUND TRUTH:      Mean = {np.mean(g_val):5.2f} km/h | σ(GT) = {sigma_gt:5.2f} km/h | Var(GT) = {np.var(g_val):6.2f}")
    print(f"BASELINE EXP6C:    Mean = {np.mean(p_base):5.2f} km/h | σ(Pred) = {sigma_base:5.2f} km/h | Ratio σ(P)/σ(GT) = {sigma_base/sigma_gt:6.4f}")
    print(f"                   Slope = {slope_base:6.4f} | Intercept = {int_base:+6.2f} km/h | Pearson r = {r_base:6.4f} | MAE = {np.mean(np.abs(p_base - g_val)):5.2f} km/h")
    print(f"                   0-10 km/h Bias: {np.mean(p_base[g_val < 10] - g_val[g_val < 10]):+6.2f} km/h | 80+ km/h Bias: {np.mean(p_base[g_val >= 80] - g_val[g_val >= 80]):+6.2f} km/h")

    # =========================================================================
    # PART 2: P_ZUPT CALIBRATION IN 0-15 KM/H TRANSITION BAND
    # =========================================================================
    print("\n" + "=" * 115)
    print(" [PART 2: P_ZUPT CALIBRATION VS. TRUE GT SPEED IN 0-15 KM/H TRANSITION BAND]")
    print("=" * 115)
    print(f"{'Speed Range (km/h)':<20} | {'Sample Count':<12} | {'Mean p_ZUPT':<14} | {'Median p_ZUPT':<14} | {'P10':<8} | {'P90':<8} | {'% > 0.5':<10} | {'% > 0.1'}")
    print("-" * 115)

    speed_bins_zupt = [
        (0.0, 0.5),
        (0.5, 1.0),
        (1.0, 2.0),
        (2.0, 3.0),
        (3.0, 5.0),
        (5.0, 8.0),
        (8.0, 10.0),
        (10.0, 15.0),
        (15.0, 25.0),
        (25.0, 120.0)
    ]

    for (low, high) in speed_bins_zupt:
        mask = (g_val >= low) & (g_val < high)
        n = np.sum(mask)
        if n > 0:
            z_sub = p_zupt_arr[mask]
            m_z = np.mean(z_sub)
            med_z = np.median(z_sub)
            p10 = np.percentile(z_sub, 10)
            p90 = np.percentile(z_sub, 90)
            pct50 = np.mean(z_sub > 0.5) * 100
            pct10 = np.mean(z_sub > 0.1) * 100
            label = f"[{low:4.1f}, {high:4.1f}) km/h"
            print(f"{label:<20} | {n:<12d} | {m_z:<14.4f} | {med_z:<14.4f} | {p10:<8.4f} | {p90:<8.4f} | {pct50:<9.1f}% | {pct10:<6.1f}%")
        else:
            print(f"[{low:4.1f}, {high:4.1f}) km/h | N=0")

    # =========================================================================
    # PART 3: UNFREEZE BACKBONE & JOINTLY TRAIN WITH VARIANCE LOSS
    # =========================================================================
    print("\n" + "=" * 115)
    print(" [PART 3: UNFREEZE CONVNEXT BACKBONE - END-TO-END TRAINING WITH VARIANCE-PRESERVING LOSS (3 EPOCHS)]")
    print("=" * 115)
    print("Loading TRAIN dataset and initializing end-to-end model...")

    train_dataset = SequencePhysicsDataset(DATA_DIR, split="train", seq_len=48, seq_stride=48)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)

    # Initialize fresh model from Exp6C checkpoint
    model_e2e = DeepSpeedKinematicsNet(in_channels=18).to(device)
    model_e2e.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    
    optimizer = torch.optim.AdamW(model_e2e.parameters(), lr=1e-4, weight_decay=1e-4)

    print("Training end-to-end model on M4 GPU for 3 epochs with Smooth L1 + Variance-Preserving Loss (λ_var=0.5)...")
    for epoch in range(1, 4):
        model_e2e.train()
        total_loss, total_huber, total_var = 0.0, 0.0, 0.0
        n_batches = 0
        
        for batch_x, batch_targets in train_loader:
            x = batch_x.to(device) # (B, 48, 18, 48)
            v_gt = batch_targets["v"].to(device) # (B, 48)
            B, L, C, W = x.shape
            
            v_state = torch.zeros(B, device=device, dtype=torch.float32)
            preds_seq = []
            
            for t in range(L):
                out_t = model_e2e(x[:, t], v_anchor=v_state)
                mu_t = out_t["mu_v"]
                preds_seq.append(mu_t)
                v_state = mu_t.detach()
                
            pred_tensor = torch.stack(preds_seq, dim=1) # (B, 48) in m/s
            
            l_huber = F.smooth_l1_loss(pred_tensor, v_gt)
            # Variance loss across sequence and batch
            p_std = torch.std(pred_tensor, dim=1)
            g_std = torch.std(v_gt, dim=1)
            l_var = F.mse_loss(p_std, g_std)
            
            loss = l_huber + 0.5 * l_var
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_e2e.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_huber += l_huber.item()
            total_var += l_var.item()
            n_batches += 1
            
        print(f"  Epoch {epoch}/3: Total Loss = {total_loss/n_batches:.4f} | Huber = {total_huber/n_batches:.4f} | Var Loss = {total_var/n_batches:.4f}")

    # Evaluate Jointly Trained Model on S-S3a Closed-Loop Rollout
    model_e2e.eval()
    preds_e2e_list = []
    v_anc = 0.0
    with torch.no_grad():
        for seq_i in range(N_val_seq):
            x_seq_i = all_val_x[seq_i].to(device)
            for t in range(48):
                x_t = x_seq_i[t:t+1]
                # v_anchor must be raw m/s — model normalizes internally via /30.0
                out = model_e2e(x_t, v_anchor=torch.tensor([v_anc], device=device, dtype=torch.float32))
                v_pred_mps = out["mu_v"].item()
                preds_e2e_list.append(v_pred_mps * 3.6)
                v_anc = v_pred_mps

    p_e2e = np.array(preds_e2e_list)
    sigma_e2e = np.std(p_e2e)
    slope_e2e = np.cov(p_e2e, g_val)[0, 1] / (np.var(g_val) + 1e-6)
    int_e2e = np.mean(p_e2e) - slope_e2e * np.mean(g_val)
    r_e2e = np.corrcoef(p_e2e, g_val)[0, 1]
    mae_e2e = np.mean(np.abs(p_e2e - g_val))

    bins = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 80), (80, 300)]
    bin_names = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-80", "80+"]
    b_maes_e2e, b_signed_e2e = [], {}
    for (bl, bh), bn in zip(bins, bin_names):
        m_b = (g_val >= bl) & (g_val < bh)
        if np.sum(m_b) > 0:
            b_maes_e2e.append(np.mean(np.abs(p_e2e[m_b] - g_val[m_b])))
            b_signed_e2e[bn] = np.mean(p_e2e[m_b] - g_val[m_b])
        else:
            b_maes_e2e.append(0.0); b_signed_e2e[bn] = 0.0

    print("\n" + "=" * 115)
    print(" [FINAL INTERMEDIATE COMPARISON: BASELINE EXP6C vs. JOINT END-TO-END VARIANCE TRAINING]")
    print("=" * 115)
    print(f"BASELINE EXP6C:    Slope = {slope_base:6.4f} | Intercept = {int_base:+6.2f} km/h | σ(P) = {sigma_base:5.2f} (σ_GT={sigma_gt:5.2f}) | r = {r_base:6.4f} | MAE = {np.mean(np.abs(p_base - g_val)):5.2f} km/h")
    print(f"                   0-10 km/h Bias: {np.mean(p_base[g_val < 10] - g_val[g_val < 10]):+6.2f} km/h | 80+ km/h Bias: {np.mean(p_base[g_val >= 80] - g_val[g_val >= 80]):+6.2f} km/h (MAE: {np.mean(np.abs(p_base[g_val>=80] - g_val[g_val>=80])):5.2f})")
    print("-" * 115)
    print(f"JOINT END-TO-END:  Slope = {slope_e2e:6.4f} | Intercept = {int_e2e:+6.2f} km/h | σ(P) = {sigma_e2e:5.2f} (σ_GT={sigma_gt:5.2f}) | r = {r_e2e:6.4f} | MAE = {mae_e2e:5.2f} km/h")
    print(f"                   0-10 km/h Bias: {b_signed_e2e['0-10']:+6.2f} km/h | 80+ km/h Bias: {b_signed_e2e['80+']:+6.2f} km/h (MAE: {b_maes_e2e[7]:5.2f})")
    print("=" * 115)

if __name__ == "__main__":
    run_intermediate_tests()
