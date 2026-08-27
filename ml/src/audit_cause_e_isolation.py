"""
Script:
1. Test 1 Variance Audit: Computes Var(GT), Std(GT), Std(Pred), Pearson r, Slope, Intercept per regime.
2. Spot-checks 80+ km/h values across Test 3 configs.
3. Cause E Head-Only Isolation Test: Freezes ConvNeXt backbone, fine-tunes only head_velocity and state_proj
   with Variance-Preserving Loss on training set, then evaluates closed-loop rollout on S-S3a.
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

def run_investigation():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[Accelerator]: {device}\n")

    ckpt_path = "ml/weights/exp6c_best_spectral_speed_filter.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "ml/weights/exp6a_best_spectral_speed_filter.pt"

    model = DeepSpeedKinematicsNet(in_channels=18).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # Load VAL
    val_dataset = SequencePhysicsDataset(DATA_DIR, split="val", seq_len=48, seq_stride=48)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)

    val_pooled_list, val_gts_list, val_zupt_list = [], [], []
    with torch.no_grad():
        for x_seq, targets in val_loader:
            x_seq = x_seq.to(device)
            v_gt_seq = targets["v"]
            B, L, C, W = x_seq.shape
            x_flat = x_seq.view(B * L, C, W)
            
            x_norm = model.input_norm(x_flat)
            feat = model.stem(x_norm)
            feat = model.trans1(model.stage1(feat))
            feat = model.trans2(model.stage2(feat))
            feat = model.trans3(model.stage3(feat))
            feat = model.stage4(feat)
            tokens = feat.permute(0, 2, 1)
            norm_tokens = model.mha_norm(tokens)
            attn_out, _ = model.mha(norm_tokens, norm_tokens, norm_tokens)
            tokens = tokens + attn_out
            pooled = model.pool_norm(tokens.mean(dim=1) + tokens[:, -1, :])
            p_zupt = torch.sigmoid(model.head_zupt(pooled).squeeze(-1))
            
            val_pooled_list.append(pooled.cpu())
            val_gts_list.append(v_gt_seq.view(-1) * 3.6)
            val_zupt_list.append(p_zupt.cpu())

    val_pooled = torch.cat(val_pooled_list, dim=0) # (24528, 128)
    val_gts = torch.cat(val_gts_list, dim=0).numpy() # (24528,)
    val_zupt = torch.cat(val_zupt_list, dim=0).numpy() # (24528,)
    N_val = len(val_gts)

    # 1. Baseline closed-loop rollout
    model.to("cpu")
    preds_base = []
    v_anc = 0.0
    with torch.no_grad():
        for i in range(N_val):
            feat_i = val_pooled[i:i+1]
            state_embed = model.state_proj(torch.tensor([[v_anc / 30.0]], dtype=torch.float32))
            fused = torch.cat([feat_i, state_embed], dim=-1)
            delta_v = model.head_velocity(fused)[0, 0].item()
            v_next = max(0.0, v_anc + delta_v)
            preds_base.append(v_next * 3.6)
            v_anc = v_next
    preds_base = np.array(preds_base)

    # Load S-S3a raw CSV for regime masks
    s3a_csv = os.path.join(DATA_DIR, "S (Driver A)", "S3a", "S-S3a.csv")
    v3a_csv = os.path.join(DATA_DIR, "S (Driver A)", "S3a", "V-S3a.csv")
    dfs = pd.read_csv(s3a_csv, encoding="latin1")
    dfv = pd.read_csv(v3a_csv, encoding="latin1")
    dfs.columns = dfs.columns.str.strip()
    dfv.columns = dfv.columns.str.strip()
    v_col = "Indicated Vehicle Speed (km/hr)" if "Indicated Vehicle Speed (km/hr)" in dfv.columns else "Velocity (km/hr)"
    az_col = [c for c in dfs.columns if "ACCELEROMETER" in c and "Z" in c][0]
    ay_col = [c for c in dfs.columns if "ACCELEROMETER" in c and "Y" in c][0]
    gz_col = [c for c in dfs.columns if "GYROSCOPE" in c and ("Yaw" in c or "Z" in c)][0]

    min_len = min(N_val, len(dfs), len(dfv))
    gts_clip = val_gts[:min_len]
    preds_clip = preds_base[:min_len]
    ay_clip = dfs[ay_col].values[:min_len]
    gz_clip = dfs[gz_col].values[:min_len]

    mask_stat = gts_clip < 1.0
    mask_turn = np.abs(gz_clip) >= 0.035
    mask_accel = np.abs(ay_clip) >= 0.5
    mask_dynamic = mask_turn | mask_accel
    mask_cruise = (np.abs(gz_clip) < 0.02) & (np.abs(ay_clip) < 0.2) & (gts_clip >= 10.0)

    print("=" * 115)
    print(" [1. RIGOROUS REGIME VARIANCE AUDIT ON S-S3a]")
    print("=" * 115)
    print(f"{'Regime':<28} | {'N':<6} | {'Var(GT)':<9} | {'Std(GT)':<9} | {'Std(Pred)':<10} | {'Pearson r':<9} | {'Slope (m)':<10} | {'Intercept'}")
    print("-" * 115)

    def print_regime_stats(name, mask):
        sub_g = gts_clip[mask]
        sub_p = preds_clip[mask]
        n = len(sub_g)
        if n < 5: return
        var_g = np.var(sub_g)
        std_g = np.std(sub_g)
        std_p = np.std(sub_p)
        r = np.corrcoef(sub_p, sub_g)[0, 1]
        slope = np.cov(sub_p, sub_g)[0, 1] / (var_g + 1e-6)
        intercept = np.mean(sub_p) - slope * np.mean(sub_g)
        mae = np.mean(np.abs(sub_p - sub_g))
        bias = np.mean(sub_p - sub_g)
        print(f"{name:<28} | {n:<6d} | {var_g:<9.2f} | {std_g:<9.2f} | {std_p:<10.2f} | {r:<9.4f} | {slope:<10.4f} | {intercept:+7.2f} km/h (MAE: {mae:4.2f})")

    print_regime_stats("ALL S-S3a Samples", np.ones(min_len, dtype=bool))
    print_regime_stats("1. Dynamic: Turning (|gz|>=0.035)", mask_turn)
    print_regime_stats("2. Dynamic: Accel/Brake (|ay|>=0.5)", mask_accel)
    print_regime_stats("3. ALL Dynamic Windows", mask_dynamic)
    print_regime_stats("4. Steady Cruise (>=10km/h)", mask_cruise)

    # 2. CAUSE E HEAD-ONLY ISOLATION TEST
    print("\n" + "=" * 115)
    print(" [2. CAUSE E ISOLATION TEST: FINE-TUNING ONLY HEAD_VELOCITY + STATE_PROJ WITH VARIANCE LOSS]")
    print("=" * 115)
    print("Extracting TRAIN backbone features (frozen ConvNeXt)...")

    model.to(device)
    train_dataset = SequencePhysicsDataset(DATA_DIR, split="train", seq_len=48, seq_stride=48)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False)

    train_pooled_list, train_gts_list = [], []
    with torch.no_grad():
        for x_seq, targets in train_loader:
            x_seq = x_seq.to(device)
            v_gt_seq = targets["v"]
            B, L, C, W = x_seq.shape
            x_flat = x_seq.view(B * L, C, W)
            
            x_norm = model.input_norm(x_flat)
            feat = model.stem(x_norm)
            feat = model.trans1(model.stage1(feat))
            feat = model.trans2(model.stage2(feat))
            feat = model.trans3(model.stage3(feat))
            feat = model.stage4(feat)
            tokens = feat.permute(0, 2, 1)
            norm_tokens = model.mha_norm(tokens)
            attn_out, _ = model.mha(norm_tokens, norm_tokens, norm_tokens)
            tokens = tokens + attn_out
            pooled = model.pool_norm(tokens.mean(dim=1) + tokens[:, -1, :])
            
            train_pooled_list.append(pooled.cpu().view(B, L, 128))
            train_gts_list.append(v_gt_seq.view(B, L))

    train_pooled = torch.cat(train_pooled_list, dim=0) # (N_seq, 48, 128)
    train_gts = torch.cat(train_gts_list, dim=0) # (N_seq, 48)
    N_train_seq = train_pooled.shape[0]
    print(f"Extracted {N_train_seq} training sequences ({N_train_seq*48} timesteps).")

    # Clone head layers
    torch.manual_seed(42)
    state_proj_tune = nn.Sequential(
        nn.Linear(1, 32),
        nn.GELU(),
        nn.Linear(32, 32)
    )
    state_proj_tune.load_state_dict(model.state_proj.state_dict())

    head_vel_tune = nn.Sequential(
        nn.Linear(160, 64),
        nn.GELU(),
        nn.Linear(64, 2)
    )
    head_vel_tune.load_state_dict(model.head_velocity.state_dict())

    state_proj_tune.to(device)
    head_vel_tune.to(device)

    opt = torch.optim.AdamW(list(state_proj_tune.parameters()) + list(head_vel_tune.parameters()), lr=1e-3, weight_decay=1e-4)

    # Train head for 5 epochs with Smooth L1 + Variance Preserving Loss (lambda_var=0.5)
    train_pooled_dev = train_pooled.to(device)
    train_gts_dev = train_gts.to(device)
    batch_size = 64
    num_batches = (N_train_seq + batch_size - 1) // batch_size

    print("\nFine-tuning output head with Variance Loss on frozen ConvNeXt features...")
    for epoch in range(1, 6):
        perm = torch.randperm(N_train_seq)
        total_loss, total_huber, total_var = 0.0, 0.0, 0.0
        
        for b in range(num_batches):
            idx = perm[b*batch_size : (b+1)*batch_size]
            b_feat = train_pooled_dev[idx] # (B, 48, 128)
            b_gt = train_gts_dev[idx] # (B, 48)
            B_curr = b_feat.shape[0]
            
            v_state = torch.zeros(B_curr, 1, device=device)
            preds_seq = []
            for t in range(48):
                f_t = b_feat[:, t] # (B, 128)
                s_t = state_proj_tune(v_state / 30.0) # (B, 32)
                fused_t = torch.cat([f_t, s_t], dim=-1) # (B, 160)
                delta_v = head_vel_tune(fused_t)[:, 0:1] # (B, 1)
                v_next = F.relu(v_state + delta_v)
                preds_seq.append(v_next)
                v_state = v_next.detach()
                
            pred_tensor = torch.stack(preds_seq, dim=1).squeeze(-1) # (B, 48)
            
            l_huber = F.smooth_l1_loss(pred_tensor, b_gt)
            # Variance preserving loss: penalize if std of pred < std of gt
            pred_std = torch.std(pred_tensor, dim=1)
            gt_std = torch.std(b_gt, dim=1)
            l_var = F.mse_loss(pred_std, gt_std)
            
            loss = l_huber + 0.5 * l_var
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            total_huber += l_huber.item()
            total_var += l_var.item()
            
        print(f"  Epoch {epoch}/5: Loss={total_loss/num_batches:.4f} (Huber={total_huber/num_batches:.4f}, VarLoss={total_var/num_batches:.4f})")

    # Evaluate Fine-Tuned Head on S-S3a Closed Loop Rollout
    state_proj_tune.eval().to("cpu")
    head_vel_tune.eval().to("cpu")

    preds_tuned = []
    v_anc = 0.0
    with torch.no_grad():
        for i in range(N_val):
            feat_i = val_pooled[i:i+1]
            s_i = state_proj_tune(torch.tensor([[v_anc / 30.0]], dtype=torch.float32))
            fused_i = torch.cat([feat_i, s_i], dim=-1)
            delta_v = head_vel_tune(fused_i)[0, 0].item()
            v_next = max(0.0, v_anc + delta_v)
            preds_tuned.append(v_next * 3.6)
            v_anc = v_next

    p_tuned = np.array(preds_tuned)
    g_arr = val_gts

    mae_tuned = np.mean(np.abs(p_tuned - g_arr))
    bias_tuned = np.mean(p_tuned - g_arr)
    slope_tuned = np.cov(p_tuned, g_arr)[0, 1] / (np.var(g_arr) + 1e-6)
    int_tuned = np.mean(p_tuned) - slope_tuned * np.mean(g_arr)
    r_tuned = np.corrcoef(p_tuned, g_arr)[0, 1]

    # Speed bins
    bins = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 80), (80, 300)]
    bin_names = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-80", "80+"]
    b_maes_tuned, b_signed_tuned = [], {}
    b_maes_base, b_signed_base = [], {}
    for (bl, bh), bn in zip(bins, bin_names):
        m_b = (g_arr >= bl) & (g_arr < bh)
        if np.sum(m_b) > 0:
            b_maes_tuned.append(np.mean(np.abs(p_tuned[m_b] - g_arr[m_b])))
            b_signed_tuned[bn] = np.mean(p_tuned[m_b] - g_arr[m_b])
            b_maes_base.append(np.mean(np.abs(preds_base[m_b] - g_arr[m_b])))
            b_signed_base[bn] = np.mean(preds_base[m_b] - g_arr[m_b])
        else:
            b_maes_tuned.append(0.0); b_signed_tuned[bn] = 0.0
            b_maes_base.append(0.0); b_signed_base[bn] = 0.0

    print("\n" + "=" * 115)
    print(" [CAUSE E ISOLATION RESULT: FROZEN BACKBONE vs. VARIANCE-TUNED HEAD]")
    print("=" * 115)
    print(f"BASELINE HEAD:     Slope = {np.cov(preds_base, g_arr)[0,1]/(np.var(g_arr)+1e-6):.4f} | Intercept = {np.mean(preds_base) - (np.cov(preds_base, g_arr)[0,1]/(np.var(g_arr)+1e-6))*np.mean(g_arr):+6.2f} | MAE = {np.mean(np.abs(preds_base-g_arr)):5.2f} | r = {np.corrcoef(preds_base, g_arr)[0,1]:.4f}")
    print(f"                   0-10 km/h: Bias = {b_signed_base['0-10']:+6.2f} km/h | 80+ km/h: Bias = {b_signed_base['80+']:+6.2f} km/h (MAE: {b_maes_base[7]:5.2f})")
    print("-" * 115)
    print(f"VARIANCE-TUNED:    Slope = {slope_tuned:.4f} | Intercept = {int_tuned:+6.2f} | MAE = {mae_tuned:5.2f} | r = {r_tuned:.4f}")
    print(f"                   0-10 km/h: Bias = {b_signed_tuned['0-10']:+6.2f} km/h | 80+ km/h: Bias = {b_signed_tuned['80+']:+6.2f} km/h (MAE: {b_maes_tuned[7]:5.2f})")
    print("=" * 115)

if __name__ == "__main__":
    run_investigation()
