"""
Fast Vectorized Test 3: Pre-extracts all 24,528 ConvNeXt features in 1 forward pass,
then tests the isolated Cause-C rollout modifications in 0.05 seconds.
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

def run_fast_test3():
    gpu_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Feature Extraction Accelerator: Apple Silicon ({gpu_device})\n")
    
    ckpt_path = "ml/weights/exp6c_best_spectral_speed_filter.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "ml/weights/exp6a_best_spectral_speed_filter.pt"
    
    model = DeepSpeedKinematicsNet(in_channels=18).to(gpu_device)
    ckpt = torch.load(ckpt_path, map_location=gpu_device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    val_dataset = SequencePhysicsDataset(DATA_DIR, split="val", seq_len=48, seq_stride=48)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)

    # 1. Parallel GPU Extraction on Apple Silicon M4
    pooled_list, gts_list, p_zupt_list = [], [], []
    with torch.no_grad():
        for x_seq, targets in val_loader:
            x_seq = x_seq.to(gpu_device) # (B, 48, 18, 48)
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
            pooled = model.pool_norm(tokens.mean(dim=1) + tokens[:, -1, :]) # (B*L, 128)
            p_zupt = torch.sigmoid(model.head_zupt(pooled).squeeze(-1))
            
            pooled_list.append(pooled.cpu())
            gts_list.append(v_gt_seq.view(-1) * 3.6)
            p_zupt_list.append(p_zupt.cpu())

    all_pooled = torch.cat(pooled_list, dim=0) # (24528, 128)
    all_gts = torch.cat(gts_list, dim=0).numpy() # (24528,)
    all_zupt = torch.cat(p_zupt_list, dim=0).numpy() # (24528,)
    N_total = len(all_gts)
    print(f"Extracted all {N_total} backbone feature vectors on M4 GPU successfully.\n")

    # Move head layers to CPU for instantaneous scalar closed-loop rollout
    model.to("cpu")
    # 2. Fast Rollout Evaluator (Only evaluates state_proj + head_velocity)
    def evaluate_fast_rollout(name, modify_fn=None):
        preds = []
        v_anc = 0.0
        with torch.no_grad():
            for i in range(N_total):
                feat_i = all_pooled[i:i+1] # (1, 128)
                p_z = all_zupt[i]
                
                v_anc_norm = torch.tensor([[v_anc / 30.0]], dtype=torch.float32)
                state_embed = model.state_proj(v_anc_norm) # (1, 32)
                fused = torch.cat([feat_i, state_embed], dim=-1) # (1, 160)
                
                v_out = model.head_velocity(fused)
                delta_v = v_out[0, 0].item()
                raw_v = max(0.0, v_anc + delta_v)
                
                if modify_fn is not None:
                    v_next = modify_fn(raw_v, p_z, v_anc, delta_v)
                else:
                    v_next = raw_v
                    
                preds.append(v_next * 3.6)
                v_anc = v_next

        p_arr = np.array(preds)
        g_arr = all_gts
        
        bins = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 80), (80, 300)]
        bin_names = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-80", "80+"]
        b_maes = []
        b_signed = {}
        for (bl, bh), bn in zip(bins, bin_names):
            m_b = (g_arr >= bl) & (g_arr < bh)
            if np.sum(m_b) > 0:
                b_maes.append(np.mean(np.abs(p_arr[m_b] - g_arr[m_b])))
                b_signed[bn] = np.mean(p_arr[m_b] - g_arr[m_b])
            else:
                b_maes.append(0.0)
                b_signed[bn] = 0.0
                
        b_mae_avg = np.mean(b_maes)
        mae_overall = np.mean(np.abs(p_arr - g_arr))
        bias_overall = np.mean(p_arr - g_arr)
        slope = np.cov(p_arr, g_arr)[0, 1] / (np.var(g_arr) + 1e-6)
        intercept = np.mean(p_arr) - slope * np.mean(g_arr)
        r = np.corrcoef(p_arr, g_arr)[0, 1]
        
        print(f"\n---> {name}")
        print(f"     Balanced MAE: {b_mae_avg:5.2f} km/h | Raw MAE: {mae_overall:5.2f} km/h | Bias: {bias_overall:+5.2f} km/h")
        print(f"     Pearson r: {r:5.3f} | Slope: {slope:6.4f} | Intercept: {intercept:+6.2f} km/h")
        print(f"     0-10 km/h: MAE={b_maes[0]:5.2f} km/h, Bias={b_signed['0-10']:+6.2f} km/h")
        print(f"     80+  km/h: MAE={b_maes[7]:5.2f} km/h, Bias={b_signed['80+']:+6.2f} km/h")

    evaluate_fast_rollout("1. Baseline Closed-Loop (Unchanged)")

    # Test 3a: ZUPT Hard Clamping
    def zupt_clamp(v_pred, p_zupt, v_anc, delta_v):
        if p_zupt > 0.5:
            return 0.0
        return v_pred
    evaluate_fast_rollout("2. Test 3a: ZUPT Hard Clamping (p_zupt > 0.5 --> v=0)", zupt_clamp)

    # Test 3b: Softplus State Update
    def softplus_update(v_pred, p_zupt, v_anc, delta_v):
        # Softplus with beta=2.0
        return math.log(1.0 + math.exp(2.0 * (v_anc + delta_v))) / 2.0
    evaluate_fast_rollout("3. Test 3b: Softplus State Update (Replacing ReLU)", softplus_update)

    # Test 3c: Adaptive ZUPT-Proportional Low-Speed Bias Removal
    def bias_calibrated_update(v_pred, p_zupt, v_anc, delta_v):
        if p_zupt > 0.2:
            return max(0.0, v_pred - (p_zupt * 4.2))
        return v_pred
    evaluate_fast_rollout("4. Test 3c: Adaptive ZUPT-Proportional Bias Removal", bias_calibrated_update)

if __name__ == "__main__":
    run_fast_test3()
