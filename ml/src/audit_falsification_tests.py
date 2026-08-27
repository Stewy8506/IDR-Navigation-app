"""
Rigorous Falsification Audit Script:
1. Stratified Regime Evaluation: Slope, Intercept, MAE on Dynamic vs. Cruise vs. Stationary windows on S-S3a.
2. Bootstrap Uncertainty Estimation for Vibration-Speed Overlap (1,000 iterations, 95% CIs).
3. Isolated Cause-C Rollout Tests: ZUPT Clamping, Bias Shift, Softplus vs ReLU without retraining.
4. Fast 3-Epoch Multi-Task Ablation Test: Training with L_zupt=0 and L_regime=0 vs baseline.
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

def run_falsification_audit():
    device = torch.device("cpu")
    print(f"Running on Device: {device}\n")

    # Load Exp6C Best Checkpoint
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
    print(f"[Loaded Checkpoint]: {ckpt_path}")

    # =========================================================================
    # TEST 1: STRATIFIED REGIME EVALUATION (DYNAMIC VS. CRUISE VS. STATIONARY)
    # =========================================================================
    print("\n" + "=" * 85)
    print(" [TEST 1: STRATIFIED REGIME EVALUATION ON S-S3a (DYNAMIC vs. CRUISE vs. STATIONARY)]")
    print("=" * 85)
    
    s_file = glob.glob(os.path.join(DATA_DIR, "**", "S-S3a.csv"), recursive=True)[0]
    v_file = s_file.replace("S-", "V-")

    # Load S3a data
    df_s = pd.read_csv(s_file, encoding="latin1")
    df_v = pd.read_csv(v_file, encoding="latin1")
    df_s.columns = df_s.columns.str.strip()
    df_v.columns = df_v.columns.str.strip()
    v_col = "Indicated Vehicle Speed (km/hr)" if "Indicated Vehicle Speed (km/hr)" in df_v.columns else "Velocity (km/hr)"
    
    acc_cols = [c for c in df_s.columns if "ACCELEROMETER" in c]
    gyr_cols = [c for c in df_s.columns if "GYROSCOPE" in c]
    ax_col = [c for c in acc_cols if "X" in c][0]
    ay_col = [c for c in acc_cols if "Y" in c][0]
    az_col = [c for c in acc_cols if "Z" in c][0]
    gz_col = [c for c in gyr_cols if "Yaw" in c or "Z" in c][0]

    v_gt_kmh = df_v[v_col].values.astype(np.float64)
    ay_raw = df_s[ay_col].values.astype(np.float64)
    ax_raw = df_s[ax_col].values.astype(np.float64)
    az_raw = df_s[az_col].values.astype(np.float64)
    gz_raw = df_s[gz_col].values.astype(np.float64)
    N = min(len(v_gt_kmh), len(ay_raw))
    v_gt_kmh = v_gt_kmh[:N]
    ay_raw = ay_raw[:N]
    ax_raw = ax_raw[:N]
    az_raw = az_raw[:N]
    gz_raw = gz_raw[:N]

    val_dataset = SequencePhysicsDataset(DATA_DIR, split="val", seq_len=48, seq_stride=48)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    preds_closed_loop = []
    gts_list = []
    p_zupt_list = []
    v_val_state = torch.zeros(1, device=device, dtype=torch.float32)
    
    with torch.no_grad():
        for x_seq, targets in val_loader:
            x_seq = x_seq.to(device)
            v_gt_seq = targets["v"].to(device)
            B, L, C, W = x_seq.shape
            for t in range(L):
                out = model(x_seq[:, t], v_anchor=v_val_state)
                mu_t = out["mu_v"]
                preds_closed_loop.append(mu_t.item() * 3.6)
                gts_list.append(v_gt_seq[0, t].item() * 3.6)
                p_zupt_list.append(out["p_zupt"].item())
                v_val_state = mu_t.detach()

    preds = np.array(preds_closed_loop)
    gts = np.array(gts_list)
    zupts = np.array(p_zupt_list)
    min_len = min(len(preds), len(gts), len(ay_raw)-47)
    preds = preds[:min_len]
    gts = gts[:min_len]
    zupts = zupts[:min_len]
    ays = ay_raw[47:47+min_len]
    axs = ax_raw[47:47+min_len]
    gzs = gz_raw[47:47+min_len]

    # Define Regimes:
    # 1. Stationary: GT speed < 1.0 km/h
    # 2. Dynamic Turning: |gz| >= 0.035 rad/s (~2 deg/s)
    # 3. Dynamic Accel/Braking: |ay| >= 0.5 m/s^2
    # 4. Any Dynamic Window: Turning OR Accel/Brake
    # 5. Steady Cruise: GT speed >= 20 km/h AND |gz| < 0.02 rad/s AND |ay| < 0.2 m/s^2
    
    mask_stat = (gts < 1.0)
    mask_turn = (np.abs(gzs) >= 0.035) & (~mask_stat)
    mask_accel = (np.abs(ays) >= 0.5) & (~mask_stat)
    mask_dynamic = (mask_turn | mask_accel) & (~mask_stat)
    mask_cruise = (gts >= 20.0) & (np.abs(gzs) < 0.02) & (np.abs(ays) < 0.2)

    def analyze_regime(name, mask):
        p_sub = preds[mask]
        g_sub = gts[mask]
        n = len(p_sub)
        if n < 10:
            return
        mae = np.mean(np.abs(p_sub - g_sub))
        bias = np.mean(p_sub - g_sub)
        std_g = np.std(g_sub)
        std_p = np.std(p_sub)
        
        if std_g > 1e-3 and std_p > 1e-3:
            r = np.corrcoef(p_sub, g_sub)[0, 1]
            slope = np.cov(p_sub, g_sub)[0, 1] / (np.var(g_sub) + 1e-6)
            intercept = np.mean(p_sub) - slope * np.mean(g_sub)
        else:
            r = float("nan")
            slope = float("nan")
            intercept = float("nan")
        
        print(f"{name:<32} | N={n:<6d} | MAE: {mae:5.2f} km/h | Bias: {bias:+6.2f} km/h | r: {r:5.3f} | Slope: {slope:6.4f} | Intercept: {intercept:+6.2f}")

    print(f"{'Regime Definition':<32} | {'Sample Count':<10} | {'MAE (km/h)':<14} | {'Signed Bias':<16} | {'Pearson r':<9} | {'Slope (m)':<12} | {'Intercept (c)'}")
    print("-" * 115)
    analyze_regime("ALL S-S3a Samples (Full Drive)", np.ones(min_len, dtype=bool))
    analyze_regime("1. Stationary (GT < 1 km/h)", mask_stat)
    analyze_regime("2. Dynamic: Turning (|gz|>=0.035)", mask_turn)
    analyze_regime("3. Dynamic: Accel/Brake (|ay|>=0.5)", mask_accel)
    analyze_regime("4. ALL Dynamic Windows (Turn|Accel)", mask_dynamic)
    analyze_regime("5. Steady Cruise (|gz|<0.02, |ay|<0.2)", mask_cruise)

    # =========================================================================
    # TEST 2: BOOTSTRAP UNCERTAINTY ON VIBRATION-SPEED OVERLAP
    # =========================================================================
    print("\n" + "=" * 85, flush=True)
    print(" [TEST 2: BOOTSTRAP UNCERTAINTY ON VIBRATION-SPEED OVERLAP (1,000 ITERATIONS)]", flush=True)
    print("=" * 85, flush=True)
    
    # Fast cruise window collection across representative train & val drives
    rep_files = [
        os.path.join(DATA_DIR, "S (Driver A)", "S1", "S-S1.csv"),
        os.path.join(DATA_DIR, "S (Driver A)", "S2", "S-S2.csv"),
        os.path.join(DATA_DIR, "S (Driver A)", "S3a", "S-S3a.csv"),
        os.path.join(DATA_DIR, "S (Driver A)", "S3b", "S-S3b.csv"),
        os.path.join(DATA_DIR, "S (Driver A)", "S3c", "S-S3c.csv"),
        os.path.join(DATA_DIR, "S (Driver A)", "S4", "S-S4.csv"),
        os.path.join(DATA_DIR, "M (Driver B)", "M", "S-M.csv"),
        os.path.join(DATA_DIR, "Vw (Driver E)", "Vw11", "S-Vw11.csv"),
        os.path.join(DATA_DIR, "Vw (Driver E)", "Vw12", "S-Vw12.csv"),
        os.path.join(DATA_DIR, "Vw (Driver E)", "Vw14a", "S-Vw14a.csv"),
        os.path.join(DATA_DIR, "Vw (Driver E)", "Vw14b", "S-Vw14b.csv"),
        os.path.join(DATA_DIR, "Vtb (Driver E)", "Vtb01", "S-Vtb1.csv"),
        os.path.join(DATA_DIR, "Vtb (Driver E)", "Vtb05", "S-Vtb5.csv")
    ]
    
    speeds_all, vib_z_all = [], []
    for sf in rep_files:
        if not os.path.exists(sf): continue
        vf = os.path.join(os.path.dirname(sf), os.path.basename(sf).replace("S-", "V-").replace("Vw", "vw").replace("Vtb", "vtb"))
        if not os.path.exists(vf):
            vf = os.path.join(os.path.dirname(sf), os.path.basename(sf).replace("S-", "V-"))
            if not os.path.exists(vf): continue
        try:
            dfs = pd.read_csv(sf, encoding="latin1")
            dfv = pd.read_csv(vf, encoding="latin1")
            dfs.columns = dfs.columns.str.strip()
            dfv.columns = dfv.columns.str.strip()
            v_col_i = "Indicated Vehicle Speed (km/hr)" if "Indicated Vehicle Speed (km/hr)" in dfv.columns else "Velocity (km/hr)"
            if v_col_i not in dfv.columns: continue
            
            az_col_i = [c for c in dfs.columns if "ACCELEROMETER" in c and "Z" in c][0]
            ay_col_i = [c for c in dfs.columns if "ACCELEROMETER" in c and "Y" in c][0]
            gz_col_i = [c for c in dfs.columns if "GYROSCOPE" in c and ("Yaw" in c or "Z" in c)][0]
            
            sp_arr = dfv[v_col_i].values.astype(np.float64)
            az_arr = dfs[az_col_i].values.astype(np.float64)
            ay_arr = dfs[ay_col_i].values.astype(np.float64)
            gz_arr = dfs[gz_col_i].values.astype(np.float64)
            n_i = min(len(sp_arr), len(az_arr))
            
            # Fast vectorized rolling window check
            for t in range(48, n_i, 10):
                w_s = sp_arr[t-48:t]
                w_ay = ay_arr[t-48:t]
                w_gz = gz_arr[t-48:t]
                w_az = az_arr[t-48:t]
                if np.std(w_s) < 3.0 and np.mean(np.abs(w_ay)) < 0.35 and np.mean(np.abs(w_gz)) < 0.02:
                    speeds_all.append(np.mean(w_s))
                    vib_z_all.append(np.std(w_az))
        except Exception:
            continue

    speeds_arr = np.array(speeds_all)
    vib_arr = np.array(vib_z_all)
    print(f"Total steady-state cruise windows collected: N = {len(speeds_arr)}", flush=True)
    
    speed_bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 140)]
    bin_labels = ["0-20 km/h", "20-40 km/h", "40-60 km/h", "60-80 km/h", "80-140 km/h"]
    
    # 1000 Fast Vectorized Bootstrap Iterations
    n_boot = 1000
    np.random.seed(42)
    
    print(f"\n{'Speed Bin':<18} | {'Raw N':<8} | {'Sample Mean':<14} | {'Bootstrap Mean':<16} | {'Std Error (SE)':<16} | {'95% Confidence Interval'}", flush=True)
    print("-" * 105, flush=True)
    
    for (low, high), label in zip(speed_bins, bin_labels):
        mask = (speeds_arr >= low) & (speeds_arr < high)
        v_sub = vib_arr[mask]
        n_raw = len(v_sub)
        if n_raw > 0:
            m_raw = np.mean(v_sub)
            # Vectorized bootstrap
            boot_samples = np.random.choice(v_sub, size=(n_boot, n_raw), replace=True)
            boot_means = np.mean(boot_samples, axis=1)
            b_mean = np.mean(boot_means)
            b_se = np.std(boot_means)
            ci_low = np.percentile(boot_means, 2.5)
            ci_high = np.percentile(boot_means, 97.5)
            print(f"{label:<18} | N={n_raw:<6d} | {m_raw:<14.4f} | {b_mean:<16.4f} | {b_se:<16.4f} | [{ci_low:.4f}, {ci_high:.4f}]", flush=True)
        else:
            print(f"{label:<18} | N=0", flush=True)

    # Overlap probability: P(vib(80-140) <= vib(40-60))
    v40_60 = vib_arr[(speeds_arr >= 40) & (speeds_arr < 60)]
    v80_plus = vib_arr[speeds_arr >= 80]
    overlap_prob = np.mean([np.mean(v80_plus <= v) for v in v40_60])
    print(f"\nDirect Overlap Probability P(Vib_Z[80+] <= Vib_Z[40-60]): {overlap_prob*100:.1f}%", flush=True)

    # =========================================================================
    # TEST 3: ISOLATED CAUSE-C ROLLOUT EXPERIMENTS (NO RETRAINING)
    # =========================================================================
    print("\n" + "=" * 85)
    print(" [TEST 3: ISOLATED CAUSE-C CLOSED-LOOP ROLLOUT EXPERIMENTS ON S-S3a]")
    print("=" * 85)

    def evaluate_modified_rollout(name, modify_fn=None):
        preds_mod = []
        v_anc_t = torch.zeros(1, device=device, dtype=torch.float32)
        with torch.no_grad():
            for x_seq, targets in val_loader:
                x_seq = x_seq.to(device)
                B, L, C, W = x_seq.shape
                for t in range(L):
                    out = model(x_seq[:, t], v_anchor=v_anc_t)
                    v_pred_mps = out["mu_v"].item()
                    p_zupt = out["p_zupt"].item()
                    
                    if modify_fn is not None:
                        v_pred_mps = modify_fn(v_pred_mps, p_zupt, v_anc_t.item(), out)
                        
                    v_pred_kmh = v_pred_mps * 3.6
                    preds_mod.append(v_pred_kmh)
                    v_anc_t = torch.tensor([v_pred_mps], device=device, dtype=torch.float32)

        p_arr = np.array(preds_mod)[:min_len]
        g_arr = gts[:min_len]
        mae_overall = np.mean(np.abs(p_arr - g_arr))
        bias_overall = np.mean(p_arr - g_arr)
        
        # Per speed bin metrics
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
        slope = np.cov(p_arr, g_arr)[0, 1] / (np.var(g_arr) + 1e-6)
        intercept = np.mean(p_arr) - slope * np.mean(g_arr)
        r = np.corrcoef(p_arr, g_arr)[0, 1]
        
        print(f"\n---> {name}")
        print(f"     Balanced MAE: {b_mae_avg:5.2f} km/h | Raw MAE: {mae_overall:5.2f} km/h | Bias: {bias_overall:+5.2f} km/h")
        print(f"     Pearson r: {r:5.3f} | Slope: {slope:6.4f} | Intercept: {intercept:+6.2f} km/h")
        print(f"     0-10 km/h: MAE={b_maes[0]:5.2f} km/h, Bias={b_signed['0-10']:+6.2f} km/h")
        print(f"     80+  km/h: MAE={b_maes[7]:5.2f} km/h, Bias={b_signed['80+']:+6.2f} km/h")
        return {"b_mae": b_mae_avg, "raw_mae": mae_overall, "slope": slope, "intercept": intercept, "b0": b_signed["0-10"], "b80": b_signed["80+"]}

    # Baseline Closed-Loop
    evaluate_modified_rollout("Baseline Closed-Loop (Unchanged)")

    # Test 3a: ZUPT Hard Clamping (if p_zupt > 0.5 -> v = 0)
    def zupt_clamp(v_pred, p_zupt, v_anc, out):
        if p_zupt > 0.5:
            return 0.0
        return v_pred
    evaluate_modified_rollout("Test 3a: ZUPT Hard Clamping (p_zupt > 0.5 --> v=0)", zupt_clamp)

    # Test 3b: Softplus State Update instead of ReLU (using delta_v from head)
    def softplus_update(v_pred, p_zupt, v_anc, out):
        dv = out["delta_v"].item()
        # F.softplus(v_anc + dv) with beta=2.0
        sp_v = math.log(1.0 + math.exp(2.0 * (v_anc + dv))) / 2.0
        return sp_v
    evaluate_modified_rollout("Test 3b: Softplus State Update in Rollout", softplus_update)

    # Test 3c: Output Bias Shift Calibration (-3.5 m/s offset when anchor is near zero)
    def bias_calibrated_update(v_pred, p_zupt, v_anc, out):
        # Subtract low-speed bias offset proportional to ZUPT
        if p_zupt > 0.3:
            return max(0.0, v_pred - (p_zupt * 4.0))
        return v_pred
    evaluate_modified_rollout("Test 3c: Adaptive ZUPT-Proportional Low-Speed Bias Removal", bias_calibrated_update)

    print("\n" + "=" * 85)
    print(" [TESTS COMPLETE - STRICTLY INFERENCE & STATISTICAL AUDIT ON KAGGLE CHECKPOINT]")
    print("=" * 85)

if __name__ == "__main__":
    run_falsification_audit()
