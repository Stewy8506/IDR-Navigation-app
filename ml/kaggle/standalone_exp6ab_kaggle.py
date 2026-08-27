"""
standalone_exp6ab_kaggle.py - Self-Contained Single-File Kaggle GPU Training Runner for Experiment 6A-B.
Speed-Balanced Weighted Sampling Controlled Experiment.

Tests the hypothesis:
"Exp6A's large high-speed underprediction is primarily caused by insufficient high-speed training representation (3.03%), causing regression-to-the-mean / prediction compression."

Controlled Variable:
- Principled WeightedRandomSampler on TRAIN set (w_b = 1 / N_b) equalizing effective exposure to all 8 speed bins (12.5% each).
- Training budget strictly preserved: 28,748 samples / 449 batches per epoch.
- All architecture, features, losses, optimizer, hyperparams, and validation logic strictly preserved.
"""

import math
import os
import random
import sys
import time
import json
import glob
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

GRAVITY = 9.80665

# ---------------------------------------------------------------------------
# 1. Physics Features & Alignment
# ---------------------------------------------------------------------------

def estimate_phone_to_vehicle_matrix(acc_static: np.ndarray) -> np.ndarray:
    a_mean = np.mean(acc_static, axis=1) if acc_static.ndim > 1 else acc_static
    norm_a = np.linalg.norm(a_mean)
    if norm_a < 1e-4:
        return np.eye(3, dtype=np.float32)
    z_phone = a_mean / norm_a
    z_vehicle = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    v = np.cross(z_phone, z_vehicle)
    s = np.linalg.norm(v)
    c = np.dot(z_phone, z_vehicle)
    if s < 1e-6:
        return np.eye(3, dtype=np.float32) if c > 0 else np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=np.float32)
    R = np.eye(3, dtype=np.float32) + vx + (vx @ vx) * ((1.0 - c) / (s ** 2))
    return R.astype(np.float32)


def align_imu_to_vehicle_frame(imu_6xN: np.ndarray) -> np.ndarray:
    R = estimate_phone_to_vehicle_matrix(imu_6xN[:3])
    acc_body = R @ imu_6xN[:3]
    gyro_spatial = np.stack([imu_6xN[5], imu_6xN[4], imu_6xN[3]], axis=0)
    gyro_body = R @ gyro_spatial
    return np.vstack([acc_body[0], acc_body[1], acc_body[2], gyro_body[2], gyro_body[1], gyro_body[0]]).astype(np.float32)


def compute_physical_pitch_series(ax: np.ndarray, ay: np.ndarray, az: np.ndarray, wy: np.ndarray, wx: np.ndarray = None, wz: np.ndarray = None, dt: float = 0.1) -> np.ndarray:
    N = len(ay)
    pitch_series = np.zeros(N, dtype=np.float32)
    if N == 0:
        return pitch_series
    sin_th0 = np.clip(ay[0] / GRAVITY, -0.99, 0.99)
    pitch_series[0] = np.arcsin(sin_th0)
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    bg = np.zeros(3, dtype=np.float64)
    Kp, Ki = 0.20, 0.001

    for t in range(N):
        a = np.array([ax[t], ay[t], az[t]], dtype=np.float64)
        norm_a = np.linalg.norm(a)
        is_dynamic = abs(norm_a - GRAVITY) > 1.2
        if wz is not None and abs(wz[t]) > 0.08:
            is_dynamic = True

        if norm_a > 1e-3 and not is_dynamic:
            a_norm = a / norm_a
            v_b = np.array([2.0 * (q[1] * q[3] - q[0] * q[2]), 2.0 * (q[0] * q[1] + q[2] * q[3]), q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2], dtype=np.float64)
            e = np.cross(a_norm, v_b)
            bg += Ki * e * dt
            w_corr = np.array([wx[t] if wx is not None else 0.0, wy[t], wz[t] if wz is not None else 0.0], dtype=np.float64) + Kp * e - bg
        else:
            w_corr = np.array([wx[t] if wx is not None else 0.0, wy[t], wz[t] if wz is not None else 0.0], dtype=np.float64) - bg

        dq = 0.5 * np.array([-q[1] * w_corr[0] - q[2] * w_corr[1] - q[3] * w_corr[2], q[0] * w_corr[0] + q[2] * w_corr[2] - q[3] * w_corr[1], q[0] * w_corr[1] - q[1] * w_corr[2] + q[3] * w_corr[0], q[0] * w_corr[2] + q[1] * w_corr[1] - q[2] * w_corr[0]], dtype=np.float64)
        q += dq * dt
        norm_q = np.linalg.norm(q)
        if norm_q > 1e-6:
            q /= norm_q

        sin_pitch = np.clip(2.0 * (q[0] * q[2] - q[3] * q[1]), -1.0, 1.0)
        pitch_series[t] = float(np.arcsin(sin_pitch))
    return pitch_series


def compute_18ch_features(w_6ch: np.ndarray, theta_phys_window: np.ndarray, fs: float = 10.0) -> np.ndarray:
    W = w_6ch.shape[1]
    ax, ay, az = w_6ch[0], w_6ch[1], w_6ch[2]
    wz_yaw, wy_pitch, wx_roll = w_6ch[3], w_6ch[4], w_6ch[5]
    dt = 1.0 / fs

    a_norm = np.sqrt(ax**2 + ay**2 + az**2) - GRAVITY
    w_norm = np.sqrt(wz_yaw**2 + wy_pitch**2 + wx_roll**2)

    vel_int = np.zeros(W, dtype=np.float32)
    acc = 0.0
    for i in range(W):
        acc = acc * 0.95 + ay[i] * dt
        vel_int[i] = acc

    az_series = pd.Series(az)
    az_var = az_series.rolling(window=5, min_periods=1).var().fillna(0.0).values.astype(np.float32)

    freqs = np.fft.rfftfreq(W, d=dt)
    az_centered = az - np.mean(az)
    ay_centered = ay - np.mean(ay)
    az_fft = np.abs(np.fft.rfft(az_centered)) ** 2 / W
    ay_fft = np.abs(np.fft.rfft(ay_centered)) ** 2 / W

    low_mask = (freqs >= 0.3) & (freqs < 1.25)
    mid_mask = (freqs >= 1.25) & (freqs < 2.5)
    high_mask = (freqs >= 2.5) & (freqs <= 5.0)

    e_low = float(np.sum(az_fft[low_mask])) if np.sum(low_mask) > 0 else 0.0
    e_mid = float(np.sum(az_fft[mid_mask])) if np.sum(mid_mask) > 0 else 0.0
    e_high = float(np.sum(az_fft[high_mask])) if np.sum(high_mask) > 0 else 0.0

    total_power = float(np.sum(az_fft)) + 1e-6
    p_ay_total = float(np.sum(ay_fft)) + 1e-6

    r_low = e_low / total_power
    r_mid = e_mid / total_power
    r_high = e_high / total_power
    spec_centroid = float(np.sum(freqs * az_fft) / total_power)

    peak_mask = (freqs >= 1.0) & (freqs <= 25.0)
    f_peak = float(freqs[peak_mask][np.argmax(az_fft[peak_mask])]) if np.sum(peak_mask) > 0 else 2.0

    wz_mag = np.abs(wz_yaw)
    turn_feat = np.where(wz_mag >= 0.035, np.clip(np.abs(ax) / wz_mag, 0.0, 40.0), 0.0).astype(np.float32)
    vib_ratio = float(p_ay_total / total_power)
    ay_grav_comp = (ay - GRAVITY * np.sin(theta_phys_window)).astype(np.float32)

    return np.stack([
        ax.astype(np.float32), ay.astype(np.float32), az.astype(np.float32),
        wz_yaw.astype(np.float32), wy_pitch.astype(np.float32), wx_roll.astype(np.float32),
        a_norm.astype(np.float32), w_norm.astype(np.float32), vel_int.astype(np.float32),
        az_var.astype(np.float32),
        np.full(W, r_low, dtype=np.float32), np.full(W, r_mid, dtype=np.float32), np.full(W, r_high, dtype=np.float32),
        np.full(W, spec_centroid, dtype=np.float32), np.full(W, f_peak, dtype=np.float32),
        turn_feat, np.full(W, vib_ratio, dtype=np.float32), ay_grav_comp,
    ], axis=0).astype(np.float32)

# ---------------------------------------------------------------------------
# 2. Sequence Dataset Loader
# ---------------------------------------------------------------------------

class SequencePhysicsDataset(Dataset):
    def __init__(self, data_dir: str, window_size: int = 48, seq_len: int = 32, seq_stride: int = 16, split: str = "train"):
        self.data_dir = data_dir
        self.window_size = window_size
        self.seq_len = seq_len
        self.seq_stride = seq_stride
        self.split = split
        self.sequences_x, self.sequences_v, self.sequences_dv = [], [], []
        self.sequences_zupt, self.sequences_pitch, self.sequences_regime = [], [], []
        self._load_dataset()

    def _classify_regime(self, v_kmh: float, accel_fwd: float, yaw_rate: float) -> int:
        if v_kmh < 1.0: return 0
        if abs(yaw_rate) > 0.15 and v_kmh > 5.0: return 6
        if accel_fwd > 0.8: return 4
        if accel_fwd < -0.8: return 5
        if v_kmh < 20.0: return 1
        if v_kmh < 60.0: return 2
        return 3

    def _load_dataset(self):
        s_pattern = os.path.join(self.data_dir, "**", "S-*.csv")
        all_s_files = sorted(glob.glob(s_pattern, recursive=True))
        selected_files = []
        for sf in all_s_files:
            is_driver_e = ("Driver E" in sf) or ("Vw" in sf) or ("Vta" in sf) or ("Vtb" in sf) or ("Vf" in sf)
            is_val_s3a = "S3a" in sf
            if self.split == "train" and not is_driver_e and not is_val_s3a: selected_files.append(sf)
            elif self.split == "val" and is_val_s3a: selected_files.append(sf)
            elif self.split == "test" and ("Vw11" in sf or "Vw12" in sf): selected_files.append(sf)

        t_start_split = time.time()
        print(f"Loading SequencePhysicsDataset [{self.split.upper()}] from {len(selected_files)} drive files (L={self.seq_len}, stride={self.seq_stride})...", flush=True)
        for idx, s_file in enumerate(selected_files):
            v_file = os.path.join(os.path.dirname(s_file), os.path.basename(s_file).replace("S-", "V-"))
            if not os.path.exists(v_file):
                print(f"  [{idx+1}/{len(selected_files)}] Skipping missing paired velocity file: {v_file}", flush=True)
                continue
            t_file = time.time()
            try:
                df_s = pd.read_csv(s_file, encoding="latin1")
                df_v = pd.read_csv(v_file, encoding="latin1")
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
                if speed_col not in df_v.columns: continue
                speed_kmh = df_v[speed_col].values.astype(np.float32)
                speed_mps = speed_kmh / 3.6

                min_len = min(raw_imu.shape[1], len(speed_mps))
                if min_len < self.window_size + self.seq_len + 5: continue
                raw_imu = raw_imu[:, :min_len]
                speed_mps, speed_kmh = speed_mps[:min_len], speed_kmh[:min_len]

                aligned_imu = align_imu_to_vehicle_frame(raw_imu)
                pitch_phys = compute_physical_pitch_series(aligned_imu[0], aligned_imu[1], aligned_imu[2], wy=aligned_imu[4], wx=aligned_imu[5], wz=aligned_imu[3], dt=0.1)

                drive_windows, drive_v, drive_dv, drive_zupt, drive_pitch, drive_regime = [], [], [], [], [], []
                for end_idx in range(self.window_size, min_len):
                    start_idx = end_idx - self.window_size
                    w_imu = aligned_imu[:, start_idx:end_idx]
                    w_pitch = pitch_phys[start_idx:end_idx]
                    v_curr = float(speed_mps[end_idx - 1])
                    v_prev = float(speed_mps[end_idx - 2])
                    is_zupt = 1.0 if (v_curr < 0.25 and np.abs(aligned_imu[1, end_idx - 1]) < 0.20 and np.abs(aligned_imu[3, end_idx - 1]) < 0.05) else 0.0
                    target_pitch = float(pitch_phys[end_idx - 1])
                    regime = self._classify_regime(float(speed_kmh[end_idx - 1]), float(aligned_imu[1, end_idx - 1]), float(aligned_imu[3, end_idx - 1]))

                    feat18 = compute_18ch_features(w_imu, w_pitch)
                    drive_windows.append(feat18)
                    drive_v.append(v_curr)
                    drive_dv.append(float(v_curr - v_prev))
                    drive_zupt.append(is_zupt)
                    drive_pitch.append(target_pitch)
                    drive_regime.append(regime)

                drive_windows = np.stack(drive_windows, axis=0)
                drive_v = np.array(drive_v, dtype=np.float32)
                drive_dv = np.array(drive_dv, dtype=np.float32)
                drive_zupt = np.array(drive_zupt, dtype=np.float32)
                drive_pitch = np.array(drive_pitch, dtype=np.float32)
                drive_regime = np.array(drive_regime, dtype=np.int64)

                N_steps = len(drive_v)
                prev_seq_count = len(self.sequences_x)
                for s_start in range(0, N_steps - self.seq_len + 1, self.seq_stride):
                    s_end = s_start + self.seq_len
                    self.sequences_x.append(drive_windows[s_start:s_end])
                    self.sequences_v.append(drive_v[s_start:s_end])
                    self.sequences_dv.append(drive_dv[s_start:s_end])
                    self.sequences_zupt.append(drive_zupt[s_start:s_end])
                    self.sequences_pitch.append(drive_pitch[s_start:s_end])
                    self.sequences_regime.append(drive_regime[s_start:s_end])
                new_seq_count = len(self.sequences_x) - prev_seq_count
                print(f"  [{idx+1}/{len(selected_files)}] Processed {os.path.basename(s_file)} ({min_len} rows -> +{new_seq_count} sequences) in {time.time()-t_file:.2f}s", flush=True)
            except Exception as e:
                print(f"Error reading {s_file}: {e}", flush=True)

        print(f"Loaded {len(self.sequences_x)} total sequences ({self.split.upper()}) in {time.time()-t_start_split:.2f}s.\n", flush=True)

    def __len__(self): return len(self.sequences_x)
    def __getitem__(self, idx: int):
        return torch.from_numpy(self.sequences_x[idx]), {
            "v": torch.from_numpy(self.sequences_v[idx]),
            "delta_v": torch.from_numpy(self.sequences_dv[idx]),
            "zupt": torch.from_numpy(self.sequences_zupt[idx]),
            "pitch": torch.from_numpy(self.sequences_pitch[idx]),
            "regime": torch.from_numpy(self.sequences_regime[idx]),
        }

# ---------------------------------------------------------------------------
# 3. Model Architecture (IDENTICAL TO EXP6A)
# ---------------------------------------------------------------------------

class ConvNeXtBlock1D(nn.Module):
    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_tensor = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None: x = self.gamma * x
        x = x.permute(0, 2, 1)
        return input_tensor + x


class DeepSpeedKinematicsNet(nn.Module):
    def __init__(self, in_channels: int = 18, window_size: int = 48, embed_dims: list = [48, 64, 96, 128], depths: list = [2, 2, 4, 2]):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(in_channels)
        self.stem = nn.Sequential(nn.Conv1d(in_channels, embed_dims[0], kernel_size=3, padding=1), nn.LayerNorm([embed_dims[0], window_size]))
        self.stage1 = nn.Sequential(*[ConvNeXtBlock1D(embed_dims[0]) for _ in range(depths[0])])
        self.trans1 = nn.Sequential(nn.Conv1d(embed_dims[0], embed_dims[1], kernel_size=1), nn.LayerNorm([embed_dims[1], window_size]))
        self.stage2 = nn.Sequential(*[ConvNeXtBlock1D(embed_dims[1]) for _ in range(depths[1])])
        self.trans2 = nn.Sequential(nn.Conv1d(embed_dims[1], embed_dims[2], kernel_size=1), nn.LayerNorm([embed_dims[2], window_size]))
        self.stage3 = nn.Sequential(*[ConvNeXtBlock1D(embed_dims[2]) for _ in range(depths[2])])
        self.trans3 = nn.Sequential(nn.Conv1d(embed_dims[2], embed_dims[3], kernel_size=1), nn.LayerNorm([embed_dims[3], window_size]))
        self.stage4 = nn.Sequential(*[ConvNeXtBlock1D(embed_dims[3]) for _ in range(depths[3])])

        self.mha_norm = nn.LayerNorm(embed_dims[3])
        self.mha = nn.MultiheadAttention(embed_dim=embed_dims[3], num_heads=4, batch_first=True)
        self.pool_norm = nn.LayerNorm(embed_dims[3])

        self.state_proj = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, 32))
        self.head_velocity = nn.Sequential(nn.Linear(embed_dims[3] + 32, 64), nn.GELU(), nn.Linear(64, 2))
        self.head_delta_v = nn.Sequential(nn.Linear(embed_dims[3], 32), nn.GELU(), nn.Linear(32, 1))
        self.head_zupt = nn.Sequential(nn.Linear(embed_dims[3], 32), nn.GELU(), nn.Linear(32, 1))
        self.head_pitch = nn.Sequential(nn.Linear(embed_dims[3], 32), nn.GELU(), nn.Linear(32, 1))
        self.head_regime = nn.Sequential(nn.Linear(embed_dims[3], 32), nn.GELU(), nn.Linear(32, 7))

    def forward(self, x: torch.Tensor, v_anchor: torch.Tensor = None) -> dict:
        x = self.input_norm(x)
        feat = self.stem(x)
        feat = self.trans1(self.stage1(feat))
        feat = self.trans2(self.stage2(feat))
        feat = self.trans3(self.stage3(feat))
        feat = self.stage4(feat)

        tokens = feat.permute(0, 2, 1)
        norm_tokens = self.mha_norm(tokens)
        attn_out, _ = self.mha(norm_tokens, norm_tokens, norm_tokens)
        tokens = tokens + attn_out
        pooled = self.pool_norm(tokens.mean(dim=1) + tokens[:, -1, :])

        if v_anchor is None: v_anchor = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        elif isinstance(v_anchor, (int, float)): v_anchor = torch.full((x.shape[0],), float(v_anchor), device=x.device, dtype=x.dtype)
        elif v_anchor.dim() == 0: v_anchor = v_anchor.expand(x.shape[0])
        elif v_anchor.dim() == 2 and v_anchor.shape[1] == 1: v_anchor = v_anchor.squeeze(-1)
        elif v_anchor.shape[0] == 1 and x.shape[0] > 1: v_anchor = v_anchor.expand(x.shape[0])

        v_anchor_norm = (v_anchor / 30.0).unsqueeze(-1)
        state_embed = self.state_proj(v_anchor_norm)
        fused_feat = torch.cat([pooled, state_embed], dim=-1)

        v_out = self.head_velocity(fused_feat)
        delta_v_pred = v_out[:, 0]
        sigma_v = 0.5 + 5.5 * torch.sigmoid(v_out[:, 1])
        var_v = sigma_v ** 2
        log_sigma2 = torch.log(var_v)
        mu_v = F.relu(v_anchor + delta_v_pred)

        return {
            "mu_v": mu_v, "sigma_v": sigma_v, "log_sigma2": log_sigma2, "var_v": var_v,
            "delta_v": delta_v_pred,
            "p_zupt": torch.sigmoid(self.head_zupt(pooled).squeeze(-1)),
            "pitch": self.head_pitch(pooled).squeeze(-1),
            "regime_logits": self.head_regime(pooled),
        }

# ---------------------------------------------------------------------------
# 4. Training Engine with Speed-Balanced Weighted Sampler
# ---------------------------------------------------------------------------

def auto_resolve_dataset_dir() -> str:
    print("\n--- Scanning for Dataset Files ---", flush=True)
    if os.path.exists("/kaggle/input"):
        print(f"Contents of /kaggle/input:", flush=True)
        for root, dirs, files in os.walk("/kaggle/input"):
            print(f"  {root} -> dirs: {dirs[:5]}, files: {files[:5]} (total {len(files)})", flush=True)
    
    # 1. Check if there are any .zip archives in /kaggle/input that need extracting
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            for f in files:
                if f.endswith(".zip") and ("iovnb" in f.lower() or "categorised" in f.lower()):
                    zip_path = os.path.join(root, f)
                    extract_dest = "/kaggle/working/extracted_dataset"
                    print(f"Found dataset archive: {zip_path}. Extracting to {extract_dest}...", flush=True)
                    import zipfile
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dest)
                    print(f"Extraction complete!", flush=True)

    # 2. Check for S-*.csv files across all candidate locations
    search_roots = ["/kaggle/working", "/kaggle/input", "ml/external", "."]
    for s_root in search_roots:
        if not os.path.exists(s_root):
            continue
        found = glob.glob(os.path.join(s_root, "**", "S-S1.csv"), recursive=True)
        if found:
            candidate_dir = os.path.dirname(os.path.dirname(os.path.dirname(found[0])))
            print(f"Discovered valid dataset root: {candidate_dir} (sample file: {found[0]})", flush=True)
            return candidate_dir

    # 3. Fallback check for common paths
    for cand in [
        "/kaggle/input/iovnb-dataset/Categorised IOVNB Dataset",
        "/kaggle/input/iovnb-dataset",
        "/kaggle/input/io-vnbd-dataset/Categorised IOVNB Dataset",
        "/kaggle/input/io-vnbd-dataset",
        "ml/external/IO-VNBD_repo/Synchronised V abd S datasets/Categorised IOVNB Dataset",
    ]:
        if os.path.exists(cand):
            print(f"Found candidate directory: {cand}", flush=True)
            return cand

    raise FileNotFoundError("Could not find dataset path with S-*.csv files anywhere under /kaggle/input or local paths!")


def run_training():
    print("\n=======================================================", flush=True)
    print("      HARDWARE & CUDA ENVIRONMENT DIAGNOSTICS          ", flush=True)
    print("=======================================================", flush=True)
    print(f"Python Version      : {sys.version.split()[0]}", flush=True)
    print(f"PyTorch Version     : {torch.__version__}", flush=True)
    print(f"CUDA Available      : {torch.cuda.is_available()}", flush=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"CUDA Runtime Version: {torch.version.cuda}", flush=True)
        print(f"Device Name         : {torch.cuda.get_device_name(0)}", flush=True)
        cap = torch.cuda.get_device_capability(0)
        print(f"Compute Capability  : sm_{cap[0]}{cap[1]}", flush=True)
        try:
            test_x = torch.randn(64, 64, device=device)
            test_y = (test_x @ test_x).sum().item()
            print(f"CUDA Smoke Test     : SUCCESS (test tensor matmul sum={test_y:.4f})", flush=True)
        except Exception as e:
            print(f"CUDA Smoke Test     : FAILED with error: {e}", flush=True)
            raise e
    else:
        print("CUDA Smoke Test     : SKIPPED (Running on CPU)", flush=True)
    print("=======================================================\n", flush=True)

    # Locate dataset
    data_dir = auto_resolve_dataset_dir()
    print(f"Using dataset directory: {data_dir}\n", flush=True)

    out_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "ml/weights"
    os.makedirs(out_dir, exist_ok=True)

    train_ds = SequencePhysicsDataset(data_dir=data_dir, split="train", seq_len=32, seq_stride=16)
    val_ds = SequencePhysicsDataset(data_dir=data_dir, split="val", seq_len=32, seq_stride=32)

    # -----------------------------------------------------------------------
    # SPEED-BALANCED WEIGHTED RANDOM SAMPLER (THE CONTROLLED INTERVENTION)
    # -----------------------------------------------------------------------
    bins = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 80), (80, 300)]
    bin_names = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-80", "80+"]
    num_bins = len(bins)

    # Calculate sequence average speed for each training window
    train_seq_speeds = np.array([np.mean(seq_v) * 3.6 for seq_v in train_ds.sequences_v])
    total_train_seqs = len(train_seq_speeds)

    seq_bin_indices = np.zeros(total_train_seqs, dtype=np.int64)
    bin_counts = np.zeros(num_bins, dtype=np.int64)

    for b_idx, (b_low, b_high) in enumerate(bins):
        mask = (train_seq_speeds >= b_low) & (train_seq_speeds < b_high)
        seq_bin_indices[mask] = b_idx
        bin_counts[b_idx] = int(mask.sum())

    # Principled sampling weights: w_b = 1 / N_b
    bin_weights = np.zeros(num_bins, dtype=np.float64)
    for b_idx in range(num_bins):
        bin_weights[b_idx] = (1.0 / bin_counts[b_idx]) if bin_counts[b_idx] > 0 else 0.0

    sample_weights = np.array([bin_weights[b_idx] for b_idx in seq_bin_indices], dtype=np.float64)
    sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.double)

    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=total_train_seqs,
        replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=64,
        sampler=sampler,
        drop_last=True,
        num_workers=2 if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # Startup Diagnostic Report
    val_seq_speeds = np.array([np.mean(seq_v) * 3.6 for seq_v in val_ds.sequences_v])
    total_val_seqs = len(val_seq_speeds)

    print("=" * 80, flush=True)
    print("                 TRAINING EXPOSURE REBALANCING AUDIT", flush=True)
    print("=" * 80, flush=True)
    print(f"{'Speed Bin (km/h)':<18} | {'TRAIN Seqs':<10} | {'Orig TRAIN %':<12} | {'Eff Sampled %':<13} | {'VAL Seqs':<9} | {'VAL %'}", flush=True)
    print("-" * 80, flush=True)

    for b_idx, ((b_low, b_high), b_name) in enumerate(zip(bins, bin_names)):
        trn_c = bin_counts[b_idx]
        orig_pct = (trn_c / total_train_seqs) * 100.0
        eff_pct = 100.0 / num_bins # 12.5% target
        val_c = int(np.sum((val_seq_speeds >= b_low) & (val_seq_speeds < b_high)))
        val_pct = (val_c / total_val_seqs) * 100.0
        w_val = bin_weights[b_idx]
        print(f"{b_name:<18} | {trn_c:<10d} | {orig_pct:<11.2f}% | {eff_pct:<12.2f}% | {val_c:<9d} | {val_pct:<.2f}% (weight={w_val:.2e})", flush=True)
    print("=" * 80, flush=True)
    print(f"Total Batches per Epoch: {len(train_loader)} (Batch Size: 64, Total Steps: {len(train_loader)*15})\n", flush=True)

    # -----------------------------------------------------------------------
    # MODEL, LOSSES & OPTIMIZER (IDENTICAL TO EXP6A)
    # -----------------------------------------------------------------------
    model = DeepSpeedKinematicsNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)

    loss_huber_v = nn.SmoothL1Loss(beta=1.0)
    loss_l1_dv = nn.L1Loss()
    loss_bce_zupt = nn.BCELoss()
    loss_huber_pitch = nn.SmoothL1Loss(beta=0.1)
    loss_ce_regime = nn.CrossEntropyLoss()

    # Pre-Flight Diagnostic Check
    print("\n--- Running Pre-Flight Diagnostic Smoke Test ---", flush=True)
    sample_x, sample_targets = next(iter(train_loader))
    sample_x = sample_x.to(device)
    B_s, L_s, C_s, W_s = sample_x.shape
    v_state_s = torch.zeros(B_s, device=device, dtype=torch.float32)
    sample_out = model(sample_x[:, 0], v_anchor=v_state_s)
    sample_loss = loss_huber_v(sample_out["mu_v"], sample_targets["v"][:, 0].to(device))
    sample_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Sample Batch Shape: x={sample_x.shape}, v={sample_targets['v'].shape}", flush=True)
    print(f"Sample Forward Pass Output: mu_v={sample_out['mu_v'].shape}, Sample Loss={sample_loss.item():.4f}", flush=True)
    print(">>> [DIAGNOSTIC PASSED] Batch loading, CUDA forward pass, loss calculation, and backward step verified! <<<\n", flush=True)

    best_balanced_val_mae = float("inf")
    best_raw_val_mae = float("inf")
    best_balanced_epoch = 0
    best_raw_epoch = 0

    save_path_best_balanced = os.path.join(out_dir, "exp6ab_best_spectral_speed_filter.pt")
    save_path_best_raw = os.path.join(out_dir, "exp6ab_best_raw_spectral_speed_filter.pt")
    save_path_final = os.path.join(out_dir, "exp6ab_final_spectral_speed_filter.pt")
    history = []

    print("Starting 15-Epoch Speed-Balanced Training on GPU...\n", flush=True)

    for epoch in range(1, 16):
        t0 = time.time()
        model.train()
        total_loss, total_mae, batches = 0.0, 0.0, 0

        for x_seq, targets in train_loader:
            x_seq = x_seq.to(device, non_blocking=True)
            B, L, C, W = x_seq.shape
            v_gt = targets["v"].to(device, non_blocking=True)
            dv_gt = targets["delta_v"].to(device, non_blocking=True)
            zupt_gt = targets["zupt"].to(device, non_blocking=True)
            pitch_gt = targets["pitch"].to(device, non_blocking=True)
            regime_gt = targets["regime"].to(device, non_blocking=True)

            optimizer.zero_grad()
            v_state = torch.zeros(B, device=device, dtype=torch.float32)
            pred_mu, pred_sigma, pred_logvar, pred_dv, pred_zupt, pred_pitch, pred_regime = [], [], [], [], [], [], []

            for t in range(L):
                out = model(x_seq[:, t], v_anchor=v_state)
                mu_t = out["mu_v"]
                pred_mu.append(mu_t)
                pred_sigma.append(out["sigma_v"])
                pred_logvar.append(out["log_sigma2"])
                pred_dv.append(out["delta_v"])
                pred_zupt.append(out["p_zupt"])
                pred_pitch.append(out["pitch"])
                pred_regime.append(out["regime_logits"])
                v_state = mu_t.detach()

            mu_all = torch.stack(pred_mu, dim=1).view(-1)
            sigma_all = torch.stack(pred_sigma, dim=1).view(-1)
            logvar_all = torch.stack(pred_logvar, dim=1).view(-1)
            dv_all = torch.stack(pred_dv, dim=1).view(-1)
            zupt_all = torch.stack(pred_zupt, dim=1).view(-1)
            pitch_all = torch.stack(pred_pitch, dim=1).view(-1)
            regime_all = torch.stack(pred_regime, dim=1).view(-1, 7)

            v_gt_all = v_gt.view(-1)
            loss = (
                loss_huber_v(mu_all, v_gt_all)
                + 0.15 * (0.5 * torch.exp(-logvar_all) * ((mu_all - v_gt_all) ** 2) + 0.5 * logvar_all).mean()
                + 0.10 * (sigma_all - torch.abs(mu_all - v_gt_all)).abs().mean()
                + 0.50 * loss_l1_dv(dv_all, dv_gt.view(-1))
                + 0.20 * loss_bce_zupt(zupt_all, zupt_gt.view(-1))
                + 0.10 * loss_huber_pitch(pitch_all, pitch_gt.view(-1))
                + 0.05 * loss_ce_regime(regime_all, regime_gt.view(-1))
            )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_mae += torch.abs(mu_all - v_gt_all).mean().item() * 3.6
            batches += 1

        scheduler.step()
        ep_time = time.time() - t0
        train_loss = total_loss / max(1, batches)
        train_mae = total_mae / max(1, batches)

        # -------------------------------------------------------------------
        # Validation Evaluation (Continuous Rollout on S-S3a)
        # -------------------------------------------------------------------
        model.eval()
        val_preds, val_gts = [], []
        with torch.no_grad():
            v_val_state = torch.zeros(1, device=device, dtype=torch.float32)
            for x_seq, targets in val_loader:
                x_seq = x_seq.to(device)
                v_gt_seq = targets["v"].to(device)
                B, L, C, W = x_seq.shape
                for t in range(L):
                    out = model(x_seq[:, t], v_anchor=v_val_state)
                    mu_t = out["mu_v"]
                    val_preds.append(mu_t.item() * 3.6)
                    val_gts.append(v_gt_seq[0, t].item() * 3.6)
                    v_val_state = mu_t.detach()

        val_preds, val_gts = np.array(val_preds), np.array(val_gts)
        val_errors = np.abs(val_preds - val_gts)
        val_signed_errors = val_preds - val_gts
        overall_val_mae = float(val_errors.mean())
        val_r = float(np.corrcoef(val_preds, val_gts)[0, 1]) if np.std(val_preds) > 0 and np.std(val_gts) > 0 else 0.0
        val_r2 = val_r ** 2

        if np.std(val_gts) > 0:
            reg_m, reg_c = np.polyfit(val_gts, val_preds, 1)
        else:
            reg_m, reg_c = 0.0, 0.0

        pred_std = float(np.std(val_preds))
        gt_std = float(np.std(val_gts))
        compression_ratio = pred_std / gt_std if gt_std > 0 else 0.0

        bin_maes = []
        bin_signed = []
        bin_mae_dict = {}
        bin_signed_dict = {}

        for b_low, b_high, bn in zip([b[0] for b in bins], [b[1] for b in bins], bin_names):
            mask = (val_gts >= b_low) & (val_gts < b_high)
            if mask.sum() > 0:
                b_mae = float(val_errors[mask].mean())
                b_signed = float(val_signed_errors[mask].mean())
            else:
                b_mae = 0.0
                b_signed = 0.0
            bin_maes.append(b_mae)
            bin_signed.append(b_signed)
            bin_mae_dict[bn] = b_mae
            bin_signed_dict[bn] = b_signed

        balanced_val_mae = float(np.mean(bin_maes))

        bin_mae_str = " | ".join([f"{bn}:{bm:4.1f}" for bn, bm in zip(bin_names, bin_maes)])
        bin_sgn_str = " | ".join([f"{bn}:{bs:+5.1f}" for bn, bs in zip(bin_names, bin_signed)])

        print(f"Epoch [{epoch:02d}/15] ({ep_time:.1f}s) | Train Loss: {train_loss:.4f} | Train MAE: {train_mae:5.2f} km/h | Val MAE: {overall_val_mae:5.2f} km/h | Bal MAE: {balanced_val_mae:5.2f} km/h | r: {val_r:.3f}", flush=True)
        print(f"   Fit: Pred = {reg_m:.4f}*GT + {reg_c:+5.2f} km/h (r²={val_r2:.3f}, std_ratio={compression_ratio:.3f})", flush=True)
        print(f"   Bin MAEs   : [ {bin_mae_str} ]", flush=True)
        print(f"   Bin Signed : [ {bin_sgn_str} ]", flush=True)

        ep_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mae": train_mae,
            "val_mae": overall_val_mae,
            "balanced_val_mae": balanced_val_mae,
            "val_r": val_r,
            "val_r2": val_r2,
            "regression_slope": float(reg_m),
            "regression_intercept": float(reg_c),
            "pred_std": pred_std,
            "gt_std": gt_std,
            "compression_ratio": compression_ratio,
            "bin_maes": bin_mae_dict,
            "bin_signed_errors": bin_signed_dict,
            "epoch_time_s": ep_time
        }
        history.append(ep_record)

        if balanced_val_mae < best_balanced_val_mae:
            best_balanced_val_mae = balanced_val_mae
            best_balanced_epoch = epoch
            torch.save(model.state_dict(), save_path_best_balanced)
            print(f"   >>> [SAVED BEST BALANCED CHECKPOINT] Epoch {epoch} (Balanced MAE: {balanced_val_mae:.2f} km/h) <<<", flush=True)

        if overall_val_mae < best_raw_val_mae:
            best_raw_val_mae = overall_val_mae
            best_raw_epoch = epoch
            torch.save(model.state_dict(), save_path_best_raw)
            print(f"   >>> [SAVED BEST RAW CHECKPOINT] Epoch {epoch} (Val MAE: {overall_val_mae:.2f} km/h) <<<", flush=True)
        print("", flush=True)

    torch.save(model.state_dict(), save_path_final)
    with open(os.path.join(out_dir, "exp6ab_history.json"), "w") as f:
        json.dump({
            "best_balanced_epoch": best_balanced_epoch,
            "best_balanced_val_mae": best_balanced_val_mae,
            "best_raw_epoch": best_raw_epoch,
            "best_raw_val_mae": best_raw_val_mae,
            "history": history
        }, f, indent=2)

    print(f"\nTraining Complete!")
    print(f"  Best Balanced MAE Checkpoint : {save_path_best_balanced} (Epoch {best_balanced_epoch})")
    print(f"  Best Raw MAE Checkpoint      : {save_path_best_raw} (Epoch {best_raw_epoch})")
    print(f"  Final Checkpoint             : {save_path_final}")

if __name__ == "__main__":
    run_training()
