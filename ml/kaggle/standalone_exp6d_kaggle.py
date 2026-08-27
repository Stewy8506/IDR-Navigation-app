"""
standalone_exp6d_kaggle.py — Experiment 6D: Dual-Stream Spectral Kinematics Net.

Self-contained Kaggle GPU training runner.

Architecture changes vs Exp 6C:
  1. Dual-stream fusion: v_t = α·v_direct + (1-α)·(v_prev + Δv)
     - Stream A: Spectral velocity (multi-scale wavelet → absolute speed)
     - Stream B: Kinematic Δv (ConvNeXt backbone, unconstrained Δv ∈ ℝ)
  2. Hysteresis ZUPT gate: hard threshold at p_ZUPT > 0.70 AND E_kinetic < 0.02
  3. No ReLU on v_prev + Δv (zero-floor via max(0, v_fused) after fusion)
  4. Variance-preserving loss: (σ(pred) - σ(gt))²
  5. Plain cosine decay LR with linear warmup (no warm restarts)
  6. Speed-balanced sampling (WeightedRandomSampler)
"""

import math
import os
import random
import sys
import time
import json
import glob
import hashlib
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
# Self-hash for provenance
# ---------------------------------------------------------------------------

def _get_script_hash() -> str:
    this_file = os.path.abspath(__file__)
    with open(this_file, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 1. Physics Features & Alignment (PRESERVED from Exp 6C)
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


def compute_physical_pitch_series(ax, ay, az, wy, wx=None, wz=None, dt=0.1):
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
            v_b = np.array([2.0*(q[1]*q[3]-q[0]*q[2]), 2.0*(q[0]*q[1]+q[2]*q[3]), q[0]**2-q[1]**2-q[2]**2+q[3]**2], dtype=np.float64)
            e = np.cross(a_norm, v_b)
            bg += Ki * e * dt
            w_corr = np.array([wx[t] if wx is not None else 0.0, wy[t], wz[t] if wz is not None else 0.0], dtype=np.float64) + Kp * e - bg
        else:
            w_corr = np.array([wx[t] if wx is not None else 0.0, wy[t], wz[t] if wz is not None else 0.0], dtype=np.float64) - bg
        dq = 0.5 * np.array([-q[1]*w_corr[0]-q[2]*w_corr[1]-q[3]*w_corr[2], q[0]*w_corr[0]+q[2]*w_corr[2]-q[3]*w_corr[1], q[0]*w_corr[1]-q[1]*w_corr[2]+q[3]*w_corr[0], q[0]*w_corr[2]+q[1]*w_corr[1]-q[2]*w_corr[0]], dtype=np.float64)
        q += dq * dt
        norm_q = np.linalg.norm(q)
        if norm_q > 1e-6:
            q /= norm_q
        sin_pitch = np.clip(2.0*(q[0]*q[2]-q[3]*q[1]), -1.0, 1.0)
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
# 2. Sequence Dataset Loader (PRESERVED, with speed-balanced sampler support)
# ---------------------------------------------------------------------------

class SequencePhysicsDataset(Dataset):
    def __init__(self, data_dir, window_size=48, seq_len=32, seq_stride=16, split="train"):
        self.data_dir = data_dir
        self.window_size = window_size
        self.seq_len = seq_len
        self.seq_stride = seq_stride
        self.split = split
        self.sequences_x, self.sequences_v, self.sequences_dv = [], [], []
        self.sequences_zupt, self.sequences_pitch, self.sequences_regime = [], [], []
        self.sequence_mean_speeds = []  # for speed-balanced sampling
        self._load_dataset()

    def _classify_regime(self, v_kmh, accel_fwd, yaw_rate):
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

        t_start = time.time()
        print(f"Loading SequencePhysicsDataset [{self.split.upper()}] from {len(selected_files)} drive files (L={self.seq_len}, stride={self.seq_stride})...", flush=True)
        for idx, s_file in enumerate(selected_files):
            v_file = os.path.join(os.path.dirname(s_file), os.path.basename(s_file).replace("S-", "V-"))
            if not os.path.exists(v_file):
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
                    # Mean speed of the sequence for balanced sampling
                    self.sequence_mean_speeds.append(float(np.mean(drive_v[s_start:s_end]) * 3.6))

                new_seq_count = len(self.sequences_x) - prev_seq_count
                print(f"  [{idx+1}/{len(selected_files)}] Processed {os.path.basename(s_file)} ({min_len} rows -> +{new_seq_count} sequences) in {time.time()-t_file:.2f}s", flush=True)
            except Exception as e:
                print(f"Error reading {s_file}: {e}", flush=True)

        print(f"Loaded {len(self.sequences_x)} total sequences ({self.split.upper()}) in {time.time()-t_start:.2f}s.\n", flush=True)

    def __len__(self): return len(self.sequences_x)
    def __getitem__(self, idx):
        return torch.from_numpy(self.sequences_x[idx]), {
            "v": torch.from_numpy(self.sequences_v[idx]),
            "delta_v": torch.from_numpy(self.sequences_dv[idx]),
            "zupt": torch.from_numpy(self.sequences_zupt[idx]),
            "pitch": torch.from_numpy(self.sequences_pitch[idx]),
            "regime": torch.from_numpy(self.sequences_regime[idx]),
        }

    def get_speed_balanced_sampler(self) -> WeightedRandomSampler:
        """Build a WeightedRandomSampler that equalizes speed-bin representation."""
        speed_arr = np.array(self.sequence_mean_speeds)
        bin_edges = [0, 10, 20, 30, 40, 50, 60, 80, 300]
        bin_indices = np.digitize(speed_arr, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)

        bin_counts = np.bincount(bin_indices, minlength=len(bin_edges) - 1).astype(np.float64)
        bin_counts = np.maximum(bin_counts, 1.0)
        bin_weights = 1.0 / bin_counts

        sample_weights = bin_weights[bin_indices]
        sample_weights = sample_weights / sample_weights.sum()

        return WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(self),
            replacement=True,
        )


# ---------------------------------------------------------------------------
# 3. Model Architecture: DualStreamSpeedNet
# ---------------------------------------------------------------------------

class ConvNeXtBlock1D(nn.Module):
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
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


class SpectralVelocityStream(nn.Module):
    """
    Multi-scale learnable wavelet filter bank for absolute speed estimation.
    Applies 3 parallel 1D convolutions at different kernel sizes to the vertical
    acceleration channel (az, channel index 2) to capture suspension vibration
    at different frequency scales, then pools spectral power and maps to speed.
    """
    def __init__(self, window_size=48):
        super().__init__()
        # Multi-scale 1D convolutions on az (channel 2)
        # Also process ax (0) and ay (1) for lateral/longitudinal vibration
        self.n_input_ch = 3  # ax, ay, az
        self.scale_small = nn.Conv1d(self.n_input_ch, 16, kernel_size=3, padding=1)
        self.scale_mid = nn.Conv1d(self.n_input_ch, 16, kernel_size=7, padding=3)
        self.scale_large = nn.Conv1d(self.n_input_ch, 16, kernel_size=15, padding=7)

        self.norm = nn.LayerNorm(48)
        self.pool_mlp = nn.Sequential(
            nn.Linear(48 * 3, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
        )
        # Output: scalar v_direct (absolute speed in m/s, unclamped)
        self.head = nn.Linear(32, 1)

    def forward(self, x_18ch):
        """
        Args:
            x_18ch: (B, 18, W) — full 18-channel IMU features

        Returns:
            v_direct: (B,) — predicted absolute speed in m/s
            spectral_features: (B, 32) — intermediate features for fusion gate
        """
        # Extract acceleration channels: ax=0, ay=1, az=2
        accel_3ch = x_18ch[:, :3, :]  # (B, 3, W)

        # Multi-scale convolution
        s1 = self.scale_small(accel_3ch)  # (B, 16, W)
        s2 = self.scale_mid(accel_3ch)    # (B, 16, W)
        s3 = self.scale_large(accel_3ch)  # (B, 16, W)

        # Spectral power: RMS across channels for each scale
        p1 = torch.sqrt((s1 ** 2).mean(dim=1) + 1e-8)  # (B, W)
        p2 = torch.sqrt((s2 ** 2).mean(dim=1) + 1e-8)  # (B, W)
        p3 = torch.sqrt((s3 ** 2).mean(dim=1) + 1e-8)  # (B, W)

        # Concatenate power profiles
        power_cat = torch.cat([p1, p2, p3], dim=1)  # (B, 3*W)

        # MLP to speed
        features = self.pool_mlp(power_cat)  # (B, 32)
        v_direct = self.head(features).squeeze(-1)  # (B,)

        # v_direct is unclamped — can be negative, zero-floor applied in fusion
        return v_direct, features


class DualStreamSpeedNet(nn.Module):
    """
    Experiment 6D architecture.

    Stream A (Spectral): Multi-scale wavelet → v_direct (absolute speed)
    Stream B (Kinematic): ConvNeXt backbone → Δv (unconstrained increment)

    Fusion: v_t = α·v_direct + (1-α)·(v_prev + Δv)
    ZUPT: hysteresis hard gate
    Zero-floor: max(0, v_fused)
    """
    def __init__(self, in_channels=18, window_size=48, embed_dims=[48, 64, 96, 128], depths=[2, 2, 4, 2]):
        super().__init__()

        # --- Shared ConvNeXt-1D Backbone ---
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

        # --- State Anchor Projection ---
        self.state_proj = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, 32))

        # --- Stream B: Kinematic Δv Head (unconstrained) ---
        self.head_delta_v = nn.Sequential(
            nn.Linear(embed_dims[3] + 32, 64),  # pooled features + state embed
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # --- Stream A: Spectral Velocity ---
        self.spectral_stream = SpectralVelocityStream(window_size=window_size)

        # --- Fusion Gate: learned α_t ∈ [0, 1] ---
        # Input: pooled backbone features + state embed + spectral features
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dims[3] + 32 + 32, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # --- Auxiliary Heads ---
        self.head_zupt = nn.Sequential(nn.Linear(embed_dims[3], 32), nn.GELU(), nn.Linear(32, 1))
        self.head_pitch = nn.Sequential(nn.Linear(embed_dims[3], 32), nn.GELU(), nn.Linear(32, 1))
        self.head_regime = nn.Sequential(nn.Linear(embed_dims[3], 32), nn.GELU(), nn.Linear(32, 7))

    def forward(self, x, v_anchor=None):
        """
        Args:
            x: (B, 18, W) — 18-channel IMU features
            v_anchor: (B,) or scalar — previous velocity state in RAW m/s

        Returns:
            dict with mu_v, v_direct, delta_v, alpha, p_zupt, pitch, regime_logits, e_kinetic
        """
        # --- Backbone feature extraction ---
        x_normed = self.input_norm(x)
        feat = self.stem(x_normed)
        feat = self.trans1(self.stage1(feat))
        feat = self.trans2(self.stage2(feat))
        feat = self.trans3(self.stage3(feat))
        feat = self.stage4(feat)

        tokens = feat.permute(0, 2, 1)
        norm_tokens = self.mha_norm(tokens)
        attn_out, _ = self.mha(norm_tokens, norm_tokens, norm_tokens)
        tokens = tokens + attn_out
        pooled = self.pool_norm(tokens.mean(dim=1) + tokens[:, -1, :])  # (B, 128)

        # --- State anchor ---
        B = x.shape[0]
        if v_anchor is None:
            v_anchor = torch.zeros(B, device=x.device, dtype=x.dtype)
        elif isinstance(v_anchor, (int, float)):
            v_anchor = torch.full((B,), float(v_anchor), device=x.device, dtype=x.dtype)
        elif v_anchor.dim() == 0:
            v_anchor = v_anchor.expand(B)
        elif v_anchor.dim() == 2 and v_anchor.shape[1] == 1:
            v_anchor = v_anchor.squeeze(-1)
        elif v_anchor.shape[0] == 1 and B > 1:
            v_anchor = v_anchor.expand(B)

        v_anchor_norm = (v_anchor / 30.0).unsqueeze(-1)  # internal normalization
        state_embed = self.state_proj(v_anchor_norm)  # (B, 32)

        # --- Stream B: Kinematic Δv ---
        fused_backbone = torch.cat([pooled, state_embed], dim=-1)  # (B, 160)
        delta_v = self.head_delta_v(fused_backbone).squeeze(-1)  # (B,) unconstrained

        # --- Stream A: Spectral velocity ---
        v_direct, spectral_feat = self.spectral_stream(x)  # (B,), (B, 32)

        # --- Fusion gate α ---
        gate_input = torch.cat([pooled, state_embed, spectral_feat], dim=-1)  # (B, 192)
        alpha = torch.sigmoid(self.fusion_gate(gate_input).squeeze(-1))  # (B,) ∈ [0,1]

        # --- Fused velocity ---
        v_kinematic = v_anchor + delta_v  # unconstrained, can be negative
        v_fused = alpha * v_direct + (1.0 - alpha) * v_kinematic

        # --- Kinematic energy for ZUPT gate ---
        # ax=ch0, ay=ch1, az=ch2 of original x (pre-normalization)
        ax_raw = x[:, 0, :]  # (B, W)
        ay_raw = x[:, 1, :]
        az_raw = x[:, 2, :]
        e_kinetic = (ax_raw.std(dim=1) ** 2 + ay_raw.std(dim=1) ** 2 + az_raw.std(dim=1) ** 2)  # (B,)

        # --- ZUPT hysteresis gate ---
        p_zupt = torch.sigmoid(self.head_zupt(pooled).squeeze(-1))  # (B,)

        # Hard gate: if p_zupt > 0.70 AND e_kinetic < 0.02, clamp to zero
        zupt_active = (p_zupt > 0.70) & (e_kinetic < 0.02)
        mu_v = torch.where(zupt_active, torch.zeros_like(v_fused), torch.clamp(v_fused, min=0.0))

        return {
            "mu_v": mu_v,
            "v_direct": v_direct,
            "delta_v": delta_v,
            "alpha": alpha,
            "p_zupt": p_zupt,
            "e_kinetic": e_kinetic,
            "pitch": self.head_pitch(pooled).squeeze(-1),
            "regime_logits": self.head_regime(pooled),
        }


# ---------------------------------------------------------------------------
# 4. Canonical Evaluation (embedded for self-contained Kaggle execution)
# ---------------------------------------------------------------------------

def evaluate_closed_loop(model, val_loader, device, verbose=True):
    """Canonical closed-loop rollout evaluation."""
    speed_bins = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 80), (80, 300)]
    bin_names = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-80", "80+"]

    model.eval()
    all_preds, all_gts = [], []
    v_state = torch.zeros(1, device=device, dtype=torch.float32)

    with torch.no_grad():
        for x_seq, targets in val_loader:
            x_seq = x_seq.to(device)
            v_gt_seq = targets["v"].to(device)
            B, L, C, W = x_seq.shape
            for t in range(L):
                out = model(x_seq[:, t], v_anchor=v_state)
                mu_t = out["mu_v"]
                all_preds.append(mu_t.item() * 3.6)
                all_gts.append(v_gt_seq[0, t].item() * 3.6)
                v_state = mu_t.detach()

    preds = np.array(all_preds)
    gts = np.array(all_gts)
    N = len(preds)

    errors = np.abs(preds - gts)
    signed = preds - gts
    raw_mae = float(errors.mean())

    sigma_pred = float(np.std(preds))
    sigma_gt = float(np.std(gts))
    sigma_ratio = sigma_pred / sigma_gt if sigma_gt > 1e-6 else 0.0

    if sigma_pred > 1e-6 and sigma_gt > 1e-6:
        pearson_r = float(np.corrcoef(preds, gts)[0, 1])
        slope, intercept = np.polyfit(gts, preds, 1)
    else:
        pearson_r, slope, intercept = 0.0, 0.0, 0.0

    bin_maes = []
    bin_signed_dict = {}
    for (low, high), bn in zip(speed_bins, bin_names):
        mask = (gts >= low) & (gts < high)
        n_bin = int(mask.sum())
        if n_bin > 0:
            bin_maes.append(float(errors[mask].mean()))
            bin_signed_dict[bn] = float(signed[mask].mean())
        else:
            bin_maes.append(0.0)
            bin_signed_dict[bn] = 0.0

    balanced_mae = float(np.mean(bin_maes))

    if verbose:
        script_hash = _get_script_hash()
        print(f"\n{'='*105}")
        print(f" CANONICAL EVALUATION (script hash: {script_hash})")
        print(f"{'='*105}")
        print(f"  N = {N:,} | Raw MAE: {raw_mae:.2f} km/h | Balanced MAE: {balanced_mae:.2f} km/h")
        print(f"  Regression: Pred = {slope:.4f} × GT {intercept:+.2f} km/h | r = {pearson_r:.4f} (r²={pearson_r**2:.4f})")
        print(f"  σ(pred) = {sigma_pred:.2f} km/h | σ(GT) = {sigma_gt:.2f} km/h | σ ratio = {sigma_ratio:.4f}")
        bin_mae_str = " | ".join([f"{bn}:{bm:4.1f}" for bn, bm in zip(bin_names, bin_maes)])
        bin_sgn_str = " | ".join([f"{bn}:{bs:+5.1f}" for bn, bs in zip(bin_names, [bin_signed_dict[b] for b in bin_names])])
        print(f"  Bin MAEs   : [ {bin_mae_str} ]")
        print(f"  Bin Signed : [ {bin_sgn_str} ]")
        print(f"{'='*105}\n")

    return {
        "raw_mae": raw_mae, "balanced_mae": balanced_mae,
        "slope": float(slope), "intercept": float(intercept),
        "pearson_r": pearson_r, "r_squared": pearson_r**2,
        "sigma_pred": sigma_pred, "sigma_gt": sigma_gt, "sigma_ratio": sigma_ratio,
        "bin_maes": dict(zip(bin_names, bin_maes)),
        "bin_signed": bin_signed_dict,
        "n_samples": N,
    }


# ---------------------------------------------------------------------------
# 5. Training Engine
# ---------------------------------------------------------------------------

def auto_resolve_dataset_dir() -> str:
    print("\n--- Scanning for Dataset Files ---", flush=True)
    if os.path.exists("/kaggle/input"):
        print(f"Contents of /kaggle/input:", flush=True)
        for root, dirs, files in os.walk("/kaggle/input"):
            print(f"  {root} -> dirs: {dirs[:5]}, files: {files[:5]} (total {len(files)})", flush=True)

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

    search_roots = ["/kaggle/working", "/kaggle/input", "ml/external", "."]
    for s_root in search_roots:
        if not os.path.exists(s_root): continue
        found = glob.glob(os.path.join(s_root, "**", "S-S1.csv"), recursive=True)
        if found:
            candidate_dir = os.path.dirname(os.path.dirname(os.path.dirname(found[0])))
            print(f"Discovered valid dataset root: {candidate_dir}", flush=True)
            return candidate_dir

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

    raise FileNotFoundError("Could not find dataset path!")


def run_training():
    script_hash = _get_script_hash()
    print("\n=======================================================", flush=True)
    print("      EXPERIMENT 6D: DUAL-STREAM SPECTRAL KINEMATICS   ", flush=True)
    print("=======================================================", flush=True)
    print(f"Script Hash         : {script_hash}", flush=True)
    print(f"Python Version      : {sys.version.split()[0]}", flush=True)
    print(f"PyTorch Version     : {torch.__version__}", flush=True)
    print(f"CUDA Available      : {torch.cuda.is_available()}", flush=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"CUDA Runtime Version: {torch.version.cuda}", flush=True)
        print(f"Device Name         : {torch.cuda.get_device_name(0)}", flush=True)
        cap = torch.cuda.get_device_capability(0)
        print(f"Compute Capability  : sm_{cap[0]}{cap[1]}", flush=True)
        test_x = torch.randn(64, 64, device=device)
        test_y = (test_x @ test_x).sum().item()
        print(f"CUDA Smoke Test     : SUCCESS (sum={test_y:.4f})", flush=True)

    print("=======================================================\n", flush=True)

    data_dir = auto_resolve_dataset_dir()
    print(f"Using dataset directory: {data_dir}\n", flush=True)

    out_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "ml/weights"
    os.makedirs(out_dir, exist_ok=True)

    # Dataset & DataLoaders
    train_ds = SequencePhysicsDataset(data_dir=data_dir, split="train", seq_len=32, seq_stride=16)
    val_ds = SequencePhysicsDataset(data_dir=data_dir, split="val", seq_len=32, seq_stride=32)

    # Speed-balanced sampler for training
    balanced_sampler = train_ds.get_speed_balanced_sampler()
    train_loader = DataLoader(
        train_ds, batch_size=64, sampler=balanced_sampler,
        drop_last=True, num_workers=2 if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # Speed bins
    bins = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 80), (80, 300)]
    bin_names = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-80", "80+"]

    # Model
    model = DualStreamSpeedNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: DualStreamSpeedNet ({total_params:,} parameters)", flush=True)

    # Optimizer & LR Schedule: linear warmup (1 epoch) + cosine decay (14 epochs)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    num_epochs = 15
    warmup_epochs = 1
    batches_per_epoch = len(train_loader)

    def lr_lambda(step):
        warmup_steps = warmup_epochs * batches_per_epoch
        total_steps = num_epochs * batches_per_epoch
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(1e-5 / 1e-3, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return 1.0

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss functions
    loss_huber = nn.SmoothL1Loss(beta=1.0)
    loss_bce_zupt = nn.BCELoss()
    loss_huber_pitch = nn.SmoothL1Loss(beta=0.1)

    best_balanced_mae = float("inf")
    best_raw_mae = float("inf")
    best_balanced_epoch = 0
    best_raw_epoch = 0

    save_best_balanced = os.path.join(out_dir, "exp6d_best_balanced.pt")
    save_best_raw = os.path.join(out_dir, "exp6d_best_raw.pt")
    save_final = os.path.join(out_dir, "exp6d_final.pt")
    history = []

    print("=======================================================================", flush=True)
    print("  Architecture  : Dual-Stream (Spectral v_direct + Kinematic Δv)       ", flush=True)
    print("  Fusion        : α·v_direct + (1-α)·(v_prev + Δv), learned α          ", flush=True)
    print("  ZUPT Gate     : Hysteresis (p_ZUPT>0.70 AND E_kin<0.02 → v=0)        ", flush=True)
    print("  Δv            : Unconstrained (no ReLU clamp)                         ", flush=True)
    print("  Loss          : Huber + 0.30·Huber(v_direct) + 0.25·VarLoss           ", flush=True)
    print("                  + 0.20·BCE(ZUPT) + 0.10·Huber(pitch)                  ", flush=True)
    print("  LR Schedule   : Warmup 1ep → Cosine decay (no restarts)              ", flush=True)
    print("  Sampling      : Speed-balanced (WeightedRandomSampler)                ", flush=True)
    print(f"  Training      : {num_epochs} epochs, batch_size=64                   ", flush=True)
    print("=======================================================================\n", flush=True)

    # Pre-flight check
    sample_x, sample_targets = next(iter(train_loader))
    sample_x = sample_x.to(device)
    B_s, L_s, C_s, W_s = sample_x.shape
    v_state_s = torch.zeros(B_s, device=device, dtype=torch.float32)
    sample_out = model(sample_x[:, 0], v_anchor=v_state_s)
    sample_loss = loss_huber(sample_out["mu_v"], sample_targets["v"][:, 0].to(device))
    sample_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(">>> [PRE-FLIGHT PASSED] Forward/backward verified <<<\n", flush=True)

    # -----------------------------------------------------------------------
    # Training Loop
    # -----------------------------------------------------------------------
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, total_mae, batches = 0.0, 0.0, 0

        for x_seq, targets in train_loader:
            x_seq = x_seq.to(device, non_blocking=True)
            B, L, C, W = x_seq.shape
            v_gt = targets["v"].to(device, non_blocking=True)
            zupt_gt = targets["zupt"].to(device, non_blocking=True)
            pitch_gt = targets["pitch"].to(device, non_blocking=True)

            optimizer.zero_grad()
            v_state = torch.zeros(B, device=device, dtype=torch.float32)

            pred_mu_list, pred_vdirect_list = [], []

            for t in range(L):
                out = model(x_seq[:, t], v_anchor=v_state)
                pred_mu_list.append(out["mu_v"])
                pred_vdirect_list.append(out["v_direct"])
                v_state = out["mu_v"].detach()

            mu_all = torch.stack(pred_mu_list, dim=1)     # (B, L)
            vd_all = torch.stack(pred_vdirect_list, dim=1) # (B, L)
            v_gt_flat = v_gt.view(-1)
            mu_flat = mu_all.view(-1)
            vd_flat = vd_all.view(-1)

            # --- Loss computation ---
            # 1. Primary Huber loss on fused velocity
            l_primary = loss_huber(mu_flat, v_gt_flat)

            # 2. Spectral stream auxiliary Huber loss
            l_spectral = loss_huber(vd_flat, v_gt_flat)

            # 3. Variance-preserving loss (per-batch)
            sigma_pred_batch = torch.std(mu_flat)
            sigma_gt_batch = torch.std(v_gt_flat)
            l_var = (sigma_pred_batch - sigma_gt_batch) ** 2

            # 4. ZUPT loss (last timestep's ZUPT prediction vs GT)
            # Get ZUPT from last forward pass
            last_out = model(x_seq[:, -1], v_anchor=mu_all[:, -2].detach() if L > 1 else torch.zeros(B, device=device))
            l_zupt = loss_bce_zupt(last_out["p_zupt"], zupt_gt[:, -1])

            # 5. Pitch loss
            l_pitch = loss_huber_pitch(last_out["pitch"], pitch_gt[:, -1])

            loss = (
                l_primary
                + 0.30 * l_spectral
                + 0.25 * l_var
                + 0.20 * l_zupt
                + 0.10 * l_pitch
            )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_mae += torch.abs(mu_flat - v_gt_flat).mean().item() * 3.6
            batches += 1

        ep_time = time.time() - t0
        train_loss = total_loss / max(1, batches)
        train_mae = total_mae / max(1, batches)
        cur_lr = optimizer.param_groups[0]["lr"]

        # -------------------------------------------------------------------
        # Validation (Canonical Closed-Loop Rollout)
        # -------------------------------------------------------------------
        eval_results = evaluate_closed_loop(model, val_loader, device, verbose=False)

        val_mae = eval_results["raw_mae"]
        balanced_mae = eval_results["balanced_mae"]
        slope = eval_results["slope"]
        intercept = eval_results["intercept"]
        pearson_r = eval_results["pearson_r"]
        sigma_ratio = eval_results["sigma_ratio"]

        bin_mae_str = " | ".join([f"{bn}:{eval_results['bin_maes'][bn]:4.1f}" for bn in bin_names])
        bin_sgn_str = " | ".join([f"{bn}:{eval_results['bin_signed'][bn]:+5.1f}" for bn in bin_names])

        print(f"Epoch [{epoch:02d}/{num_epochs}] ({ep_time:.1f}s, lr={cur_lr:.1e}) | Train Loss: {train_loss:.4f} | Train MAE: {train_mae:5.2f} | Val MAE: {val_mae:5.2f} | Bal MAE: {balanced_mae:5.2f} | r: {pearson_r:.3f}", flush=True)
        print(f"   Fit: Pred = {slope:.4f}*GT {intercept:+5.2f} km/h | σ_ratio={sigma_ratio:.3f} | σ(P)={eval_results['sigma_pred']:.1f} | σ(GT)={eval_results['sigma_gt']:.1f}", flush=True)
        print(f"   Bin MAEs   : [ {bin_mae_str} ]", flush=True)
        print(f"   Bin Signed : [ {bin_sgn_str} ]", flush=True)

        ep_record = {
            "epoch": epoch, "train_loss": train_loss, "train_mae": train_mae,
            "val_mae": val_mae, "balanced_mae": balanced_mae,
            "slope": slope, "intercept": intercept,
            "pearson_r": pearson_r, "sigma_ratio": sigma_ratio,
            "sigma_pred": eval_results["sigma_pred"], "sigma_gt": eval_results["sigma_gt"],
            "bin_maes": eval_results["bin_maes"], "bin_signed": eval_results["bin_signed"],
            "lr": cur_lr, "epoch_time_s": ep_time,
        }
        history.append(ep_record)

        if balanced_mae < best_balanced_mae:
            best_balanced_mae = balanced_mae
            best_balanced_epoch = epoch
            torch.save(model.state_dict(), save_best_balanced)
            print(f"   >>> [SAVED BEST BALANCED] Epoch {epoch} (Balanced MAE: {balanced_mae:.2f} km/h) <<<", flush=True)

        if val_mae < best_raw_mae:
            best_raw_mae = val_mae
            best_raw_epoch = epoch
            torch.save(model.state_dict(), save_best_raw)
            print(f"   >>> [SAVED BEST RAW] Epoch {epoch} (Val MAE: {val_mae:.2f} km/h) <<<", flush=True)
        print("", flush=True)

    # Save final
    torch.save(model.state_dict(), save_final)
    with open(os.path.join(out_dir, "exp6d_history.json"), "w") as f:
        json.dump({
            "script_hash": script_hash,
            "best_balanced_epoch": best_balanced_epoch,
            "best_balanced_mae": best_balanced_mae,
            "best_raw_epoch": best_raw_epoch,
            "best_raw_mae": best_raw_mae,
            "history": history,
        }, f, indent=2)

    # Final evaluation on best balanced checkpoint
    print("\n=== FINAL EVALUATION ON BEST BALANCED CHECKPOINT ===", flush=True)
    model.load_state_dict(torch.load(save_best_balanced, map_location=device, weights_only=False))
    final_results = evaluate_closed_loop(model, val_loader, device, verbose=True)

    print(f"\nTraining Complete!")
    print(f"  Best Balanced MAE : {save_best_balanced} (Epoch {best_balanced_epoch})")
    print(f"  Best Raw MAE      : {save_best_raw} (Epoch {best_raw_epoch})")
    print(f"  Final Checkpoint  : {save_final}")
    print(f"  Script Hash       : {script_hash}")


if __name__ == "__main__":
    run_training()
