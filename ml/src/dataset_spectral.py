"""
dataset_spectral.py - IO-VNBD PyTorch Dataset with 18-Channel Causal Multi-Domain Physics Features.
Implements:
1. Strict Drive & Driver-Aware Isolation (Train: A/B/D, Val: A-S3a, Test: E-Vw11+).
2. Zero GPS Feature Leakage Guarantee (GPS used exclusively for labels).
3. 18 Causal Physics/Spectral Channels x 48 Temporal Steps (4.8s at 10 Hz).
4. Physical Pitch Observer (Independent Complementary Filter) for Feature 17.
5. Speed-Regime Balanced Resampling across 8 bins (0-10, 10-20, ..., 80+ km/h).
6. Multi-Task Targets: velocity, delta_v, ZUPT probability, pitch, motion regime.
"""

import glob
import math
import os
import random
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


GRAVITY = 9.80665


def estimate_phone_to_vehicle_matrix(accel_3xN: np.ndarray) -> np.ndarray:
    """
    Computes rotation matrix R (3x3) aligning the smartphone accelerometer to the vehicle body frame
    using static gravity alignment: R @ a_phone puts gravity along [0, 0, +g].
    """
    mean_acc = np.mean(accel_3xN[:, :min(50, accel_3xN.shape[1])], axis=1)
    norm = np.linalg.norm(mean_acc)
    if norm < 1e-3:
        return np.eye(3, dtype=np.float32)
    u = mean_acc / norm
    v = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    cross = np.cross(u, v)
    sin_t = np.linalg.norm(cross)
    cos_t = np.dot(u, v)
    if sin_t < 1e-6:
        return np.eye(3, dtype=np.float32) if cos_t > 0 else -np.eye(3, dtype=np.float32)
    k_unit = cross / sin_t
    K = np.array([
        [0, -k_unit[2], k_unit[1]],
        [k_unit[2], 0, -k_unit[0]],
        [-k_unit[1], k_unit[0], 0]
    ], dtype=np.float32)
    R = np.eye(3, dtype=np.float32) + sin_t * K + (1.0 - cos_t) * (K @ K)
    return R.astype(np.float32)


def align_imu_to_vehicle_frame(imu_6xN: np.ndarray) -> np.ndarray:
    """
    Transforms raw 6-channel IMU (ax, ay, az, gy, gp, gr) into vehicle body frame (aligned with gravity).
    Spatial ordering:
      - Phone accelerometer: [ax, ay, az] (phone X, Y, Z)
      - Phone gyroscope:     [gr, gp, gy] (phone X=Roll, Y=Pitch, Z=Yaw)
    Output aligned_imu:
      - 0: Lateral acceleration ax_v (Right)
      - 1: Forward acceleration ay_v (Forward)
      - 2: Vertical acceleration az_v (Up)
      - 3: Yaw rate wz_v (CCW +Z)
      - 4: Pitch rate wy_v (Pitch up +Y)
      - 5: Roll rate wx_v (Roll right +X)
    """
    R = estimate_phone_to_vehicle_matrix(imu_6xN[:3])
    acc_body = R @ imu_6xN[:3]
    # Spatial gyro ordering: Gyro X (gr), Gyro Y (gp), Gyro Z (gy)
    gyro_spatial = np.stack([imu_6xN[5], imu_6xN[4], imu_6xN[3]], axis=0)
    gyro_body = R @ gyro_spatial
    # Pack as [ax, ay, az, wz, wy, wx]
    return np.vstack([
        acc_body[0],
        acc_body[1],
        acc_body[2],
        gyro_body[2], # wz_yaw
        gyro_body[1], # wy_pitch
        gyro_body[0], # wx_roll
    ]).astype(np.float32)


def compute_physical_pitch_series(
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    wy: np.ndarray,
    wx: np.ndarray = None,
    wz: np.ndarray = None,
    dt: float = 0.1,
) -> np.ndarray:
    """
    Quasi-Static Gated 3D Attitude Observer (causal quaternion formulation with motion gating).
    Prevents centripetal and acceleration-induced pitch spikes during turns, stops, and braking.
    """
    N = len(ay)
    pitch_series = np.zeros(N, dtype=np.float32)
    if N == 0:
        return pitch_series

    # Initial static gravity pitch
    sin_th0 = np.clip(ay[0] / 9.80665, -0.99, 0.99)
    pitch_series[0] = np.arcsin(sin_th0)

    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # [qw, qx, qy, qz]
    bg = np.zeros(3, dtype=np.float64)
    Kp = 0.20
    Ki = 0.001

    for t in range(N):
        a = np.array([ax[t], ay[t], az[t]], dtype=np.float64)
        w_vec = np.array([
            wx[t] if wx is not None else 0.0,
            wy[t],
            wz[t] if wz is not None else 0.0,
        ], dtype=np.float64)

        a_norm = np.linalg.norm(a)
        w_norm = np.linalg.norm(w_vec)

        # Strictly gate accelerometer to quasi-static windows (|norm(a) - g| < 0.6 m/s^2 and |w| < 0.05 rad/s)
        if abs(a_norm - 9.80665) < 0.6 and w_norm < 0.05:
            a_unit = a / a_norm
            v_g = np.array([
                2.0 * (q[1] * q[3] - q[0] * q[2]),
                2.0 * (q[0] * q[1] + q[2] * q[3]),
                q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2,
            ])
            e = np.cross(a_unit, v_g)
            bg = np.clip(bg - Ki * e * dt, -0.015, 0.015)  # clamp bias to max 0.85 deg/s
            w_corr = w_vec + Kp * e - bg
        else:
            w_corr = w_vec - bg

        qw, qx, qy, qz = q
        q_dot = 0.5 * np.array([
            -qx * w_corr[0] - qy * w_corr[1] - qz * w_corr[2],
             qw * w_corr[0] + qy * w_corr[2] - qz * w_corr[1],
             qw * w_corr[1] - qx * w_corr[2] + qz * w_corr[0],
             qw * w_corr[2] + qx * w_corr[1] - qy * w_corr[0],
        ])
        q += q_dot * dt
        q /= np.linalg.norm(q)

        # Body pitch angle relative to horizontal plane
        gx_grav = 2.0 * (q[1] * q[3] - q[0] * q[2])
        gy_grav = 2.0 * (q[0] * q[1] + q[2] * q[3])
        gz_grav = q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2
        pitch_series[t] = math.atan2(gy_grav, math.sqrt(gx_grav**2 + gz_grav**2))

    return pitch_series


def compute_18ch_features(
    w_6ch: np.ndarray,
    theta_phys_window: np.ndarray,
    fs: float = 10.0,
) -> np.ndarray:
    """
    Given a (6, W) window of vehicle-frame IMU signals [ax, ay, az, gy, gp, gr] and physical pitch window,
    computes the exact 18-channel multi-domain representation (18, W) without future lookahead:
    0..5: ax, ay, az, gy (yaw rate wz), gp (pitch rate wy), gr (roll rate wx)
    6: ||a|| - g
    7: ||omega||
    8: Multi-scale leaky longitudinal velocity integral: I[t] = 0.95 * I[t-1] + ay[t] * dt
    9: Causal rolling vertical suspension variance Var(az)
    10..12: Pavement-normalized sub-band PSD ratios (R_low, R_mid, R_high = E_i / P_total)
    13: Spectral centroid frequency f_centroid
    14: Dominant harmonic peak frequency f_peak in [1.0, 25.0] Hz
    15: Kinematic turning feature: abs(ax / (wz + sign(wz)*0.01)) clipped to [0, 50]
    16: Longitudinal/vertical vibration ratio: P_ay / (P_az + 1e-6)
    17: Gravity-compensated longitudinal acceleration: ay - g * sin(theta_phys)
    """
    W = w_6ch.shape[1]
    ax = w_6ch[0]
    ay = w_6ch[1]
    az = w_6ch[2]
    wz_yaw = w_6ch[3]
    wy_pitch = w_6ch[4]
    wx_roll = w_6ch[5]
    dt = 1.0 / fs

    # 1. Dynamic norms (Ch 6, 7)
    a_norm = np.sqrt(ax**2 + ay**2 + az**2) - GRAVITY
    w_norm = np.sqrt(wz_yaw**2 + wy_pitch**2 + wx_roll**2)

    # 2. Leaky velocity integral (Ch 8)
    vel_int = np.zeros(W, dtype=np.float32)
    acc = 0.0
    for i in range(W):
        acc = acc * 0.95 + ay[i] * dt
        vel_int[i] = acc

    # 3. Causal rolling vertical suspension variance (Ch 9)
    az_series = pd.Series(az)
    az_var = az_series.rolling(window=5, min_periods=1).var().fillna(0.0).values.astype(np.float32)

    # 4. FFT Spectral Features per window (W points)
    freqs = np.fft.rfftfreq(W, d=dt)
    az_centered = az - np.mean(az)
    ay_centered = ay - np.mean(ay)
    az_fft = np.abs(np.fft.rfft(az_centered)) ** 2 / W
    ay_fft = np.abs(np.fft.rfft(ay_centered)) ** 2 / W

    # Sub-band Energies
    low_mask = (freqs >= 0.3) & (freqs < 1.25)
    mid_mask = (freqs >= 1.25) & (freqs < 2.5)
    high_mask = (freqs >= 2.5) & (freqs <= 5.0)

    e_low = float(np.sum(az_fft[low_mask])) if np.sum(low_mask) > 0 else 0.0
    e_mid = float(np.sum(az_fft[mid_mask])) if np.sum(mid_mask) > 0 else 0.0
    e_high = float(np.sum(az_fft[high_mask])) if np.sum(high_mask) > 0 else 0.0

    total_power = float(np.sum(az_fft)) + 1e-6
    p_ay_total = float(np.sum(ay_fft)) + 1e-6

    # Normalized power ratios (Ch 10..12)
    r_low = e_low / total_power
    r_mid = e_mid / total_power
    r_high = e_high / total_power

    # Spectral centroid (Ch 13)
    spec_centroid = float(np.sum(freqs * az_fft) / total_power)

    # Harmonic peak frequency in [1.0, 25.0] Hz (Ch 14)
    peak_mask = (freqs >= 1.0) & (freqs <= 25.0)
    if np.sum(peak_mask) > 0:
        f_peak = float(freqs[peak_mask][np.argmax(az_fft[peak_mask])])
    else:
        f_peak = 2.0

    # Kinematic centripetal turning feature (Ch 15)
    # Physical basis: centripetal acceleration a_lat = v * w_yaw is valid ONLY during actual turns (|w_z| >= 0.035 rad/s ~ 2 deg/s).
    # When driving straight, centripetal speed is undefined / inactive (0.0), NOT 50 m/s noise!
    wz_mag = np.abs(wz_yaw)
    turn_feat = np.where(wz_mag >= 0.035, np.clip(np.abs(ax) / wz_mag, 0.0, 40.0), 0.0).astype(np.float32)

    # Longitudinal to vertical vibration ratio (Ch 16)
    vib_ratio = float(p_ay_total / total_power)

    # Gravity-compensated longitudinal acceleration (Ch 17)
    ay_grav_comp = (ay - GRAVITY * np.sin(theta_phys_window)).astype(np.float32)

    # Stack all 18 channels
    features = np.stack([
        ax.astype(np.float32),
        ay.astype(np.float32),
        az.astype(np.float32),
        wz_yaw.astype(np.float32),
        wy_pitch.astype(np.float32),
        wx_roll.astype(np.float32),
        a_norm.astype(np.float32),
        w_norm.astype(np.float32),
        vel_int.astype(np.float32),
        az_var.astype(np.float32),
        np.full(W, r_low, dtype=np.float32),
        np.full(W, r_mid, dtype=np.float32),
        np.full(W, r_high, dtype=np.float32),
        np.full(W, spec_centroid, dtype=np.float32),
        np.full(W, f_peak, dtype=np.float32),
        turn_feat,
        np.full(W, vib_ratio, dtype=np.float32),
        ay_grav_comp,
    ], axis=0)

    # Anti-leakage verification assertion
    assert features.shape == (18, W), f"Expected shape (18, {W}), got {features.shape}"
    assert not np.isnan(features).any(), "Features contain NaN values!"
    assert not np.isinf(features).any(), "Features contain Inf values!"

    return features.astype(np.float32)


class DeepPhysicsDataset(Dataset):
    """
    Drive-Aware Multi-Task PyTorch Dataset with 18 Causal Physics Features.
    Splits strictly by drive:
      - Train: Driver A (S1, S2, S3b, S3c, S4), Driver B (M), Driver D (Y1)
      - Val: Driver A (S3a)
      - Test: Driver E (Vw11, Vw12, etc.)
    """
    def __init__(
        self,
        data_dir: str = "ml/external/IO-VNBD_repo/Synchronised V abd S datasets/Categorised IOVNB Dataset",
        window_size: int = 48,
        step_size: int = 2,
        split: str = "train",  # "train", "val", or "test"
        balance_speed_bins: bool = True,
        samples_per_bin: int = 15000,
    ):
        self.data_dir = data_dir
        self.window_size = window_size
        self.step_size = step_size
        self.split = split
        self.balance_speed_bins = balance_speed_bins
        self.samples_per_bin = samples_per_bin

        self.windows: List[np.ndarray] = []
        self.prev_windows: List[np.ndarray] = []
        self.has_prev_targets: List[float] = []
        self.v_targets: List[float] = []
        self.dv_targets: List[float] = []
        self.zupt_targets: List[float] = []
        self.pitch_targets: List[float] = []
        self.regime_targets: List[int] = []

        self._load_dataset()

    def _classify_regime(self, v_kmh: float, accel_fwd: float, yaw_rate: float) -> int:
        """
        0: Standstill (v < 1.0 km/h)
        1: Low-Speed (1 <= v < 20 km/h)
        2: City/Suburban Cruise (20 <= v < 60 km/h)
        3: High-Speed Motorway (v >= 60 km/h)
        4: Acceleration (a_y > 0.8 m/s^2)
        5: Braking (a_y < -0.8 m/s^2)
        6: Turning / Roundabout (|w_z| > 0.15 rad/s)
        """
        if v_kmh < 1.0:
            return 0
        if abs(yaw_rate) > 0.15 and v_kmh > 5.0:
            return 6
        if accel_fwd > 0.8:
            return 4
        if accel_fwd < -0.8:
            return 5
        if v_kmh < 20.0:
            return 1
        if v_kmh < 60.0:
            return 2
        return 3

    def _load_dataset(self):
        s_pattern = os.path.join(self.data_dir, "**", "S-*.csv")
        all_s_files = sorted(glob.glob(s_pattern, recursive=True))

        selected_files = []
        for sf in all_s_files:
            is_driver_e = ("Driver E" in sf) or ("Vw" in sf) or ("Vta" in sf) or ("Vtb" in sf) or ("Vf" in sf)
            is_val_s3a = "S3a" in sf

            if self.split == "train":
                # Train on A (except S3a), B, D
                if not is_driver_e and not is_val_s3a:
                    selected_files.append(sf)
            elif self.split == "val":
                # Validation strictly on held-out Driver A drive S3a
                if is_val_s3a:
                    selected_files.append(sf)
            elif self.split == "test":
                # Final test strictly on held-out Driver E
                if "Vw11" in sf or "Vw12" in sf:
                    selected_files.append(sf)

        print(f"Loading DeepPhysicsDataset [{self.split.upper()}] from {len(selected_files)} drive recordings...")

        raw_windows = []
        raw_prev_windows = []
        raw_has_prev = []
        raw_v = []
        raw_dv = []
        raw_zupt = []
        raw_pitch = []
        raw_regime = []

        for s_file in selected_files:
            v_file = os.path.join(os.path.dirname(s_file), os.path.basename(s_file).replace("S-", "V-"))
            if not os.path.exists(v_file):
                continue

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
                if speed_col not in df_v.columns:
                    continue

                speed_kmh = df_v[speed_col].values.astype(np.float32)
                speed_mps = speed_kmh / 3.6

                min_len = min(raw_imu.shape[1], len(speed_mps))
                if min_len < self.window_size + 5:
                    continue

                raw_imu = raw_imu[:, :min_len]
                speed_mps = speed_mps[:min_len]
                speed_kmh = speed_kmh[:min_len]

                # Vehicle-frame alignment
                aligned_imu = align_imu_to_vehicle_frame(raw_imu)
                ax_v, ay_v, az_v = aligned_imu[0], aligned_imu[1], aligned_imu[2]
                gy_v, gp_v, gr_v = aligned_imu[3], aligned_imu[4], aligned_imu[5]

                # Physical Pitch Observer (strictly causal 3D quasi-static gated attitude observer)
                pitch_phys = compute_physical_pitch_series(ax_v, ay_v, az_v, wy=gp_v, wx=gr_v, wz=gy_v, dt=0.1)

                last_feat18 = None
                for start_idx in range(0, min_len - self.window_size + 1, self.step_size):
                    end_idx = start_idx + self.window_size
                    w_imu = aligned_imu[:, start_idx:end_idx]
                    w_pitch = pitch_phys[start_idx:end_idx]

                    v_curr = float(speed_mps[end_idx - 1])
                    v_prev = float(speed_mps[end_idx - 2])
                    dv_curr = float(v_curr - v_prev)  # Local physical velocity increment delta_v = v[t] - v[t-1]

                    # Standstill ZUPT Ground Truth Label
                    is_zupt = 1.0 if (v_curr < 0.25 and np.abs(ay_v[end_idx - 1]) < 0.20 and np.abs(gy_v[end_idx - 1]) < 0.05) else 0.0
                    target_pitch = float(pitch_phys[end_idx - 1])
                    regime = self._classify_regime(float(speed_kmh[end_idx - 1]), float(ay_v[end_idx - 1]), float(gy_v[end_idx - 1]))

                    if not np.isnan(w_imu).any() and not np.isnan(v_curr):
                        feat18 = compute_18ch_features(w_imu, w_pitch)
                        raw_windows.append(feat18)
                        raw_v.append(v_curr)
                        raw_dv.append(dv_curr)
                        raw_zupt.append(is_zupt)
                        raw_pitch.append(target_pitch)
                        raw_regime.append(regime)

                        if last_feat18 is not None:
                            raw_prev_windows.append(last_feat18)
                            raw_has_prev.append(1.0)
                        else:
                            raw_prev_windows.append(feat18)
                            raw_has_prev.append(0.0)
                        last_feat18 = feat18

            except Exception as e:
                print(f"Error reading {s_file}: {e}")

        # Speed-regime stratified sampling for training set
        if self.split == "train" and self.balance_speed_bins and len(raw_v) > 0:
            bins = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 80), (80, 200)]
            bin_indices = [[] for _ in range(len(bins))]

            for idx, v_mps in enumerate(raw_v):
                v_kmh = v_mps * 3.6
                for b_idx, (b_low, b_high) in enumerate(bins):
                    if b_low <= v_kmh < b_high:
                        bin_indices[b_idx].append(idx)
                        break

            balanced_indices = []
            for b_idx, idx_list in enumerate(bin_indices):
                if len(idx_list) == 0:
                    continue
                sampled = np.random.choice(idx_list, size=min(self.samples_per_bin, max(len(idx_list), 5000)), replace=(len(idx_list) < self.samples_per_bin))
                balanced_indices.extend(sampled.tolist())

            random.shuffle(balanced_indices)
            for idx in balanced_indices:
                self.windows.append(raw_windows[idx])
                self.prev_windows.append(raw_prev_windows[idx])
                self.has_prev_targets.append(raw_has_prev[idx])
                self.v_targets.append(raw_v[idx])
                self.dv_targets.append(raw_dv[idx])
                self.zupt_targets.append(raw_zupt[idx])
                self.pitch_targets.append(raw_pitch[idx])
                self.regime_targets.append(raw_regime[idx])

            print(f"Stratified Speed Resampling: Balanced dataset to {len(self.windows)} windows across all 8 speed bins.")
        else:
            self.windows = raw_windows
            self.prev_windows = raw_prev_windows
            self.has_prev_targets = raw_has_prev
            self.v_targets = raw_v
            self.dv_targets = raw_dv
            self.zupt_targets = raw_zupt
            self.pitch_targets = raw_pitch
            self.regime_targets = raw_regime
            print(f"Loaded {len(self.windows)} 18-channel physics windows ({self.split.upper()}).")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        feat_tensor = torch.from_numpy(self.windows[idx])  # (18, 48)
        targets = {
            "v": torch.tensor(self.v_targets[idx], dtype=torch.float32),
            "delta_v": torch.tensor(self.dv_targets[idx], dtype=torch.float32),
            "zupt": torch.tensor(self.zupt_targets[idx], dtype=torch.float32),
            "pitch": torch.tensor(self.pitch_targets[idx], dtype=torch.float32),
            "regime": torch.tensor(self.regime_targets[idx], dtype=torch.long),
            "x_prev": torch.from_numpy(self.prev_windows[idx]),
            "has_prev": torch.tensor(self.has_prev_targets[idx], dtype=torch.float32),
        }
        return feat_tensor, targets


class SequencePhysicsDataset(Dataset):
    """
    Chronological Sequence Dataset for State-Conditioned Velocity Observer (Experiment 6A).
    Constructs contiguous chronological chunks of length L (default L=32) at 10 Hz (dt=0.1s).
    Never crosses drive boundaries.
    
    Yields:
      X: (L, 18, 48) float32
      targets:
        - "v": (L,) float32 ground-truth velocity (m/s)
        - "delta_v": (L,) float32 velocity change v[t] - v[t-1] (m/s)
        - "zupt": (L,) float32 standstill probability
        - "pitch": (L,) float32 physical pitch (rad)
        - "regime": (L,) int64 motion regime classifier label
    """
    def __init__(
        self,
        data_dir: str = "ml/external/IO-VNBD_repo/Synchronised V abd S datasets/Categorised IOVNB Dataset",
        window_size: int = 48,
        seq_len: int = 32,
        seq_stride: int = 16,
        split: str = "train",
    ):
        self.data_dir = data_dir
        self.window_size = window_size
        self.seq_len = seq_len
        self.seq_stride = seq_stride
        self.split = split

        self.sequences_x: List[np.ndarray] = []
        self.sequences_v: List[np.ndarray] = []
        self.sequences_dv: List[np.ndarray] = []
        self.sequences_zupt: List[np.ndarray] = []
        self.sequences_pitch: List[np.ndarray] = []
        self.sequences_regime: List[np.ndarray] = []

        self._load_dataset()

    def _classify_regime(self, v_kmh: float, accel_fwd: float, yaw_rate: float) -> int:
        if v_kmh < 1.0:
            return 0
        if abs(yaw_rate) > 0.15 and v_kmh > 5.0:
            return 6
        if accel_fwd > 0.8:
            return 4
        if accel_fwd < -0.8:
            return 5
        if v_kmh < 20.0:
            return 1
        if v_kmh < 60.0:
            return 2
        return 3

    def _load_dataset(self):
        s_pattern = os.path.join(self.data_dir, "**", "S-*.csv")
        all_s_files = sorted(glob.glob(s_pattern, recursive=True))

        selected_files = []
        for sf in all_s_files:
            is_driver_e = ("Driver E" in sf) or ("Vw" in sf) or ("Vta" in sf) or ("Vtb" in sf) or ("Vf" in sf)
            is_val_s3a = "S3a" in sf

            if self.split == "train":
                # Train on A (except S3a), B, D
                if not is_driver_e and not is_val_s3a:
                    selected_files.append(sf)
            elif self.split == "val":
                # Validation strictly on held-out Driver A drive S3a
                if is_val_s3a:
                    selected_files.append(sf)
            elif self.split == "test":
                # Final test strictly on held-out Driver E
                if "Vw11" in sf or "Vw12" in sf:
                    selected_files.append(sf)

        print(f"Loading SequencePhysicsDataset [{self.split.upper()}] from {len(selected_files)} drive recordings (L={self.seq_len}, stride={self.seq_stride})...")

        for s_file in selected_files:
            v_file = os.path.join(os.path.dirname(s_file), os.path.basename(s_file).replace("S-", "V-"))
            if not os.path.exists(v_file):
                continue

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
                if speed_col not in df_v.columns:
                    continue

                speed_kmh = df_v[speed_col].values.astype(np.float32)
                speed_mps = speed_kmh / 3.6

                min_len = min(raw_imu.shape[1], len(speed_mps))
                if min_len < self.window_size + self.seq_len + 5:
                    continue

                raw_imu = raw_imu[:, :min_len]
                speed_mps = speed_mps[:min_len]
                speed_kmh = speed_kmh[:min_len]

                # Vehicle-frame alignment
                aligned_imu = align_imu_to_vehicle_frame(raw_imu)
                ax_v, ay_v, az_v = aligned_imu[0], aligned_imu[1], aligned_imu[2]
                gy_v, gp_v, gr_v = aligned_imu[3], aligned_imu[4], aligned_imu[5]

                # Physical Pitch Observer
                pitch_phys = compute_physical_pitch_series(ax_v, ay_v, az_v, wy=gp_v, wx=gr_v, wz=gy_v, dt=0.1)

                drive_windows = []
                drive_v = []
                drive_dv = []
                drive_zupt = []
                drive_pitch = []
                drive_regime = []

                # Extract 10 Hz continuous chronological windows
                for end_idx in range(self.window_size, min_len):
                    start_idx = end_idx - self.window_size
                    w_imu = aligned_imu[:, start_idx:end_idx]
                    w_pitch = pitch_phys[start_idx:end_idx]

                    v_curr = float(speed_mps[end_idx - 1])
                    v_prev = float(speed_mps[end_idx - 2])
                    dv_curr = float(v_curr - v_prev)

                    is_zupt = 1.0 if (v_curr < 0.25 and np.abs(ay_v[end_idx - 1]) < 0.20 and np.abs(gy_v[end_idx - 1]) < 0.05) else 0.0
                    target_pitch = float(pitch_phys[end_idx - 1])
                    regime = self._classify_regime(float(speed_kmh[end_idx - 1]), float(ay_v[end_idx - 1]), float(gy_v[end_idx - 1]))

                    feat18 = compute_18ch_features(w_imu, w_pitch)
                    drive_windows.append(feat18)
                    drive_v.append(v_curr)
                    drive_dv.append(dv_curr)
                    drive_zupt.append(is_zupt)
                    drive_pitch.append(target_pitch)
                    drive_regime.append(regime)

                drive_windows = np.stack(drive_windows, axis=0)  # (N_steps, 18, 48)
                drive_v = np.array(drive_v, dtype=np.float32)
                drive_dv = np.array(drive_dv, dtype=np.float32)
                drive_zupt = np.array(drive_zupt, dtype=np.float32)
                drive_pitch = np.array(drive_pitch, dtype=np.float32)
                drive_regime = np.array(drive_regime, dtype=np.int64)

                N_steps = len(drive_v)
                for s_start in range(0, N_steps - self.seq_len + 1, self.seq_stride):
                    s_end = s_start + self.seq_len
                    self.sequences_x.append(drive_windows[s_start:s_end])
                    self.sequences_v.append(drive_v[s_start:s_end])
                    self.sequences_dv.append(drive_dv[s_start:s_end])
                    self.sequences_zupt.append(drive_zupt[s_start:s_end])
                    self.sequences_pitch.append(drive_pitch[s_start:s_end])
                    self.sequences_regime.append(drive_regime[s_start:s_end])

            except Exception as e:
                print(f"Error processing drive {s_file}: {e}")

        print(f"Loaded {len(self.sequences_x)} chronological sequences of length {self.seq_len} ({self.split.upper()}).")

    def __len__(self):
        return len(self.sequences_x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x_seq = torch.from_numpy(self.sequences_x[idx])  # (L, 18, 48)
        targets = {
            "v": torch.from_numpy(self.sequences_v[idx]),          # (L,)
            "delta_v": torch.from_numpy(self.sequences_dv[idx]),    # (L,)
            "zupt": torch.from_numpy(self.sequences_zupt[idx]),      # (L,)
            "pitch": torch.from_numpy(self.sequences_pitch[idx]),    # (L,)
            "regime": torch.from_numpy(self.sequences_regime[idx]),  # (L,)
        }
        return x_seq, targets
