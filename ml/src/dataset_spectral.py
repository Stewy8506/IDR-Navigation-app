"""
dataset_spectral.py - IO-VNBD PyTorch Dataset with Spectral FFT/PSD + Time-Domain Physics Features.
Extracts frequency-domain wheel/road harmonics, spectral centroids, sub-band energies, and time-domain dynamics.
"""

import glob
import math
import os
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def compute_spectral_physics_features(w_6ch: np.ndarray, fs: float = 10.0) -> np.ndarray:
    """
    Given a (6, W) window of IMU signals [ax, ay, az, gy, gp, gr],
    computes a multi-domain representation combining:
    1. Time-domain physics signals (ax, ay, az, gy, gp, gr, a_norm, w_norm, vel_integral, az_var)
    2. FFT Power Spectral Density (PSD) magnitudes across frequency bins
    3. Sub-band spectral energy (Low: 0.3-1.2Hz, Mid: 1.2-2.5Hz, High: 2.5-5.0Hz)
    4. Spectral centroid (wheel/engine frequency shift with speed)
    """
    W = w_6ch.shape[1]
    ax, ay, az = w_6ch[0], w_6ch[1], w_6ch[2]
    gy, gp, gr = w_6ch[3], w_6ch[4], w_6ch[5]
    dt = 1.0 / fs

    # 1. Time-Domain Physics Features (10 channels)
    a_norm = np.sqrt(ax**2 + ay**2 + az**2) - 9.80665
    w_norm = np.sqrt(gy**2 + gp**2 + gr**2)

    vel_int = np.zeros(W, dtype=np.float32)
    acc = 0.0
    for i in range(W):
        acc = acc * 0.95 + ay[i] * dt
        vel_int[i] = acc

    az_series = pd.Series(az)
    az_var = az_series.rolling(window=5, min_periods=1).var().fillna(0.0).values.astype(np.float32)

    # 2. FFT Spectral Features per window
    # rfft yields (W//2 + 1) frequency bins
    freqs = np.fft.rfftfreq(W, d=dt)
    
    # Compute PSD for vertical accel az and forward accel ay
    az_fft = np.abs(np.fft.rfft(az - np.mean(az))) ** 2 / W
    ay_fft = np.abs(np.fft.rfft(ay - np.mean(ay))) ** 2 / W
    w_fft = np.abs(np.fft.rfft(w_norm - np.mean(w_norm))) ** 2 / W

    # Sub-band Energies
    low_mask = (freqs >= 0.3) & (freqs < 1.25)
    mid_mask = (freqs >= 1.25) & (freqs < 2.5)
    high_mask = (freqs >= 2.5) & (freqs <= 5.0)

    e_low = np.sum(az_fft[low_mask]) if np.sum(low_mask) > 0 else 0.0
    e_mid = np.sum(az_fft[mid_mask]) if np.sum(mid_mask) > 0 else 0.0
    e_high = np.sum(az_fft[high_mask]) if np.sum(high_mask) > 0 else 0.0

    # Spectral Centroid (mean frequency weighted by power)
    sum_power = np.sum(az_fft) + 1e-6
    spec_centroid = np.sum(freqs * az_fft) / sum_power
    spec_energy_ay = np.sum(ay_fft)

    # Broadcast scalar spectral descriptors across temporal length W
    e_low_ch = np.full(W, np.log1p(e_low), dtype=np.float32)
    e_mid_ch = np.full(W, np.log1p(e_mid), dtype=np.float32)
    e_high_ch = np.full(W, np.log1p(e_high), dtype=np.float32)
    spec_cent_ch = np.full(W, spec_centroid, dtype=np.float32)
    spec_power_ch = np.full(W, np.log1p(sum_power), dtype=np.float32)
    spec_ay_ch = np.full(W, np.log1p(spec_energy_ay), dtype=np.float32)

    # Stack: 10 Time-Domain Channels + 6 Spectral Energy Channels = 16 Channels
    features = np.stack([
        ax, ay, az, gy, gp, gr, a_norm, w_norm, vel_int, az_var,
        e_low_ch, e_mid_ch, e_high_ch, spec_cent_ch, spec_power_ch, spec_ay_ch
    ], axis=0)

    return features.astype(np.float32)


class SpectralIOVNBDDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "ml/external/IO-VNBD_repo/Synchronised V abd S datasets/Categorised IOVNB Dataset",
        window_size: int = 32,
        step_size: int = 2,
        is_train: bool = True,
        val_split: bool = False,
    ):
        self.data_dir = data_dir
        self.window_size = window_size
        self.step_size = step_size
        self.is_train = is_train
        self.val_split = val_split

        self.windows: List[np.ndarray] = []
        self.targets: List[float] = []

        self._load_dataset()

    def _load_dataset(self):
        s_pattern = os.path.join(self.data_dir, "**", "S-*.csv")
        all_s_files = glob.glob(s_pattern, recursive=True)

        selected_files = []
        for sf in all_s_files:
            is_driver_e = "Driver E" in sf
            if self.is_train and not self.val_split:
                if not is_driver_e:
                    selected_files.append(sf)
            elif self.val_split:
                if "Driver D" in sf:
                    selected_files.append(sf)
            else:
                if is_driver_e:
                    selected_files.append(sf)

        for s_file in selected_files:
            v_file = os.path.join(os.path.dirname(s_file), os.path.basename(s_file).replace("S-", "V-"))
            if not os.path.exists(v_file):
                continue

            try:
                df_s = pd.read_csv(s_file, encoding="latin1")
                df_v = pd.read_csv(v_file, encoding="latin1")
                df_s.columns = df_s.columns.str.strip()
                df_v.columns = df_v.columns.str.strip()

                ax = df_s["ACCELEROMETER X (m/s²)"].values
                ay = df_s["ACCELEROMETER Y (m/s²)"].values
                az = df_s["ACCELEROMETER Z (m/s²)"].values
                gy = df_s["GYROSCOPE Yaw (rad/s)"].values
                gp = df_s["GYROSCOPE Pitch (rad/s)"].values
                gr = df_s["GYROSCOPE Roll (rad/s)"].values

                raw_imu = np.stack([ax, ay, az, gy, gp, gr], axis=0)

                if "Indicated Vehicle Speed (km/hr)" in df_v.columns:
                    speed_kmh = df_v["Indicated Vehicle Speed (km/hr)"].values
                elif "Velocity (km/hr)" in df_v.columns:
                    speed_kmh = df_v["Velocity (km/hr)"].values
                else:
                    continue

                speed_mps = speed_kmh / 3.6
                min_len = min(raw_imu.shape[1], len(speed_mps))
                if min_len < self.window_size:
                    continue

                raw_imu = raw_imu[:, :min_len]
                speed_mps = speed_mps[:min_len]

                for start_idx in range(0, min_len - self.window_size + 1, self.step_size):
                    end_idx = start_idx + self.window_size
                    w_raw = raw_imu[:, start_idx:end_idx]
                    target = speed_mps[end_idx - 1]

                    if not np.isnan(w_raw).any() and not np.isnan(target):
                        feat16 = compute_spectral_physics_features(w_raw)
                        self.windows.append(feat16)
                        self.targets.append(float(target))

            except Exception as e:
                print(f"Error loading {s_file}: {e}")

        print(f"Loaded {len(self.windows)} 16-channel spectral windows.")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.windows[idx]),
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )
