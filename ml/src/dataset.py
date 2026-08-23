r"""
dataset.py - IO-VNBD PyTorch Dataset loader with 10 Physics Features + Mounting Invariance Augmentation.
Applies random 3D rotation jitter (Euler angles +/- 15 deg) and sensor bias/noise injection during training.
"""

import glob
import os
import math
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _get_random_rotation_matrix(max_angle_deg: float = 15.0) -> np.ndarray:
    """Generates a random 3D rotation matrix within +/- max_angle_deg."""
    angles = np.radians(np.random.uniform(-max_angle_deg, max_angle_deg, size=3))
    roll, pitch, yaw = angles[0], angles[1], angles[2]

    # Rx, Ry, Rz
    rx = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
    ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
    rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])

    return rz @ ry @ rx


class IOVNBDDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "ml/external/IO-VNBD_repo/Synchronised V abd S datasets/Categorised IOVNB Dataset",
        window_size: int = 20,
        step_size: int = 2,
        is_train: bool = True,
        val_split: bool = False,
        augment: bool = True,
    ):
        self.data_dir = data_dir
        self.window_size = window_size
        self.step_size = step_size
        self.is_train = is_train
        self.val_split = val_split
        self.augment = augment and is_train and not val_split

        # Store raw 6 IMU channels per window for on-the-fly 3D rotation augmentation
        self.raw_windows: List[np.ndarray] = []  # shape (6, window_size)
        self.targets: List[float] = []

        if os.path.exists(data_dir):
            self._load_and_process_dataset()

    def _extract_physics_features(self, ax, ay, az, gy, gp, gr, dt=0.1) -> np.ndarray:
        """Extracts 10 physics-informed channels from IMU signals (shape: 10, W)."""
        # Dynamic accel norm (gravity removed)
        a_norm = np.sqrt(ax**2 + ay**2 + az**2) - 9.80665
        # Gyro total angular rate
        w_norm = np.sqrt(gy**2 + gp**2 + gr**2)

        # Leaky forward velocity integral
        vel_integral = np.zeros_like(ay)
        acc = 0.0
        for i in range(len(ay)):
            acc = acc * 0.95 + ay[i] * dt
            vel_integral[i] = acc

        # Vertical vibration moving variance
        az_series = pd.Series(az)
        az_var = az_series.rolling(window=5, min_periods=1).var().fillna(0.0).values

        return np.stack([ax, ay, az, gy, gp, gr, a_norm, w_norm, vel_integral, az_var], axis=0)

    def _load_and_process_dataset(self):
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
                    w = raw_imu[:, start_idx:end_idx]
                    target = speed_mps[end_idx - 1]

                    if not np.isnan(w).any() and not np.isnan(target):
                        self.raw_windows.append(w.astype(np.float32))
                        self.targets.append(float(target))

            except Exception as e:
                print(f"Error reading {s_file}: {e}")

        print(f"Loaded {len(self.raw_windows)} raw windows (augment={self.augment}).")

    def __len__(self):
        return len(self.raw_windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        w = self.raw_windows[idx].copy()  # shape (6, window_size)

        ax, ay, az = w[0], w[1], w[2]
        gy, gp, gr = w[3], w[4], w[5]

        # Apply 3D Mounting Rotation Jitter and Sensor Noise Augmentation
        if self.augment and np.random.rand() > 0.3:
            rot = _get_random_rotation_matrix(max_angle_deg=12.0)
            accel_3d = np.stack([ax, ay, az], axis=0)  # (3, W)
            gyro_3d = np.stack([gy, gp, gr], axis=0)   # (3, W)

            # Rotate vectors
            accel_rot = rot @ accel_3d
            gyro_rot = rot @ gyro_3d

            # Add small sensor bias jitter
            accel_bias = np.random.uniform(-0.1, 0.1, size=(3, 1))
            gyro_bias = np.random.uniform(-0.02, 0.02, size=(3, 1))

            accel_rot += accel_bias
            gyro_rot += gyro_bias

            ax, ay, az = accel_rot[0], accel_rot[1], accel_rot[2]
            gy, gp, gr = gyro_rot[0], gyro_rot[1], gyro_rot[2]

        # Extract 10 physics features from (augmented) raw signals
        features = self._extract_physics_features(ax, ay, az, gy, gp, gr)

        return (
            torch.from_numpy(features.astype(np.float32)),
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )
