"""
dataset_recurrent.py - Sequential Chunks Dataset with Prior GNSS Speed Conditioning & Stratified Motorway Slicing.
Extracts continuous sequences of length T with 16 multi-domain spectral features and prior speed v_prior.
Includes IMU noise jittering, speed prior perturbation, and balanced speed representation.
"""

import glob
import math
import os
import random
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .dataset_spectral import compute_spectral_physics_features, align_imu_to_vehicle_frame


class RecurrentIOVNBDDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "ml/external/IO-VNBD_repo/Synchronised V abd S datasets/Categorised IOVNB Dataset",
        window_size: int = 32,
        seq_len: int = 16,
        step_size: int = 4,
        is_train: bool = True,
        val_split: bool = False,
        augment: bool = True,
    ):
        self.data_dir = data_dir
        self.window_size = window_size
        self.seq_len = seq_len
        self.step_size = step_size
        self.is_train = is_train
        self.val_split = val_split
        self.augment = augment and is_train and not val_split

        # List of sequence chunks: (seq_len, 16, window_size) features, scalar v_prior, and (seq_len,) targets
        self.sequences: List[np.ndarray] = []
        self.priors: List[float] = []
        self.targets: List[np.ndarray] = []

        self._load_dataset()

    def _load_dataset(self):
        s_pattern = os.path.join(self.data_dir, "**", "S-*.csv")
        all_s_files = glob.glob(s_pattern, recursive=True)

        selected_files = []
        train_driver_counts = {"Driver A": 0, "Driver B": 0, "Driver C": 0, "Driver E": 0}
        val_driver_counts = {"Driver D": 0, "Driver E": 0}

        for sf in sorted(all_s_files):
            is_held_out = ("Driver D" in sf) or ("Vw11" in sf) or ("Vw12" in sf)
            if self.is_train and not self.val_split:
                if not is_held_out:
                    for d_key in train_driver_counts:
                        if d_key in sf and train_driver_counts[d_key] < 5:
                            selected_files.append((sf, "train"))
                            train_driver_counts[d_key] += 1
                            break
            elif self.val_split:
                if is_held_out:
                    if "Driver D" in sf and val_driver_counts["Driver D"] < 4:
                        selected_files.append((sf, "val"))
                        val_driver_counts["Driver D"] += 1
                    elif ("Vw11" in sf or "Vw12" in sf) and val_driver_counts["Driver E"] < 2:
                        selected_files.append((sf, "val"))
                        val_driver_counts["Driver E"] += 1

        print(f"Loading Recurrent Dataset (Train={self.is_train}, Val={self.val_split}) from {len(selected_files)} entries...")

        for s_file, split_mode in selected_files:
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
                if min_len < self.window_size + self.seq_len:
                    continue

                raw_imu = raw_imu[:, :min_len]
                aligned_imu = align_imu_to_vehicle_frame(raw_imu)
                speed_mps = speed_mps[:min_len]

                # Precompute all consecutive window features for this drive
                drive_windows = []
                drive_targets = []
                for start_idx in range(0, min_len - self.window_size + 1, 1):
                    end_idx = start_idx + self.window_size
                    w_aligned = aligned_imu[:, start_idx:end_idx]
                    target = speed_mps[end_idx - 1]

                    if not np.isnan(w_aligned).any() and not np.isnan(target):
                        feat16 = compute_spectral_physics_features(w_aligned)
                        drive_windows.append(feat16)
                        drive_targets.append(float(target))
                    else:
                        break

                drive_windows = np.array(drive_windows, dtype=np.float32)  # (N_windows, 16, W)
                drive_targets = np.array(drive_targets, dtype=np.float32)  # (N_windows,)

                # Slice drive into sequential chunks of length seq_len with prior speed v_prior
                num_windows = len(drive_windows)
                for seq_start in range(0, num_windows - self.seq_len + 1, self.step_size):
                    seq_end = seq_start + self.seq_len
                    v_prior_val = float(drive_targets[seq_start])  # True speed at blackout/sequence start

                    self.sequences.append(drive_windows[seq_start:seq_end])  # (Seq_Len, 16, W)
                    self.priors.append(v_prior_val)                          # scalar float
                    self.targets.append(drive_targets[seq_start:seq_end])    # (Seq_Len,)

            except Exception as e:
                print(f"Error loading {s_file}: {e}")

        print(f"Loaded {len(self.sequences)} sequential trajectory chunks (seq_len={self.seq_len}).")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_feat = self.sequences[idx].copy()  # (Seq_Len, 16, W)
        v_prior = float(self.priors[idx])
        seq_target = self.targets[idx].copy()  # (Seq_Len,)

        # Data Augmentation to prevent overfitting
        if self.augment:
            # 1. Subtle Gaussian sensor jitter on raw channels (channels 0..5)
            noise = np.random.normal(0.0, 0.02, size=seq_feat[:, :6, :].shape).astype(np.float32)
            seq_feat[:, :6, :] += noise

            # 2. Random scaling factor (0.96 to 1.04) representing chassis stiffness variation
            scale = np.random.uniform(0.96, 1.04)
            seq_feat[:, 10:, :] *= scale

            # 3. GNSS Prior noise jittering (±0.4 m/s uncertainty in last known fix)
            v_prior += float(np.random.normal(0.0, 0.4))
            v_prior = max(0.0, v_prior)

        v_prior_tensor = torch.tensor([v_prior], dtype=torch.float32)

        return (
            torch.from_numpy(seq_feat),
            v_prior_tensor,
            torch.from_numpy(seq_target),
        )
