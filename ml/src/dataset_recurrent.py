"""Dataset for the sequential NO-PRIOR recurrent speed estimator."""

import glob
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .dataset_spectral import (
    align_imu_to_vehicle_frame,
    compute_spectral_physics_features,
)


SPEED_BINS_KMH = (
    (0.0, 10.0),
    (10.0, 30.0),
    (30.0, 50.0),
    (50.0, 70.0),
    (70.0, 90.0),
    (90.0, 140.0),
)


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

        self.sequences: List[np.ndarray] = []
        self.priors: List[float] = []
        self.targets: List[np.ndarray] = []
        self.sequence_bin_indices: List[int] = []
        self._load_dataset()

    @staticmethod
    def _driver_name(path: str) -> str:
        normalized = path.replace("\\", "/")
        for driver in ("Driver A", "Driver B", "Driver C", "Driver D", "Driver E"):
            if driver in normalized:
                return driver
        return ""

    def _selected_files(self) -> List[str]:
        all_s_files = sorted(
            glob.glob(os.path.join(self.data_dir, "**", "S-*.csv"), recursive=True)
        )
        if not all_s_files:
            raise FileNotFoundError(
                f"No sensor files found under '{self.data_dir}' matching **/S-*.csv."
            )

        if self.is_train and not self.val_split:
            # Driver E is held out completely; all available A/B/C files are used.
            return [
                path for path in all_s_files
                if self._driver_name(path) in {"Driver A", "Driver B", "Driver C"}
            ]
        if self.val_split:
            # Validation is exclusively Driver D and is never balanced.
            return [path for path in all_s_files if self._driver_name(path) == "Driver D"]
        return [path for path in all_s_files if self._driver_name(path) == "Driver E"]

    def _load_dataset(self):
        selected_files = self._selected_files()
        split_name = "train A/B/C" if self.is_train and not self.val_split else "validation D"
        print(f"Loading recurrent {split_name} dataset from {len(selected_files)} files...")

        for s_file in selected_files:
            v_file = os.path.join(
                os.path.dirname(s_file),
                os.path.basename(s_file).replace("S-", "V-"),
            )
            if not os.path.exists(v_file):
                continue

            try:
                df_s = pd.read_csv(s_file, encoding="latin1")
                df_v = pd.read_csv(v_file, encoding="latin1")
                df_s.columns = df_s.columns.astype(str).str.strip()
                df_v.columns = df_v.columns.astype(str).str.strip()

                # Keep the established six-channel ordering and feature extractor.
                raw_imu = np.stack(
                    [
                        df_s["ACCELEROMETER X (m/s²)"].to_numpy(),
                        df_s["ACCELEROMETER Y (m/s²)"].to_numpy(),
                        df_s["ACCELEROMETER Z (m/s²)"].to_numpy(),
                        df_s["GYROSCOPE Yaw (rad/s)"].to_numpy(),
                        df_s["GYROSCOPE Pitch (rad/s)"].to_numpy(),
                        df_s["GYROSCOPE Roll (rad/s)"].to_numpy(),
                    ],
                    axis=0,
                )
                if "Indicated Vehicle Speed (km/hr)" in df_v.columns:
                    speed_kmh = df_v["Indicated Vehicle Speed (km/hr)"].to_numpy()
                elif "Velocity (km/hr)" in df_v.columns:
                    speed_kmh = df_v["Velocity (km/hr)"].to_numpy()
                else:
                    continue

                min_len = min(raw_imu.shape[1], len(speed_kmh))
                if min_len < self.window_size + self.seq_len:
                    continue
                raw_imu = raw_imu[:, :min_len]
                speed_mps = np.asarray(speed_kmh[:min_len], dtype=np.float32) / 3.6
                aligned_imu = align_imu_to_vehicle_frame(raw_imu)

                drive_windows = []
                drive_targets = []
                for start_idx in range(0, min_len - self.window_size + 1):
                    end_idx = start_idx + self.window_size
                    window = aligned_imu[:, start_idx:end_idx]
                    target = speed_mps[end_idx - 1]
                    if np.isfinite(window).all() and np.isfinite(target):
                        features = compute_spectral_physics_features(window)
                        drive_windows.append(features)
                        drive_targets.append(float(target))

                drive_windows = np.asarray(drive_windows, dtype=np.float32)
                drive_targets = np.asarray(drive_targets, dtype=np.float32)
                if len(drive_windows) < self.seq_len:
                    continue

                # Each item is 16 temporal feature windows; each target is the speed
                # at the end of its corresponding 32-sample source window.
                for seq_start in range(
                    0, len(drive_windows) - self.seq_len + 1, self.step_size
                ):
                    seq_end = seq_start + self.seq_len
                    sequence_target = drive_targets[seq_start:seq_end]
                    self.sequences.append(drive_windows[seq_start:seq_end])
                    self.priors.append(float(sequence_target[0]))
                    self.targets.append(sequence_target)
                    self.sequence_bin_indices.append(self.speed_bin_index(sequence_target))
            except Exception as exc:
                print(f"Error loading {s_file}: {exc}")

        print(f"Loaded {len(self.sequences)} sequences (seq_len={self.seq_len}).")

    @staticmethod
    def speed_bin_index(target_sequence: np.ndarray) -> int:
        mean_kmh = float(np.mean(target_sequence)) * 3.6
        for index, (low, high) in enumerate(SPEED_BINS_KMH):
            if low <= mean_kmh < high:
                return index
        return -1

    def speed_bin_counts(self) -> List[int]:
        counts = [0] * len(SPEED_BINS_KMH)
        for bin_index in self.sequence_bin_indices:
            if bin_index >= 0:
                counts[bin_index] += 1
        return counts

    def sequence_sampling_weights(self) -> torch.DoubleTensor:
        counts = self.speed_bin_counts()
        weights = [
            1.0 / counts[bin_index] if 0 <= bin_index < len(counts) else 0.0
            for bin_index in self.sequence_bin_indices
        ]
        return torch.as_tensor(weights, dtype=torch.double)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_feat = self.sequences[idx].copy()
        v_prior = float(self.priors[idx])
        seq_target = self.targets[idx].copy()
        if self.augment:
            noise = np.random.normal(
                0.0, 0.02, size=seq_feat[:, :6, :].shape
            ).astype(np.float32)
            seq_feat[:, :6, :] += noise
            scale = np.random.uniform(0.96, 1.04)
            seq_feat[:, 10:, :] *= scale
            v_prior = max(0.0, v_prior + float(np.random.normal(0.0, 0.4)))

        return (
            torch.from_numpy(seq_feat.astype(np.float32)),
            torch.tensor([v_prior], dtype=torch.float32),
            torch.from_numpy(seq_target.astype(np.float32)),
        )
