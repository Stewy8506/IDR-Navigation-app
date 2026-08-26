"""Shared V5 data discovery, windowing, and training-only normalization."""

import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


IMU_COLUMNS = (
    "ACCELEROMETER X (m/s²)",
    "ACCELEROMETER Y (m/s²)",
    "ACCELEROMETER Z (m/s²)",
    "GYROSCOPE Yaw (rad/s)",
    "GYROSCOPE Pitch (rad/s)",
    "GYROSCOPE Roll (rad/s)",
)
TARGET_COLUMN = "Indicated Vehicle Speed (km/hr)"
WINDOW_SIZE = 64
STEP_SIZE = 2
SPEED_BINS_KMH = ((0, 10), (10, 30), (30, 50), (50, 70), (70, 90), (90, 140))


def _normalized_name(value: str) -> str:
    return os.path.basename(value).strip().lower()


def find_vehicle_file(s_file: str) -> Optional[str]:
    """Find the same-recording V-file, tolerating prefix and case differences."""
    folder = os.path.dirname(s_file)
    sensor_name = _normalized_name(s_file)
    sensor_stem = sensor_name[2:-4] if sensor_name.startswith("s-") else ""
    candidates = glob.glob(os.path.join(folder, "*.csv"))
    matching = []
    for candidate in candidates:
        name = _normalized_name(candidate)
        if not name.startswith("v-") or not name.endswith(".csv"):
            continue
        if name[2:-4] == sensor_stem:
            matching.append(candidate)

    for candidate in sorted(matching):
        try:
            columns = {str(column).strip().lower() for column in pd.read_csv(candidate, nrows=0, encoding="latin1").columns}
        except Exception:
            continue
        if TARGET_COLUMN.lower() in columns:
            return candidate
    return None


def discover_pairs(data_dir: str) -> Tuple[List[Tuple[str, str]], int, int]:
    s_files = sorted(glob.glob(os.path.join(data_dir, "**", "S-*.csv"), recursive=True))
    pairs = []
    skipped = 0
    for s_file in s_files:
        v_file = find_vehicle_file(s_file)
        if v_file is None:
            print(f"WARNING: skipped S-file without a valid matching V-file: {s_file}")
            skipped += 1
        else:
            pairs.append((s_file, v_file))
    return pairs, len(s_files), skipped


def driver_group(path: str) -> str:
    normalized = path.replace("\\", "/").lower()
    for letter in "abcde":
        if f"driver {letter}" in normalized:
            return letter.upper()
    return ""


def select_pairs(pairs: List[Tuple[str, str]], split: str) -> List[Tuple[str, str]]:
    if split == "train":
        return [(s, v) for s, v in pairs if driver_group(s) not in {"D", "E"}]
    if split == "validation":
        return [(s, v) for s, v in pairs if driver_group(s) == "D"]
    if split == "test":
        return [(s, v) for s, v in pairs if driver_group(s) == "E"]
    raise ValueError(f"Unknown split: {split}")


def _read_pair(s_file: str, v_file: str) -> Tuple[np.ndarray, np.ndarray]:
    df_s = pd.read_csv(s_file, encoding="latin1")
    df_v = pd.read_csv(v_file, encoding="latin1")
    df_s.columns = df_s.columns.astype(str).str.strip()
    df_v.columns = df_v.columns.astype(str).str.strip()
    missing = [column for column in IMU_COLUMNS if column not in df_s.columns]
    if missing:
        raise KeyError(f"Missing IMU columns: {missing}")
    if TARGET_COLUMN not in df_v.columns:
        raise KeyError(f"Missing V-file target column: {TARGET_COLUMN}")

    imu = np.stack(
        [pd.to_numeric(df_s[column], errors="coerce").to_numpy(dtype=np.float32) for column in IMU_COLUMNS],
        axis=0,
    )
    target = pd.to_numeric(df_v[TARGET_COLUMN], errors="coerce").to_numpy(dtype=np.float32)
    row_count = min(imu.shape[1], len(target))
    if imu.shape[1] != len(target):
        print(f"WARNING: row mismatch, truncating paired files to {row_count}: {s_file}")
    return imu[:, :row_count], target[:row_count]


def load_windows(pairs: List[Tuple[str, str]], window_size: int = WINDOW_SIZE, step_size: int = STEP_SIZE):
    windows: List[np.ndarray] = []
    targets: List[float] = []
    source_indices: List[int] = []
    for pair_index, (s_file, v_file) in enumerate(pairs):
        try:
            imu, target = _read_pair(s_file, v_file)
            for start in range(0, imu.shape[1] - window_size + 1, step_size):
                end = start + window_size
                window_target = target[end - 1]
                window = imu[:, start:end]
                if np.isfinite(window).all() and np.isfinite(window_target):
                    windows.append(window.astype(np.float32))
                    targets.append(float(window_target))
                    source_indices.append(pair_index)
        except Exception as exc:
            print(f"WARNING: skipped invalid pair {s_file}: {exc}")
    if not windows:
        return np.empty((0, 6, window_size), dtype=np.float32), np.empty(0, dtype=np.float32), source_indices
    return np.stack(windows).astype(np.float32), np.asarray(targets, dtype=np.float32), source_indices


def fit_normalization(windows: np.ndarray, targets_kmh: np.ndarray) -> Dict[str, np.ndarray]:
    if len(windows) == 0 or len(targets_kmh) == 0:
        raise ValueError("Cannot fit V5 normalization on an empty training distribution.")
    feature_mean = windows.mean(axis=(0, 2)).astype(np.float32)
    feature_std = windows.std(axis=(0, 2)).astype(np.float32)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std).astype(np.float32)
    target_mean = np.float32(targets_kmh.mean())
    target_std = np.float32(max(targets_kmh.std(), 1e-6))
    return {"feature_mean": feature_mean, "feature_std": feature_std, "target_mean": target_mean, "target_std": target_std}


def save_normalization(path: str, stats: Dict[str, np.ndarray]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, **stats)


def load_normalization(path: str) -> Dict[str, np.ndarray]:
    values = np.load(path)
    required = {"feature_mean", "feature_std", "target_mean", "target_std"}
    if not required.issubset(values.files):
        raise ValueError(f"V5 normalization file must contain {sorted(required)}")
    return {key: values[key].astype(np.float32) for key in required}


class V5SpeedDataset(Dataset):
    def __init__(self, windows: np.ndarray, targets_kmh: np.ndarray, stats: Dict[str, np.ndarray]):
        if len(windows) != len(targets_kmh) or len(windows) == 0:
            raise ValueError("V5 dataset requires a non-empty, aligned windows/targets array.")
        self.windows = ((windows - stats["feature_mean"][None, :, None]) / stats["feature_std"][None, :, None]).astype(np.float32)
        self.targets = ((targets_kmh - stats["target_mean"]) / stats["target_std"]).astype(np.float32)
        if not np.isfinite(self.windows).all() or not np.isfinite(self.targets).all():
            raise ValueError("NaN/Inf detected after V5 normalization.")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return torch.from_numpy(self.windows[index]), torch.tensor(self.targets[index], dtype=torch.float32)


def describe_targets(targets_kmh: np.ndarray) -> str:
    return f"min={targets_kmh.min():.2f}, max={targets_kmh.max():.2f}, mean={targets_kmh.mean():.2f}, std={targets_kmh.std():.2f} km/h"
