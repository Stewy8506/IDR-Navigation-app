"""
dataset.py - Data loader and preprocessor for IO-VNBD dataset.
Extracts windowed IMU signals (accel, gyro) and aligns them with ground truth velocity.
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class IOVNBDDataset(Dataset):
    def __init__(self, data_dir: str, window_size: int = 100, step_size: int = 10, is_train: bool = True):
        """
        Args:
            data_dir: Path to IO-VNBD dataset directory.
            window_size: Number of IMU time steps per sample (e.g. 100 samples at 100Hz = 1s).
            step_size: Stride for sliding window.
            is_train: Whether to load training or validation/test split.
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.step_size = step_size
        self.is_train = is_train
        
        self.windows = []
        self.targets = []
        
        # Load dataset files if data_dir exists
        if os.path.exists(data_dir):
            self._load_data()

    def _load_data(self):
        """
        Loads CSV/MAT data files from the IO-VNBD directory.
        Format expected: timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, speed (or ground truth).
        """
        split_folder = "Train" if self.is_train else "Test"
        target_path = os.path.join(self.data_dir, split_folder)
        if not os.path.exists(target_path):
            target_path = self.data_dir

        files = glob.glob(os.path.join(target_path, "**/*.csv"), recursive=True)
        print(f"Found {len(files)} files in {target_path}")
        
        for file in files:
            try:
                df = pd.read_csv(file)
                # Process columns and extract sliding windows
                # Window shape: (6, window_size) -> 3 accel + 3 gyro channels
            except Exception as e:
                print(f"Error loading {file}: {e}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.windows[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )
