"""
train_heading.py - Trains DeepHeadingObserverNet for Gyro Bias & Yaw Rate Estimation.
Optimizes Huber loss on true yaw rate innovation + Gaussian NLL for bias uncertainty.
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from .model import DeepHeadingObserverNet
from .dataset_spectral import align_imu_to_vehicle_frame


class HeadingDataset(Dataset):
    def __init__(self, data_dir: str, window_size: int = 48, step_size: int = 4, split: str = "train"):
        self.window_size = window_size
        self.step_size = step_size
        self.windows = []
        self.bias_targets = []
        self.dw_targets = []

        s_pattern = os.path.join(data_dir, "**", "S-*.csv")
        all_s_files = sorted(glob.glob(s_pattern, recursive=True))

        for sf in all_s_files:
            is_driver_e = ("Driver E" in sf) or ("Vw" in sf) or ("Vta" in sf) or ("Vtb" in sf) or ("Vf" in sf)
            is_val_s3a = "S3a" in sf

            if split == "train":
                if is_driver_e or is_val_s3a: continue
            elif split == "val":
                if not is_val_s3a: continue

            vf = os.path.join(os.path.dirname(sf), os.path.basename(sf).replace("S-", "V-"))
            if not os.path.exists(vf): continue

            try:
                df_s = pd.read_csv(sf, encoding="latin1")
                df_v = pd.read_csv(vf, encoding="latin1")
                df_s.columns = df_s.columns.str.strip()
                df_v.columns = df_v.columns.str.strip()

                ax = df_s["ACCELEROMETER X (m/s²)"].values.astype(np.float32)
                ay = df_s["ACCELEROMETER Y (m/s²)"].values.astype(np.float32)
                az = df_s["ACCELEROMETER Z (m/s²)"].values.astype(np.float32)
                gy = df_s["GYROSCOPE Yaw (rad/s)"].values.astype(np.float32)
                gp = df_s["GYROSCOPE Pitch (rad/s)"].values.astype(np.float32)
                gr = df_s["GYROSCOPE Roll (rad/s)"].values.astype(np.float32)

                raw_imu = np.stack([ax, ay, az, gy, gp, gr], axis=0)
                aligned_imu = align_imu_to_vehicle_frame(raw_imu)

                if "Yaw Rate (deg/sec)" in df_v.columns:
                    true_yaw_rate_rads = (df_v["Yaw Rate (deg/sec)"].values * np.pi / 180.0).astype(np.float32)
                else:
                    continue

                min_len = min(aligned_imu.shape[1], len(true_yaw_rate_rads))
                if min_len < self.window_size: continue

                aligned_imu = aligned_imu[:, :min_len]
                true_yaw_rate_rads = true_yaw_rate_rads[:min_len]

                for start_idx in range(0, min_len - self.window_size + 1, self.step_size):
                    end_idx = start_idx + self.window_size
                    w = aligned_imu[:, start_idx:end_idx]
                    
                    meas_wz = aligned_imu[3, end_idx - 1]
                    true_wz = true_yaw_rate_rads[end_idx - 1]
                    bias_gt = meas_wz - true_wz
                    dw_gt = true_wz

                    self.windows.append(w)
                    self.bias_targets.append(float(bias_gt))
                    self.dw_targets.append(float(dw_gt))
            except Exception as e:
                pass

        print(f"Loaded HeadingDataset [{split.upper()}]: {len(self.windows)} windows.")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.windows[idx]),
            torch.tensor(self.bias_targets[idx], dtype=torch.float32),
            torch.tensor(self.dw_targets[idx], dtype=torch.float32),
        )


def train_heading(epochs: int = 15, batch_size: int = 128, lr: float = 1e-3):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data_dir = "ml/external/IO-VNBD_repo/Synchronised V abd S datasets/Categorised IOVNB Dataset"

    train_ds = HeadingDataset(data_dir=data_dir, split="train", step_size=4)
    val_ds = HeadingDataset(data_dir=data_dir, split="val", step_size=4)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = DeepHeadingObserverNet(in_channels=6).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    loss_fn = nn.SmoothL1Loss(beta=0.01)
    save_path = "ml/weights/best_heading_observer.pt"
    os.makedirs("ml/weights", exist_ok=True)
    best_val_loss = float("inf")

    print(f"\nTraining DeepHeadingObserverNet on {len(train_ds)} samples, validating on {len(val_ds)} samples...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0

        for x, bias_gt, dw_gt in train_loader:
            x = x.to(device)
            bias_gt = bias_gt.to(device)
            dw_gt = dw_gt.to(device)

            optimizer.zero_grad()
            out = model(x)
            bias_pred = out["gyro_bias_z"]
            dw_pred = out["delta_wz"]

            l_bias = loss_fn(bias_pred, bias_gt)
            l_dw = loss_fn(dw_pred, dw_gt)
            loss = l_bias + l_dw

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1

        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for x, bias_gt, dw_gt in val_loader:
                x = x.to(device)
                bias_gt = bias_gt.to(device)
                dw_gt = dw_gt.to(device)
                out = model(x)
                l = loss_fn(out["gyro_bias_z"], bias_gt) + loss_fn(out["delta_wz"], dw_gt)
                val_loss += l.item()
                val_batches += 1

        avg_val_loss = val_loss / max(1, val_batches)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {total_loss/batches:.5f} | Val Loss: {avg_val_loss:.5f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best heading observer to {save_path}")


if __name__ == "__main__":
    train_heading(epochs=15, batch_size=128)
