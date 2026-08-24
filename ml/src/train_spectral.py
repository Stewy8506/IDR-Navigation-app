"""
train_spectral.py - Multi-Task Physics-Guided Training for DeepSpeedKinematicsNet.
Optimizes:
  L_total = L_Huber(v) + 0.15 * L_NLL(v, sigma^2) + 0.5 * L_L1(delta_v) + 0.2 * L_BCE(ZUPT) + 0.1 * L_pitch + 0.05 * L_regime
Reports speed-bin MAE, ZUPT F1 & motion FPR, and validation metrics per epoch.
"""

import argparse
import math
import os
import random
import time
from typing import Tuple, Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .model import DeepSpeedKinematicsNet
from .dataset_spectral import DeepPhysicsDataset


def calibrated_heteroscedastic_loss(
    mu: torch.Tensor, target: torch.Tensor, sigma_v: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Laplace Negative Log-Likelihood with explicit calibration penalty:
      L_nll = |target - mu| / sigma_v + ln(sigma_v)
      L_cal = (sigma_v - detach(|target - mu|))^2
    """
    abs_err = torch.abs(target - mu)
    l_nll = (abs_err / sigma_v + torch.log(sigma_v)).mean()
    l_cal = ((sigma_v - abs_err.detach()) ** 2).mean()
    return l_nll, l_cal


def train(
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 1e-3,
    in_channels: int = 18,
    window_size: int = 48,
    samples_per_bin: int = 15000,
):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"==========================================================================")
    print(f"   TRAINING DEEP SPEED KINEMATICS NET (CALIBRATED HETEROSCEDASTIC)")
    print(f"==========================================================================")
    print(f"Compute Device:      {device}")
    print(f"Epochs:              {epochs}")
    print(f"Batch Size:          {batch_size}")
    print(f"Learning Rate:       {lr}")
    print(f"Context Window:      {window_size} samples (4.8s at 10 Hz)")

    # Data Loaders with Drive-Aware & Driver-Aware Isolation
    train_ds = DeepPhysicsDataset(window_size=window_size, step_size=2, split="train", balance_speed_bins=True, samples_per_bin=samples_per_bin)
    val_ds = DeepPhysicsDataset(window_size=window_size, step_size=2, split="val", balance_speed_bins=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = DeepSpeedKinematicsNet(in_channels=in_channels, window_size=window_size).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters:    {param_count:,} parameters")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)

    loss_huber_v = nn.SmoothL1Loss(beta=1.0)
    loss_l1_dv = nn.L1Loss()
    loss_bce_zupt = nn.BCELoss()
    loss_huber_pitch = nn.SmoothL1Loss(beta=0.1)
    loss_ce_regime = nn.CrossEntropyLoss()

    os.makedirs("ml/weights", exist_ok=True)
    best_val_mae = float("inf")
    save_path = "ml/weights/best_spectral_speed_filter.pt"

    bins = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 80), (80, 200)]
    bin_names = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-80", "80+"]

    print(f"\nTraining DeepSpeedKinematicsNet on {len(train_ds)} samples, validating on {len(val_ds)} samples...\n")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_mae_v = 0.0
        batches = 0

        for x, targets in train_loader:
            x = x.to(device)  # (B, 18, 48)
            v_gt = targets["v"].to(device)
            dv_gt = targets["delta_v"].to(device)
            zupt_gt = targets["zupt"].to(device)
            pitch_gt = targets["pitch"].to(device)
            regime_gt = targets["regime"].to(device)

            optimizer.zero_grad()

            preds = model(x)
            mu_v = preds["mu_v"]
            sigma_v = preds["sigma_v"]
            delta_v = preds["delta_v"]
            p_zupt = preds["p_zupt"]
            pitch_pred = preds["pitch"]
            regime_logits = preds["regime_logits"]

            # Multi-Task Physics Losses with Calibrated Laplace NLL
            l_v_huber = loss_huber_v(mu_v, v_gt)
            l_v_nll, l_v_cal = calibrated_heteroscedastic_loss(mu_v, v_gt, sigma_v)
            l_dv = loss_l1_dv(delta_v, dv_gt)
            l_zupt = loss_bce_zupt(p_zupt, zupt_gt)
            l_pitch = loss_huber_pitch(pitch_pred, pitch_gt)
            l_regime = loss_ce_regime(regime_logits, regime_gt)

            loss = (
                l_v_huber
                + 0.15 * l_v_nll
                + 0.10 * l_v_cal
                + 0.50 * l_dv
                + 0.20 * l_zupt
                + 0.10 * l_pitch
                + 0.05 * l_regime
            )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_mae_v += torch.abs(mu_v - v_gt).mean().item()
            batches += 1

        scheduler.step()

        # Validation Evaluation
        model.eval()
        val_v_errors = []
        val_v_gt_kmh = []
        val_v_pred_kmh = []
        val_sigmas_kmh = []
        val_zupt_preds = []
        val_zupt_gt = []

        with torch.no_grad():
            for x, targets in val_loader:
                x = x.to(device)
                v_gt = targets["v"].to(device)
                zupt_gt = targets["zupt"].to(device)

                preds = model(x)
                mu_v = preds["mu_v"]
                sigma_v = preds["sigma_v"]
                p_zupt = preds["p_zupt"]

                err_kmh = (torch.abs(mu_v - v_gt) * 3.6).cpu().numpy()
                gt_kmh = (v_gt * 3.6).cpu().numpy()
                pred_kmh = (mu_v * 3.6).cpu().numpy()
                sig_kmh = (sigma_v * 3.6).cpu().numpy()

                val_v_errors.extend(err_kmh.tolist())
                val_v_gt_kmh.extend(gt_kmh.tolist())
                val_v_pred_kmh.extend(pred_kmh.tolist())
                val_sigmas_kmh.extend(sig_kmh.tolist())
                val_zupt_preds.extend(p_zupt.cpu().numpy().tolist())
                val_zupt_gt.extend(zupt_gt.cpu().numpy().tolist())

        val_v_errors = np.array(val_v_errors)
        val_v_gt_kmh = np.array(val_v_gt_kmh)
        val_v_pred_kmh = np.array(val_v_pred_kmh)
        val_sigmas_kmh = np.array(val_sigmas_kmh)
        val_zupt_preds = np.array(val_zupt_preds)
        val_zupt_gt = np.array(val_zupt_gt)

        avg_val_mae = np.mean(val_v_errors)
        avg_val_rmse = np.sqrt(np.mean(val_v_errors ** 2))
        val_bias = np.mean(val_v_pred_kmh - val_v_gt_kmh)
        val_corr = np.corrcoef(val_v_pred_kmh, val_v_gt_kmh)[0, 1] if len(val_v_gt_kmh) > 1 else 0.0
        cal_error = np.mean(np.abs(val_sigmas_kmh - val_v_errors))
        unc_corr = np.corrcoef(val_sigmas_kmh, val_v_errors)[0, 1] if len(val_sigmas_kmh) > 1 else 0.0

        # ZUPT Metrics
        zupt_binary_pred = (val_zupt_preds > 0.5).astype(float)
        tp = np.sum((zupt_binary_pred == 1.0) & (val_zupt_gt == 1.0))
        fp = np.sum((zupt_binary_pred == 1.0) & (val_zupt_gt == 0.0))
        fn = np.sum((zupt_binary_pred == 0.0) & (val_zupt_gt == 1.0))
        tn = np.sum((zupt_binary_pred == 0.0) & (val_zupt_gt == 0.0))

        zupt_prec = tp / (tp + fp + 1e-6)
        zupt_rec = tp / (tp + fn + 1e-6)
        zupt_f1 = 2 * (zupt_prec * zupt_rec) / (zupt_prec + zupt_rec + 1e-6)
        motion_mask = val_v_gt_kmh > 3.6  # Moving > 1 m/s
        motion_fpr = (np.sum(zupt_binary_pred[motion_mask] == 1.0) / (np.sum(motion_mask) + 1e-6)) * 100.0

        # Speed-bin MAE and mean predicted sigma
        bin_maes = []
        bin_sigmas = []
        for (b_low, b_high) in bins:
            mask = (val_v_gt_kmh >= b_low) & (val_v_gt_kmh < b_high)
            bin_mae = np.mean(val_v_errors[mask]) if np.sum(mask) > 0 else 0.0
            bin_sig = np.mean(val_sigmas_kmh[mask]) if np.sum(mask) > 0 else 0.0
            bin_maes.append(f"{bin_mae:.1f}")
            bin_sigmas.append(f"{bin_sig:.1f}")

        bin_report = " | ".join([f"{bn}:{bm}" for bn, bm in zip(bin_names, bin_maes)])
        sig_report = " | ".join([f"{bn}:{bs}" for bn, bs in zip(bin_names, bin_sigmas)])

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] - Train MAE: {total_mae_v/batches*3.6:.2f} km/h | "
            f"Val MAE: {avg_val_mae:.2f} km/h | RMSE: {avg_val_rmse:.2f} | Bias: {val_bias:+.2f} | r: {val_corr:.3f} | "
            f"Mean Sigma: {np.mean(val_sigmas_kmh):.2f} km/h | CalErr: {cal_error:.2f} | UncCorr: {unc_corr:.3f}"
        )
        print(f"   --> Speed-Bin MAE (km/h):   [ {bin_report} ]")
        print(f"   --> Speed-Bin Sigma (km/h): [ {sig_report} ]", flush=True)

        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            torch.save(model.state_dict(), save_path)
            print(f"  --> [SAVED BEST CHECKPOINT] Val MAE: {best_val_mae:.2f} km/h\n", flush=True)

    print(f"\n==========================================================================")
    print(f"   TRAINING COMPLETE: Best Val MAE = {best_val_mae:.2f} km/h")
    print(f"   Model Saved To: {save_path}")
    print(f"==========================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
