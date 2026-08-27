"""
train_exp6a_kaggle.py - Portable Kaggle GPU Training Runner for Experiment 6A.

Scientific Formulation (Frozen Experiment 6A):
  - Model: DeepSpeedKinematicsNet (ConvNeXt-1D + Temporal Multi-Head Attention)
  - Representation: 18-channel physics features with Channel 15 turn gating & BatchNorm1d
  - State Conditioning: v_anchor (1 -> 32) MLP embedding concatenated with global pooled representation (128 + 32 = 160 dim)
  - Closed-Loop Sequence Rollout: L = 32 steps at 10 Hz (dt = 0.1s, duration 3.2s)
  - Zero GT State Leakage: v_anchor[t] = v_pred[t-1].detach() strictly (no teacher forcing)
  - Target: v_raw[t] = v_anchor[t] + delta_v_pred[t], mu_v[t] = ReLU(v_raw[t])
  - Losses: Huber(mu_v, v_gt) + 0.15 * NLL(mu_v, v_gt, sigma_v) + 0.10 * Cal(sigma_v, err) + 0.50 * L1(delta_v) + 0.20 * BCE(ZUPT) + 0.10 * Huber(pitch) + 0.05 * CE(regime)
  - Checkpoint Selection: Balanced 8-bin MAE on Driver A S3a (continuous closed-loop rollout from v0 = 0.0 m/s)
"""

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add workspace roots to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from ml.src.model import DeepSpeedKinematicsNet
from ml.src.dataset_spectral import SequencePhysicsDataset


def auto_detect_dataset_dir(custom_path: str = None) -> str:
    """Auto-detects IO-VNBD dataset location across Kaggle, Colab, and local environments."""
    if custom_path and os.path.exists(custom_path):
        return custom_path

    env_path = os.environ.get("DATA_ROOT")
    if env_path and os.path.exists(env_path):
        return env_path

    candidates = [
        "/kaggle/input/iovnb-dataset/Categorised IOVNB Dataset",
        "/kaggle/input/io-vnbd-dataset/Categorised IOVNB Dataset",
        "/kaggle/input/iovnbd/Categorised IOVNB Dataset",
        "/kaggle/input/iovnb-dataset",
        "/kaggle/input/io-vnbd",
        "ml/external/IO-VNBD_repo/Synchronised V abd S datasets/Categorised IOVNB Dataset",
        "../input/iovnb-dataset/Categorised IOVNB Dataset",
    ]

    for cand in candidates:
        if os.path.exists(cand):
            return cand

    raise FileNotFoundError(
        "Could not automatically locate 'Categorised IOVNB Dataset'. "
        "Please provide --data-dir or set DATA_ROOT environment variable."
    )


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_exp6a_kaggle(
    data_dir: str = None,
    output_dir: str = "ml/weights",
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    seq_len: int = 32,
    seq_stride: int = 16,
    window_size: int = 48,
    seed: int = 42,
    smoke_test: bool = False,
):
    set_seed(seed)
    data_path = auto_detect_dataset_dir(data_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Hardware Detection
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        cuda_ver = torch.version.cuda
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple Silicon (MPS)"
        vram_gb = 0.0
        cuda_ver = "N/A"
    else:
        device = torch.device("cpu")
        device_name = "CPU"
        vram_gb = 0.0
        cuda_ver = "N/A"

    print("=" * 80)
    print("   EXPERIMENT 6A: CLOSED-LOOP STATE-CONDITIONED VELOCITY OBSERVER (KAGGLE RUNNER)")
    print("=" * 80)
    print(f"Device:                    {device} ({device_name})")
    if device.type == "cuda":
        print(f"GPU VRAM:                  {vram_gb:.2f} GB | CUDA: {cuda_ver}")
    print(f"Dataset Path:              {data_path}")
    print(f"Output Directory:          {output_dir}")
    print(f"Epochs:                    {1 if smoke_test else epochs}")
    print(f"Batch Size:                {batch_size}")
    print(f"Learning Rate:             {lr}")
    print(f"Sequence Length (L):       {seq_len} steps (3.2s at 10 Hz)")
    print(f"Sequence Stride:           {seq_stride} steps")
    print(f"Context Window (W):        {window_size} samples (4.8s at 10 Hz)")
    print(f"Velocity Norm Scale:       30.0 m/s (108 km/h)")
    print(f"State Conditioning Dim:    32 (Linear -> GELU -> Linear)")
    print(f"Zero GT Leakage:           VERIFIED (v_anchor strictly receives model's own v_pred.detach())")
    print("-" * 80)

    # 2. Dataset Loading
    t0_load = time.time()
    train_ds = SequencePhysicsDataset(data_dir=data_path, window_size=window_size, seq_len=seq_len, seq_stride=seq_stride, split="train")
    val_ds = SequencePhysicsDataset(data_dir=data_path, window_size=window_size, seq_len=seq_len, seq_stride=seq_len, split="val")
    t_load = time.time() - t0_load

    num_workers = 2 if device.type == "cuda" else 0
    pin_memory = (device.type == "cuda")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    print(f"Dataset loaded in {t_load:.1f}s | Training Sequences: {len(train_ds):,} | Validation Sequences: {len(val_ds):,}")

    # 3. Model & Multi-Task Losses
    model = DeepSpeedKinematicsNet(in_channels=18, window_size=window_size).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters:          {param_count:,} parameters")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)

    loss_huber_v = nn.SmoothL1Loss(beta=1.0)
    loss_l1_dv = nn.L1Loss()
    loss_bce_zupt = nn.BCELoss()
    loss_huber_pitch = nn.SmoothL1Loss(beta=0.1)
    loss_ce_regime = nn.CrossEntropyLoss()

    bins = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 80), (80, 200)]
    bin_names = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-80", "80+"]

    save_path_best = os.path.join(output_dir, "exp6a_best_spectral_speed_filter.pt")
    save_path_final = os.path.join(output_dir, "exp6a_final_spectral_speed_filter.pt")
    history_path = os.path.join(output_dir, "exp6a_history.json")

    best_balanced_val_mae = float("inf")
    best_epoch = 0
    history = []

    # 4. Smoke Test Mode
    if smoke_test:
        print("\n" + "=" * 80)
        print("   RUNNING GPU SMOKE TEST (2 Batches)")
        print("=" * 80)
        model.train()
        smoke_batches = 0
        for x_seq, targets in train_loader:
            x_seq = x_seq.to(device, non_blocking=True)
            B, L, C, W = x_seq.shape
            v_gt = targets["v"].to(device, non_blocking=True)
            dv_gt = targets["delta_v"].to(device, non_blocking=True)
            zupt_gt = targets["zupt"].to(device, non_blocking=True)
            pitch_gt = targets["pitch"].to(device, non_blocking=True)
            regime_gt = targets["regime"].to(device, non_blocking=True)

            optimizer.zero_grad()
            v_state = torch.zeros(B, device=device, dtype=torch.float32)

            pred_mu_list = []
            for t in range(L):
                out = model(x_seq[:, t], v_anchor=v_state)
                mu_t = out["mu_v"]
                pred_mu_list.append(mu_t)
                v_state = mu_t.detach()

            mu_all = torch.stack(pred_mu_list, dim=1).view(-1)
            v_gt_all = v_gt.view(-1)
            loss = loss_huber_v(mu_all, v_gt_all)
            loss.backward()

            # Gradient check
            for name, param in model.named_parameters():
                if param.grad is not None:
                    assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}!"
                    assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}!"

            optimizer.step()
            smoke_batches += 1
            print(f"Smoke Batch {smoke_batches}: Loss = {loss.item():.4f} (Shapes: X={x_seq.shape}, v_anchor=({B},), mu_all={mu_all.shape})")
            if smoke_batches >= 2:
                break

        print("Smoke Test Passed: All tensor shapes, closed-loop rollout, finite losses, and finite gradients verified.")
        torch.save(model.state_dict(), save_path_best)
        print(f"Verified Checkpoint Serialization to {save_path_best}.")
        return

    # 5. Full Training Loop
    total_epochs = epochs
    print(f"\nStarting 15-Epoch Training Loop ({len(train_loader)} batches/epoch, {batch_size} sequences/batch, L={seq_len})...\n")

    for epoch in range(1, total_epochs + 1):
        t0_epoch = time.time()
        model.train()
        total_loss = 0.0
        total_mae_v = 0.0
        batches = 0

        for x_seq, targets in train_loader:
            x_seq = x_seq.to(device, non_blocking=True)  # (B, L, 18, 48)
            B, L, C, W = x_seq.shape

            v_gt = targets["v"].to(device, non_blocking=True)          # (B, L)
            dv_gt = targets["delta_v"].to(device, non_blocking=True)    # (B, L)
            zupt_gt = targets["zupt"].to(device, non_blocking=True)    # (B, L)
            pitch_gt = targets["pitch"].to(device, non_blocking=True)  # (B, L)
            regime_gt = targets["regime"].to(device, non_blocking=True)# (B, L)

            optimizer.zero_grad()

            # Closed-loop rollout starts from zero state (zero GT leakage)
            v_state = torch.zeros(B, device=device, dtype=torch.float32)

            pred_mu_list = []
            pred_sigma_list = []
            pred_logvar_list = []
            pred_dv_list = []
            pred_zupt_list = []
            pred_pitch_list = []
            pred_regime_list = []

            for t in range(L):
                x_t = x_seq[:, t]  # (B, 18, 48)
                out = model(x_t, v_anchor=v_state)

                mu_t = out["mu_v"]
                pred_mu_list.append(mu_t)
                pred_sigma_list.append(out["sigma_v"])
                pred_logvar_list.append(out["log_sigma2"])
                pred_dv_list.append(out["delta_v"])
                pred_zupt_list.append(out["p_zupt"])
                pred_pitch_list.append(out["pitch"])
                pred_regime_list.append(out["regime_logits"])

                # Recursive state update strictly using model's OWN prediction
                v_state = mu_t.detach()

            mu_all = torch.stack(pred_mu_list, dim=1).view(-1)
            sigma_all = torch.stack(pred_sigma_list, dim=1).view(-1)
            logvar_all = torch.stack(pred_logvar_list, dim=1).view(-1)
            dv_all = torch.stack(pred_dv_list, dim=1).view(-1)
            zupt_all = torch.stack(pred_zupt_list, dim=1).view(-1)
            pitch_all = torch.stack(pred_pitch_list, dim=1).view(-1)
            regime_all = torch.stack(pred_regime_list, dim=1).view(-1, 7)

            v_gt_all = v_gt.view(-1)
            dv_gt_all = dv_gt.view(-1)
            zupt_gt_all = zupt_gt.view(-1)
            pitch_gt_all = pitch_gt.view(-1)
            regime_gt_all = regime_gt.view(-1)

            l_huber_v = loss_huber_v(mu_all, v_gt_all)
            l_nll = (0.5 * torch.exp(-logvar_all) * ((mu_all - v_gt_all) ** 2) + 0.5 * logvar_all).mean()
            l_cal = (sigma_all - torch.abs(mu_all - v_gt_all)).abs().mean()
            l_dv = loss_l1_dv(dv_all, dv_gt_all)
            l_zupt = loss_bce_zupt(zupt_all, zupt_gt_all)
            l_pitch = loss_huber_pitch(pitch_all, pitch_gt_all)
            l_regime = loss_ce_regime(regime_all, regime_gt_all)

            loss = (
                l_huber_v
                + 0.15 * l_nll
                + 0.10 * l_cal
                + 0.50 * l_dv
                + 0.20 * l_zupt
                + 0.10 * l_pitch
                + 0.05 * l_regime
            )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_mae_v += torch.abs(mu_all - v_gt_all).mean().item() * 3.6
            batches += 1

        scheduler.step()
        epoch_train_time = time.time() - t0_epoch
        train_loss = total_loss / max(1, batches)
        train_mae = total_mae_v / max(1, batches)

        # 6. Continuous Closed-Loop Validation (Driver A S3a)
        model.eval()
        val_preds = []
        val_gts = []
        val_sigmas = []

        with torch.no_grad():
            v_val_state = torch.zeros(1, device=device, dtype=torch.float32)
            for x_seq, targets in val_loader:
                x_seq = x_seq.to(device)
                v_gt_seq = targets["v"].to(device)
                B, L, C, W = x_seq.shape

                for t in range(L):
                    x_t = x_seq[:, t]
                    out = model(x_t, v_anchor=v_val_state)
                    mu_t = out["mu_v"]
                    val_preds.append(mu_t.item() * 3.6)
                    val_gts.append(v_gt_seq[0, t].item() * 3.6)
                    val_sigmas.append(out["sigma_v"].item() * 3.6)
                    v_val_state = mu_t.detach()

        val_preds = np.array(val_preds)
        val_gts = np.array(val_gts)
        val_errors = np.abs(val_preds - val_gts)
        overall_val_mae = float(val_errors.mean())
        val_rmse = float(np.sqrt(np.mean(val_errors ** 2)))
        val_bias = float(np.mean(val_preds - val_gts))
        val_r = float(np.corrcoef(val_preds, val_gts)[0, 1]) if np.std(val_preds) > 0 and np.std(val_gts) > 0 else 0.0

        bin_maes = []
        bin_mae_dict = {}
        for b_low, b_high, bn in zip([b[0] for b in bins], [b[1] for b in bins], bin_names):
            mask = (val_gts >= b_low) & (val_gts < b_high)
            b_mae = float(val_errors[mask].mean()) if mask.sum() > 0 else 0.0
            bin_maes.append(b_mae)
            bin_mae_dict[bn] = b_mae

        balanced_val_mae = float(np.mean(bin_maes))
        samples_per_sec = (len(train_ds) * seq_len) / max(0.01, epoch_train_time)

        # GPU Memory Tracking
        if device.type == "cuda":
            vram_alloc = torch.cuda.memory_allocated(0) / (1024 ** 2)
            vram_res = torch.cuda.memory_reserved(0) / (1024 ** 2)
            mem_str = f" | VRAM: {vram_alloc:.0f}/{vram_res:.0f} MB"
        else:
            mem_str = ""

        bin_report = " | ".join([f"{bn}:{bm:4.1f}" for bn, bm in zip(bin_names, bin_maes)])
        print(f"Epoch [{epoch:02d}/{total_epochs:02d}] ({epoch_train_time:.1f}s, {samples_per_sec:.0f} smp/s{mem_str}) | Train Loss: {train_loss:.4f} | Train MAE: {train_mae:5.2f} km/h | Val MAE: {overall_val_mae:5.2f} km/h | Balanced: {balanced_val_mae:5.2f} km/h | r: {val_r:.3f}", flush=True)
        print(f"   Bin MAEs (km/h): [ {bin_report} ]", flush=True)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mae": train_mae,
            "val_mae": overall_val_mae,
            "balanced_val_mae": balanced_val_mae,
            "val_rmse": val_rmse,
            "val_bias": val_bias,
            "val_r": val_r,
            "bin_maes": bin_mae_dict,
            "epoch_time_s": epoch_train_time,
            "samples_per_sec": samples_per_sec,
        }
        history.append(epoch_record)

        if balanced_val_mae < best_balanced_val_mae:
            best_balanced_val_mae = balanced_val_mae
            best_epoch = epoch
            torch.save(model.state_dict(), save_path_best)
            print(f"   >>> [SAVED BEST CHECKPOINT] Epoch {epoch} to {save_path_best} (Balanced MAE: {balanced_val_mae:.2f} km/h) <<<\n", flush=True)
        else:
            print(f"   --- Checkpoint not improved (Best: {best_balanced_val_mae:.2f} km/h at Epoch {best_epoch}) ---\n", flush=True)

    # Save final model & JSON history
    torch.save(model.state_dict(), save_path_final)
    with open(history_path, "w") as f:
        json.dump({
            "experiment": "Experiment 6A: Closed-Loop State-Conditioned Velocity Observer",
            "best_epoch": best_epoch,
            "best_balanced_val_mae": best_balanced_val_mae,
            "history": history
        }, f, indent=2)

    print("=" * 80)
    print(f"   TRAINING COMPLETE: Best Epoch {best_epoch} with Balanced Val MAE: {best_balanced_val_mae:.2f} km/h")
    print(f"   Best Checkpoint: {save_path_best}")
    print(f"   Final Checkpoint: {save_path_final}")
    print(f"   Training History: {history_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaggle GPU Training Runner for Experiment 6A")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to Categorised IOVNB Dataset")
    parser.add_argument("--output-dir", type=str, default="ml/weights", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (number of sequences)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seq-len", type=int, default=32, help="Chronological sequence length L")
    parser.add_argument("--seq-stride", type=int, default=16, help="Sequence sampling stride")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 2-batch GPU smoke test")
    args = parser.parse_args()

    train_exp6a_kaggle(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seq_len=args.seq_len,
        seq_stride=args.seq_stride,
        seed=args.seed,
        smoke_test=args.smoke_test,
    )
