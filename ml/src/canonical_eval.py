"""
canonical_eval.py — Canonical Versioned Evaluation for INSS Speed Estimation.

RULES:
  1. This is the ONLY evaluation function that may be used to report official metrics.
  2. Every call logs this file's SHA256 hash so results are traceable to exact eval code.
  3. v_anchor is always raw m/s — the model normalizes internally.
  4. Closed-loop: v_t feeds back as v_anchor for t+1, starting from v_anchor=0.
  5. No sequence boundary resets — the entire validation drive is one continuous rollout.

Version: 1.0.0
"""

import hashlib
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Self-hash for provenance tracking
# ---------------------------------------------------------------------------

def _get_eval_script_hash() -> str:
    """Return SHA256 of this file's contents."""
    this_file = os.path.abspath(__file__)
    with open(this_file, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


EVAL_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Core Evaluation
# ---------------------------------------------------------------------------

def evaluate_closed_loop(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    speed_bins: Optional[List[Tuple[float, float]]] = None,
    verbose: bool = True,
) -> Dict:
    """
    Canonical closed-loop rollout evaluation on a validation drive.

    The model receives one timestep at a time. The predicted velocity (mu_v)
    is fed back as v_anchor for the next timestep. No ground-truth leakage.

    Args:
        model: DeepSpeedKinematicsNet (or any model with same forward signature)
        val_loader: DataLoader yielding (x_seq, targets) with batch_size=1
                    x_seq shape: (1, L, C, W), targets["v"] shape: (1, L)
        device: torch device
        speed_bins: list of (low, high) in km/h for per-bin metrics
        verbose: whether to print results

    Returns:
        Dict with all metrics
    """
    if speed_bins is None:
        speed_bins = [(0, 10), (10, 20), (20, 30), (30, 40),
                      (40, 50), (50, 60), (60, 80), (80, 300)]

    bin_names = []
    for low, high in speed_bins:
        if high >= 300:
            bin_names.append(f"{low}+")
        else:
            bin_names.append(f"{low}-{high}")

    eval_hash = _get_eval_script_hash()

    model.eval()
    all_preds_kmh = []
    all_gts_kmh = []

    # Single continuous closed-loop rollout — v_anchor starts at 0 m/s
    v_state = torch.zeros(1, device=device, dtype=torch.float32)

    t_start = time.time()
    with torch.no_grad():
        for x_seq, targets in val_loader:
            x_seq = x_seq.to(device)
            v_gt_seq = targets["v"].to(device)  # (1, L) in m/s
            B, L, C, W = x_seq.shape

            for t in range(L):
                # v_anchor is raw m/s — model normalizes internally
                out = model(x_seq[:, t], v_anchor=v_state)
                mu_t = out["mu_v"]  # (1,) in m/s, already ReLU'd >= 0

                all_preds_kmh.append(mu_t.item() * 3.6)
                all_gts_kmh.append(v_gt_seq[0, t].item() * 3.6)

                # Closed-loop feedback
                v_state = mu_t.detach()

    eval_time = time.time() - t_start

    preds = np.array(all_preds_kmh)
    gts = np.array(all_gts_kmh)
    N = len(preds)

    # --- Global metrics ---
    errors = np.abs(preds - gts)
    signed_errors = preds - gts
    raw_mae = float(errors.mean())
    raw_bias = float(signed_errors.mean())

    sigma_pred = float(np.std(preds))
    sigma_gt = float(np.std(gts))
    sigma_ratio = sigma_pred / sigma_gt if sigma_gt > 1e-6 else 0.0

    if sigma_pred > 1e-6 and sigma_gt > 1e-6:
        pearson_r = float(np.corrcoef(preds, gts)[0, 1])
        slope, intercept = np.polyfit(gts, preds, 1)
    else:
        pearson_r = 0.0
        slope, intercept = 0.0, 0.0

    # --- Per-bin metrics ---
    bin_results = {}
    bin_maes_for_balanced = []
    for (low, high), bn in zip(speed_bins, bin_names):
        mask = (gts >= low) & (gts < high)
        n_bin = int(mask.sum())
        if n_bin > 0:
            b_mae = float(errors[mask].mean())
            b_bias = float(signed_errors[mask].mean())
        else:
            b_mae = 0.0
            b_bias = 0.0
        bin_results[bn] = {"n": n_bin, "mae": b_mae, "bias": b_bias}
        bin_maes_for_balanced.append(b_mae)

    balanced_mae = float(np.mean(bin_maes_for_balanced))

    results = {
        "eval_version": EVAL_VERSION,
        "eval_hash": eval_hash,
        "n_samples": N,
        "eval_time_s": eval_time,
        "raw_mae": raw_mae,
        "balanced_mae": balanced_mae,
        "raw_bias": raw_bias,
        "slope": float(slope),
        "intercept": float(intercept),
        "pearson_r": pearson_r,
        "r_squared": pearson_r ** 2,
        "sigma_pred": sigma_pred,
        "sigma_gt": sigma_gt,
        "sigma_ratio": sigma_ratio,
        "bins": bin_results,
    }

    if verbose:
        print(f"\n{'='*100}")
        print(f" CANONICAL EVALUATION v{EVAL_VERSION} (hash: {eval_hash})")
        print(f"{'='*100}")
        print(f"  N = {N:,} timesteps | Eval time: {eval_time:.1f}s")
        print(f"  Raw MAE       : {raw_mae:6.2f} km/h")
        print(f"  Balanced MAE  : {balanced_mae:6.2f} km/h")
        print(f"  Regression    : Pred = {slope:.4f} × GT {intercept:+.2f} km/h")
        print(f"  Pearson r     : {pearson_r:.4f} (r² = {pearson_r**2:.4f})")
        print(f"  σ(pred)       : {sigma_pred:.2f} km/h")
        print(f"  σ(GT)         : {sigma_gt:.2f} km/h")
        print(f"  σ ratio       : {sigma_ratio:.4f}")
        print(f"  Raw bias      : {raw_bias:+.2f} km/h")
        print()
        print(f"  {'Bin':<10} | {'N':>6} | {'MAE':>8} | {'Bias':>8}")
        print(f"  {'-'*10}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}")
        for bn in bin_names:
            br = bin_results[bn]
            print(f"  {bn:<10} | {br['n']:>6d} | {br['mae']:>7.2f}  | {br['bias']:>+7.2f}")
        print(f"{'='*100}\n")

    return results


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import glob
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    from ml.kaggle.standalone_exp6c_kaggle import (
        DeepSpeedKinematicsNet,
        SequencePhysicsDataset,
    )

    DATA_DIR = "ml/external/IO-VNBD_repo/Synchronised V abd S datasets/Categorised IOVNB Dataset"

    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "ml/weights/exp6c_best_spectral_speed_filter.pt"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Loading checkpoint: {ckpt_path}")
    print(f"Device: {device}")

    model = DeepSpeedKinematicsNet(in_channels=18).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    # Use seq_len=32, stride=32 to match training validation config
    val_ds = SequencePhysicsDataset(DATA_DIR, split="val", seq_len=32, seq_stride=32)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    results = evaluate_closed_loop(model, val_loader, device, verbose=True)
    print(f"Eval script hash: {results['eval_hash']}")
