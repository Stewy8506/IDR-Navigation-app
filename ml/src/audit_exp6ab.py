"""
Comparison & Audit Script for Experiment 6A vs Experiment 6A-B
"""
import os
import json
import numpy as np

def run_comparison():
    exp6a_hist_path = "ml/weights/exp6a_history.json"
    exp6ab_hist_path = "ml/weights/exp6ab_run/exp6ab_history.json"
    if not os.path.exists(exp6ab_hist_path):
        exp6ab_hist_path = "ml/weights/exp6ab_history.json"

    if not os.path.exists(exp6a_hist_path) or not os.path.exists(exp6ab_hist_path):
        print(f"Waiting for files: exp6a={os.path.exists(exp6a_hist_path)}, exp6ab={os.path.exists(exp6ab_hist_path)}")
        return

    with open(exp6a_hist_path) as f:
        h6a = json.load(f)
    with open(exp6ab_hist_path) as f:
        h6ab = json.load(f)

    print("=" * 80)
    print("      EXPERIMENT 6A vs EXPERIMENT 6A-B: COMPLETE COMPARISON AUDIT")
    print("=" * 80)

    # 1. Best Balanced Checkpoint comparison
    best_bal_6a_ep = h6a["best_epoch"]
    best_bal_6a = [e for e in h6a["history"] if e["epoch"] == best_bal_6a_ep][0]

    best_bal_6ab_ep = h6ab["best_balanced_epoch"]
    best_bal_6ab = [e for e in h6ab["history"] if e["epoch"] == best_bal_6ab_ep][0]

    # Best Raw Checkpoint
    best_raw_6a = min(h6a["history"], key=lambda x: x["val_mae"])
    best_raw_6ab = min(h6ab["history"], key=lambda x: x["val_mae"])

    print(f"\n--- 1. OVERALL HEAD-TO-HEAD SUMMARY ---")
    print(f"{'Metric':<30} | {'Exp6A (Baseline)':<20} | {'Exp6A-B (Speed-Bal)':<20} | {'Delta':<15}")
    print("-" * 90)

    def print_row(name, v6a, v6ab, unit="", fmt=".2f", lower_is_better=True):
        d = v6ab - v6a
        better = (d < 0) if lower_is_better else (d > 0)
        sign = "+" if d > 0 else ""
        flag = " [IMPROVED]" if better else (" [REGRESSED]" if d != 0 else "")
        v6a_s = f"{v6a:{fmt}} {unit}".strip()
        v6ab_s = f"{v6ab:{fmt}} {unit}".strip()
        d_s = f"{sign}{d:{fmt}} {unit}{flag}".strip()
        print(f"{name:<30} | {v6a_s:<20} | {v6ab_s:<20} | {d_s:<15}")

    print_row("Best Balanced Val MAE", best_bal_6a["balanced_val_mae"], best_bal_6ab["balanced_val_mae"], "km/h")
    print_row("Best Raw Val MAE", best_raw_6a["val_mae"], best_raw_6ab["val_mae"], "km/h")
    print_row("Best Pearson r", max(e["val_r"] for e in h6a["history"]), max(e["val_r"] for e in h6ab["history"]), "", fmt=".3f", lower_is_better=False)
    
    if "regression_slope" in best_bal_6ab:
        print_row("Regression Slope (m)", 0.5675, best_bal_6ab["regression_slope"], "", fmt=".4f", lower_is_better=False)
        print_row("Regression Intercept (c)", 14.96, best_bal_6ab["regression_intercept"], "km/h", fmt=".2f", lower_is_better=True)
        print_row("Std Ratio (Pred/GT)", 0.830, best_bal_6ab["compression_ratio"], "", fmt=".3f", lower_is_better=False)

    print(f"\n--- 2. PER-BIN MAE COMPARISON (AT BEST BALANCED CHECKPOINT) ---")
    print(f"{'Speed Bin (km/h)':<20} | {'Exp6A MAE':<15} | {'Exp6A-B MAE':<15} | {'Delta':<15}")
    print("-" * 70)
    for bn in ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-80", "80+"]:
        m6a = best_bal_6a["bin_maes"].get(bn, 0.0)
        m6ab = best_bal_6ab["bin_maes"].get(bn, 0.0)
        d = m6ab - m6a
        flag = " [IMPROVED]" if d < 0 else " [REGRESSED]"
        print(f"{bn:<20} | {m6a:<15.2f} | {m6ab:<15.2f} | {d:+6.2f} km/h{flag}")

    if "bin_signed_errors" in best_bal_6ab:
        print(f"\n--- 3. PER-BIN SIGNED ERROR (BIAS) COMPARISON ---")
        print(f"{'Speed Bin (km/h)':<20} | {'Exp6A Signed':<15} | {'Exp6A-B Signed':<15}")
        print("-" * 55)
        # Exp6A Epoch 7 signed errors from audit:
        exp6a_signed = {
            "0-10": +14.23,
            "10-20": +10.93,
            "20-30": +4.62,
            "30-40": -3.69,
            "40-50": -4.86,
            "50-60": -5.45,
            "60-80": -11.99,
            "80+": -25.42
        }
        for bn in ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-80", "80+"]:
            s6a = exp6a_signed.get(bn, 0.0)
            s6ab = best_bal_6ab["bin_signed_errors"].get(bn, 0.0)
            print(f"{bn:<20} | {s6a:+15.2f} | {s6ab:+15.2f}")

    print(f"\n--- 4. FULL 15-EPOCH PROGRESSION ---")
    print(f"{'Epoch':<6} | {'6A Trn':<8} | {'6AB Trn':<8} | {'6A Val':<8} | {'6AB Val':<8} | {'6A Bal':<8} | {'6AB Bal':<8} | {'6A r':<6} | {'6AB r':<6} | {'6AB Slope':<9} | {'6AB c':<8}")
    print("-" * 95)
    for e6a, e6ab in zip(h6a["history"], h6ab["history"]):
        ep = e6a["epoch"]
        sl = e6ab.get("regression_slope", 0.0)
        ic = e6ab.get("regression_intercept", 0.0)
        print(f"{ep:02d}     | {e6a['train_mae']:<8.2f} | {e6ab['train_mae']:<8.2f} | {e6a['val_mae']:<8.2f} | {e6ab['val_mae']:<8.2f} | {e6a['balanced_val_mae']:<8.2f} | {e6ab['balanced_val_mae']:<8.2f} | {e6a['val_r']:<6.3f} | {e6ab['val_r']:<6.3f} | {sl:<9.4f} | {ic:<+8.2f}")

if __name__ == "__main__":
    run_comparison()
