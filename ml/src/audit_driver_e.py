"""
Audit script for Driver E data:
1. List all subcategories/drives of Driver E (Vta, Vtb, Vw, Vf).
2. Compute sample count, duration, min, max, mean, med, std, percentiles (P5, P25, P75, P95, P99) for each drive.
3. Compute speed bin distribution across all standard speed bins.
4. Compare training vs validation vs Driver E held-out test distribution.
"""

import os
import glob
import math
import numpy as np
import pandas as pd

DATA_DIR = "ml/external/IO-VNBD_repo/Synchronised V abd S datasets/Categorised IOVNB Dataset"

def audit_driver_e():
    s_files = sorted(glob.glob(os.path.join(DATA_DIR, "**", "S-*.csv"), recursive=True))
    
    driver_e_files = []
    other_train_files = []
    val_s3a_file = []

    for sf in s_files:
        is_e = ("Driver E" in sf) or ("Vw" in sf) or ("Vta" in sf) or ("Vtb" in sf) or ("Vf" in sf)
        if is_e:
            driver_e_files.append(sf)
        elif "S3a" in sf:
            val_s3a_file.append(sf)
        else:
            other_train_files.append(sf)

    print(f"Total S-*.csv files found: {len(s_files)}")
    print(f"  - Train Drives (Drivers A, B, C, D) : {len(other_train_files)}")
    print(f"  - Validation Drive (Driver A - S3a) : {len(val_s3a_file)}")
    print(f"  - Driver E Drives (Held-Out Test)   : {len(driver_e_files)}\n")

    bins = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 80), (80, 300)]
    bin_names = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-80", "80+"]

    def read_speeds(s_file_list):
        speeds = []
        drive_details = []
        for sf in s_file_list:
            vf = os.path.join(os.path.dirname(sf), os.path.basename(sf).replace("S-", "V-"))
            if not os.path.exists(vf):
                # Try lowercase
                vf_alt = os.path.join(os.path.dirname(sf), os.path.basename(sf).replace("S-", "V-").replace("Vta", "vta").replace("Vtb", "vtb").replace("Vw", "vw").replace("Vf", "vf"))
                if os.path.exists(vf_alt):
                    vf = vf_alt
                else:
                    continue
            try:
                df_v = pd.read_csv(vf, encoding="latin1")
                df_v.columns = df_v.columns.str.strip()
                col = "Indicated Vehicle Speed (km/hr)" if "Indicated Vehicle Speed (km/hr)" in df_v.columns else "Velocity (km/hr)"
                if col in df_v.columns:
                    v_kmh = df_v[col].values.astype(np.float64)
                    speeds.extend(v_kmh)
                    drive_details.append({
                        "file": os.path.relpath(sf, DATA_DIR),
                        "count": len(v_kmh),
                        "duration_s": len(v_kmh) * 0.1,
                        "min": np.min(v_kmh),
                        "max": np.max(v_kmh),
                        "mean": np.mean(v_kmh),
                        "med": np.median(v_kmh),
                        "std": np.std(v_kmh),
                        "p5": np.percentile(v_kmh, 5),
                        "p25": np.percentile(v_kmh, 25),
                        "p75": np.percentile(v_kmh, 75),
                        "p95": np.percentile(v_kmh, 95),
                        "p99": np.percentile(v_kmh, 99),
                        "speeds": v_kmh
                    })
            except Exception as e:
                print(f"Error reading {vf}: {e}")
        return np.array(speeds), drive_details

    train_speeds, train_drives = read_speeds(other_train_files)
    val_speeds, val_drives = read_speeds(val_s3a_file)
    driver_e_speeds, driver_e_drives = read_speeds(driver_e_files)

    # Benchmark test drives
    vw11_files = [f for f in driver_e_files if "Vw11" in f]
    vw12_files = [f for f in driver_e_files if "Vw12" in f]
    vw11_speeds, _ = read_speeds(vw11_files)
    vw12_speeds, _ = read_speeds(vw12_files)

    print("=" * 115)
    print("                DRIVER E DRIVES BREAKDOWN (INDIVIDUAL DRIVES)")
    print("=" * 115)
    print(f"{'Drive / Profile':<35} | {'Count':<7} | {'Dur (min)':<9} | {'Min':<5} | {'Max':<5} | {'Mean':<5} | {'Med':<5} | {'Std':<5} | {'P5':<5} | {'P95':<5} | {'P99':<5}")
    print("-" * 115)
    for d in driver_e_drives:
        dur_min = d['duration_s'] / 60.0
        print(f"{d['file']:<35} | {d['count']:<7d} | {dur_min:<9.1f} | {d['min']:<5.1f} | {d['max']:<5.1f} | {d['mean']:<5.1f} | {d['med']:<5.1f} | {d['std']:<5.1f} | {d['p5']:<5.1f} | {d['p95']:<5.1f} | {d['p99']:<5.1f}")

    print("\n" + "=" * 115)
    print("            SUMMARY: TRAIN vs VALIDATION vs DRIVER E (TEST) DISTRIBUTIONS")
    print("=" * 115)
    print(f"{'Dataset Split / Role':<35} | {'Count':<7} | {'Dur (min)':<9} | {'Min':<5} | {'Max':<5} | {'Mean':<5} | {'Med':<5} | {'Std':<5} | {'P5':<5} | {'P95':<5} | {'P99':<5}")
    print("-" * 115)

    def print_summary_row(name, sp):
        dur_min = (len(sp) * 0.1) / 60.0
        print(f"{name:<35} | {len(sp):<7d} | {dur_min:<9.1f} | {np.min(sp):<5.1f} | {np.max(sp):<5.1f} | {np.mean(sp):<5.1f} | {np.median(sp):<5.1f} | {np.std(sp):<5.1f} | {np.percentile(sp, 5):<5.1f} | {np.percentile(sp, 95):<5.1f} | {np.percentile(sp, 99):<5.1f}")

    print_summary_row("TRAIN (Drivers A, B, C, D - 7 files)", train_speeds)
    print_summary_row("VAL (Driver A - S3a)", val_speeds)
    print_summary_row("TEST: Driver E (Vw11 Benchmark)", vw11_speeds)
    print_summary_row("TEST: Driver E (Vw12 Benchmark)", vw12_speeds)
    print_summary_row("TEST: Driver E (ALL 25 Drives)", driver_e_speeds)

    print("\n" + "=" * 90)
    print("           SPEED-BIN SAMPLE COUNTS & PERCENTAGES ACROSS DATASET SPLITS")
    print("=" * 90)
    print(f"{'Speed Bin (km/h)':<18} | {'TRAIN % (N)':<20} | {'VAL S3a % (N)':<20} | {'Driver E Vw11 % (N)':<22} | {'Driver E All % (N)'}")
    print("-" * 90)

    for (bl, bh), bn in zip(bins, bin_names):
        t_c = int(np.sum((train_speeds >= bl) & (train_speeds < bh)))
        t_p = (t_c / len(train_speeds)) * 100.0
        v_c = int(np.sum((val_speeds >= bl) & (val_speeds < bh)))
        v_p = (v_c / len(val_speeds)) * 100.0
        vw11_c = int(np.sum((vw11_speeds >= bl) & (vw11_speeds < bh)))
        vw11_p = (vw11_c / len(vw11_speeds)) * 100.0
        e_c = int(np.sum((driver_e_speeds >= bl) & (driver_e_speeds < bh)))
        e_p = (e_c / len(driver_e_speeds)) * 100.0

        t_str = f"{t_p:5.1f}% ({t_c:<6d})"
        v_str = f"{v_p:5.1f}% ({v_c:<5d})"
        vw11_str = f"{vw11_p:5.1f}% ({vw11_c:<5d})"
        e_str = f"{e_p:5.1f}% ({e_c:<6d})"

        print(f"{bn:<18} | {t_str:<20} | {v_str:<20} | {vw11_str:<22} | {e_str}")

if __name__ == "__main__":
    audit_driver_e()
