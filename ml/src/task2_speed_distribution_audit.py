"""
task2_speed_distribution_audit.py - Comprehensive Speed Distribution Audit of the Training Dataset.
Examines Drivers A, B, and D across speed bins: 0-10, 10-30, 30-50, 50-70, 70-90, 90-140 km/h.
Calculates sample counts, percentage share, total drive time, and high-speed coverage.
"""

import glob
import os
import numpy as np
import pandas as pd


def main():
    data_dir = "ml/external/IO-VNBD_repo/Synchronised V abd S datasets/Categorised IOVNB Dataset"
    s_pattern = os.path.join(data_dir, "**", "S-*.csv")
    all_s_files = glob.glob(s_pattern, recursive=True)

    # Exclude Driver E (held-out test set)
    train_s_files = [f for f in all_s_files if "Driver E" not in f]
    print(f"Auditing training dataset across {len(train_s_files)} training files (Drivers A, B, D)...")

    all_speeds_kmh = []
    files_audited = 0

    for sf in train_s_files:
        vf = os.path.join(os.path.dirname(sf), os.path.basename(sf).replace("S-", "V-"))
        if not os.path.exists(vf):
            continue

        try:
            df_v = pd.read_csv(vf, encoding="latin1")
            df_v.columns = df_v.columns.str.strip()

            if "Indicated Vehicle Speed (km/hr)" in df_v.columns:
                speed = df_v["Indicated Vehicle Speed (km/hr)"].values
            elif "Velocity (km/hr)" in df_v.columns:
                speed = df_v["Velocity (km/hr)"].values
            else:
                continue

            # Filter NaNs
            speed = speed[~np.isnan(speed)]
            all_speeds_kmh.extend(speed.tolist())
            files_audited += 1
        except Exception as e:
            print(f"Error reading {vf}: {e}")

    all_speeds = np.array(all_speeds_kmh)
    N_total = len(all_speeds)
    total_time_hours = (N_total * 0.1) / 3600.0

    bins = [
        (0, 10, "Stationary / Creep"),
        (10, 30, "Low-Speed City"),
        (30, 50, "Urban Arterial"),
        (50, 70, "Suburban / Country"),
        (70, 90, "Fast A-Road"),
        (90, 140, "Motorway / Highway"),
    ]

    print("\n" + "=" * 85)
    print("           TASK 2: TRAINING SET SPEED DISTRIBUTION AUDIT (Drivers A, B, D)")
    print("=" * 85)
    print(f"Total Synchronized Files:   {files_audited} drives")
    print(f"Total 10Hz Samples:         {N_total:,} samples")
    print(f"Total Cumulative Drive Time:{total_time_hours:.2f} hours ({total_time_hours * 60:.1f} minutes)")
    print(f"Mean Training Speed:        {np.mean(all_speeds):.2f} km/h (std: {np.std(all_speeds):.2f} km/h)")
    print(f"Median Training Speed (P50):{np.percentile(all_speeds, 50):.2f} km/h")
    print(f"P90 Speed:                  {np.percentile(all_speeds, 90):.2f} km/h")
    print(f"P95 Speed:                  {np.percentile(all_speeds, 95):.2f} km/h")
    print(f"Max Recorded Speed:         {np.max(all_speeds):.2f} km/h")
    print("-" * 85)
    print(f"{'Speed Bin (km/h)':<20} | {'Regime Description':<20} | {'Sample Count':<12} | {'Time (min)':<10} | {'Percentage':<10}")
    print("-" * 85)

    high_speed_count = 0

    for b_low, b_high, label in bins:
        mask = (all_speeds >= b_low) & (all_speeds < b_high) if b_high < 140 else (all_speeds >= b_low)
        count = np.sum(mask)
        pct = (count / N_total) * 100.0
        time_min = (count * 0.1) / 60.0

        if b_low >= 70:
            high_speed_count += count

        print(f"{f'{b_low}-{b_high} km/h':<20} | {label:<20} | {count:<12,d} | {time_min:<10.1f} | {pct:<9.2f}%")

    pct_above_70 = (high_speed_count / N_total) * 100.0
    time_above_70 = (high_speed_count * 0.1) / 60.0

    print("-" * 85)
    print(f"TOTAL SUSTAINED HIGH-SPEED REGIME (≥ 70 km/h):")
    print(f"  Sample Count:             {high_speed_count:,} samples out of {N_total:,}")
    print(f"  Cumulative Time:          {time_above_70:.1f} minutes ({time_above_70/60.0:.2f} hours)")
    print(f"  Percentage of Training Set: {pct_above_70:.2f}%")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
