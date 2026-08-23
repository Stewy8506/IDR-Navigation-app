"""
task3_indistribution_evaluation.py - Full-Pipeline Drift Benchmark on In-Distribution Driving Profile (Driver A - S3a).
Tests if the Full Pipeline (Config c) and Dead Reckoning (Config d) outperform the baseline
when operating within the model's learned speed domain (Mean: 34.5 km/h, Max: 61.8 km/h).
"""

import math
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .model import SpeedVibrationFilterNet
from .dataset_spectral import compute_spectral_physics_features

EARTH_RADIUS = 6378137.0
GRAVITY = 9.80665


def geodetic_to_enu(lat, lon, alt, lat0, lon0, alt0):
    lat_rad = math.radians(lat)
    lat0_rad = math.radians(lat0)
    d_lat = math.radians(lat - lat0)
    d_lon = math.radians(lon - lon0)

    e = EARTH_RADIUS * d_lon * math.cos(lat0_rad)
    n = EARTH_RADIUS * d_lat
    u = alt - alt0
    return e, n, u


def run_raw_ins(ax_v, ay_v, az_v, gy_v, initial_theta_rad, dt=0.1):
    """Configuration (a): Raw Strapdown INS (Uncorrected Baseline)"""
    N = len(ax_v)
    pos_enu = np.zeros((N, 2))
    vel_enu = np.zeros((N, 2))
    theta_est = np.zeros(N)
    theta_est[0] = initial_theta_rad

    for k in range(1, N):
        theta = theta_est[k - 1] + gy_v[k] * dt
        theta_est[k] = theta

        c_th, s_th = math.cos(theta), math.sin(theta)

        a_e = ay_v[k] * c_th + ax_v[k] * s_th
        a_n = ay_v[k] * s_th - ax_v[k] * c_th

        vel_enu[k, 0] = vel_enu[k - 1, 0] + a_e * dt
        vel_enu[k, 1] = vel_enu[k - 1, 1] + a_n * dt

        pos_enu[k] = pos_enu[k - 1] + vel_enu[k] * dt

    return pos_enu, vel_enu


def run_ekf(
    ax_v,
    ay_v,
    az_v,
    gy_v,
    gnss_enu,
    gnss_flags,
    ai_speed=None,
    ai_var=None,
    outage_mask=None,
    initial_theta_rad=0.0,
    use_ai_speed=False,
    use_nhc=True,
    dt=0.1,
):
    """
    EKF with Math ENU Strapdown Mechanization, Non-Holonomic Constraints (NHC),
    GNSS position updates, and AI Speed updates.
    """
    N = len(ax_v)
    pos_enu = np.zeros((N, 2))
    vel_enu = np.zeros((N, 2))
    theta_est = np.zeros(N)

    pos_enu[0] = gnss_enu[0, :2]
    theta_est[0] = initial_theta_rad

    P_pos = np.eye(2) * 4.0
    P_vel = np.eye(2) * 0.5
    Q_pos = np.eye(2) * 0.01
    Q_vel = np.eye(2) * 0.05
    R_gnss = np.eye(2) * (2.5 ** 2)

    diverged = False
    bg_smooth = 0.0
    is_in_outage_prev = False

    for k in range(1, N):
        is_in_outage = outage_mask is not None and outage_mask[k]

        # Online Gyro Bias Smoothing during GNSS-aided driving
        if not is_in_outage and gnss_flags[k]:
            bg_smooth = 0.995 * bg_smooth + 0.005 * gy_v[k] * 0.05

        # 1. Heading integration with locked pre-outage bias
        eff_gyro = gy_v[k] - (bg_smooth if is_in_outage else 0.0)
        theta = theta_est[k - 1] + eff_gyro * dt
        c_th, s_th = math.cos(theta), math.sin(theta)

        # 2. Acceleration in ENU
        a_e = ay_v[k] * c_th + ax_v[k] * s_th
        a_n = ay_v[k] * s_th - ax_v[k] * c_th

        # 3. Velocity & Position prediction
        v_pred = vel_enu[k - 1].copy()
        v_pred[0] += a_e * dt
        v_pred[1] += a_n * dt

        # Non-Holonomic Constraints (NHC): Lateral velocity ~ 0
        if use_nhc:
            v_lat = v_pred[0] * s_th - v_pred[1] * c_th
            v_pred[0] -= 0.35 * (v_lat * s_th)
            v_pred[1] -= 0.35 * (-v_lat * c_th)

        p_pred = pos_enu[k - 1] + v_pred * dt

        P_vel += Q_vel * dt
        P_pos += P_vel * dt + Q_pos * dt

        # 4. Centripetal Kinematic Velocity Constraint: a_lateral = v * omega_yaw
        omega_mag = abs(eff_gyro)
        if omega_mag >= 0.035: # Turning maneuver (>= 2 deg/sec)
            # Physical speed from lateral centripetal acceleration
            v_centripetal = abs(ax_v[k]) / omega_mag
            if 2.0 <= v_centripetal <= 40.0: # 7 - 144 km/h valid vehicle speed range
                v_fwd_est = v_pred[0] * c_th + v_pred[1] * s_th
                innov_centripetal = v_centripetal - v_fwd_est
                r_centripetal = max(1.0, (0.25**2) / (omega_mag**2))
                k_gain = min(0.25, P_vel[0, 0] / (P_vel[0, 0] + r_centripetal))
                v_pred[0] += k_gain * innov_centripetal * c_th
                v_pred[1] += k_gain * innov_centripetal * s_th

        # 5. AI-Driven Zero-Velocity Update (ZUPT) & Spectral Vibration Scaling (10 Hz)
        if use_ai_speed and ai_speed is not None and k < len(ai_speed):
            z_speed = ai_speed[k]
            v_mag = np.linalg.norm(v_pred)
            
            # Physical ZUPT: ONLY when vehicle is already stopped (v_mag < 0.5 m/s)
            if v_mag < 0.5 and (z_speed < 1.0 or abs(ay_v[k]) < 0.10) and abs(gy_v[k]) < 0.04:
                v_pred[0] = 0.0
                v_pred[1] = 0.0
                P_vel[0, 0] = 0.0001
                P_vel[1, 1] = 0.0001
            else:
                # Dynamic vibration noise adaptation: road roughness scales filter process noise
                vib_energy = max(0.0, ai_var[k] if ai_var is not None else 0.5)
                P_vel += Q_vel * dt * (0.05 * math.log1p(vib_energy))

        # 6. GNSS Measurement Update
        is_gnss_valid = gnss_flags[k] and not is_in_outage

        if is_gnss_valid:
            z_pos = gnss_enu[k, :2]
            innov_pos = z_pos - p_pred

            S = P_pos + R_gnss
            K_pos = P_pos @ np.linalg.inv(S)
            p_pred += K_pos @ innov_pos
            v_pred += (K_pos @ innov_pos) * 0.5
            P_pos = (np.eye(2) - K_pos) @ P_pos

        if np.isnan(p_pred).any() or np.isnan(v_pred).any():
            diverged = True
            p_pred = np.nan_to_num(p_pred)
            v_pred = np.nan_to_num(v_pred)

        pos_enu[k] = p_pred
        vel_enu[k] = v_pred
        theta_est[k] = theta
        is_in_outage_prev = is_in_outage

    return pos_enu, vel_enu, diverged


def main():
    test_s_file = "ml/external/IO-VNBD_repo/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S3a/S-S3a.csv"
    test_v_file = "ml/external/IO-VNBD_repo/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S3a/V-S3a.csv"
    weights_path = "ml/weights/best_spectral_speed_filter.pt"

    print("=" * 75)
    print("   TASK 3: FULL-PIPELINE DRIFT BENCHMARK ON IN-DISTRIBUTION DRIVER (S3a)")
    print("=" * 75)
    print(f"Test Segment: {os.path.basename(test_s_file)} (In-Distribution Driver A - City/Suburban Driving)")

    df_s = pd.read_csv(test_s_file, encoding="latin1")
    df_v = pd.read_csv(test_v_file, encoding="latin1")
    df_s.columns = df_s.columns.str.strip()
    df_v.columns = df_v.columns.str.strip()

    # Evaluate representative 5,000 samples (8.33 minutes)
    N = min(5000, len(df_s), len(df_v))
    df_s = df_s.iloc[:N]
    df_v = df_v.iloc[:N]

    dt = 0.1
    time_sec = np.arange(N) * dt
    duration_min = (N * dt) / 60.0

    gt_lat = df_v["Latitude (degrees)"].values
    gt_lon = df_v["Longitude (degrees)"].values
    gt_alt = df_v["Height (km)"].values * 1000.0 if "Height (km)" in df_v.columns else np.zeros(N)
    gt_speed_kmh = df_v["Indicated Vehicle Speed (km/hr)"].values
    gt_speed_mps = gt_speed_kmh / 3.6

    lat0, lon0, alt0 = gt_lat[0], gt_lon[0], gt_alt[0]

    gt_enu = np.zeros((N, 3))
    for i in range(N):
        gt_enu[i] = geodetic_to_enu(gt_lat[i], gt_lon[i], gt_alt[i], lat0, lon0, alt0)

    dist_increments = np.sqrt(np.diff(gt_enu[:, 0])**2 + np.diff(gt_enu[:, 1])**2)
    total_distance_m = np.sum(dist_increments)

    # Initial Math ENU Angle theta
    theta0 = math.atan2(gt_enu[10, 1] - gt_enu[0, 1], gt_enu[10, 0] - gt_enu[0, 0])
    print(f"Drive Duration:            {duration_min:.2f} minutes ({N} samples)")
    print(f"Total Trajectory Distance: {total_distance_m:.1f} meters ({total_distance_m/1000.0:.2f} km)")
    print(f"Mean Ground Truth Speed:   {np.mean(gt_speed_kmh):.1f} km/h (Max: {np.max(gt_speed_kmh):.1f} km/h)")

    ax = df_s["ACCELEROMETER X (m/s²)"].values
    ay = df_s["ACCELEROMETER Y (m/s²)"].values
    az = df_s["ACCELEROMETER Z (m/s²)"].values
    gy = df_s["GYROSCOPE Yaw (rad/s)"].values
    gp = df_s["GYROSCOPE Pitch (rad/s)"].values
    gr = df_s["GYROSCOPE Roll (rad/s)"].values

    # Standard 1.0 Hz GNSS (Every 10 samples = 1s, with +/-2.5m noise)
    np.random.seed(42)
    gnss_1hz_enu = gt_enu.copy()
    gnss_1hz_enu[:, 0] += np.random.normal(0, 2.5, N)
    gnss_1hz_enu[:, 1] += np.random.normal(0, 2.5, N)

    gnss_1hz_flags = np.zeros(N, dtype=bool)
    gnss_1hz_flags[::10] = True

    # Load 16-Channel Spectral Speed Model
    device = torch.device("cpu")
    # 16-Channel Spectral Speed Model Inference with Causal EMA Smoothing
    window_size = 32
    model = SpeedVibrationFilterNet(in_channels=16, window_size=window_size)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded Spectral model weights from {weights_path}")
    model.eval()

    raw_6ch = np.stack([ax, ay, az, gy, gp, gr], axis=0)
    ai_speed_raw = np.zeros(N)
    ai_var_raw = np.zeros(N)

    print("Running 16-Channel Spectral Speed model inference...")
    with torch.no_grad():
        for i in range(window_size, N):
            w = raw_6ch[:, i - window_size : i]
            feat16 = compute_spectral_physics_features(w)
            out = model(torch.from_numpy(feat16).unsqueeze(0).float()).squeeze(0)
            ai_speed_raw[i] = max(0.0, out[0].item() * 3.6) # km/h
            ai_var_raw[i] = max(0.1, out[1].item())

    # Initial window fill
    ai_speed_raw[:window_size] = ai_speed_raw[window_size]
    ai_var_raw[:window_size] = ai_var_raw[window_size]

    # Apply Causal Exponential Moving Average (EMA) Smoothing (alpha = 0.20)
    ai_speed_kmh = np.zeros(N)
    ai_speed_kmh[0] = ai_speed_raw[0]
    alpha = 0.20
    for i in range(1, N):
        ai_speed_kmh[i] = (1.0 - alpha) * ai_speed_kmh[i - 1] + alpha * ai_speed_raw[i]

    ai_speed_mps = ai_speed_kmh / 3.6
    ai_var = ai_var_raw

    speed_mae = np.mean(np.abs(ai_speed_kmh[window_size:] - gt_speed_kmh[window_size:]))
    speed_rmse = np.sqrt(np.mean((ai_speed_kmh[window_size:] - gt_speed_kmh[window_size:]) ** 2))
    speed_corr = np.corrcoef(ai_speed_kmh[window_size:], gt_speed_kmh[window_size:])[0, 1]

    print(f"In-Distribution Speed Estimation Accuracy: MAE = {speed_mae:.2f} km/h, Pearson r = {speed_corr:.3f}")

    print("\nExecuting Pipeline Configurations...")

    # --- CONFIG (a): Raw Strapdown INS Only ---
    pos_a, vel_a = run_raw_ins(ax, ay, az, gy, theta0, dt=dt)
    drift_a = np.linalg.norm(pos_a[-1] - gt_enu[-1, :2])
    drift_pct_a = (drift_a / total_distance_m) * 100.0

    # --- CONFIG (b): EKF + NHC + GNSS (No AI Speed) ---
    pos_b, vel_b, div_b = run_ekf(
        ax, ay, az, gy,
        gnss_1hz_enu,
        gnss_1hz_flags,
        initial_theta_rad=theta0,
        use_ai_speed=False,
        use_nhc=True,
        dt=dt,
    )
    drift_b = np.linalg.norm(pos_b[-1] - gt_enu[-1, :2])
    drift_pct_b = (drift_b / total_distance_m) * 100.0
    err_b_series = np.linalg.norm(pos_b - gt_enu[:, :2], axis=1)

    # --- CONFIG (c): Full Pipeline: EKF + NHC + GNSS + Spectral AI Speed ---
    pos_c, vel_c, div_c = run_ekf(
        ax, ay, az, gy,
        gnss_1hz_enu,
        gnss_1hz_flags,
        ai_speed=ai_speed_mps,
        ai_var=ai_var,
        initial_theta_rad=theta0,
        use_ai_speed=True,
        use_nhc=True,
        dt=dt,
    )
    drift_c = np.linalg.norm(pos_c[-1] - gt_enu[-1, :2])
    drift_pct_c = (drift_c / total_distance_m) * 100.0
    err_c_series = np.linalg.norm(pos_c - gt_enu[:, :2], axis=1)

    # --- CONFIG (d): 90-Second Simulated Tunnel Outage Scenario ---
    outage_start_k = 1200  # t = 120s
    outage_end_k = 2100    # t = 210s
    outage_mask = np.zeros(N, dtype=bool)
    outage_mask[outage_start_k:outage_end_k] = True
    outage_dist_m = np.sum(dist_increments[outage_start_k : outage_end_k - 1])

    pos_outage_no_ai, _, _ = run_ekf(
        ax, ay, az, gy,
        gnss_1hz_enu,
        gnss_1hz_flags,
        outage_mask=outage_mask,
        initial_theta_rad=theta0,
        use_ai_speed=False,
        use_nhc=True,
        dt=dt,
    )
    drift_outage_no_ai_end = np.linalg.norm(pos_outage_no_ai[outage_end_k] - gt_enu[outage_end_k, :2])
    drift_outage_no_ai_pct = (drift_outage_no_ai_end / outage_dist_m) * 100.0

    pos_outage_ai, _, div_d = run_ekf(
        ax, ay, az, gy,
        gnss_1hz_enu,
        gnss_1hz_flags,
        ai_speed=ai_speed_mps,
        ai_var=ai_var,
        outage_mask=outage_mask,
        initial_theta_rad=theta0,
        use_ai_speed=True,
        use_nhc=True,
        dt=dt,
    )
    drift_outage_ai_end = np.linalg.norm(pos_outage_ai[outage_end_k] - gt_enu[outage_end_k, :2])
    drift_outage_ai_pct = (drift_outage_ai_end / outage_dist_m) * 100.0

    err_outage_no_ai = np.linalg.norm(pos_outage_no_ai - gt_enu[:, :2], axis=1)
    err_outage_ai = np.linalg.norm(pos_outage_ai - gt_enu[:, :2], axis=1)

    print("\n" + "=" * 75)
    print("        TASK 3: IN-DISTRIBUTION FULL-PIPELINE DRIFT RESULTS")
    print("=" * 75)
    print(f"Drive Duration:            {duration_min:.2f} minutes ({N} samples at 10 Hz)")
    print(f"Total Trajectory Distance: {total_distance_m:.1f} meters ({total_distance_m/1000.0:.2f} km)")
    print(f"Outage Window Duration:    {(outage_end_k - outage_start_k)*dt:.1f} seconds ({outage_dist_m:.1f} meters traveled in outage)")
    print("-" * 75)
    print(f"{'Configuration':<45} | {'Mean Error (m)':<14} | {'Max Error (m)':<13} | {'Final Drift':<12}")
    print("-" * 75)
    print(f"{'(a) Raw Strapdown INS (Uncorrected)':<45} | {'-':<14} | {'-':<13} | {drift_a:.1f}m ({drift_pct_a:.1f}%)")
    print(f"{'(b) EKF + NHC + GNSS (No AI Speed)':<45} | {np.mean(err_b_series):<14.2f} | {np.max(err_b_series):<13.2f} | {drift_b:.2f}m ({drift_pct_b:.2f}%)")
    print(f"{'(c) Full Pipeline (EKF + NHC + GNSS + AI)':<45} | {np.mean(err_c_series):<14.2f} | {np.max(err_c_series):<13.2f} | {drift_c:.2f}m ({drift_pct_c:.2f}%)")
    print("-" * 75)
    print("GNSS-DENIED OUTAGE BENCHMARK (90s Outage Window):")
    print(f"{'  - Outage without AI (Pure INS + NHC only)':<45} | Drift at Outage End: {drift_outage_no_ai_end:.2f}m ({drift_outage_no_ai_pct:.2f}%)")
    print(f"{'  - Outage with Spectral AI Speed Model':<45} | Drift at Outage End: {drift_outage_ai_end:.2f}m ({drift_outage_ai_pct:.2f}%)")
    print("=" * 75)

    # Generate 4-Quadrant Benchmark Plots
    os.makedirs("ml/evaluation_plots", exist_ok=True)
    plot_path = "ml/evaluation_plots/indistribution_trajectory_drift_benchmark.png"

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    plt.suptitle("IDR-Nav In-Distribution Full-Pipeline Evaluation (Driver A - Drive S3a)", fontsize=14, fontweight="bold")

    # Plot 1: Trajectory Overlay
    ax1 = axs[0, 0]
    ax1.plot(gt_enu[:, 0], gt_enu[:, 1], "k-", linewidth=2.5, label="Ground Truth (ECU GPS)")
    ax1.plot(pos_b[:, 0], pos_b[:, 1], "b--", linewidth=1.5, alpha=0.8, label="Config (b): EKF (No AI)")
    ax1.plot(pos_c[:, 0], pos_c[:, 1], "g-", linewidth=1.8, alpha=0.9, label="Config (c): Full Pipeline (With AI)")
    ax1.plot(pos_outage_ai[:, 0], pos_outage_ai[:, 1], "r-.", linewidth=1.8, label="Config (d): 90s GNSS Outage")
    ax1.scatter([gt_enu[outage_start_k, 0]], [gt_enu[outage_start_k, 1]], color="red", s=90, zorder=5, label="Outage Start (t=120s)")
    ax1.scatter([gt_enu[outage_end_k, 0]], [gt_enu[outage_end_k, 1]], color="darkred", s=90, marker="x", zorder=5, label="Outage End (t=210s)")
    ax1.set_title("2D Local ENU Trajectory Overlay")
    ax1.set_xlabel("East (meters)")
    ax1.set_ylabel("North (meters)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=9)

    # Plot 2: Euclidean Error Over Time
    ax2 = axs[0, 1]
    ax2.plot(time_sec, err_b_series, "b--", alpha=0.8, label="Config (b): EKF GNSS-aided (No AI)")
    ax2.plot(time_sec, err_c_series, "g-", alpha=0.9, label="Config (c): Full Pipeline GNSS-aided (With AI)")
    ax2.plot(time_sec, err_outage_no_ai, "m:", label="90s Outage (Without AI Model)")
    ax2.plot(time_sec, err_outage_ai, "r-", linewidth=1.8, label="90s Outage (With AI Model)")
    ax2.axvspan(outage_start_k * dt, outage_end_k * dt, color="gray", alpha=0.2, label="GNSS Outage Window (90s)")
    ax2.set_title("Euclidean Positional Error Over Time (meters)")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Error (meters)")
    ax2.set_ylim(0, max(50, np.max(err_outage_ai) * 1.1))
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", fontsize=9)

    # Plot 3: Speed Tracking Comparison (Smooth Causal EMA)
    ax3 = axs[1, 0]
    ax3.plot(time_sec, gt_speed_kmh, "k-", linewidth=1.8, label="Ground Truth Speed (km/h)")
    ax3.plot(time_sec, ai_speed_kmh, "g-", linewidth=1.5, label="AI Speed Prediction (Smooth EMA)")
    
    # Smooth 1-sigma uncertainty band
    sigma_kmh = np.sqrt(ai_var) * 1.5
    ax3.fill_between(
        time_sec,
        np.maximum(0.0, ai_speed_kmh - sigma_kmh),
        ai_speed_kmh + sigma_kmh,
        color="green",
        alpha=0.15,
        label="AI ±1σ Uncertainty Band",
    )
    ax3.set_title(f"Forward Speed Tracking (MAE: {speed_mae:.2f} km/h, r: {speed_corr:.3f})")
    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel("Speed (km/h)")
    ax3.set_ylim(-2, max(85, np.max(gt_speed_kmh) * 1.15))
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right", fontsize=9)

    # Plot 4: Log Scale Comparison
    ax4 = axs[1, 1]
    err_a = np.linalg.norm(pos_a - gt_enu[:, :2], axis=1)
    ax4.semilogy(time_sec, err_a, "k-", label="Raw Strapdown INS Drift (m)")
    ax4.semilogy(time_sec, err_outage_no_ai, "m:", label="90s Outage without AI (m)")
    ax4.semilogy(time_sec, err_outage_ai, "r-", label="90s Outage with AI Model (m)")
    ax4.semilogy(time_sec, err_c_series, "g-", label="Full Pipeline GNSS-Aided (m)")
    ax4.set_title("Drift Mitigation Comparison (Log Scale)")
    ax4.set_xlabel("Time (seconds)")
    ax4.set_ylabel("Error in Meters (Log Scale)")
    ax4.grid(True, alpha=0.3, which="both")
    ax4.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    print(f"Saved in-distribution benchmark plot to {plot_path}")


if __name__ == "__main__":
    main()
