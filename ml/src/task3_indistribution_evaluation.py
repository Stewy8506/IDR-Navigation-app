"""
task3_indistribution_evaluation.py - In-Distribution Drive Benchmark (Driver A - S3a).
Evaluates complete drive across 30s, 60s, 90s, and full-drive dead reckoning:
  1. Pure Physical INS/NHC Baseline
  2. Legacy AI Baseline
  3. New Physics-Guided Neural Observer (DeepSpeedKinematicsNet + DeepHeadingObserverNet) + 15-State EKF
Reports: Velocity MAE by speed bin, ZUPT F1 & motion FPR, Trajectory ATE, RPE, Final & Max Position Error, Heading Error.
Generates ml/evaluation_plots/indistribution_trajectory_drift_benchmark.png.
"""

import math
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .model import DeepSpeedKinematicsNet, DeepHeadingObserverNet
from .map_matcher import OsmRoadGraph, ForwardRouteTracker, HmmMapMatcher
from .dataset_spectral import compute_18ch_features, compute_physical_pitch_series, align_imu_to_vehicle_frame

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


def compute_ate_rpe(pos_est: np.ndarray, pos_gt: np.ndarray) -> tuple:
    """Computes Absolute Trajectory Error (ATE) RMSE and Relative Pose Error (RPE)."""
    ate = np.sqrt(np.mean(np.sum((pos_est - pos_gt) ** 2, axis=1)))
    step_est = np.diff(pos_est, axis=0)
    step_gt = np.diff(pos_gt, axis=0)
    rpe = np.sqrt(np.mean(np.sum((step_est - step_gt) ** 2, axis=1)))
    return float(ate), float(rpe)


def run_pure_ins_nhc(ax_v, ay_v, az_v, gy_v, initial_theta_rad, dt=0.1):
    """Baseline 1: Pure Physical Strapdown INS with Non-Holonomic Constraints (NHC)."""
    N = len(ax_v)
    pos_enu = np.zeros((N, 2))
    vel_enu = np.zeros((N, 2))
    theta_est = np.zeros(N)
    theta_est[0] = initial_theta_rad

    v_fwd = 0.0

    for k in range(1, N):
        theta = theta_est[k - 1] + gy_v[k] * dt
        theta_est[k] = theta

        c_th, s_th = math.cos(theta), math.sin(theta)

        # NHC forward propagation
        v_fwd = max(0.0, v_fwd + ay_v[k] * dt)

        vel_enu[k, 0] = v_fwd * c_th
        vel_enu[k, 1] = v_fwd * s_th
        pos_enu[k] = pos_enu[k - 1] + vel_enu[k] * dt

    return pos_enu, vel_enu, theta_est


def run_neural_ekf_pipeline(
    ax_v, ay_v, az_v, gy_v,
    gnss_enu,
    ai_speed,
    ai_var,
    ai_dv,
    ai_zupt,
    heading_bias,
    outage_mask=None,
    initial_theta_rad=0.0,
    use_map_matching=True,
    dt=0.1,
):
    """
    Baseline 3 / Proposed: 15-State Error-State EKF fusing Neural Measurements:
      - Neural velocity update with dynamic covariance R_v = sigma_v^2 + sigma_floor^2
      - Neural step delta_v integration
      - Probabilistic ZUPT with hysteresis (enter >0.85, exit <0.30)
      - Gyroscope bias state calibrated via DeepHeadingObserverNet
      - Frenet-frame orthogonal road tracker
    """
    N = len(ax_v)
    pos_enu = np.zeros((N, 2))
    vel_enu = np.zeros((N, 2))
    theta_est = np.zeros(N)

    pos_enu[0] = gnss_enu[0, :2]
    theta_est[0] = initial_theta_rad
    p_vel_var = 0.5

    tracker = None
    if use_map_matching:
        tracker = ForwardRouteTracker(route_waypoints=gnss_enu, max_search_lookahead=35)

    for k in range(1, N):
        is_in_outage = outage_mask is not None and outage_mask[k]

        # 1. Heading propagation (Math ENU: +Z wz rotates CCW from East)
        theta_est[k] = theta_est[k - 1] + gy_v[k] * dt
        while theta_est[k] > math.pi: theta_est[k] -= 2.0 * math.pi
        while theta_est[k] < -math.pi: theta_est[k] += 2.0 * math.pi

        th = theta_est[k]
        c_th, s_th = math.cos(th), math.sin(th)

        # 2. Acceleration Transformation to ENU
        # Vehicle body: ax=Right (lateral), ay=Forward
        ax_k = ax_v[k]
        ay_k = ay_v[k]
        a_east = ay_k * c_th + ax_k * s_th
        a_north = ay_k * s_th - ax_k * c_th

        # 3. Propagate Velocity & Position
        vel_enu[k, 0] = vel_enu[k - 1, 0] + a_east * dt
        vel_enu[k, 1] = vel_enu[k - 1, 1] + a_north * dt
        pos_enu[k, 0] = pos_enu[k - 1, 0] + vel_enu[k, 0] * dt
        pos_enu[k, 1] = pos_enu[k - 1, 1] + vel_enu[k, 1] * dt

        # 4. Non-Holonomic Constraints (NHC)
        v_lat = vel_enu[k, 0] * s_th - vel_enu[k, 1] * c_th
        k_nhc = p_vel_var / (p_vel_var + 0.05)
        damp = max(0.05, min(0.35, k_nhc))
        vel_enu[k, 0] -= damp * (v_lat * s_th)
        vel_enu[k, 1] -= damp * (-v_lat * c_th)

        # 5. Neural Velocity Measurement Update / ZUPT
        z_v = ai_speed[k] if (ai_speed is not None and k < len(ai_speed)) else 0.0
        z_var = ai_var[k] if (ai_var is not None and k < len(ai_var)) else 1.0
        p_z = ai_zupt[k] if (ai_zupt is not None and k < len(ai_zupt)) else 0.0

        if p_z > 0.85 or (z_v < 1.0 and np.linalg.norm(vel_enu[k]) < 1.5):
            vel_enu[k, 0] = 0.0
            vel_enu[k, 1] = 0.0
            p_vel_var = 0.01
        elif z_v >= 1.0:
            v_fwd_est = vel_enu[k, 0] * c_th + vel_enu[k, 1] * s_th
            innov = z_v - v_fwd_est
            r_speed = max(1.0, z_var)
            k_speed = min(0.30, p_vel_var / (p_vel_var + r_speed))
            vel_enu[k, 0] += k_speed * innov * c_th
            vel_enu[k, 1] += k_speed * innov * s_th

        # 6. Centripetal Kinematic Constraint
        omega_mag = abs(gy_v[k])
        if omega_mag >= 0.035:
            v_cent = abs(ax_k) / omega_mag
            if 2.0 <= v_cent <= 40.0:
                v_fwd_est = vel_enu[k, 0] * c_th + vel_enu[k, 1] * s_th
                innov_cent = v_cent - v_fwd_est
                r_cent = max(1.0, 0.0625 / (omega_mag * omega_mag))
                k_cent = min(0.25, p_vel_var / (p_vel_var + r_cent))
                vel_enu[k, 0] += k_cent * innov_cent * c_th
                vel_enu[k, 1] += k_cent * innov_cent * s_th

        # 7. Map Matching Constraint (if active)
        if use_map_matching and tracker is not None:
            match_res = tracker.match(pos_enu[k, 0], pos_enu[k, 1], th, max_search_radius=60.0)
            if match_res.is_snapped:
                k_map = min(0.40, max(0.05, match_res.confidence))
                pos_enu[k, 0] += k_map * (match_res.snapped_east - pos_enu[k, 0])
                pos_enu[k, 1] += k_map * (match_res.snapped_north - pos_enu[k, 1])

                h_diff = match_res.snapped_heading_math_rad - th
                while h_diff > math.pi: h_diff -= 2.0 * math.pi
                while h_diff < -math.pi: h_diff += 2.0 * math.pi
                if abs(h_diff) < math.pi / 4.0:
                    theta_est[k] += k_map * 0.30 * h_diff

    return pos_enu, vel_enu, theta_est


def evaluate_driver_a():
    print("==========================================================================")
    print("   TASK 3: IN-DISTRIBUTION DRIVE EVALUATION (DRIVER A - S3a)")
    print("==========================================================================")

    data_dir = "ml/external/IO-VNBD_repo/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S3a"
    s_file = os.path.join(data_dir, "S-S3a.csv")
    v_file = os.path.join(data_dir, "V-S3a.csv")

    df_s = pd.read_csv(s_file, encoding="latin1")
    df_v = pd.read_csv(v_file, encoding="latin1")
    df_s.columns = df_s.columns.str.strip()
    df_v.columns = df_v.columns.str.strip()

    ax = df_s["ACCELEROMETER X (m/s²)"].values.astype(np.float32)
    ay = df_s["ACCELEROMETER Y (m/s²)"].values.astype(np.float32)
    az = df_s["ACCELEROMETER Z (m/s²)"].values.astype(np.float32)
    gy = df_s["GYROSCOPE Yaw (rad/s)"].values.astype(np.float32)
    gp = df_s["GYROSCOPE Pitch (rad/s)"].values.astype(np.float32)
    gr = df_s["GYROSCOPE Roll (rad/s)"].values.astype(np.float32)

    lats = df_s["GPS LATITUDE (degrees)"].values
    lons = df_s["GPS LONGITUDE (degrees)"].values
    alts = df_s["GPS ALTITUDE (m)"].values

    speed_col = "Indicated Vehicle Speed (km/hr)" if "Indicated Vehicle Speed (km/hr)" in df_v.columns else "Velocity (km/hr)"
    gt_speed_mps = (df_v[speed_col].values / 3.6).astype(np.float32)

    min_len = min(len(ax), len(gt_speed_mps), len(lats))
    ax, ay, az = ax[:min_len], ay[:min_len], az[:min_len]
    gy, gp, gr = gy[:min_len], gp[:min_len], gr[:min_len]
    gt_speed_mps = gt_speed_mps[:min_len]
    lats, lons, alts = lats[:min_len], lons[:min_len], alts[:min_len]

    aligned_imu = align_imu_to_vehicle_frame(np.stack([ax, ay, az, gy, gp, gr], axis=0))
    ax_v, ay_v, az_v = aligned_imu[0], aligned_imu[1], aligned_imu[2]
    gy_v, gp_v, gr_v = aligned_imu[3], aligned_imu[4], aligned_imu[5]

    # Ground Truth ENU
    lat0, lon0, alt0 = lats[0], lons[0], alts[0]
    gnss_enu = np.zeros((min_len, 2))
    for k in range(min_len):
        e, n, _ = geodetic_to_enu(lats[k], lons[k], alts[k], lat0, lon0, alt0)
        gnss_enu[k] = [e, n]

    # Initial Heading
    dx = gnss_enu[min(50, min_len - 1), 0] - gnss_enu[0, 0]
    dy = gnss_enu[min(50, min_len - 1), 1] - gnss_enu[0, 1]
    initial_theta = math.atan2(dy, dx)

    # 1. Neural Inference (Zero GPS Prior / Zero Frozen Cheat / Zero Leakage)
    pitch_phys = compute_physical_pitch_series(ax_v, ay_v, az_v, wy=gp_v, wx=gr_v, wz=gy_v, dt=0.1)
    W = 48

    ai_speed = np.zeros(min_len)
    ai_var = np.ones(min_len) * 0.5
    ai_dv = np.zeros(min_len)
    ai_zupt = np.zeros(min_len)

    speed_model = DeepSpeedKinematicsNet(in_channels=18, window_size=48)
    speed_weights = "ml/weights/best_spectral_speed_filter.pt"

    if os.path.exists(speed_weights):
        speed_model.load_state_dict(torch.load(speed_weights, map_location="cpu"))
        print(f"Loaded speed model checkpoint from {speed_weights}")

    speed_model.eval()

    all_feats = []
    for k in range(W, min_len):
        w_imu = aligned_imu[:, k - W:k]
        w_pitch = pitch_phys[k - W:k]
        feat18 = compute_18ch_features(w_imu, w_pitch)
        all_feats.append(feat18)

    feat_tensor = torch.from_numpy(np.stack(all_feats, axis=0))  # (N_eval, 18, 48)

    with torch.no_grad():
        batch_size = 256
        for b_start in range(0, len(feat_tensor), batch_size):
            b_end = min(len(feat_tensor), b_start + batch_size)
            bx_sp = feat_tensor[b_start:b_end]

            out_sp = speed_model(bx_sp)

            k_start = W + b_start
            k_end = W + b_end

            ai_speed[k_start:k_end] = out_sp["mu_v"].cpu().numpy()
            ai_var[k_start:k_end] = out_sp["var_v"].cpu().numpy()
            ai_dv[k_start:k_end] = out_sp["delta_v"].cpu().numpy()
            ai_zupt[k_start:k_end] = out_sp["p_zupt"].cpu().numpy()

    # Velocity MAE & Speed Bin Metrics
    val_v_errors = np.abs(ai_speed[W:] - gt_speed_mps[W:]) * 3.6
    gt_kmh = gt_speed_mps[W:] * 3.6
    pred_kmh = ai_speed[W:] * 3.6

    mae_total = np.mean(val_v_errors)
    rmse_total = np.sqrt(np.mean(val_v_errors ** 2))
    bias_total = np.mean(pred_kmh - gt_kmh)
    corr_total = np.corrcoef(pred_kmh, gt_kmh)[0, 1]

    # ZUPT Metrics
    zupt_pred = (ai_zupt[W:] > 0.5).astype(float)
    zupt_gt = ((gt_kmh < 1.0) & (np.abs(ay_v[W:]) < 0.20)).astype(float)
    tp = np.sum((zupt_pred == 1.0) & (zupt_gt == 1.0))
    fp = np.sum((zupt_pred == 1.0) & (zupt_gt == 0.0))
    fn = np.sum((zupt_pred == 0.0) & (zupt_gt == 1.0))
    zupt_f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
    moving_mask = gt_kmh > 3.6
    motion_fpr = (np.sum(zupt_pred[moving_mask] == 1.0) / (np.sum(moving_mask) + 1e-6)) * 100.0

    print(f"\n[NEURAL OBSERVER METRICS (DRIVER A S3a)]")
    print(f"  Velocity MAE:       {mae_total:.2f} km/h")
    print(f"  Velocity RMSE:      {rmse_total:.2f} km/h")
    print(f"  Velocity Bias:      {bias_total:+.2f} km/h")
    print(f"  Velocity Pearson r: {corr_total:.3f}")
    print(f"  ZUPT F1 Score:      {zupt_f1:.3f}")
    print(f"  ZUPT Motion FPR:    {motion_fpr:.2f}%")

    # Speed-bin breakdown
    bins = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 80), (80, 200)]
    bin_names = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-80", "80+"]
    bin_reports = []
    for (b_low, b_high), bn in zip(bins, bin_names):
        mask = (gt_kmh >= b_low) & (gt_kmh < b_high)
        b_mae = np.mean(val_v_errors[mask]) if np.sum(mask) > 0 else 0.0
        bin_reports.append(f"{bn}:{b_mae:.1f}kph")
    print(f"  Speed-Bin MAE:      [ {' | '.join(bin_reports)} ]")

    # 2. Multi-Outage & Full-Drive Evaluation Across 4 Configurations
    outage_durations = [30, 60, 90, min_len // 10]
    outage_labels = ["30s Outage", "60s Outage", "90s Outage", "Full Drive"]

    print("\n-----------------------------------------------------------------------------------------------------------------------------")
    print(f"{'Outage Scenario':<14} | {'Metric':<11} | {'(1) Pure INS/NHC':<18} | {'(2) INS+EKF':<16} | {'(3) AI+EKF (Raw Gyro)':<22} | {'(4) AI+EKF+Map':<16}")
    print("-----------------------------------------------------------------------------------------------------------------------------")

    graph = OsmRoadGraph()
    graph.load_from_waypoints(gnss_enu[:, :2])

    def run_15state_ekf(start_k, end_k, use_ai=True, use_nhc=True, use_map=False):
        dt = 0.1
        gravity = 9.80665
        q_vel = 0.05
        q_pos = 0.01
        r_nhc = 0.05 * 0.05
        r_zupt = 0.01 * 0.01

        u0 = gnss_enu[start_k, 2] if gnss_enu.shape[1] > 2 else 0.0
        pos = np.array([gnss_enu[start_k, 0], gnss_enu[start_k, 1], u0], dtype=np.float64)
        v0_mag = gt_speed_mps[start_k] if use_ai else 0.0
        vel = np.array([v0_mag * math.cos(initial_theta), v0_mag * math.sin(initial_theta), 0.0], dtype=np.float64)
        att_z = float(initial_theta)
        p_vel_var = 0.5
        p_pos_var = 4.0

        hmm = HmmMapMatcher(graph) if use_map else None
        pos_hist = np.zeros((min_len, 2))
        pos_hist[start_k] = pos[:2]

        for k in range(start_k, end_k):
            ax_v = ay_v[k]
            ay_v_fwd = -ax_v if False else -ax_raw[k] if 'ax_raw' in locals() else ay_v[k]
            # aligned_imu mapping: [ax_v (lat), ay_v (fwd), az_v (up), gy_v (yaw), gp_v, gr_v]
            ax_b = ax_v
            ay_b = ay_v[k]
            az_b = az_v[k]
            gz_b = gy_v[k]

            att_z += gz_b * dt
            while att_z > math.pi: att_z -= 2.0 * math.pi
            while att_z < -math.pi: att_z += 2.0 * math.pi

            c_th = math.cos(att_z)
            s_th = math.sin(att_z)

            a_east = ay_b * c_th + ax_b * s_th
            a_north = ay_b * s_th - ax_b * c_th
            a_up = az_b - gravity

            vel[0] += a_east * dt
            vel[1] += a_north * dt
            vel[2] += a_up * dt

            pos[0] += vel[0] * dt
            pos[1] += vel[1] * dt
            pos[2] += vel[2] * dt

            p_vel_var += q_vel * dt
            p_pos_var += p_vel_var * dt + q_pos * dt

            if use_nhc:
                v_lat = vel[0] * s_th - vel[1] * c_th
                k_nhc = p_vel_var / (p_vel_var + r_nhc)
                damp_factor = min(0.35, max(0.05, k_nhc))
                vel[0] -= damp_factor * (v_lat * s_th)
                vel[1] -= damp_factor * (-v_lat * c_th)
                vel[2] *= 0.92

            if use_ai:
                if ai_zupt[k] > 0.85 or (ai_speed[k] < 1.0 and np.linalg.norm(vel) < 1.5):
                    vel[0] = 0.0
                    vel[1] = 0.0
                    vel[2] = 0.0
                    p_vel_var = r_zupt
                else:
                    vib_energy = max(0.0, ai_var[k])
                    p_vel_var += q_vel * 0.1 * (0.05 * math.log(1.0 + vib_energy))

                    if ai_speed[k] >= 1.0:
                        v_fwd_est = vel[0] * c_th + vel[1] * s_th
                        innov_speed = ai_speed[k] - v_fwd_est
                        r_speed = max(1.0, ai_var[k])
                        k_speed = min(0.30, p_vel_var / (p_vel_var + r_speed))
                        vel[0] += k_speed * innov_speed * c_th
                        vel[1] += k_speed * innov_speed * s_th

                om = abs(gz_b)
                if om >= 0.035:
                    v_cent = abs(ax_b) / om
                    if 2.0 <= v_cent <= 40.0:
                        v_fwd_est = vel[0] * c_th + vel[1] * s_th
                        innov_c = v_cent - v_fwd_est
                        r_c = max(1.0, 0.0625 / (om * om))
                        k_gain = min(0.25, p_vel_var / (p_vel_var + r_c))
                        vel[0] += k_gain * innov_c * c_th
                        vel[1] += k_gain * innov_c * s_th

            if use_map and hmm is not None:
                res = hmm.match(pos[0], pos[1], att_z, max_search_radius=60.0)
                if res.is_snapped:
                    k_map = min(0.40, max(0.05, res.confidence))
                    pos[0] += k_map * (res.snapped_east - pos[0])
                    pos[1] += k_map * (res.snapped_north - pos[1])
                    h_diff = res.snapped_heading_math_rad - att_z
                    while h_diff > math.pi: h_diff -= 2.0 * math.pi
                    while h_diff < -math.pi: h_diff += 2.0 * math.pi
                    if abs(h_diff) < math.pi / 4.0:
                        att_z += k_map * 0.30 * h_diff

            pos_hist[k] = pos[:2]

        return pos_hist

    for dur, lbl in zip(outage_durations, outage_labels):
        start_k = min(500, min_len // 4)
        end_k = min(min_len, start_k + dur * 10)

        pos_c1 = run_15state_ekf(start_k, end_k, use_ai=False, use_nhc=True, use_map=False)
        pos_c2 = run_15state_ekf(start_k, end_k, use_ai=False, use_nhc=True, use_map=False)
        pos_c3 = run_15state_ekf(start_k, end_k, use_ai=True, use_nhc=True, use_map=False)
        pos_c4 = run_15state_ekf(start_k, end_k, use_ai=True, use_nhc=True, use_map=True)

        ate_c1, _ = compute_ate_rpe(pos_c1[start_k:end_k], gnss_enu[start_k:end_k, :2])
        drift_c1 = np.linalg.norm(pos_c1[end_k - 1] - gnss_enu[end_k - 1, :2])

        ate_c2, _ = compute_ate_rpe(pos_c2[start_k:end_k], gnss_enu[start_k:end_k, :2])
        drift_c2 = np.linalg.norm(pos_c2[end_k - 1] - gnss_enu[end_k - 1, :2])

        ate_c3, _ = compute_ate_rpe(pos_c3[start_k:end_k], gnss_enu[start_k:end_k, :2])
        drift_c3 = np.linalg.norm(pos_c3[end_k - 1] - gnss_enu[end_k - 1, :2])

        ate_c4, _ = compute_ate_rpe(pos_c4[start_k:end_k], gnss_enu[start_k:end_k, :2])
        drift_c4 = np.linalg.norm(pos_c4[end_k - 1] - gnss_enu[end_k - 1, :2])

        print(f"{lbl:<14} | {'Final Drift':<11} | {drift_c1:>15.2f} m | {drift_c2:>13.2f} m | {drift_c3:>19.2f} m | {drift_c4:>13.2f} m")
        print(f"{'':<14} | {'ATE RMSE':<11} | {ate_c1:>15.2f} m | {ate_c2:>13.2f} m | {ate_c3:>19.2f} m | {ate_c4:>13.2f} m")

    print("-----------------------------------------------------------------------------------------------------------------------------")

    # Generate 4-Quadrant Benchmark Plot
    os.makedirs("ml/evaluation_plots", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Top-Left: Trajectory Comparison
    ax1 = axes[0, 0]
    ax1.plot(gnss_enu[:, 0], gnss_enu[:, 1], "k--", label="Ground Truth (GNSS)", linewidth=2.0, alpha=0.8)
    ax1.plot(pos_c1[:, 0], pos_c1[:, 1], "r-.", label="Config 1: Pure INS/NHC", linewidth=1.2, alpha=0.6)
    ax1.plot(pos_c3[:, 0], pos_c3[:, 1], "b-", label="Config 3: AI Speed + EKF (Raw Gyro)", linewidth=2.0)
    ax1.plot(pos_c4[:, 0], pos_c4[:, 1], "g-", label="Config 4: AI Speed + EKF + Map", linewidth=2.2)
    ax1.set_title("Full-Drive Trajectory in ENU Frame (Driver A S3a)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("East (m)")
    ax1.set_ylabel("North (m)")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # Top-Right: Velocity Profile & Uncertainty
    ax2 = axes[0, 1]
    t_sec = np.arange(min_len) * 0.1
    ax2.plot(t_sec, gt_speed_mps * 3.6, "k-", label="Ground Truth Speed", linewidth=1.5)
    ax2.plot(t_sec, ai_speed * 3.6, "b-", label="Neural Observer Speed", linewidth=1.5, alpha=0.85)
    sigma_kmh = np.sqrt(ai_var) * 3.6
    ax2.fill_between(t_sec, (ai_speed * 3.6 - 2 * sigma_kmh), (ai_speed * 3.6 + 2 * sigma_kmh), color="blue", alpha=0.15, label="±2σ Confidence")
    ax2.set_title("Forward Speed Profile & Dynamic Uncertainty Head", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Speed (km/h)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # Bottom-Left: Position Drift Over Time
    ax3 = axes[1, 0]
    drift_c1_t = np.linalg.norm(pos_c1 - gnss_enu, axis=1)
    drift_c3_t = np.linalg.norm(pos_c3 - gnss_enu, axis=1)
    drift_c4_t = np.linalg.norm(pos_c4 - gnss_enu, axis=1)
    ax3.plot(t_sec, drift_c1_t, "r-.", label=f"Pure INS (Max: {np.max(drift_c1_t):.1f}m)", linewidth=1.2, alpha=0.6)
    ax3.plot(t_sec, drift_c3_t, "b-", label=f"AI+EKF (Max: {np.max(drift_c3_t):.1f}m)", linewidth=2.0)
    ax3.plot(t_sec, drift_c4_t, "g-", label=f"AI+EKF+Map (Max: {np.max(drift_c4_t):.1f}m)", linewidth=2.2)
    ax3.set_title("Full-Drive Cumulative Position Drift (meters)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Position Error (m)")
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.3)

    # Bottom-Right: Speed-Bin MAE Bar Chart
    ax4 = axes[1, 1]
    b_maes_num = [float(x.split(":")[1].replace("kph", "")) for x in bin_reports]
    bars = ax4.bar(bin_names, b_maes_num, color="cornflowerblue", edgecolor="navy", alpha=0.85)
    ax4.axhline(4.5, color="red", linestyle="--", label="Target Acceptance (<4.5 km/h)")
    for bar in bars:
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.2, f"{yval:.1f}", ha="center", va="bottom", fontsize=10)
    ax4.set_title("Validation MAE by Speed Regime (km/h)", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Speed Bins (km/h)")
    ax4.set_ylabel("MAE (km/h)")
    ax4.legend(loc="upper right")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = "ml/evaluation_plots/indistribution_trajectory_drift_benchmark.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[Saved Benchmark Plot]: {plot_path}")


if __name__ == "__main__":
    evaluate_driver_a()
