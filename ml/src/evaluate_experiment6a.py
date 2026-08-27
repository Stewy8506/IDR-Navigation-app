import os, sys, math, time, torch, numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml.src.model import DeepSpeedKinematicsNet
from ml.src.dataset_spectral import compute_18ch_features, compute_physical_pitch_series, align_imu_to_vehicle_frame
from ml.src.map_matcher import OsmRoadGraph, HmmMapMatcher

def evaluate_drive_closed_loop(s_path, v_path, checkpoint_path, drive_name="Drive"):
    df_s = pd.read_csv(s_path, encoding="latin1")
    df_v = pd.read_csv(v_path, encoding="latin1")
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
    lat_acc = aligned_imu[0]
    fwd_acc = aligned_imu[1]
    vert_acc = aligned_imu[2]
    yaw_rate = aligned_imu[3]

    lat0, lon0, alt0 = lats[0], lons[0], alts[0]
    earthR = 6378137.0
    gnss_enu = np.zeros((min_len, 2))
    for k in range(min_len):
        d_lat = math.radians(lats[k] - lat0)
        d_lon = math.radians(lons[k] - lon0)
        gnss_enu[k, 0] = earthR * d_lon * math.cos(math.radians(lat0))
        gnss_enu[k, 1] = earthR * d_lat

    dx = gnss_enu[min(50, min_len - 1), 0] - gnss_enu[0, 0]
    dy = gnss_enu[min(50, min_len - 1), 1] - gnss_enu[0, 1]
    initial_theta = math.atan2(dy, dx)

    pitch_phys = compute_physical_pitch_series(lat_acc, fwd_acc, vert_acc, wy=aligned_imu[4], wx=aligned_imu[5], wz=yaw_rate, dt=0.1)
    W = 48

    ai_speed = np.zeros(min_len)
    ai_var = np.ones(min_len) * 0.5
    ai_dv = np.zeros(min_len)
    ai_zupt = np.zeros(min_len)
    ai_sigma = np.ones(min_len)

    speed_model = DeepSpeedKinematicsNet(in_channels=18, window_size=48)
    speed_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    speed_model.eval()

    all_feats = [compute_18ch_features(aligned_imu[:, k - W:k], pitch_phys[k - W:k]) for k in range(W, min_len)]
    feat_tensor = torch.from_numpy(np.stack(all_feats, axis=0))

    # Sequential Closed-Loop State Rollout (zero GT leakage, starts at v0=0)
    v_state_val = torch.tensor([0.0], dtype=torch.float32)

    with torch.no_grad():
        for k_idx in range(len(feat_tensor)):
            bx_sp = feat_tensor[k_idx:k_idx + 1]
            out_sp = speed_model(bx_sp, v_anchor=v_state_val)

            k_curr = W + k_idx
            mu_t = out_sp["mu_v"].cpu().item()
            ai_speed[k_curr] = mu_t
            ai_var[k_curr] = out_sp["var_v"].cpu().item()
            ai_sigma[k_curr] = out_sp["sigma_v"].cpu().item()
            ai_dv[k_curr] = out_sp["delta_v"].cpu().item()
            ai_zupt[k_curr] = out_sp["p_zupt"].cpu().item()

            # Closed-loop update for next step
            v_state_val = torch.tensor([mu_t], dtype=torch.float32)

    val_v_errors = np.abs(ai_speed[W:] - gt_speed_mps[W:]) * 3.6
    gt_kmh = gt_speed_mps[W:] * 3.6
    pred_kmh = ai_speed[W:] * 3.6

    mae_total = np.mean(val_v_errors)
    rmse_total = np.sqrt(np.mean(val_v_errors ** 2))
    bias_total = np.mean(pred_kmh - gt_kmh)
    corr_total = np.corrcoef(pred_kmh, gt_kmh)[0, 1]

    bins = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 80), (80, 200)]
    bin_names = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-80", "80+"]
    bin_mae_dict = {}
    for (b_low, b_high), bn in zip(bins, bin_names):
        mask = (gt_kmh >= b_low) & (gt_kmh < b_high)
        bin_mae_dict[bn] = np.mean(val_v_errors[mask]) if np.sum(mask) > 0 else 0.0

    balanced_mae = np.mean(list(bin_mae_dict.values()))
    seq_diffs = np.abs(np.diff(pred_kmh))
    acc_corr = np.corrcoef(ai_dv[W:], fwd_acc[W:])[0, 1]

    # Corrected EKF Implementation
    def run_ekf(use_ai=True, use_map=False, start_k=0, end_k=min_len, force_v0_zero=True):
        dt = 0.1
        gravity = 9.80665
        q_vel = 0.05
        q_pos = 0.01
        r_nhc = 0.05 * 0.05
        r_zupt = 0.01 * 0.01

        pos = np.array([gnss_enu[start_k, 0], gnss_enu[start_k, 1], 0.0], dtype=np.float64)
        v0_mag = 0.0 if force_v0_zero else (gt_speed_mps[start_k] if use_ai else 0.0)
        vel = np.array([v0_mag * math.cos(initial_theta), v0_mag * math.sin(initial_theta), 0.0], dtype=np.float64)
        att_z = float(initial_theta)
        p_vel_var = 0.5
        p_pos_var = 4.0

        graph = OsmRoadGraph()
        graph.load_from_waypoints(gnss_enu[:, :2])
        hmm = HmmMapMatcher(graph) if use_map else None
        pos_hist = np.zeros((end_k - start_k, 2))

        for idx, k in enumerate(range(start_k, end_k)):
            ax_b = lat_acc[k]
            ay_b = fwd_acc[k]
            az_b = vert_acc[k]
            gz_b = yaw_rate[k]

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

            pos_hist[idx] = pos[:2]
        return pos_hist

    # Full Drive Evaluation
    pos_full = run_ekf(use_ai=True, use_map=False, start_k=0, end_k=min_len, force_v0_zero=True)
    drift_full = np.linalg.norm(pos_full - gnss_enu, axis=1)

    # SIH GNSS Blackout Evaluations
    outage_results = {}
    outage_durations = [10, 30, 60, 90, 120]
    blackout_start = min(500, min_len // 4)  # 50 seconds in

    for dur in outage_durations:
        dur_samples = dur * 10
        end_k = min(min_len, blackout_start + dur_samples)
        pos_out = run_ekf(use_ai=True, use_map=False, start_k=blackout_start, end_k=end_k, force_v0_zero=False)
        ref_seg = gnss_enu[blackout_start:end_k]
        
        # Position drift at blackout termination
        final_pos_err = np.linalg.norm(pos_out[-1] - ref_seg[-1])
        # Traveled distance during blackout
        dist_travelled = np.sum(np.linalg.norm(np.diff(ref_seg, axis=0), axis=1))
        drift_pct = (final_pos_err / max(1.0, dist_travelled)) * 100.0
        
        outage_results[dur] = {
            "drift_m": final_pos_err,
            "dist_m": dist_travelled,
            "drift_pct": drift_pct,
            "passed_sih": drift_pct < 10.0
        }

    # 1 km Outage Test
    cum_dist = np.cumsum(np.linalg.norm(np.diff(gnss_enu, axis=0), axis=1))
    target_1km = cum_dist[blackout_start] + 1000.0
    k_1km = np.searchsorted(cum_dist, target_1km) if blackout_start < len(cum_dist) else min_len
    k_1km = min(min_len, max(blackout_start + 100, k_1km))
    pos_1km = run_ekf(use_ai=True, use_map=False, start_k=blackout_start, end_k=k_1km, force_v0_zero=False)
    ref_1km = gnss_enu[blackout_start:k_1km]
    final_1km_err = np.linalg.norm(pos_1km[-1] - ref_1km[-1])
    dist_1km = np.sum(np.linalg.norm(np.diff(ref_1km, axis=0), axis=1))
    drift_1km_pct = (final_1km_err / max(1.0, dist_1km)) * 100.0

    outage_results["1km"] = {
        "drift_m": final_1km_err,
        "dist_m": dist_1km,
        "drift_pct": drift_1km_pct,
        "passed_sih": drift_1km_pct < 10.0
    }

    # Observer Dynamics Metrics
    v_err_1s = val_v_errors[:10].mean() if len(val_v_errors) >= 10 else 0.0
    v_err_2s = val_v_errors[:20].mean() if len(val_v_errors) >= 20 else 0.0
    v_err_5s = val_v_errors[:50].mean() if len(val_v_errors) >= 50 else 0.0

    # Uncertainty Calibration Metrics in 60-80 and 80+
    mask_60_80 = (gt_kmh >= 60.0) & (gt_kmh < 80.0)
    mask_80 = (gt_kmh >= 80.0)
    sig_60_80 = ai_sigma[W:][mask_60_80].mean() if np.sum(mask_60_80) > 0 else 0.0
    sig_80 = ai_sigma[W:][mask_80].mean() if np.sum(mask_80) > 0 else 0.0
    unc_corr_80 = np.corrcoef(ai_sigma[W:][mask_80], val_v_errors[mask_80])[0, 1] if np.sum(mask_80) > 5 else 0.0

    results = {
        "drive_name": drive_name,
        "duration_s": min_len * 0.1,
        "samples": min_len,
        "overall_mae": mae_total,
        "balanced_mae": balanced_mae,
        "rmse": rmse_total,
        "bias": bias_total,
        "pearson_r": corr_total,
        "bin_maes": bin_mae_dict,
        "jitter_mean": seq_diffs.mean(),
        "jitter_max": seq_diffs.max(),
        "acc_corr": acc_corr,
        "full_drift_final": drift_full[-1],
        "full_drift_max": np.max(drift_full),
        "v_err_1s": v_err_1s,
        "v_err_2s": v_err_2s,
        "v_err_5s": v_err_5s,
        "outages": outage_results,
        "sig_60_80": sig_60_80,
        "sig_80": sig_80,
        "unc_corr_80": unc_corr_80,
        "pred_kmh": pred_kmh,
        "gt_kmh": gt_kmh,
    }
    return results

if __name__ == "__main__":
    cp = "ml/weights/exp6a_best_spectral_speed_filter.pt"
    if not os.path.exists(cp):
        cp = "ml/weights/best_spectral_speed_filter.pt"

    s3a_s = "ml/external/IO-VNBD_repo/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S3a/S-S3a.csv"
    s3a_v = "ml/external/IO-VNBD_repo/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S3a/V-S3a.csv"

    vw11_s = "ml/external/IO-VNBD_repo/Synchronised V abd S datasets/Categorised IOVNB Dataset/Vw (Driver E)/Vw11/S-Vw11.csv"
    vw11_v = "ml/external/IO-VNBD_repo/Synchronised V abd S datasets/Categorised IOVNB Dataset/Vw (Driver E)/Vw11/V-Vw11.csv"

    print("Running Experiment 6A Evaluation on S3a...")
    res_s3a = evaluate_drive_closed_loop(s3a_s, s3a_v, cp, "Driver A S3a")
    print("Running Experiment 6A Evaluation on Driver E Vw11...")
    res_vw11 = evaluate_drive_closed_loop(vw11_s, vw11_v, cp, "Driver E Vw11")

    print("\n" + "="*90)
    print("EXPERIMENT 6A MASTER BENCHMARK REPORT")
    print("="*90)
    print(f"{'Metric':<35} | {'Driver A S3a':<22} | {'Driver E Vw11':<22}")
    print("-"*90)
    print(f"{'Velocity MAE':<35} | {res_s3a['overall_mae']:>6.2f} km/h              | {res_vw11['overall_mae']:>6.2f} km/h")
    print(f"{'Balanced 8-Bin MAE':<35} | {res_s3a['balanced_mae']:>6.2f} km/h              | {res_vw11['balanced_mae']:>6.2f} km/h")
    print(f"{'0-10 km/h MAE':<35} | {res_s3a['bin_maes']['0-10']:>6.2f} km/h              | {res_vw11['bin_maes']['0-10']:>6.2f} km/h")
    print(f"{'60-80 km/h MAE':<35} | {res_s3a['bin_maes']['60-80']:>6.2f} km/h              | {res_vw11['bin_maes']['60-80']:>6.2f} km/h")
    print(f"{'80+ km/h MAE':<35} | {res_s3a['bin_maes']['80+']:>6.2f} km/h              | {res_vw11['bin_maes']['80+']:>6.2f} km/h")
    print(f"{'High-Speed (80+) Signed Bias':<35} | {res_s3a['bias']:>+6.2f} km/h              | {res_vw11['bias']:>+6.2f} km/h")
    print(f"{'Pearson r':<35} | {res_s3a['pearson_r']:>6.3f}                    | {res_vw11['pearson_r']:>6.3f}")
    print(f"{'10 Hz Step Jitter (Mean)':<35} | {res_s3a['jitter_mean']:>6.3f} km/h              | {res_vw11['jitter_mean']:>6.3f} km/h")
    print(f"{'Acceleration Correlation':<35} | {res_s3a['acc_corr']:>+6.4f}                    | {res_vw11['acc_corr']:>+6.4f}")
    print(f"{'Corrected EKF Final DR Drift':<35} | {res_s3a['full_drift_final']:>7.2f} m             | {res_vw11['full_drift_final']:>7.2f} m")
    print(f"{'Corrected EKF Max Peak Drift':<35} | {res_s3a['full_drift_max']:>7.2f} m             | {res_vw11['full_drift_max']:>7.2f} m")

    print("\n" + "="*90)
    print("SIH GNSS BLACKOUT BENCHMARK (<10% DRIFT REQUIREMENT)")
    print("="*90)
    print(f"{'Outage Scenario':<18} | {'Dist Travelled':<15} | {'S3a Drift (m / %)':<20} | {'Driver E Drift (m / %)':<22} | {'SIH Status':<10}")
    print("-"*90)
    for dur in [10, 30, 60, 90, 120, "1km"]:
        label = f"{dur}s Blackout" if isinstance(dur, int) else "1 km Blackout"
        s3a_d = res_s3a['outages'][dur]
        vw_d = res_vw11['outages'][dur]
        s_str = f"{s3a_d['drift_m']:5.1f}m ({s3a_d['drift_pct']:4.1f}%)"
        v_str = f"{vw_d['drift_m']:5.1f}m ({vw_d['drift_pct']:4.1f}%)"
        status = "PASSED" if (s3a_d['passed_sih'] and vw_d['passed_sih']) else "PARTIAL"
        print(f"{label:<18} | {s3a_d['dist_m']:>6.1f} m        | {s_str:<20} | {v_str:<22} | {status}")
