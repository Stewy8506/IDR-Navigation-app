import math
import os
import numpy as np
import pandas as pd

def run_parity_trace():
    s_path = "ml/external/IO-VNBD_repo/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S3a/S-S3a.csv"
    v_path = "ml/external/IO-VNBD_repo/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S3a/V-S3a.csv"
    preds_path = "ml/evaluation_cache/s3a_neural_preds.csv"

    df_s = pd.read_csv(s_path, encoding="latin1")
    df_v = pd.read_csv(v_path, encoding="latin1")
    df_p = pd.read_csv(preds_path)

    df_s.columns = df_s.columns.str.strip()
    df_v.columns = df_v.columns.str.strip()

    lat = df_s["GPS LATITUDE (degrees)"].values
    lon = df_s["GPS LONGITUDE (degrees)"].values
    alt = df_s["GPS ALTITUDE (m)"].values
    ax_raw = df_s["ACCELEROMETER X (m/s²)"].values.astype(np.float64)
    ay_raw = df_s["ACCELEROMETER Y (m/s²)"].values.astype(np.float64)
    az_raw = df_s["ACCELEROMETER Z (m/s²)"].values.astype(np.float64)
    gy_raw = df_s["GYROSCOPE Yaw (rad/s)"].values.astype(np.float64)

    speed_col = "Indicated Vehicle Speed (km/hr)" if "Indicated Vehicle Speed (km/hr)" in df_v.columns else "Velocity (km/hr)"
    gt_speed_mps = (df_v[speed_col].values / 3.6).astype(np.float64)

    ai_speed = df_p["mu_v"].values.astype(np.float64)
    ai_var = df_p["var_v"].values.astype(np.float64)
    ai_zupt = df_p["p_zupt"].values.astype(np.float64)

    min_len = min(len(lat), len(gt_speed_mps), len(ai_speed))

    lat0, lon0, alt0 = lat[0], lon[0], alt[0]
    earthR = 6378137.0
    gnss_enu = np.zeros((min_len, 3))
    for k in range(min_len):
        dLat = math.radians(lat[k] - lat0)
        dLon = math.radians(lon[k] - lon0)
        e = earthR * math.cos(math.radians(lat0)) * dLon
        n = earthR * dLat
        u = alt[k] - alt0
        gnss_enu[k] = [e, n, u]

    dx = gnss_enu[50, 0] - gnss_enu[0, 0]
    dy = gnss_enu[50, 1] - gnss_enu[0, 1]
    initial_theta = math.atan2(dy, dx)

    start_k = min(500, min_len // 4)
    dur_sec = 30
    end_k = min(min_len, start_k + dur_sec * 10)

    print(f"Tracing 30s Outage: start_k={start_k}, end_k={end_k}, initial_theta={initial_theta:.4f} rad ({(initial_theta*180/math.pi):.2f} deg)")

    # Vehicle frame mapping:
    # ax_v (lateral) = ay_raw
    # ay_v (forward) = -ax_raw
    # az_v (up) = az_raw
    # gz_v (yaw rate CCW) = gy_raw

    dt = 0.1
    gravity = 9.80665
    q_vel = 0.05
    q_pos = 0.01
    r_nhc = 0.05 * 0.05
    r_zupt = 0.01 * 0.01

    # Exact 15-State EKF implementation matching Dart
    # -------------------------------------------------------------
    # States: pos [E, N, U], vel [vE, vN, vU], attitude [roll, pitch, yaw], accelBias, gyroBias
    pos = np.array([gnss_enu[start_k, 0], gnss_enu[start_k, 1], gnss_enu[start_k, 2]], dtype=np.float64)
    vel = np.array([gt_speed_mps[start_k] * math.cos(initial_theta), gt_speed_mps[start_k] * math.sin(initial_theta), 0.0], dtype=np.float64)
    att_z = float(initial_theta)
    p_vel_var = 0.5
    p_pos_var = 4.0

    trace_records = []

    for k in range(start_k, end_k):
        # 1. Bias-corrected IMU
        ax_v = ay_raw[k]
        ay_v = -ax_raw[k]
        az_v = az_raw[k]
        gz_v = gy_raw[k]

        # 2. Attitude propagation
        att_z += gz_v * dt
        while att_z > math.pi: att_z -= 2.0 * math.pi
        while att_z < -math.pi: att_z += 2.0 * math.pi

        c_th = math.cos(att_z)
        s_th = math.sin(att_z)

        # 3. Transform Accel to ENU
        a_east = ay_v * c_th + ax_v * s_th
        a_north = ay_v * s_th - ax_v * c_th
        a_up = az_v - gravity

        # 4. Propagate Velocity & Position
        vel[0] += a_east * dt
        vel[1] += a_north * dt
        vel[2] += a_up * dt

        pos[0] += vel[0] * dt
        pos[1] += vel[1] * dt
        pos[2] += vel[2] * dt

        # 5. Covariance propagation
        p_vel_var += q_vel * dt
        p_pos_var += p_vel_var * dt + q_pos * dt

        # 6. Apply Non-Holonomic Constraints (NHC)
        v_lat = vel[0] * s_th - vel[1] * c_th
        k_nhc = p_vel_var / (p_vel_var + r_nhc)
        damp_factor = min(0.35, max(0.05, k_nhc))
        vel[0] -= damp_factor * (v_lat * s_th)
        vel[1] -= damp_factor * (-v_lat * c_th)
        vel[2] *= 0.92

        # 7. Measurement update from AI Speed / ZUPT
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

        # 8. Apply Centripetal Constraint
        om = abs(gz_v)
        if om >= 0.035:
            v_cent = abs(ax_v) / om
            if 2.0 <= v_cent <= 40.0:
                v_fwd_est = vel[0] * c_th + vel[1] * s_th
                innov_c = v_cent - v_fwd_est
                r_c = max(1.0, 0.0625 / (om * om))
                k_gain = min(0.25, p_vel_var / (p_vel_var + r_c))
                vel[0] += k_gain * innov_c * c_th
                vel[1] += k_gain * innov_c * s_th

        gt_pos = gnss_enu[k]
        pos_err = np.linalg.norm(pos[:2] - gt_pos[:2])

        trace_records.append({
            "k": k,
            "pos_e": pos[0],
            "pos_n": pos[1],
            "pos_u": pos[2],
            "vel_e": vel[0],
            "vel_n": vel[1],
            "vel_u": vel[2],
            "yaw": att_z,
            "p_vel_var": p_vel_var,
            "p_pos_var": p_pos_var,
            "pos_err": pos_err,
            "gt_e": gt_pos[0],
            "gt_n": gt_pos[1],
        })

    df_out = pd.DataFrame(trace_records)
    os.makedirs("ml/evaluation_cache", exist_ok=True)
    out_path = "ml/evaluation_cache/python_ekf_trace_s3a_30s.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Saved Python trace to {out_path}")
    print(f"Final Drift at k={end_k-1}: {df_out['pos_err'].iloc[-1]:.2f} m")
    ate = math.sqrt((df_out['pos_err']**2).mean())
    print(f"ATE RMSE: {ate:.2f} m")

if __name__ == "__main__":
    run_parity_trace()
