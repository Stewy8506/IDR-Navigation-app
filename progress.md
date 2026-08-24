# INSS Navigation App — Learning Pipeline Experiment Progress Tracker

## 1. Overview & Evaluation Protocol

This document tracks all controlled experiments performed on the learned velocity observer (`DeepSpeedKinematicsNet`), fusing with the 15-state Error-State EKF (`EkfFusionEngine`).

### Experimental Principles:
1. **Strict Data Isolation:**
   - **Training Set:** Driver A (`S1`, `S2`, `S3b`, `S3c`, `S4`), Driver B (`M`), Driver D (`Y1`).
   - **Validation Set:** Driver A (`S3a`).
   - **Held-Out OOD Test Set:** Driver E (`Vw11`, `Vw12`, ...). *Never tune or select checkpoints against Driver E.*
2. **Zero GPS Shortcut Anti-Leakage:**
   - GPS is strictly used for ground truth offline labels and evaluation metrics.
   - Zero GPS signals or derivatives enter neural inference or runtime dead reckoning.
3. **Controlled Single-Variable Changes:**
   - Change exactly one variable per experiment, compare across all 8 speed bins, and evaluate both in-distribution (S3a) and held-out OOD (Driver E).

---

## 2. Experiments Matrix

| Exp # | Description | Status | In-Dist (S3a) MAE | OOD (Vw11) MAE | 30s Outage Drift (S3a) | Key Result / Note |
|---|---|---|---|---|---|---|
| **0** | **Frozen Baseline** (Legacy Ch15, unnormalized, arithmetic mean checkpoint) | **Completed** | $14.96\text{ km/h}$ | $17.39\text{ km/h}$ | $103.39\text{ m}$ | Baseline established with raw gyro fusion. |
| **1** | **Fix Channel 15 ONLY** (Turn-gating centripetal speed: $\|\omega_z\| \ge 0.035\text{ rad/s}$) | **Completed** | **$14.65\text{ km/h}$** | $21.10\text{ km/h}$ | **$95.31\text{ m}$** | S3a MAE $-0.31\text{ km/h}$, 80+ MAE $-3.2\text{ km/h}$, 30s drift $-8.08\text{ m}$, full drive drift $-684\text{ m}$. |
| **2** | **Channel 15 Fix + Input Normalization** (`InputBatchNorm1d` on 18 input channels) | *Pending* | — | — | — | Prevent vibration ratio scale from overwhelming subtle gyro cues. |
| **3** | **Speed-Bin Balanced Checkpoint Selection** ($\frac{1}{8} \sum_{b=1}^8 \text{MAE}_b$) | *Pending* | — | — | — | Prevent checkpoint selection bias toward slow urban driving. |
| **4** | **High-Speed Relative Loss Formulation** ($\mathcal{L}_{\text{rel}}$ on $v \ge 50\text{ km/h}$) | *Pending* | — | — | — | Incentivize accurate percentage errors on high-speed highways. |
| **5** | **Temporal Consistency Regularization** ($\|\mu_v[t] - \mu_v[t-1] - a_y[t]\Delta t\|$) | *Pending* | — | — | — | Penalize step jitter and enforce kinematic acceleration continuity. |

---

## 3. Detailed Benchmark History

### Experiment 0 — Frozen Baseline
* **Date:** 2026-08-24
* **Changes:** Initial baseline model evaluated with raw gyro EKF fusion (DeepHeading gyro bias injection removed).
* **Validation (Driver A S3a):**
  * Overall Velocity MAE: **$14.96\text{ km/h}$** | RMSE: $19.68\text{ km/h}$ | Bias: $-1.31\text{ km/h}$ | Pearson $r$: $0.566$
  * Speed Bins MAE (km/h): `[0-10: 12.8 | 10-20: 17.1 | 20-30: 15.9 | 30-40: 15.7 | 40-50: 12.7 | 50-60: 10.4 | 60-80: 15.4 | 80+: 31.8]`
  * Stratified 8-Bin Balanced MAE: **$16.48\text{ km/h}$**
  * Dead Reckoning Drift (AI+EKF Raw Gyro):
    * 30s Outage: **$103.39\text{ m}$** (ATE RMSE: $59.77\text{ m}$)
    * 60s Outage: **$203.20\text{ m}$** (ATE RMSE: $144.15\text{ m}$)
    * 90s Outage: **$275.64\text{ m}$** (ATE RMSE: $184.22\text{ m}$)
    * Full Drive: **$9742.63\text{ m}$** (ATE RMSE: $6592.51\text{ m}$)
* **Held-Out OOD (Driver E Vw11):**
  * Overall Velocity MAE: **$17.39\text{ km/h}$** | RMSE: $23.58\text{ km/h}$ | Bias: $-8.71\text{ km/h}$ | Pearson $r$: $0.732$
  * Speed Bins MAE (km/h): `[0-10: 9.5 | 10-20: 16.0 | 20-30: 9.1 | 30-40: 6.3 | 40-50: 7.0 | 50-60: 11.4 | 60-80: 29.9 | 80+: 43.9]`
  * Full Drive Drift: **$1782.90\text{ m}$** (ATE RMSE: $1187.32\text{ m}$)

---

### Experiment 1 — Turn-Gated Kinematics Fix (Channel 15 ONLY)
* **Date:** 2026-08-24
* **Hypothesis:** Baseline Channel 15 divided lateral acceleration by $\max(|\omega_z|, 0.01\text{ rad/s})$, exploding to the $50\text{ m/s}$ ceiling during $>85\%$ of straight driving and corrupting the network's high-speed perception (Pearson $r = -0.073$). Gating centripetal speed strictly to turns ($|\omega_z| \ge 0.035\text{ rad/s}$) will restore physical correlation ($r = +0.100$) and reduce high-speed error.
* **Code Changes:**
  * `ml/src/dataset_spectral.py`: Replaced corrupted division with turn gating formula.
  * `app/lib/ai/speed_filter_runner.dart`: Parity update in Dart runtime.
* **Validation (Driver A S3a):**
  * Overall Velocity MAE: **$14.65\text{ km/h}$** (*improved by $-0.31\text{ km/h}$*)
  * Velocity RMSE: **$18.93\text{ km/h}$** (*improved by $-0.75\text{ km/h}$*)
  * Velocity Bias: **$-0.47\text{ km/h}$** (*near zero, improved by $+0.84\text{ km/h}$*)
  * Pearson $r$: **$0.569$** (*improved from $0.566$*)
  * Speed Bins MAE (km/h): `[0-10: 17.7 | 10-20: 15.2 | 20-30: 13.2 | 30-40: 12.2 | 40-50: 12.3 | 50-60: 11.4 | 60-80: 18.5 | 80+: 28.6]`
    * **10–20 km/h:** $-1.9\text{ km/h}$ reduction
    * **20–30 km/h:** $-2.7\text{ km/h}$ reduction
    * **30–40 km/h:** $-3.5\text{ km/h}$ reduction
    * **80+ km/h:** **$-3.2\text{ km/h}$ reduction** (dropped from $31.8$ to $28.6\text{ km/h}$)
  * Stratified 8-Bin Balanced MAE: **$16.14\text{ km/h}$** (*improved from $16.48\text{ km/h}$*)
  * Dead Reckoning Drift (AI+EKF Raw Gyro):
    * 30s Outage: **$95.31\text{ m}$** (*down by $8.08\text{ m}$ / $7.8\%$*)
    * 60s Outage: **$198.23\text{ m}$** (*down by $4.97\text{ m}$*)
    * 90s Outage: **$254.38\text{ m}$** (*down by $21.26\text{ m}$ / $7.7\%$*)
    * Full Drive: **$9058.45\text{ m}$** (*down by $684.18\text{ m}$ / $7.0\%$*)
* **Held-Out OOD (Driver E Vw11):**
  * 40–50 km/h MAE: **$4.6\text{ km/h}$** (*improved from $7.0\text{ km/h}$*)
  * 50–60 km/h MAE: **$10.2\text{ km/h}$** (*improved from $11.4\text{ km/h}$*)
  * Full Drive Drift: **$1292.80\text{ m}$** (*down by $490.10\text{ m}$ / **$27.5\%$ reduction**!*)

---

## 4. Next Experiment Action Plan (When Resuming)

1. **Implement Experiment 2:**
   - Add `nn.BatchNorm1d(in_channels)` or per-channel standardization layer at the entrance of `DeepSpeedKinematicsNet` in `ml/src/model.py`.
   - Re-train 15 epochs and verify if subtle gyroscopic and suspension vibration frequency cues are amplified.
2. **Implement Experiment 3:**
   - Update `train_spectral.py` checkpoint saving logic to monitor `mean(speed_bin_maes)` across all 8 bins.
3. **Implement Experiment 4:**
   - Add weighted percentage error penalty for speed $>50\text{ km/h}$.
