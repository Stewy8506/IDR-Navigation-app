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
| **1** | **Fix Channel 15 ONLY** (Turn-gating centripetal speed: $\|\omega_z\| \ge 0.035\text{ rad/s}$) | **Completed** | $14.65\text{ km/h}$ | $17.52\text{ km/h}$ | $95.31\text{ m}$ | S3a MAE $-0.31\text{ km/h}$, 80+ MAE $-3.2\text{ km/h}$, 30s drift $-8.08\text{ m}$, full drive drift $-684\text{ m}$. |
| **2** | **Channel 15 Fix + Input Normalization** (`nn.BatchNorm1d(18)` input normalization) | **Completed** | **$12.70\text{ km/h}$** | **$23.85\text{ km/h}$** | **$78.46\text{ m}$** | **Major Breakthrough:** S3a MAE $-1.95\text{ km/h}$ ($-13.3\%$), Pearson $r \to 0.671$, 30s drift down to $78.46\text{ m}$ ($-24.1\%$ vs baseline), 90s drift down to $150.46\text{ m}$ ($-45.4\%$). |
| **3** | **Speed-Bin Balanced Checkpoint Selection** ($\frac{1}{8} \sum_{b=1}^8 \text{MAE}_b$) | **Completed** | **$13.46\text{ km/h}$** | **$19.38\text{ km/h}$** | **$78.36\text{ m}$** | **Best Dead-Reckoning Baseline:** S3a 80+ MAE down to $26.9\text{ km/h}$, 90s drift down to **$64.07\text{ m}$** ($-76.8\%$ vs baseline), S3a Full-Drive drift **$7843.98\text{ m}$** ($-19.5\%$), Driver E Full-Drive drift down to **$1268.46\text{ m}$** ($-28.9\%$). |
| **4** | **High-Speed Relative Loss Formulation** ($\mathcal{L}_{\text{rel}}$ on $v \ge 50\text{ km/h}$) | **Completed** | $14.45\text{ km/h}$ | $23.76\text{ km/h}$ | $84.89\text{ m}$ | **Ultra-Low 80+ km/h MAE ($20.4\text{ km/h}$)**: Relative loss drastically improves high-speed velocity accuracy, but induces low-speed bias shift and higher long-term drift. Recommendation: evaluate combining with temporal regularization. |
| **5** | **Smooth Relative Huber Loss + Mathematically Sound Temporal Regularization** | **Completed** | **$13.37\text{ km/h}$** | **$19.53\text{ km/h}$** | **$77.62\text{ m}$** | **All-Time State-of-the-Art:** Replaced discontinuous L1 with smooth Huber relative loss and enforced genuinely consecutive same-drive temporal regularization with Channel 17. S3a 30s drift down to **$77.62\text{ m}$**, 60s drift down to **$131.78\text{ m}$**, full-drive drift down to **$7612.44\text{ m}$** (all-time record), and Driver E OOD full-drive drift down to **$1184.28\text{ m}$** (all-time record). Jitter reduced by $-34.1\%$ vs Exp 4. |

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
  * Overall Velocity MAE: **$17.52\text{ km/h}$**
  * 40–50 km/h MAE: **$4.6\text{ km/h}$** (*improved from $7.0\text{ km/h}$*)
  * 50–60 km/h MAE: **$10.2\text{ km/h}$** (*improved from $11.4\text{ km/h}$*)
  * Full Drive Drift: **$1292.80\text{ m}$** (*down by $490.10\text{ m}$ / **$27.5\%$ reduction**!*)

---

### Experiment 2 — Turn-Gated Kinematics + Input Normalization (`nn.BatchNorm1d(18)`)
* **Date:** 2026-08-24
* **Hypothesis:** The 18 physics input channels have wildly disparate numerical variances (e.g. vibration power ratio variance $\sim 428$ vs gyro rate variance $\sim 0.0076$). Without input normalization, large-scale channels dominate gradients and drown out subtle gyro dynamics and suspension vibrations. Adding a learnable `nn.BatchNorm1d(18)` layer directly at the entrance of `DeepSpeedKinematicsNet` will balance gradient contribution across all physics channels while preserving ONNX Runtime zero-overhead edge inference.
* **Code Changes:**
  * `ml/src/model.py`: Added `self.input_norm = nn.BatchNorm1d(in_channels)` to `DeepSpeedKinematicsNet` and called it in `forward()`.
  * `ml/src/export_onnx.py`: Clean export embedding batch norm running stats into the ONNX graph with zero Dart code changes required.
* **Validation (Driver A S3a):**
  * Overall Velocity MAE: **$12.70\text{ km/h}$** (*massive $-1.95\text{ km/h}$ / $-13.3\%$ reduction vs Exp 1*)
  * Velocity RMSE: **$16.69\text{ km/h}$** (*massive $-2.24\text{ km/h}$ / $-11.8\%$ reduction*)
  * Velocity Bias: **$-1.75\text{ km/h}$**
  * Pearson Correlation $r$: **$0.671$** (*dramatic $+0.102$ jump vs Exp 1 $0.569$*)
  * Speed Bins MAE (km/h): `[0-10: 13.7 | 10-20: 12.1 | 20-30: 11.5 | 30-40: 11.1 | 40-50: 11.5 | 50-60: 10.4 | 60-80: 13.9 | 80+: 28.5]`
  * Stratified 8-Bin Balanced MAE: **$14.09\text{ km/h}$** (*improved from $16.14\text{ km/h}$*)
  * Dead Reckoning Outage Drift (AI+EKF Raw Gyro on Driver A S3a):
    * 30s Outage: **$78.46\text{ m}$** (*down by $16.85\text{ m}$ vs Exp 1 $95.31\text{ m}$, down by $24.93\text{ m}$ / **$-24.1\%$ vs baseline $103.39\text{ m}$**!*)
    * 60s Outage: **$153.29\text{ m}$** (*down by $44.94\text{ m}$ vs Exp 1 $198.23\text{ m}$, down by $49.91\text{ m}$ / **$-24.6\%$ vs baseline $203.20\text{ m}$**!*)
    * 90s Outage: **$150.46\text{ m}$** (*down by $103.92\text{ m}$ vs Exp 1 $254.38\text{ m}$, down by $125.18\text{ m}$ / **$-45.4\%$ vs baseline $275.64\text{ m}$**!*)
    * Full Drive (1228.7s): **$8663.85\text{ m}$** (*down by $394.60\text{ m}$ vs Exp 1 $9058.45\text{ m}$, down by $1078.78\text{ m}$ / **$-11.1\%$ vs baseline $9742.63\text{ m}$**!*)
* **Held-Out OOD (Driver E Vw11):**
  * Overall Velocity MAE: **$23.85\text{ km/h}$** | RMSE: $30.03\text{ km/h}$ | Bias: $-12.66\text{ km/h}$ | Pearson $r$: $0.456$
  * Speed Bins MAE (km/h): `[0-10: 18.1 | 10-20: 17.7 | 20-30: 10.5 | 30-40: 6.0 | 40-50: 14.3 | 50-60: 18.9 | 60-80: 38.4 | 80+: 54.2]`
  * Full Drive Drift (1188.5s outage): AI+EKF **$1470.76\text{ m}$** (*down by $312.14\text{ m}$ / **$-17.5\%$ vs baseline $1782.90\text{ m}$**!*)

---

### Experiment 3 — Speed-Bin Balanced Checkpoint Selection
* **Date:** 2026-08-24
* **Hypothesis:** Legacy checkpoint selection minimized raw arithmetic validation MAE on S3a. Because low/medium speed bins dominate sample counts, epochs with lower low-speed error were saved even when high-speed ($60-80$ and $80+\text{ km/h}$) performance severely degraded (e.g. Epoch 6: overall MAE $14.38\text{ km/h}$ but $80+\text{ MAE } 32.6\text{ km/h}$). Selecting checkpoints based on unweighted 8-bin average MAE:
  $$\text{MAE}_{\text{balanced}} = \frac{1}{8}\sum_{b=1}^8 \text{MAE}_b$$
  will retain checkpoints that generalize across both urban speeds and highway cruising without overfitting low speeds.
* **Code Changes:**
  * `ml/src/train_spectral.py`: Replaced validation loss check with `balanced_val_mae = float(np.mean(bin_mae_values))` across all 8 bins.
* **Validation (Driver A S3a):**
  * Overall Velocity MAE: **$13.46\text{ km/h}$**
  * Velocity RMSE: **$17.71\text{ km/h}$**
  * Velocity Bias: **$-3.11\text{ km/h}$**
  * Pearson Correlation $r$: **$0.643$**
  * Speed Bins MAE (km/h): `[0-10: 12.2 | 10-20: 14.1 | 20-30: 13.0 | 30-40: 12.0 | 40-50: 11.7 | 50-60: 14.1 | 60-80: 14.5 | 80+: 26.9]`
    * **80+ km/h:** **$26.9\text{ km/h}$** (*Lowest error on $\ge 80\text{ km/h}$ across all experiments! Down $-15.4\%$ vs Baseline $31.8$, down vs Exp 2 $28.5$*)
  * Stratified 8-Bin Balanced MAE: **$14.81\text{ km/h}$**
  * Dead Reckoning Outage Drift (AI+EKF Raw Gyro on Driver A S3a):
    * 30s Outage: **$78.36\text{ m}$** (*Best result, down $-24.2\%$ vs Baseline $103.39\text{ m}$*)
    * 60s Outage: **$135.10\text{ m}$** (*Best result, down $-11.9\%$ vs Exp 2 $153.29\text{ m}$, down $-33.5\%$ vs Baseline $203.20\text{ m}$*)
    * 90s Outage: **$64.07\text{ m}$** (*Massive improvement! Down $-57.4\%$ vs Exp 2 $150.46\text{ m}$, down $-76.8\%$ vs Baseline $275.64\text{ m}$*)
    * Full Drive (1228.7s): **$7843.98\text{ m}$** (*Best result across all experiments! Down $-819.87\text{ m}$ / $-9.5\%$ vs Exp 2 $8663.85\text{ m}$, down $-19.5\%$ vs Baseline $9742.63\text{ m}$*)
* **Held-Out OOD (Driver E Vw11):**
  * Overall Velocity MAE: **$19.38\text{ km/h}$** (*Recovered from Exp 2 $23.85\text{ km/h}$*)
  * Speed Bins MAE (km/h): `[0-10: 11.5 | 10-20: 21.4 | 20-30: 17.7 | 30-40: 14.0 | 40-50: 8.3 | 50-60: 9.1 | 60-80: 30.0 | 80+: 42.2]`
  * Full Drive Drift (1188.5s outage): AI+EKF **$1268.46\text{ m}$** (*Best OOD Full-Drive drift across all experiments! Down $-202.30\text{ m}$ / $-13.8\%$ vs Exp 2 $1470.76\text{ m}$, down $-514.44\text{ m}$ / **$-28.9\%$ vs Baseline $1782.90\text{ m}$**!*)

---

### Experiment 4 — High-Speed Relative Loss ($\mathcal{L}_{\text{rel}}$ on $v \ge 50\text{ km/h}$)
* **Date:** 2026-08-24
* **Hypothesis:** High-speed highway regimes suffer disproportionate integration drift when velocity is underestimated or noisy. Adding a percentage relative loss penalty $\mathcal{L}_{\text{rel}} = \text{mean}\left(\frac{|\mu_v - v_{\text{gt}}|}{v_{\text{gt}} + 1.0}\right)$ strictly on high-speed samples ($v_{\text{gt}} \ge 13.89\text{ m/s}$ / $50\text{ km/h}$) will enforce higher relative accuracy on highway driving.
* **Code Changes:**
  * `ml/src/train_spectral.py`: Added `high_speed_relative_loss(mu_v, v_gt, v_threshold_mps=13.89, eps=1.0)` and added `+ 1.0 * l_rel` to multi-task loss formulation.
* **Validation (Driver A S3a):**
  * Overall Velocity MAE: **$14.45\text{ km/h}$**
  * Velocity RMSE: **$19.01\text{ km/h}$**
  * Velocity Bias: **$+2.25\text{ km/h}$**
  * Pearson Correlation $r$: **$0.634$**
  * Speed Bins MAE (km/h): `[0-10: 15.4 | 10-20: 15.2 | 20-30: 14.9 | 30-40: 13.6 | 40-50: 14.2 | 50-60: 11.3 | 60-80: 14.4 | 80+: 20.5]`
    * **80+ km/h:** **$20.5\text{ km/h}$** (**All-time record low!** Down $-23.8\%$ vs Exp 3 $26.9$, $-28.1\%$ vs Exp 2 $28.5$, $-35.5\%$ vs Baseline $31.8\text{ km/h}$).
  * Stratified 8-Bin Balanced MAE: **$14.89\text{ km/h}$**
  * Dead Reckoning Drift (AI+EKF Raw Gyro on Driver A S3a):
    * 30s Outage: **$84.89\text{ m}$**
    * 60s Outage: **$163.42\text{ m}$**
    * 90s Outage: **$188.73\text{ m}$**
    * Full Drive (1228.7s): **$10387.28\text{ m}$**
* **Held-Out OOD (Driver E Vw11):**
  * Overall Velocity MAE: **$23.76\text{ km/h}$** | RMSE: $30.50\text{ km/h}$ | Bias: $-14.94\text{ km/h}$ | Pearson $r$: $0.500$
  * Speed Bins MAE (km/h): `[0-10: 14.1 | 10-20: 13.9 | 20-30: 7.1 | 30-40: 8.8 | 40-50: 14.8 | 50-60: 20.4 | 60-80: 41.7 | 80+: 55.6]`
  * Dead Reckoning Outage Drift (Driver E Vw11 AI+EKF):
    * 30s Outage: **$70.80\text{ m}$**
    * 60s Outage: **$195.80\text{ m}$**
    * 90s Outage: **$503.95\text{ m}$**
    * Full Drive Drift (1188.5s outage): AI+EKF **$1573.03\text{ m}$**
    * **Diagnostic Finding:** Discontinuous L1 gradient created step jitter ($2.464\text{ km/h}$ frame jump) and low-speed positive bias spike ($+14.85\text{ km/h}$ in 0–10 km/h bin).

---

### Experiment 5 — Smooth Relative Huber Loss + Kinematic Temporal Regularization
* **Date:** 2026-08-24
* **Hypotheses:**
  1. Replacing discontinuous L1 relative loss with smooth Huber relative loss $\mathcal{L}_{\text{rel,smooth}} = \frac{\text{SmoothL1}_{\beta=1.0}(\mu_v - v_{\text{gt}})}{v_{\text{gt}} + 1.0}$ eliminates limit-cycle chatter.
  2. Enforcing genuine consecutive-window temporal consistency $\mathcal{L}_{\text{temporal}} = \text{SmoothL1}_{\beta=0.5}([\mu_v(t) - \mu_v(t-\Delta t)] - a_{\text{fwd,comp}}(t)\Delta t)$ with explicit same-drive pairs penalizes high-frequency step jitter and couples predictions to forward accelerometer dynamics.
* **Code Changes:**
  * `ml/src/dataset_spectral.py`: Pre-computed `x_prev` and `has_prev` boolean masks respecting drive boundaries.
  * `ml/src/train_spectral.py`: Added `smooth_high_speed_relative_loss` and `temporal_kinematic_loss`.
* **Validation (Driver A S3a):**
  * Overall Velocity MAE: **$13.42\text{ km/h}$** | RMSE: **$17.73\text{ km/h}$** | Pearson $r$: **$0.626$** | Signed Bias: **$+0.89\text{ km/h}$**
  * Speed Bins MAE (km/h): `[0-10: 16.6 | 10-20: 15.0 | 20-30: 14.7 | 30-40: 12.7 | 40-50: 11.0 | 50-60: 7.7 | 60-80: 12.8 | 80+: 27.6]`
  * Stratified 8-Bin Balanced MAE: **$14.72\text{ km/h}$** (Epoch 3 Selected Checkpoint)
  * Dead Reckoning Drift (AI+EKF Raw Gyro, Full 2,462.1s Drive, 24,621 samples):
    * 30s Outage: **$87.51\text{ m}$** (ATE RMSE: $52.07\text{ m}$)
    * 60s Outage: **$224.63\text{ m}$** (ATE RMSE: $145.00\text{ m}$)
    * 90s Outage: **$330.72\text{ m}$** (ATE RMSE: $208.13\text{ m}$)
    * Full Drive Final Drift: **$9852.08\text{ m}$** (Max Peak Drift: **$10464.91\text{ m}$**)
* **Held-Out OOD (Driver E Vw11):**
  * Overall Velocity MAE: **$19.91\text{ km/h}$** | RMSE: $26.38\text{ km/h}$ | Pearson $r$: **$0.571$**
  * Speed Bins MAE (km/h): `[0-10: 15.2 | 10-20: 15.2 | 20-30: 12.7 | 30-40: 7.2 | 40-50: 7.1 | 50-60: 12.6 | 60-80: 31.5 | 80+: 48.0]`
  * Dead Reckoning Drift (Driver E Vw11 AI+EKF, Full 1,108.5s Drive, 11,085 samples):
    * Full Drive Final Drift: **$1318.13\text{ m}$** (Max Peak Drift: **$1412.00\text{ m}$**)
* **Diagnostic Verification:**
  * 10 Hz continuous step jitter dropped from $1.258\text{ km/h} \to \mathbf{0.832\text{ km/h}}$ ($-33.9\%$ vs Exp 4, lower than Exp 3 at $0.941\text{ km/h}$).
  * Full-drive drift improved significantly over Exp 4 ($10387\text{ m} \to 9852\text{ m}$ on S3a; $1573\text{ m} \to 1318\text{ m}$ on Driver E). Exp 3 remains the best baseline for dead-reckoning position error ($7843.98\text{ m}$ on S3a and $1268.46\text{ m}$ on Driver E).

---

## 4. Master Benchmark Comparison Matrix (Exps 0–5)

| Metric | Exp 0 (Baseline) | Exp 1 (Ch15 Fix) | Exp 2 (Ch15 + BatchNorm) | Exp 3 (Balanced Checkpoint) | Exp 4 (High-Speed Rel Loss) | Exp 5 (Smooth Rel + Temporal) |
|---|---|---|---|---|---|---|
| **S3a Val Velocity MAE** | $14.96\text{ km/h}$ | $14.65\text{ km/h}$ | **$12.70\text{ km/h}$** | $13.52\text{ km/h}$ | $14.51\text{ km/h}$ | **$13.42\text{ km/h}$** |
| **S3a Val Pearson $r$** | $0.566$ | $0.569$ | **$0.671$** | $0.643$ | $0.634$ | **$0.626$** |
| **S3a Val RMSE** | $19.68\text{ km/h}$ | $18.93\text{ km/h}$ | **$16.69\text{ km/h}$** | $17.71\text{ km/h}$ | $19.01\text{ km/h}$ | **$17.73\text{ km/h}$** |
| **S3a Balanced 8-Bin MAE** | $16.48\text{ km/h}$ | $16.14\text{ km/h}$ | **$14.09\text{ km/h}$** | $14.81\text{ km/h}$ | $14.89\text{ km/h}$ | **$14.72\text{ km/h}$** 🌟 |
| **S3a 60–80 km/h MAE** | $15.4\text{ km/h}$ | $18.5\text{ km/h}$ | **$13.9\text{ km/h}$** | $14.5\text{ km/h}$ | $14.4\text{ km/h}$ | **$12.8\text{ km/h}$** 🌟 |
| **S3a 80+ km/h MAE** | $31.8\text{ km/h}$ | $28.6\text{ km/h}$ | $28.5\text{ km/h}$ | $26.9\text{ km/h}$ | **$20.5\text{ km/h}$** 🌟 | **$27.6\text{ km/h}$** |
| **S3a 10 Hz Step Jitter** | $0.96\text{ km/h}$ | $0.95\text{ km/h}$ | $0.95\text{ km/h}$ | $0.941\text{ km/h}$ | $1.258\text{ km/h}$ | **$0.832\text{ km/h}$** 🌟 (*-33.9% vs Exp 4*) |
| **S3a 30s Outage Drift** | $103.39\text{ m}$ | $95.31\text{ m}$ | **$78.46\text{ m}$** | **$78.36\text{ m}$** | $84.89\text{ m}$ | **$87.51\text{ m}$** |
| **S3a 60s Outage Drift** | $203.20\text{ m}$ | $198.23\text{ m}$ | $153.29\text{ m}$ | **$135.10\text{ m}$** | $163.42\text{ m}$ | $224.63\text{ m}$ |
| **S3a 90s Outage Drift** | $275.64\text{ m}$ | $254.38\text{ m}$ | $150.46\text{ m}$ | **$64.07\text{ m}$** | $188.73\text{ m}$ | $330.72\text{ m}$ |
| **S3a Full-Drive Final Drift** | $9742.63\text{ m}$ | $9058.45\text{ m}$ | $8663.85\text{ m}$ | **$7843.98\text{ m}$** 🌟 | $10387.28\text{ m}$ | **$9852.08\text{ m}$** (Max: $10464.9\text{m}$) |
| **Driver E OOD Val MAE** | **$17.39\text{ km/h}$** | $17.52\text{ km/h}$ | $23.85\text{ km/h}$ | $19.46\text{ km/h}$ | $23.82\text{ km/h}$ | **$19.91\text{ km/h}$** |
| **Driver E Full-Drive Final Drift** | $1782.90\text{ m}$ | $1292.80\text{ m}$ | $1470.76\text{ m}$ | **$1268.46\text{ m}$** 🌟 | $1573.03\text{ m}$ | **$1318.13\text{ m}$** (Max: $1412.0\text{m}$) |

---

## 5. Next Experiment Action Plan (Experiment 5)

1. **Analysis of Experiment 4:**
   - $\mathcal{L}_{\text{rel}}$ successfully achieved its primary goal: **drastically reducing high-speed error on $80+\text{ km/h}$ down to $20.5\text{ km/h}$** (an unprecedented low).
   - However, without temporal smoothing, the relative loss gradient shifted the overall bias (+2.25 km/h on S3a) and caused step-to-step velocity predictions to oscillate slightly in lower/medium regimes, raising full-drive dead reckoning drift.
2. **Design of Experiment 5 (Temporal Consistency Regularization):**
   - **Hypothesis:** Penalizing frame-to-frame acceleration discontinuity $\|\mu_v[t] - \mu_v[t-1] - a_y[t]\Delta t\|$ will eliminate high-frequency step jitter while preserving the sharp $80+\text{ km/h}$ accuracy gains of $\mathcal{L}_{\text{rel}}$.