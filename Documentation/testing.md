# IDR-Nav — Master Verification & Testing Playbook

**Document:** `Documentation/testing.md`  
**Target System:** IDR-Nav (Offline Intelligent Dead Reckoning Navigation Layer)  
**Target Platform:** Flutter (Dart runtime) + Python (PyTorch / ONNX ML Pipeline)  
**Authoritative Reference:** [`Documentation/prd.md`](file:///Users/anv./Development/INSS%20Navigation%20app/Documentation/prd.md)  
**Target Audience:** Autonomous Agents, Core Engineers, Reviewers, and Testers  

---

# 1. Executive Testing Philosophy & The 5 Golden Rules

When modifying any component of IDR-Nav (sensor ingestion, calibration, neural speed regression, strapdown mechanization, EKF fusion, mode management, or map-matching), **never evaluate upstream training metrics (e.g. loss curves) in isolation**. Every change must be validated against real geodetic ground truth through the complete integrated pipeline.

### The 5 Golden Rules:
1. **Decouple Speed Prediction (mu) from Uncertainty (sigma^2):**  
   Never evaluate model quality solely via Gaussian Negative Log-Likelihood (NLL). An unconstrained neural network can deceptively minimize NLL by collapsing mu to the prior dataset mean (~32.6 km/h) and inflating its predicted variance sigma^2. Always mandate mu-only RMSE and binned speed bias evaluations.
2. **Mandate Out-of-Distribution (OOD) Uncertainty Calibration:**  
   Predicted standard deviation sigma must scale proportionally with actual prediction error |mu - y|. If predicted sigma drops or stays flat on difficult, high-speed, or unmodeled dynamics, the uncertainty head is miscalibrated and will corrupt the Kalman filter.
3. **Enforce Right-Hand Mathematical Coordinate Conventions:**  
   In Android `SensorManager`, +Z gyroscope rate (omega_z) is **counter-clockwise (CCW)**. In Math ENU coordinates (x = East, y = North), heading angle theta is measured CCW from East (d(theta)/dt = omega_z). Geodetic compass heading is measured clockwise from North (psi = 90 deg - theta). Inverting this convention reverses vehicle turns and causes catastrophic spatial divergence.
4. **Never Confuse Host CPU Latency with Physical Mobile Execution:**  
   `flutter test` executed on macOS compiles to native host machine code (e.g., Apple Silicon ARM64/x86_64 Darwin). Physical mobile execution must be measured directly on a connected device via Flutter Integration Test (`integration_test/`) over at least 1,000 real sensor cycles.
5. **Never Present Estimated Pipeline Numbers as Measured:**  
   Clearly differentiate between measured on-device execution times and unmeasured architectural projections.

---

# 2. Complete Phase-by-Phase Acceptance Criteria (PRD Mapping)

| PRD Phase | Component | Key Metric / Verification Requirement | Target Acceptance Threshold |
|---|---|---|---|
| **Phase A** | `SensorService` & `Alignment` | Raw hardware streaming + static gravity & dynamic GPS course alignment | Gravity error < 0.1 m/s^2; yaw offset convergence < 2.0 deg |
| **Phase B** | `StrapdownIns` & `EkfFusion` | 15-State ES-EKF with Non-Holonomic Constraints (NHC) | 0 NaNs, 0 Divergences; covariance positive semi-definite |
| **Phase C** | `ModeManager` & GNSS Fusion | 1 Hz GNSS fusion + smooth blackout transition | GNSS-aided mean error < 6.0 m; final drift < 0.1% of total trip |
| **Phase D** | `SpeedFilterRunner` (AI) | 16-Channel Spectral Model (FFT PSD + physics) | MAE < 8.0 km/h across all bins; 90s blackout drift < 3.0% |
| **Phase E** | `OsmMapMatcher` (HMM) | Offline OSM road-network topology snapping | Snapping latency < 5 ms; multi-minute blackout drift < 2.0% |
| **Phase F** | `DebugDashboard` (UI) | Real-time 3D orientation gizmo + 2D trail rendering | Sustained 10 Hz update rate; per-cycle latency < 1.0 ms |

---

# 3. Environment Setup & Pre-Flight Checklist

### Step 1: Python Virtual Environment Verification
```bash
# Verify virtual environment and packages
source ml/venv/bin/activate
python3 -c "import torch, numpy, pandas, onnxruntime, pypdf; print('PyTorch MPS available:', torch.backends.mps.is_available())"
```

### Step 2: Flutter Workspace Verification & Device Inspection
```bash
cd app
flutter pub get
flutter analyze
flutter test
flutter devices
cd ..
```

---

# 4. Detailed Verification Protocols

---

## Protocol 1: mu-Only Speed Accuracy & Binned Bias Audit

### Purpose:
Ensures the neural network dynamically tracks the true speed trajectory across all velocity regimes (0 to 120 km/h) and has not collapsed to the dataset prior mean (~32.6 km/h).

### Command to Execute:
```bash
ml/venv/bin/python3 -m ml.src.evaluate_spectral_binned
```

### Interpretation Table & Passing Criteria:
```text
=====================================================================================
       16-CHANNEL SPECTRAL MODEL EVALUATION (HELD-OUT DRIVER E)
=====================================================================================
Speed Bin (km/h)     | Regime Description   | Target Mean Pred  | Max Allowable Bias
-------------------------------------------------------------------------------------
0-10 km/h            | Stationary / Creep   | < 5.0 km/h        | <= +4.0 km/h (ZUPT active)
10-30 km/h           | Low-Speed City       | 18.0 - 24.0 km/h  | <= +/- 5.0 km/h
30-50 km/h           | Urban Arterial       | 38.0 - 44.0 km/h  | <= +/- 5.0 km/h
50-70 km/h           | Suburban / Country   | 55.0 - 65.0 km/h  | <= +/- 7.0 km/h
70-90 km/h           | Fast A-Road          | 75.0 - 85.0 km/h  | <= +/- 10.0 km/h
90-140 km/h          | Motorway / Highway   | 85.0 - 105.0 km/h | <= +/- 15.0 km/h
=====================================================================================
```
* **Failure Condition:** If `Mean Pred` stays clamped near 30 to 35 km/h for the >= 70 km/h bins, the model is suffering from prior-mean collapse due to training set speed imbalance.

---

## Protocol 2: Uncertainty Calibration (sigma^2) Check

### Purpose:
Verifies that the heteroscedastic uncertainty head (sigma^2) provides a mathematically reliable measurement covariance R = sigma^2 for the EKF.

### Command to Execute:
```bash
ml/venv/bin/python3 -m ml.src.task1_uncertainty_calibration
```

### Acceptance Checklist:
1. **Monotonic Error-Uncertainty Correlation:** When actual error |mu - y| increases, predicted sigma must increase proportionally.
2. **OOD Sensitivity:** Out-of-distribution (OOD) or high-speed samples must have higher predicted sigma (>= 15 km/h) than in-distribution urban samples (<= 9 km/h).
3. **Scatter Plot Verification:** Inspect [`ml/evaluation_plots/uncertainty_calibration_scatter.png`](file:///Users/anv./Development/INSS%20Navigation%20app/ml/evaluation_plots/uncertainty_calibration_scatter.png). Samples must fall below the 2-sigma confidence boundary (|Error| <= 2 * sigma).

---

## Protocol 3: Full-Pipeline Drift & Outage Benchmarking

### Purpose:
Tests the entire closed-loop positioning engine (INS Mechanization -> NHC Constraints -> EKF Fusion -> AI Speed Updates) against high-precision vehicle ECU ground truth over 5+ km drives with complex maneuvers (roundabouts, sharp turns, speed transitions) and a **90-second simulated GNSS blackout**.

### Configurations Tested:
* **Config (a): Raw Strapdown INS Only** (Uncorrected baseline; expected quadratic divergence).
* **Config (b): EKF + NHC + GNSS (No AI Speed)** (Trustworthy baseline with 1 Hz GNSS).
* **Config (c): Full Pipeline (EKF + NHC + GNSS + AI Speed Model)**.
* **Config (d): 90-Second Simulated Tunnel Outage** (t = 120s to 210s).

### Commands to Execute:
```bash
# 1. Benchmark on Held-Out Driver E (5 Roundabouts + Motorway Driving)
ml/venv/bin/python3 -m ml.src.evaluate_full_pipeline

# 2. Benchmark on In-Distribution Driver A (City/Suburban Driving Profile)
ml/venv/bin/python3 -m ml.src.task3_indistribution_evaluation
```

### Passing Thresholds:

```text
===========================================================================
                 AUDITED FULL-PIPELINE PASSING THRESHOLDS
===========================================================================
Metric                                        | Passing Target Threshold
---------------------------------------------------------------------------
Config (b) GNSS-Aided Mean Error              | <= 6.00 meters
Config (b) GNSS-Aided Final Drift             | <= 0.20% of total distance
Config (c) Full Pipeline Mean Error           | <= 8.00 meters
Config (c) Full Pipeline Final Drift          | <= 1.00% of total distance
90s Outage Drift (Dead Reckoning Mode)        | <= 3.00% of outage distance
Numerical Stability                           | 0 NaNs, 0 Divergences
===========================================================================
```

---

## Protocol 4: On-Device Physical ARM64 Latency Profiling

### Purpose:
Measures the true on-device CPU execution time inside the native Android ART runtime on physical ARM64 hardware across 1,000 consecutive real driving cycles.

### Command to Execute:
```bash
flutter test integration_test/on_device_latency_test.dart -d <device_serial>
```
*(Example: `flutter test integration_test/on_device_latency_test.dart -d RZCY5159SVJ`)*

### Target Execution Profile:
```text
======================================================================
   ON-DEVICE PHYSICAL ANDROID ARM64 TARGET PROFILE
======================================================================
Subsystem Stage                                | Target P95 Latency
----------------------------------------------------------------------
1. 10/16-Ch Feature Extraction & Windowing     | < 0.010 ms (10 µs)
2. Strapdown INS Prediction & NHC Dampening    | < 0.015 ms (15 µs)
3. EKF Measurement Update & NavState Emission  | < 0.005 ms (5 µs)
----------------------------------------------------------------------
Total Dart Per-Cycle Loop Time                 | < 0.030 ms (30 µs)
Native ONNX Runtime Inference (ARM64 NDK)      | < 0.500 ms (500 µs)
======================================================================
TOTAL END-TO-END PER-CYCLE TIME:               | < 1.000 ms (1.0% budget)
Real-Time Budget Available (10 Hz):            | 100.00 ms
======================================================================
```

---

# 5. Diagnostic Tree & Troubleshooting Guide

```text
                                  PIPELINE FAILURE DETECTED
                                              │
               ┌──────────────────────────────┴──────────────────────────────┐
               ▼                                                             ▼
     [Trajectory Diverges]                                         [Speed Prediction Flawed]
               │                                                             │
    ┌──────────┴──────────┐                                       ┌──────────┴──────────┐
    ▼                     ▼                                       ▼                     ▼
[Inverted Gyro Yaw]  [GPS Logging Glitch]                    [Prior-Mean Collapse]  [Engine Idle Rumble]
• Check Right-Hand   • IO-VNBD logs GPS at                   • Check speed-bin      • Check ZUPT detector
  Sensor frame:        0.1Hz (every 10s).                      distribution.          variance threshold.
  +Z is CCW (dθ = ωz). • Never differentiate                 • Model collapses to   • Clamps speed to 0.0
• Compass heading      positions directly:                     32 km/h if >70km/h     when Var(a) < 0.02
  increases CW.          use valid flag arrivals.              data is <5% of set.    and ||w|| < 0.05.
```

### Detailed Diagnostic Steps:

1. **Symptom: Trajectory curves in opposite direction of road during maneuvers.**
   * **Root Cause:** Gyro yaw rate sign inverted.
   * **Fix:** In `StrapdownIns`, ensure d(theta)/dt = omega_z (where theta is Math ENU angle measured CCW from East). Verify compass heading conversion psi = 90 deg - theta.
2. **Symptom: Error plot exhibits periodic 100 to 600m spikes during GNSS availability.**
   * **Root Cause:** Calculating GPS velocity by dividing static repeated coordinates by dt = 0.1s, producing artificial 10x velocity spikes on the 10th tick.
   * **Fix:** Only process GNSS position/velocity updates when a fresh, non-duplicate coordinate timestamp arrives.
3. **Symptom: Predicted speed remains 10 to 15 km/h when the car is stopped.**
   * **Root Cause:** Vehicle engine idle rumble (800 RPM ~ 13.3 Hz) shaking dashboard phone cradle.
   * **Fix:** Activate the physical Zero-Velocity Update (ZUPT) detector in `SpeedFilterRunner`. Clamps velocity to 0.0 m/s when Var(a) < threshold.
4. **Symptom: Predicted speed underpredicts highway cruising (80 to 110 km/h) by >30%.**
   * **Root Cause:** Training dataset speed imbalance (>81% urban data <50 km/h).
   * **Fix:** Use `dataset_spectral.py` with 16-channel FFT PSD features and speed-stratified balanced sampling across all speed bins.
5. **Symptom: EKF produces NaNs or negative diagonal covariance values.**
   * **Root Cause:** Numerical round-off in covariance update P = (I - K*H) * P.
   * **Fix:** Enforce symmetric Joseph form covariance update:
     ```text
     P = (I - K*H) * P * (I - K*H)^T + K * R * K^T
     ```
     and enforce positive semi-definite diagonal floor P_ii >= 1e-6.

---

# 6. Automated One-Command Master Verification Script

To execute the entire verification suite in a single automated step:

```bash
# Run all Python ML & Fusion verification protocols
ml/venv/bin/python3 -c "
import subprocess
print('=== EXECUTING MASTER VERIFICATION SUITE ===')
scripts = [
    'ml.src.task1_uncertainty_calibration',
    'ml.src.task2_speed_distribution_audit',
    'ml.src.evaluate_spectral_binned',
    'ml.src.evaluate_full_pipeline',
    'ml.src.task3_indistribution_evaluation'
]
for s in scripts:
    print(f'\n--> RUNNING: {s}')
    res = subprocess.run(['ml/venv/bin/python3', '-m', s], capture_output=False)
    if res.returncode != 0:
        print(f'FAILED: {s}')
        exit(1)
print('\n=== ALL PYTHON VERIFICATION PROTOCOLS PASSED ===')
"
```

To run all Flutter unit and integration benchmarks:
```bash
cd app && flutter test && cd ..
```
