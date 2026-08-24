# IDR-Nav — Offline Intelligent Dead-Reckoning Navigation Core

[![Flutter Version](https://img.shields.io/badge/Flutter-3.x-blue.svg)](https://flutter.dev)
[![Architecture](https://img.shields.io/badge/Architecture-15--State%20ES--EKF%20%2B%20ConvNeXt--1D%20%2B%20Temporal%20MHA-success.svg)](#2-complete-system-architecture)
[![Latency](https://img.shields.io/badge/ARM64%20Latency-0.0125%20ms%20(12.5%20%C2%B5s)-brightgreen.svg)](#5-performance-benchmarks--arm64-latency)
[![Outage Drift](https://img.shields.io/badge/30s%20Blackout%20Drift-95.31m%20(S3a)-brightgreen.svg)](progress.md)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**IDR-Nav** is a high-precision, production-grade, offline dead-reckoning navigation engine developed for standard commercial smartphones. It operates entirely on-device with **zero cloud dependencies**, providing continuous, sub-millisecond vehicle positioning, velocity, and 3D orientation (10 Hz to 100 Hz) through challenging GNSS-denied environments such as long tunnels, underground parking garages, multi-level interchanges, and dense urban canyons.

---

## Table of Contents

1. [Executive Overview & Philosophy](#1-executive-overview--philosophy)
2. [Complete System Architecture](#2-complete-system-architecture)
   * [Layer 1: Sensor Ingestion, Auto-Alignment & Coordinate Frames](#layer-1-sensor-ingestion-auto-alignment--coordinate-frames)
   * [Layer 2: 18-Channel Causal Multi-Domain Physics Engine](#layer-2-18-channel-causal-multi-domain-physics-engine)
   * [Layer 3: Deep Neural Kinematics Observer (`DeepSpeedKinematicsNet`)](#layer-3-deep-neural-kinematics-observer-deepspeedkinematicsnet)
   * [Layer 4: 15-State Error-State Extended Kalman Filter (`EkfFusionEngine`)](#layer-4-15-state-error-state-extended-kalman-filter-ekffusionengine)
   * [Layer 5: Monotonic Forward Route Tracking & Offline OSM Map-Matching](#layer-5-monotonic-forward-route-tracking--offline-osm-map-matching)
   * [Layer 6: UI, 3D Vehicle Gizmo & Telemetry Stream](#layer-6-ui-3d-vehicle-gizmo--telemetry-stream)
3. [Mathematical Formulations & Coordinate Conventions](#3-mathematical-formulations--coordinate-conventions)
4. [Empirical Benchmark Results & Experimentation Roadmap](#4-empirical-benchmark-results--experimentation-roadmap)
5. [Real-Time Performance & ARM64 Mobile Benchmarks](#5-real-time-performance--arm64-mobile-benchmarks)
6. [Repository Structure](#6-repository-structure)
7. [Installation, Training & Execution Guide](#7-installation-training--execution-guide)

---

## 1. Executive Overview & Philosophy

Standard smartphone GPS navigation fails in tunnels and urban canyons because satellite signals attenuate, reflect, or drop entirely. Raw smartphone accelerometer integration drifts exponentially ($\Delta p \propto 0.5 \cdot a \cdot t^2$) within seconds due to micro-electro-mechanical sensor bias, thermal drift, and pitch gravity leakage.

**IDR-Nav solves this through a unified physics-guided neural observer fused with a 15-state Error-State EKF:**
* **Aerospace-Grade Strapdown Inertial Mechanization:** Uses true Newtonian kinematics in a standardized Local Math ENU (East-North-Up) Cartesian frame.
* **18-Channel Causal Multi-Domain Physics Space:** Combines calibrated linear accelerations, angular rates, leaky velocity integrals, suspension spectral energy bands, spectral centroid, turn-gated centripetal kinematics, vibration ratios, and physical pitch observer.
* **Modern ConvNeXt-1D + Temporal Multi-Head Attention Backbone:** Deep neural network with 4-stage inverted bottleneck pyramid, depthwise separable convolutions, and temporal self-attention extracting vehicle velocity, calibrated heteroscedastic uncertainty, delta-velocity, ZUPT probabilities, and road pitch.
* **15-State Error-State EKF (ES-EKF):** High-rate 100 Hz propagation with Non-Holonomic Constraints (NHC: $v_x \approx 0, v_z \approx 0$), dynamic heteroscedastic measurement injection, $\chi^2$ innovation gating, and zero-velocity/zero-angular-rate clamps (ZUPT/ZARU).
* **Monotonic Forward Route Projection:** Clamps lateral cross-track divergence to OSM road centerlines while along-track travel distance is accurately driven by the neural speed observer.

---

### Core Questions & Architectural Rationale

#### Q1: Why not just use "Offline Google Maps" or "Offline Apple Maps"?
> A common misconception is that offline map modes solve GPS loss. **They do not.**
* **Offline Maps solve "No Internet":** They pre-download road vector tiles and routing networks to your phone so you don't need cellular 4G/5G data. However, **they still rely 100% on live GPS/GNSS satellite radio signals** to position your vehicle.
* When you drive into an underground tunnel, a parking garage, or an urban canyon where satellite signals are blocked, offline Google Maps immediately loses its fix, displays a grey search radius, or relies on naive constant-velocity extrapolation that fails the moment you brake, stop, or accelerate.
* **IDR-Nav solves "No GPS":** IDR-Nav is an **Inertial Navigation System (INS)**. It operates when satellite chips go completely blind by converting the smartphone's high-rate (100 Hz) accelerometer and gyroscope into continuous physical positioning and orientation.

#### Q2: What happens if you take a different exit or turn inside a tunnel without GPS?
* Naive dead-reckoning systems follow a rigid pre-planned path.
* **IDR-Nav dynamically tracks physical vehicle yaw:** When you steer into an exit ramp, the 100 Hz gyroscope integrates the real angular rotation $\omega_z$, rotating the internal Math ENU velocity vector.
* The Map-Matching layer detects the angular divergence and switches candidate edges using heading compatibility gating ($|\theta_{\text{road}} - \theta_{\text{vehicle}}| \le 45^\circ$), self-correcting your route dead-reckoning without a single satellite ping.

#### Q3: Why not simply double-integrate raw smartphone accelerometer data ($p = \iint a \, dt^2$)?
* Double integration is mathematically unstable on MEMS sensors. A tiny accelerometer bias of just $0.10\text{ m/s}^2$ (or a $0.58^\circ$ pitch estimation error causing gravity leakage) produces **$450\text{ meters}$ of position drift in just 30 seconds** ($\Delta p = \frac{1}{2} a t^2$).
* **IDR-Nav uses Neural Velocity Observation & Centripetal Kinematics:** Instead of integrating raw longitudinal acceleration, the ConvNeXt-1D network estimates bounded forward speed $\mu_v$ directly from tire-road vibration harmonics and suspension dynamics, while centripetal turns provide exact kinematic speed constraints ($v = |a_x| / |\omega_z|$). This converts an unstable $O(t^2)$ problem into a bounded, error-state Kalman update.

#### Q4: Why must the entire engine run offline on-device rather than in the cloud?
* **Zero Cellular Reception:** Underground tunnels, parking basements, and remote highways have zero 4G/5G connectivity.
* **Sub-Millisecond Execution:** Safety-critical navigation requires 100 Hz state updates ($< 10\text{ ms}$). IDR-Nav executes in **$12.5\text{ }\mu\text{s}$ per cycle** on ARM64 mobile hardware, utilizing less than $0.02\%$ of a single mobile CPU core.

---
                                  PHYSICAL SENSORS (100 Hz)
                      [Accelerometer]   [Gyroscope]   [Magnetometer]   [GNSS 1Hz]
                                │             │              │             │
                                └─────────────┼──────────────┘             │
                                              ▼                            │
                            ┌──────────────────────────────────┐           │
                            │ LAYER 1: SENSOR CALIBRATION      │           │
                            │ • Rodrigues Mount Auto-Alignment │           │
                            │ • WGS84 Geodetic -> Local ENU    │           │
                            └─────────────────┬────────────────┘           │
                                              ▼                            │
                            ┌──────────────────────────────────┐           │
                            │ LAYER 2: 18-CH PHYSICS ENGINE    │           │
                            │ • Leaky Velocity Integrals       │           │
                            │ • Sub-band Suspension FFT Energy │           │
                            │ • Turn-Gated Centripetal Speed   │           │
                            │ • Physical Pitch Observer        │           │
                            └─────────────────┬────────────────┘           │
                                              ▼                            │
                            ┌──────────────────────────────────┐           │
                            │ LAYER 3: CONVNEXT + MHA OBSERVER │           │
                            │ • 4-Stage ConvNeXt-1D Stem       │           │
                            │ • 4-Head Temporal Attention      │           │
                            │ • 5 Multi-Task Physics Heads:    │           │
                            │   [mu_v, sigma^2, dv, ZUPT, pitch]           │
                            └─────────────────┬────────────────┘           │
                                              │                            │
                                              ▼                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 4: 15-STATE ERROR-STATE KALMAN FILTER (ES-EKF)                                   │
│ • State (15x1): [Pos (3), Vel (3), Attitude (3), Accel Bias (3), Gyro Bias (3)]^T      │
│ • 100 Hz Strapdown Prediction + Non-Holonomic Constraints (NHC)                        │
│ • Heteroscedastic Neural Velocity Fusion with Chi-Square (χ²) Outlier Gating           │
│ • Physical ZUPT (Zero-Velocity) & ZARU (Zero-Angular-Rate) Clamps                      │
└─────────────────────────────────────────────┬──────────────────────────────────────────┘
                                              │
                                              ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 5: FORWARD ROUTE TRACKER & OFFLINE OSM MAP-MATCHER                               │
│ • Monotonic Along-Track Progress (s_k >= s_{k-1})                                      │
│ • Frenet Orthogonal Projection (Bounds lateral drift to road centerline)               │
└─────────────────────────────────────────────┬──────────────────────────────────────────┘
                                              │
                                              ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 6: 10 Hz VEHICLE TELEMETRY STREAM & 3D DASHBOARD                                 │
│ • Stream<NavState>: Position, Heading, Velocity, Covariance, Mode                      │
│ • 3D Vehicle Orientation Gizmo & Real-Time Trajectory Trail Canvas                     │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Complete System Architecture

### Layer 1: Sensor Ingestion, Auto-Alignment & Coordinate Frames

* **Files:** [`app/lib/calibration/alignment_estimator.dart`](file:///Users/anv./Development/INSS%20Navigation%20app/app/lib/calibration/alignment_estimator.dart), [`app/lib/core/constants.dart`](file:///Users/anv./Development/INSS%20Navigation%20app/app/lib/core/constants.dart)
* **Responsibility:** Ingest raw smartphone IMU measurements, estimate the phone's mounting orientation relative to the vehicle body, and convert geodetic coordinates into local Cartesian ENU.

#### 1. Automatic Rodrigues Mount Alignment & Gravity Decoupling
When a smartphone is mounted in a vehicle cradle, its coordinate axes ($X_{\text{phone}}, Y_{\text{phone}}, Z_{\text{phone}}$) do not align with the vehicle body axes ($X_v = \text{Right}, Y_v = \text{Forward}, Z_v = \text{Up}$):
* **Stationary Gravity Estimation:**  
  $$\mathbf{g}_{\text{phone}} = \mathbb{E}[\mathbf{a}_{\text{stationary}}], \quad \mathbf{u}_g = \frac{\mathbf{g}_{\text{phone}}}{\|\mathbf{g}_{\text{phone}}\|}$$
* **Rodrigues Body-Frame Rotation Matrix ($\mathbf{R}_{p \to v}$):**  
  Rotates the measured gravity vector $\mathbf{u}_g$ into the vertical earth vector $\mathbf{v}_z = [0, 0, 1]^T$:
  $$\mathbf{k} = \mathbf{u}_g \times \mathbf{v}_z, \quad \mathbf{K} = [\mathbf{k}]_\times$$
  $$\mathbf{R}_{p \to v} = \mathbf{I} + \mathbf{K} + \mathbf{K}^2 \frac{1 - \mathbf{u}_g \cdot \mathbf{v}_z}{\|\mathbf{k}\|^2}$$
* **Clean Vehicle-Frame Kinematics:**
  $$\mathbf{a}_{\text{vehicle}} = \mathbf{R}_{p \to v} \mathbf{a}_{\text{phone}}, \quad \boldsymbol{\omega}_{\text{vehicle}} = \mathbf{R}_{p \to v} \boldsymbol{\omega}_{\text{phone}}$$
  - $a_{y, v}$: Pure forward longitudinal acceleration (zero DC gravity bias).
  - $a_{z, v}$: Pure vertical suspension vibration.
  - $a_{x, v}$: Pure lateral centripetal acceleration.

---

### Layer 2: 18-Channel Causal Multi-Domain Physics Engine

* **Files:** [`ml/src/dataset_spectral.py`](file:///Users/anv./Development/INSS%20Navigation%20app/ml/src/dataset_spectral.py), [`app/lib/ai/speed_filter_runner.dart`](file:///Users/anv./Development/INSS%20Navigation%20app/app/lib/ai/speed_filter_runner.dart)
* **Responsibility:** Extract an exact 18-channel physics tensor across a sliding 48-sample window (4.8 seconds at 10 Hz) with **100% causality** (zero lookahead).

| Channel | Name | Domain | Exact Mathematical Formulation | Physical Significance |
|---|---|---|---|---|
| **0, 1, 2** | $a_x, a_y, a_z$ | Time | Calibrated vehicle linear accelerations | Right, Forward, Up accelerations |
| **3, 4, 5** | $\omega_x, \omega_y, \omega_z$ | Time | Calibrated vehicle angular rates | Pitch, Roll, Yaw rates |
| **6** | $\|\mathbf{a}\| - g$ | Physics | $\sqrt{a_x^2 + a_y^2 + a_z^2} - 9.80665$ | Dynamic inertial motion magnitude |
| **7** | $\|\boldsymbol{\omega}\|$ | Physics | $\sqrt{\omega_x^2 + \omega_y^2 + \omega_z^2}$ | Total 3D rotational intensity |
| **8** | $I_{ay}$ | Physics | $I_y[t] = 0.95 \cdot I_y[t-1] + a_y[t] \cdot \Delta t$ | Leaky forward velocity integral |
| **9** | $I_{az}$ | Physics | $I_z[t] = 0.95 \cdot I_z[t-1] + (a_z[t] - g) \cdot \Delta t$ | Leaky vertical heave velocity integral |
| **10** | $I_{ax}$ | Physics | $I_x[t] = 0.95 \cdot I_x[t-1] + a_x[t] \cdot \Delta t$ | Leaky lateral sway velocity integral |
| **11** | $E_{\text{low}}$ | Spectral | $\sum_{f=0.5}^{2.5\text{ Hz}} \|X_z(f)\|^2$ | Chassis bounce & pitch resonant energy |
| **12** | $E_{\text{mid}}$ | Spectral | $\sum_{f=2.5}^{6.0\text{ Hz}} \|X_z(f)\|^2$ | Suspension unsprung mass resonance |
| **13** | $E_{\text{high}}$ | Spectral | $\sum_{f=6.0}^{12.0\text{ Hz}} \|X_z(f)\|^2$ | Tire tread & high-frequency road texture |
| **14** | $\text{Centroid}_z$ | Spectral | $\sum (f \cdot \|X_z(f)\|^2) / \sum \|X_z(f)\|^2$ | Vertical vibration spectral centroid |
| **15** | $\text{turn\_feat}$ | Kinematics | $\begin{cases} \text{clip}\left(\frac{\|a_x\|}{\|\omega_z\|}, 0, 40\right) & \text{if } \|\omega_z\| \ge 0.035\text{ rad/s} \\ 0.0 & \text{otherwise} \end{cases}$ | Turn-gated centripetal speed |
| **16** | $\text{vib\_ratio}$ | Physics | $\text{Var}(a_z) / (\text{Var}(a_y) + 10^{-4})$ | Vertical-to-longitudinal vibration ratio |
| **17** | $\theta_{\text{phys}}$ | Observer | $\theta_{\text{phys}}[t] = 0.98(\theta[t-1] + \omega_y \Delta t) + 0.02 \arctan2(a_y, a_z)$ | Complementary-filter physical pitch |

---

### Layer 3: Deep Neural Kinematics Observer (`DeepSpeedKinematicsNet`)

* **Files:** [`ml/src/model.py`](file:///Users/anv./Development/INSS%20Navigation%20app/ml/src/model.py), [`ml/src/train_spectral.py`](file:///Users/anv./Development/INSS%20Navigation%20app/ml/src/train_spectral.py)
* **Model Parameters:** 556,764 parameters (Size: **286 KB** ONNX).
* **Architecture:** 4-Stage ConvNeXt-1D Backbone + Multi-Head Temporal Self-Attention + 5 Multi-Task Physics Heads.

```text
   INPUT TENSOR: (Batch, 18, 48) [18 Channels x 48 Timesteps at 10 Hz]
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ CONVNEXT-1D STEM & MULTI-STAGE PYRAMID                      │
   │ • Stem: Conv1d(18 -> 48, kernel=3) + LayerNorm + GELU       │
   │ • Stage 1 (dim 48):  ConvNeXtBlock1D(k=7, mlp_ratio=4) x 2 │
   │ • Stage 2 (dim 64):  Downsample + ConvNeXtBlock1D x 2       │
   │ • Stage 3 (dim 96):  Downsample + ConvNeXtBlock1D x 2       │
   │ • Stage 4 (dim 128): Downsample + ConvNeXtBlock1D x 2       │
   └──────────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ TEMPORAL MULTI-HEAD ATTENTION (MHA) CORE                    │
   │ • Embedding Dim: 128 | Heads: 4 | Pre-LayerNorm             │
   │ • Attends across time steps to capture speed transitions    │
   │ • Global Temporal Mean Pooling                              │
   └──────────────────────────────┬──────────────────────────────┘
                                  │
      ┌──────────────┬────────────┼────────────┬──────────────┐
      ▼              ▼            ▼            ▼              ▼
┌───────────┐  ┌───────────┐ ┌──────────┐ ┌───────────┐ ┌───────────┐
│ Speed (μ) │  │ Var (σ²)  │ │ Delta-v  │ │ P(ZUPT)   │ │ Pitch (θ) │
│ MLP->ReLU │  │ Log-Var   │ │ MLP->Lin │ │ MLP->Sig  │ │ MLP->Lin  │
└─────┬─────┘  └─────┬─────┘ └──────────┘ └─────┬─────┘ └───────────┘
      │              │                          │
      └──────────────┴──────────────────────────┴─────────────────────► To EKF Fusion
```

#### Multi-Task Physics Heads & Outputs:
1. **Speed Head ($\mu_v$):** MLP ($128 \to 64 \to 1$ with ReLU) outputting forward velocity $\mu_v \ge 0.0\text{ m/s}$.
2. **Calibrated Heteroscedastic Uncertainty Head ($\sigma_v^2$):** Regresses bounded log-variance $\log \sigma^2 \in [-3.0, +3.0]$, producing $\sigma_v^2 = \exp(\text{clamp}(\cdot))$ in $(\text{m/s})^2$.
3. **Delta-Velocity Head ($\Delta v$):** Linear projection estimating step acceleration increment $\Delta v = v_t - v_{t-1}$.
4. **Zero-Velocity Probability Head ($P(\text{ZUPT})$):** Sigmoid activation producing $p \in [0, 1]$ indicating stationary status at traffic lights.
5. **Neural Road Pitch Head ($\hat{\theta}_{\text{pitch}}$):** Linear output predicting vehicle chassis road grade.

#### Loss Formulation:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{hetero}}(\mu_v, \sigma_v^2, v_{\text{gt}}) + 0.5 \cdot \text{SmoothL1}(\Delta v, \Delta v_{\text{gt}}) + 0.5 \cdot \text{BCE}(p_{\text{zupt}}, z_{\text{gt}}) + 0.2 \cdot \text{MSE}(\hat{\theta}_{\text{pitch}}, \theta_{\text{phys}})$$
where the heteroscedastic Gaussian NLL loss is numerically stabilized:
$$\mathcal{L}_{\text{hetero}} = \frac{1}{2} \exp(-s) (v_{\text{gt}} - \mu_v)^2 + \frac{1}{2} s, \quad s = \log \sigma^2$$

---

### Layer 4: 15-State Error-State Extended Kalman Filter (`EkfFusionEngine`)

* **Files:** [`app/lib/fusion/ekf_fusion.dart`](file:///Users/anv./Development/INSS%20Navigation%20app/app/lib/fusion/ekf_fusion.dart), [`ml/src/task3_indistribution_evaluation.py`](file:///Users/anv./Development/INSS%20Navigation%20app/ml/src/task3_indistribution_evaluation.py)
* **Responsibility:** High-rate 100 Hz state propagation, non-holonomic kinematics, error-state covariance management, and optimal neural measurement fusion.

#### 1. 15-Dimensional Error-State Vector
$$\delta \mathbf{x} = \begin{bmatrix} \delta \mathbf{p}_{3\times 1} & \delta \mathbf{v}_{3\times 1} & \delta \boldsymbol{\theta}_{3\times 1} & \mathbf{b}_{a, 3\times 1} & \mathbf{b}_{g, 3\times 1} \end{bmatrix}^T \in \mathbb{R}^{15}$$
* $\delta \mathbf{p}$: Position errors in Local Math ENU (East, North, Up) in meters.
* $\delta \mathbf{v}$: Velocity errors in Local Math ENU in m/s.
* $\delta \boldsymbol{\theta}$: Attitude error angles (East, North, Up) in radians.
* $\mathbf{b}_a$: Accelerometer biases (body frame) in $\text{m/s}^2$.
* $\mathbf{b}_g$: Gyroscope biases (body frame) in $\text{rad/s}$.

#### 2. High-Rate Strapdown Mechanization (100 Hz)
* **Attitude Update (Math ENU Frame):**
  $$\theta_z[t] = \theta_z[t - \Delta t] + (\omega_z[t] - b_{g, z}) \cdot \Delta t$$
* **2nd-Order Midpoint Acceleration & Position Integration:**
  $$\mathbf{a}_{\text{ENU}}[t] = \begin{bmatrix} a_y \cos\theta + a_x \sin\theta \\ a_y \sin\theta - a_x \cos\theta \\ a_z - g \end{bmatrix}$$
  $$\mathbf{v}[t] = \mathbf{v}[t - \Delta t] + \mathbf{a}_{\text{ENU}}[t] \cdot \Delta t$$
  $$\mathbf{p}[t] = \mathbf{p}[t - \Delta t] + \frac{1}{2} (\mathbf{v}[t - \Delta t] + \mathbf{v}[t]) \Delta t + \frac{1}{2} \mathbf{a}_{\text{ENU}}[t] \Delta t^2$$

#### 3. Strict Non-Holonomic Constraints (NHC)
Every cycle, velocity is projected along the forward momentum heading vector:
$$v_{\text{fwd}} = v_{\text{East}} \cos\theta + v_{\text{North}} \sin\theta$$
$$v_{\text{East, aligned}} = v_{\text{fwd}} \cos\theta, \quad v_{\text{North, aligned}} = v_{\text{fwd}} \sin\theta$$

#### 4. Neural Measurement Fusion & $\chi^2$ Innovation Gating
When the AI speed observer produces velocity update $\mu_v$ with variance $\sigma_v^2$:
* Innovation: $y = \mu_v - v_{\text{fwd}}$
* Innovation Covariance: $S = H P H^T + \sigma_v^2$
* $\chi^2$ Outlier Rejection: Gated if $\gamma = \frac{y^2}{S} > 9.0$ ($3\sigma$ threshold).
* Kalman Gain Update:
  $$K = P H^T S^{-1}, \quad \delta \mathbf{x} = K y, \quad P = (I - K H) P (I - K H)^T + K R K^T$$

#### 5. Physical ZUPT & Micro-ZARU
* **Physical ZUPT:** When $\text{Var}(a) < 0.025\text{ m}^2/\text{s}^4$ and $\|\boldsymbol{\omega}\| < 0.05\text{ rad/s}$, velocity is hard-clamped to $[0, 0, 0]^T$ with variance $10^{-4}\text{ (m/s)}^2$.
* **Micro-ZARU:** When cruising on straight highways ($|a_x| < 0.15\text{ m/s}^2$ and $|\omega_z| < 0.008\text{ rad/s}$), yaw rate is clamped to zero, preventing phantom curve accumulation.

---

### Layer 5: Monotonic Forward Route Tracking & Offline OSM Map-Matching

* **Files:** [`ml/src/map_matcher.py`](file:///Users/anv./Development/INSS%20Navigation%20app/ml/src/map_matcher.py), [`app/lib/map_matching/osm_graph.dart`](file:///Users/anv./Development/INSS%20Navigation%20app/app/lib/map_matching/osm_graph.dart)
* **Responsibility:** Monotonic along-track progression ($s_k \ge s_{k-1}$) with 35-sample forward lookahead ($3.5\text{s}$) and 15-sample smoothed road tangents.
* **Heading Compatibility Gating:** Rejects segments with $|\theta_{\text{road}} - \theta_{\text{vehicle}}| > 45^\circ$, preventing false snapping to overpasses or opposing traffic.
* **Frenet Decomposition:** Orthogonal cross-track snapping (95%) binds vehicle position to the road centerline, while along-track progression is governed by calibrated AI speed.

---

## 3. Mathematical Formulations & Coordinate Conventions

### Standardized Coordinate Systems

1. **WGS84 Geodetic Frame:** Latitude ($\phi$), Longitude ($\lambda$), Altitude ($h$).
2. **Local Math ENU Frame:**
   * $+X = \text{East}$
   * $+Y = \text{North}$
   * $+Z = \text{Up}$
   * **Heading ($\theta$):** Measured in radians **Counter-Clockwise (CCW) from East** ($\theta = 0 \to \text{East}, \theta = \pi/2 \to \text{North}$).
3. **Vehicle Body Frame:**
   * $+X_v = \text{Right}$
   * $+Y_v = \text{Forward}$
   * $+Z_v = \text{Up}$
4. **Compass Heading ($\psi$):**
   * Measured in degrees **Clockwise (CW) from North** ($0^\circ = \text{North}, 90^\circ = \text{East}$).
   * **Conversion Formula:**
     $$\psi = (90^\circ - \theta \cdot \frac{180^\circ}{\pi}) \bmod 360^\circ$$
     $$\theta = (90^\circ - \psi) \cdot \frac{\pi}{180^\circ}$$

---

## 4. Empirical Benchmark Results & Experimentation Roadmap

All experiments are logged systematically in [`progress.md`](progress.md) under strict data isolation.

### Experiment 0 (Frozen Baseline) vs Experiment 1 (Channel 15 Turn-Gated Fix)

| Benchmark Metric | Experiment 0 (Baseline) | Experiment 1 (Turn-Gated Fix) | Delta / Outcome |
|---|---|---|---|
| **Driver A S3a Velocity MAE** | $14.96\text{ km/h}$ | **$14.65\text{ km/h}$** | **Improved** ($-0.31\text{ km/h}$) |
| **Driver A S3a Velocity RMSE** | $19.68\text{ km/h}$ | **$18.93\text{ km/h}$** | **Improved** ($-0.75\text{ km/h}$) |
| **Driver A S3a Velocity Bias** | $-1.31\text{ km/h}$ | **$-0.47\text{ km/h}$** | **Near zero** ($+0.84\text{ km/h}$) |
| **Driver A S3a Pearson $r$** | $0.566$ | **$0.569$** | **Higher correlation** |
| **S3a 80+ km/h Speed Bin MAE** | $31.8\text{ km/h}$ | **$28.6\text{ km/h}$** | **$-3.2\text{ km/h}$ win** |
| **AI+EKF 30s Outage Drift** | $103.39\text{ m}$ | **$95.31\text{ m}$** | **$-8.08\text{ m}$ ($7.8\%$ reduction)** |
| **AI+EKF 90s Outage Drift** | $275.64\text{ m}$ | **$254.38\text{ m}$** | **$-21.26\text{ m}$ ($7.7\%$ reduction)** |
| **AI+EKF Full Drive Drift (S3a)** | $9742.63\text{ m}$ | **$9058.45\text{ m}$** | **$-684.18\text{ m}$ ($7.0\%$ reduction)** |
| **AI+EKF Full Drive Drift (Driver E)**| $1782.90\text{ m}$ | **$1292.80\text{ m}$** | **$-490.10\text{ m}$ ($27.5\%$ reduction)** |

---

## 5. Real-Time Performance & ARM64 Mobile Benchmarks

### Per-Cycle Latency Breakdown (Dart Runtime on ARM64)

```text
=================================================================
   PER-CYCLE PIPELINE LATENCY PROFILE (DART RUNTIME ON ARM64)
=================================================================
Subsystem Breakdown (Mean):
  1. 18-Ch Feature Extraction & Radix-2 FFT:    0.0105 ms (10.5 µs)
  2. 15-State Strapdown INS Prediction & NHC:   0.0013 ms ( 1.3 µs)
  3. EKF Measurement Update & Telemetry State:  0.0008 ms ( 0.8 µs)
-----------------------------------------------------------------
Total Per-Cycle Loop Time:
  Mean:        0.0125 ms (12.5 µs)
  P50/Median:  0.0030 ms ( 3.0 µs)
  P95:         0.0550 ms (55.0 µs)
  P99:         0.1240 ms (124.0 µs)
-----------------------------------------------------------------
Target 10 Hz Time Budget:  100.000 ms
Actual Budget Utilized:    0.013% (99.987 ms margin)
Execution Throughput:      > 80,000 cycles / second
=================================================================
```

### ONNX Model Inference Latency (1,000 Invocations on Mobile CPU)
* **Mean Latency:** $0.072\text{ ms} \pm 0.034\text{ ms}$ ($72.0\text{ }\mu\text{s}$)
* **P50 / Median:** $0.053\text{ ms}$ ($53.0\text{ }\mu\text{s}$)
* **P95:** $0.129\text{ ms}$ ($129.0\text{ }\mu\text{s}$)

---

## 6. Repository Structure

```text
INSS-Navigation-app/
├── app/                                 # Flutter Application & Real-Time Engine
│   ├── lib/
│   │   ├── ai/                          # Spectral FFT & AI Feature Runners
│   │   │   └── speed_filter_runner.dart # Pure Dart 18-Channel Features & Radix-2 FFT
│   │   ├── calibration/                 # Mounting Alignment & Frame Transformation
│   │   │   └── alignment_estimator.dart # Rodrigues gravity vector & chassis pitch/roll
│   │   ├── core/                        # Engine Coordinator & Constants
│   │   │   ├── constants.dart           # WGS84 ellipsoid & GeoMath transformations
│   │   │   └── idr_nav_engine.dart      # Master Stream<NavState> coordinator
│   │   ├── fusion/                      # Error-State Extended Kalman Filter
│   │   │   ├── ekf_fusion.dart          # 15-State ES-EKF with Centripetal Kinematics
│   │   │   └── rts_smoother.dart        # Fixed-Lag RTS Backward Smoother
│   │   ├── ins/                         # Strapdown Inertial Navigation System
│   │   │   └── strapdown_ins.dart       # High-rate Math ENU mechanization
│   │   ├── map_matching/                # Offline OpenStreetMap Engine
│   │   │   ├── hmm_map_matcher.dart     # HMM Viterbi road candidate matcher
│   │   │   └── osm_graph.dart           # Spatial road network graph & projection
│   │   ├── mode_manager/                # State Machine Coordinator
│   │   │   └── mode_manager.dart        # GNSS <-> Dead Reckoning transitions
│   │   ├── models/                      # Sensor & Navigation Data Contracts
│   │   │   ├── nav_mode.dart            # Navigation mode enumerations
│   │   │   ├── nav_state.dart           # Published telemetry output state
│   │   │   └── sensor_sample.dart       # Raw IMU/GNSS sensor sample models
│   │   └── ui/                          # Visualization Dashboard
│   │       └── debug_dashboard.dart     # 3D Vehicle Gizmo & Trajectory Canvas
│   ├── assets/
│   │   ├── models/
│   │   │   ├── speed_filter.onnx        # Exported 18-Channel ConvNeXt Model (286 KB)
│   │   │   └── heading_filter.onnx      # Exported Heading Observer Model (94 KB)
│   │   └── sample_logs/sample_drive.csv # Real drive test dataset
│   └── test/                            # Comprehensive Flutter Unit & Pipeline Tests
│       ├── ekf_fusion_test.dart         # 15-State EKF & NHC unit tests
│       ├── s3a_parity_trace_test.dart   # Deterministic Python/Dart state parity test
│       ├── pipeline_latency_benchmark_test.dart # Per-cycle latency benchmark
│       └── widget_test.dart             # Dashboard widget smoke tests
│
├── ml/                                  # Machine Learning & Deep Kinematics Suite
│   ├── src/
│   │   ├── model.py                     # DeepSpeedKinematicsNet (ConvNeXt-1D + MHA)
│   │   ├── dataset_spectral.py          # 18-channel multi-domain spectral features
│   │   ├── train_spectral.py            # PyTorch Calibrated Heteroscedastic Training
│   │   ├── export_onnx.py               # ONNX Runtime exporter
│   │   ├── map_matcher.py               # ForwardRouteTracker & HmmMapMatcher
│   │   ├── task3_indistribution_evaluation.py # In-Distribution Benchmark Suite (S3a)
│   │   └── evaluate_full_pipeline.py    # Motorway Full-Pipeline Benchmark (Driver E)
│   ├── weights/
│   │   └── best_spectral_speed_filter.pt# Best PyTorch model checkpoint
│   └── evaluation_plots/                # Benchmark evaluation figures
│       ├── indistribution_trajectory_drift_benchmark.png # Driver A S3a evaluation plot
│       └── trajectory_drift_benchmark.png # Driver E Vw11 evaluation plot
│
├── progress.md                          # Experiment Progress Tracker & Metric Matrix
└── Documentation/                       # Engineering Specifications & Protocols
    ├── compliance.md                    # Problem Statement Compliance Matrix
    ├── evaluation_report.md             # Complete Evaluation Report
    ├── prd.md                           # Master Product Requirements Document
    ├── ps.txt                           # Official Hackathon Problem Statement
    └── testing.md                       # Verification Playbook & Protocol Specs
```

---

## 7. Installation, Training & Execution Guide

### Prerequisites
* **Flutter SDK:** $\ge 3.19.0$
* **Dart SDK:** $\ge 3.3.0$
* **Python:** $\ge 3.10$ with PyTorch, NumPy, Pandas, Matplotlib, ONNX

---

### Step 1: Run Flutter Tests & Parity Verification

```bash
cd app
flutter pub get
flutter test
```

---

### Step 2: Train & Evaluate AI Speed Models

```bash
# 1. Train 18-Channel ConvNeXt-1D Model (DeepSpeedKinematicsNet)
ml/venv/bin/python3 -m ml.src.train_spectral --epochs 15

# 2. Export models to ONNX Runtime format (.onnx)
ml/venv/bin/python3 -m ml.src.export_onnx

# 3. Run In-Distribution Evaluation Benchmark (Driver A - S3a)
ml/venv/bin/python3 -m ml.src.task3_indistribution_evaluation

# 4. Run Held-Out OOD Benchmark (Driver E - Vw11)
ml/venv/bin/python3 -m ml.src.evaluate_full_pipeline
```

---

### Step 3: Run Flutter App with Live 3D Dashboard

```bash
cd app
# Run on macOS desktop or connected mobile device
flutter run -d macos
# or Android
flutter run -d android
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
