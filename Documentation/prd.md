# Product Requirements Document (PRD)
## Project: IDR-Nav — Offline Intelligent Dead Reckoning Navigation Layer

**Version:** 0.1 (Prototype phase)
**Owner:** Anuvab
**Target platform:** Flutter (Android primary, iOS secondary)
**Document purpose:** This PRD is written to be handed to a coding agent (Cursor/Antigravity) as the authoritative spec for implementation. It describes what to build, in what order, and to what standard. Where a decision is ambiguous, the agent should stop and ask rather than assume.

---

## 1. Overview

IDR-Nav is a self-contained navigation *layer* — not a full map/routing app — that estimates a vehicle's real-time position and orientation using the phone's raw IMU sensors (accelerometer, gyroscope, magnetometer), optionally corrected by GNSS when available. It is designed to keep producing accurate, drift-bounded position estimates during GNSS outages (tunnels, underground parking, urban canyons, dense forest) and to seamlessly hand off between GNSS-aided and pure inertial (dead-reckoning) modes.

This layer sits *underneath* a navigation UI (map view, turn-by-turn, etc.) and is responsible only for answering one question continuously: **"Where is the vehicle, and which way is it facing, right now?"**

This is the offline/on-device engine — no server calls, no cloud inference. Everything computed above (map rendering, routing, UI) is out of scope for this document except where it consumes this layer's output.

---

## 2. Goals

1. Estimate vehicle position (lat/lon or local ENU coordinates), velocity, and orientation (heading/pitch/roll) at a minimum of 10 Hz.
2. Maintain positional drift under **10% of distance traveled** during GNSS-denied periods (target benchmark: <100m drift over 1km at 60km/h; <5m drift over 50m in <1 minute).
3. Automatically detect phone mounting orientation relative to the vehicle (pitch/roll/yaw offset) without manual user calibration.
4. Filter out non-navigation motion — potholes, engine vibration, phone handling/misalignment — from the position/velocity estimate.
5. Seamlessly transition between GNSS-aided and GNSS-denied (pure dead-reckoning) modes within milliseconds, with no visible jump/freeze in the output.
6. Snap the estimated trajectory onto the actual road network using offline map data (OpenStreetMap) once map-matching is implemented (Phase 4+).
7. Expose a clean, well-documented output interface (position, heading, velocity, confidence/uncertainty, current mode) that any UI layer can consume.
8. Run entirely on-device, in real time, on a mid-range Android phone, without cloud dependency.

---

## 3. Non-Goals (explicitly out of scope for this build)

- Turn-by-turn routing, map rendering, POI search — this is a positioning engine, not a maps app.
- OBD-II / vehicle CAN bus integration. Phone sensors only.
- Pedestrian dead reckoning (step counting, etc.) — vehicle motion model only.
- Multi-vehicle or fleet features.
- Cloud sync, accounts, telemetry upload.
- Production-grade UI polish — the visualization is a debug/demo instrument, not a shipped consumer UI, until explicitly stated otherwise.
- iOS-specific tuning in the first build (build Android-first; keep code portable but don't spend cycles on iOS-specific sensor quirks yet).

---

## 5. Functional Requirements by Module

### 5.1 SensorService
- Streams raw accelerometer, gyroscope, magnetometer via `sensors_plus`, using **raw uncorrected sensor data** (not OS-fused rotation vectors).
- Streams GNSS position/velocity/bearing via `geolocator` (or platform location API), including accuracy/HDOP if available.
- Every sample carries a real hardware/monotonic timestamp — never assume fixed dt.
- Exposes each stream independently (accel stream, gyro stream, mag stream, GNSS stream) so downstream modules subscribe only to what they need.
- Must handle sensor unavailability gracefully (e.g., no magnetometer on device) without crashing downstream modules.

### 5.2 Calibration & Alignment Module
- On app start / remount detection, estimate the phone's static pitch/roll relative to level ground using the gravity vector (phone stationary assumption, detected via low variance in accelerometer readings over a short window).
- Once GNSS is available and the vehicle is moving above a minimum speed threshold, estimate yaw offset (phone-frame vs. vehicle-frame heading) by comparing GNSS course-over-ground to the phone's gyro-integrated heading.
- Continuously refine this alignment estimate in the background (not just once at startup) — mounting can shift mid-drive.
- Expose current alignment as a rotation (quaternion or rotation matrix) that transforms phone-frame IMU readings into vehicle-frame readings.
- Provide a manual "reset alignment" trigger for debugging, but automatic detection is the primary and required mechanism — the end user should never be asked to manually align anything.

### 5.3 AI Speed & Vibration Filter
- A trained model (initially: 1D CNN or small LSTM/TCN, trained offline on IO-VNBD) that takes a windowed segment of vehicle-frame accelerometer + gyroscope data and outputs an estimate of forward vehicle speed, filtering out high-frequency road/engine noise.
- Model is trained in Python (PyTorch), exported to a mobile-inference format (TFLite or ONNX Runtime Mobile), and invoked from Dart via platform channel / plugin (e.g., `tflite_flutter`).
- This module's output is a **measurement input to the Kalman filter**, not a replacement for it — it does not directly set position.
- Must run inference fast enough to keep up with the target 10 Hz update rate with margin to spare.
- Model versioning: the engine must tolerate swapping in an updated model file without code changes (load from assets/local storage, not hardcoded).

### 5.4 INS Mechanization (Strapdown)
- Classical strapdown INS integration: gyro-integrated orientation, gravity-compensated accelerometer double-integration for position, all in vehicle/local navigation frame (after Calibration module's frame transform is applied).
- This produces the *raw* (uncorrected, will-drift) inertial trajectory that feeds into the Kalman filter as the process model / prediction step.
- Must expose intermediate state (raw drifted position/velocity/orientation) for debugging and for the Round 1 "before correction" comparison plot.

### 5.5 Kalman Filter (EKF or UKF) — Fusion Core
- State vector should include, at minimum: position (2D or 3D), velocity, heading, and relevant sensor bias terms (accelerometer bias, gyro bias) — bias estimation is required, not optional, since MEMS sensors have significant deterministic bias/drift.
- **Prediction step:** driven by INS mechanization output (Section 5.4) and/or the AI speed filter's velocity estimate.
- **Update step:** corrects using GNSS position/velocity (when available and above a quality threshold) and, once implemented, map-matching constraints (Section 5.6).
- Must implement **Non-Holonomic Constraints (NHC)** as an additional pseudo-measurement: lateral and vertical vehicle velocity ≈ 0 in the vehicle body frame (a car doesn't slide sideways or fly). This alone significantly bounds drift and should be treated as a first-class measurement update, not an afterthought.
- Filter must track and expose its own uncertainty (covariance) — this is required output, not internal-only state, since the Mode Manager (5.7) depends on it.
- Implementation choice: start with EKF (simpler Jacobians, well-documented for this exact GNSS/INS use case); leave the door open to swap to UKF later if EKF linearization error proves significant during testing. Document this decision point clearly in code comments.

### 5.6 Map-Matching (Phase 4+, not required for first working prototype)
- Consumes an offline OpenStreetMap extract (region around the test drive area, bundled or downloaded ahead of time — no live map API calls).
- Implements Hidden Markov Model based map-matching (candidate road segments per position estimate, transition probabilities based on road network connectivity and distance/heading plausibility) to snap the fused trajectory onto plausible roads.
- Feeds back into the Kalman filter as an additional correction/constraint once a confident road-segment match exists.
- Must degrade gracefully (i.e., not snap incorrectly) when off-road, in a parking lot, or in low-confidence conditions — false snapping is worse than no snapping.

### 5.7 Mode Manager (GNSS-aided ⇄ Dead Reckoning)
- Continuously evaluates GNSS signal quality (availability, accuracy/HDOP, number of satellites if exposed) to decide current mode: `GNSS_AIDED` or `DEAD_RECKONING`.
- Transition must be smooth in the *output* stream — no discontinuous jump in reported position even though the underlying estimation mode changed. Achieve this via the Kalman filter's natural handling of intermittent measurements (simply stop feeding GNSS updates when unavailable) rather than any separate blending/smoothing hack.
- Exposes current mode as part of the output stream (Section 5.8) so the UI can indicate this to the user.
- Target transition detection latency: within one sensor cycle (~100ms) of GNSS loss/reacquisition, not several seconds.

### 5.8 Output Interface
- A single well-defined Dart data class, e.g.:
```dart
class NavState {
  final DateTime timestamp;
  final double latitude;
  final double longitude;
  final double headingDegrees;
  final double pitchDegrees;
  final double rollDegrees;
  final double speedMps;
  final double positionUncertaintyMeters; // from filter covariance
  final NavMode mode; // GNSS_AIDED | DEAD_RECKONING
}
```
- Exposed as a `Stream<NavState>` that any consumer (the 3D visualizer, later a map UI) subscribes to.
- This is the seam between "engine" and "app" — the 3D visualizer built earlier should be refactored to consume this stream instead of raw orientation, once this engine exists.

---

## 6. Non-Functional Requirements

- **Update rate:** minimum 10 Hz sustained output, matching the PS benchmark for smartphone-class deployment.
- **Latency:** end-to-end sensor-sample-to-output-update latency should stay well under 100ms to feel "live" in the visualizer.
- **Battery/CPU:** must not peg a single core continuously — profile and optimize the inference and filter update loop if it becomes a bottleneck; this is a background-runnable service, not a one-shot computation.
- **Robustness:** must not crash or produce NaN/garbage state on sensor dropout, GNSS unavailability, or malformed sensor data — always fail toward "hold last known good state with growing uncertainty," never toward crashing or silently returning zero/garbage.
- **Testability:** every module (Calibration, INS mechanization, Kalman filter, Mode Manager) must be unit-testable independent of live sensors — accept injected/mocked sensor streams for testing, not just live device streams.
- **Portability:** engine logic should be pure Dart with no direct UI dependency, so it can later run headless (e.g., for offline batch evaluation against IO-VNBD logs) as well as live on-device.

---

## 7. Offline Evaluation Requirement (ties to Round 1 deliverable)

The engine's core algorithms (INS mechanization + Kalman fusion + AI speed filter) must be runnable **offline against logged data**, not only live on a phone. This is required so that:
- The AI speed filter can be trained/validated in Python against IO-VNBD.
- The full fusion pipeline (once ported/mirrored in Dart, or run in Python for evaluation purposes) can be tested against IO-VNBD or self-collected logs, producing the drift/RMSE metrics and position plots needed for submission.
- A `SensorService` implementation that reads from a recorded log file (CSV/JSON of timestamped sensor+GNSS rows) instead of live hardware must exist, implementing the same interface as the live version, so the rest of the engine is unaware whether it's running live or replayed.

---

## 8. Build Phases (for the agent to follow in order)

**Phase A — Sensor & Calibration scaffold**
- SensorService (live + log-replay implementations)
- Calibration & Alignment module
- Basic debug UI showing raw streams + computed alignment

**Phase B — INS + Kalman core**
- Strapdown INS mechanization
- EKF implementation with NHC pseudo-measurements
- Bias estimation in state vector
- Debug output comparing raw INS drift vs. filtered estimate

**Phase C — GNSS fusion + Mode Manager**
- GNSS measurement updates into the EKF
- Mode Manager with smooth transition behavior
- Full `NavState` output stream

**Phase D — AI Speed/Vibration Filter integration**
- Bring in the trained TFLite/ONNX model
- Wire its output as an EKF measurement input
- Compare filter performance with/without this module active

**Phase E — Map-Matching**
- OSM offline extract loading
- HMM-based matching
- Feed back into EKF as constraint

**Phase F — Visualization integration**
- Refactor the existing 3D visualizer to consume `Stream<NavState>` instead of raw orientation
- Add position trail rendering (not just orientation gizmo)

Do not skip ahead to a later phase before the current phase produces sane, testable output — each phase's correctness is a precondition for the next phase's fusion inputs being trustworthy.

---

## 9. Tech Stack Summary

| Layer | Choice |
|---|---|
| App framework | Flutter (Dart) |
| Raw sensors | `sensors_plus` |
| GNSS | `geolocator` |
| On-device ML inference | `tflite_flutter` (or `onnxruntime` Flutter bindings) |
| Offline map data | OpenStreetMap extract, loaded locally |
| Model training (offline, separate from app repo) | Python, PyTorch, trained against IO-VNBD |
| Filter math | Hand-implemented EKF in Dart (using `vector_math` for linear algebra), no black-box filter library |

---

## 10. Open Questions / Decisions the Agent Should Flag Rather Than Assume

- Exact state vector dimensionality and bias-modeling approach for the EKF (random walk vs. first-order Gauss-Markov bias model) — propose an approach with reasoning, don't silently pick one.
- Whether GNSS velocity (Doppler-derived) or GNSS position alone is used as the primary correction measurement — likely both, but confirm before implementing.
- Format/schema for the offline log-replay files — propose a CSV schema (columns, units, timestamp format) before implementing the log-replay SensorService.
- How the AI speed filter's output uncertainty (if any) should be represented and fed into the EKF's measurement noise covariance — the PS explicitly asks for AI-driven drift mitigation, so this coupling deserves a deliberate design, not a fixed constant.

---

*End of document.*
