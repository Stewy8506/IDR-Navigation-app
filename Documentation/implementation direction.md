IDR-Nav — High-Capacity Physics-Guided Neural Navigation Implementation Specification

Objective

Rework the current AI speed and heading estimators into high-capacity, physics-guided neural observers designed for full-drive inertial navigation, not merely short GPS-blackout windows.

The system must:

* use zero GPS-derived inference features
* use no frozen GPS priors or position/speed priors
* operate from the 18-channel temporal IMU representation
* produce velocity, velocity increment, uncertainty, ZUPT probability, and pitch information
* provide neural measurements to the existing 15-state error-state EKF
* generalize across complete drives and unseen driving regimes
* specifically address the current Driver E high-speed generalization failure
* remain comfortably below the 10–20 ms execution budget
* preserve deterministic, reproducible evaluation

The neural models are observers/measurement generators, not replacements for the physical navigation filter.

⸻

1. Critical Architecture Rules

1.1 Zero GPS inference dependency

GPS may be used only for:

* training labels
* offline validation
* offline evaluation
* trajectory visualization

GPS must NEVER be passed to the neural networks or runtime EKF as an input feature or frozen prior.

No:

* frozen speed
* frozen position
* GPS-derived velocity
* GPS-derived acceleration
* GPS-derived heading
* GPS-derived trajectory features

may enter runtime inference.

⸻

2. Correct the Pitch Feature Design

Do NOT create a circular dependency where the network’s predicted pitch is fed back into the same network as an input.

The pitch/gravity feature must come from an independent physical attitude observer.

Implement:

raw IMU
    ↓
physical pitch observer
    ↓
pitch_phys
    ↓
gravity-compensated longitudinal acceleration

For example:

theta_acc =
    atan2(a_y, sqrt(a_x^2 + a_z^2))

with appropriate filtering/observer logic.

Then construct:

a_y_gravity_corrected =
    a_y - g * sin(theta_phys)

The neural network may independently predict:

pitch_neural

but pitch_neural must not be used to construct its own input feature.

⸻

3. Dataset

Modify:

ml/src/dataset_spectral.py

3.1 Temporal window

Use:

48 samples × 18 channels

At 10 Hz this represents:

4.8 seconds

of temporal context.

The window must be causal with respect to the current output. Do not accidentally include future samples.

⸻

4. 18 Input Features

Use the following feature representation.

Channels 0–5

Vehicle-frame IMU:

0: ax
1: ay
2: az
3: wx
4: wy
5: wz

Channels 6–7

Dynamic norms:

6: ||a|| - g
7: ||omega||

Channel 8

Multi-scale leaky longitudinal velocity-integral features.

Use carefully bounded/normalized integration with:

tau = 0.90
tau = 0.98

Do not use GPS to initialize these integrals.

The purpose is to provide inertial temporal context, not a GPS-derived speed prior.

Channel 9

Vertical suspension dynamic variance:

Var(az)

Use a causal temporal estimator.

Channels 10–12

Pavement-normalized spectral ratios:

R_low
R_mid
R_high

Ensure normalization is derived only from training-set statistics and is frozen during inference.

Do not calculate normalization using the entire evaluation drive.

Channel 13

Spectral centroid frequency:

f_centroid

Channel 14

Dominant/harmonic peak frequency:

f_peak

constrained to:

1–25 Hz

Channel 15

Kinematic turning feature:

abs(ax / (wz + epsilon))

with a numerically safe epsilon and sensible clipping.

Channel 16

Longitudinal/vertical vibration ratio:

P_ay / P_az

with safe denominator handling.

Channel 17

Gravity-compensated longitudinal acceleration:

ay - g * sin(theta_phys)

where theta_phys comes exclusively from the independent physical pitch observer.

⸻

5. Dataset Splitting — CRITICAL

Do NOT randomly split individual temporal windows.

Adjacent windows from the same drive must not appear simultaneously in train and validation/test sets.

The split must be drive-aware and preferably driver-aware.

Primary configuration:

Training:
Drivers A, B, D
Validation:
held-out drives from A/B/D
Final test:
Driver E

Additionally implement leave-one-driver-out evaluation where practical.

The Driver E test set must remain completely untouched during model development and hyperparameter selection.

⸻

6. Address the Existing Driver E Failure Explicitly

The previous system demonstrated severe high-speed distribution-shift behavior on Driver E.

Do not assume increasing parameter count will fix this.

Before training, generate velocity-distribution statistics for every driver:

0–10 km/h
10–20
20–30
30–40
40–50
50–60
60–80
80+

Report the sample count in each regime.

Implement balanced or weighted training so rare high-speed regimes are not ignored.

Prefer:

1. velocity-regime-aware sampling
2. velocity-regime-aware loss weighting

Use both if computationally practical.

Do not allow the model to simply learn the mean velocity of the training distribution.

⸻

7. DeepSpeedKinematicsNet

Modify:

ml/src/model.py

Implement:

DeepSpeedKinematicsNet

Target approximately:

~1M–2M parameters

Do not artificially inflate the model merely to hit a parameter count.

The objective is useful capacity, not maximum parameter count.

⸻

8. Backbone

Use a 1D temporal ConvNeXt-style architecture.

Recommended:

Input:
48 × 18
Stage 1:
Conv1D
kernel = 7
LayerNorm
GELU
depthwise temporal convolution
pointwise expansion ≈ 4×
Stage 2:
same general structure
Stage 3:
same general structure
Stage 4:
same general structure

Maintain temporal resolution where useful.

Use residual connections.

Avoid unnecessary downsampling that destroys temporal information.

⸻

9. Temporal Self-Attention

After the ConvNeXt temporal backbone, use:

4-head temporal self-attention

over the temporal dimension.

Input:

48 temporal tokens

The attention layer must remain lightweight enough for mobile CPU inference.

Use residual connection + normalization.

⸻

10. Multi-Task Outputs

The network must output:

Head 1 — Velocity

mu_v
log_variance_v

Represent variance using a numerically stable parameterization such as:

log_sigma2

rather than directly predicting unrestricted variance.

The final variance must be positive and bounded to reasonable limits.

⸻

Head 2 — Velocity Increment

Predict:

delta_v

The primary target should be the local physical velocity increment:

delta_v[t] = v[t] - v[t-1]

Do not ambiguously define this as an arbitrary multi-second integral.

Optionally include a longer-horizon velocity-difference auxiliary target if useful, but the primary delta target must remain clearly defined.

⸻

Head 3 — ZUPT Probability

Predict:

P_ZUPT

Use sigmoid output.

The ZUPT target must be generated from the ground-truth motion state using an explicit and documented threshold/hysteresis procedure.

Do not use raw ZUPT accuracy as the main metric.

Evaluate:

precision
recall
F1
false-positive rate during motion
false-negative rate during standstill
PR-AUC

False positive ZUPT events during motion are especially important because they can corrupt EKF velocity.

⸻

Head 4 — Dynamic Pitch

Predict:

pitch_neural

This is an independent learned pitch estimate/correction.

Do not use it to construct the input feature that the network consumes.

⸻

11. Optional Dynamic-Regime Head

If parameter/latency budget permits, add a lightweight auxiliary head predicting a dynamic regime/confidence representation.

For example:

stationary
low-speed
cruise
acceleration
braking
turning
high-speed

This may be implemented as a compact classification head.

It is auxiliary only.

Do not allow it to become a shortcut for velocity.

If it materially hurts generalization or latency, remove it.

⸻

12. DeepHeadingObserverNet

Implement:

DeepHeadingObserverNet

using a lightweight temporal convolutional architecture.

Input should include the appropriate raw IMU channels and temporal context.

Outputs:

gyro_bias_z
delta_omega_z
bias_uncertainty

Conceptually:

omega_z_corrected =
    omega_z
    - gyro_bias_z
    + delta_omega_z

The model must learn bias dynamics rather than merely denoise instantaneous gyro measurements.

The uncertainty output must be usable by the EKF.

⸻

13. Training Objective

Modify:

ml/src/train_spectral.py

Use AdamW.

Use:

Cosine Annealing with Warm Restarts

and gradient clipping.

Start with the following multi-task objective:

L_total =
    L_velocity_Huber
    + 0.15 * L_velocity_NLL
    + 0.5 * L_delta_velocity_L1
    + 0.2 * L_ZUPT_BCE
    + L_pitch

The pitch loss weight should be tuned rather than assumed.

Use appropriate normalization for each task.

⸻

14. Heteroscedastic Velocity Loss

The model predicts:

mu_v
log_sigma2_v

Use Gaussian/heteroscedastic NLL:

L_NLL =
0.5 * (
    exp(-log_sigma2) * (v - mu_v)^2
    + log_sigma2
)

with numerical clamping.

Do not allow the network to inflate uncertainty indefinitely to reduce the loss.

Use sensible minimum/maximum variance bounds.

⸻

15. Make Uncertainty Actually Useful

The predicted velocity uncertainty must not be merely logged.

Convert it into the EKF measurement covariance:

R_velocity =
sigma_v^2 + sigma_floor^2

and use that measurement covariance during the neural velocity update.

The model should therefore communicate:

estimated velocity
+
how much the EKF should trust it

⸻

16. EKF Integration

Modify the existing 15-state error-state EKF integration.

The neural network is a measurement provider.

The physical INS remains responsible for state propagation.

The architecture should conceptually be:

                 Raw IMU
                    │
                    ↓
             Physical propagation
                    │
                    ↓
              15-state EKF
                    ↑
       ┌────────────┼─────────────┐
       │            │             │
       │            │             │
 neural velocity   ZUPT      heading observer
       │            │             │
       │            │             │
       └────────────┴─────────────┘
                    │
                    ↓
              corrected state
                    │
                    ↓
             Frenet tracker
                    │
                    ↓
             final trajectory

Do NOT replace the EKF with direct neural position regression.

⸻

17. ZUPT Integration

ZUPT must be probabilistic.

Do not blindly apply a ZUPT update whenever:

P_ZUPT > 0.5

Investigate a threshold/hysteresis mechanism.

For example:

enter ZUPT:
P_ZUPT > threshold_high
remain ZUPT:
P_ZUPT > threshold_low

with:

threshold_high > threshold_low

to prevent chattering.

The EKF should also reject implausible ZUPT events using physical consistency checks.

⸻

18. Heading Integration

The heading observer should feed:

gyro bias estimate
gyro correction
bias uncertainty

into the EKF.

Do not simply overwrite the physical gyro measurement.

The EKF should maintain the underlying bias state.

⸻

19. Export

Modify:

ml/src/export_onnx.py

Export:

DeepSpeedKinematicsNet
→ app/assets/models/speed_filter.onnx
DeepHeadingObserverNet
→ app/assets/models/heading_filter.onnx

Ensure:

* fixed input dimensions where practical
* deterministic inference
* compatible ONNX operators
* no training-only operations
* FP32 baseline first
* quantization only after baseline correctness is established

⸻

20. Latency Benchmark

Modify:

ml/src/benchmark_latency.py

Run at least:

1000 iterations

after warmup.

Report:

mean
median
P90
P95
P99
min
max

Acceptance:

P95 < 5 ms preferred
P99 < 10 ms preferred
absolute requirement < 20 ms

Do not claim 1.5–2.5 ms before measuring it.

⸻

21. Full-Drive Evaluation

Modify:

ml/src/task3_indistribution_evaluation.py
ml/src/evaluate_full_pipeline.py

Evaluate complete drives, not only short blackout segments.

For every drive report:

velocity MAE
velocity RMSE
velocity bias
velocity correlation
velocity MAE by speed regime
ZUPT precision
ZUPT recall
ZUPT F1
trajectory ATE
trajectory RPE
final position error
maximum position error
heading error

⸻

22. Required Baselines

Every evaluation must compare against:

Baseline 1

Pure physical INS / NHC baseline.

Baseline 2

Current production AI system.

Baseline 3

New neural observer + EKF.

The new system must demonstrate that the additional model capacity actually improves navigation rather than merely improving standalone velocity prediction.

⸻

23. Critical Benchmark

The existing benchmark must be preserved.

The new model must beat the existing strong physical baseline, not merely beat the current neural model.

Use the previous results as regression references.

The target hierarchy is:

New neural + EKF
        ↓
must beat current AI pipeline
        ↓
should beat pure INS/NHC baseline
        ↓
stretch goal:
very low full-drive drift

Do not declare success merely because Driver E improves from the current failure.

⸻

24. Full-Drive Robustness Evaluation

Evaluate at multiple outage durations:

30 s
60 s
90 s
full drive

For each, report:

ATE
RPE
final error
maximum error
velocity MAE
heading error

The full-drive test is the primary objective.

Short blackout performance is secondary.

⸻

25. Driver E Must Remain a True Holdout

Do not tune hyperparameters against Driver E.

The development cycle must be:

train A/B/D
        ↓
validate held-out A/B/D drives
        ↓
freeze architecture + hyperparameters
        ↓
evaluate E once as final OOD test

If Driver E is repeatedly inspected during development, create a new untouched holdout for final scientific evaluation.

⸻

26. Flutter Runtime

Modify:

app/lib/ai/speed_filter_runner.dart

Implement:

48-step rolling buffer
18-channel feature extraction
ONNX speed model
ONNX heading model

Expose:

SpeedEstimate.velocity
SpeedEstimate.velocityVariance
SpeedEstimate.zuptProbability
SpeedEstimate.deltaVelocity
SpeedEstimate.pitch

Ensure the feature extractor is numerically identical to the Python implementation.

Do not maintain two subtly different feature definitions.

⸻

27. Python ↔ Dart Feature Parity

Create a deterministic parity test.

For identical raw IMU samples:

Python feature extractor
vs
Dart feature extractor

must agree within a defined numerical tolerance.

Test every one of the 18 channels.

This is mandatory.

A model that works in Python but receives different features on-device is considered a failure.

⸻

28. Flutter Latency

Modify:

app/test/pipeline_latency_benchmark_test.dart

Measure:

feature extraction
+
speed ONNX inference
+
heading ONNX inference
+
per-cycle overhead

Report:

mean
P95
P99

Ensure the entire 10 Hz cycle comfortably fits within:

100 ms

and the neural portion remains below:

20 ms

⸻

29. Automated Verification

Run:

ml/venv/bin/python3 -m ml.src.train_spectral

Then:

ml/venv/bin/python3 -m ml.src.export_onnx
ml/venv/bin/python3 -m ml.src.benchmark_latency

Then:

ml/venv/bin/python3 -m ml.src.task3_indistribution_evaluation
ml/venv/bin/python3 -m ml.src.evaluate_full_pipeline

Then:

cd app
flutter test

⸻

30. Initial Acceptance Criteria

Training

Target:

validation velocity MAE < 4.5 km/h

but do NOT accept this criterion alone.

Also require:

no severe velocity-regime collapse

and report MAE independently for each speed bin.

⸻

ZUPT

Do not use:

accuracy > 99%

as the acceptance criterion.

Instead establish:

high F1
low motion false-positive rate
acceptable standstill recall

with exact measured values reported.

⸻

Latency

Required:

P99 < 20 ms

Preferred:

P95 < 5 ms

⸻

Navigation

Primary goals:

Driver A full-drive drift < 5 m
Driver E full-drive drift < 30 m

These are stretch targets, not reasons to hide regressions elsewhere.

Also compare directly against all existing baselines.

⸻

31. Do Not Implement Blindly

Before modifying files:

1. Inspect the existing dataset schema.
2. Inspect current feature extraction.
3. Inspect current model architecture.
4. Inspect current training targets.
5. Inspect current EKF state definition.
6. Inspect current ONNX input/output contracts.
7. Inspect current Dart feature extraction.
8. Inspect existing tests.
9. Inspect the current V4 evaluation outputs.
10. Identify any existing assumptions about sample rate, coordinate frames, units, and timestamps.

Do not overwrite existing functionality until the current pipeline is understood.

Preserve backward compatibility where practical.

⸻

32. Unit/Frame Audit

Before training, explicitly verify:

accelerometer units
gyro units
velocity units
position units
timestamp units
coordinate frames
vehicle-frame convention
sign conventions
gravity direction
yaw convention
pitch convention

Especially verify whether:

ay

is actually the longitudinal axis in the current dataset.

Do not assume the axis labels in this specification override the actual dataset schema.

⸻

33. Reproducibility

Training must be reproducible.

Record:

random seed
dataset split
normalization statistics
model parameter count
training configuration
optimizer configuration
learning-rate schedule
epoch count
best checkpoint
git commit

Save all validation metrics.

⸻

34. Deliverables

At completion provide:

1. Modified Python source
2. Trained model checkpoints
3. ONNX models
4. Updated Dart runtime
5. Updated tests
6. Training metrics
7. Latency benchmark
8. Python/Dart feature parity results
9. Full-drive evaluation results
10. Driver A results
11. Driver E results
12. Baseline comparison
13. Parameter counts
14. Exact train/validation/test split
15. Any known failure modes

⸻

35. Final Reporting Format

The final agent report must contain:

MODEL
- parameter count
- architecture
- input shape
- output shape
TRAINING
- dataset
- split
- epochs
- best validation MAE
- speed-bin MAE
ZUPT
- precision
- recall
- F1
- motion FPR
LATENCY
- mean
- P95
- P99
- max
NAVIGATION
- baseline INS/NHC
- old AI
- new AI + EKF
- Driver A
- Driver E
- full-drive error
- 30/60/90s outage error
RUNTIME
- Python/Dart feature parity
- Flutter latency
- test count
- failures
REGRESSIONS
- any metric that became worse
- any known failure modes

⸻

Final Instruction

Implement this as a physics-guided learned inertial observer, not as a black-box position predictor.

The physical system remains responsible for propagation and state consistency.

The neural system provides:

velocity
velocity increment
velocity uncertainty
ZUPT probability
pitch estimate
gyro correction
gyro bias
uncertainty

The EKF performs the actual state fusion.

The central scientific objective is:

Extract the maximum useful information from the temporal IMU signal while preventing GPS shortcuts and improving generalization across unseen drives and high-speed regimes.

Do not optimize solely for validation MAE.

Do not optimize solely for Driver E.

Do not increase model size without measuring whether it improves OOD full-drive navigation.

Every architectural change must ultimately be judged by:

full-drive inertial navigation accuracy
+
cross-drive generalization
+
physical consistency
+
runtime latency

Proceed incrementally. After each major component, run the relevant tests before moving to the next component. Do not make unrelated changes to the application.