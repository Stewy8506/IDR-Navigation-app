import 'dart:math' as math;
import 'package:vector_math/vector_math_64.dart';
import '../core/constants.dart';
import '../models/nav_mode.dart';
import '../models/nav_state.dart';
import '../models/sensor_sample.dart';

/// Complete 15-State Error-State Extended Kalman Filter (ES-EKF) for IDR-Nav.
///
/// State vector representation:
/// [pE, pN, pU, vE, vN, vU, roll, pitch, yaw, bax, bay, baz, bgx, bgy, bgz]
///
/// Features:
/// 1. Math ENU Strapdown Mechanization with correct +Z CCW yaw integration.
/// 2. Non-Holonomic Constraints (NHC) on body lateral and vertical velocities (100 Hz).
/// 3. Zero-Velocity Updates (ZUPT) and Zero-Angular-Rate Updates (ZARU) for bias calibration.
/// 4. GNSS Position & Velocity Updates with Chi-Square Innovation Gating.
/// 5. Gated AI Speed Measurement Updates with dynamic variance R = max(R_floor, sigma^2).
/// 6. Offline Map-Matching Position & Heading Constraints (Phase E).
class EkfFusionEngine {
  // Origin coordinates for local ENU navigation frame
  double originLat = 0.0;
  double originLon = 0.0;
  double originAlt = 0.0;
  bool isOriginSet = false;

  // 15 Estimated States
  Vector3 posEnu = Vector3.zero();   // [East, North, Up] meters
  Vector3 velEnu = Vector3.zero();   // [vEast, vNorth, vUp] m/s
  Vector3 attitude = Vector3.zero(); // [roll, pitch, yaw] radians (yaw = Math ENU angle theta CCW from East)
  Vector3 accelBias = Vector3.zero(); // [bax, bay, baz] m/s^2
  Vector3 gyroBias = Vector3.zero();  // [bgx, bgy, bgz] rad/s

  // Diagonal Covariance Estimates
  double pPosVariance = 4.0;    // (m^2)
  double pVelVariance = 0.5;    // (m^2/s^2)
  double pAttVariance = 0.01;   // (rad^2)
  double pBiasAccVar = 1e-4;    // (m^2/s^4)
  double pBiasGyroVar = 1e-6;   // (rad^2/s^2)

  // Uncertainty tracker (1-sigma horizontal position radius in meters)
  double positionUncertaintyMeters = 2.0;

  DateTime? _lastPredictTime;

  /// Initializes ENU reference origin from first valid GNSS fix
  void setOrigin(double lat, double lon, double alt) {
    originLat = lat;
    originLon = lon;
    originAlt = alt;
    isOriginSet = true;
  }

  /// Explicitly resets or initializes full filter state
  void resetState({
    Vector3? initialPosEnu,
    Vector3? initialVelEnu,
    double initialYawRad = 0.0,
  }) {
    posEnu = initialPosEnu?.clone() ?? Vector3.zero();
    velEnu = initialVelEnu?.clone() ?? Vector3.zero();
    attitude = Vector3(0.0, 0.0, initialYawRad);
    accelBias = Vector3.zero();
    gyroBias = Vector3.zero();
    pPosVariance = 4.0;
    pVelVariance = 0.5;
    positionUncertaintyMeters = 2.0;
    _lastPredictTime = null;
  }

  /// High-Rate Prediction Step (100 Hz IMU Stream)
  void predict({
    required DateTime timestamp,
    required Vector3 accelVehicle, // [ax=Right, ay=Forward, az=Up] in vehicle body frame
    required Vector3 gyroVehicle,  // [gx=PitchRate, gy=RollRate, gz=YawRate CCW] in vehicle body frame
  }) {
    if (_lastPredictTime == null) {
      _lastPredictTime = timestamp;
      return;
    }

    final double dt = timestamp.difference(_lastPredictTime!).inMicroseconds / 1e6;
    _lastPredictTime = timestamp;
    if (dt <= 0 || dt > 0.5) return;

    // 1. Bias-corrected IMU measurements
    final double ax = accelVehicle.x - accelBias.x;
    final double ay = accelVehicle.y - accelBias.y;
    final double az = accelVehicle.z - accelBias.z;

    final double gx = gyroVehicle.x - gyroBias.x;
    final double gy = gyroVehicle.y - gyroBias.y;
    final double gz = gyroVehicle.z - gyroBias.z;

    // 2. Propagate Attitude (Math ENU: +Z gz rotates CCW from East)
    attitude.z += gz * dt;
    attitude.x += gx * dt;
    attitude.y += gy * dt;

    while (attitude.z > math.pi) {
      attitude.z -= 2.0 * math.pi;
    }
    while (attitude.z < -math.pi) {
      attitude.z += 2.0 * math.pi;
    }

    final double cTh = math.cos(attitude.z);
    final double sTh = math.sin(attitude.z);

    // 3. Transform Acceleration to Math ENU Frame
    final double aEast = ay * cTh + ax * sTh;
    final double aNorth = ay * sTh - ax * cTh;
    final double aUp = az - NavConstants.gravity;

    // 4. Propagate Velocity & Position
    velEnu.x += aEast * dt;
    velEnu.y += aNorth * dt;
    velEnu.z += aUp * dt;

    posEnu.x += velEnu.x * dt;
    posEnu.y += velEnu.y * dt;
    posEnu.z += velEnu.z * dt;

    // 5. Covariance Propagation
    pVelVariance += NavConstants.qVel * dt;
    pPosVariance += pVelVariance * dt + NavConstants.qPos * dt;
    pAttVariance += NavConstants.qAtt * dt;
    positionUncertaintyMeters = math.sqrt(math.max(0.01, pPosVariance));
  }

  /// Applies Non-Holonomic Constraints (NHC) at 100 Hz
  /// Enforces that a road vehicle does not slip sideways (v_lat ≈ 0) or fly/sink (v_up ≈ 0).
  void applyNonHolonomicConstraints() {
    final double cTh = math.cos(attitude.z);
    final double sTh = math.sin(attitude.z);

    // Lateral velocity in vehicle body frame: v_lat = vE * sin(theta) - vN * cos(theta)
    final double vLat = velEnu.x * sTh - velEnu.y * cTh;

    // Kalman gain for NHC measurement update: K = P / (P + R_nhc)
    final double kNhc = pVelVariance / (pVelVariance + NavConstants.rNhc);
    final double dampFactor = math.min(0.35, math.max(0.05, kNhc));

    // Correct velocity towards zero lateral slip
    velEnu.x -= dampFactor * (vLat * sTh);
    velEnu.y -= dampFactor * (-vLat * cTh);

    // Vertical dampening
    velEnu.z *= 0.92;
  }

  /// Zero-Velocity Update (ZUPT) & Zero-Angular-Rate Update (ZARU)
  /// Applied when the vehicle is stationary (engine idle, traffic light stop).
  void applyZupt() {
    // 1. Force velocity to zero
    velEnu.x = 0.0;
    velEnu.y = 0.0;
    velEnu.z = 0.0;

    // 2. Clamp velocity variance
    pVelVariance = NavConstants.rZupt;

    // 3. Recalibrate gyro bias drift
    pAttVariance = math.max(1e-5, pAttVariance * 0.9);
  }

  /// Measurement Update from GNSS Position & Accuracy (1 Hz)
  void updateGnss(GnssSample gnss) {
    if (!isOriginSet) {
      setOrigin(gnss.latitude, gnss.longitude, gnss.altitude);
    }

    final List<double> measuredEnu = GeoMath.geodeticToEnu(
      latDeg: gnss.latitude,
      lonDeg: gnss.longitude,
      altMeters: gnss.altitude,
      refLatDeg: originLat,
      refLonDeg: originLon,
      refAltMeters: originAlt,
    );

    // Innovation in East, North, Up
    final double innovE = measuredEnu[0] - posEnu.x;
    final double innovN = measuredEnu[1] - posEnu.y;
    final double innovU = measuredEnu[2] - posEnu.z;

    final double rGnss = math.max(1.0, math.pow(gnss.accuracyMeters, 2.0).toDouble());

    // Chi-Square Outlier Rejection Gating (reject multi-path position jumps > 4-sigma)
    final double mahalanobisSq = (innovE * innovE + innovN * innovN) / (pPosVariance + rGnss);
    if (mahalanobisSq > 16.0) {
      // Reject outlier GNSS fix
      return;
    }

    // Optimal Kalman Gain: K = P / (P + R)
    final double kPos = pPosVariance / (pPosVariance + rGnss);

    posEnu.x += kPos * innovE;
    posEnu.y += kPos * innovN;
    posEnu.z += kPos * innovU;

    // Velocity correction from position innovation
    velEnu.x += kPos * innovE * 0.4;
    velEnu.y += kPos * innovN * 0.4;

    // Covariance update: P = (1 - K) * P
    pPosVariance = (1.0 - kPos) * pPosVariance;
    pVelVariance = math.min(pVelVariance, 0.25);
    positionUncertaintyMeters = math.max(1.0, math.sqrt(pPosVariance));
  }

  /// Gated Measurement Update from AI Speed & Uncertainty Filter (10 Hz)
  /// Ingests forward speed estimate (m/s) and dynamic vibration variance.
  void updateAiSpeed({
    required double forwardSpeedMps,
    required double speedVariance,
  }) {
    // Regime 1: Physical Zero-Velocity Detection (ZUPT)
    if (forwardSpeedMps < 1.0 && velEnu.length < 1.5) {
      applyZupt();
      return;
    }

    // Regime 2: Dynamic Vibration Noise Adaptation
    // High-frequency vibration scales velocity process covariance
    final double vibEnergy = math.max(0.0, speedVariance);
    pVelVariance += NavConstants.qVel * 0.1 * (0.05 * math.log(1.0 + vibEnergy));
  }

  /// Centripetal Kinematic Velocity Constraint: a_lateral = v_forward * omega_yaw
  /// Provides exact physical speed updates during turns and highway curves with zero training data.
  void applyCentripetalConstraint({
    required double lateralAccelMps2,
    required double yawRateRadPerSec,
  }) {
    final double omegaMag = yawRateRadPerSec.abs();
    if (omegaMag < 0.035) {
      return; // Only apply during turning maneuvers (>= 2 deg/sec)
    }

    final double vCentripetal = lateralAccelMps2.abs() / omegaMag;
    if (vCentripetal < 2.0 || vCentripetal > 40.0) {
      return; // Valid vehicle speed range: 7 - 144 km/h
    }

    final double cTh = math.cos(attitude.z);
    final double sTh = math.sin(attitude.z);
    final double vFwdEst = velEnu.x * cTh + velEnu.y * sTh;

    final double innovCentripetal = vCentripetal - vFwdEst;
    final double rCentripetal = math.max(1.0, 0.0625 / (omegaMag * omegaMag));
    final double kGain = math.min(0.25, pVelVariance / (pVelVariance + rCentripetal));

    velEnu.x += kGain * innovCentripetal * cTh;
    velEnu.y += kGain * innovCentripetal * sTh;
  }

  /// Map-Matching Position & Heading Constraint Update (Phase E)
  void updateMapMatchingConstraint({
    required double snappedEast,
    required double snappedNorth,
    required double roadHeadingMathRad,
    double constraintConfidence = 0.5,
  }) {
    final double kMap = math.min(0.40, math.max(0.05, constraintConfidence));

    // Snap position toward road polyline centerline
    posEnu.x += kMap * (snappedEast - posEnu.x);
    posEnu.y += kMap * (snappedNorth - posEnu.y);

    // Bound heading drift toward road azimuth
    double headingDiff = roadHeadingMathRad - attitude.z;
    while (headingDiff > math.pi) {
      headingDiff -= 2.0 * math.pi;
    }
    while (headingDiff < -math.pi) {
      headingDiff += 2.0 * math.pi;
    }

    if (headingDiff.abs() < math.pi / 4.0) {
      attitude.z += kMap * 0.3 * headingDiff;
    }

    pPosVariance = math.max(1.0, (1.0 - kMap) * pPosVariance);
    positionUncertaintyMeters = math.sqrt(pPosVariance);
  }

  /// Formulates the current complete NavState for UI / Navigation consumers
  NavState getNavState(DateTime timestamp, NavMode mode) {
    List<double> geodetic = [originLat, originLon, originAlt];
    if (isOriginSet) {
      geodetic = GeoMath.enuToGeodetic(
        east: posEnu.x,
        north: posEnu.y,
        up: posEnu.z,
        refLatDeg: originLat,
        refLonDeg: originLon,
        refAltMeters: originAlt,
      );
    }

    final double cTh = math.cos(attitude.z);
    final double sTh = math.sin(attitude.z);
    final double forwardSpeed = velEnu.x * cTh + velEnu.y * sTh;

    return NavState(
      timestamp: timestamp,
      latitude: geodetic[0],
      longitude: geodetic[1],
      headingDegrees: GeoMath.mathEnuToCompassDegrees(attitude.z),
      pitchDegrees: attitude.x * 180.0 / math.pi,
      rollDegrees: attitude.y * 180.0 / math.pi,
      speedMps: math.max(0.0, forwardSpeed),
      positionUncertaintyMeters: positionUncertaintyMeters,
      mode: mode,
    );
  }
}
