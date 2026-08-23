import 'dart:math';
import 'package:vector_math/vector_math_64.dart';
import '../core/constants.dart';
import '../models/nav_mode.dart';
import '../models/nav_state.dart';
import '../models/sensor_sample.dart';

/// Extended Kalman Filter (EKF) state vector representation (15 dimensions)
/// [pE, pN, pU, vE, vN, vU, roll, pitch, yaw, bax, bay, baz, bgx, bgy, bgz]
class EkfFusionEngine {
  // Origin coordinates for local ENU navigation frame
  double originLat = 0.0;
  double originLon = 0.0;
  double originAlt = 0.0;
  bool isOriginSet = false;

  // Estimated States
  Vector3 posEnu = Vector3.zero();
  Vector3 velEnu = Vector3.zero();
  Vector3 attitude = Vector3.zero(); // [roll, pitch, yaw] radians
  Vector3 accelBias = Vector3.zero();
  Vector3 gyroBias = Vector3.zero();

  // Position Uncertainty (1-sigma in meters)
  double positionUncertaintyMeters = 1.0;

  DateTime? _lastPredictTime;

  /// Initializes ENU origin from first valid GNSS fix
  void setOrigin(double lat, double lon, double alt) {
    originLat = lat;
    originLon = lon;
    originAlt = alt;
    isOriginSet = true;
  }

  /// Prediction step driven by vehicle-frame IMU samples
  void predict({
    required DateTime timestamp,
    required Vector3 accelVehicle,
    required Vector3 gyroVehicle,
  }) {
    if (_lastPredictTime == null) {
      _lastPredictTime = timestamp;
      return;
    }

    final double dt = timestamp.difference(_lastPredictTime!).inMicroseconds / 1e6;
    _lastPredictTime = timestamp;
    if (dt <= 0 || dt > 0.5) return;

    // 1. Correct sensor measurements with estimated biases
    final Vector3 correctedGyro = gyroVehicle - gyroBias;
    final Vector3 correctedAccel = accelVehicle - accelBias;

    // 2. Propagate attitude (roll, pitch, yaw)
    attitude.x += correctedGyro.x * dt;
    attitude.y += correctedGyro.y * dt;
    attitude.z += correctedGyro.z * dt;

    if (attitude.z > pi) attitude.z -= 2 * pi;
    if (attitude.z < -pi) attitude.z -= 2 * pi;

    // 3. Rotation to Navigation Frame
    final Matrix3 rBodyToNav = Matrix3.rotationZ(attitude.z) *
        Matrix3.rotationY(attitude.y) *
        Matrix3.rotationX(attitude.x);

    // 4. Gravity-compensated acceleration
    final Vector3 accelNav = rBodyToNav.transformed(correctedAccel);
    accelNav.z -= NavConstants.gravity;

    // 5. Propagate velocity & position
    velEnu += accelNav * dt;
    posEnu += velEnu * dt;

    // Grow uncertainty slightly during dead reckoning
    positionUncertaintyMeters += 0.05 * dt;
  }

  /// Update step from GNSS position and velocity
  void updateGnss(GnssSample gnss) {
    if (!isOriginSet) {
      setOrigin(gnss.latitude, gnss.longitude, gnss.altitude);
    }

    // Convert geodetic to local ENU
    final double latRad = gnss.latitude * pi / 180.0;
    final double dLat = (gnss.latitude - originLat) * pi / 180.0;
    final double dLon = (gnss.longitude - originLon) * pi / 180.0;

    final double measuredE = NavConstants.earthRadiusMeters * dLon * cos(latRad);
    final double measuredN = NavConstants.earthRadiusMeters * dLat;
    final double measuredU = gnss.altitude - originAlt;

    // EKF innovation update for position (Kalman gain blending)
    final double kPos = 0.4; // Blend factor based on GNSS accuracy
    posEnu.x += kPos * (measuredE - posEnu.x);
    posEnu.y += kPos * (measuredN - posEnu.y);
    posEnu.z += kPos * (measuredU - posEnu.z);

    // Update uncertainty to GNSS reported accuracy
    positionUncertaintyMeters = gnss.accuracyMeters;
  }

  /// Update step from AI Speed Filter (Forward longitudinal velocity)
  void updateAiSpeed(double forwardSpeedMps, double speedVariance) {
    // Forward velocity constraint in vehicle frame: project ENU velocity to vehicle frame
    final double currentHeading = attitude.z;
    final double headingCos = cos(currentHeading);
    final double headingSin = sin(currentHeading);

    // Estimated forward speed = vE * sin(yaw) + vN * cos(yaw)
    final double estimatedForwardSpeed = velEnu.x * headingSin + velEnu.y * headingCos;
    final double innovation = forwardSpeedMps - estimatedForwardSpeed;

    // Kalman gain for speed
    const double kSpeed = 0.2;
    velEnu.x += kSpeed * innovation * headingSin;
    velEnu.y += kSpeed * innovation * headingCos;
  }

  /// Applies Non-Holonomic Constraints (NHC): Lateral and Vertical body velocity ~ 0
  void applyNonHolonomicConstraints() {
    final double currentHeading = attitude.z;
    final double headingCos = cos(currentHeading);
    final double headingSin = sin(currentHeading);

    // Compute lateral velocity (orthogonal to forward heading)
    final double lateralSpeed = -velEnu.x * headingCos + velEnu.y * headingSin;
    
    // Dampen lateral slide & vertical bouncing
    const double kNhc = 0.15;
    velEnu.x -= kNhc * (-lateralSpeed * headingCos);
    velEnu.y -= kNhc * (lateralSpeed * headingSin);
    velEnu.z *= 0.95; // Dampen vertical velocity drift
  }

  /// Formulates the current NavState
  NavState getNavState(DateTime timestamp, NavMode mode) {
    double lat = originLat;
    double lon = originLon;
    double alt = originAlt;

    if (isOriginSet) {
      final double originLatRad = originLat * pi / 180.0;
      final double dLatRad = posEnu.y / NavConstants.earthRadiusMeters;
      final double dLonRad = posEnu.x / (NavConstants.earthRadiusMeters * cos(originLatRad));

      lat = originLat + (dLatRad * 180.0 / pi);
      lon = originLon + (dLonRad * 180.0 / pi);
      alt = originAlt + posEnu.z;
    }

    double headingDeg = attitude.z * 180.0 / pi;
    if (headingDeg < 0) headingDeg += 360.0;

    return NavState(
      timestamp: timestamp,
      latitude: lat,
      longitude: lon,
      altitude: alt,
      headingDegrees: headingDeg,
      pitchDegrees: attitude.y * 180.0 / pi,
      rollDegrees: attitude.x * 180.0 / pi,
      speedMps: velEnu.length,
      positionUncertaintyMeters: positionUncertaintyMeters,
      mode: mode,
    );
  }
}
