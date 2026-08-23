import 'dart:math' as math;
import 'package:vector_math/vector_math_64.dart';
import '../core/constants.dart';

/// State of strapdown inertial navigation integration
class InsState {
  Vector3 positionEnu; // [East, North, Up] in meters
  Vector3 velocityEnu; // [vEast, vNorth, vUp] in m/s
  double roll;         // rad (rotation around forward Y)
  double pitch;        // rad (rotation around right X)
  double yaw;          // rad (Math ENU angle theta from East CCW)

  InsState({
    required this.positionEnu,
    required this.velocityEnu,
    this.roll = 0.0,
    this.pitch = 0.0,
    this.yaw = 0.0,
  });

  /// Compass heading in degrees (CW from North [0, 360))
  double get compassHeadingDegrees => GeoMath.mathEnuToCompassDegrees(yaw);

  /// Forward velocity in vehicle body frame (m/s)
  double get forwardSpeedMps {
    final double cTh = math.cos(yaw);
    final double sTh = math.sin(yaw);
    return velocityEnu.x * cTh + velocityEnu.y * sTh;
  }

  /// Lateral velocity in vehicle body frame (m/s)
  double get lateralSpeedMps {
    final double cTh = math.cos(yaw);
    final double sTh = math.sin(yaw);
    return velocityEnu.x * sTh - velocityEnu.y * cTh;
  }

  InsState clone() => InsState(
        positionEnu: positionEnu.clone(),
        velocityEnu: velocityEnu.clone(),
        roll: roll,
        pitch: pitch,
        yaw: yaw,
      );
}

/// Classical Strapdown Inertial Navigation System (INS) Mechanization
class StrapdownIns {
  InsState _state = InsState(
    positionEnu: Vector3.zero(),
    velocityEnu: Vector3.zero(),
  );

  DateTime? _lastTimestamp;

  InsState get state => _state;

  /// Resets or initializes state
  void setState(InsState newState) {
    _state = newState.clone();
  }

  /// Mechanization step: updates orientation, velocity, and position
  /// given vehicle-frame acceleration and angular rates.
  void step({
    required DateTime timestamp,
    required Vector3 accelVehicle, // [ax=Right, ay=Forward, az=Up] m/s^2
    required Vector3 gyroVehicle,  // [gx=PitchRate, gy=RollRate, gz=YawRate CCW] rad/s
    Vector3? accelBias,
    Vector3? gyroBias,
  }) {
    if (_lastTimestamp == null) {
      _lastTimestamp = timestamp;
      return;
    }

    final double dt = (timestamp.difference(_lastTimestamp!).inMicroseconds) / 1e6;
    _lastTimestamp = timestamp;

    if (dt <= 0 || dt > 0.5) {
      return;
    }

    // Apply estimated sensor biases
    final double ax = accelVehicle.x - (accelBias?.x ?? 0.0);
    final double ay = accelVehicle.y - (accelBias?.y ?? 0.0);
    final double az = accelVehicle.z - (accelBias?.z ?? 0.0);

    final double gx = gyroVehicle.x - (gyroBias?.x ?? 0.0);
    final double gy = gyroVehicle.y - (gyroBias?.y ?? 0.0);
    final double gz = gyroVehicle.z - (gyroBias?.z ?? 0.0);

    // 1. Attitude Update (Math ENU: +Z gz rotates CCW from East)
    _state.yaw += gz * dt;
    _state.pitch += gx * dt;
    _state.roll += gy * dt;

    // Normalize yaw to [-pi, pi]
    while (_state.yaw > math.pi) _state.yaw -= 2.0 * math.pi;
    while (_state.yaw < -math.pi) _state.yaw += 2.0 * math.pi;

    final double cTh = math.cos(_state.yaw);
    final double sTh = math.sin(_state.yaw);

    // 2. Acceleration Transformation to Math ENU Frame
    // Forward ay -> [cos(theta), sin(theta)], Lateral ax -> [sin(theta), -cos(theta)]
    final double aEast = ay * cTh + ax * sTh;
    final double aNorth = ay * sTh - ax * cTh;
    final double aUp = az - NavConstants.gravity;

    // 3. Velocity integration
    _state.velocityEnu.x += aEast * dt;
    _state.velocityEnu.y += aNorth * dt;
    _state.velocityEnu.z += aUp * dt;

    // 4. Position integration
    _state.positionEnu.x += _state.velocityEnu.x * dt;
    _state.positionEnu.y += _state.velocityEnu.y * dt;
    _state.positionEnu.z += _state.velocityEnu.z * dt;
  }
}
