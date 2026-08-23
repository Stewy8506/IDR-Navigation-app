import 'dart:math';
import 'package:vector_math/vector_math_64.dart';
import '../core/constants.dart';

/// State of strapdown inertial navigation integration
class InsState {
  Vector3 positionEnu; // [East, North, Up] meters
  Vector3 velocityEnu; // [vE, vN, vU] m/s
  double roll;         // radians
  double pitch;        // radians
  double yaw;          // radians

  InsState({
    required this.positionEnu,
    required this.velocityEnu,
    this.roll = 0.0,
    this.pitch = 0.0,
    this.yaw = 0.0,
  });

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
    required Vector3 accelVehicle, // m/s^2 (in vehicle body frame)
    required Vector3 gyroVehicle,  // rad/s (in vehicle body frame)
  }) {
    if (_lastTimestamp == null) {
      _lastTimestamp = timestamp;
      return;
    }

    final double dt = (timestamp.difference(_lastTimestamp!).inMicroseconds) / 1e6;
    _lastTimestamp = timestamp;

    if (dt <= 0 || dt > 0.5) {
      // Guard against abnormal dt jumps
      return;
    }

    // 1. Attitude Update (Euler integration)
    _state.yaw += gyroVehicle.z * dt;
    _state.pitch += gyroVehicle.y * dt;
    _state.roll += gyroVehicle.x * dt;

    // Normalize yaw to [-pi, pi]
    if (_state.yaw > pi) _state.yaw -= 2 * pi;
    if (_state.yaw < -pi) _state.yaw += 2 * pi;

    // 2. Rotation matrix from Vehicle Body to Navigation Frame (ENU)
    final Matrix3 rBodyToNav = Matrix3.rotationZ(_state.yaw) *
        Matrix3.rotationY(_state.pitch) *
        Matrix3.rotationX(_state.roll);

    // 3. Accelerometer transformation to Navigation Frame & Gravity removal
    final Vector3 accelNav = rBodyToNav.transformed(accelVehicle);
    accelNav.z -= NavConstants.gravity; // Remove gravity in ENU Up-axis

    // 4. Velocity integration
    _state.velocityEnu += accelNav * dt;

    // 5. Position integration
    _state.positionEnu += _state.velocityEnu * dt;
  }
}
