import 'dart:math';
import 'package:vector_math/vector_math_64.dart';
import '../core/constants.dart';

/// Output from the AI Speed & Vibration Estimator
class SpeedEstimate {
  final double speedMps;
  final double variance;

  const SpeedEstimate({required this.speedMps, required this.variance});
}

/// Runner that maintains a sliding window of vehicle IMU data,
/// computes 10 physics-informed feature channels, and invokes the ML model.
class SpeedFilterRunner {
  final int windowSize;
  final List<List<double>> _featureWindow = [];
  double _leakyVelocityIntegral = 0.0;
  final List<double> _recentAzBuffer = [];
  bool _isModelLoaded = false;

  SpeedFilterRunner({this.windowSize = 20});

  bool get isModelLoaded => _isModelLoaded;

  /// Loads the model from assets
  Future<void> loadModel() async {
    // Scaffold: Model asset is at assets/models/speed_filter.onnx
    _isModelLoaded = false;
  }

  /// Adds a new vehicle-frame IMU sample and extracts 10 physics channels
  void addSample(Vector3 accelVehicle, Vector3 gyroVehicle, {double dt = 0.1}) {
    final ax = accelVehicle.x;
    final ay = accelVehicle.y;
    final az = accelVehicle.z;

    final gy = gyroVehicle.z; // yaw rate
    final gp = gyroVehicle.y; // pitch rate
    final gr = gyroVehicle.x; // roll rate

    // Channel 7: Dynamic acceleration norm
    final aNorm = sqrt(ax * ax + ay * ay + az * az) - NavConstants.gravity;

    // Channel 8: Gyro total angular rate
    final wNorm = sqrt(gy * gy + gp * gp + gr * gr);

    // Channel 9: Leaky velocity integral
    _leakyVelocityIntegral = _leakyVelocityIntegral * 0.95 + ay * dt;

    // Channel 10: Vertical vibration variance
    _recentAzBuffer.add(az);
    if (_recentAzBuffer.length > 5) {
      _recentAzBuffer.removeAt(0);
    }
    final meanAz = _recentAzBuffer.reduce((a, b) => a + b) / _recentAzBuffer.length;
    double azVar = 0.0;
    for (var val in _recentAzBuffer) {
      azVar += (val - meanAz) * (val - meanAz);
    }
    azVar /= _recentAzBuffer.length;

    // 10 Physics Features
    final featureVector = [
      ax,
      ay,
      az,
      gy,
      gp,
      gr,
      aNorm,
      wNorm,
      _leakyVelocityIntegral,
      azVar,
    ];

    _featureWindow.add(featureVector);
    if (_featureWindow.length > windowSize) {
      _featureWindow.removeAt(0);
    }
  }

  /// Predicts forward speed (m/s) and measurement variance (R) if window is full
  SpeedEstimate? predictSpeed() {
    if (!_isModelLoaded || _featureWindow.length < windowSize) {
      return null;
    }
    // Returns [mu, var] from dual-head 10-channel model
    return const SpeedEstimate(speedMps: 0.0, variance: 0.5);
  }
}
