import 'dart:async';
import 'package:vector_math/vector_math_64.dart';

/// Runner that maintains a sliding window of vehicle IMU data
/// and invokes the ML speed model for inference.
class SpeedFilterRunner {
  final int windowSize;
  final List<List<double>> _imuBuffer = [];
  bool _isModelLoaded = false;

  SpeedFilterRunner({this.windowSize = 100});

  bool get isModelLoaded => _isModelLoaded;

  /// Loads the TFLite / ONNX model from assets
  Future<void> loadModel() async {
    // Scaffold: Load interpreter when model asset is placed in assets/models/
    _isModelLoaded = false;
  }

  /// Adds a new vehicle-frame IMU sample to the sliding window
  void addSample(Vector3 accelVehicle, Vector3 gyroVehicle) {
    _imuBuffer.add([
      accelVehicle.x,
      accelVehicle.y,
      accelVehicle.z,
      gyroVehicle.x,
      gyroVehicle.y,
      gyroVehicle.z,
    ]);

    if (_imuBuffer.length > windowSize) {
      _imuBuffer.removeAt(0);
    }
  }

  /// Predicts forward speed (m/s) if window is full and model is loaded
  double? predictSpeed() {
    if (!_isModelLoaded || _imuBuffer.length < windowSize) {
      return null;
    }
    // Perform tensor inference here
    return 0.0;
  }
}
