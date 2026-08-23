import 'dart:math';
import 'package:flutter/foundation.dart';
import 'package:vector_math/vector_math_64.dart';
import '../core/constants.dart';
import '../models/sensor_sample.dart';

/// Estimates the static and dynamic mounting orientation of the smartphone
/// relative to the vehicle chassis (transforms Phone Frame -> Vehicle Frame).
class AlignmentEstimator {
  final List<Vector3> _stationaryAccelBuffer = [];
  Matrix3 _rPhoneToVehicle = Matrix3.identity();
  bool _isPitchRollCalibrated = false;
  bool _isYawCalibrated = false;

  double _estimatedPitch = 0.0;
  double _estimatedRoll = 0.0;
  double _estimatedYawOffset = 0.0;

  Matrix3 get rotationMatrix => _rPhoneToVehicle;
  bool get isCalibrated => _isPitchRollCalibrated;
  bool get isYawCalibrated => _isYawCalibrated;

  /// Processes raw accelerometer sample to find static gravity vector
  void processAccelerometer(AccelSample sample) {
    if (_isPitchRollCalibrated) return;

    _stationaryAccelBuffer.add(sample.acceleration);
    if (_stationaryAccelBuffer.length >= NavConstants.staticWindowSamples) {
      _computePitchRollFromGravity();
    }
  }

  void _computePitchRollFromGravity() {
    Vector3 mean = Vector3.zero();
    for (var v in _stationaryAccelBuffer) {
      mean += v;
    }
    mean.scale(1.0 / _stationaryAccelBuffer.length);

    // Compute variance to confirm device is stationary
    double variance = 0.0;
    for (var v in _stationaryAccelBuffer) {
      variance += (v - mean).length2;
    }
    variance /= _stationaryAccelBuffer.length;

    if (variance < NavConstants.stationaryVarianceThreshold) {
      // Stationary detected: gravity vector is along mean
      final double gNorm = mean.length;
      if (gNorm > 8.0 && gNorm < 11.5) {
        final double ax = mean.x;
        final double ay = mean.y;
        final double az = mean.z;

        // Pitch & Roll relative to down gravity vector [0, 0, -g]
        _estimatedPitch = atan2(-ay, sqrt(ax * ax + az * az));
        _estimatedRoll = atan2(ax, az);

        _updateRotationMatrix();
        _isPitchRollCalibrated = true;
        debugPrint('Alignment: Pitch=${(_estimatedPitch * 180 / pi).toStringAsFixed(1)}°, Roll=${(_estimatedRoll * 180 / pi).toStringAsFixed(1)}°');
      }
    }
    _stationaryAccelBuffer.clear();
  }

  /// Processes GNSS fix to align vehicle heading with integrated gyro heading
  void processGnss(GnssSample gnss, double integratedGyroHeadingRad) {
    if (!_isPitchRollCalibrated || gnss.speedMps < NavConstants.minAlignmentSpeedMps) {
      return;
    }

    final double gnssCogRad = gnss.headingDegrees * pi / 180.0;
    _estimatedYawOffset = gnssCogRad - integratedGyroHeadingRad;
    _updateRotationMatrix();
    _isYawCalibrated = true;
  }

  void _updateRotationMatrix() {
    final Matrix3 rZ = Matrix3.rotationZ(_estimatedYawOffset);
    final Matrix3 rY = Matrix3.rotationY(_estimatedPitch);
    final Matrix3 rX = Matrix3.rotationX(_estimatedRoll);
    _rPhoneToVehicle = rZ * rY * rX;
  }

  /// Transforms a 3D vector from Phone Frame to Vehicle Body Frame
  Vector3 transformToVehicleFrame(Vector3 phoneVector) {
    return _rPhoneToVehicle.transformed(phoneVector);
  }

  /// Reset calibration
  void reset() {
    _stationaryAccelBuffer.clear();
    _rPhoneToVehicle = Matrix3.identity();
    _isPitchRollCalibrated = false;
    _isYawCalibrated = false;
  }
}
