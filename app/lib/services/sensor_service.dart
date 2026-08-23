import '../models/sensor_sample.dart';

/// Abstract SensorService defining stream contracts for IMU and GNSS.
/// Can be backed by live mobile hardware or offline log replay.
abstract class SensorService {
  /// Stream of raw uncalibrated accelerometer samples (m/s^2)
  Stream<AccelSample> get accelStream;

  /// Stream of raw gyroscope angular velocities (rad/s)
  Stream<GyroSample> get gyroStream;

  /// Stream of raw magnetometer readings (uT)
  Stream<MagSample> get magStream;

  /// Stream of GNSS position/velocity fixes
  Stream<GnssSample> get gnssStream;

  /// Start streaming sensor data
  Future<void> start();

  /// Stop streaming sensor data
  Future<void> stop();

  /// Dispose resources
  Future<void> dispose();
}
