import 'package:vector_math/vector_math_64.dart';

/// Raw 3-axis accelerometer sample with hardware/monotonic timestamp
class AccelSample {
  final DateTime timestamp;
  final Vector3 acceleration; // m/s^2

  AccelSample({required this.timestamp, required this.acceleration});
}

/// Raw 3-axis gyroscope sample with hardware/monotonic timestamp
class GyroSample {
  final DateTime timestamp;
  final Vector3 angularVelocity; // rad/s

  GyroSample({required this.timestamp, required this.angularVelocity});
}

/// Raw 3-axis magnetometer sample with hardware/monotonic timestamp
class MagSample {
  final DateTime timestamp;
  final Vector3 magneticField; // microtesla (uT)

  MagSample({required this.timestamp, required this.magneticField});
}

/// GNSS position & velocity fix
class GnssSample {
  final DateTime timestamp;
  final double latitude;
  final double longitude;
  final double altitude;
  final double speedMps;
  final double headingDegrees;
  final double accuracyMeters;
  final bool isValid;

  GnssSample({
    required this.timestamp,
    required this.latitude,
    required this.longitude,
    this.altitude = 0.0,
    required this.speedMps,
    required this.headingDegrees,
    required this.accuracyMeters,
    this.isValid = true,
  });
}
