import 'nav_mode.dart';

/// Comprehensive estimated navigation state emitted continuously by the engine.
class NavState {
  final DateTime timestamp;
  final double latitude;
  final double longitude;
  final double altitude;
  final double headingDegrees;
  final double pitchDegrees;
  final double rollDegrees;
  final double speedMps;
  final double positionUncertaintyMeters;
  final NavMode mode;

  const NavState({
    required this.timestamp,
    required this.latitude,
    required this.longitude,
    this.altitude = 0.0,
    required this.headingDegrees,
    this.pitchDegrees = 0.0,
    this.rollDegrees = 0.0,
    required this.speedMps,
    required this.positionUncertaintyMeters,
    required this.mode,
  });

  @override
  String toString() {
    return 'NavState(lat: ${latitude.toStringAsFixed(6)}, lon: ${longitude.toStringAsFixed(6)}, '
        'heading: ${headingDegrees.toStringAsFixed(1)}°, speed: ${(speedMps * 3.6).toStringAsFixed(1)} km/h, '
        'uncert: ${positionUncertaintyMeters.toStringAsFixed(2)}m, mode: ${mode.name})';
  }
}
