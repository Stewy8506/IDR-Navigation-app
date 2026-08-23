class NavConstants {
  /// Standard gravity constant (m/s^2)
  static const double gravity = 9.80665;

  /// Target fusion update frequency (Hz)
  static const double targetUpdateRateHz = 10.0;

  /// Minimum vehicle speed (m/s) to reliably estimate yaw alignment from GNSS course
  static const double minAlignmentSpeedMps = 3.0; // ~10.8 km/h

  /// Earth radius in meters (WGS-84 mean radius)
  static const double earthRadiusMeters = 6378137.0;

  /// Maximum acceptable GNSS horizontal accuracy (meters) to accept for EKF update
  static const double maxGnssAccuracyThresholdMeters = 15.0;

  /// Sliding window length for stationary gravity calibration (samples)
  static const int staticWindowSamples = 50;

  /// Variance threshold to declare phone stationary on mount
  static const double stationaryVarianceThreshold = 0.05;
}
