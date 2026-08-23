import 'dart:math' as math;

/// Core physical, geodetic, and algorithmic constants for the IDR-Nav engine.
class NavConstants {
  /// Standard gravity constant (m/s^2)
  static const double gravity = 9.80665;

  /// Target fusion update frequency (Hz)
  static const double targetUpdateRateHz = 10.0;
  static const double targetDt = 1.0 / targetUpdateRateHz;

  /// Minimum vehicle speed (m/s) to reliably estimate yaw alignment from GNSS course (~10.8 km/h)
  static const double minAlignmentSpeedMps = 3.0;

  /// WGS-84 Earth Semi-major axis (meters)
  static const double wgs84A = 6378137.0;
  static const double earthRadiusMeters = 6378137.0;

  /// WGS-84 Earth Semi-minor axis (meters)
  static const double wgs84B = 6356752.314245;

  /// WGS-84 Earth Eccentricity Squared
  static const double wgs84E2 = 0.00669437999014;

  /// Maximum acceptable GNSS horizontal accuracy (meters) to accept for EKF update
  static const double maxGnssAccuracyThresholdMeters = 15.0;

  /// Sliding window length for stationary gravity calibration (samples at 10Hz = 5s)
  static const int staticWindowSamples = 50;

  /// Variance threshold to declare phone stationary on mount
  static const double stationaryVarianceThreshold = 0.04;

  /// ZUPT Acceleration Variance Threshold for engine idle detection (m^2/s^4)
  static const double zuptAccVarianceThreshold = 0.025;

  /// ZUPT Angular Rate Norm Threshold (rad/s)
  static const double zuptAngularRateThreshold = 0.05;

  /// Process Noise Covariances for 15-State ES-EKF
  static const double qPos = 0.01;      // (m^2/s)
  static const double qVel = 0.05;      // (m^2/s^3)
  static const double qAtt = 0.0001;    // (rad^2/s)
  static const double qAccBias = 1e-5;  // (m^2/s^5)
  static const double qGyroBias = 1e-7; // (rad^2/s^3)

  /// Measurement Noise Covariances
  static const double rGnssPos = 2.5 * 2.5; // (m^2)
  static const double rGnssVel = 0.3 * 0.3; // (m^2/s^2)
  static const double rNhc = 0.05 * 0.05;   // (m^2/s^2)
  static const double rZupt = 0.01 * 0.01;  // (m^2/s^2)
  static const double rAiSpeedFloor = 0.50; // (m^2/s^2) minimum variance for AI speed
}

/// Geodetic & Coordinate Math Transformations
class GeoMath {
  /// Converts Geodetic (Lat, Lon, Alt) to Local ENU (East, North, Up) relative to reference origin.
  static List<double> geodeticToEnu({
    required double latDeg,
    required double lonDeg,
    required double altMeters,
    required double refLatDeg,
    required double refLonDeg,
    required double refAltMeters,
  }) {
    final double refLatRad = refLatDeg * math.pi / 180.0;
    final double dLatRad = (latDeg - refLatDeg) * math.pi / 180.0;
    final double dLonRad = (lonDeg - refLonDeg) * math.pi / 180.0;

    final double east = NavConstants.wgs84A * dLonRad * math.cos(refLatRad);
    final double north = NavConstants.wgs84A * dLatRad;
    final double up = altMeters - refAltMeters;

    return [east, north, up];
  }

  /// Converts Local ENU (East, North, Up) back to Geodetic (Lat, Lon, Alt).
  static List<double> enuToGeodetic({
    required double east,
    required double north,
    required double up,
    required double refLatDeg,
    required double refLonDeg,
    required double refAltMeters,
  }) {
    final double refLatRad = refLatDeg * math.pi / 180.0;
    final double dLatRad = north / NavConstants.wgs84A;
    final double dLonRad = east / (NavConstants.wgs84A * math.cos(refLatRad));

    final double latDeg = refLatDeg + (dLatRad * 180.0 / math.pi);
    final double lonDeg = refLonDeg + (dLonRad * 180.0 / math.pi);
    final double altMeters = refAltMeters + up;

    return [latDeg, lonDeg, altMeters];
  }

  /// Converts Mathematical ENU Heading theta (CCW from East in radians) to Compass Heading (CW from North in degrees).
  static double mathEnuToCompassDegrees(double thetaRad) {
    final double mathDegrees = thetaRad * 180.0 / math.pi;
    double compass = 90.0 - mathDegrees;
    while (compass < 0.0) {
      compass += 360.0;
    }
    while (compass >= 360.0) {
      compass -= 360.0;
    }
    return compass;
  }

  /// Converts Compass Heading (CW from North in degrees) to Mathematical ENU Heading theta (CCW from East in radians).
  static double compassDegreesToMathEnu(double compassDegrees) {
    final double mathDegrees = 90.0 - compassDegrees;
    double thetaRad = mathDegrees * math.pi / 180.0;
    // Normalize to [-pi, pi]
    while (thetaRad > math.pi) {
      thetaRad -= 2.0 * math.pi;
    }
    while (thetaRad < -math.pi) {
      thetaRad += 2.0 * math.pi;
    }
    return thetaRad;
  }
}
