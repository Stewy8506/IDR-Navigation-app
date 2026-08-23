import 'dart:math';
import 'package:vector_math/vector_math_64.dart';
import 'constants.dart';

/// Utilities for geodetic (WGS84) to local East-North-Up (ENU) coordinates
/// and 3D frame rotations.
class CoordinateTransform {
  /// Converts Geodetic (lat, lon, alt) to Local ENU coordinates relative to an origin.
  static Vector3 geodeticToEnu({
    required double latDeg,
    required double lonDeg,
    required double altMeters,
    required double originLatDeg,
    required double originLonDeg,
    required double originAltMeters,
  }) {
    final double originLatRad = originLatDeg * pi / 180.0;
    final double dLatRad = (latDeg - originLatDeg) * pi / 180.0;
    final double dLonRad = (lonDeg - originLonDeg) * pi / 180.0;

    final double east = NavConstants.earthRadiusMeters * dLonRad * cos(originLatRad);
    final double north = NavConstants.earthRadiusMeters * dLatRad;
    final double up = altMeters - originAltMeters;

    return Vector3(east, north, up);
  }

  /// Converts Local ENU coordinates back to Geodetic (lat, lon, alt) given origin.
  static (double latDeg, double lonDeg, double altMeters) enuToGeodetic({
    required Vector3 enu,
    required double originLatDeg,
    required double originLonDeg,
    required double originAltMeters,
  }) {
    final double originLatRad = originLatDeg * pi / 180.0;
    final double dLatRad = enu.y / NavConstants.earthRadiusMeters;
    final double dLonRad = enu.x / (NavConstants.earthRadiusMeters * cos(originLatRad));

    final double latDeg = originLatDeg + (dLatRad * 180.0 / pi);
    final double lonDeg = originLonDeg + (dLonRad * 180.0 / pi);
    final double altMeters = originAltMeters + enu.z;

    return (latDeg, lonDeg, altMeters);
  }

  /// Creates a rotation matrix from Euler angles (Roll, Pitch, Yaw) in radians.
  static Matrix3 eulerToRotationMatrix(double roll, double pitch, double yaw) {
    final Matrix3 rZ = Matrix3.rotationZ(yaw);
    final Matrix3 rY = Matrix3.rotationY(pitch);
    final Matrix3 rX = Matrix3.rotationX(roll);

    // Z-Y-X rotation sequence
    return rZ * rY * rX;
  }
}
