import 'dart:math' as math;
import 'package:flutter_test/flutter_test.dart';
import 'package:idr_nav/core/constants.dart';
import 'package:idr_nav/ins/strapdown_ins.dart';
import 'package:vector_math/vector_math_64.dart';

void main() {
  group('StrapdownIns & GeoMath Unit Tests', () {
    test('GeoMath converts between Math ENU and Compass Heading correctly', () {
      // 0 rad Math ENU (East) -> 90 deg Compass
      expect(GeoMath.mathEnuToCompassDegrees(0.0), closeTo(90.0, 1e-4));

      // pi/2 rad Math ENU (North) -> 0 deg Compass
      expect(GeoMath.mathEnuToCompassDegrees(math.pi / 2.0), closeTo(0.0, 1e-4));

      // pi rad Math ENU (West) -> 270 deg Compass
      expect(GeoMath.mathEnuToCompassDegrees(math.pi), closeTo(270.0, 1e-4));

      // -pi/2 rad Math ENU (South) -> 180 deg Compass
      expect(GeoMath.mathEnuToCompassDegrees(-math.pi / 2.0), closeTo(180.0, 1e-4));

      // Round trip conversion
      const double testCompass = 135.0; // South-East
      final double thetaMath = GeoMath.compassDegreesToMathEnu(testCompass);
      expect(GeoMath.mathEnuToCompassDegrees(thetaMath), closeTo(testCompass, 1e-4));
    });

    test('StrapdownIns integrates forward acceleration into East/North velocity', () {
      final ins = StrapdownIns();
      final t0 = DateTime(2026, 1, 1, 12, 0, 0);

      // Initialize facing North (yaw = pi/2 rad)
      ins.setState(InsState(
        positionEnu: Vector3.zero(),
        velocityEnu: Vector3.zero(),
        yaw: math.pi / 2.0,
      ));

      // Step 1: Initial timestamp registration
      ins.step(
        timestamp: t0,
        accelVehicle: Vector3(0.0, 2.0, 9.80665), // 2.0 m/s^2 forward
        gyroVehicle: Vector3.zero(),
      );

      // Step 2: 1 second later (dt = 1.0s, forward accel = 2.0 m/s^2 facing North)
      final t1 = t0.add(const Duration(milliseconds: 100)); // dt = 0.1s
      ins.step(
        timestamp: t1,
        accelVehicle: Vector3(0.0, 2.0, 9.80665),
        gyroVehicle: Vector3.zero(),
      );

      // North velocity should increase by 2.0 * 0.1 = 0.2 m/s, East velocity ~ 0
      expect(ins.state.velocityEnu.y, closeTo(0.2, 1e-3));
      expect(ins.state.velocityEnu.x, closeTo(0.0, 1e-3));
      expect(ins.state.forwardSpeedMps, closeTo(0.2, 1e-3));
    });

    test('StrapdownIns turns CCW when +Z gyro yaw rate is positive', () {
      final ins = StrapdownIns();
      final t0 = DateTime(2026, 1, 1, 12, 0, 0);

      ins.setState(InsState(
        positionEnu: Vector3.zero(),
        velocityEnu: Vector3.zero(),
        yaw: 0.0, // Facing East
      ));

      ins.step(
        timestamp: t0,
        accelVehicle: Vector3(0.0, 0.0, 9.80665),
        gyroVehicle: Vector3(0.0, 0.0, 0.5), // +0.5 rad/s CCW
      );

      final t1 = t0.add(const Duration(milliseconds: 100)); // dt = 0.1s
      ins.step(
        timestamp: t1,
        accelVehicle: Vector3(0.0, 0.0, 9.80665),
        gyroVehicle: Vector3(0.0, 0.0, 0.5),
      );

      // Yaw should increase by 0.5 * 0.1 = 0.05 rad (turning toward North CCW)
      expect(ins.state.yaw, closeTo(0.05, 1e-4));
      // Compass heading should decrease from 90 deg toward 0 deg (North)
      expect(ins.state.compassHeadingDegrees, lessThan(90.0));
    });
  });
}
