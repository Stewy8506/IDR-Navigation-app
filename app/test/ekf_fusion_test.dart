import 'dart:math' as math;
import 'package:flutter_test/flutter_test.dart';
import 'package:idr_nav/fusion/ekf_fusion.dart';
import 'package:idr_nav/models/sensor_sample.dart';
import 'package:vector_math/vector_math_64.dart';

void main() {
  group('EkfFusionEngine Unit Tests', () {
    test('Non-Holonomic Constraints (NHC) reduce lateral slip velocity', () {
      final ekf = EkfFusionEngine();
      ekf.resetState(
        initialPosEnu: Vector3.zero(),
        initialVelEnu: Vector3(5.0, 0.0, 0.0), // 5.0 m/s East
        initialYawRad: math.pi / 2.0,          // Facing North (so East is lateral slide!)
      );

      // East velocity is purely lateral relative to vehicle's North heading
      expect(ekf.velEnu.x, closeTo(5.0, 1e-4));

      // Apply NHC
      for (int i = 0; i < 10; i++) {
        ekf.applyNonHolonomicConstraints();
      }

      // Lateral East velocity should be dampened significantly
      expect(ekf.velEnu.x, lessThan(3.0));
    });

    test('applyZupt zeroes velocity and clamps velocity variance', () {
      final ekf = EkfFusionEngine();
      ekf.resetState(
        initialPosEnu: Vector3(10.0, 20.0, 0.0),
        initialVelEnu: Vector3(2.5, -1.0, 0.0),
      );

      ekf.applyZupt();

      expect(ekf.velEnu.x, equals(0.0));
      expect(ekf.velEnu.y, equals(0.0));
      expect(ekf.velEnu.z, equals(0.0));
      expect(ekf.pVelVariance, lessThanOrEqualTo(0.01));
    });

    test('Chi-Square Outlier Gating rejects extreme GNSS multipath spikes', () {
      final ekf = EkfFusionEngine();
      ekf.setOrigin(52.4862, -1.8904, 100.0);
      ekf.resetState(initialPosEnu: Vector3.zero());

      // Normal GNSS fix within 2m
      final normalGnss = GnssSample(
        timestamp: DateTime.now(),
        latitude: 52.48621,
        longitude: -1.8904,
        altitude: 100.0,
        speedMps: 0.0,
        headingDegrees: 0.0,
        accuracyMeters: 2.0,
      );
      ekf.updateGnss(normalGnss);
      expect(ekf.posEnu.y, greaterThan(0.0));

      // Multipath spike (1,000 meters jump)
      final spikeGnss = GnssSample(
        timestamp: DateTime.now(),
        latitude: 52.5000,
        longitude: -1.8904,
        altitude: 100.0,
        speedMps: 0.0,
        headingDegrees: 0.0,
        accuracyMeters: 2.0,
      );
      final prevPosY = ekf.posEnu.y;
      ekf.updateGnss(spikeGnss);

      // Should be rejected by Chi-Square gate -> position unchanged
      expect(ekf.posEnu.y, equals(prevPosY));
    });
  });
}
