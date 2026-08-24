import 'package:flutter_test/flutter_test.dart';
import 'package:idr_nav/ai/speed_filter_runner.dart';
import 'package:vector_math/vector_math_64.dart';

void main() {
  group('Adaptive Duty Cycle & Calibration Tests (Method C)', () {
    test('SpeedFilterRunner updates online calibration ratio towards ground truth', () {
      final runner = SpeedFilterRunner(windowSize: 20);

      for (int i = 0; i < 40; i++) {
        final vib = (i % 2 == 0 ? 0.35 : -0.35);
        final accel = Vector3(0.2, 0.5, 9.81 + vib);
        final gyro = Vector3(0.01, 0.01, 0.05 + vib * 0.1);
        runner.addSample(accel, gyro);
      }

      final uncalibratedEstimate = runner.predictSpeed(applyCalibration: false);
      expect(uncalibratedEstimate, isNotNull);
      expect(uncalibratedEstimate!.isZupt, isFalse);
      expect(uncalibratedEstimate.speedMps, greaterThan(0.0));

      final double initialScale = runner.calibrationScaleFactor;
      expect(initialScale, equals(1.0));

      // Simulate a GNSS fix indicating actual speed is 20% higher than uncalibrated raw estimate
      final double simulatedGnssSpeed = uncalibratedEstimate.speedMps * 1.20;

      // Update calibration across several GNSS ticks
      for (int i = 0; i < 15; i++) {
        final vib = (i % 2 == 0 ? 0.35 : -0.35);
        final accel = Vector3(0.2, 0.5, 9.81 + vib);
        final gyro = Vector3(0.01, 0.01, 0.05 + vib * 0.1);
        runner.addSample(accel, gyro);
        runner.updateGnssCalibration(simulatedGnssSpeed, accuracyMeters: 1.5);
      }

      // Calibration scale factor should have adapted upwards towards 1.20
      expect(runner.calibrationScaleFactor, greaterThan(1.05));
      expect(runner.calibrationCount, equals(15));

      // Calibrated speed should now reflect higher scaled speed
      final calibratedEstimate = runner.predictSpeed(applyCalibration: true);
      expect(calibratedEstimate!.speedMps, greaterThan(uncalibratedEstimate.speedMps));
    });

    test('SpeedFilterRunner maintains warm ring buffer for zero cold-start wake-up', () {
      final runner = SpeedFilterRunner(windowSize: 20);
      final accel = Vector3(0.1, 0.3, 9.81);
      final gyro = Vector3(0.01, 0.01, 0.01);

      // Fill buffer during "sleep" mode
      for (int i = 0; i < 25; i++) {
        runner.addSample(accel, gyro);
      }

      // Wake-up instant prediction should immediately return a valid estimate with 0 delay
      final instantWakeupEstimate = runner.predictSpeed(applyCalibration: true);
      expect(instantWakeupEstimate, isNotNull);
      expect(instantWakeupEstimate!.speedMps, isNot(isNaN));
    });

    test('SpeedFilterRunner ignores noisy GNSS fixes during calibration', () {
      final runner = SpeedFilterRunner(windowSize: 20);
      final accel = Vector3(0.2, 0.5, 9.81);
      final gyro = Vector3(0.01, 0.01, 0.02);

      for (int i = 0; i < 25; i++) {
        runner.addSample(accel, gyro);
      }

      // Try updating calibration with a degraded GNSS accuracy of 25 meters (e.g. multipath jump)
      runner.updateGnssCalibration(50.0, accuracyMeters: 25.0);

      // Scale factor should remain unchanged (rejected by quality threshold)
      expect(runner.calibrationScaleFactor, equals(1.0));
      expect(runner.calibrationCount, equals(0));
    });
  });
}
