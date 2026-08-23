import 'package:flutter_test/flutter_test.dart';
import 'package:idr_nav/fusion/rts_smoother.dart';

void main() {
  group('RtsSmoother Unit Tests', () {
    test('RTS backward smoothing retroactively removes blackout drift upon exit', () {
      final smoother = RtsSmoother(maxBufferSize: 100);

      // Simulate 10 seconds of dead-reckoning with 10m drift
      final startTime = DateTime.now();
      for (int i = 0; i < 100; i++) {
        final double t = i * 0.1;
        smoother.recordOutageSample(
          timestamp: startTime.add(Duration(milliseconds: (t * 1000).toInt())),
          east: t * 10.0,
          north: 0.0,
          headingDegrees: 90.0,
          speedMps: 10.0,
          positionUncertaintyMeters: 2.0 + t * 0.5,
        );
      }

      expect(smoother.hasOutageHistory, isTrue);

      // Upon exit at east=100m, GNSS fix reports actual position was east=105m (5m discrepancy)
      final smoothedStates = smoother.smoothTrajectoryUponExit(105.0, 0.0);

      expect(smoothedStates.length, equals(100));
      // First state should have minimal correction (near 0)
      expect(smoothedStates.first.east, closeTo(0.05, 0.01));
      // Last state should match the exit GNSS fix exactly (105.0)
      expect(smoothedStates.last.east, closeTo(105.0, 1e-4));
      // Mid-point state should be linearly smoothed
      expect(smoothedStates[49].east, closeTo(49.0 * 0.1 * 10.0 + 2.5, 0.5));
    });
  });
}
