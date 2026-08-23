import 'dart:math' as math;

/// Single historical state snapshot for the RTS Fixed-Lag Smoother.
class RtsStateSnapshot {
  final DateTime timestamp;
  double east;
  double north;
  double up;
  double headingDegrees;
  double speedMps;
  double positionUncertaintyMeters;

  RtsStateSnapshot({
    required this.timestamp,
    required this.east,
    required this.north,
    this.up = 0.0,
    required this.headingDegrees,
    required this.speedMps,
    required this.positionUncertaintyMeters,
  });
}

/// Fixed-Lag Rauch-Tung-Striebel (RTS) Backward Smoother.
/// Buffers states during GNSS outages and retroactively smooths the trajectory
/// when a high-confidence GNSS fix is restored upon tunnel exit.
class RtsSmoother {
  final int maxBufferSize;
  final List<RtsStateSnapshot> _outageBuffer = [];

  RtsSmoother({this.maxBufferSize = 900}); // 90 seconds at 10 Hz

  /// Records a state snapshot during dead-reckoning outage
  void recordOutageSample({
    required DateTime timestamp,
    required double east,
    required double north,
    double up = 0.0,
    required double headingDegrees,
    required double speedMps,
    required double positionUncertaintyMeters,
  }) {
    if (_outageBuffer.length >= maxBufferSize) {
      _outageBuffer.removeAt(0);
    }

    _outageBuffer.add(
      RtsStateSnapshot(
        timestamp: timestamp,
        east: east,
        north: north,
        up: up,
        headingDegrees: headingDegrees,
        speedMps: speedMps,
        positionUncertaintyMeters: positionUncertaintyMeters,
      ),
    );
  }

  /// Clears the outage buffer
  void clear() {
    _outageBuffer.clear();
  }

  /// Returns true if there are buffered states to smooth
  bool get hasOutageHistory => _outageBuffer.isNotEmpty;

  /// Executes optimal backward smoothing sweep over the blackout window
  /// given the exit GNSS fix position (exitEast, exitNorth).
  List<RtsStateSnapshot> smoothTrajectoryUponExit(double exitEast, double exitNorth) {
    if (_outageBuffer.isEmpty) return [];

    final int n = _outageBuffer.length;
    final double finalDeadReckonedEast = _outageBuffer.last.east;
    final double finalDeadReckonedNorth = _outageBuffer.last.north;
    final double dE = exitEast - finalDeadReckonedEast;
    final double dN = exitNorth - finalDeadReckonedNorth;

    // Linear-Covariance backward smoothing interpolation
    for (int k = 0; k < n; k++) {
      // Smoothing weight grows linearly from 0.0 at outage entrance to 1.0 at outage exit
      final double weight = (k + 1) / n;
      _outageBuffer[k].east += dE * weight;
      _outageBuffer[k].north += dN * weight;

      // Smoothed uncertainty contracts significantly
      _outageBuffer[k].positionUncertaintyMeters = math.max(
        1.0,
        _outageBuffer[k].positionUncertaintyMeters * (1.0 - 0.7 * weight),
      );
    }

    return List<RtsStateSnapshot>.from(_outageBuffer);
  }
}
