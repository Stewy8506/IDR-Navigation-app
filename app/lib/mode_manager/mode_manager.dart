import '../core/constants.dart';
import '../models/nav_mode.dart';
import '../models/sensor_sample.dart';

/// Manages smooth state transitions between GNSS-aided and pure Dead Reckoning.
class ModeManager {
  NavMode _currentMode = NavMode.calibrating;
  DateTime? _lastValidGnssTime;
  DateTime? _outageStartTime;
  int _consecutiveDroppedFixes = 0;
  int _gnssOutageCount = 0;

  NavMode get currentMode => _currentMode;
  int get gnssOutageCount => _gnssOutageCount;
  int get consecutiveDroppedFixes => _consecutiveDroppedFixes;

  /// Returns total duration in seconds of current active GNSS blackout
  double get currentOutageDurationSeconds {
    if (_currentMode != NavMode.deadReckoning || _outageStartTime == null) {
      return 0.0;
    }
    return DateTime.now().difference(_outageStartTime!).inMicroseconds / 1e6;
  }

  /// Evaluates and transitions mode using real sensor timestamp (supports replay and live)
  NavMode evaluateMode({
    required DateTime currentTimestamp,
    required GnssSample? latestGnss,
    required bool isCalibrated,
  }) {
    if (!isCalibrated) {
      _currentMode = NavMode.calibrating;
      return _currentMode;
    }

    final bool isGnssValid = latestGnss != null &&
        latestGnss.isValid &&
        latestGnss.accuracyMeters <= NavConstants.maxGnssAccuracyThresholdMeters;

    if (isGnssValid) {
      // Check if recovering from a blackout
      if (_currentMode == NavMode.deadReckoning) {
        _gnssOutageCount++;
      }
      _consecutiveDroppedFixes = 0;
      _lastValidGnssTime = currentTimestamp;
      _outageStartTime = null;
      _currentMode = NavMode.gnssAided;
    } else {
      _consecutiveDroppedFixes++;
      // If 1.5 seconds have elapsed without a valid GNSS fix, declare DEAD_RECKONING
      if (_lastValidGnssTime == null ||
          currentTimestamp.difference(_lastValidGnssTime!).inMilliseconds > 1500) {
        if (_currentMode != NavMode.deadReckoning) {
          _outageStartTime = currentTimestamp;
        }
        _currentMode = NavMode.deadReckoning;
      }
    }

    return _currentMode;
  }

  /// Resets state manager
  void reset() {
    _currentMode = NavMode.calibrating;
    _lastValidGnssTime = null;
    _outageStartTime = null;
    _consecutiveDroppedFixes = 0;
    _gnssOutageCount = 0;
  }
}
