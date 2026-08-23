import '../core/constants.dart';
import '../models/nav_mode.dart';
import '../models/sensor_sample.dart';

/// Manages smooth state transitions between GNSS-aided and pure Dead Reckoning.
class ModeManager {
  NavMode _currentMode = NavMode.calibrating;
  DateTime? _lastValidGnssTime;

  NavMode get currentMode => _currentMode;

  /// Updates mode based on latest GNSS fix and calibration status
  NavMode evaluateMode({
    required GnssSample? latestGnss,
    required bool isCalibrated,
  }) {
    if (!isCalibrated) {
      _currentMode = NavMode.calibrating;
      return _currentMode;
    }

    final now = DateTime.now();
    if (latestGnss != null &&
        latestGnss.isValid &&
        latestGnss.accuracyMeters <= NavConstants.maxGnssAccuracyThresholdMeters) {
      _lastValidGnssTime = now;
      _currentMode = NavMode.gnssAided;
    } else {
      // Check if GNSS outage threshold has elapsed (e.g. > 2.0s without fix)
      if (_lastValidGnssTime == null ||
          now.difference(_lastValidGnssTime!).inMilliseconds > 2000) {
        _currentMode = NavMode.deadReckoning;
      }
    }

    return _currentMode;
  }
}
