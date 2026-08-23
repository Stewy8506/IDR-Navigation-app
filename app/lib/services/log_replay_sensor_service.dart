import 'dart:async';
import 'package:vector_math/vector_math_64.dart';
import '../models/sensor_sample.dart';
import 'sensor_service.dart';

/// Replays sensor recordings from pre-recorded datasets (e.g. IO-VNBD / CSV)
/// for deterministic unit testing and offline drift evaluation.
class LogReplaySensorService implements SensorService {
  final List<String> csvLines;
  final double playbackSpeed;

  final _accelController = StreamController<AccelSample>.broadcast();
  final _gyroController = StreamController<GyroSample>.broadcast();
  final _magController = StreamController<MagSample>.broadcast();
  final _gnssController = StreamController<GnssSample>.broadcast();

  bool _isPlaying = false;
  Timer? _playbackTimer;

  LogReplaySensorService({
    required this.csvLines,
    this.playbackSpeed = 1.0,
  });

  @override
  Stream<AccelSample> get accelStream => _accelController.stream;

  @override
  Stream<GyroSample> get gyroStream => _gyroController.stream;

  @override
  Stream<MagSample> get magStream => _magController.stream;

  @override
  Stream<GnssSample> get gnssStream => _gnssController.stream;

  @override
  Future<void> start() async {
    _isPlaying = true;
    int lineIndex = 0;

    // Simulate 100Hz playback ticks
    final intervalMs = (10 / playbackSpeed).round();
    _playbackTimer = Timer.periodic(Duration(milliseconds: intervalMs), (timer) {
      if (!_isPlaying || lineIndex >= csvLines.length) {
        timer.cancel();
        return;
      }

      final line = csvLines[lineIndex].trim();
      lineIndex++;
      if (line.isEmpty || line.startsWith('#') || line.startsWith('timestamp')) {
        return;
      }

      final parts = line.split(',');
      if (parts.length >= 7) {
        final now = DateTime.now();
        // Schema: [timestamp, ax, ay, az, gx, gy, gz, (optional lat, lon, speed)]
        final ax = double.tryParse(parts[1]) ?? 0.0;
        final ay = double.tryParse(parts[2]) ?? 0.0;
        final az = double.tryParse(parts[3]) ?? 0.0;
        final gx = double.tryParse(parts[4]) ?? 0.0;
        final gy = double.tryParse(parts[5]) ?? 0.0;
        final gz = double.tryParse(parts[6]) ?? 0.0;

        _accelController.add(AccelSample(timestamp: now, acceleration: Vector3(ax, ay, az)));
        _gyroController.add(GyroSample(timestamp: now, angularVelocity: Vector3(gx, gy, gz)));
      }
    });
  }

  @override
  Future<void> stop() async {
    _isPlaying = false;
    _playbackTimer?.cancel();
  }

  @override
  Future<void> dispose() async {
    await stop();
    await _accelController.close();
    await _gyroController.close();
    await _magController.close();
    await _gnssController.close();
  }
}
