import 'dart:async';
import 'package:vector_math/vector_math_64.dart';
import '../ai/speed_filter_runner.dart';
import '../calibration/alignment_estimator.dart';
import '../fusion/ekf_fusion.dart';
import '../map_matching/hmm_map_matcher.dart';
import '../map_matching/osm_graph.dart';
import '../models/nav_mode.dart';
import '../models/nav_state.dart';
import '../models/sensor_sample.dart';
import '../mode_manager/mode_manager.dart';
import '../services/sensor_service.dart';

/// Top-level coordinator for the IDR-Nav offline navigation engine.
class IdrNavEngine {
  final SensorService sensorService;
  final AlignmentEstimator alignmentEstimator = AlignmentEstimator();
  final EkfFusionEngine ekfEngine = EkfFusionEngine();
  final ModeManager modeManager = ModeManager();
  final SpeedFilterRunner speedFilter = SpeedFilterRunner();
  final OsmRoadGraph roadGraph = OsmRoadGraph();
  late final HmmMapMatcher mapMatcher;

  final _navStateController = StreamController<NavState>.broadcast();
  StreamSubscription? _accelSub;
  StreamSubscription? _gyroSub;
  StreamSubscription? _gnssSub;

  Vector3 _latestAccel = Vector3.zero();
  Vector3 _latestGyro = Vector3.zero();
  GnssSample? _latestGnss;
  Timer? _fusionLoopTimer;

  IdrNavEngine({required this.sensorService}) {
    roadGraph.loadBundledTestNetwork();
    mapMatcher = HmmMapMatcher(graph: roadGraph);
  }

  /// Output stream consumed by UI and visualizers
  Stream<NavState> get navStateStream => _navStateController.stream;

  Future<void> start() async {
    await sensorService.start();

    // 1. Accelerometer stream (100 Hz)
    _accelSub = sensorService.accelStream.listen((sample) {
      _latestAccel = sample.acceleration;
      alignmentEstimator.processAccelerometer(sample);
    });

    // 2. Gyroscope stream (100 Hz)
    _gyroSub = sensorService.gyroStream.listen((sample) {
      _latestGyro = sample.angularVelocity;
    });

    // 3. GNSS stream (1 Hz)
    _gnssSub = sensorService.gnssStream.listen((sample) {
      _latestGnss = sample;
      ekfEngine.updateGnss(sample);
      alignmentEstimator.processGnss(sample, ekfEngine.attitude.z);
      if (sample.isValid) {
        speedFilter.updateGnssCalibration(
          sample.speedMps,
          accuracyMeters: sample.accuracyMeters,
        );
      }
    });

    // 4. 10 Hz Fusion Loop
    _fusionLoopTimer = Timer.periodic(const Duration(milliseconds: 100), (_) {
      _runFusionStep();
    });
  }

  void _runFusionStep() {
    final now = DateTime.now();

    // 1. Evaluate Operational Mode first
    final mode = modeManager.evaluateMode(
      currentTimestamp: now,
      latestGnss: _latestGnss,
      isCalibrated: alignmentEstimator.isCalibrated,
    );

    // 2. Transform raw phone IMU readings to vehicle body frame
    final Vector3 accelVehicle = alignmentEstimator.transformToVehicleFrame(_latestAccel);
    final Vector3 gyroVehicle = alignmentEstimator.transformToVehicleFrame(_latestGyro);

    // 3. High-rate Strapdown INS Prediction
    ekfEngine.predict(
      timestamp: now,
      accelVehicle: accelVehicle,
      gyroVehicle: gyroVehicle,
    );

    // 4. Apply Non-Holonomic Constraints (NHC)
    ekfEngine.applyNonHolonomicConstraints();

    // 5. Buffer IMU sample to maintain warm 3.2s sliding window
    speedFilter.addSample(accelVehicle, gyroVehicle);

    // 6. Adaptive AI Duty Cycle Execution (Method C)
    if (mode == NavMode.deadReckoning) {
      // WAKE UP: Full 10 Hz calibrated AI speed estimation during blackout
      final speedEstimate = speedFilter.predictSpeed(applyCalibration: true);
      if (speedEstimate != null) {
        if (speedEstimate.isZupt) {
          ekfEngine.applyZupt();
        } else {
          ekfEngine.updateAiSpeed(
            forwardSpeedMps: speedEstimate.speedMps,
            speedVariance: speedEstimate.variance,
          );
        }
      }
    } else {
      // SLEEP / GNSS-AIDED: AI speed injection is dormant to keep clean GNSS trajectory.
      // Cheap physical ZUPT check is still performed for red light stops.
      final speedEstimate = speedFilter.predictSpeed(applyCalibration: false);
      if (speedEstimate != null && speedEstimate.isZupt) {
        ekfEngine.applyZupt();
      }
    }

    // 7. Offline OSM Map-Matching Constraint (Phase E)
    final match = mapMatcher.match(
      currentEast: ekfEngine.posEnu.x,
      currentNorth: ekfEngine.posEnu.y,
      currentHeadingMathRad: ekfEngine.attitude.z,
    );

    if (match.isSnapped) {
      ekfEngine.updateMapMatchingConstraint(
        snappedEast: match.snappedEast,
        snappedNorth: match.snappedNorth,
        roadHeadingMathRad: match.snappedHeadingMathRad,
        constraintConfidence: match.confidence,
      );
    }

    // 8. Formulate and emit continuous NavState
    final state = ekfEngine.getNavState(now, mode);
    _navStateController.add(state);
  }

  Future<void> stop() async {
    _fusionLoopTimer?.cancel();
    await _accelSub?.cancel();
    await _gyroSub?.cancel();
    await _gnssSub?.cancel();
    await sensorService.stop();
  }

  Future<void> dispose() async {
    await stop();
    await _navStateController.close();
    await sensorService.dispose();
  }
}
