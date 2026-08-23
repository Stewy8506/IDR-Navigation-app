// ignore_for_file: avoid_print, prefer_interpolation_to_compose_strings

import 'dart:convert';
import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:idr_nav/ai/speed_filter_runner.dart';
import 'package:idr_nav/calibration/alignment_estimator.dart';
import 'package:idr_nav/fusion/ekf_fusion.dart';
import 'package:idr_nav/models/nav_mode.dart';
import 'package:vector_math/vector_math_64.dart';

void main() {
  test('TASK 2: Per-Cycle Pipeline Latency Benchmark (Dart Native VM)', () async {
    final sampleFile = File('assets/sample_logs/sample_drive.csv');
    expect(sampleFile.existsSync(), isTrue, reason: 'sample_drive.csv must exist');

    final lines = await sampleFile.readAsLines(encoding: latin1);
    final alignmentEstimator = AlignmentEstimator();
    final ekfEngine = EkfFusionEngine();
    final speedFilter = SpeedFilterRunner(windowSize: 20);

    // Warmup
    final warmupAccel = Vector3(0.1, 0.2, 9.81);
    final warmupGyro = Vector3(0.01, 0.01, 0.01);
    for (int i = 0; i < 50; i++) {
      speedFilter.addSample(warmupAccel, warmupGyro);
      ekfEngine.predict(
        timestamp: DateTime.now(),
        accelVehicle: warmupAccel,
        gyroVehicle: warmupGyro,
      );
      ekfEngine.applyNonHolonomicConstraints();
      ekfEngine.updateAiSpeed(forwardSpeedMps: 10.0, speedVariance: 0.5);
    }

    // Benchmark Measurements
    final List<double> featureExtractionTimesUs = [];
    final List<double> ekfPredictionTimesUs = [];
    final List<double> ekfUpdateTimesUs = [];
    final List<double> totalCycleTimesUs = [];

    int evaluatedSamples = 0;
    final stopwatch = Stopwatch();

    for (final line in lines) {
      if (line.isEmpty || line.startsWith('#') || line.startsWith('GPS')) continue;
      final parts = line.split(',');
      if (parts.length < 18) continue;

      final ax = double.tryParse(parts[9]) ?? 0.0;
      final ay = double.tryParse(parts[10]) ?? 0.0;
      final az = double.tryParse(parts[11]) ?? 9.81;
      final gy = double.tryParse(parts[15]) ?? 0.0;
      final gp = double.tryParse(parts[16]) ?? 0.0;
      final gr = double.tryParse(parts[17]) ?? 0.0;

      final rawAccel = Vector3(ax, ay, az);
      final rawGyro = Vector3(gr, gp, gy);

      // --- MEASURE 1: Feature Extraction & Alignment ---
      stopwatch.reset();
      stopwatch.start();
      final accelVehicle = alignmentEstimator.transformToVehicleFrame(rawAccel);
      final gyroVehicle = alignmentEstimator.transformToVehicleFrame(rawGyro);
      speedFilter.addSample(accelVehicle, gyroVehicle);
      final speedEst = speedFilter.predictSpeed();
      stopwatch.stop();
      final tFeatUs = stopwatch.elapsedMicroseconds.toDouble();
      featureExtractionTimesUs.add(tFeatUs);

      // --- MEASURE 2: EKF Prediction + NHC ---
      stopwatch.reset();
      stopwatch.start();
      final now = DateTime.now();
      ekfEngine.predict(
        timestamp: now,
        accelVehicle: accelVehicle,
        gyroVehicle: gyroVehicle,
      );
      ekfEngine.applyNonHolonomicConstraints();
      stopwatch.stop();
      final tPredUs = stopwatch.elapsedMicroseconds.toDouble();
      ekfPredictionTimesUs.add(tPredUs);

      // --- MEASURE 3: EKF AI Speed Measurement Update ---
      stopwatch.reset();
      stopwatch.start();
      if (speedEst != null) {
        ekfEngine.updateAiSpeed(
          forwardSpeedMps: speedEst.speedMps,
          speedVariance: speedEst.variance,
        );
      }
      final _ = ekfEngine.getNavState(now, NavMode.deadReckoning);
      stopwatch.stop();
      final tUpdateUs = stopwatch.elapsedMicroseconds.toDouble();
      ekfUpdateTimesUs.add(tUpdateUs);

      // Total per-cycle time
      totalCycleTimesUs.add(tFeatUs + tPredUs + tUpdateUs);
      evaluatedSamples++;
      if (evaluatedSamples >= 500) break;
    }

    // Compute percentiles
    totalCycleTimesUs.sort();
    final meanTotalUs = totalCycleTimesUs.reduce((a, b) => a + b) / totalCycleTimesUs.length;
    final p50TotalUs = totalCycleTimesUs[(totalCycleTimesUs.length * 0.50).toInt()];
    final p95TotalUs = totalCycleTimesUs[(totalCycleTimesUs.length * 0.95).toInt()];
    final p99TotalUs = totalCycleTimesUs[(totalCycleTimesUs.length * 0.99).toInt()];

    final meanFeatUs = featureExtractionTimesUs.reduce((a, b) => a + b) / featureExtractionTimesUs.length;
    final meanPredUs = ekfPredictionTimesUs.reduce((a, b) => a + b) / ekfPredictionTimesUs.length;
    final meanUpdateUs = ekfUpdateTimesUs.reduce((a, b) => a + b) / ekfUpdateTimesUs.length;

    print('\n' + '=' * 65);
    print('   TASK 2: PER-CYCLE PIPELINE LATENCY BENCHMARK (DART RUNTIME)');
    print('=' * 65);
    print('Evaluated Samples:     $evaluatedSamples consecutive real drive cycles');
    print('-' * 65);
    print('Breakdown by Subsystem (Mean):');
    print('  1. 10-Ch Feature Extraction & Windowing: ${(meanFeatUs / 1000.0).toStringAsFixed(4)} ms (${meanFeatUs.toStringAsFixed(1)} µs)');
    print('  2. Strapdown INS Prediction & NHC:       ${(meanPredUs / 1000.0).toStringAsFixed(4)} ms (${meanPredUs.toStringAsFixed(1)} µs)');
    print('  3. EKF Measurement Update & NavState:    ${(meanUpdateUs / 1000.0).toStringAsFixed(4)} ms (${meanUpdateUs.toStringAsFixed(1)} µs)');
    print('-' * 65);
    print('Total Per-Cycle Latency (Dart Pipeline):');
    print('  Mean: ${(meanTotalUs / 1000.0).toStringAsFixed(4)} ms (${meanTotalUs.toStringAsFixed(1)} µs)');
    print('  P50:  ${(p50TotalUs / 1000.0).toStringAsFixed(4)} ms (${p50TotalUs.toStringAsFixed(1)} µs)');
    print('  P95:  ${(p95TotalUs / 1000.0).toStringAsFixed(4)} ms (${p95TotalUs.toStringAsFixed(1)} µs)');
    print('  P99:  ${(p99TotalUs / 1000.0).toStringAsFixed(4)} ms (${p99TotalUs.toStringAsFixed(1)} µs)');
    print('-' * 65);
    final budgetPercent = (meanTotalUs / 1000.0 / 100.0) * 100.0;
    print('Target 10 Hz Budget:       100.00 ms');
    print('Actual Budget Used:        ${budgetPercent.toStringAsFixed(3)}%');
    print('Margin Remaining:          ${(100.0 - meanTotalUs / 1000.0).toStringAsFixed(3)} ms');
    print('=' * 65 + '\n');
  });
}
