import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'package:flutter_test/flutter_test.dart';
import 'package:idr_nav/ai/speed_filter_runner.dart';

void main() {
  test('Deterministic Python <-> Dart 18-Channel Feature Parity Test', () {
    final file = File('test/test_parity_data.json');
    expect(file.existsSync(), isTrue, reason: 'test_parity_data.json must exist');

    final jsonMap = jsonDecode(file.readAsStringSync()) as Map<String, dynamic>;
    final rawImu = (jsonMap['raw_imu'] as List)
        .map((row) => (row as List).map((v) => (v as num).toDouble()).toList())
        .toList();
    final expectedFeat = (jsonMap['expected_feat_18ch'] as List)
        .map((row) => (row as List).map((v) => (v as num).toDouble()).toList())
        .toList();

    expect(rawImu.length, equals(6));
    expect(expectedFeat.length, equals(18));
    final int numSamples = rawImu[0].length;
    expect(numSamples, equals(48));

    final runner = SpeedFilterRunner(windowSize: 48);

    for (int t = 0; t < numSamples; t++) {
      runner.addRawSample6ch(
        rawImu[0][t], // ax
        rawImu[1][t], // ay
        rawImu[2][t], // az
        rawImu[3][t], // wz
        rawImu[4][t], // wy
        rawImu[5][t], // wx
        dt: 0.1,
      );
    }

    final actualFeat = runner.extractFeatureMatrix();
    expect(actualFeat, isNotNull);
    expect(actualFeat!.length, equals(18));

    final channelNames = [
      'ax', 'ay', 'az', 'wz', 'wy', 'wx',
      'a_norm', 'w_norm', 'vel_int', 'az_var',
      'r_low', 'r_mid', 'r_high', 'spec_centroid', 'f_peak',
      'turn_feat', 'vib_ratio', 'ay_grav_comp'
    ];

    print('=================================================================');
    print('       PYTHON <-> DART 18-CHANNEL FEATURE PARITY RESULTS');
    print('=================================================================');

    double maxOverallDiff = 0.0;

    for (int ch = 0; ch < 18; ch++) {
      double maxChDiff = 0.0;
      for (int t = 0; t < numSamples; t++) {
        final double diff = (actualFeat[ch][t] - expectedFeat[ch][t]).abs();
        if (diff > maxChDiff) maxChDiff = diff;
        if (diff > maxOverallDiff) maxOverallDiff = diff;
      }
      print('Ch ${ch.toString().padLeft(2)}: ${channelNames[ch].padRight(16)} | Max Diff: ${maxChDiff.toStringAsExponential(3)}');
      expect(maxChDiff, lessThan(1e-4), reason: 'Channel ${channelNames[ch]} differs from Python reference');
    }

    print('-----------------------------------------------------------------');
    print('Max Overall Discrepancy: ${maxOverallDiff.toStringAsExponential(3)}');
    print('Status: ALL 18 CHANNELS MATCH WITHIN NUMERICAL TOLERANCE (1e-4) [PASS]');
    print('=================================================================');
  });
}
