import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'package:flutter_test/flutter_test.dart';
import 'package:vector_math/vector_math_64.dart';
import 'package:idr_nav/fusion/ekf_fusion.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('TASK 3: Generate Deterministic S3a 30s Outage Trace (Dart Native)', () async {
    const String sPath = '../ml/external/IO-VNBD_repo/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S3a/S-S3a.csv';
    const String vPath = '../ml/external/IO-VNBD_repo/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S3a/V-S3a.csv';
    const String predsPath = '../ml/evaluation_cache/s3a_neural_preds.csv';

    final sFile = File(sPath);
    final vFile = File(vPath);
    final predsFile = File(predsPath);

    final sLines = await sFile.readAsLines(encoding: latin1);
    final vLines = await vFile.readAsLines(encoding: latin1);
    final predLines = await predsFile.readAsLines();

    final sHeader = sLines[0].split(',').map((e) => e.trim()).toList();
    final vHeader = vLines[0].split(',').map((e) => e.trim()).toList();

    final int idxLat = sHeader.indexOf('GPS LATITUDE (degrees)');
    final int idxLon = sHeader.indexOf('GPS LONGITUDE (degrees)');
    final int idxAlt = sHeader.indexOf('GPS ALTITUDE (m)');
    final int idxAx = sHeader.indexOf('ACCELEROMETER X (m/s²)');
    final int idxAy = sHeader.indexOf('ACCELEROMETER Y (m/s²)');
    final int idxAz = sHeader.indexOf('ACCELEROMETER Z (m/s²)');
    final int idxGy = sHeader.indexOf('GYROSCOPE Yaw (rad/s)');

    int idxSpeed = vHeader.indexOf('Indicated Vehicle Speed (km/hr)');
    if (idxSpeed == -1) idxSpeed = vHeader.indexOf('Velocity (km/hr)');

    final int minLen = math.min(sLines.length - 1, math.min(vLines.length - 1, predLines.length - 1));

    final double lat0 = double.parse(sLines[1].split(',')[idxLat].trim());
    final double lon0 = double.parse(sLines[1].split(',')[idxLon].trim());
    final double alt0 = double.parse(sLines[1].split(',')[idxAlt].trim());

    final List<Vector3> gnssEnu = [];
    final List<double> gtSpeedMps = [];
    final List<Vector3> imuRaw = [];
    final List<double> gyRaw = [];

    const double earthR = 6378137.0;

    for (int i = 1; i <= minLen; i++) {
      final sParts = sLines[i].split(',');
      final vParts = vLines[i].split(',');

      final double lat = double.parse(sParts[idxLat].trim());
      final double lon = double.parse(sParts[idxLon].trim());
      final double alt = double.parse(sParts[idxAlt].trim());

      final double ax = double.parse(sParts[idxAx].trim());
      final double ay = double.parse(sParts[idxAy].trim());
      final double az = double.parse(sParts[idxAz].trim());
      final double gy = double.parse(sParts[idxGy].trim());

      final double spKmh = double.parse(vParts[idxSpeed].trim());
      gtSpeedMps.add(spKmh / 3.6);

      imuRaw.add(Vector3(ay, -ax, az)); // lateral, forward, vertical
      gyRaw.add(gy);

      final double dLat = (lat - lat0) * (math.pi / 180.0);
      final double dLon = (lon - lon0) * (math.pi / 180.0);
      final double e = earthR * math.cos(lat0 * math.pi / 180.0) * dLon;
      final double n = earthR * dLat;
      gnssEnu.add(Vector3(e, n, alt - alt0));
    }

    final List<double> aiSpeed = [];
    final List<double> aiVar = [];
    final List<double> aiZupt = [];

    for (int i = 1; i <= minLen; i++) {
      final parts = predLines[i].split(',');
      aiSpeed.add(double.parse(parts[0]));
      aiVar.add(double.parse(parts[1]));
      aiZupt.add(double.parse(parts[2]));
    }

    final double dx = gnssEnu[50].x - gnssEnu[0].x;
    final double dy = gnssEnu[50].y - gnssEnu[0].y;
    final double initialTheta = math.atan2(dy, dx);

    final int startK = math.min(500, minLen ~/ 4);
    final int endK = math.min(minLen, startK + 300);

    final ekf = EkfFusionEngine();
    ekf.setOrigin(lat0, lon0, alt0);
    ekf.resetState(
      initialPosEnu: gnssEnu[startK],
      initialVelEnu: Vector3(gtSpeedMps[startK] * math.cos(initialTheta), gtSpeedMps[startK] * math.sin(initialTheta), 0.0),
      initialYawRad: initialTheta,
      initialTimestamp: DateTime(2026, 1, 1, 0, 0, 0),
    );

    final List<String> traceCsv = [
      'k,pos_e,pos_n,pos_u,vel_e,vel_n,vel_u,yaw,p_vel_var,p_pos_var,pos_err,gt_e,gt_n'
    ];

    DateTime t = DateTime(2026, 1, 1, 0, 0, 0);

    for (int k = startK; k < endK; k++) {
      t = t.add(const Duration(milliseconds: 100));
      final double gz = gyRaw[k];

      ekf.predict(
        timestamp: t,
        accelVehicle: imuRaw[k],
        gyroVehicle: Vector3(0.0, 0.0, gz),
      );
      ekf.applyNonHolonomicConstraints();

      if (aiZupt[k] > 0.85) {
        ekf.applyZupt();
      } else {
        ekf.updateAiSpeed(
          forwardSpeedMps: aiSpeed[k],
          speedVariance: aiVar[k],
        );
      }

      ekf.applyCentripetalConstraint(
        lateralAccelMps2: imuRaw[k].x,
        yawRateRadPerSec: gz,
      );

      final double posErr = math.sqrt(
        math.pow(ekf.posEnu.x - gnssEnu[k].x, 2) + math.pow(ekf.posEnu.y - gnssEnu[k].y, 2)
      );

      traceCsv.add(
        '$k,${ekf.posEnu.x},${ekf.posEnu.y},${ekf.posEnu.z},'
        '${ekf.velEnu.x},${ekf.velEnu.y},${ekf.velEnu.z},'
        '${ekf.attitude.z},${ekf.pVelVariance},${ekf.pPosVariance},'
        '$posErr,${gnssEnu[k].x},${gnssEnu[k].y}'
      );
    }

    final outDartTrace = File('../ml/evaluation_cache/dart_ekf_trace_s3a_30s.csv');
    await outDartTrace.writeAsString(traceCsv.join('\n'));
    print('Saved Dart trace to ${outDartTrace.path}');
  });
}
