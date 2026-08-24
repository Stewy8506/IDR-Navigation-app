import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'package:flutter_test/flutter_test.dart';
import 'package:vector_math/vector_math_64.dart';
import 'package:idr_nav/fusion/ekf_fusion.dart';
import 'package:idr_nav/models/sensor_sample.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('TASK 3: Real 15-State EkfFusionEngine Evaluation on Driver A (S3a)', () async {
    const String sPath = '../ml/external/IO-VNBD_repo/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S3a/S-S3a.csv';
    const String vPath = '../ml/external/IO-VNBD_repo/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S3a/V-S3a.csv';

    final sFile = File(sPath);
    final vFile = File(vPath);

    expect(sFile.existsSync(), isTrue, reason: 'S-S3a.csv must exist');
    expect(vFile.existsSync(), isTrue, reason: 'V-S3a.csv must exist');

    final sLines = await sFile.readAsLines(encoding: latin1);
    final vLines = await vFile.readAsLines(encoding: latin1);

    final sHeader = sLines[0].split(',').map((e) => e.trim()).toList();
    final vHeader = vLines[0].split(',').map((e) => e.trim()).toList();

    final int idxLat = sHeader.indexOf('GPS LATITUDE (degrees)');
    final int idxLon = sHeader.indexOf('GPS LONGITUDE (degrees)');
    final int idxAlt = sHeader.indexOf('GPS ALTITUDE (m)');
    final int idxAx = sHeader.indexOf('ACCELEROMETER X (m/s²)');
    final int idxAy = sHeader.indexOf('ACCELEROMETER Y (m/s²)');
    final int idxAz = sHeader.indexOf('ACCELEROMETER Z (m/s²)');
    final int idxGy = sHeader.indexOf('GYROSCOPE Yaw (rad/s)');
    final int idxGp = sHeader.indexOf('GYROSCOPE Pitch (rad/s)');
    final int idxGr = sHeader.indexOf('GYROSCOPE Roll (rad/s)');

    int idxSpeed = vHeader.indexOf('Indicated Vehicle Speed (km/hr)');
    if (idxSpeed == -1) idxSpeed = vHeader.indexOf('Velocity (km/hr)');

    final int minLen = math.min(sLines.length - 1, vLines.length - 1);
    expect(minLen > 1000, isTrue);

    // Parse Ground Truth ENU
    final List<Vector3> gnssEnu = [];
    final List<double> gtSpeedMps = [];
    final List<Vector3> imuRaw = [];

    final double lat0 = double.parse(sLines[1].split(',')[idxLat].trim());
    final double lon0 = double.parse(sLines[1].split(',')[idxLon].trim());
    final double alt0 = double.parse(sLines[1].split(',')[idxAlt].trim());

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
      final double gp = double.parse(sParts[idxGp].trim());
      final double gr = double.parse(sParts[idxGr].trim());

      final double spKmh = double.parse(vParts[idxSpeed].trim());
      gtSpeedMps.add(spKmh / 3.6);

      // Vehicle frame alignment: [ax_v, ay_v, az_v] = [ay, -ax, az], [wz_v, wy_v, wx_v] = [gy, gp, gr]
      imuRaw.add(Vector3(ay, -ax, az)); // lateral, forward, vertical

      // Geodetic to ENU
      final double dLat = (lat - lat0) * (math.pi / 180.0);
      final double dLon = (lon - lon0) * (math.pi / 180.0);
      const double earthR = 6378137.0;
      final double e = earthR * math.cos(lat0 * math.pi / 180.0) * dLon;
      final double n = earthR * dLat;
      gnssEnu.add(Vector3(e, n, alt - alt0));
    }

    // Initial heading from GNSS
    final double dx = gnssEnu[50].x - gnssEnu[0].x;
    final double dy = gnssEnu[50].y - gnssEnu[0].y;
    final double initialTheta = math.atan2(dy, dx);

    print('=================================================================');
    print('   REAL 15-STATE EKF SIMULATION ON DRIVER A S3a (FLUTTER ENGINE)');
    print('=================================================================');
    print('Total Evaluated Cycles: $minLen (24 minutes at 10 Hz)');
    print('Initial ENU Heading:   ${(initialTheta * 180 / math.pi).toStringAsFixed(2)} deg');

    // Load neural predictions
    final predsFile = File('../ml/evaluation_cache/s3a_neural_preds.csv');
    expect(predsFile.existsSync(), isTrue);
    final predLines = await predsFile.readAsLines();
    final List<double> aiSpeed = [];
    final List<double> aiVar = [];
    final List<double> aiZupt = [];
    final List<double> aiBwz = [];

    for (int i = 1; i < predLines.length; i++) {
      final parts = predLines[i].split(',');
      aiSpeed.add(double.parse(parts[0]));
      aiVar.add(double.parse(parts[1]));
      aiZupt.add(double.parse(parts[2]));
      aiBwz.add(double.parse(parts[3]));
    }

    final int evalLen = math.min(minLen, aiSpeed.length);

    // Evaluate Multi-Outage Scenarios (30s, 60s, 90s, Full Drive)
    final List<int> durationsSec = [30, 60, 90, evalLen ~/ 10];
    final List<String> labels = ['30s Outage', '60s Outage', '90s Outage', 'Full Drive'];

    print('=================================================================');
    print('   REAL 15-STATE EKF MULTI-OUTAGE EVALUATION (DRIVER A S3a)');
    print('=================================================================');
    print('Outage Window   | Metric       | Pure INS/NHC    | Real AI + 15-State EKF');
    print('-----------------------------------------------------------------');

    for (int d = 0; d < durationsSec.length; d++) {
      final int dur = durationsSec[d];
      final String lbl = labels[d];

      final int startK = math.min(500, evalLen ~/ 4);
      final int endK = math.min(evalLen, startK + dur * 10);

      // 1. Pure INS + NHC
      final ekfIns = EkfFusionEngine();
      ekfIns.setOrigin(lat0, lon0, alt0);
      ekfIns.resetState(
        initialPosEnu: gnssEnu[startK],
        initialVelEnu: Vector3.zero(),
        initialYawRad: initialTheta,
        initialTimestamp: DateTime(2026, 1, 1, 0, 0, 0),
      );

      final List<Vector3> posIns = [];
      DateTime t = DateTime(2026, 1, 1, 0, 0, 0);

      for (int k = startK; k < endK; k++) {
        t = t.add(const Duration(milliseconds: 100));
        ekfIns.predict(
          timestamp: t,
          accelVehicle: imuRaw[k],
          gyroVehicle: Vector3(0.0, 0.0, double.parse(sLines[k + 1].split(',')[idxGy].trim())),
        );
        ekfIns.applyNonHolonomicConstraints();
        posIns.add(ekfIns.posEnu.clone());
      }

      final double driftIns = (posIns.last - gnssEnu[endK - 1]).length;
      double sumSqIns = 0.0;
      for (int k = 0; k < posIns.length; k++) {
        sumSqIns += (posIns[k] - gnssEnu[startK + k]).length2;
      }
      final double ateIns = math.sqrt(sumSqIns / posIns.length);

      // 2. Real AI + 15-State EKF (Pure Dead Reckoning, Raw Gyro)
      final ekfAi = EkfFusionEngine();
      ekfAi.setOrigin(lat0, lon0, alt0);
      ekfAi.resetState(
        initialPosEnu: gnssEnu[startK],
        initialVelEnu: Vector3(gtSpeedMps[startK] * math.cos(initialTheta), gtSpeedMps[startK] * math.sin(initialTheta), 0.0),
        initialYawRad: initialTheta,
        initialTimestamp: DateTime(2026, 1, 1, 0, 0, 0),
      );

      final List<Vector3> posAi = [];
      t = DateTime(2026, 1, 1, 0, 0, 0);

      for (int k = startK; k < endK; k++) {
        t = t.add(const Duration(milliseconds: 100));
        final double rawGz = double.parse(sLines[k + 1].split(',')[idxGy].trim());

        ekfAi.predict(
          timestamp: t,
          accelVehicle: imuRaw[k],
          gyroVehicle: Vector3(0.0, 0.0, rawGz),
        );
        ekfAi.applyNonHolonomicConstraints();

        // Feed Neural Speed Measurement
        if (aiZupt[k] > 0.85) {
          ekfAi.applyZupt();
        } else {
          ekfAi.updateAiSpeed(
            forwardSpeedMps: aiSpeed[k],
            speedVariance: aiVar[k],
          );
        }

        // Apply Centripetal constraint
        ekfAi.applyCentripetalConstraint(
          lateralAccelMps2: imuRaw[k].x,
          yawRateRadPerSec: rawGz,
        );

        posAi.add(ekfAi.posEnu.clone());
      }

      final double driftAi = (posAi.last - gnssEnu[endK - 1]).length;
      double sumSqAi = 0.0;
      for (int k = 0; k < posAi.length; k++) {
        sumSqAi += (posAi[k] - gnssEnu[startK + k]).length2;
      }
      final double ateAi = math.sqrt(sumSqAi / posAi.length);

      print('${lbl.padRight(15)} | Final Drift  | ${driftIns.toStringAsFixed(2).padLeft(12)} m | ${driftAi.toStringAsFixed(2).padLeft(18)} m');
      print('${"".padRight(15)} | ATE RMSE     | ${ateIns.toStringAsFixed(2).padLeft(12)} m | ${ateAi.toStringAsFixed(2).padLeft(18)} m');
      print('-----------------------------------------------------------------');
    }
    print('=================================================================');
  });
}
