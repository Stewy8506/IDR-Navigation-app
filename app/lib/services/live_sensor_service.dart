import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:geolocator/geolocator.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:vector_math/vector_math_64.dart';
import '../models/sensor_sample.dart';
import 'sensor_service.dart';

/// Live sensor service capturing hardware IMU and GNSS streams on device.
class LiveSensorService implements SensorService {
  final _accelController = StreamController<AccelSample>.broadcast();
  final _gyroController = StreamController<GyroSample>.broadcast();
  final _magController = StreamController<MagSample>.broadcast();
  final _gnssController = StreamController<GnssSample>.broadcast();

  StreamSubscription? _accelSub;
  StreamSubscription? _gyroSub;
  StreamSubscription? _magSub;
  StreamSubscription? _gnssSub;

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
    // 1. Raw Accelerometer
    _accelSub = accelerometerEventStream(samplingPeriod: SensorInterval.gameInterval).listen(
      (event) {
        _accelController.add(
          AccelSample(
            timestamp: DateTime.now(),
            acceleration: Vector3(event.x, event.y, event.z),
          ),
        );
      },
      onError: (err) => debugPrint('Accel error: $err'),
    );

    // 2. Raw Gyroscope
    _gyroSub = gyroscopeEventStream(samplingPeriod: SensorInterval.gameInterval).listen(
      (event) {
        _gyroController.add(
          GyroSample(
            timestamp: DateTime.now(),
            angularVelocity: Vector3(event.x, event.y, event.z),
          ),
        );
      },
      onError: (err) => debugPrint('Gyro error: $err'),
    );

    // 3. Raw Magnetometer
    _magSub = magnetometerEventStream(samplingPeriod: SensorInterval.gameInterval).listen(
      (event) {
        _magController.add(
          MagSample(
            timestamp: DateTime.now(),
            magneticField: Vector3(event.x, event.y, event.z),
          ),
        );
      },
      onError: (err) => debugPrint('Mag error: $err'),
    );

    // 4. GNSS Stream
    LocationPermission permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
    }

    if (permission == LocationPermission.always || permission == LocationPermission.whileInUse) {
      const locationSettings = LocationSettings(
        accuracy: LocationAccuracy.bestForNavigation,
        distanceFilter: 0,
      );

      _gnssSub = Geolocator.getPositionStream(locationSettings: locationSettings).listen(
        (Position pos) {
          _gnssController.add(
            GnssSample(
              timestamp: pos.timestamp,
              latitude: pos.latitude,
              longitude: pos.longitude,
              altitude: pos.altitude,
              speedMps: pos.speed,
              headingDegrees: pos.heading,
              accuracyMeters: pos.accuracy,
            ),
          );
        },
        onError: (err) => debugPrint('GNSS error: $err'),
      );
    }
  }

  @override
  Future<void> stop() async {
    await _accelSub?.cancel();
    await _gyroSub?.cancel();
    await _magSub?.cancel();
    await _gnssSub?.cancel();
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
