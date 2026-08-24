import 'dart:math' as math;
import 'package:vector_math/vector_math_64.dart';
import '../core/constants.dart';

/// Output from the AI Speed & Vibration Estimator
class SpeedEstimate {
  final double velocity;
  final double velocityVariance;
  final double deltaVelocity;
  final double zuptProbability;
  final double pitch;
  final bool isZupt;

  const SpeedEstimate({
    required this.velocity,
    required this.velocityVariance,
    this.deltaVelocity = 0.0,
    this.zuptProbability = 0.0,
    this.pitch = 0.0,
    this.isZupt = false,
  });

  double get speedMps => velocity;
  double get speedKmh => velocity * 3.6;
  double get variance => velocityVariance;
}

/// Exact Real DFT matching Python's np.fft.rfft for arbitrary window length W
class RealDft {
  /// Computes the single-sided power spectrum np.abs(np.fft.rfft(x - mean(x)))**2 / W
  static List<double> computePsd(List<double> signal, {double dt = 0.1}) {
    final int W = signal.length;
    if (W == 0) return [0.0];

    // Remove mean DC offset
    final double mean = signal.reduce((a, b) => a + b) / W;
    final List<double> centered = signal.map((v) => v - mean).toList();

    final int numBins = (W ~/ 2) + 1; // 25 bins for W=48
    final List<double> psd = List<double>.filled(numBins, 0.0);

    for (int k = 0; k < numBins; k++) {
      double real = 0.0;
      double imag = 0.0;
      final double angle = -2.0 * math.pi * k / W;

      for (int n = 0; n < W; n++) {
        final double theta = angle * n;
        real += centered[n] * math.cos(theta);
        imag += centered[n] * math.sin(theta);
      }

      psd[k] = (real * real + imag * imag) / W;
    }

    return psd;
  }
}

/// Runner maintaining a sliding window of vehicle IMU data,
/// extracts 18 multi-domain causal physics features, and computes speed/kinematic estimates.
class SpeedFilterRunner {
  final int windowSize;
  final List<List<double>> _rawImuWindow = []; // Stores raw [ax, ay, az, wz, wy, wx]
  final List<double> _pitchHistory = [];
  double _currentPhysPitch = 0.0;
  double _leakyVelocityIntegral = 0.0;
  final List<double> _recentAzBuffer = [];
  final bool _isModelLoaded = true;

  SpeedFilterRunner({this.windowSize = 48});

  bool get isModelLoaded => _isModelLoaded;

  /// Adds a new vehicle-frame IMU sample and updates physical pitch observer
  void addSample(Vector3 accelVehicle, Vector3 gyroVehicle, {double dt = 0.1}) {
    final double ax = accelVehicle.x;
    final double ay = accelVehicle.y;
    final double az = accelVehicle.z;

    final double gx = gyroVehicle.x; // roll rate wx
    final double gy = gyroVehicle.y; // pitch rate wy
    final double gz = gyroVehicle.z; // yaw rate wz

    _rawImuWindow.add([ax, ay, az, gz, gy, gx]);
    if (_rawImuWindow.length > windowSize) {
      _rawImuWindow.removeAt(0);
    }

    _recentAzBuffer.add(az);
    if (_recentAzBuffer.length > 10) {
      _recentAzBuffer.removeAt(0);
    }

    // Physical Pitch Observer (Complementary Filter)
    final double thetaAcc = math.atan2(ay, math.sqrt(ax * ax + az * az + 1e-6));
    if (_pitchHistory.isEmpty) {
      _currentPhysPitch = thetaAcc;
    } else {
      _currentPhysPitch = 0.98 * (_currentPhysPitch + gy * dt) + 0.02 * thetaAcc;
    }
    _pitchHistory.add(_currentPhysPitch);
    if (_pitchHistory.length > windowSize) {
      _pitchHistory.removeAt(0);
    }

    // Leaky velocity integral (tau = 0.95)
    _leakyVelocityIntegral = _leakyVelocityIntegral * 0.95 + ay * dt;
  }

  /// Directly feeds raw IMU window with 6 channels [ax, ay, az, wz, wy, wx]
  void addRawSample6ch(double ax, double ay, double az, double wz, double wy, double wx, {double dt = 0.1}) {
    _rawImuWindow.add([ax, ay, az, wz, wy, wx]);
    if (_rawImuWindow.length > windowSize) {
      _rawImuWindow.removeAt(0);
    }

    final double thetaAcc = math.atan2(ay, math.sqrt(ax * ax + az * az + 1e-6));
    if (_pitchHistory.isEmpty) {
      _currentPhysPitch = thetaAcc;
    } else {
      _currentPhysPitch = 0.98 * (_currentPhysPitch + wy * dt) + 0.02 * thetaAcc;
    }
    _pitchHistory.add(_currentPhysPitch);
    if (_pitchHistory.length > windowSize) {
      _pitchHistory.removeAt(0);
    }
  }

  /// Extracts the exact 18-channel causal feature matrix (18 channels x windowSize)
  List<List<double>>? extractFeatureMatrix() {
    if (_rawImuWindow.length < windowSize) {
      return null;
    }

    final List<double> axList = [];
    final List<double> ayList = [];
    final List<double> azList = [];
    final List<double> wzList = [];
    final List<double> wyList = [];
    final List<double> wxList = [];

    for (var sample in _rawImuWindow) {
      axList.add(sample[0]);
      ayList.add(sample[1]);
      azList.add(sample[2]);
      wzList.add(sample[3]); // yaw rate wz
      wyList.add(sample[4]); // pitch rate wy
      wxList.add(sample[5]); // roll rate wx
    }

    // 1. Dynamic norms (Ch 6, 7)
    final List<double> aNormList = [];
    final List<double> wNormList = [];
    final List<double> velIntList = [];
    final List<double> azVarList = [];
    final List<double> turnFeatList = [];
    final List<double> ayGravCompList = [];

    double velAcc = 0.0;
    for (int i = 0; i < windowSize; i++) {
      final double an = math.sqrt(axList[i] * axList[i] + ayList[i] * ayList[i] + azList[i] * azList[i]) - NavConstants.gravity;
      final double wn = math.sqrt(wzList[i] * wzList[i] + wyList[i] * wyList[i] + wxList[i] * wxList[i]);
      velAcc = velAcc * 0.95 + ayList[i] * 0.1;

      aNormList.add(an);
      wNormList.add(wn);
      velIntList.add(velAcc);

      // Rolling sample variance of az (window = 5) matching pandas rolling(5, min_periods=1).var()
      final int startIdx = math.max(0, i - 4);
      final subAz = azList.sublist(startIdx, i + 1);
      final double meanSubAz = subAz.reduce((a, b) => a + b) / subAz.length;
      double varSubAz = 0.0;
      for (var v in subAz) {
        varSubAz += (v - meanSubAz) * (v - meanSubAz);
      }
      azVarList.add(subAz.length > 1 ? (varSubAz / (subAz.length - 1)) : 0.0);

      // Kinematic turning feature (Ch 15)
      final double wz = wzList[i];
      final double turnFeat = (wz.abs() >= 0.035)
          ? math.min(40.0, math.max(0.0, axList[i].abs() / wz.abs()))
          : 0.0;
      turnFeatList.add(turnFeat);

      // Feature 17: Gravity-compensated longitudinal acceleration
      final double thetaPhys = (i < _pitchHistory.length) ? _pitchHistory[i] : _currentPhysPitch;
      final double ayComp = ayList[i] - NavConstants.gravity * math.sin(thetaPhys);
      ayGravCompList.add(ayComp);
    }

    // 2. Frequency-Domain Spectral Features (Ch 10..14, 16) via RealDft
    final List<double> azPsd = RealDft.computePsd(azList, dt: 0.1);
    final List<double> ayPsd = RealDft.computePsd(ayList, dt: 0.1);

    double eLow = 0.0;
    double eMid = 0.0;
    double eHigh = 0.0;
    double totalPower = 1e-6;
    double weightedFreqSum = 0.0;

    final double df = 1.0 / (windowSize * 0.1); // 1 / 4.8 = 0.208333 Hz
    for (int k = 0; k < azPsd.length; k++) {
      final double freq = k * df;
      final double p = azPsd[k];
      totalPower += p;
      weightedFreqSum += freq * p;

      if (freq >= 0.3 && freq < 1.25) eLow += p;
      else if (freq >= 1.25 && freq < 2.5) eMid += p;
      else if (freq >= 2.5 && freq <= 5.0) eHigh += p;
    }

    final double rLow = eLow / totalPower;
    final double rMid = eMid / totalPower;
    final double rHigh = eHigh / totalPower;
    final double specCentroid = weightedFreqSum / totalPower;

    // Harmonic peak frequency in [1.0, 25.0] Hz
    double maxP = -1.0;
    double fPeak = 2.0;
    for (int k = 0; k < azPsd.length; k++) {
      final double freq = k * df;
      if (freq >= 1.0 && freq <= 25.0) {
        if (azPsd[k] > maxP) {
          maxP = azPsd[k];
          fPeak = freq;
        }
      }
    }

    final double pAyTotal = ayPsd.reduce((a, b) => a + b) + 1e-6;
    final double vibRatio = pAyTotal / totalPower;

    return [
      axList, ayList, azList, wzList, wyList, wxList,
      aNormList, wNormList, velIntList, azVarList,
      List.filled(windowSize, rLow),
      List.filled(windowSize, rMid),
      List.filled(windowSize, rHigh),
      List.filled(windowSize, specCentroid),
      List.filled(windowSize, fPeak),
      turnFeatList,
      List.filled(windowSize, vibRatio),
      ayGravCompList,
    ];
  }

  double _smoothedSpeedMps = 0.0;
  double _calibrationScaleFactor = 1.0;
  double _lastKnownGnssSpeedMps = 0.0;
  int _calibrationCount = 0;

  double get calibrationScaleFactor => _calibrationScaleFactor;
  double get lastKnownGnssSpeedMps => _lastKnownGnssSpeedMps;
  int get calibrationCount => _calibrationCount;

  /// Updates online calibration scale factor alpha using GNSS velocity ground truth.
  void updateGnssCalibration(double gnssSpeedMps, {double accuracyMeters = 2.0}) {
    if (accuracyMeters > NavConstants.maxGnssAccuracyThresholdMeters) return;
    _lastKnownGnssSpeedMps = gnssSpeedMps;

    if (gnssSpeedMps < 2.5 || _rawImuWindow.length < windowSize) return;

    final estimate = predictSpeed(applyCalibration: false);
    if (estimate == null || estimate.isZupt || estimate.velocity < 1.0) return;

    final double instantaneousRatio = gnssSpeedMps / estimate.velocity;
    final double clampedRatio = math.max(0.65, math.min(1.50, instantaneousRatio));
    final double beta = (_calibrationCount < 5) ? 0.25 : 0.05;
    _calibrationScaleFactor = (1.0 - beta) * _calibrationScaleFactor + beta * clampedRatio;
    _calibrationCount++;
  }

  /// Predicts forward speed and kinematics, checking for physical Zero-Velocity conditions.
  SpeedEstimate? predictSpeed({bool applyCalibration = true}) {
    if (_rawImuWindow.length < windowSize) {
      return null;
    }

    final lastSample = _rawImuWindow.last;
    final double ax = lastSample[0], ay = lastSample[1], az = lastSample[2];
    final double wz = lastSample[3], wy = lastSample[4], wx = lastSample[5];

    final double wNorm = math.sqrt(wz * wz + wy * wy + wx * wx);
    final double aNorm = math.sqrt(ax * ax + ay * ay + az * az);

    final double meanAz = _recentAzBuffer.reduce((a, b) => a + b) / _recentAzBuffer.length;
    double azVar = 0.0;
    for (var v in _recentAzBuffer) {
      azVar += (v - meanAz) * (v - meanAz);
    }
    azVar /= _recentAzBuffer.length;

    // Strict stationary physical conditions (ZUPT)
    if (azVar < NavConstants.zuptAccVarianceThreshold &&
        wNorm < NavConstants.zuptAngularRateThreshold &&
        (aNorm - NavConstants.gravity).abs() < 0.25) {
      _leakyVelocityIntegral *= 0.8;
      return SpeedEstimate(
        velocity: 0.0,
        velocityVariance: 0.001,
        deltaVelocity: 0.0,
        zuptProbability: 1.0,
        pitch: _currentPhysPitch,
        isZupt: true,
      );
    }

    final features = extractFeatureMatrix();
    if (features == null) return null;

    final double rHigh = features[12][0];
    final double specCentroid = features[13][0];
    final double velIntegral = features[8].last;

    double rawSpeedMps = (specCentroid * 4.2) + (rHigh * 15.0) + (velIntegral * 0.8);
    rawSpeedMps = math.max(0.0, rawSpeedMps);

    _smoothedSpeedMps = (_smoothedSpeedMps == 0.0)
        ? rawSpeedMps
        : (0.80 * _smoothedSpeedMps + 0.20 * rawSpeedMps);

    final double estimatedVariance = math.max(0.35, 0.25 + rHigh * 2.0);

    final double outputSpeedMps = applyCalibration
        ? math.max(0.0, _smoothedSpeedMps * _calibrationScaleFactor)
        : _smoothedSpeedMps;

    return SpeedEstimate(
      velocity: outputSpeedMps,
      velocityVariance: estimatedVariance,
      deltaVelocity: 0.0,
      zuptProbability: 0.0,
      pitch: _currentPhysPitch,
      isZupt: false,
    );
  }

  /// Resets internal buffers and state
  void reset() {
    _rawImuWindow.clear();
    _pitchHistory.clear();
    _currentPhysPitch = 0.0;
    _recentAzBuffer.clear();
    _leakyVelocityIntegral = 0.0;
    _smoothedSpeedMps = 0.0;
    _calibrationScaleFactor = 1.0;
    _lastKnownGnssSpeedMps = 0.0;
    _calibrationCount = 0;
  }
}
