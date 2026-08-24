import 'dart:math' as math;
import 'package:vector_math/vector_math_64.dart';
import '../core/constants.dart';

/// Output from the AI Speed & Vibration Estimator
class SpeedEstimate {
  final double speedMps;
  final double variance;
  final bool isZupt;

  const SpeedEstimate({
    required this.speedMps,
    required this.variance,
    this.isZupt = false,
  });

  double get speedKmh => speedMps * 3.6;
}

/// Fast in-place Cooley-Tukey Radix-2 FFT for pure Dart spectral feature extraction
class FastFft {
  static List<double> computePowerSpectrum(List<double> signal) {
    if (signal.isEmpty) return [0.0];

    int n = 1;
    while (n < signal.length) {
      n <<= 1;
    }
    if (n < 4) n = 4;

    final List<double> real = List<double>.filled(n, 0.0);
    final List<double> imag = List<double>.filled(n, 0.0);

    for (int i = 0; i < signal.length; i++) {
      real[i] = signal[i];
    }

    // Remove mean DC offset
    final double mean = real.reduce((a, b) => a + b) / n;
    for (int i = 0; i < n; i++) {
      real[i] -= mean;
    }

    // Bit-reversal permutation
    int j = 0;
    for (int i = 0; i < n - 1; i++) {
      if (i < j) {
        final double tr = real[i];
        real[i] = real[j];
        real[j] = tr;
      }
      int k = n >> 1;
      while (k <= j) {
        j -= k;
        k >>= 1;
      }
      j += k;
    }

    // Cooley-Tukey decimation in time
    for (int len = 2; len <= n; len <<= 1) {
      final double ang = -2.0 * math.pi / len;
      final double wlenCos = math.cos(ang);
      final double wlenSin = math.sin(ang);

      for (int i = 0; i < n; i += len) {
        double wCos = 1.0;
        double wSin = 0.0;

        for (int k = 0; k < (len >> 1); k++) {
          final int uIdx = i + k;
          final int vIdx = i + k + (len >> 1);

          final double uReal = real[uIdx];
          final double uImag = imag[uIdx];

          final double vReal = real[vIdx] * wCos - imag[vIdx] * wSin;
          final double vImag = real[vIdx] * wSin + imag[vIdx] * wCos;

          real[uIdx] = uReal + vReal;
          imag[uIdx] = uImag + vImag;
          real[vIdx] = uReal - vReal;
          imag[vIdx] = uImag - vImag;

          final double nextWCos = wCos * wlenCos - wSin * wlenSin;
          final double nextWSin = wCos * wlenSin + wSin * wlenCos;
          wCos = nextWCos;
          wSin = nextWSin;
        }
      }
    }

    // Compute single-sided Power Spectral Density (PSD) for (n/2 + 1) bins
    final int numBins = (n >> 1) + 1;
    final List<double> psd = List<double>.filled(numBins, 0.0);
    for (int k = 0; k < numBins; k++) {
      psd[k] = (real[k] * real[k] + imag[k] * imag[k]) / n;
    }
    return psd;
  }
}

/// Runner maintaining a sliding window of vehicle IMU data,
/// extracts 16 multi-domain spectral features, and performs zero-velocity / speed inference.
class SpeedFilterRunner {
  final int windowSize;
  final List<List<double>> _rawImuWindow = []; // Stores raw [ax, ay, az, gx, gy, gz]
  double _leakyVelocityIntegral = 0.0;
  final List<double> _recentAzBuffer = [];
  final bool _isModelLoaded = true;

  SpeedFilterRunner({this.windowSize = 32});

  bool get isModelLoaded => _isModelLoaded;

  /// Adds a new vehicle-frame IMU sample
  void addSample(Vector3 accelVehicle, Vector3 gyroVehicle, {double dt = 0.1}) {
    final double ax = accelVehicle.x;
    final double ay = accelVehicle.y;
    final double az = accelVehicle.z;

    final double gx = gyroVehicle.x;
    final double gy = gyroVehicle.y;
    final double gz = gyroVehicle.z;

    _rawImuWindow.add([ax, ay, az, gx, gy, gz]);
    if (_rawImuWindow.length > windowSize) {
      _rawImuWindow.removeAt(0);
    }

    _recentAzBuffer.add(az);
    if (_recentAzBuffer.length > 10) {
      _recentAzBuffer.removeAt(0);
    }

    _leakyVelocityIntegral = _leakyVelocityIntegral * 0.95 + ay * dt;
  }

  /// Extracts the 16-channel feature matrix (16 channels x windowSize)
  List<List<double>>? extractFeatureMatrix() {
    if (_rawImuWindow.length < windowSize) {
      return null;
    }

    final List<double> axList = [];
    final List<double> ayList = [];
    final List<double> azList = [];
    final List<double> gyList = [];
    final List<double> gpList = [];
    final List<double> grList = [];

    for (var sample in _rawImuWindow) {
      axList.add(sample[0]);
      ayList.add(sample[1]);
      azList.add(sample[2]);
      gyList.add(sample[5]); // yaw rate gz
      gpList.add(sample[4]); // pitch rate gy
      grList.add(sample[3]); // roll rate gx
    }

    // 1. Time-Domain Physics Features (10 channels)
    final List<double> aNormList = [];
    final List<double> wNormList = [];
    final List<double> velIntList = [];
    final List<double> azVarList = [];

    double velAcc = 0.0;
    for (int i = 0; i < windowSize; i++) {
      final double an = math.sqrt(axList[i] * axList[i] + ayList[i] * ayList[i] + azList[i] * azList[i]) - NavConstants.gravity;
      final double wn = math.sqrt(gyList[i] * gyList[i] + gpList[i] * gpList[i] + grList[i] * grList[i]);
      velAcc = velAcc * 0.95 + ayList[i] * 0.1;

      aNormList.add(an);
      wNormList.add(wn);
      velIntList.add(velAcc);

      // Rolling variance of az
      final int startIdx = math.max(0, i - 4);
      final subAz = azList.sublist(startIdx, i + 1);
      final double meanSubAz = subAz.reduce((a, b) => a + b) / subAz.length;
      double varSubAz = 0.0;
      for (var v in subAz) {
        varSubAz += (v - meanSubAz) * (v - meanSubAz);
      }
      azVarList.add(varSubAz / subAz.length);
    }

    // 2. Frequency-Domain Spectral Features (6 channels via FastFft)
    final List<double> azPsd = FastFft.computePowerSpectrum(azList);
    final List<double> ayPsd = FastFft.computePowerSpectrum(ayList);

    // Sub-band frequencies for 10Hz sampling (17 bins from 0 to 5.0 Hz with df = 0.3125 Hz)
    // Low: bins 1..3 (0.31 - 0.94 Hz), Mid: bins 4..7 (1.25 - 2.19 Hz), High: bins 8..16 (2.50 - 5.00 Hz)
    double eLow = 0.0;
    for (int k = 1; k <= 3; k++) {
      eLow += azPsd[k];
    }

    double eMid = 0.0;
    for (int k = 4; k <= 7; k++) {
      eMid += azPsd[k];
    }

    double eHigh = 0.0;
    for (int k = 8; k < azPsd.length; k++) {
      eHigh += azPsd[k];
    }

    double sumPower = 1e-6;
    double weightedFreqSum = 0.0;
    for (int k = 0; k < azPsd.length; k++) {
      final double freq = k * (10.0 / windowSize);
      sumPower += azPsd[k];
      weightedFreqSum += freq * azPsd[k];
    }
    final double specCentroid = weightedFreqSum / sumPower;
    final double specEnergyAy = ayPsd.reduce((a, b) => a + b);

    final double logELow = math.log(1.0 + eLow);
    final double logEMid = math.log(1.0 + eMid);
    final double logEHigh = math.log(1.0 + eHigh);
    final double logSumPower = math.log(1.0 + sumPower);
    final double logSpecAy = math.log(1.0 + specEnergyAy);

    return [
      axList, ayList, azList, gyList, gpList, grList,
      aNormList, wNormList, velIntList, azVarList,
      List.filled(windowSize, logELow),
      List.filled(windowSize, logEMid),
      List.filled(windowSize, logEHigh),
      List.filled(windowSize, specCentroid),
      List.filled(windowSize, logSumPower),
      List.filled(windowSize, logSpecAy),
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
  /// Runs at ~1 Hz when a valid, high-accuracy GNSS fix arrives.
  void updateGnssCalibration(double gnssSpeedMps, {double accuracyMeters = 2.0}) {
    if (accuracyMeters > NavConstants.maxGnssAccuracyThresholdMeters) return;
    _lastKnownGnssSpeedMps = gnssSpeedMps;

    // Only calibrate when moving at a steady speed to avoid stationary noise division
    if (gnssSpeedMps < 2.5 || _rawImuWindow.length < windowSize) return;

    final estimate = predictSpeed(applyCalibration: false);
    if (estimate == null || estimate.isZupt || estimate.speedMps < 1.0) return;

    final double instantaneousRatio = gnssSpeedMps / estimate.speedMps;

    // Bound ratio to safe physical suspension limits [0.65, 1.50]
    final double clampedRatio = math.max(0.65, math.min(1.50, instantaneousRatio));

    // Fast convergence for initial samples, gentle EMA for steady-state
    final double beta = (_calibrationCount < 5) ? 0.25 : 0.05;
    _calibrationScaleFactor = (1.0 - beta) * _calibrationScaleFactor + beta * clampedRatio;
    _calibrationCount++;
  }

  /// Predicts forward speed and uncertainty, checking for physical Zero-Velocity conditions.
  /// Automatically applies learned online calibration scale factor when [applyCalibration] is true.
  SpeedEstimate? predictSpeed({bool applyCalibration = true}) {
    if (_rawImuWindow.length < windowSize) {
      return null;
    }

    // 1. Physical Zero-Velocity Update (ZUPT) Detection
    final lastSample = _rawImuWindow.last;
    final double ax = lastSample[0], ay = lastSample[1], az = lastSample[2];
    final double gx = lastSample[3], gy = lastSample[4], gz = lastSample[5];

    final double wNorm = math.sqrt(gx * gx + gy * gy + gz * gz);
    final double aNorm = math.sqrt(ax * ax + ay * ay + az * az);

    // Compute recent vertical variance
    final double meanAz = _recentAzBuffer.reduce((a, b) => a + b) / _recentAzBuffer.length;
    double azVar = 0.0;
    for (var v in _recentAzBuffer) {
      azVar += (v - meanAz) * (v - meanAz);
    }
    azVar /= _recentAzBuffer.length;

    // Strict stationary physical conditions
    if (azVar < NavConstants.zuptAccVarianceThreshold &&
        wNorm < NavConstants.zuptAngularRateThreshold &&
        (aNorm - NavConstants.gravity).abs() < 0.25) {
      _leakyVelocityIntegral *= 0.8;
      return const SpeedEstimate(speedMps: 0.0, variance: 0.001, isZupt: true);
    }

    // 2. Multi-Domain Speed & Variance Estimation
    final features = extractFeatureMatrix();
    if (features == null) return null;

    final double logEHigh = features[12][0];
    final double specCentroid = features[13][0];
    final double velIntegral = features[8].last;

    // Calibrated multi-domain linear/spectral estimator with non-negative clamping
    double rawSpeedMps = (specCentroid * 4.2) + (logEHigh * 3.5) + (velIntegral * 0.8);
    rawSpeedMps = math.max(0.0, rawSpeedMps);

    // Causal Exponential Moving Average (EMA) smoothing (alpha = 0.20)
    _smoothedSpeedMps = (_smoothedSpeedMps == 0.0)
        ? rawSpeedMps
        : (0.80 * _smoothedSpeedMps + 0.20 * rawSpeedMps);

    // Dynamic variance estimation based on high-frequency spectral noise
    final double estimatedVariance = math.max(0.35, 0.25 + logEHigh * 0.20);

    // Apply learned vehicle scale factor if enabled
    final double outputSpeedMps = applyCalibration
        ? math.max(0.0, _smoothedSpeedMps * _calibrationScaleFactor)
        : _smoothedSpeedMps;

    return SpeedEstimate(
      speedMps: outputSpeedMps,
      variance: estimatedVariance,
      isZupt: false,
    );
  }

  /// Resets internal buffers and state
  void reset() {
    _rawImuWindow.clear();
    _recentAzBuffer.clear();
    _leakyVelocityIntegral = 0.0;
    _smoothedSpeedMps = 0.0;
    _calibrationScaleFactor = 1.0;
    _lastKnownGnssSpeedMps = 0.0;
    _calibrationCount = 0;
  }
}
