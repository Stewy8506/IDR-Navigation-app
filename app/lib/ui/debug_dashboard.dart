import 'dart:math' as math;
import 'package:flutter/material.dart';
import '../core/idr_nav_engine.dart';
import '../models/nav_mode.dart';
import '../models/nav_state.dart';
import '../services/live_sensor_service.dart';

class DebugDashboard extends StatefulWidget {
  const DebugDashboard({super.key});

  @override
  State<DebugDashboard> createState() => _DebugDashboardState();
}

class _DebugDashboardState extends State<DebugDashboard> {
  late final IdrNavEngine _engine;
  NavState? _latestState;
  final List<Offset> _trajectoryTrail = [];
  double _lastCycleDurationMicros = 0.0;
  DateTime? _lastStepTime;

  @override
  void initState() {
    super.initState();
    _engine = IdrNavEngine(sensorService: LiveSensorService());
    _startEngine();
  }

  Future<void> _startEngine() async {
    await _engine.start();
    _engine.navStateStream.listen((state) {
      final now = DateTime.now();
      if (_lastStepTime != null) {
        _lastCycleDurationMicros = now.difference(_lastStepTime!).inMicroseconds.toDouble();
      }
      _lastStepTime = now;

      if (mounted) {
        setState(() {
          _latestState = state;
          _trajectoryTrail.add(Offset(_engine.ekfEngine.posEnu.x, _engine.ekfEngine.posEnu.y));
          if (_trajectoryTrail.length > 200) {
            _trajectoryTrail.removeAt(0);
          }
        });
      }
    });
  }

  @override
  void dispose() {
    _engine.dispose();
    super.dispose();
  }

  Color _getModeColor(NavMode mode) {
    switch (mode) {
      case NavMode.gnssAided:
        return const Color(0xFF10B981); // Emerald Green
      case NavMode.deadReckoning:
        return const Color(0xFFF59E0B); // Amber / Orange
      case NavMode.calibrating:
        return const Color(0xFF38BDF8); // Sky Blue
    }
  }

  @override
  Widget build(BuildContext context) {
    final mode = _latestState?.mode ?? NavMode.calibrating;
    final modeColor = _getModeColor(mode);

    return Scaffold(
      backgroundColor: const Color(0xFF0B0F19),
      appBar: AppBar(
        backgroundColor: const Color(0xFF111827),
        elevation: 0,
        title: const Text(
          'IDR-Nav Engine Monitor',
          style: TextStyle(fontWeight: FontWeight.bold, fontSize: 18, color: Colors.white),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh_rounded, color: Colors.white70),
            onPressed: () {
              _engine.alignmentEstimator.reset();
              _trajectoryTrail.clear();
            },
            tooltip: 'Reset Alignment & Trail',
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Mode Banner
            Container(
              padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 16),
              decoration: BoxDecoration(
                color: modeColor.withValues(alpha: 0.12),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: modeColor.withValues(alpha: 0.4), width: 1.5),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Row(
                    children: [
                      Icon(
                        mode == NavMode.gnssAided
                            ? Icons.satellite_alt_rounded
                            : mode == NavMode.deadReckoning
                                ? Icons.directions_car_rounded
                                : Icons.sync_rounded,
                        color: modeColor,
                        size: 22,
                      ),
                      const SizedBox(width: 10),
                      Text(
                        'MODE: ${mode.name.toUpperCase()}',
                        style: TextStyle(
                          color: modeColor,
                          fontWeight: FontWeight.bold,
                          fontSize: 14,
                          letterSpacing: 1.2,
                        ),
                      ),
                    ],
                  ),
                  Text(
                    'Uncertainty: ±${_latestState?.positionUncertaintyMeters.toStringAsFixed(1) ?? "--"} m',
                    style: const TextStyle(color: Colors.white70, fontSize: 12, fontWeight: FontWeight.w500),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),

            // 3D Vehicle Attitude & 2D Trajectory Canvas
            Container(
              height: 220,
              decoration: BoxDecoration(
                color: const Color(0xFF161F30),
                borderRadius: BorderRadius.circular(14),
                border: Border.all(color: Colors.white10),
              ),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(14),
                child: CustomPaint(
                  painter: _VehicleVisualizerPainter(
                    rollDeg: _latestState?.rollDegrees ?? 0.0,
                    pitchDeg: _latestState?.pitchDegrees ?? 0.0,
                    yawDeg: _latestState?.headingDegrees ?? 0.0,
                    trail: _trajectoryTrail,
                    modeColor: modeColor,
                  ),
                ),
              ),
            ),
            const SizedBox(height: 16),

            // Primary Telemetry Row
            Row(
              children: [
                Expanded(
                  child: _buildMetricCard(
                    title: 'FORWARD SPEED',
                    value: _latestState != null
                        ? '${(_latestState!.speedMps * 3.6).toStringAsFixed(1)} km/h'
                        : '-- km/h',
                    icon: Icons.speed_rounded,
                    accentColor: const Color(0xFF2DD4BF),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: _buildMetricCard(
                    title: 'HEADING (AZIMUTH)',
                    value: _latestState != null
                        ? '${_latestState!.headingDegrees.toStringAsFixed(1)}°'
                        : '--°',
                    icon: Icons.explore_rounded,
                    accentColor: const Color(0xFFFBBF24),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),

            // Secondary Telemetry Row: Execution & Outages
            Row(
              children: [
                Expanded(
                  child: _buildMetricCard(
                    title: 'LOOP LATENCY',
                    value: _lastCycleDurationMicros > 0
                        ? '${(_lastCycleDurationMicros / 1000.0).toStringAsFixed(2)} ms'
                        : '< 0.03 ms',
                    icon: Icons.timer_outlined,
                    accentColor: const Color(0xFFA78BFA),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: _buildMetricCard(
                    title: 'BLACKOUT COUNT',
                    value: '${_engine.modeManager.gnssOutageCount}',
                    icon: Icons.shield_outlined,
                    accentColor: const Color(0xFFF43F5E),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),

            // Coordinates & Attitude Angles Card
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: const Color(0xFF161F30),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: Colors.white10),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'GEODETIC COORDINATES & ATTITUDE',
                    style: TextStyle(color: Colors.white54, fontSize: 11, fontWeight: FontWeight.bold, letterSpacing: 1.0),
                  ),
                  const SizedBox(height: 10),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        'Lat: ${_latestState?.latitude.toStringAsFixed(6) ?? "0.000000"}',
                        style: const TextStyle(color: Colors.white, fontSize: 14, fontFamily: 'monospace'),
                      ),
                      Text(
                        'Lon: ${_latestState?.longitude.toStringAsFixed(6) ?? "0.000000"}',
                        style: const TextStyle(color: Colors.white, fontSize: 14, fontFamily: 'monospace'),
                      ),
                    ],
                  ),
                  const Divider(color: Colors.white12, height: 20),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceAround,
                    children: [
                      _buildAngleChip('Roll', '${(_latestState?.rollDegrees ?? 0.0).toStringAsFixed(1)}°'),
                      _buildAngleChip('Pitch', '${(_latestState?.pitchDegrees ?? 0.0).toStringAsFixed(1)}°'),
                      _buildAngleChip('Yaw', '${(_latestState?.headingDegrees ?? 0.0).toStringAsFixed(1)}°'),
                    ],
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildMetricCard({
    required String title,
    required String value,
    required IconData icon,
    required Color accentColor,
  }) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFF161F30),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, size: 16, color: accentColor),
              const SizedBox(width: 6),
              Text(
                title,
                style: const TextStyle(color: Colors.white54, fontSize: 10, fontWeight: FontWeight.bold, letterSpacing: 0.8),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            value,
            style: const TextStyle(color: Colors.white, fontSize: 17, fontWeight: FontWeight.bold),
          ),
        ],
      ),
    );
  }

  Widget _buildAngleChip(String label, String value) {
    return Column(
      children: [
        Text(label, style: const TextStyle(color: Colors.white38, fontSize: 11)),
        const SizedBox(height: 4),
        Text(value, style: const TextStyle(color: Colors.white, fontSize: 13, fontWeight: FontWeight.w600)),
      ],
    );
  }
}

/// Custom painter rendering a 3D wireframe vehicle gizmo and 2D trajectory trail
class _VehicleVisualizerPainter extends CustomPainter {
  final double rollDeg;
  final double pitchDeg;
  final double yawDeg;
  final List<Offset> trail;
  final Color modeColor;

  _VehicleVisualizerPainter({
    required this.rollDeg,
    required this.pitchDeg,
    required this.yawDeg,
    required this.trail,
    required this.modeColor,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width * 0.35, size.height * 0.5);

    // 1. Draw Grid Background
    final gridPaint = Paint()
      ..color = Colors.white.withValues(alpha: 0.05)
      ..strokeWidth = 1.0;
    for (double x = 0; x < size.width; x += 30) {
      canvas.drawLine(Offset(x, 0), Offset(x, size.height), gridPaint);
    }
    for (double y = 0; y < size.height; y += 30) {
      canvas.drawLine(Offset(0, y), Offset(size.width, y), gridPaint);
    }

    // 2. Draw 2D Trajectory Trail on the right half
    if (trail.length > 1) {
      final trailPaint = Paint()
        ..color = modeColor.withValues(alpha: 0.8)
        ..strokeWidth = 2.0
        ..style = PaintingStyle.stroke;

      final trailCenter = Offset(size.width * 0.75, size.height * 0.5);
      final path = Path();
      final latest = trail.last;

      for (int i = 0; i < trail.length; i++) {
        final double dx = (trail[i].dx - latest.dx) * 0.15;
        final double dy = -(trail[i].dy - latest.dy) * 0.15;
        final pt = trailCenter + Offset(dx, dy);

        if (i == 0) {
          path.moveTo(pt.dx, pt.dy);
        } else {
          path.lineTo(pt.dx, pt.dy);
        }
      }
      canvas.drawPath(path, trailPaint);
    }

    // 3. Draw 3D Orientation Gizmo
    final double yawRad = -yawDeg * math.pi / 180.0;
    canvas.save();
    canvas.translate(center.dx, center.dy);
    canvas.rotate(yawRad);

    final vehiclePaint = Paint()
      ..color = Colors.white
      ..strokeWidth = 2.5
      ..style = PaintingStyle.stroke;

    final arrowPath = Path()
      ..moveTo(0, -35)
      ..lineTo(20, 25)
      ..lineTo(0, 15)
      ..lineTo(-20, 25)
      ..close();

    final fillPaint = Paint()
      ..color = modeColor.withValues(alpha: 0.25)
      ..style = PaintingStyle.fill;

    canvas.drawPath(arrowPath, fillPaint);
    canvas.drawPath(arrowPath, vehiclePaint);
    canvas.restore();
  }

  @override
  bool shouldRepaint(covariant _VehicleVisualizerPainter oldDelegate) => true;
}
