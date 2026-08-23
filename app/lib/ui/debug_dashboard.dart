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

  @override
  void initState() {
    super.initState();
    _engine = IdrNavEngine(sensorService: LiveSensorService());
    _startEngine();
  }

  Future<void> _startEngine() async {
    await _engine.start();
    _engine.navStateStream.listen((state) {
      if (mounted) {
        setState(() {
          _latestState = state;
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
        return Colors.greenAccent;
      case NavMode.deadReckoning:
        return Colors.orangeAccent;
      case NavMode.calibrating:
        return Colors.lightBlueAccent;
    }
  }

  @override
  Widget build(BuildContext context) {
    final mode = _latestState?.mode ?? NavMode.calibrating;

    return Scaffold(
      backgroundColor: const Color(0xFF0F172A),
      appBar: AppBar(
        backgroundColor: const Color(0xFF1E293B),
        title: const Text(
          'IDR-Nav Engine Monitor',
          style: TextStyle(fontWeight: FontWeight.bold, fontSize: 18, color: Colors.white),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh, color: Colors.white70),
            onPressed: () {
              _engine.alignmentEstimator.reset();
            },
            tooltip: 'Reset Alignment',
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
                color: _getModeColor(mode).withValues(alpha: 0.15),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: _getModeColor(mode).withValues(alpha: 0.5)),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Row(
                    children: [
                      Icon(
                        mode == NavMode.gnssAided
                            ? Icons.satellite_alt
                            : mode == NavMode.deadReckoning
                                ? Icons.directions_car
                                : Icons.sync,
                        color: _getModeColor(mode),
                      ),
                      const SizedBox(width: 10),
                      Text(
                        'MODE: ${mode.name.toUpperCase()}',
                        style: TextStyle(
                          color: _getModeColor(mode),
                          fontWeight: FontWeight.bold,
                          fontSize: 14,
                          letterSpacing: 1.1,
                        ),
                      ),
                    ],
                  ),
                  Text(
                    '±${_latestState?.positionUncertaintyMeters.toStringAsFixed(1) ?? "--"} m',
                    style: const TextStyle(color: Colors.white70, fontSize: 13),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),

            // Telemetry Grid
            Row(
              children: [
                Expanded(
                  child: _buildMetricCard(
                    title: 'FORWARD SPEED',
                    value: _latestState != null
                        ? '${(_latestState!.speedMps * 3.6).toStringAsFixed(1)} km/h'
                        : '-- km/h',
                    icon: Icons.speed,
                    accentColor: Colors.tealAccent,
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: _buildMetricCard(
                    title: 'HEADING',
                    value: _latestState != null
                        ? '${_latestState!.headingDegrees.toStringAsFixed(1)}°'
                        : '--°',
                    icon: Icons.explore,
                    accentColor: Colors.amberAccent,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),

            // Coordinates Card
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: const Color(0xFF1E293B),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'ESTIMATED POSITION (WGS84)',
                    style: TextStyle(color: Colors.white54, fontSize: 11, fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 8),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        'Lat: ${_latestState?.latitude.toStringAsFixed(6) ?? "0.000000"}',
                        style: const TextStyle(color: Colors.white, fontSize: 15, fontFamily: 'monospace'),
                      ),
                      Text(
                        'Lon: ${_latestState?.longitude.toStringAsFixed(6) ?? "0.000000"}',
                        style: const TextStyle(color: Colors.white, fontSize: 15, fontFamily: 'monospace'),
                      ),
                    ],
                  ),
                ],
              ),
            ),
            const SizedBox(height: 12),

            // Attitude Angles Card
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: const Color(0xFF1E293B),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'VEHICLE ATTITUDE',
                    style: TextStyle(color: Colors.white54, fontSize: 11, fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 8),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceAround,
                    children: [
                      _buildAngleItem('Roll', _latestState?.rollDegrees ?? 0.0),
                      _buildAngleItem('Pitch', _latestState?.pitchDegrees ?? 0.0),
                      _buildAngleItem('Yaw', _latestState?.headingDegrees ?? 0.0),
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
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF1E293B),
        borderRadius: BorderRadius.circular(12),
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
                style: const TextStyle(color: Colors.white54, fontSize: 10, fontWeight: FontWeight.bold),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Text(
            value,
            style: const TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold),
          ),
        ],
      ),
    );
  }

  Widget _buildAngleItem(String label, double valueDeg) {
    return Column(
      children: [
        Text(
          label,
          style: const TextStyle(color: Colors.white54, fontSize: 12),
        ),
        const SizedBox(height: 4),
        Text(
          '${valueDeg.toStringAsFixed(1)}°',
          style: const TextStyle(color: Colors.white, fontSize: 14, fontWeight: FontWeight.bold, fontFamily: 'monospace'),
        ),
      ],
    );
  }
}
