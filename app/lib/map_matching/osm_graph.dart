import 'dart:math' as math;

/// Represents a single directed road segment within the offline local road network.
class RoadSegment {
  final String id;
  final String name;
  final double startEast;
  final double startNorth;
  final double endEast;
  final double endNorth;
  final double speedLimitKmh;
  final bool isOneWay;

  // Cached segment properties
  late final double lengthMeters;
  late final double headingMathRad; // Math ENU angle theta CCW from East
  late final double headingCompassDeg;

  RoadSegment({
    required this.id,
    required this.name,
    required this.startEast,
    required this.startNorth,
    required this.endEast,
    required this.endNorth,
    this.speedLimitKmh = 50.0,
    this.isOneWay = false,
  }) {
    final double dE = endEast - startEast;
    final double dN = endNorth - startNorth;
    lengthMeters = math.sqrt(dE * dE + dN * dN);
    headingMathRad = math.atan2(dN, dE);

    double compass = 90.0 - (headingMathRad * 180.0 / math.pi);
    while (compass < 0.0) {
      compass += 360.0;
    }
    while (compass >= 360.0) {
      compass -= 360.0;
    }
    headingCompassDeg = compass;
  }

  /// Calculates the orthogonal distance and projected point on this segment from query point (pE, pN).
  Map<String, double> projectPoint(double pE, double pN) {
    if (lengthMeters < 1e-3) {
      final double dist = math.sqrt((pE - startEast) * (pE - startEast) + (pN - startNorth) * (pN - startNorth));
      return {'dist': dist, 'projE': startEast, 'projN': startNorth, 't': 0.0};
    }

    final double dE = endEast - startEast;
    final double dN = endNorth - startNorth;

    // Linear projection parameter t in [0, 1]
    double t = ((pE - startEast) * dE + (pN - startNorth) * dN) / (lengthMeters * lengthMeters);
    t = math.max(0.0, math.min(1.0, t));

    final double projE = startEast + t * dE;
    final double projN = startNorth + t * dN;
    final double dist = math.sqrt((pE - projE) * (pE - projE) + (pN - projN) * (pN - projN));

    return {'dist': dist, 'projE': projE, 'projN': projN, 't': t};
  }
}

/// In-memory spatial index representing the offline OpenStreetMap road network.
class OsmRoadGraph {
  final List<RoadSegment> _segments = [];

  List<RoadSegment> get segments => List.unmodifiable(_segments);

  void addSegment(RoadSegment segment) {
    _segments.add(segment);
  }

  /// Loads bundled offline road geometry for the test region (or synthesized urban grid)
  void loadBundledTestNetwork() {
    _segments.clear();
    // Default urban arterial grid & roundabouts matching IO-VNBD test area
    _segments.add(RoadSegment(
      id: 'seg_1',
      name: 'Main Arterial Eastbound',
      startEast: 0.0,
      startNorth: 0.0,
      endEast: 1500.0,
      endNorth: 0.0,
      speedLimitKmh: 60.0,
    ));
    _segments.add(RoadSegment(
      id: 'seg_2',
      name: 'Roundabout 1 Entry',
      startEast: 1500.0,
      startNorth: 0.0,
      endEast: 1800.0,
      endNorth: 400.0,
      speedLimitKmh: 40.0,
    ));
    _segments.add(RoadSegment(
      id: 'seg_3',
      name: 'Roundabout 1 North Exit',
      startEast: 1800.0,
      startNorth: 400.0,
      endEast: 1800.0,
      endNorth: 2000.0,
      speedLimitKmh: 70.0,
    ));
    _segments.add(RoadSegment(
      id: 'seg_4',
      name: 'Motorway Dual Carriageway',
      startEast: 1800.0,
      startNorth: 2000.0,
      endEast: 4500.0,
      endNorth: 3500.0,
      speedLimitKmh: 100.0,
    ));
  }

  /// Finds the N nearest candidate road segments within searchRadiusMeters
  List<RoadSegment> findCandidateSegments({
    required double east,
    required double north,
    double searchRadiusMeters = 50.0,
  }) {
    final List<MapEntry<RoadSegment, double>> candidates = [];

    for (var seg in _segments) {
      final proj = seg.projectPoint(east, north);
      final double dist = proj['dist']!;
      if (dist <= searchRadiusMeters) {
        candidates.add(MapEntry(seg, dist));
      }
    }

    candidates.sort((a, b) => a.value.compareTo(b.value));
    return candidates.map((e) => e.key).toList();
  }
}
