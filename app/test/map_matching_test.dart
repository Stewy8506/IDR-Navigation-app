import 'dart:math' as math;
import 'package:flutter_test/flutter_test.dart';
import 'package:idr_nav/map_matching/hmm_map_matcher.dart';
import 'package:idr_nav/map_matching/osm_graph.dart';

void main() {
  group('OsmRoadGraph & HmmMapMatcher Unit Tests', () {
    test('RoadSegment orthogonal projection works accurately', () {
      final seg = RoadSegment(
        id: 'test_1',
        name: 'Test Road',
        startEast: 0.0,
        startNorth: 0.0,
        endEast: 100.0,
        endNorth: 0.0,
      );

      // Point (50, 10) should project to (50, 0) with distance = 10m
      final proj = seg.projectPoint(50.0, 10.0);
      expect(proj['dist'], closeTo(10.0, 1e-4));
      expect(proj['projE'], closeTo(50.0, 1e-4));
      expect(proj['projN'], closeTo(0.0, 1e-4));
    });

    test('HmmMapMatcher snaps nearby noisy point to road centerline', () {
      final graph = OsmRoadGraph();
      graph.loadBundledTestNetwork();
      final matcher = HmmMapMatcher(graph: graph);

      // Point near Main Arterial Eastbound (start: 0,0, end: 1500, 0)
      final match = matcher.match(
        currentEast: 200.0,
        currentNorth: 8.0, // 8m off centerline
        currentHeadingMathRad: 0.0, // Heading East
      );

      expect(match.isSnapped, isTrue);
      expect(match.snappedNorth, closeTo(0.0, 1e-4));
      expect(match.snappedEast, closeTo(200.0, 1e-4));
      expect(match.roadName, contains('Main Arterial'));
    });
  });
}
