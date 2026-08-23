import 'dart:math' as math;
import 'osm_graph.dart';

/// Result of an HMM Map-Matching step
class MapMatchResult {
  final double snappedEast;
  final double snappedNorth;
  final double snappedHeadingMathRad;
  final double confidence; // [0.0, 1.0]
  final String? roadName;
  final bool isSnapped;

  const MapMatchResult({
    required this.snappedEast,
    required this.snappedNorth,
    required this.snappedHeadingMathRad,
    required this.confidence,
    this.roadName,
    required this.isSnapped,
  });

  const MapMatchResult.unmatched({
    required double rawEast,
    required double rawNorth,
    required double rawHeadingMathRad,
  }) : snappedEast = rawEast,
       snappedNorth = rawNorth,
       snappedHeadingMathRad = rawHeadingMathRad,
       confidence = 0.0,
       roadName = null,
       isSnapped = false;
}

/// Hidden Markov Model (HMM) Viterbi Map-Matcher for offline road snapping
class HmmMapMatcher {
  final OsmRoadGraph graph;
  final double measurementSigmaMeters;
  final double betaTransition;

  RoadSegment? _lastMatchedSegment;
  double? _lastEast;
  double? _lastNorth;

  HmmMapMatcher({
    required this.graph,
    this.measurementSigmaMeters = 10.0,
    this.betaTransition = 15.0,
  });

  /// Evaluates candidates and returns the most probable road match
  MapMatchResult match({
    required double currentEast,
    required double currentNorth,
    required double currentHeadingMathRad,
    double maxSearchRadiusMeters = 40.0,
  }) {
    final candidates = graph.findCandidateSegments(
      east: currentEast,
      north: currentNorth,
      searchRadiusMeters: maxSearchRadiusMeters,
    );

    if (candidates.isEmpty) {
      _lastMatchedSegment = null;
      _lastEast = currentEast;
      _lastNorth = currentNorth;
      return MapMatchResult.unmatched(
        rawEast: currentEast,
        rawNorth: currentNorth,
        rawHeadingMathRad: currentHeadingMathRad,
      );
    }

    RoadSegment? bestCandidate;
    double bestScore = -double.infinity;
    Map<String, double>? bestProj;

    final double deltaDistTraveled = (_lastEast != null && _lastNorth != null)
        ? math.sqrt(math.pow(currentEast - _lastEast!, 2) + math.pow(currentNorth - _lastNorth!, 2))
        : 0.0;

    for (var seg in candidates) {
      final proj = seg.projectPoint(currentEast, currentNorth);
      final double dist = proj['dist']!;

      // 1. Emission Probability: Gaussian on distance + heading alignment penalty
      double headingDiff = (seg.headingMathRad - currentHeadingMathRad).abs();
      while (headingDiff > math.pi) {
        headingDiff -= 2.0 * math.pi;
      }
      while (headingDiff < -math.pi) {
        headingDiff += 2.0 * math.pi;
      }
      headingDiff = headingDiff.abs();

      // Heading alignment weight
      final double headingCos = math.max(0.0, math.cos(headingDiff));
      final double logEmission = -0.5 * math.pow(dist / measurementSigmaMeters, 2) + math.log(0.1 + 0.9 * headingCos);

      // 2. Transition Probability
      double logTransition = 0.0;
      if (_lastMatchedSegment != null) {
        if (_lastMatchedSegment!.id == seg.id) {
          logTransition = 0.0; // Same segment continuity bonus
        } else {
          // Topology transition penalty
          logTransition = -1.0 * (deltaDistTraveled / betaTransition);
        }
      }

      final double totalScore = logEmission + logTransition;
      if (totalScore > bestScore) {
        bestScore = totalScore;
        bestCandidate = seg;
        bestProj = proj;
      }
    }

    if (bestCandidate == null || bestProj == null) {
      return MapMatchResult.unmatched(
        rawEast: currentEast,
        rawNorth: currentNorth,
        rawHeadingMathRad: currentHeadingMathRad,
      );
    }

    final double confidence = math.exp(math.min(0.0, bestScore));
    _lastMatchedSegment = bestCandidate;
    _lastEast = bestProj['projE']!;
    _lastNorth = bestProj['projN']!;

    return MapMatchResult(
      snappedEast: bestProj['projE']!,
      snappedNorth: bestProj['projN']!,
      snappedHeadingMathRad: bestCandidate.headingMathRad,
      confidence: confidence,
      roadName: bestCandidate.name,
      isSnapped: confidence > 0.35,
    );
  }

  void reset() {
    _lastMatchedSegment = null;
    _lastEast = null;
    _lastNorth = null;
  }
}
