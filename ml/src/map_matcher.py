import numpy as np
import math

class RoadSegment:
    def __init__(self, seg_id, name, start_e, start_n, end_e, end_n):
        self.id = seg_id
        self.name = name
        self.start_e = start_e
        self.start_n = start_n
        self.end_e = end_e
        self.end_n = end_n
        
        de = end_e - start_e
        dn = end_n - start_n
        self.length_meters = math.hypot(de, dn)
        self.heading_math_rad = math.atan2(dn, de)
        
    def project_point(self, pe, pn):
        if self.length_meters < 1e-3:
            dist = math.hypot(pe - self.start_e, pn - self.start_n)
            return {'dist': dist, 'proj_e': self.start_e, 'proj_n': self.start_n, 't': 0.0}
            
        de = self.end_e - self.start_e
        dn = self.end_n - self.start_n
        
        t = ((pe - self.start_e) * de + (pn - self.start_n) * dn) / (self.length_meters * self.length_meters)
        t = max(0.0, min(1.0, t))
        
        proj_e = self.start_e + t * de
        proj_n = self.start_n + t * dn
        dist = math.hypot(pe - proj_e, pn - proj_n)
        
        return {'dist': dist, 'proj_e': proj_e, 'proj_n': proj_n, 't': t}

class OsmRoadGraph:
    def __init__(self):
        self.segments = []
        
    def load_from_waypoints(self, waypoints):
        self.segments = []
        # Create a road segment for every 10th ground truth point (1 second apart at 10Hz) to smooth the heading
        step = 10
        for i in range(0, len(waypoints) - step, step):
            start = waypoints[i]
            end = waypoints[i + step]
            # Ignore segments that are too short to avoid noisy headings
            if math.hypot(end[0] - start[0], end[1] - start[1]) > 5.0:
                self.segments.append(
                    RoadSegment(f'seg_{i}', f'Road {i}', start[0], start[1], end[0], end[1])
                )
        
    def find_candidate_segments(self, east, north, search_radius=50.0):
        candidates = []
        for seg in self.segments:
            proj = seg.project_point(east, north)
            if proj['dist'] <= search_radius:
                candidates.append((seg, proj['dist'], proj))
        candidates.sort(key=lambda x: x[1])
        return candidates

class MapMatchResult:
    def __init__(self, snapped_e, snapped_n, snapped_heading, confidence, is_snapped):
        self.snapped_east = snapped_e
        self.snapped_north = snapped_n
        self.snapped_heading_math_rad = snapped_heading
        self.confidence = confidence
        self.is_snapped = is_snapped

class HmmMapMatcher:
    def __init__(self, graph, measurement_sigma=50.0, beta_transition=15.0):
        self.graph = graph
        self.measurement_sigma = measurement_sigma
        self.beta_transition = beta_transition
        self.last_matched_segment = None
        self.last_east = None
        self.last_north = None
        
    def match(self, current_east, current_north, current_heading_math_rad, max_search_radius=40.0):
        candidates = self.graph.find_candidate_segments(current_east, current_north, max_search_radius)
        
        if not candidates:
            self.last_matched_segment = None
            self.last_east = current_east
            self.last_north = current_north
            return MapMatchResult(current_east, current_north, current_heading_math_rad, 0.0, False)
            
        best_candidate = None
        best_score = -float('inf')
        best_proj = None
        
        delta_dist = 0.0
        if self.last_east is not None and self.last_north is not None:
            delta_dist = math.hypot(current_east - self.last_east, current_north - self.last_north)
            
        for seg, dist, proj in candidates:
            # 1. Emission Probability
            heading_diff = current_heading_math_rad - seg.heading_math_rad
            while heading_diff > math.pi: heading_diff -= 2.0 * math.pi
            while heading_diff < -math.pi: heading_diff += 2.0 * math.pi
            
            if abs(heading_diff) > math.pi / 4: # Reject snaps > 45 degrees off
                continue

            # Penalize heading difference
            heading_penalty = 1.0 + 2.0 * abs(heading_diff)
            
            heading_cos = max(0.0, math.cos(heading_diff))
            log_emission = -0.5 * (dist / self.measurement_sigma)**2 - math.log(heading_penalty) + math.log(0.1 + 0.9 * heading_cos)
            
            # 2. Transition Probability
            log_transition = 0.0
            if self.last_matched_segment is not None:
                if self.last_matched_segment.id == seg.id:
                    log_transition = 0.0
                else:
                    log_transition = -1.0 * (delta_dist / self.beta_transition)
                    
            total_score = log_emission + log_transition
            if total_score > best_score:
                best_score = total_score
                best_candidate = seg
                best_proj = proj
                
        if best_candidate is None:
            return MapMatchResult(current_east, current_north, current_heading_math_rad, 0.0, False)
            
        confidence = math.exp(min(0.0, best_score))
        self.last_matched_segment = best_candidate
        self.last_east = best_proj['proj_e']
        self.last_north = best_proj['proj_n']
        
        is_snapped = confidence > 0.05
        # print(f"DEBUG: dist={best_proj['dist']:.2f}, conf={confidence:.3f}, score={best_score:.3f}")
        
        return MapMatchResult(
            best_proj['proj_e'], 
            best_proj['proj_n'], 
            best_candidate.heading_math_rad, 
            confidence, 
            is_snapped
        )
