"""
Enhanced Machine Learning Engine for TPMS Signal Analysis
Includes predictive encounter modeling and advanced vehicle association
"""
import numpy as np
import time
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import List, Dict, Optional, Tuple, Set, Any
import math
from datetime import datetime, timedelta

@dataclass
class SignalCharacteristics:
    """Characteristics of a detected signal"""
    frequency: float
    power: float
    snr: float
    modulation: str
    baud_rate: int
    timestamp: float
    decoded: bool = False
    protocol: Optional[str] = None
    characteristics: Dict = field(default_factory=dict)

# ======================================================================================
# Enhanced Vehicle Clustering with Predictive Capabilities
# ======================================================================================

class VehicleClusteringEngine:
    """
    Enhanced clustering with:
    - Temporal pattern learning (commute schedules, parking patterns)
    - Location-aware clustering (home/work/frequent locations)
    - Signal strength fingerprinting per location
    - Predictive next encounter modeling
    """

    def __init__(
        self,
        db,
        # --- temporal mate logic ---
        base_mate_window_s: float = 60.0,
        relaxed_mate_window_s: float = 120.0,
        observation_window_s: float = 10.0,
        min_same_sensor_gap_s: float = 0.005,

        # --- scoring / thresholds ---
        suspect_score: float = 120.0,
        link_score: float = 2.0,
        stable_pair_ratio: float = 0.60,

        # --- forgetting ---
        decay_halflife_s: float = 180.0,

        # --- safety gates ---
        max_vehicle_size: int = 8,
        min_vehicle_size: int = 2,

        # --- DB cache ---
        refresh_vehicle_cache_s: float = 300.0,

        # --- NEW: predictive parameters ---
        enable_prediction: bool = True,
        location_cell_deg: float = 0.001,  # ~100m cells for location clustering
        min_pattern_observations: int = 5,
    ):
        self.db = db

        self.base_mate_window_s = float(base_mate_window_s)
        self.relaxed_mate_window_s = float(relaxed_mate_window_s)
        self.observation_window_s = float(observation_window_s)
        self.min_same_sensor_gap_s = float(min_same_sensor_gap_s)

        self.suspect_score = float(suspect_score)
        self.link_score = float(link_score)
        self.stable_pair_ratio = float(stable_pair_ratio)

        self.decay_halflife_s = float(decay_halflife_s)

        self.max_vehicle_size = int(max_vehicle_size)
        self.min_vehicle_size = int(min_vehicle_size)

        # Original clustering state
        self.pair_scores = defaultdict(float)
        self._last_decay_ts = time.time()
        self._recent_detections = deque()
        self._last_seen_sensor_ts = {}

        self._vehicle_cache = []
        self._vehicle_cache_ts = 0.0
        self._vehicle_cache_ttl = float(refresh_vehicle_cache_s)

        self.recent_windows = deque(maxlen=300)
        self.vehicle_profiles: Dict[int, Dict] = {}
        self.clusters: Dict[int, List[str]] = {}

        # NEW: Predictive modeling components
        self.enable_prediction = enable_prediction
        self.location_cell_deg = float(location_cell_deg)
        self.min_pattern_observations = int(min_pattern_observations)

        # Temporal patterns: vehicle_id -> hour -> encounter_count
        self.temporal_patterns: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

        # Location patterns: vehicle_id -> (lat_cell, lon_cell) -> encounter_count
        self.location_patterns: Dict[int, Dict[Tuple[int, int], int]] = defaultdict(lambda: defaultdict(int))

        # Encounter intervals: vehicle_id -> [time_gaps_between_encounters]
        self.encounter_intervals: Dict[int, deque] = defaultdict(lambda: deque(maxlen=50))

        # Last encounter: vehicle_id -> timestamp
        self.last_encounter: Dict[int, float] = {}

        # Signal fingerprints: vehicle_id -> location_cell -> {avg_rssi, avg_snr, count}
        self.signal_fingerprints: Dict[int, Dict[Tuple[int, int], Dict]] = defaultdict(lambda: defaultdict(lambda: {'rssi': [], 'snr': [], 'count': 0}))

        # Missed encounter tracking: vehicle_id -> [expected_but_missed_timestamps]
        self.missed_encounters: Dict[int, List[float]] = defaultdict(list)

    # -----------------------------
    # Utilities (unchanged)
    # -----------------------------

    @staticmethod
    def _pair_key(a: str, b: str) -> Tuple[str, str]:
        return (a, b) if a <= b else (b, a)

    def _decay(self, now: float):
        """Exponentially decay pair scores over time."""
        dt = max(0.0, now - self._last_decay_ts)
        if dt <= 0:
            return
        decay_factor = 0.5 ** (dt / self.decay_halflife_s)
        if decay_factor < 0.999:
            for k in list(self.pair_scores.keys()):
                self.pair_scores[k] *= decay_factor
                if self.pair_scores[k] < 0.05:
                    del self.pair_scores[k]
        self._last_decay_ts = now

    def _effective_window(self, a: str, b: str) -> float:
        """Relax mate window if this pair is already suspected."""
        score = self.pair_scores.get(self._pair_key(a, b), 0.0)
        return self.relaxed_mate_window_s if score >= self.suspect_score else self.base_mate_window_s

    def get_pair_likelihood(self, a: str, b: str) -> float:
        """Map raw score -> [0,1) likelihood."""
        s = self.pair_scores.get(self._pair_key(a, b), 0.0)
        return 1.0 - math.exp(-s / max(1e-6, self.link_score))

    # -----------------------------
    # NEW: Location utilities
    # -----------------------------

    def _location_cell(self, lat: Optional[float], lon: Optional[float]) -> Optional[Tuple[int, int]]:
        """Convert lat/lon to grid cell for location clustering."""
        if lat is None or lon is None:
            return None
        try:
            lat_cell = int(np.floor(float(lat) / self.location_cell_deg))
            lon_cell = int(np.floor(float(lon) / self.location_cell_deg))
            return (lat_cell, lon_cell)
        except (ValueError, TypeError):
            return None

    # -----------------------------
    # Ingestion (enhanced)
    # -----------------------------

    def _ingest_detection(self, sid: str, ts: float):
        """For each new detection: compare against other detections within relaxed_mate_window_s"""
        if not sid:
            return

        # debounce same sensor spam
        last = self._last_seen_sensor_ts.get(sid)
        if last is not None and (ts - last) < self.min_same_sensor_gap_s:
            return
        self._last_seen_sensor_ts[sid] = ts

        self._decay(ts)

        # purge stale detections
        cutoff = ts - self.relaxed_mate_window_s
        while self._recent_detections and self._recent_detections[0][1] < cutoff:
            self._recent_detections.popleft()

        # compare to recent detections
        for other_sid, other_ts in reversed(self._recent_detections):
            dt = ts - other_ts
            if dt > self.relaxed_mate_window_s:
                break
            if other_sid == sid:
                continue

            window = self._effective_window(sid, other_sid)
            if dt <= window:
                k = self._pair_key(sid, other_sid)
                self.pair_scores[k] += 1.0

        self._recent_detections.append((sid, ts))

    def _ingest_signal_buffer(self, signal_buffer: List[Dict]) -> float:
        """Ingest all signals in time order."""
        events: List[Tuple[str, float]] = []
        now = time.time()

        for s in signal_buffer:
            sid = s.get("tpms_id")
            if not sid:
                continue
            ts = float(s.get("timestamp", now))
            events.append((sid, ts))

        if not events:
            return now

        events.sort(key=lambda x: x[1])
        for sid, ts in events:
            self._ingest_detection(sid, ts)

        return max(ts for _, ts in events)

    # -----------------------------
    # Graph building (unchanged)
    # -----------------------------

    def _adjacency(self, min_score: float) -> Dict[str, Set[str]]:
        adj = defaultdict(set)
        for (a, b), score in self.pair_scores.items():
            if score >= min_score:
                adj[a].add(b)
                adj[b].add(a)
        return adj

    def _components(self, min_score: float) -> List[List[str]]:
        """Connected components among edges with score >= min_score."""
        adj = self._adjacency(min_score)
        seen = set()
        comps: List[List[str]] = []

        for node in adj.keys():
            if node in seen:
                continue
            stack = [node]
            seen.add(node)
            comp = []

            while stack:
                x = stack.pop()
                comp.append(x)
                for y in adj[x]:
                    if y not in seen:
                        seen.add(y)
                        stack.append(y)

            comps.append(sorted(comp))

        return comps

    def _is_group_stable(self, ids: List[str]) -> bool:
        """Stable group heuristic."""
        ids = sorted(set(ids))
        n = len(ids)
        if n < 2:
            return False

        good = 0
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += 1
                if self.pair_scores.get(self._pair_key(ids[i], ids[j]), 0.0) >= self.link_score:
                    good += 1

        return total > 0 and (good / total) >= self.stable_pair_ratio

    def _observed_ids(self, now_ts: float) -> List[str]:
        """Sensors seen within observation_window_s of now_ts."""
        cutoff = now_ts - self.observation_window_s
        obs = {sid for sid, ts in self._recent_detections if ts >= cutoff}
        return sorted(obs)

    def _expand_with_suspected_mates(self, ids: List[str]) -> List[str]:
        """Expand the set by one-hop mate edges."""
        if not ids:
            return ids

        adj_suspect = self._adjacency(self.suspect_score)

        expanded = set(ids)
        changed = True
        for _ in range(2):
            if not changed:
                break
            changed = False
            for sid in list(expanded):
                for mate in adj_suspect.get(sid, set()):
                    if mate not in expanded:
                        expanded.add(mate)
                        changed = True

        return sorted(expanded)

    # -----------------------------
    # NEW: Pattern Learning
    # -----------------------------

    def _update_patterns(self, vehicle_id: int, ts: float, location: Optional[Tuple[float, float]], signal_buffer: List[Dict]):
        """Update temporal and spatial patterns for a vehicle."""
        if not self.enable_prediction:
            return

        # Update temporal patterns (hour of day)
        try:
            dt = datetime.fromtimestamp(ts)
            hour = dt.hour
            self.temporal_patterns[vehicle_id][hour] += 1
        except Exception:
            pass

        # Update location patterns
        if location and location[0] is not None and location[1] is not None:
            cell = self._location_cell(location[0], location[1])
            if cell:
                self.location_patterns[vehicle_id][cell] += 1

                # Update signal fingerprints for this location
                for sig in signal_buffer:
                    if sig.get('tpms_id') in self.vehicle_profiles.get(vehicle_id, {}).get('sensor_ids', []):
                        rssi = sig.get('signal_strength')
                        snr = sig.get('snr')
                        if rssi is not None:
                            self.signal_fingerprints[vehicle_id][cell]['rssi'].append(float(rssi))
                        if snr is not None:
                            self.signal_fingerprints[vehicle_id][cell]['snr'].append(float(snr))
                        self.signal_fingerprints[vehicle_id][cell]['count'] += 1

                        # Keep only recent samples
                        if len(self.signal_fingerprints[vehicle_id][cell]['rssi']) > 100:
                            self.signal_fingerprints[vehicle_id][cell]['rssi'] = self.signal_fingerprints[vehicle_id][cell]['rssi'][-100:]
                        if len(self.signal_fingerprints[vehicle_id][cell]['snr']) > 100:
                            self.signal_fingerprints[vehicle_id][cell]['snr'] = self.signal_fingerprints[vehicle_id][cell]['snr'][-100:]

        # Update encounter intervals
        if vehicle_id in self.last_encounter:
            interval = ts - self.last_encounter[vehicle_id]
            if interval > 0:  # Sanity check
                self.encounter_intervals[vehicle_id].append(interval)

        self.last_encounter[vehicle_id] = ts

    # -----------------------------
    # NEW: Predictive Analytics
    # -----------------------------

    def predict_next_encounter(self, vehicle_id: int, current_time: Optional[float] = None) -> Dict[str, Any]:
        """
        Predict when we'll next see this vehicle.

        Returns:
            {
                'predicted_time': timestamp,
                'confidence': 0-1,
                'interval_mean': average time between encounters,
                'interval_std': standard deviation,
                'likely_hours': [hours of day with highest probability],
                'likely_locations': [(lat_cell, lon_cell, probability), ...]
            }
        """
        if current_time is None:
            current_time = time.time()

        result = {
            'predicted_time': None,
            'confidence': 0.0,
            'interval_mean': None,
            'interval_std': None,
            'likely_hours': [],
            'likely_locations': []
        }

        # Need minimum observations
        intervals = list(self.encounter_intervals.get(vehicle_id, []))
        if len(intervals) < self.min_pattern_observations:
            result['confidence'] = 0.0
            return result

        # Calculate interval statistics
        interval_mean = float(np.mean(intervals))
        interval_std = float(np.std(intervals))

        result['interval_mean'] = interval_mean
        result['interval_std'] = interval_std

        # Predict next encounter based on last encounter + mean interval
        last_enc = self.last_encounter.get(vehicle_id)
        if last_enc:
            result['predicted_time'] = last_enc + interval_mean

            # Confidence based on consistency of intervals (lower std = higher confidence)
            cv = interval_std / interval_mean if interval_mean > 0 else float('inf')
            result['confidence'] = float(np.clip(1.0 / (1.0 + cv), 0.0, 1.0))

        # Find likely hours (top 3)
        hour_counts = self.temporal_patterns.get(vehicle_id, {})
        if hour_counts:
            sorted_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)
            result['likely_hours'] = [h for h, _ in sorted_hours[:3]]

        # Find likely locations (top 5)
        loc_counts = self.location_patterns.get(vehicle_id, {})
        if loc_counts:
            total = sum(loc_counts.values())
            sorted_locs = sorted(loc_counts.items(), key=lambda x: x[1], reverse=True)
            result['likely_locations'] = [
                (cell, count / total) for cell, count in sorted_locs[:5]
            ]

        return result

    def detect_missed_encounters(self, vehicle_id: int, current_time: Optional[float] = None, threshold_multiplier: float = 2.0) -> Dict[str, Any]:
        """
        Detect if we likely missed an encounter with this vehicle.

        Args:
            vehicle_id: Vehicle to check
            current_time: Current timestamp
            threshold_multiplier: How many standard deviations past mean to flag as missed

        Returns:
            {
                'missed': bool,
                'expected_time': when we expected to see it,
                'overdue_by': seconds overdue,
                'confidence': 0-1
            }
        """
        if current_time is None:
            current_time = time.time()

        result = {
            'missed': False,
            'expected_time': None,
            'overdue_by': 0.0,
            'confidence': 0.0
        }

        # Need pattern data
        intervals = list(self.encounter_intervals.get(vehicle_id, []))
        if len(intervals) < self.min_pattern_observations:
            return result

        last_enc = self.last_encounter.get(vehicle_id)
        if not last_enc:
            return result

        # Calculate expected next encounter
        interval_mean = float(np.mean(intervals))
        interval_std = float(np.std(intervals))

        expected_time = last_enc + interval_mean
        threshold = interval_mean + (threshold_multiplier * interval_std)

        time_since_last = current_time - last_enc

        result['expected_time'] = expected_time
        result['overdue_by'] = max(0.0, time_since_last - interval_mean)

        if time_since_last > threshold:
            result['missed'] = True
            # Confidence increases with how overdue we are
            overdue_ratio = (time_since_last - interval_mean) / interval_std if interval_std > 0 else 0
            result['confidence'] = float(np.clip(overdue_ratio / threshold_multiplier, 0.0, 1.0))

            # Log the missed encounter
            self.missed_encounters[vehicle_id].append(expected_time)
            # Keep only recent missed encounters
            if len(self.missed_encounters[vehicle_id]) > 20:
                self.missed_encounters[vehicle_id] = self.missed_encounters[vehicle_id][-20:]

        return result

    def get_incomplete_sensor_sets(self, vehicle_id: int, observed_sensors: List[str]) -> Dict[str, Any]:
        """
        Analyze if we're seeing an incomplete sensor set for a known vehicle.

        Returns:
            {
                'complete': bool,
                'expected_sensors': [all known sensors],
                'missing_sensors': [sensors we expected but didn't see],
                'unexpected_sensors': [sensors we saw but don't recognize],
                'completeness_ratio': 0-1
            }
        """
        profile = self.vehicle_profiles.get(vehicle_id)
        if not profile:
            return {'complete': True, 'expected_sensors': [], 'missing_sensors': [], 'unexpected_sensors': [], 'completeness_ratio': 1.0}

        expected = set(profile.get('sensor_ids', []))
        observed = set(observed_sensors)

        missing = expected - observed
        unexpected = observed - expected

        completeness = len(observed & expected) / len(expected) if expected else 1.0

        return {
            'complete': len(missing) == 0,
            'expected_sensors': sorted(expected),
            'missing_sensors': sorted(missing),
            'unexpected_sensors': sorted(unexpected),
            'completeness_ratio': float(completeness)
        }

    def suggest_sensor_associations(self, orphan_sensor_id: str, current_time: Optional[float] = None,
                                   location: Optional[Tuple[float, float]] = None) -> List[Dict[str, Any]]:
        """
        Suggest which vehicle an orphan sensor might belong to based on:
        - Temporal patterns (is this vehicle expected now?)
        - Location patterns (is this vehicle expected here?)
        - Signal fingerprints (does RSSI/SNR match expected for this location?)
        - Incomplete sensor sets (which vehicles are we seeing partial sets for?)

        Returns:
            List of suggestions sorted by confidence:
            [{
                'vehicle_id': int,
                'confidence': 0-1,
                'reasons': ['temporal_match', 'location_match', 'signal_match', 'incomplete_set'],
                'score_breakdown': {...}
            }]
        """
        if current_time is None:
            current_time = time.time()

        suggestions = []

        for vehicle_id in self.vehicle_profiles.keys():
            scores = {
                'temporal': 0.0,
                'location': 0.0,
                'signal': 0.0,
                'incomplete': 0.0
            }
            reasons = []

            # Temporal matching
            try:
                dt = datetime.fromtimestamp(current_time)
                current_hour = dt.hour
                hour_counts = self.temporal_patterns.get(vehicle_id, {})
                if hour_counts:
                    total_encounters = sum(hour_counts.values())
                    hour_prob = hour_counts.get(current_hour, 0) / total_encounters
                    scores['temporal'] = float(hour_prob)
                    if hour_prob > 0.1:
                        reasons.append('temporal_match')
            except Exception:
                pass

            # Location matching
            if location and location[0] is not None and location[1] is not None:
                cell = self._location_cell(location[0], location[1])
                if cell:
                    loc_counts = self.location_patterns.get(vehicle_id, {})
                    if loc_counts:
                        total_encounters = sum(loc_counts.values())
                        loc_prob = loc_counts.get(cell, 0) / total_encounters
                        scores['location'] = float(loc_prob)
                        if loc_prob > 0.1:
                            reasons.append('location_match')

            # Check if vehicle has incomplete sensor set recently
            profile = self.vehicle_profiles.get(vehicle_id, {})
            expected_sensors = set(profile.get('sensor_ids', []))
            observed_recently = self._observed_ids(current_time)

            if expected_sensors:
                overlap = len(set(observed_recently) & expected_sensors)
                if overlap > 0 and overlap < len(expected_sensors):
                    scores['incomplete'] = 1.0 - (overlap / len(expected_sensors))
                    reasons.append('incomplete_set')

            # Overall confidence (weighted combination)
            confidence = (
                0.3 * scores['temporal'] +
                0.3 * scores['location'] +
                0.2 * scores['signal'] +
                0.2 * scores['incomplete']
            )

            if confidence > 0.1:  # Only suggest if some evidence
                suggestions.append({
                    'vehicle_id': vehicle_id,
                    'confidence': float(confidence),
                    'reasons': reasons,
                    'score_breakdown': scores
                })

        # Sort by confidence
        suggestions.sort(key=lambda x: x['confidence'], reverse=True)
        return suggestions

    # -----------------------------
    # Main Processing (enhanced)
    # -----------------------------

    def process_signals(self, signal_buffer: List[Dict]) -> List[int]:
        """
        Enhanced signal processing with pattern learning.
        """
        if not signal_buffer:
            return []

        # (1) update mate graph
        window_ts = self._ingest_signal_buffer(signal_buffer)

        # (2) candidates
        strong_components = self._components(self.link_score)

        observed = self._observed_ids(window_ts)
        observed_expanded = self._expand_with_suspected_mates(observed)

        if observed:
            self.recent_windows.append((window_ts, frozenset(observed)))

        candidates: List[List[str]] = []
        candidates.extend(strong_components)
        if observed_expanded:
            candidates.append(observed_expanded)

        # de-dupe candidates
        uniq = []
        seen_sets = set()
        for c in candidates:
            cset = tuple(sorted(set(c)))
            if cset not in seen_sets:
                seen_sets.add(cset)
                uniq.append(list(cset))
        candidates = uniq

        vehicle_ids: List[int] = []
        self.clusters = {}

        # location passthrough
        lat = next((s.get("latitude") for s in signal_buffer if s.get("latitude") is not None), None)
        lon = next((s.get("longitude") for s in signal_buffer if s.get("longitude") is not None), None)
        location = (lat, lon) if (lat is not None and lon is not None) else None

        for idx, ids in enumerate(candidates):
            if len(ids) < self.min_vehicle_size:
                continue
            if len(ids) > self.max_vehicle_size:
                continue

            stable = self._is_group_stable(ids)
            vehicle_id = self.identify_vehicle(ids)

            if vehicle_id is None:
                if not stable:
                    continue
                vehicle_id = self.db.upsert_vehicle(ids, window_ts, location=location)
                print(f"🚗 New vehicle (stable mates): ID={vehicle_id}, sensors={ids}", flush=True)
            else:
                self.db.upsert_vehicle(ids, window_ts, location=location)

            vehicle_ids.append(vehicle_id)
            self.update_vehicle_profile(vehicle_id, ids, window_ts, signal_buffer)

            # NEW: Update patterns
            self._update_patterns(vehicle_id, window_ts, location, signal_buffer)

            self.clusters[idx] = ids

        return sorted(set(vehicle_ids))

    # -----------------------------
    # Matching (unchanged)
    # -----------------------------

    def _refresh_vehicle_cache(self):
        now = time.time()
        if now - self._vehicle_cache_ts > self._vehicle_cache_ttl:
            self._vehicle_cache = self.db.get_all_vehicles(min_encounters=1)
            self._vehicle_cache_ts = now

    def identify_vehicle(self, sensor_ids: List[str]) -> Optional[int]:
        """Match sensors to an existing vehicle using Jaccard score."""
        if not sensor_ids:
            return None

        self._refresh_vehicle_cache()

        input_sensors = set(sensor_ids)
        best_id = None
        best_score = 0.0

        for v in self._vehicle_cache:
            v_sensors = set(v.get("tpms_ids", []))
            if not v_sensors:
                continue

            overlap = len(v_sensors & input_sensors)
            if overlap < 2:
                continue

            union = len(v_sensors | input_sensors)
            score = overlap / union

            if score > best_score and score >= 0.35:
                best_score = score
                best_id = v["id"]

        return best_id

    # -----------------------------
    # Profiles (unchanged)
    # -----------------------------

    def update_vehicle_profile(self, vehicle_id: int, sensor_ids: List[str], ts: float, signal_buffer: List[Dict]):
        if vehicle_id not in self.vehicle_profiles:
            self.vehicle_profiles[vehicle_id] = {
                "sensor_ids": [],
                "first_seen": ts,
                "last_seen": ts,
                "detection_count": 0,
                "avg_signal_strength": None,
                "avg_snr": None,
                "typical_frequency": None,
            }

        p = self.vehicle_profiles[vehicle_id]
        p["last_seen"] = max(p["last_seen"], ts)
        p["first_seen"] = min(p["first_seen"], ts)
        p["detection_count"] += 1

        for sid in sensor_ids:
            if sid not in p["sensor_ids"]:
                p["sensor_ids"].append(sid)

        rel = [s for s in signal_buffer if s.get("tpms_id") in set(sensor_ids)]
        if rel:
            vals_rssi = [s.get("signal_strength") for s in rel if s.get("signal_strength") is not None]
            vals_snr = [s.get("snr") for s in rel if s.get("snr") is not None]
            vals_f = [s.get("frequency") for s in rel if s.get("frequency") is not None]

            p["avg_signal_strength"] = float(np.mean(vals_rssi)) if vals_rssi else p["avg_signal_strength"]
            p["avg_snr"] = float(np.mean(vals_snr)) if vals_snr else p["avg_snr"]
            p["typical_frequency"] = float(np.mean(vals_f)) if vals_f else p["typical_frequency"]

    def get_statistics(self) -> Dict:
        """Enhanced statistics including predictive metrics."""
        base_stats = {
            "pair_edges": len(self.pair_scores),
            "num_components_strong": len(self._components(self.link_score)),
            "num_vehicle_profiles": len(self.vehicle_profiles),
            "clusters_last_run": {k: len(v) for k, v in self.clusters.items()},
        }

        if self.enable_prediction:
            base_stats.update({
                "vehicles_with_patterns": len(self.temporal_patterns),
                "total_location_cells": sum(len(locs) for locs in self.location_patterns.values()),
                "avg_encounters_per_vehicle": float(np.mean([len(intervals) for intervals in self.encounter_intervals.values()])) if self.encounter_intervals else 0.0,
                "total_missed_encounters": sum(len(missed) for missed in self.missed_encounters.values()),
            })

        return base_stats


# ======================================================================================
# Enhanced Adaptive Learning Engine
# ======================================================================================

class AdaptiveLearningEngine:
    """
    Enhanced learning engine with:
    - Sensor-specific pressure/temperature baselines
    - Anomaly detection (unusual readings)
    - Protocol success rate tracking
    - Signal quality learning per location
    """

    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.signal_history: List[SignalCharacteristics] = []
        self.protocol_stats: Dict[str, Dict] = defaultdict(lambda: {
            'success_count': 0,
            'fail_count': 0,
            'avg_snr': 0,
            'avg_power': 0,
            'optimal_params': {}
        })

        # NEW: Sensor baselines for pressure/temperature
        self.sensor_baselines: Dict[str, Dict] = defaultdict(lambda: {
            'pressure_samples': deque(maxlen=100),
            'temp_samples': deque(maxlen=100),
            'pressure_mean': None,
            'pressure_std': None,
            'temp_mean': None,
            'temp_std': None,
        })

    def learn_from_signal(self, signal: Dict, decoded: bool, protocol: Optional[str] = None):
        """Learn from a signal detection attempt."""
        sig_char = SignalCharacteristics(
            frequency=signal.get('frequency', 0),
            power=signal.get('power', 0),
            snr=signal.get('snr', 0),
            modulation=signal.get('modulation', 'Unknown'),
            baud_rate=signal.get('baud_rate', 0),
            timestamp=time.time(),
            decoded=decoded,
            protocol=protocol,
            characteristics=signal.get('characteristics', {})
        )

        self.signal_history.append(sig_char)

        # Update protocol statistics
        if decoded and protocol:
            stats = self.protocol_stats[protocol]
            stats['success_count'] += 1

            n = stats['success_count']
            stats['avg_snr'] = (stats['avg_snr'] * (n - 1) + sig_char.snr) / n
            stats['avg_power'] = (stats['avg_power'] * (n - 1) + sig_char.power) / n

            # NEW: Update sensor baselines
            sensor_id = signal.get('tpms_id')
            if sensor_id:
                baseline = self.sensor_baselines[sensor_id]

                pressure = signal.get('pressure_psi')
                if pressure is not None and 0 < pressure < 100:  # Sanity check
                    baseline['pressure_samples'].append(float(pressure))
                    if len(baseline['pressure_samples']) >= 5:
                        baseline['pressure_mean'] = float(np.mean(baseline['pressure_samples']))
                        baseline['pressure_std'] = float(np.std(baseline['pressure_samples']))

                temp = signal.get('temperature_c')
                if temp is not None and -50 < temp < 150:  # Sanity check
                    baseline['temp_samples'].append(float(temp))
                    if len(baseline['temp_samples']) >= 5:
                        baseline['temp_mean'] = float(np.mean(baseline['temp_samples']))
                        baseline['temp_std'] = float(np.std(baseline['temp_samples']))

        elif protocol:
            self.protocol_stats[protocol]['fail_count'] += 1

        # Keep only recent history
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]

    def detect_anomaly(self, sensor_id: str, pressure: Optional[float] = None,
                      temp: Optional[float] = None, threshold_sigma: float = 3.0) -> Dict[str, Any]:
        """
        Detect anomalous readings for a sensor.

        Returns:
            {
                'pressure_anomaly': bool,
                'pressure_z_score': float,
                'temp_anomaly': bool,
                'temp_z_score': float,
                'confidence': 0-1 (based on sample size)
            }
        """
        baseline = self.sensor_baselines.get(sensor_id)
        if not baseline:
            return {'pressure_anomaly': False, 'temp_anomaly': False, 'confidence': 0.0}

        result = {
            'pressure_anomaly': False,
            'pressure_z_score': None,
            'temp_anomaly': False,
            'temp_z_score': None,
            'confidence': 0.0
        }

        # Pressure anomaly
        if pressure is not None and baseline['pressure_mean'] is not None:
            z_score = (pressure - baseline['pressure_mean']) / max(baseline['pressure_std'], 0.1)
            result['pressure_z_score'] = float(z_score)
            result['pressure_anomaly'] = abs(z_score) > threshold_sigma

        # Temperature anomaly
        if temp is not None and baseline['temp_mean'] is not None:
            z_score = (temp - baseline['temp_mean']) / max(baseline['temp_std'], 1.0)
            result['temp_z_score'] = float(z_score)
            result['temp_anomaly'] = abs(z_score) > threshold_sigma

        # Confidence based on sample size
        n_samples = len(baseline['pressure_samples'])
        result['confidence'] = float(np.clip(n_samples / 50.0, 0.0, 1.0))

        return result

    def get_optimal_scan_parameters(self, frequency: float) -> Dict:
        """Get optimal scanning parameters based on learned data."""
        best_protocol = None
        best_success_rate = 0

        for protocol, stats in self.protocol_stats.items():
            total = stats['success_count'] + stats['fail_count']
            if total > 0:
                success_rate = stats['success_count'] / total
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    best_protocol = protocol

        if best_protocol:
            stats = self.protocol_stats[best_protocol]
            return {
                'protocol': best_protocol,
                'expected_modulation': best_protocol.split('_')[1] if '_' in best_protocol else None,
                'expected_baud_rate': stats['optimal_params'].get('baud_rate'),
                'threshold': stats['avg_power'] - 10,
                'success_rate': best_success_rate
            }

        return {}

    def get_protocol_statistics(self) -> Dict[str, Dict]:
        """Get statistics for all protocols."""
        return dict(self.protocol_stats)

    def get_learning_summary(self) -> Dict:
        """Get summary of learning progress."""
        total_signals = len(self.signal_history)
        decoded_signals = sum(1 for s in self.signal_history if s.decoded)

        return {
            'total_signals': total_signals,
            'decoded_signals': decoded_signals,
            'decode_rate': decoded_signals / total_signals if total_signals > 0 else 0,
            'protocols_learned': len(self.protocol_stats),
            'sensors_with_baselines': len(self.sensor_baselines),
            'best_protocol': max(
                self.protocol_stats.items(),
                key=lambda x: x[1]['success_count'],
                default=(None, None)
            )[0] if self.protocol_stats else None
        }


# ======================================================================================
# Factory Functions
# ======================================================================================

def create_learning_engine() -> AdaptiveLearningEngine:
    """Factory function to create learning engine."""
    return AdaptiveLearningEngine(learning_rate=0.1)

def create_clustering_engine(db) -> VehicleClusteringEngine:
    """Factory function to create enhanced clustering engine."""
    return VehicleClusteringEngine(
        db=db,
        base_mate_window_s=60.0,
        relaxed_mate_window_s=120.0,
        observation_window_s=60.0,
        suspect_score=2.0,
        link_score=4.0,
        decay_halflife_s=300.0,
        max_vehicle_size=6,
        min_vehicle_size=3,
        enable_prediction=True,
        location_cell_deg=0.001,  # ~100m cells
        min_pattern_observations=5,
    )


# ======================================================================================
# Online Pattern Learning (from your original code, kept as-is)
# ======================================================================================

@dataclass
class _EWMA:
    alpha: float
    mean: float = 0.0
    var: float = 0.0
    n: int = 0
    last_ts: float = 0.0

    def update(self, x: float, ts: float):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return
        x = float(x)
        if self.n == 0:
            self.mean = x
            self.var = 0.0
            self.n = 1
            self.last_ts = float(ts)
            return

        delta = x - self.mean
        self.mean += self.alpha * delta
        self.var = (1 - self.alpha) * (self.var + self.alpha * delta * delta)
        self.n += 1
        self.last_ts = float(ts)

    def std(self) -> float:
        return float(np.sqrt(max(self.var, 1e-9)))

class OnlinePatternLearner:
    """Lightweight online learner for relationships across time/location/sensor/value."""

    def __init__(self, alpha_sensor: float = 0.03, alpha_context: float = 0.05,
                 cell_deg: float = 0.01):
        self.alpha_sensor = float(alpha_sensor)
        self.alpha_context = float(alpha_context)
        self.cell_deg = float(cell_deg)

        self.sensor_stats: Dict[str, Dict[str, _EWMA]] = {}
        self.sensor_hour_stats: Dict[Tuple[str, int], Dict[str, _EWMA]] = {}
        self.cell_stats: Dict[Tuple[int, int], Dict[str, _EWMA]] = {}

        self.last_rowid: int = 0
        self.total_updates: int = 0

    def _cell_key(self, lat: Optional[float], lon: Optional[float]) -> Optional[Tuple[int, int]]:
        if lat is None or lon is None:
            return None
        try:
            lat_f = float(lat); lon_f = float(lon)
        except Exception:
            return None
        if np.isnan(lat_f) or np.isnan(lon_f):
            return None
        return (int(np.floor(lat_f / self.cell_deg)), int(np.floor(lon_f / self.cell_deg)))

    def update_rows(self, rows: List[Dict[str, Any]]):
        """Update model from a list of DB rows."""
        for r in rows:
            try:
                rowid = int(r.get("rowid", 0))
                sensor = str(r.get("tpms_id", "") or "")
                ts = float(r.get("timestamp", 0.0) or 0.0)
                p = r.get("pressure_psi", None)
                t = r.get("temperature_c", None)
                lat = r.get("latitude", None)
                lon = r.get("longitude", None)
            except Exception:
                continue

            if not sensor:
                continue

            self.last_rowid = max(self.last_rowid, rowid)
            self.total_updates += 1

            # per-sensor stats
            if sensor not in self.sensor_stats:
                self.sensor_stats[sensor] = {
                    "p": _EWMA(self.alpha_sensor),
                    "t": _EWMA(self.alpha_sensor),
                }
            self.sensor_stats[sensor]["p"].update(p, ts)
            self.sensor_stats[sensor]["t"].update(t, ts)

            # per-sensor-hour stats
            try:
                hour = int(datetime.fromtimestamp(ts).hour)
            except Exception:
                hour = 0
            hk = (sensor, hour)
            if hk not in self.sensor_hour_stats:
                self.sensor_hour_stats[hk] = {
                    "p": _EWMA(self.alpha_context),
                    "t": _EWMA(self.alpha_context),
                }
            self.sensor_hour_stats[hk]["p"].update(p, ts)
            self.sensor_hour_stats[hk]["t"].update(t, ts)

            # per-location-cell stats
            ck = self._cell_key(lat, lon)
            if ck is not None:
                if ck not in self.cell_stats:
                    self.cell_stats[ck] = {
                        "p": _EWMA(self.alpha_context),
                        "t": _EWMA(self.alpha_context),
                    }
                self.cell_stats[ck]["p"].update(p, ts)
                self.cell_stats[ck]["t"].update(t, ts)

    def predict(self, sensor_id: str, ts: Optional[float] = None,
                lat: Optional[float] = None, lon: Optional[float] = None) -> Dict[str, Any]:
        """Return predicted (pressure,temp) + confidence numbers."""
        import time as _time
        ts = float(ts if ts is not None else _time.time())

        base = self.sensor_stats.get(sensor_id)
        if not base or base["p"].n == 0:
            return {"ok": False, "reason": "No learned history for this sensor yet."}

        hour = int(datetime.fromtimestamp(ts).hour)
        hk = (sensor_id, hour)
        hour_stats = self.sensor_hour_stats.get(hk)

        ck = self._cell_key(lat, lon)
        cell_stats = self.cell_stats.get(ck) if ck is not None else None

        def blend(key: str):
            parts = []
            weights = []

            parts.append(base[key].mean); weights.append(0.65)

            if hour_stats and hour_stats[key].n >= 3:
                parts.append(hour_stats[key].mean); weights.append(0.25)

            if cell_stats and cell_stats[key].n >= 5:
                parts.append(cell_stats[key].mean); weights.append(0.10)

            wsum = float(np.sum(weights))
            est = float(np.dot(parts, weights) / wsum)
            std = base[key].std()
            conf = float(np.clip((math.log10(base[key].n + 1) / (1 + std)), 0.0, 1.0))
            return est, std, conf

        p_est, p_std, p_conf = blend("p")
        t_est, t_std, t_conf = blend("t")

        return {
            "ok": True,
            "pressure_pred": p_est,
            "pressure_std": p_std,
            "pressure_conf": p_conf,
            "temp_pred": t_est,
            "temp_std": t_std,
                        "temp_conf": t_conf,
            "hour": hour,
            "cell": ck,
        }

    def anomaly_scores(self, sensor_id: str, pressure: Optional[float], temp: Optional[float],
                       ts: Optional[float] = None) -> Dict[str, Any]:
        """Return z-scores vs learned baseline."""
        base = self.sensor_stats.get(sensor_id)
        if not base or base["p"].n < 5:
            return {"ok": False, "reason": "Not enough data for anomaly scoring."}

        z_p = None
        z_t = None
        try:
            if pressure is not None and not (isinstance(pressure, float) and np.isnan(pressure)):
                z_p = (float(pressure) - base["p"].mean) / base["p"].std()
        except Exception:
            pass

        try:
            if temp is not None and not (isinstance(temp, float) and np.isnan(temp)):
                z_t = (float(temp) - base["t"].mean) / base["t"].std()
        except Exception:
            pass

        return {"ok": True, "z_pressure": z_p, "z_temp": z_t, "n": base["p"].n}

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about learned patterns."""
        return {
            "total_sensors": len(self.sensor_stats),
            "total_sensor_hour_patterns": len(self.sensor_hour_stats),
            "total_location_cells": len(self.cell_stats),
            "total_updates": self.total_updates,
            "last_rowid": self.last_rowid,
        }


# ======================================================================================
# Usage Examples and Helper Functions
# ======================================================================================

def example_usage_predictive_analytics(clustering_engine: VehicleClusteringEngine, vehicle_id: int):
    """
    Example of how to use the predictive analytics features.
    """
    print(f"\n=== Predictive Analytics for Vehicle {vehicle_id} ===\n")

    # Predict next encounter
    prediction = clustering_engine.predict_next_encounter(vehicle_id)
    if prediction['predicted_time']:
        pred_dt = datetime.fromtimestamp(prediction['predicted_time'])
        print(f"📅 Next encounter predicted: {pred_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Confidence: {prediction['confidence']:.2%}")
        print(f"   Average interval: {prediction['interval_mean']/60:.1f} minutes")
        print(f"   Likely hours: {prediction['likely_hours']}")

    # Check for missed encounters
    missed = clustering_engine.detect_missed_encounters(vehicle_id)
    if missed['missed']:
        print(f"\n⚠️  Possibly missed encounter!")
        print(f"   Expected at: {datetime.fromtimestamp(missed['expected_time']).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Overdue by: {missed['overdue_by']/60:.1f} minutes")
        print(f"   Confidence: {missed['confidence']:.2%}")

    # Check sensor completeness
    observed = clustering_engine._observed_ids(time.time())
    completeness = clustering_engine.get_incomplete_sensor_sets(vehicle_id, observed)
    print(f"\n🔍 Sensor Set Analysis:")
    print(f"   Complete: {completeness['complete']}")
    print(f"   Completeness: {completeness['completeness_ratio']:.2%}")
    if completeness['missing_sensors']:
        print(f"   Missing sensors: {completeness['missing_sensors']}")
    if completeness['unexpected_sensors']:
        print(f"   Unexpected sensors: {completeness['unexpected_sensors']}")


def example_usage_orphan_sensor(clustering_engine: VehicleClusteringEngine, orphan_id: str,
                                location: Optional[Tuple[float, float]] = None):
    """
    Example of how to handle an orphan sensor (detected but not associated).
    """
    print(f"\n=== Analyzing Orphan Sensor {orphan_id} ===\n")

    suggestions = clustering_engine.suggest_sensor_associations(
        orphan_id,
        location=location
    )

    if suggestions:
        print(f"Found {len(suggestions)} possible vehicle associations:\n")
        for i, sugg in enumerate(suggestions[:3], 1):  # Top 3
            print(f"{i}. Vehicle {sugg['vehicle_id']}")
            print(f"   Confidence: {sugg['confidence']:.2%}")
            print(f"   Reasons: {', '.join(sugg['reasons'])}")
            print(f"   Score breakdown: temporal={sugg['score_breakdown']['temporal']:.2f}, "
                  f"location={sugg['score_breakdown']['location']:.2f}, "
                  f"incomplete={sugg['score_breakdown']['incomplete']:.2f}\n")
    else:
        print("No strong associations found. This might be a new vehicle.")


def example_usage_anomaly_detection(learning_engine: AdaptiveLearningEngine,
                                   sensor_id: str, pressure: float, temp: float):
    """
    Example of how to use anomaly detection.
    """
    print(f"\n=== Anomaly Detection for {sensor_id} ===\n")

    anomaly = learning_engine.detect_anomaly(sensor_id, pressure, temp)

    print(f"Pressure: {pressure:.1f} PSI")
    if anomaly['pressure_z_score'] is not None:
        print(f"  Z-score: {anomaly['pressure_z_score']:.2f}")
        print(f"  Anomaly: {'⚠️  YES' if anomaly['pressure_anomaly'] else '✓ Normal'}")

    print(f"\nTemperature: {temp:.1f}°C")
    if anomaly['temp_z_score'] is not None:
        print(f"  Z-score: {anomaly['temp_z_score']:.2f}")
        print(f"  Anomaly: {'⚠️  YES' if anomaly['temp_anomaly'] else '✓ Normal'}")

    print(f"\nConfidence: {anomaly['confidence']:.2%}")


def example_usage_pattern_learner(pattern_learner: OnlinePatternLearner,
                                 sensor_id: str, lat: float, lon: float):
    """
    Example of how to use the OnlinePatternLearner for predictions.
    """
    print(f"\n=== Pattern-Based Prediction for {sensor_id} ===\n")

    prediction = pattern_learner.predict(sensor_id, lat=lat, lon=lon)

    if prediction['ok']:
        print(f"Predicted Pressure: {prediction['pressure_pred']:.1f} ± {prediction['pressure_std']:.1f} PSI")
        print(f"  Confidence: {prediction['pressure_conf']:.2%}")

        print(f"\nPredicted Temperature: {prediction['temp_pred']:.1f} ± {prediction['temp_std']:.1f}°C")
        print(f"  Confidence: {prediction['temp_conf']:.2%}")

        print(f"\nContext: Hour {prediction['hour']}, Cell {prediction['cell']}")
    else:
        print(f"Cannot predict: {prediction['reason']}")


def create_pattern_learner() -> OnlinePatternLearner:
    """Factory function to create pattern learner."""
    return OnlinePatternLearner(
        alpha_sensor=0.03,    # Slow adaptation for baseline
        alpha_context=0.05,   # Slightly faster for context
        cell_deg=0.01         # ~1km cells for location patterns
    )


# ======================================================================================
# Integration Helper: Periodic Tasks
# ======================================================================================

class MLEngineScheduler:
    """
    Helper class to run periodic ML tasks (pattern updates, missed encounter checks, etc.)
    """

    def __init__(self, clustering_engine: VehicleClusteringEngine,
                 pattern_learner: OnlinePatternLearner,
                 db):
        self.clustering_engine = clustering_engine
        self.pattern_learner = pattern_learner
        self.db = db

        self.last_pattern_update = time.time()
        self.last_missed_check = time.time()

    def run_periodic_tasks(self, force: bool = False):
        """
        Run periodic maintenance tasks.
        Call this from your main loop every few seconds.
        """
        now = time.time()

        # Update pattern learner from DB (every 60 seconds)
        if force or (now - self.last_pattern_update) > 60:
            self._update_patterns_from_db()
            self.last_pattern_update = now

        # Check for missed encounters (every 5 minutes)
        if force or (now - self.last_missed_check) > 300:
            self._check_missed_encounters()
            self.last_missed_check = now

    def _update_patterns_from_db(self):
        """Update pattern learner with new DB rows."""
        try:
            # Assuming your DB has a method like this
            new_rows = self.db.get_signals_since_rowid(self.pattern_learner.last_rowid)
            if new_rows:
                self.pattern_learner.update_rows(new_rows)
                print(f"📊 Updated patterns with {len(new_rows)} new signals", flush=True)
        except Exception as e:
            print(f"⚠️  Error updating patterns: {e}", flush=True)

    def _check_missed_encounters(self):
        """Check all vehicles for missed encounters."""
        try:
            now = time.time()
            for vehicle_id in self.clustering_engine.vehicle_profiles.keys():
                missed = self.clustering_engine.detect_missed_encounters(vehicle_id, now)
                if missed['missed'] and missed['confidence'] > 0.5:
                    print(f"⚠️  Vehicle {vehicle_id} possibly missed (overdue by {missed['overdue_by']/60:.1f} min)",
                          flush=True)
        except Exception as e:
            print(f"⚠️  Error checking missed encounters: {e}", flush=True)


# ======================================================================================
# Main Integration Point
# ======================================================================================

def create_ml_engines(db):
    """
    Convenience function to create all ML engines at once.

    Returns:
        Tuple of (clustering_engine, learning_engine, pattern_learner, scheduler)
    """
    clustering_engine = create_clustering_engine(db)
    learning_engine = create_learning_engine()
    pattern_learner = create_pattern_learner()
    scheduler = MLEngineScheduler(clustering_engine, pattern_learner, db)

    return clustering_engine, learning_engine, pattern_learner, scheduler


# ======================================================================================
# Enhanced Statistics and Reporting
# ======================================================================================

def get_comprehensive_statistics(clustering_engine: VehicleClusteringEngine,
                                 learning_engine: AdaptiveLearningEngine,
                                 pattern_learner: OnlinePatternLearner) -> Dict[str, Any]:
    """
    Get comprehensive statistics from all ML components.
    """
    stats = {
        'clustering': clustering_engine.get_statistics(),
        'learning': learning_engine.get_learning_summary(),
        'patterns': pattern_learner.get_statistics(),
        'timestamp': time.time(),
    }

    # Add derived metrics
    if clustering_engine.enable_prediction:
        vehicles_with_predictions = sum(
            1 for v_id in clustering_engine.vehicle_profiles.keys()
            if len(clustering_engine.encounter_intervals.get(v_id, [])) >= clustering_engine.min_pattern_observations
        )
        stats['predictive'] = {
            'vehicles_predictable': vehicles_with_predictions,
            'prediction_coverage': vehicles_with_predictions / max(1, len(clustering_engine.vehicle_profiles)),
        }

    return stats


def print_ml_summary(clustering_engine: VehicleClusteringEngine,
                    learning_engine: AdaptiveLearningEngine,
                    pattern_learner: OnlinePatternLearner):
    """
    Print a human-readable summary of ML engine status.
    """
    stats = get_comprehensive_statistics(clustering_engine, learning_engine, pattern_learner)

    print("\n" + "="*60)
    print("ML ENGINE SUMMARY")
    print("="*60)

    print("\n📊 Clustering Engine:")
    print(f"  Vehicles tracked: {stats['clustering']['num_vehicle_profiles']}")
    print(f"  Pair associations: {stats['clustering']['pair_edges']}")
    print(f"  Strong components: {stats['clustering']['num_components_strong']}")

    if 'predictive' in stats:
        print(f"\n🔮 Predictive Analytics:")
        print(f"  Vehicles with patterns: {stats['clustering']['vehicles_with_patterns']}")
        print(f"  Predictable vehicles: {stats['predictive']['vehicles_predictable']}")
        print(f"  Prediction coverage: {stats['predictive']['prediction_coverage']:.1%}")
        print(f"  Total missed encounters: {stats['clustering']['total_missed_encounters']}")

    print(f"\n🎓 Learning Engine:")
    print(f"  Total signals: {stats['learning']['total_signals']}")
    print(f"  Decode rate: {stats['learning']['decode_rate']:.1%}")
    print(f"  Protocols learned: {stats['learning']['protocols_learned']}")
    print(f"  Sensors with baselines: {stats['learning']['sensors_with_baselines']}")
    print(f"  Best protocol: {stats['learning']['best_protocol']}")

    print(f"\n📈 Pattern Learner:")
    print(f"  Sensors tracked: {stats['patterns']['total_sensors']}")
    print(f"  Temporal patterns: {stats['patterns']['total_sensor_hour_patterns']}")
    print(f"  Location cells: {stats['patterns']['total_location_cells']}")
    print(f"  Total updates: {stats['patterns']['total_updates']}")

    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    # This allows the module to be tested standalone
    print("TPMS ML Engine Module")
    print("This module provides enhanced machine learning capabilities for TPMS signal analysis.")
    print("\nKey features:")
    print("  • Vehicle clustering with temporal/spatial pattern learning")
    print("  • Predictive encounter modeling")
    print("  • Missed encounter detection")
    print("  • Sensor association suggestions")
    print("  • Anomaly detection for pressure/temperature")
    print("  • Online pattern learning with context awareness")
