"""
Machine Learning Engine for TPMS Signal Analysis
Modern Python (3.10+) compatible
"""
import numpy as np  
import time
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import List, Dict, Optional, Tuple, Set, Any
import math
from datetime import datetime

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

class VehicleClusteringEngine:
    """
    Mate-graph vehicle clustering:

    - Any two sensors detected within base_mate_window_s are "potential mates" and get a score bump.
    - If a pair becomes "suspected" (score >= suspect_score), the mate window relaxes to relaxed_mate_window_s.
    - Vehicles are inferred as connected components over strong mate edges.

    This matches the mental model:
      "I think these sensors belong together; later sightings confirm it (and the confirmation window is wider)."
    """

    def __init__(
        self,
        db,
        # --- temporal mate logic ---
        base_mate_window_s: float = 60.0,        # strict "mate" window
        relaxed_mate_window_s: float = 120.0,     # used once pair is suspected
        observation_window_s: float = 10.0,      # which sensors count as "currently together"
        min_same_sensor_gap_s: float = 0.005,    # debounce repeats from same sensor

        # --- scoring / thresholds ---
        suspect_score: float = 120.0,             # when we start relaxing the window for that pair
        link_score: float = 2.0,                # edge threshold for building vehicle components
        stable_pair_ratio: float = 0.60,        # % of pairs that must be strong to create a *new* vehicle

        # --- forgetting ---
        decay_halflife_s: float = 180.0,        # decay old associations

        # --- safety gates ---
        max_vehicle_size: int = 8,
        min_vehicle_size: int = 2,

        # --- DB cache ---
        refresh_vehicle_cache_s: float = 300.0,
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

        # (a,b) -> score (decayed)
        self.pair_scores = defaultdict(float)
        self._last_decay_ts = time.time()

        # recent detections for mate matching: (tpms_id, ts)
        # kept only for the last relaxed_mate_window_s window
        self._recent_detections = deque()

        # debounce repeats per sensor to avoid score inflation
        self._last_seen_sensor_ts = {}

        # vehicle cache for identify_vehicle()
        self._vehicle_cache = []
        self._vehicle_cache_ts = 0.0
        self._vehicle_cache_ttl = float(refresh_vehicle_cache_s)

        # Debuggable outputs
        self.recent_windows = deque(maxlen=300)  # (ts, frozenset(ids_observed))
        self.vehicle_profiles: Dict[int, Dict] = {}
        self.clusters: Dict[int, List[str]] = {}

    # -----------------------------
    # Utilities
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
        """
        Map raw score -> [0,1) likelihood.
        This is just a convenience scalar for UI/debug.
        """
        s = self.pair_scores.get(self._pair_key(a, b), 0.0)
        # saturating curve: 0->0, 4->~0.63, 8->~0.86, ...
        return 1.0 - math.exp(-s / max(1e-6, self.link_score))

    # -----------------------------
    # Ingestion: "mates within 1s"
    # -----------------------------

    def _ingest_detection(self, sid: str, ts: float):
        """
        For each new detection:
          - compare against other detections within relaxed_mate_window_s
          - if dt <= effective_window(pair), bump the pair score
        """
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

        # compare to recent detections (iterate newest->oldest)
        # deque is time-ordered as we append
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
        """
        Ingest all signals in time order.
        Returns a robust window timestamp (max ts).
        """
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
    # Graph building / grouping
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
        """
        Stable group heuristic:
          require at least stable_pair_ratio of all pairs to be >= link_score
        """
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
        """
        If we already suspect mates, expand the set by one-hop mate edges.
        This models: "I saw a car, I think I know its sensor set, but I can't verify until it moves."
        """
        if not ids:
            return ids

        adj_suspect = self._adjacency(self.suspect_score)

        expanded = set(ids)
        changed = True
        # iterative expansion (cap depth to avoid explosions)
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
    # Public API
    # -----------------------------

    def process_signals(self, signal_buffer: List[Dict]) -> List[int]:
        """
        Called with a short list of decoded signals (your app already buffers).
        We:
          1) ingest detections -> update mate likelihoods
          2) form candidates from strong components + current observation expanded with suspected mates
          3) identify/upsert vehicles
        """
        if not signal_buffer:
            return []

        # (1) update mate graph
        window_ts = self._ingest_signal_buffer(signal_buffer)

        # (2) candidates
        strong_components = self._components(self.link_score)

        observed = self._observed_ids(window_ts)
        observed_expanded = self._expand_with_suspected_mates(observed)

        # record for debugging
        if observed:
            self.recent_windows.append((window_ts, frozenset(observed)))

        candidates: List[List[str]] = []
        candidates.extend(strong_components)
        if observed_expanded:
            candidates.append(observed_expanded)

        # de-dupe candidates by set
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

        # optional location passthrough
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
                # only create new vehicles from stable groups
                if not stable:
                    continue
                vehicle_id = self.db.upsert_vehicle(ids, window_ts, location=location)
                print(f"🚗 New vehicle (stable mates): ID={vehicle_id}, sensors={ids}", flush=True)
            else:
                self.db.upsert_vehicle(ids, window_ts, location=location)
                # print(f"🚗 Known vehicle: ID={vehicle_id}, sensors={ids}", flush=True)

            vehicle_ids.append(vehicle_id)
            self.update_vehicle_profile(vehicle_id, ids, window_ts, signal_buffer)
            self.clusters[idx] = ids

        return sorted(set(vehicle_ids))

    # -----------------------------
    # Matching (unchanged-ish)
    # -----------------------------

    def _refresh_vehicle_cache(self):
        now = time.time()
        if now - self._vehicle_cache_ts > self._vehicle_cache_ttl:
            self._vehicle_cache = self.db.get_all_vehicles(min_encounters=1)
            self._vehicle_cache_ts = now

    def identify_vehicle(self, sensor_ids: List[str]) -> Optional[int]:
        """
        Match sensors to an existing vehicle using Jaccard score.
        """
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
    # Profiles / stats (keep your old behavior)
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
        return {
            "pair_edges": len(self.pair_scores),
            "num_components_strong": len(self._components(self.link_score)),
            "num_vehicle_profiles": len(self.vehicle_profiles),
            "clusters_last_run": {k: len(v) for k, v in self.clusters.items()},
        }
    
class AdaptiveLearningEngine:
    """
    Adaptive learning engine for TPMS signal detection
    Learns optimal parameters from successful decodes
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

    def learn_from_signal(self, signal: Dict, decoded: bool, protocol: Optional[str] = None):
        """
        Learn from a signal detection attempt

        Args:
            signal: Signal characteristics
            decoded: Whether the signal was successfully decoded
            protocol: Protocol name if decoded
        """
        # Create signal characteristics
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

            # Update running averages
            n = stats['success_count']
            stats['avg_snr'] = (stats['avg_snr'] * (n - 1) + sig_char.snr) / n
            stats['avg_power'] = (stats['avg_power'] * (n - 1) + sig_char.power) / n
        elif protocol:
            self.protocol_stats[protocol]['fail_count'] += 1

        # Keep only recent history
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]

    def get_optimal_scan_parameters(self, frequency: float) -> Dict:
        """
        Get optimal scanning parameters based on learned data

        Args:
            frequency: Target frequency

        Returns:
            Dictionary of optimal parameters
        """
        # Find most successful protocol at this frequency
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
                'threshold': stats['avg_power'] - 10,  # 10 dB below average
                'success_rate': best_success_rate
            }

        return {}

    def get_protocol_statistics(self) -> Dict[str, Dict]:
        """Get statistics for all protocols"""
        return dict(self.protocol_stats)

    def get_learning_summary(self) -> Dict:
        """Get summary of learning progress"""
        total_signals = len(self.signal_history)
        decoded_signals = sum(1 for s in self.signal_history if s.decoded)

        return {
            'total_signals': total_signals,
            'decoded_signals': decoded_signals,
            'decode_rate': decoded_signals / total_signals if total_signals > 0 else 0,
            'protocols_learned': len(self.protocol_stats),
            'best_protocol': max(
                self.protocol_stats.items(),
                key=lambda x: x[1]['success_count'],
                default=(None, None)
            )[0] if self.protocol_stats else None
        }

def create_learning_engine() -> AdaptiveLearningEngine:
    """Factory function to create learning engine"""
    return AdaptiveLearningEngine(learning_rate=0.1)

def create_clustering_engine(db) -> VehicleClusteringEngine:
    return VehicleClusteringEngine(
        db=db,
        base_mate_window_s=60.0,        # ✅ Changed from 1.0
        relaxed_mate_window_s=120.0,    # ✅ Changed from 3.0
        observation_window_s=60.0,      # ✅ Changed from 1.0
        suspect_score=2.0,
        link_score=4.0,
        decay_halflife_s=300.0,         # ✅ Increased from 180
        max_vehicle_size=6,
        min_vehicle_size=3,             # ✅ Requires 3+ sensors (allows incomplete sets)
    )




# ======================================================================================
# Online Pattern Learning (mixed real-time + batch-friendly)
# ======================================================================================

from dataclasses import dataclass
import numpy as np
import math
from typing import DefaultDict

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

        # Exponential moving stats
        delta = x - self.mean
        self.mean += self.alpha * delta
        # EW variance update (approx; good enough for anomaly scoring)
        self.var = (1 - self.alpha) * (self.var + self.alpha * delta * delta)
        self.n += 1
        self.last_ts = float(ts)

    def std(self) -> float:
        return float(np.sqrt(max(self.var, 1e-9)))

class OnlinePatternLearner:
    """Lightweight online learner for relationships across time/location/sensor/value.

    What it learns (incrementally, per incoming signal):
      - Per-sensor EWMA mean/std for pressure & temperature
      - Per-sensor *hour-of-day* EWMA mean/std (captures commute / daily cycles)
      - Per-location-cell EWMA mean/std (captures 'hotspots' like parking lots)

    What it can do:
      - Predict expected pressure/temp for a sensor at a given time/location
      - Compute anomaly z-scores ("this reading is weird for this sensor/time/place")
    """

    def __init__(self, alpha_sensor: float = 0.03, alpha_context: float = 0.05,
                 cell_deg: float = 0.01):
        self.alpha_sensor = float(alpha_sensor)
        self.alpha_context = float(alpha_context)
        self.cell_deg = float(cell_deg)

        # sensor_id -> {'p': _EWMA, 't': _EWMA}
        self.sensor_stats: Dict[str, Dict[str, _EWMA]] = {}

        # (sensor_id, hour) -> {'p': _EWMA, 't': _EWMA}
        self.sensor_hour_stats: Dict[Tuple[str, int], Dict[str, _EWMA]] = {}

        # (cell_lat, cell_lon) -> {'p': _EWMA, 't': _EWMA}
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
        """Update model from a list of DB rows produced by get_signals_since_rowid."""
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

            # --- per-sensor stats ---
            if sensor not in self.sensor_stats:
                self.sensor_stats[sensor] = {
                    "p": _EWMA(self.alpha_sensor),
                    "t": _EWMA(self.alpha_sensor),
                }
            self.sensor_stats[sensor]["p"].update(p, ts)
            self.sensor_stats[sensor]["t"].update(t, ts)

            # --- per-sensor-hour stats ---
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

            # --- per-location-cell stats ---
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
        """Return predicted (pressure,temp) + simple confidence numbers."""
        import time as _time
        ts = float(ts if ts is not None else _time.time())

        # Base: per-sensor
        base = self.sensor_stats.get(sensor_id)
        if not base or base["p"].n == 0:
            return {"ok": False, "reason": "No learned history for this sensor yet."}

        # Context: hour-of-day
        hour = int(datetime.fromtimestamp(ts).hour)
        hk = (sensor_id, hour)
        hour_stats = self.sensor_hour_stats.get(hk)

        # Context: location cell
        ck = self._cell_key(lat, lon)
        cell_stats = self.cell_stats.get(ck) if ck is not None else None

        # Weighted blending
        def blend(key: str):
            parts = []
            weights = []

            # sensor baseline
            parts.append(base[key].mean); weights.append(0.65)

            if hour_stats and hour_stats[key].n >= 3:
                parts.append(hour_stats[key].mean); weights.append(0.25)

            if cell_stats and cell_stats[key].n >= 5:
                parts.append(cell_stats[key].mean); weights.append(0.10)

            wsum = float(np.sum(weights))
            est = float(np.dot(parts, weights) / wsum)
            # confidence: more samples + lower std is higher confidence (heuristic)
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
