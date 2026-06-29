"""
tpms_vehicle_engine.py  (v2)
============================
Vehicle clustering and learning engine for TPMS data.

v2 improvements:
  - PSI histogram approach: instead of sliding window, build PSI histogram
    per burst and find natural "clusters of 4" using a density-based method
  - RSSI-assisted grouping: sensors with very similar RSSI are likely co-located
    (same vehicle lane/direction)
  - Temporal micro-grouping: within a burst, sub-group by 10s micro-windows
    to find which sensors arrived together (same car)
  - Richer confidence scoring
"""

import sqlite3, json, math, time, hashlib, struct
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import statistics


@dataclass
class Signal:
    timestamp: float
    tpms_id: str
    protocol: str
    pressure_psi: Optional[float]
    temperature_c: Optional[float]
    battery_low: bool
    signal_strength: Optional[float]
    snr: Optional[float]
    frequency: Optional[float]
    latitude: Optional[float]
    longitude: Optional[float]
    confidence: Optional[float] = None

    @property
    def has_pressure(self) -> bool:
        return self.pressure_psi is not None and 10.0 <= self.pressure_psi <= 100.0


@dataclass
class BurstEvent:
    burst_id: str
    start_ts: float
    end_ts: float
    center_lat: Optional[float]
    center_lon: Optional[float]
    signals: List[Signal]

    @property
    def duration(self) -> float:
        return self.end_ts - self.start_ts

    @property
    def signal_count(self) -> int:
        return len(self.signals)

    @property
    def pressured_signals(self) -> List[Signal]:
        return [s for s in self.signals if s.has_pressure]


@dataclass
class VehicleCluster:
    cluster_id: str
    sensor_ids: List[str]
    psi_values: List[float]
    psi_fingerprint: Tuple[float, ...]
    protocol: str
    burst_id: str
    timestamp: float
    latitude: Optional[float]
    longitude: Optional[float]
    confidence: float
    method: str = "unknown"
    notes: str = ""


class BurstDetector:
    GAP_THRESHOLD = 90.0
    GPS_THRESHOLD = 0.008
    MIN_BURST_SIZE = 4

    def detect_bursts(self, signals: List[Signal]) -> List[BurstEvent]:
        if not signals:
            return []
        signals = sorted(signals, key=lambda s: s.timestamp)
        bursts, current = [], [signals[0]]

        for sig in signals[1:]:
            prev = current[-1]
            time_gap = sig.timestamp - prev.timestamp
            gps_gap = self._gps_dist(sig, prev)

            if time_gap > self.GAP_THRESHOLD or gps_gap > self.GPS_THRESHOLD:
                if len(current) >= self.MIN_BURST_SIZE:
                    bursts.append(self._make_burst(current))
                current = [sig]
            else:
                current.append(sig)

        if len(current) >= self.MIN_BURST_SIZE:
            bursts.append(self._make_burst(current))
        return bursts

    def _gps_dist(self, a, b) -> float:
        if a.latitude is None or b.latitude is None:
            return 0.0
        return math.sqrt((a.latitude - b.latitude)**2 + (a.longitude - b.longitude)**2)

    def _make_burst(self, signals) -> BurstEvent:
        lats = [s.latitude for s in signals if s.latitude is not None]
        lons = [s.longitude for s in signals if s.longitude is not None]
        clat = sum(lats)/len(lats) if lats else None
        clon = sum(lons)/len(lons) if lons else None
        key = f"{signals[0].timestamp:.0f}_{clat or 0:.4f}_{clon or 0:.4f}"
        bid = hashlib.md5(key.encode()).hexdigest()[:12]
        return BurstEvent(bid, signals[0].timestamp, signals[-1].timestamp, clat, clon, signals)


class WithinBurstClusterer:
    """
    Multi-strategy clusterer:
      1. MICRO-WINDOW: sub-group burst into 10s windows, each window may contain
         a single passing vehicle. If a window has 3-6 sensors → label as vehicle.
      2. RSSI-GROUPING: cluster by RSSI similarity (same vehicle = same distance)
      3. PSI-QUARTETS: PSI histogram approach to find natural groups of 4
    """

    PSI_ROUND = 2.5
    MIN_PSI = 12.0
    MAX_PSI = 100.0
    PSI_TOLERANCE = 4.0

    def cluster_burst(self, burst: BurstEvent) -> List[VehicleCluster]:
        valid = [s for s in burst.pressured_signals]
        if len(valid) < 3:
            return []

        clusters = []

        # Strategy 1: Micro-window (10s slices → single vehicle passing)
        micro = self._micro_window_cluster(valid, burst)
        clusters.extend(micro)
        used = set(sid for c in micro for sid in c.sensor_ids)

        # Strategy 2: RSSI grouping on remaining sensors
        remaining = [s for s in valid if s.tpms_id not in used
                     and s.signal_strength is not None]
        if remaining:
            rssi_clusters = self._rssi_cluster(remaining, burst)
            clusters.extend(rssi_clusters)
            used.update(sid for c in rssi_clusters for sid in c.sensor_ids)

        # Strategy 3: PSI quartets on whatever's left
        leftover = [s for s in valid if s.tpms_id not in used]
        if len(leftover) >= 4:
            psi_clusters = self._psi_quartet_cluster(leftover, burst)
            clusters.extend(psi_clusters)

        return clusters

    def _round_psi(self, p):
        return round(p / self.PSI_ROUND) * self.PSI_ROUND

    def _fingerprint(self, sensors):
        return tuple(sorted(self._round_psi(s.pressure_psi) for s in sensors))

    # --- Strategy 1: Micro-window ---
    def _micro_window_cluster(self, signals, burst) -> List[VehicleCluster]:
        """
        In a 10s window, 3-6 sensors with similar timing are likely one vehicle.
        Works best for highway captures where each car takes <3s to broadcast.
        """
        clusters = []
        used = set()
        n = len(signals)
        sorted_by_ts = sorted(signals, key=lambda s: s.timestamp)

        i = 0
        while i < n:
            t0 = sorted_by_ts[i].timestamp
            window = []
            j = i
            while j < n and sorted_by_ts[j].timestamp - t0 <= 10.0:
                if sorted_by_ts[j].tpms_id not in used:
                    window.append(sorted_by_ts[j])
                j += 1

            if 3 <= len(window) <= 6:
                # Validate PSI spread
                psi_vals = [s.pressure_psi for s in window]
                psi_range = max(psi_vals) - min(psi_vals)
                if psi_range >= 5.0:
                    fp = self._fingerprint(window)
                    cid = hashlib.md5((burst.burst_id + "mw" + str(i)).encode()).hexdigest()[:12]
                    conf = self._confidence(window, method="micro")
                    clusters.append(VehicleCluster(
                        cluster_id=cid, sensor_ids=[s.tpms_id for s in window],
                        psi_values=psi_vals, psi_fingerprint=fp,
                        protocol=window[0].protocol, burst_id=burst.burst_id,
                        timestamp=t0, latitude=burst.center_lat, longitude=burst.center_lon,
                        confidence=conf, method="micro_window",
                        notes=f"10s_window, psi_range={psi_range:.1f}, size={len(window)}"
                    ))
                    for s in window:
                        used.add(s.tpms_id)
                    i = j
                    continue
            i += 1

        return clusters

    # --- Strategy 2: RSSI grouping ---
    def _rssi_cluster(self, signals, burst) -> List[VehicleCluster]:
        """
        Group by RSSI similarity. Within a burst, sensors with RSSI within 2dB
        of each other are equidistant from the scanner → possibly same vehicle.
        """
        clusters = []
        used = set()

        by_rssi = sorted(signals, key=lambda s: s.signal_strength)
        i = 0
        while i < len(by_rssi):
            if by_rssi[i].tpms_id in used:
                i += 1
                continue
            anchor_rssi = by_rssi[i].signal_strength
            group = []
            for s in by_rssi[i:]:
                if s.tpms_id in used:
                    continue
                if abs(s.signal_strength - anchor_rssi) <= 2.0:
                    group.append(s)
                else:
                    break

            if 3 <= len(group) <= 6:
                psi_vals = [s.pressure_psi for s in group]
                psi_range = max(psi_vals) - min(psi_vals)
                if psi_range >= 5.0:
                    fp = self._fingerprint(group)
                    cid = hashlib.md5((burst.burst_id + "rssi" + str(i)).encode()).hexdigest()[:12]
                    conf = self._confidence(group, method="rssi")
                    clusters.append(VehicleCluster(
                        cluster_id=cid, sensor_ids=[s.tpms_id for s in group],
                        psi_values=psi_vals, psi_fingerprint=fp,
                        protocol=group[0].protocol, burst_id=burst.burst_id,
                        timestamp=group[0].timestamp,
                        latitude=burst.center_lat, longitude=burst.center_lon,
                        confidence=conf, method="rssi_group",
                        notes=f"rssi={anchor_rssi:.1f}±2dB, psi_range={psi_range:.1f}"
                    ))
                    for s in group:
                        used.add(s.tpms_id)
            i += 1
        return clusters

    # --- Strategy 3: PSI quartets ---
    def _psi_quartet_cluster(self, signals, burst) -> List[VehicleCluster]:
        """Sliding-window PSI sort — find groups of 4 with plausible tire pressure spread."""
        sorted_s = sorted(signals, key=lambda s: s.pressure_psi)
        clusters = []
        used = set()

        for window_size in [4, 5]:
            for i in range(len(sorted_s) - window_size + 1):
                window = sorted_s[i:i+window_size]
                if any(s.tpms_id in used for s in window):
                    continue
                psi_vals = [s.pressure_psi for s in window]
                psi_range = max(psi_vals) - min(psi_vals)
                if psi_range < 5.0 or psi_range > 50.0:
                    continue
                fp = self._fingerprint(window)
                if len(set(fp)) < 2:
                    continue
                cid = hashlib.md5((burst.burst_id + "psi" + str(i)).encode()).hexdigest()[:12]
                conf = self._confidence(window, method="psi")
                clusters.append(VehicleCluster(
                    cluster_id=cid, sensor_ids=[s.tpms_id for s in window],
                    psi_values=psi_vals, psi_fingerprint=fp,
                    protocol=window[0].protocol, burst_id=burst.burst_id,
                    timestamp=window[0].timestamp,
                    latitude=burst.center_lat, longitude=burst.center_lon,
                    confidence=conf, method="psi_quartet",
                    notes=f"psi_range={psi_range:.1f}, window={window_size}"
                ))
                for s in window:
                    used.add(s.tpms_id)
        return clusters

    def _confidence(self, sensors, method="unknown") -> float:
        psi_vals = [s.pressure_psi for s in sensors]
        psi_range = max(psi_vals) - min(psi_vals)

        # Pressure spread score: ideal ~15-30 PSI range
        if psi_range < 5:
            psi_score = 0.1
        elif psi_range <= 30:
            psi_score = 0.5 + (psi_range / 60)
        else:
            psi_score = max(0.3, 1.0 - (psi_range - 30) / 40)

        size_score = {3: 0.7, 4: 1.0, 5: 0.9, 6: 0.8}.get(len(sensors), 0.5)
        method_score = {"micro_window": 0.85, "rssi_group": 0.75, "psi_quartet": 0.65}.get(method, 0.5)

        # Temperature consistency bonus
        temps = [s.temperature_c for s in sensors if s.temperature_c is not None]
        temp_bonus = 0.0
        if len(temps) >= 2:
            temp_std = statistics.stdev(temps) if len(temps) > 1 else 0
            temp_bonus = 0.1 if temp_std < 10 else 0.0

        return min(1.0, psi_score * size_score * method_score + temp_bonus)


class VehicleProfileDB:
    PSI_MATCH_THRESHOLD = 5.0
    MIN_ENCOUNTERS_CONFIDENT = 3

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_schema()

    def _init_schema(self):
        con = sqlite3.connect(self.db_path)
        con.executescript("""
            CREATE TABLE IF NOT EXISTS vehicle_clusters (
                cluster_id      TEXT PRIMARY KEY,
                burst_id        TEXT NOT NULL,
                timestamp       REAL NOT NULL,
                sensor_ids      TEXT NOT NULL,
                psi_fingerprint TEXT NOT NULL,
                psi_values      TEXT NOT NULL,
                protocol        TEXT,
                latitude        REAL,
                longitude       REAL,
                confidence      REAL,
                vehicle_hash    TEXT,
                method          TEXT,
                notes           TEXT,
                created_at      REAL DEFAULT (unixepoch('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_vc_burst   ON vehicle_clusters(burst_id);
            CREATE INDEX IF NOT EXISTS idx_vc_ts      ON vehicle_clusters(timestamp);
            CREATE INDEX IF NOT EXISTS idx_vc_vh      ON vehicle_clusters(vehicle_hash);

            CREATE TABLE IF NOT EXISTS vehicle_profiles (
                vehicle_hash        TEXT PRIMARY KEY,
                nickname            TEXT,
                sensor_ids          TEXT NOT NULL,
                psi_fingerprint     TEXT NOT NULL,
                psi_tolerance       REAL DEFAULT 4.0,
                encounter_count     INTEGER DEFAULT 1,
                first_seen          REAL,
                last_seen           REAL,
                encounter_lats      TEXT DEFAULT '[]',
                encounter_lons      TEXT DEFAULT '[]',
                protocol            TEXT,
                avg_confidence      REAL DEFAULT 0.5,
                metadata            TEXT DEFAULT '{}',
                updated_at          REAL DEFAULT (unixepoch('now'))
            );

            CREATE TABLE IF NOT EXISTS burst_events (
                burst_id        TEXT PRIMARY KEY,
                start_ts        REAL NOT NULL,
                end_ts          REAL NOT NULL,
                duration_s      REAL,
                center_lat      REAL,
                center_lon      REAL,
                signal_count    INTEGER,
                pressured_count INTEGER,
                clusters_found  INTEGER DEFAULT 0,
                processed_at    REAL DEFAULT (unixepoch('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_be_ts ON burst_events(start_ts);

            CREATE TABLE IF NOT EXISTS processing_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                source_db   TEXT,
                since_ts    REAL,
                signals_processed INTEGER,
                bursts_detected   INTEGER,
                clusters_found    INTEGER,
                new_profiles      INTEGER,
                known_matches     INTEGER,
                elapsed_s   REAL,
                processed_at REAL DEFAULT (unixepoch('now'))
            );
        """)
        con.commit()
        con.close()

    def save_burst(self, burst: BurstEvent, clusters: List[VehicleCluster]):
        con = sqlite3.connect(self.db_path)
        try:
            con.execute("""
                INSERT OR REPLACE INTO burst_events
                (burst_id, start_ts, end_ts, duration_s, center_lat, center_lon,
                 signal_count, pressured_count, clusters_found)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (burst.burst_id, burst.start_ts, burst.end_ts, burst.duration,
                  burst.center_lat, burst.center_lon, burst.signal_count,
                  len(burst.pressured_signals), len(clusters)))

            for c in clusters:
                con.execute("""
                    INSERT OR IGNORE INTO vehicle_clusters
                    (cluster_id, burst_id, timestamp, sensor_ids, psi_fingerprint,
                     psi_values, protocol, latitude, longitude, confidence, method, notes)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """, (c.cluster_id, c.burst_id, c.timestamp,
                      json.dumps(c.sensor_ids), json.dumps(list(c.psi_fingerprint)),
                      json.dumps(c.psi_values), c.protocol,
                      c.latitude, c.longitude, c.confidence, c.method, c.notes))
            con.commit()
        finally:
            con.close()

    def match_and_learn(self, cluster: VehicleCluster) -> Tuple[str, bool]:
        con = sqlite3.connect(self.db_path)
        try:
            profiles = con.execute(
                "SELECT vehicle_hash, psi_fingerprint, psi_tolerance, sensor_ids, "
                "encounter_count, encounter_lats, encounter_lons, avg_confidence "
                "FROM vehicle_profiles WHERE protocol=? OR protocol IS NULL",
                [cluster.protocol]).fetchall()

            best_match, best_dist = None, float('inf')
            for row in profiles:
                fp = tuple(json.loads(row[1]))
                tol = row[2]
                dist = self._fp_dist(cluster.psi_fingerprint, fp)
                if dist < best_dist and dist < tol * max(1, len(fp)**0.5):
                    best_dist, best_match = dist, row

            if best_match:
                vh = best_match[0]
                new_sids = list(set(json.loads(best_match[3])) | set(cluster.sensor_ids))
                lats = (json.loads(best_match[5]) + ([cluster.latitude] if cluster.latitude else []))[-200:]
                lons = (json.loads(best_match[6]) + ([cluster.longitude] if cluster.longitude else []))[-200:]
                new_tol = min(10.0, best_match[2] * 1.02)
                old_conf = best_match[7] or 0.5
                new_conf = old_conf * 0.8 + cluster.confidence * 0.2

                con.execute("""
                    UPDATE vehicle_profiles SET
                        sensor_ids=?, encounter_count=encounter_count+1,
                        last_seen=?, encounter_lats=?, encounter_lons=?,
                        psi_tolerance=?, avg_confidence=?, updated_at=unixepoch('now')
                    WHERE vehicle_hash=?
                """, (json.dumps(new_sids), cluster.timestamp,
                      json.dumps(lats), json.dumps(lons), new_tol, new_conf, vh))
                con.execute("UPDATE vehicle_clusters SET vehicle_hash=? WHERE cluster_id=?",
                            [vh, cluster.cluster_id])
                con.commit()
                return vh, False
            else:
                vh = hashlib.md5(json.dumps(list(cluster.psi_fingerprint)).encode()).hexdigest()[:16]
                con.execute("""
                    INSERT OR IGNORE INTO vehicle_profiles
                    (vehicle_hash, nickname, sensor_ids, psi_fingerprint, psi_tolerance,
                     encounter_count, first_seen, last_seen, encounter_lats, encounter_lons,
                     protocol, avg_confidence)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """, (vh, f"Vehicle-{vh[:6].upper()}", json.dumps(cluster.sensor_ids),
                      json.dumps(list(cluster.psi_fingerprint)), self.PSI_MATCH_THRESHOLD,
                      1, cluster.timestamp, cluster.timestamp,
                      json.dumps([cluster.latitude] if cluster.latitude else []),
                      json.dumps([cluster.longitude] if cluster.longitude else []),
                      cluster.protocol, cluster.confidence))
                con.execute("UPDATE vehicle_clusters SET vehicle_hash=? WHERE cluster_id=?",
                            [vh, cluster.cluster_id])
                con.commit()
                return vh, True
        finally:
            con.close()

    def _fp_dist(self, fp1, fp2) -> float:
        if len(fp1) != len(fp2):
            return float('inf')
        return math.sqrt(sum((a-b)**2 for a,b in zip(sorted(fp1), sorted(fp2))))

    def get_stats(self) -> dict:
        con = sqlite3.connect(self.db_path)
        s = {}
        s["bursts"] = con.execute("SELECT COUNT(*) FROM burst_events").fetchone()[0]
        s["clusters"] = con.execute("SELECT COUNT(*) FROM vehicle_clusters").fetchone()[0]
        s["profiles"] = con.execute("SELECT COUNT(*) FROM vehicle_profiles").fetchone()[0]
        s["recurring_2plus"] = con.execute("SELECT COUNT(*) FROM vehicle_profiles WHERE encounter_count>=2").fetchone()[0]
        s["confirmed_3plus"] = con.execute("SELECT COUNT(*) FROM vehicle_profiles WHERE encounter_count>=3").fetchone()[0]
        by_method = con.execute("SELECT method, COUNT(*) FROM vehicle_clusters GROUP BY method").fetchall()
        s["by_method"] = dict(by_method)
        con.close()
        return s

    def log_run(self, source_db, since_ts, result, elapsed):
        con = sqlite3.connect(self.db_path)
        con.execute("""
            INSERT INTO processing_log
            (source_db, since_ts, signals_processed, bursts_detected, clusters_found,
             new_profiles, known_matches, elapsed_s)
            VALUES (?,?,?,?,?,?,?,?)
        """, (source_db, since_ts, result["signals_processed"], result["bursts_detected"],
              result["clusters_found"], result["new_vehicle_profiles"],
              result["known_vehicle_matches"], elapsed))
        con.commit()
        con.close()


class TPMSVehicleProcessor:
    def __init__(self, source_db: str, profile_db: str):
        self.source_db = source_db
        self.profile_db = profile_db
        self.detector = BurstDetector()
        self.clusterer = WithinBurstClusterer()
        self.pdb = VehicleProfileDB(profile_db)

    def load_signals(self, since_ts=946684800, limit=100000) -> List[Signal]:
        con = sqlite3.connect(self.source_db)
        rows = con.execute("""
            SELECT timestamp, tpms_id, protocol, pressure_psi, temperature_c,
                   battery_low, signal_strength, snr, frequency, latitude, longitude, confidence
            FROM tpms_signals WHERE timestamp > ? ORDER BY timestamp LIMIT ?
        """, [since_ts, limit]).fetchall()
        con.close()

        signals = []
        for r in rows:
            ss = r[6]
            if isinstance(ss, (bytes, bytearray)) and len(ss) == 4:
                try: ss = struct.unpack('<f', ss)[0]
                except: ss = None
            try: ss = float(ss) if ss is not None else None
            except: ss = None
            signals.append(Signal(r[0], r[1], r[2], r[3], r[4], bool(r[5]),
                                  ss, r[7], r[8], r[9], r[10], r[11]))
        return signals

    def process(self, since_ts=946684800, limit=100000, verbose=True) -> dict:
        t0 = time.time()
        if verbose: print(f"[v2] Loading signals since {since_ts:.0f}...")
        signals = self.load_signals(since_ts, limit)
        if verbose: print(f"[v2] {len(signals)} signals loaded")

        bursts = self.detector.detect_bursts(signals)
        if verbose: print(f"[v2] {len(bursts)} bursts detected")

        total_c, new_v, known_v = 0, 0, 0
        for burst in bursts:
            clusters = self.clusterer.cluster_burst(burst)
            self.pdb.save_burst(burst, clusters)
            for c in clusters:
                vh, is_new = self.pdb.match_and_learn(c)
                total_c += 1
                new_v += is_new
                known_v += (not is_new)

        elapsed = time.time() - t0
        result = {
            "signals_processed": len(signals),
            "bursts_detected": len(bursts),
            "clusters_found": total_c,
            "new_vehicle_profiles": new_v,
            "known_vehicle_matches": known_v,
            "elapsed_s": round(elapsed, 2),
            "db_stats": self.pdb.get_stats(),
        }
        self.pdb.log_run(self.source_db, since_ts, result, elapsed)
        if verbose: print(f"[v2] Done in {elapsed:.1f}s: {result}")
        return result
