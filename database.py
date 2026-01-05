import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd


class TPMSDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()

    def _connect(self, read_only: bool = False) -> sqlite3.Connection:
        """
        Centralized SQLite connection helper.

        - Enables WAL + reasonable pragmas for better concurrency and reduced corruption risk.
        - Uses busy_timeout so readers don't instantly fail while scanner writes.
        """
        if read_only:
            # Best-effort read-only mode (works on newer SQLite). Falls back to normal connect.
            try:
                conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=30, check_same_thread=False)
            except Exception:
                conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        else:
            conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)

        conn.row_factory = sqlite3.Row

        try:
            conn.execute("PRAGMA busy_timeout=5000;")
            conn.execute("PRAGMA foreign_keys=ON;")
            # These pragmas materially reduce the chance of corruption on abrupt shutdown
            # and improve write/read concurrency for the Streamlit UI.
            if not read_only:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                conn.execute("PRAGMA temp_store=MEMORY;")
        except Exception:
            # If a particular SQLite build rejects a pragma, don't kill the app.
            pass

        return conn

    def init_database(self):
        """Initialize database schema"""
        conn = self._connect()
        cursor = conn.cursor()

        # Raw TPMS signals
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS tpms_signals
                       (
                           id              INTEGER PRIMARY KEY AUTOINCREMENT,
                           tpms_id         TEXT NOT NULL,
                           timestamp       REAL NOT NULL,
                           latitude        REAL,
                           longitude       REAL,
                           frequency       REAL,
                           signal_strength REAL,
                           snr             REAL,
                           pressure_psi    REAL,
                           temperature_c   REAL,
                           battery_low     INTEGER,
                           protocol        TEXT,
                           raw_data        BLOB
                       )
                       ''')

        # Vehicle clusters (groups of 4 TPMS)
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS vehicles
                       (
                           id              INTEGER PRIMARY KEY AUTOINCREMENT,
                           vehicle_hash    TEXT UNIQUE NOT NULL,
                           first_seen      REAL        NOT NULL,
                           last_seen       REAL        NOT NULL,
                           encounter_count INTEGER DEFAULT 1,
                           tpms_ids        TEXT        NOT NULL, -- JSON array
                           nickname        TEXT,
                           notes           TEXT,
                           metadata        TEXT                  -- JSON for additional data
                       )
                       ''')

        # Encounters (when a vehicle is detected)
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS encounters
                       (
                           id             INTEGER PRIMARY KEY AUTOINCREMENT,
                           vehicle_id     INTEGER NOT NULL,
                           timestamp      REAL    NOT NULL,
                           latitude       REAL,
                           longitude      REAL,
                           duration       REAL,
                           signal_quality REAL,
                           FOREIGN KEY (vehicle_id) REFERENCES vehicles (id)
                       )
                       ''')

        # Maintenance tracking
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS maintenance_history
                       (
                           id                INTEGER PRIMARY KEY AUTOINCREMENT,
                           vehicle_id        INTEGER NOT NULL,
                           tpms_id           TEXT    NOT NULL,
                           timestamp         REAL    NOT NULL,
                           avg_pressure      REAL,
                           min_pressure      REAL,
                           max_pressure      REAL,
                           avg_temperature   REAL,
                           pressure_variance REAL,
                           alert_type        TEXT,
                           FOREIGN KEY (vehicle_id) REFERENCES vehicles (id)
                       )
                       ''')

        # Create indices
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tpms_id ON tpms_signals(tpms_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON tpms_signals(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_vehicle_hash ON vehicles(vehicle_hash)')

        conn.commit()
        conn.close()

    def insert_signal(self, signal_data: Dict) -> int:
        """Insert a raw TPMS signal"""
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute('''
                       INSERT INTO tpms_signals
                       (tpms_id, timestamp, latitude, longitude, frequency,
                        signal_strength, snr, pressure_psi, temperature_c,
                        battery_low, protocol, raw_data)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ''', (
                           signal_data['tpms_id'],
                           signal_data['timestamp'],
                           signal_data.get('latitude'),
                           signal_data.get('longitude'),
                           signal_data['frequency'],
                           signal_data['signal_strength'],
                           signal_data['snr'],
                           signal_data.get('pressure_psi'),
                           signal_data.get('temperature_c'),
                           signal_data.get('battery_low', 0),
                           signal_data.get('protocol', 'unknown'),
                           signal_data.get('raw_data')
                       ))

        signal_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return signal_id

    def insert_signals_batch(self, signals: List[Dict]) -> int:
        """Insert multiple signals in a batch for better performance"""
        if not signals:
            return 0
        
        conn = self._connect()
        cursor = conn.cursor()
        
        rows = []
        for signal_data in signals:
            rows.append((
                signal_data.get('tpms_id'),
                signal_data.get('timestamp'),
                signal_data.get('latitude'),
                signal_data.get('longitude'),
                signal_data.get('frequency'),
                signal_data.get('signal_strength'),
                signal_data.get('snr'),
                signal_data.get('pressure_psi'),
                signal_data.get('temperature_c'),
                signal_data.get('battery_low', 0),
                signal_data.get('protocol', 'unknown'),
                signal_data.get('raw_data')
            ))
        
        cursor.executemany('''
            INSERT INTO tpms_signals
            (tpms_id, timestamp, latitude, longitude, frequency,
             signal_strength, snr, pressure_psi, temperature_c,
             battery_low, protocol, raw_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', rows)
        
        count = cursor.rowcount
        conn.commit()
        conn.close()
        return count

    def get_recent_signals(self, time_window: int = 30) -> List[Dict]:
        """Get signals from the last N seconds"""
        conn = self._connect(read_only=True)
        cursor = conn.cursor()

        cutoff_time = datetime.now().timestamp() - time_window

        cursor.execute('''
                       SELECT *
                       FROM tpms_signals
                       WHERE timestamp > ?
                       ORDER BY timestamp DESC
                       ''', (cutoff_time,))

        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]

        conn.close()
        return results

    def get_all_unique_sensors(self):
        """Get all unique TPMS sensors ever seen"""
        query = """
                SELECT tpms_id,
                       protocol,
                       COUNT(*)             as signal_count,
                       MIN(timestamp)       as first_seen,
                       MAX(timestamp)       as last_seen,
                       AVG(signal_strength) as avg_rssi,
                       AVG(pressure_psi)    as avg_pressure,
                       AVG(temperature_c)   as avg_temperature,
                       MAX(frequency)       as frequency
                FROM tpms_signals
                GROUP BY tpms_id
                ORDER BY last_seen DESC
                """
        conn = self._connect(read_only=True)
        result = pd.read_sql_query(query, conn)
        conn.close()
        return result

    def get_sensor_history(self, tpms_id):
        """Get full history for a specific sensor"""
        query = """
                SELECT *
                FROM tpms_signals
                WHERE tpms_id = ?
                ORDER BY timestamp DESC
                """
        conn = self._connect(read_only=True)
        result = pd.read_sql_query(query, conn, params=(tpms_id,))
        conn.close()
        return result

    def get_sensor_statistics(self, tpms_id):
        """Get detailed statistics for a sensor"""
        query = """
                SELECT COUNT(*)             as total_signals,
                       MIN(timestamp)       as first_seen,
                       MAX(timestamp)       as last_seen,
                       AVG(signal_strength) as avg_rssi,
                       MIN(signal_strength) as min_rssi,
                       MAX(signal_strength) as max_rssi,
                       AVG(pressure_psi)    as avg_pressure,
                       MIN(pressure_psi)    as min_pressure,
                       MAX(pressure_psi)    as max_pressure,
                       AVG(temperature_c)   as avg_temp,
                       MIN(temperature_c)   as min_temp,
                       MAX(temperature_c)   as max_temp,
                       protocol
                FROM tpms_signals
                WHERE tpms_id = ?
                """
        conn = self._connect(read_only=True)
        conn.row_factory = sqlite3.Row
        result = conn.execute(query, (tpms_id,)).fetchone()
        conn.close()
        return dict(result) if result else None

    def get_orphaned_sensors(self):
        """Get sensors not assigned to any vehicle"""
        query = """
                SELECT DISTINCT ts.tpms_id, ts.protocol, MAX(ts.timestamp) as last_seen
                FROM tpms_signals ts
                         LEFT JOIN vehicles v ON ts.tpms_id IN (SELECT json_each.value
                                                                FROM vehicles, json_each(vehicles.tpms_ids)
                                                                WHERE vehicles.id = v.id)
                WHERE v.id IS NULL
                GROUP BY ts.tpms_id
                ORDER BY last_seen DESC
                """
        conn = self._connect(read_only=True)
        result = pd.read_sql_query(query, conn)
        conn.close()
        return result

    def assign_sensor_to_vehicle(self, tpms_id, vehicle_id):
        """Manually assign a sensor to a vehicle"""
        conn = self._connect()

        # Get current tpms_ids for vehicle
        vehicle = conn.execute(
            "SELECT tpms_ids FROM vehicles WHERE id = ?",
            (vehicle_id,)
        ).fetchone()

        if vehicle:
            tpms_ids = json.loads(vehicle[0])
            if tpms_id not in tpms_ids:
                tpms_ids.append(tpms_id)
                conn.execute(
                    "UPDATE vehicles SET tpms_ids = ? WHERE id = ?",
                    (json.dumps(tpms_ids), vehicle_id)
                )
                conn.commit()
                conn.close()
                return True

        conn.close()
        return False

    def upsert_vehicle(self, tpms_ids: List[str], timestamp: float,
                       location: Optional[Tuple[float, float]] = None) -> int:
        """Create or update a vehicle cluster"""
        vehicle_hash = self._generate_vehicle_hash(tpms_ids)

        conn = self._connect()
        cursor = conn.cursor()

        # Check if vehicle exists
        cursor.execute('SELECT id, encounter_count FROM vehicles WHERE vehicle_hash = ?',
                       (vehicle_hash,))
        result = cursor.fetchone()

        if result:
            vehicle_id, encounter_count = result
            cursor.execute('''
                           UPDATE vehicles
                           SET last_seen       = ?,
                               encounter_count = ?
                           WHERE id = ?
                           ''', (timestamp, encounter_count + 1, vehicle_id))
        else:
            cursor.execute('''
                           INSERT INTO vehicles (vehicle_hash, first_seen, last_seen, tpms_ids)
                           VALUES (?, ?, ?, ?)
                           ''', (vehicle_hash, timestamp, timestamp, json.dumps(sorted(tpms_ids))))
            vehicle_id = cursor.lastrowid

        # Record encounter
        cursor.execute('''
                       INSERT INTO encounters (vehicle_id, timestamp, latitude, longitude)
                       VALUES (?, ?, ?, ?)
                       ''', (vehicle_id, timestamp,
                             location[0] if location else None,
                             location[1] if location else None))

        conn.commit()
        conn.close()
        return vehicle_id

    def get_vehicle_history(self, vehicle_id: int) -> Dict:
        """Get complete history for a vehicle"""
        conn = self._connect(read_only=True)
        cursor = conn.cursor()

        # Vehicle info
        cursor.execute('SELECT * FROM vehicles WHERE id = ?', (vehicle_id,))
        vehicle = dict(zip([d[0] for d in cursor.description], cursor.fetchone()))
        vehicle['tpms_ids'] = json.loads(vehicle['tpms_ids'])

        # Encounters
        cursor.execute('''
                       SELECT *
                       FROM encounters
                       WHERE vehicle_id = ?
                       ORDER BY timestamp DESC
                       ''', (vehicle_id,))
        encounters = [dict(zip([d[0] for d in cursor.description], row))
                      for row in cursor.fetchall()]

        # Maintenance data
        cursor.execute('''
            SELECT tpms_id, AVG(pressure_psi) as avg_pressure, 
                   AVG(temperature_c) as avg_temp,
                   MIN(pressure_psi) as min_pressure,
                   MAX(pressure_psi) as max_pressure
            FROM tpms_signals
            WHERE tpms_id IN ({})
            GROUP BY tpms_id
        '''.format(','.join('?' * len(vehicle['tpms_ids']))), vehicle['tpms_ids'])

        maintenance = [dict(zip([d[0] for d in cursor.description], row))
                       for row in cursor.fetchall()]

        conn.close()

        return {
            'vehicle': vehicle,
            'encounters': encounters,
            'maintenance': maintenance
        }

    def get_all_vehicles(self, min_encounters: int = 1) -> List[Dict]:
        """Get all known vehicles"""
        conn = self._connect(read_only=True)
        cursor = conn.cursor()

        cursor.execute('''
                       SELECT *
                       FROM vehicles
                       WHERE encounter_count >= ?
                       ORDER BY last_seen DESC
                       ''', (min_encounters,))

        columns = [d[0] for d in cursor.description]
        vehicles = []
        for row in cursor.fetchall():
            vehicle = dict(zip(columns, row))
            vehicle['tpms_ids'] = json.loads(vehicle['tpms_ids'])
            vehicles.append(vehicle)

        conn.close()
        return vehicles

    def analyze_maintenance(self, vehicle_id: int, days: int = 30) -> Dict:
        """Analyze tire maintenance for a vehicle"""
        conn = self._connect(read_only=True)
        cursor = conn.cursor()

        # Get TPMS IDs for this vehicle
        cursor.execute('SELECT tpms_ids FROM vehicles WHERE id = ?', (vehicle_id,))
        result = cursor.fetchone()
        if not result:
            conn.close()
            return {}
        
        tpms_ids = json.loads(result[0])
        cutoff_time = datetime.now().timestamp() - (days * 86400)

        # Get pressure and temperature trends
        cursor.execute('''
            SELECT tpms_id, timestamp, pressure_psi, temperature_c
            FROM tpms_signals
            WHERE tpms_id IN ({}) AND timestamp > ?
            ORDER BY timestamp
        '''.format(','.join('?' * len(tpms_ids))), tpms_ids + [cutoff_time])

        data = cursor.fetchall()
        conn.close()

        # Analyze per tire
        tire_analysis = {}
        for tpms_id in tpms_ids:
            tire_data = [(t, p, temp) for tid, t, p, temp in data if tid == tpms_id]

            if tire_data:
                pressures = [p for _, p, _ in tire_data if p is not None]
                temps = [t for _, _, t in tire_data if t is not None]

                tire_analysis[tpms_id] = {
                    'avg_pressure': np.mean(pressures) if pressures else None,
                    'pressure_std': np.std(pressures) if pressures else None,
                    'min_pressure': min(pressures) if pressures else None,
                    'max_pressure': max(pressures) if pressures else None,
                    'avg_temp': np.mean(temps) if temps else None,
                    'readings_count': len(tire_data),
                    'alerts': self._generate_alerts(pressures, temps)
                }

        return tire_analysis

    def _generate_alerts(self, pressures: List[float], temps: List[float]) -> List[str]:
        """Generate maintenance alerts"""
        alerts = []

        if pressures:
            avg_pressure = np.mean(pressures)
            if avg_pressure < 28:
                alerts.append('LOW_PRESSURE')
            elif avg_pressure > 40:
                alerts.append('HIGH_PRESSURE')

            if len(pressures) > 5:
                pressure_variance = np.std(pressures)
                if pressure_variance > 5:
                    alerts.append('UNSTABLE_PRESSURE')

        if temps:
            avg_temp = np.mean(temps)
            if avg_temp > 80:
                alerts.append('HIGH_TEMPERATURE')

        return alerts

    def _generate_vehicle_hash(self, tpms_ids: List[str]) -> str:
        """Generate a unique hash for a vehicle based on its TPMS IDs"""
        return '-'.join(sorted(tpms_ids))

    def update_vehicle_nickname(self, vehicle_id: int, nickname: str):
        """Set a friendly name for a vehicle"""
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute('UPDATE vehicles SET nickname = ? WHERE id = ?',
                       (nickname, vehicle_id))
        conn.commit()
        conn.close()

    # -----------------------------
    # Fast, UI-friendly helpers
    # -----------------------------

    def get_realtime_stats(self, now_ts: Optional[float] = None,
                           minute_window_s: int = 60,
                           hour_window_s: int = 3600) -> Dict[str, Any]:
        """Return lightweight stats for the sidebar (fast, indexed queries)."""
        import time as _time
        now_ts = float(now_ts if now_ts is not None else _time.time())
        ts_min = now_ts - float(minute_window_s)
        ts_hr = now_ts - float(hour_window_s)

        conn = self._connect(read_only=True)
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) AS n FROM tpms_signals WHERE timestamp >= ?", (ts_min,))
        n_last_min = int(cur.fetchone()[0] or 0)

        cur.execute("SELECT COUNT(*) AS n FROM tpms_signals WHERE timestamp >= ?", (ts_hr,))
        n_last_hour = int(cur.fetchone()[0] or 0)

        cur.execute("SELECT COUNT(DISTINCT tpms_id) AS n FROM tpms_signals WHERE timestamp >= ?", (ts_hr,))
        unique_sensors_last_hour = int(cur.fetchone()[0] or 0)

        repeats_last_hour = max(0, n_last_hour - unique_sensors_last_hour)
        rate_per_hour_last_min = n_last_min * 60

        # Best-effort counts that should be fast even on large DBs
        try:
            cur.execute("SELECT COUNT(*) FROM vehicles")
            known_vehicles = int(cur.fetchone()[0] or 0)
        except Exception:
            known_vehicles = 0

        try:
            cur.execute("SELECT COUNT(DISTINCT tpms_id) FROM tpms_signals")
            known_sensors = int(cur.fetchone()[0] or 0)
        except Exception:
            known_sensors = 0

        conn.close()

        return {
            "now_ts": now_ts,
            "n_last_min": n_last_min,
            "n_last_hour": n_last_hour,
            "signals_last_hour": n_last_hour,
            "unique_sensors_last_hour": unique_sensors_last_hour,
            "repeats_last_hour": repeats_last_hour,
            "repeated_signals_last_hour": repeats_last_hour,
            "rate_per_hour_last_min": rate_per_hour_last_min,
            "known_vehicles": known_vehicles,
            "known_sensors": known_sensors,
        }

    def get_signals_since_rowid(self, last_rowid: int = 0, limit: int = 5000) -> List[Dict[str, Any]]:
        """Fetch signals incrementally for online learning (ordered by rowid)."""
        conn = self._connect(read_only=True)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                rowid,
                tpms_id,
                protocol,
                timestamp,
                frequency,
                pressure_psi,
                temperature_c,
                battery_low,
                signal_strength,
                snr,
                latitude,
                longitude,
                raw_data
            FROM tpms_signals
            WHERE rowid > ?
            ORDER BY rowid ASC
            LIMIT ?
            """,
            (int(last_rowid), int(limit)),
        )
        rows = cur.fetchall()
        conn.close()

        out: List[Dict[str, Any]] = []
        for r in rows:
            # sqlite3.Row supports mapping-like access
            out.append(dict(r))
        return out

    def ensure_performance_indexes(self) -> None:
        """Create extra indexes that make dashboard queries fast."""
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_tpms_signals_timestamp ON tpms_signals(timestamp)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_tpms_signals_tpmsid ON tpms_signals(tpms_id)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_tpms_signals_timestamp_tpmsid ON tpms_signals(timestamp, tpms_id)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_tpms_signals_tpmsid_timestamp ON tpms_signals(tpms_id, timestamp)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_encounters_timestamp ON encounters(timestamp)"
            )
            conn.commit()
        finally:
            conn.close()

