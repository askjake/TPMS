import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np


class TPMSDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
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
        conn = sqlite3.connect(self.db_path)
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

    def get_recent_signals(self, time_window: int = 30) -> List[Dict]:
        """Get signals from the last N seconds"""
        conn = sqlite3.connect(self.db_path)
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

    def upsert_vehicle(self, tpms_ids: List[str], timestamp: float,
                       location: Optional[Tuple[float, float]] = None) -> int:
        """Create or update a vehicle cluster"""
        vehicle_hash = self._generate_vehicle_hash(tpms_ids)

        conn = sqlite3.connect(self.db_path)
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
        conn = sqlite3.connect(self.db_path)
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
        conn = sqlite3.connect(self.db_path)
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
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get TPMS IDs for this vehicle
        cursor.execute('SELECT tpms_ids FROM vehicles WHERE id = ?', (vehicle_id,))
        tpms_ids = json.loads(cursor.fetchone()[0])

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
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('UPDATE vehicles SET nickname = ? WHERE id = ?',
                       (nickname, vehicle_id))
        conn.commit()
        conn.close()
