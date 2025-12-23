import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import time


class VehicleClusteringEngine:
    """ML engine for identifying and tracking vehicles"""

    def __init__(self, db):
        self.db = db
        self.active_clusters = {}  # Current detection window
        self.cluster_timeout = 60  # seconds

    def process_signals(self, signals: List[Dict]) -> List[int]:
        """Process new signals and identify vehicles"""
        if not signals:
            return []

        current_time = time.time()
        vehicle_ids = []

        # Group signals by time proximity
        time_clusters = self._cluster_by_time(signals)

        for cluster in time_clusters:
            # Check if this could be a vehicle (4 TPMS)
            tpms_ids = list(set([s['tpms_id'] for s in cluster]))

            if len(tpms_ids) >= 3:  # At least 3 sensors (allow for missed signals)
                # Check if this matches an existing active cluster
                vehicle_id = self._match_or_create_vehicle(tpms_ids, current_time, cluster)
                if vehicle_id:
                    vehicle_ids.append(vehicle_id)

        # Clean up old active clusters
        self._cleanup_active_clusters(current_time)

        return vehicle_ids

    def _cluster_by_time(self, signals: List[Dict], time_window: float = 5.0) -> List[List[Dict]]:
        """Cluster signals that appear close in time"""
        if not signals:
            return []

        # Sort by timestamp
        sorted_signals = sorted(signals, key=lambda x: x['timestamp'])

        clusters = []
        current_cluster = [sorted_signals[0]]

        for signal in sorted_signals[1:]:
            if signal['timestamp'] - current_cluster[-1]['timestamp'] <= time_window:
                current_cluster.append(signal)
            else:
                if len(current_cluster) >= 3:
                    clusters.append(current_cluster)
                current_cluster = [signal]

        if len(current_cluster) >= 3:
            clusters.append(current_cluster)

        return clusters

    def _match_or_create_vehicle(self, tpms_ids: List[str], timestamp: float,
                                 signals: List[Dict]) -> int:
        """Match signals to existing vehicle or create new one"""
        tpms_set = set(tpms_ids)

        # Check active clusters first (currently nearby vehicles)
        for vehicle_hash, cluster_info in self.active_clusters.items():
            existing_ids = set(cluster_info['tpms_ids'])

            # Calculate similarity (Jaccard index)
            intersection = len(tpms_set & existing_ids)
            union = len(tpms_set | existing_ids)
            similarity = intersection / union if union > 0 else 0

            if similarity >= 0.5:  # At least 50% match (2 out of 4 tires)
                # Update active cluster
                cluster_info['last_seen'] = timestamp
                cluster_info['tpms_ids'] = list(tpms_set | existing_ids)

                # Update database
                location = self._extract_location(signals)
                vehicle_id = self.db.upsert_vehicle(
                    cluster_info['tpms_ids'],
                    timestamp,
                    location
                )
                cluster_info['vehicle_id'] = vehicle_id
                return vehicle_id

        # No match found, create new active cluster
        vehicle_hash = '-'.join(sorted(tpms_ids))
        self.active_clusters[vehicle_hash] = {
            'tpms_ids': tpms_ids,
            'first_seen': timestamp,
            'last_seen': timestamp,
            'vehicle_id': None
        }

        # Create in database
        location = self._extract_location(signals)
        vehicle_id = self.db.upsert_vehicle(tpms_ids, timestamp, location)
        self.active_clusters[vehicle_hash]['vehicle_id'] = vehicle_id

        return vehicle_id

    def _cleanup_active_clusters(self, current_time: float):
        """Remove stale active clusters"""
        to_remove = []
        for vehicle_hash, cluster_info in self.active_clusters.items():
            if current_time - cluster_info['last_seen'] > self.cluster_timeout:
                to_remove.append(vehicle_hash)

        for vehicle_hash in to_remove:
            del self.active_clusters[vehicle_hash]

    def _extract_location(self, signals: List[Dict]) -> Tuple[float, float]:
        """Extract location from signals (if GPS available)"""
        for signal in signals:
            if signal.get('latitude') and signal.get('longitude'):
                return (signal['latitude'], signal['longitude'])
        return None

    def find_patterns(self, days: int = 30) -> Dict:
        """Analyze patterns in vehicle encounters"""
        vehicles = self.db.get_all_vehicles(min_encounters=3)

        patterns = {
            'frequent_vehicles': [],
            'time_patterns': defaultdict(list),
            'location_patterns': defaultdict(list)
        }

        for vehicle in vehicles:
            history = self.db.get_vehicle_history(vehicle['id'])
            encounters = history['encounters']

            if len(encounters) >= 3:
                patterns['frequent_vehicles'].append({
                    'vehicle_id': vehicle['id'],
                    'nickname': vehicle.get('nickname', 'Unknown'),
                    'encounter_count': len(encounters),
                    'first_seen': datetime.fromtimestamp(vehicle['first_seen']),
                    'last_seen': datetime.fromtimestamp(vehicle['last_seen']),
                    'tpms_ids': vehicle['tpms_ids']
                })

                # Analyze time patterns
                for encounter in encounters:
                    dt = datetime.fromtimestamp(encounter['timestamp'])
                    hour = dt.hour
                    day_of_week = dt.strftime('%A')

                    patterns['time_patterns'][vehicle['id']].append({
                        'hour': hour,
                        'day': day_of_week,
                        'timestamp': encounter['timestamp']
                    })

        # Find common encounter times
        for vehicle_id, times in patterns['time_patterns'].items():
            hours = [t['hour'] for t in times]
            if hours:
                patterns['time_patterns'][vehicle_id] = {
                    'common_hours': self._find_common_hours(hours),
                    'encounters': times
                }

        return patterns

    def _find_common_hours(self, hours: List[int]) -> List[int]:
        """Find most common hours of encounter"""
        hour_counts = defaultdict(int)
        for hour in hours:
            hour_counts[hour] += 1

        # Return hours that appear more than once
        common = [hour for hour, count in hour_counts.items() if count > 1]
        return sorted(common)

    def predict_next_encounter(self, vehicle_id: int) -> Dict:
        """Predict when a vehicle might be seen again"""
        history = self.db.get_vehicle_history(vehicle_id)
        encounters = history['encounters']

        if len(encounters) < 3:
            return {'prediction': 'insufficient_data'}

        # Calculate time intervals between encounters
        timestamps = sorted([e['timestamp'] for e in encounters])
        intervals = np.diff(timestamps)

        if len(intervals) > 0:
            avg_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            last_seen = timestamps[-1]

            predicted_time = last_seen + avg_interval
            confidence = 1.0 / (1.0 + std_interval / avg_interval)

            return {
                'prediction': 'estimated',
                'predicted_timestamp': predicted_time,
                'predicted_datetime': datetime.fromtimestamp(predicted_time),
                'confidence': confidence,
                'avg_interval_hours': avg_interval / 3600,
                'last_seen': datetime.fromtimestamp(last_seen)
            }

        return {'prediction': 'insufficient_data'}
