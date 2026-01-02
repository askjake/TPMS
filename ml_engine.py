"""
Machine Learning Engine for TPMS Signal Analysis
Modern Python (3.10+) compatible
"""
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Optional, Tuple
import time
from dataclasses import dataclass, field
from collections import defaultdict

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
    ML engine for clustering TPMS sensors by vehicle
    Uses DBSCAN for spatial-temporal clustering
    """
    
    def __init__(self, min_samples: int = 3, eps: float = 0.5):
        self.min_samples = min_samples
        self.eps = eps
        self.scaler = StandardScaler()
        self.signal_history: List[SignalCharacteristics] = []
        self.clusters: Dict[int, List[str]] = {}
        self.vehicle_profiles: Dict[int, Dict] = {}
        
    def add_signal(self, signal: SignalCharacteristics):
        """Add a signal to the history"""
        self.signal_history.append(signal)
        
        # Keep only recent signals (last 1000)
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]
    
    def cluster_sensors(self, sensor_data: List[Dict]) -> Dict[int, List[str]]:
        """
        Cluster TPMS sensors by vehicle using DBSCAN
        
        Args:
            sensor_data: List of sensor readings with timestamps and signal characteristics
            
        Returns:
            Dictionary mapping cluster_id to list of sensor IDs
        """
        if len(sensor_data) < self.min_samples:
            return {}
        
        # Extract features for clustering
        features = []
        sensor_ids = []
        
        for sensor in sensor_data:
            # Features: time_of_day, signal_strength, frequency_offset, etc.
            features.append([
                sensor.get('timestamp', 0) % 86400,  # Time of day in seconds
                sensor.get('signal_strength', 0),
                sensor.get('frequency', 0),
                sensor.get('snr', 0),
            ])
            sensor_ids.append(sensor.get('id', ''))
        
        features = np.array(features)
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = clustering.fit_predict(features_scaled)
        
        # Group sensors by cluster
        clusters = defaultdict(list)
        for sensor_id, label in zip(sensor_ids, labels):
            if label != -1:  # -1 is noise in DBSCAN
                clusters[label].append(sensor_id)
        
        self.clusters = dict(clusters)
        return self.clusters
    
    def identify_vehicle(self, sensor_ids: List[str]) -> Optional[int]:
        """
        Identify which vehicle a set of sensors belongs to
        
        Args:
            sensor_ids: List of TPMS sensor IDs
            
        Returns:
            Vehicle cluster ID or None if no match
        """
        # Find cluster with most matching sensors
        best_match = None
        best_score = 0
        
        for cluster_id, cluster_sensors in self.clusters.items():
            # Calculate overlap
            overlap = len(set(sensor_ids) & set(cluster_sensors))
            score = overlap / len(sensor_ids) if sensor_ids else 0
            
            if score > best_score and score >= 0.5:  # At least 50% match
                best_score = score
                best_match = cluster_id
        
        return best_match
    
    def get_vehicle_profile(self, cluster_id: int) -> Optional[Dict]:
        """Get the profile for a vehicle cluster"""
        return self.vehicle_profiles.get(cluster_id)
    
    def update_vehicle_profile(self, cluster_id: int, sensor_data: List[Dict]):
        """Update vehicle profile with new sensor data"""
        if cluster_id not in self.vehicle_profiles:
            self.vehicle_profiles[cluster_id] = {
                'sensor_ids': [],
                'first_seen': time.time(),
                'last_seen': time.time(),
                'detection_count': 0,
                'avg_signal_strength': 0,
                'typical_frequency': 0,
            }
        
        profile = self.vehicle_profiles[cluster_id]
        
        # Update profile
        profile['last_seen'] = time.time()
        profile['detection_count'] += 1
        
        # Update sensor IDs
        for sensor in sensor_data:
            sensor_id = sensor.get('id', '')
            if sensor_id and sensor_id not in profile['sensor_ids']:
                profile['sensor_ids'].append(sensor_id)
        
        # Update statistics
        if sensor_data:
            profile['avg_signal_strength'] = np.mean([
                s.get('signal_strength', 0) for s in sensor_data
            ])
            profile['typical_frequency'] = np.mean([
                s.get('frequency', 0) for s in sensor_data
            ])
    
    def get_statistics(self) -> Dict:
        """Get clustering statistics"""
        return {
            'total_signals': len(self.signal_history),
            'num_clusters': len(self.clusters),
            'num_vehicles': len(self.vehicle_profiles),
            'signals_per_cluster': {
                cid: len(sensors) for cid, sensors in self.clusters.items()
            }
        }

    def find_patterns(self, days: int = 30) -> Dict:
        """
        Find patterns in vehicle encounters
    
        Args:
            days: Number of days to analyze
        
        Returns:
            Dictionary with pattern analysis
        """
        from datetime import datetime, timedelta
    
        cutoff_time = datetime.now().timestamp() - (days * 86400)
    
        # Get vehicles from database
        if not hasattr(self, 'db') or self.db is None:
            return {
                'frequent_vehicles': [],
                'time_patterns': {},
                'location_clusters': {}
            }
    
        vehicles = self.db.get_all_vehicles(min_encounters=2)
    
        # Filter by time window
        frequent_vehicles = []
        for vehicle in vehicles:
            if vehicle['last_seen'] >= cutoff_time:
                history = self.db.get_vehicle_history(vehicle['id'])
            
                # Count encounters in time window
                recent_encounters = [
                    e for e in history['encounters'] 
                    if e['timestamp'] >= cutoff_time
                ]
            
                if len(recent_encounters) >= 3:
                    frequent_vehicles.append({
                        'id': vehicle['id'],
                        'nickname': vehicle.get('nickname', f"Vehicle {vehicle['id']}"),
                        'encounter_count': len(recent_encounters),
                        'first_seen': datetime.fromtimestamp(vehicle['first_seen']),
                        'last_seen': datetime.fromtimestamp(vehicle['last_seen']),
                        'tpms_ids': vehicle['tpms_ids']
                    })
    
        # Sort by encounter count
        frequent_vehicles.sort(key=lambda x: x['encounter_count'], reverse=True)
    
        return {
            'frequent_vehicles': frequent_vehicles,
            'time_patterns': self._analyze_time_patterns(frequent_vehicles),
            'location_clusters': {}
        }

    def _analyze_time_patterns(self, vehicles: List[Dict]) -> Dict:
        """Analyze temporal patterns in vehicle encounters"""
        patterns = {
            'peak_hours': [],
            'peak_days': [],
            'regular_commuters': []
        }
    
        if not vehicles:
            return patterns
    
        # Analyze hour-of-day patterns
        hour_counts = {}
        for vehicle in vehicles:
            if hasattr(self, 'db') and self.db:
                history = self.db.get_vehicle_history(vehicle['id'])
                for encounter in history['encounters']:
                    from datetime import datetime
                    dt = datetime.fromtimestamp(encounter['timestamp'])
                    hour = dt.hour
                    hour_counts[hour] = hour_counts.get(hour, 0) + 1
    
        if hour_counts:
            # Find peak hours
            sorted_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)
            patterns['peak_hours'] = [
                {'hour': h, 'count': c} for h, c in sorted_hours[:3]
            ]
    
        return patterns

    def predict_next_encounter(self, vehicle_id: int) -> Dict:
        """
        Predict next encounter time for a vehicle
    
        Args:
            vehicle_id: Vehicle ID
        
        Returns:
            Dictionary with prediction details
        """
        from datetime import datetime, timedelta
    
        if not hasattr(self, 'db') or self.db is None:
            return {
                'prediction': 'insufficient_data',
                'confidence': 0.0
            }
    
        history = self.db.get_vehicle_history(vehicle_id)
        encounters = history['encounters']
    
        if len(encounters) < 3:
            return {
                'prediction': 'insufficient_data',
                'confidence': 0.0
            }
        
        # Calculate average time between encounters
        timestamps = sorted([e['timestamp'] for e in encounters])
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
    
        if not intervals:
            return {
                'prediction': 'insufficient_data',
                'confidence': 0.0
            }
    
        avg_interval = sum(intervals) / len(intervals)
        std_interval = (sum((x - avg_interval)**2 for x in intervals) / len(intervals)) ** 0.5
    
        # Predict next encounter
        last_encounter = timestamps[-1]
        predicted_time = last_encounter + avg_interval
        predicted_datetime = datetime.fromtimestamp(predicted_time)
    
        # Calculate confidence based on consistency
        if std_interval > 0:
            coefficient_of_variation = std_interval / avg_interval
            confidence = max(0.0, min(1.0, 1.0 - coefficient_of_variation))
        else:
            confidence = 0.9
    
        return {
            'prediction': 'estimated',
            'predicted_datetime': predicted_datetime,
            'predicted_timestamp': predicted_time,
            'confidence': confidence,
            'avg_interval_hours': avg_interval / 3600,
            'last_encounter': datetime.fromtimestamp(last_encounter)
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

def create_clustering_engine() -> VehicleClusteringEngine:
    """Factory function to create clustering engine"""
    return VehicleClusteringEngine(min_samples=3, eps=0.5)


