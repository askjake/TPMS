"""
Machine Learning for Signal Pattern Recognition
Learns from detected signals to improve future detection
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import pickle
import time
from dataclasses import dataclass
from collections import deque


@dataclass
class SignalProfile:
    """Learned profile for a signal type"""
    frequency: float
    bandwidth_range: Tuple[float, float]
    modulation_type: str
    baud_rate_range: Tuple[int, int]
    power_range: Tuple[float, float]
    snr_threshold: float
    confidence: float
    sample_count: int
    last_updated: float
    characteristics: Dict


class SignalLearningEngine:
    """ML engine for learning and recognizing signal patterns"""

    def __init__(self, db):
        self.db = db
        self.signal_profiles = {}  # Learned signal profiles
        self.training_buffer = deque(maxlen=1000)  # Recent signals for training

        # Adaptive thresholds
        self.adaptive_thresholds = {
            'power_threshold': -80.0,  # dBm
            'snr_threshold': 5.0,  # dB
            'bandwidth_tolerance': 0.2,  # MHz
            'frequency_tolerance': 0.1  # MHz
        }

        # ML models
        self.modulation_classifier = None
        self.protocol_classifier = None
        self.scaler = StandardScaler()

        # Statistics
        self.detection_stats = {
            'total_attempts': 0,
            'successful_decodes': 0,
            'false_positives': 0,
            'missed_signals': 0
        }

        self._load_models()

    # Add to SignalLearningEngine class

    def get_decoder_hints(self, frequency: float, signal_strength: float) -> Dict:
        """Get hints for decoder based on learned patterns"""
        hints = {
            'try_protocols': [],  # Ordered list of protocols to try
            'baud_rates': [],  # Suggested baud rates
            'expected_length': None,
            'confidence': 0.0
        }

        # Find matching profiles
        matching_profiles = []
        for key, profile in self.signal_profiles.items():
            if abs(profile.frequency - frequency) < self.adaptive_thresholds['frequency_tolerance']:
                # Check if signal strength is in expected range
                if profile.power_range[0] <= signal_strength <= profile.power_range[1] + 10:
                    matching_profiles.append(profile)

        if not matching_profiles:
            return hints

        # Sort by confidence
        matching_profiles.sort(key=lambda p: p.confidence, reverse=True)

        # Build hints from top profiles
        for profile in matching_profiles[:3]:  # Top 3 matches
            if profile.modulation_type not in hints['try_protocols']:
                hints['try_protocols'].append(profile.modulation_type)

            # Add baud rate range
            hints['baud_rates'].extend([
                profile.baud_rate_range[0],
                int(np.mean(profile.baud_rate_range)),
                profile.baud_rate_range[1]
            ])

        # Remove duplicates and sort
        hints['baud_rates'] = sorted(list(set(hints['baud_rates'])))
        hints['confidence'] = matching_profiles[0].confidence if matching_profiles else 0.0

        return hints

    def should_skip_frequency(self, frequency: float) -> bool:
        """Determine if a frequency should be skipped based on learning"""
        # Check if we have enough data
        total_attempts = 0
        total_successes = 0

        for key, profile in self.signal_profiles.items():
            if abs(profile.frequency - frequency) < 0.5:
                total_attempts += profile.sample_count
                # Estimate successes based on confidence
                total_successes += int(profile.sample_count * profile.confidence)

        # Need at least 100 attempts to make a decision
        if total_attempts < 100:
            return False

        # If success rate is very low, consider skipping
        success_rate = total_successes / total_attempts if total_attempts > 0 else 0

        # Skip if less than 0.5% success rate
        return success_rate < 0.005

    def get_frequency_priority(self, frequency: float) -> float:
        """Get priority score for a frequency (higher = more important)"""
        priority = 0.5  # Default

        # Find profiles for this frequency
        for key, profile in self.signal_profiles.items():
            if abs(profile.frequency - frequency) < 0.5:
                # Higher confidence and more samples = higher priority
                priority += profile.confidence * 0.3
                priority += min(profile.sample_count / 1000, 0.2)  # Cap at 0.2

        return min(priority, 1.0)

    def learn_from_signal(self, signal_data: Dict, decoded: bool,
                          protocol: Optional[str] = None):
        """Learn from a detected signal"""
        self.training_buffer.append({
            'timestamp': time.time(),
            'frequency': signal_data['frequency'],
            'power': signal_data['power'],
            'snr': signal_data['snr'],
            'bandwidth': signal_data.get('bandwidth', 0),
            'modulation': signal_data.get('modulation', 'unknown'),
            'baud_rate': signal_data.get('baud_rate', 0),
            'decoded': decoded,
            'protocol': protocol,
            'characteristics': signal_data.get('characteristics', {})
        })

        # Update statistics
        self.detection_stats['total_attempts'] += 1
        if decoded:
            self.detection_stats['successful_decodes'] += 1

            # Update or create signal profile
            self._update_signal_profile(signal_data, protocol)

        # Retrain models periodically
        if len(self.training_buffer) >= 100 and \
                self.detection_stats['total_attempts'] % 50 == 0:
            self._retrain_models()

    def _update_signal_profile(self, signal_data: Dict, protocol: str):
        """Update learned profile for this signal type"""
        key = f"{signal_data['frequency']:.1f}_{protocol}"

        if key in self.signal_profiles:
            profile = self.signal_profiles[key]

            # Update ranges using exponential moving average
            alpha = 0.3  # Learning rate

            # Update bandwidth range
            bw = signal_data.get('bandwidth', profile.bandwidth_range[0])
            profile.bandwidth_range = (
                min(profile.bandwidth_range[0], bw),
                max(profile.bandwidth_range[1], bw)
            )

            # Update power range
            power = signal_data['power']
            profile.power_range = (
                min(profile.power_range[0], power),
                max(profile.power_range[1], power)
            )

            # Update baud rate range
            if signal_data.get('baud_rate', 0) > 0:
                br = signal_data['baud_rate']
                profile.baud_rate_range = (
                    min(profile.baud_rate_range[0], br),
                    max(profile.baud_rate_range[1], br)
                )

            # Increase confidence
            profile.confidence = min(1.0, profile.confidence + 0.05)
            profile.sample_count += 1
            profile.last_updated = time.time()

        else:
            # Create new profile
            self.signal_profiles[key] = SignalProfile(
                frequency=signal_data['frequency'],
                bandwidth_range=(
                    signal_data.get('bandwidth', 0.1),
                    signal_data.get('bandwidth', 0.3)
                ),
                modulation_type=signal_data.get('modulation', 'FSK'),
                baud_rate_range=(
                    signal_data.get('baud_rate', 5000),
                    signal_data.get('baud_rate', 20000)
                ),
                power_range=(signal_data['power'] - 5, signal_data['power'] + 5),
                snr_threshold=max(5.0, signal_data['snr'] - 3),
                confidence=0.5,
                sample_count=1,
                last_updated=time.time(),
                characteristics=signal_data.get('characteristics', {})
            )

    def get_optimal_scan_parameters(self, frequency: float) -> Dict:
        """Get optimal scanning parameters based on learned profiles"""
        # Find profiles near this frequency
        nearby_profiles = [
            p for f, p in self.signal_profiles.items()
            if abs(p.frequency - frequency) < 1.0
        ]

        if not nearby_profiles:
            # Use defaults
            return {
                'bandwidth': 0.5,
                'gain': 40,
                'duration': 1.0,
                'threshold': self.adaptive_thresholds['power_threshold']
            }

        # Use learned parameters
        best_profile = max(nearby_profiles, key=lambda p: p.confidence)

        return {
            'bandwidth': np.mean(best_profile.bandwidth_range),
            'gain': self._calculate_optimal_gain(best_profile),
            'duration': self._calculate_optimal_duration(best_profile),
            'threshold': best_profile.power_range[0] - 5,
            'expected_modulation': best_profile.modulation_type,
            'expected_baud_rate': int(np.mean(best_profile.baud_rate_range))
        }

    def _calculate_optimal_gain(self, profile: SignalProfile) -> int:
        """Calculate optimal gain for this signal type"""
        avg_power = np.mean(profile.power_range)

        # Target: bring signal to around -40 dBm
        target_power = -40
        gain_adjustment = target_power - avg_power

        # Clamp to valid range
        optimal_gain = int(np.clip(20 + gain_adjustment, 0, 47))

        # Ensure even number
        return (optimal_gain // 2) * 2

    def _calculate_optimal_duration(self, profile: SignalProfile) -> float:
        """Calculate optimal capture duration"""
        # Shorter duration for strong, well-known signals
        if profile.confidence > 0.8 and profile.sample_count > 20:
            return 0.5
        elif profile.confidence > 0.5:
            return 1.0
        else:
            return 2.0

    def _retrain_models(self):
        """Retrain ML models with accumulated data"""
        if len(self.training_buffer) < 50:
            return

        # Prepare training data
        X = []
        y_modulation = []
        y_protocol = []

        for sample in self.training_buffer:
            if not sample['decoded']:
                continue

            features = [
                sample['power'],
                sample['snr'],
                sample['bandwidth'],
                sample.get('characteristics', {}).get('amp_var', 0),
                sample.get('characteristics', {}).get('phase_var', 0),
                sample.get('characteristics', {}).get('freq_var', 0)
            ]

            X.append(features)
            y_modulation.append(sample['modulation'])
            if sample['protocol']:
                y_protocol.append(sample['protocol'])

        if len(X) < 10:
            return

        X = np.array(X)

        # Train modulation classifier
        try:
            from sklearn.ensemble import RandomForestClassifier

            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)

            self.modulation_classifier = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                random_state=42
            )
            self.modulation_classifier.fit(X_scaled, y_modulation)

            print(f"✅ Retrained modulation classifier with {len(X)} samples")

        except Exception as e:
            print(f"⚠️  Error retraining models: {e}")

    def predict_modulation(self, signal_characteristics: Dict) -> Tuple[str, float]:
        """Predict modulation type using ML"""
        if self.modulation_classifier is None:
            return "Unknown", 0.0

        try:
            features = [[
                signal_characteristics.get('power', -80),
                signal_characteristics.get('snr', 0),
                signal_characteristics.get('bandwidth', 0.2),
                signal_characteristics.get('amp_var', 0),
                signal_characteristics.get('phase_var', 0),
                signal_characteristics.get('freq_var', 0)
            ]]

            X_scaled = self.scaler.transform(features)
            prediction = self.modulation_classifier.predict(X_scaled)[0]
            probabilities = self.modulation_classifier.predict_proba(X_scaled)[0]
            confidence = np.max(probabilities)

            return prediction, confidence

        except Exception as e:
            print(f"Error predicting modulation: {e}")
            return "Unknown", 0.0

    def get_learning_statistics(self) -> Dict:
        """Get statistics about learning progress"""
        success_rate = 0
        if self.detection_stats['total_attempts'] > 0:
            success_rate = (self.detection_stats['successful_decodes'] /
                            self.detection_stats['total_attempts'])

        return {
            'total_attempts': self.detection_stats['total_attempts'],
            'successful_decodes': self.detection_stats['successful_decodes'],
            'success_rate': success_rate,
            'learned_profiles': len(self.signal_profiles),
            'training_samples': len(self.training_buffer),
            'model_trained': self.modulation_classifier is not None,
            'profiles': {
                key: {
                    'frequency': p.frequency,
                    'confidence': p.confidence,
                    'sample_count': p.sample_count,
                    'modulation': p.modulation_type
                }
                for key, p in self.signal_profiles.items()
            }
        }

    def _save_models(self):
        """Save learned models to disk"""
        try:
            with open('ml_signal_models.pkl', 'wb') as f:
                pickle.dump({
                    'profiles': self.signal_profiles,
                    'thresholds': self.adaptive_thresholds,
                    'modulation_classifier': self.modulation_classifier,
                    'scaler': self.scaler,
                    'stats': self.detection_stats
                }, f)
        except Exception as e:
            print(f"Error saving models: {e}")

    def _load_models(self):
        """Load previously learned models"""
        try:
            with open('ml_signal_models.pkl', 'rb') as f:
                data = pickle.load(f)
                self.signal_profiles = data.get('profiles', {})
                self.adaptive_thresholds = data.get('thresholds', self.adaptive_thresholds)
                self.modulation_classifier = data.get('modulation_classifier')
                self.scaler = data.get('scaler', StandardScaler())
                self.detection_stats = data.get('stats', self.detection_stats)
            print(f"✅ Loaded {len(self.signal_profiles)} learned signal profiles")
        except FileNotFoundError:
            print("ℹ️  No previous learning data found, starting fresh")
        except Exception as e:
            print(f"⚠️  Error loading models: {e}")
