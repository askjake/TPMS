"""
TPMS Signal Decoder with Protocol Detection
"""
import numpy as np
from scipy import signal as scipy_signal
from dataclasses import dataclass
from typing import List, Optional, Dict
import time
from config import config

@dataclass
class TPMSSignal:
    tpms_id: str
    timestamp: float
    frequency: float
    signal_strength: float
    snr: float
    pressure_psi: Optional[float]
    temperature_c: Optional[float]
    battery_low: bool
    protocol: str
    raw_data: bytes
    confidence: float = 0.0

@dataclass
class UnknownSignal:
    timestamp: float
    frequency: float
    signal_strength: float
    modulation_type: str
    baud_rate: Optional[int]
    packet_length: int
    pattern_signature: str
    raw_samples: np.ndarray

class TPMSDecoder:
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.unknown_signals = []
        self.protocol_patterns = {}
        self._init_protocol_patterns()

    # Add to TPMSDecoder class after __init__

    def set_learning_engine(self, learning_engine):
        """Set reference to learning engine for adaptive decoding"""
        self.learning_engine = learning_engine

    def process_samples(self, iq_samples: np.ndarray, frequency: float, signal_strength_dbm: Optional[float] = None) -> List[TPMSSignal]:
        """Process IQ samples and decode TPMS signals with ML guidance"""
        signals = []
    
        # Calculate signal power
        power = np.abs(iq_samples) ** 2
        avg_power = np.mean(power)
        signal_strength = 10 * np.log10(avg_power + 1e-10) - 60
    
        # Get ML-guided parameters if available
        if hasattr(self, 'learning_engine') and self.learning_engine:
            optimal_params = self.learning_engine.get_optimal_scan_parameters(frequency)
            threshold = optimal_params.get('threshold', config.SIGNAL_THRESHOLD)
            expected_modulation = optimal_params.get('expected_modulation', None)
            expected_baud = optimal_params.get('expected_baud_rate', None)
        else:
            threshold = config.SIGNAL_THRESHOLD
            expected_modulation = None
            expected_baud = None
    
        # Check if signal is strong enough
        if signal_strength < threshold:
            return signals
    
        # If we have ML guidance, try expected protocol first
        if expected_modulation and expected_modulation in self.protocol_patterns:
            pattern = self.protocol_patterns[expected_modulation]
        
            # Override baud rate if we have learned value
            if expected_baud:
                pattern = pattern.copy()
                pattern['baud_rate'] = expected_baud
        
            decoded = self._try_decode_protocol(
                iq_samples, expected_modulation, pattern, frequency, signal_strength
            )
        
            if decoded:
                signals.append(decoded)
            
                # Learn from this successful decode
                if hasattr(self, 'learning_engine'):
                    self.learning_engine.learn_from_signal(
                        {
                            'frequency': frequency,
                            'power': signal_strength,
                            'snr': decoded.snr,
                            'modulation': expected_modulation,
                            'baud_rate': pattern['baud_rate'],
                            'characteristics': {}
                        },
                        decoded=True,
                        protocol=expected_modulation
                    )
            
                return signals
    
        # Try all known protocols (original behavior)
        for protocol_name, pattern in self.protocol_patterns.items():
            # Skip if we already tried this one
            if protocol_name == expected_modulation:
                continue
        
            decoded = self._try_decode_protocol(
                iq_samples, protocol_name, pattern, frequency, signal_strength
            )
        
            if decoded:
                signals.append(decoded)
            
                # Learn from this successful decode
                if hasattr(self, 'learning_engine'):
                    self.learning_engine.learn_from_signal(
                        {
                            'frequency': frequency,
                            'power': signal_strength,
                            'snr': decoded.snr,
                            'modulation': protocol_name,
                            'baud_rate': pattern['baud_rate'],
                            'characteristics': {}
                        },
                        decoded=True,
                        protocol=protocol_name
                    )
            
                return signals
    
        # If no known protocol matched, analyze as unknown
        if config.PROTOCOL_DETECTION_ENABLED:
            unknown = self._analyze_unknown_signal(iq_samples, frequency, signal_strength)
            if unknown:
                self.unknown_signals.append(unknown)
            
                # Learn from failed decode
                if hasattr(self, 'learning_engine'):
                    self.learning_engine.learn_from_signal(
                        {
                            'frequency': frequency,
                            'power': signal_strength,
                            'snr': 0,
                            'modulation': unknown.modulation_type,
                            'baud_rate': unknown.baud_rate or 0,
                            'characteristics': {}
                        },
                        decoded=False,
                        protocol=None
                    )
    
        return signals

    def _init_protocol_patterns(self):
        """Initialize known TPMS protocol patterns"""
        self.protocol_patterns = {
            'Toyota/Lexus': {
                'preamble': [0xAA, 0xAA],
                'packet_length': 10,
                'modulation': 'FSK',
                'baud_rate': 10000
            },
            'Schrader': {
                'preamble': [0x55, 0x55],
                'packet_length': 8,
                'modulation': 'Manchester',
                'baud_rate': 19200
            },
            'Continental': {
                'preamble': [0xFF, 0x00],
                'packet_length': 9,
                'modulation': 'FSK',
                'baud_rate': 9600
            }
        }
    
    def _try_decode_protocol(self, iq_samples: np.ndarray, protocol_name: str, 
                            pattern: dict, frequency: float, signal_strength: float) -> Optional[TPMSSignal]:
        """Attempt to decode signal with specific protocol"""
        try:
            # Demodulate based on modulation type
            if pattern['modulation'] == 'FSK':
                bits = self._demodulate_fsk(iq_samples, pattern['baud_rate'])
            elif pattern['modulation'] == 'Manchester':
                bits = self._demodulate_manchester(iq_samples, pattern['baud_rate'])
            else:
                return None
            
            if bits is None or len(bits) < pattern['packet_length'] * 8:
                return None
            
            # Look for preamble
            preamble_bits = self._bytes_to_bits(pattern['preamble'])
            preamble_pos = self._find_preamble(bits, preamble_bits)
            
            if preamble_pos == -1:
                return None
            
            # Extract packet
            packet_start = preamble_pos + len(preamble_bits)
            packet_bits = bits[packet_start:packet_start + pattern['packet_length'] * 8]
            
            if len(packet_bits) < pattern['packet_length'] * 8:
                return None
            
            # Convert to bytes
            packet_bytes = self._bits_to_bytes(packet_bits)
            
            # Decode packet based on protocol
            decoded = self._decode_packet(packet_bytes, protocol_name)
            
            if decoded:
                # Calculate SNR
                snr = self._calculate_snr(iq_samples)
                
                return TPMSSignal(
                    tpms_id=decoded['id'],
                    timestamp=time.time(),
                    frequency=frequency,
                    signal_strength=signal_strength,
                    snr=snr,
                    pressure_psi=decoded.get('pressure'),
                    temperature_c=decoded.get('temperature'),
                    battery_low=decoded.get('battery_low', False),
                    protocol=protocol_name,
                    raw_data=packet_bytes,
                    confidence=decoded.get('confidence', 0.8)
                )
        
        except Exception as e:
            print(f"Error decoding {protocol_name}: {e}")
            return None
    
    def _analyze_unknown_signal(self, iq_samples: np.ndarray, frequency: float, 
                                signal_strength: float) -> Optional[UnknownSignal]:
        """Analyze unknown signal characteristics"""
        try:
            # Detect modulation type
            modulation = self._detect_modulation(iq_samples)
            
            # Estimate baud rate
            baud_rate = self._estimate_baud_rate(iq_samples)
            
            # Estimate packet length
            packet_length = len(iq_samples) // (self.sample_rate // (baud_rate or 10000))
            
            # Create pattern signature
            pattern_sig = self._create_pattern_signature(iq_samples)
            
            return UnknownSignal(
                timestamp=time.time(),
                frequency=frequency,
                signal_strength=signal_strength,
                modulation_type=modulation,
                baud_rate=baud_rate,
                packet_length=packet_length,
                pattern_signature=pattern_sig,
                raw_samples=iq_samples[:1000]  # Store first 1000 samples
            )
        
        except Exception as e:
            print(f"Error analyzing unknown signal: {e}")
            return None
    
    def _detect_modulation(self, iq_samples: np.ndarray) -> str:
        """Detect modulation type"""
        # Check phase variations
        phase = np.angle(iq_samples)
        phase_diff = np.diff(phase)
        phase_var = np.var(phase_diff)
        
        # Check amplitude variations
        amplitude = np.abs(iq_samples)
        amp_var = np.var(amplitude)
        
        if phase_var > 0.5 and amp_var < 0.1:
            return "FSK/PSK"
        elif amp_var > 0.3:
            return "ASK/OOK"
        elif phase_var > 0.3:
            return "PSK"
        else:
            return "Unknown"
    
    def _estimate_baud_rate(self, iq_samples: np.ndarray) -> Optional[int]:
        """Estimate symbol/baud rate"""
        try:
            # Use autocorrelation to find symbol period
            amplitude = np.abs(iq_samples)
            autocorr = np.correlate(amplitude, amplitude, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find peaks
            peaks, _ = scipy_signal.find_peaks(autocorr, distance=10)
            
            if len(peaks) > 1:
                # Average distance between peaks
                avg_distance = np.mean(np.diff(peaks[:5]))
                baud_rate = int(self.sample_rate / avg_distance)
                return baud_rate
        
        except Exception:
            pass
        
        return None
    
    def _create_pattern_signature(self, iq_samples: np.ndarray) -> str:
        """Create a signature for pattern matching"""
        # Simplified signature based on amplitude envelope
        amplitude = np.abs(iq_samples[:100])
        # Normalize
        amplitude = (amplitude - np.min(amplitude)) / (np.max(amplitude) - np.min(amplitude) + 1e-10)
        # Quantize to 4 levels
        quantized = (amplitude * 3).astype(int)
        # Convert to string
        return ''.join(map(str, quantized))
    
    def _demodulate_fsk(self, iq_samples: np.ndarray, baud_rate: int) -> Optional[np.ndarray]:
        """Demodulate FSK signal"""
        # Simplified FSK demodulation
        instantaneous_freq = np.diff(np.unwrap(np.angle(iq_samples)))
        samples_per_bit = self.sample_rate // baud_rate
        
        bits = []
        for i in range(0, len(instantaneous_freq) - samples_per_bit, samples_per_bit):
            bit_sample = instantaneous_freq[i:i+samples_per_bit]
            bits.append(1 if np.mean(bit_sample) > 0 else 0)
        
        return np.array(bits) if bits else None
    
    def _demodulate_manchester(self, iq_samples: np.ndarray, baud_rate: int) -> Optional[np.ndarray]:
        """Demodulate Manchester encoded signal"""
        # Simplified Manchester demodulation
        amplitude = np.abs(iq_samples)
        samples_per_bit = self.sample_rate // baud_rate
        
        bits = []
        for i in range(0, len(amplitude) - samples_per_bit * 2, samples_per_bit * 2):
            first_half = np.mean(amplitude[i:i+samples_per_bit])
            second_half = np.mean(amplitude[i+samples_per_bit:i+samples_per_bit*2])
            
            if first_half > second_half:
                bits.append(1)
            else:
                bits.append(0)
        
        return np.array(bits) if bits else None
    
    def _bytes_to_bits(self, bytes_data: List[int]) -> np.ndarray:
        """Convert bytes to bit array"""
        bits = []
        for byte in bytes_data:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        return np.array(bits)
    
    def _bits_to_bytes(self, bits: np.ndarray) -> bytes:
        """Convert bit array to bytes"""
        bytes_data = []
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                if i + j < len(bits):
                    byte = (byte << 1) | int(bits[i + j])
            bytes_data.append(byte)
        return bytes(bytes_data)
    
    def _find_preamble(self, bits: np.ndarray, preamble: np.ndarray) -> int:
        """Find preamble in bit stream"""
        for i in range(len(bits) - len(preamble)):
            if np.array_equal(bits[i:i+len(preamble)], preamble):
                return i
        return -1
    
    def _decode_packet(self, packet: bytes, protocol: str) -> Optional[Dict]:
        """Decode packet based on protocol"""
        # Simplified decoding - in reality this would be protocol-specific
        if len(packet) < 4:
            return None
        
        # Extract ID (first 4 bytes typically)
        tpms_id = ''.join(f'{b:02X}' for b in packet[:4])
        
        # Mock pressure and temperature (would be protocol-specific)
        pressure = None
        temperature = None
        battery_low = False
        
        if len(packet) > 4:
            pressure = packet[4] * 0.5  # Example conversion
        if len(packet) > 5:
            temperature = packet[5] - 40  # Example conversion
        if len(packet) > 6:
            battery_low = bool(packet[6] & 0x80)
        
        return {
            'id': tpms_id,
            'pressure': pressure,
            'temperature': temperature,
            'battery_low': battery_low,
            'confidence': 0.7
        }
    
    def _calculate_snr(self, iq_samples: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio"""
        power = np.abs(iq_samples) ** 2
        signal_power = np.max(power)
        noise_power = np.median(power)
        snr = 10 * np.log10((signal_power / (noise_power + 1e-10)))
        return snr
    
    def get_unknown_signals(self, max_age: float = 60.0) -> List[UnknownSignal]:
        """Get recent unknown signals"""
        current_time = time.time()
        return [s for s in self.unknown_signals if current_time - s.timestamp < max_age]
    
    def get_protocol_statistics(self) -> Dict:
        """Get statistics on detected protocols"""
        recent_unknown = self.get_unknown_signals(300)  # Last 5 minutes
        
        modulation_counts = {}
        baud_rates = []
        
        for signal in recent_unknown:
            mod = signal.modulation_type
            modulation_counts[mod] = modulation_counts.get(mod, 0) + 1
            if signal.baud_rate:
                baud_rates.append(signal.baud_rate)
        
        return {
            'total_unknown': len(recent_unknown),
            'modulation_types': modulation_counts,
            'common_baud_rates': list(set(baud_rates)) if baud_rates else [],
            'avg_signal_strength': np.mean([s.signal_strength for s in recent_unknown]) if recent_unknown else 0
        }

