"""
TPMS Signal Decoder with Protocol Detection
Matching Maurader TPMSRX implementation
"""
import numpy as np
from scipy import signal as scipy_signal
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import time
from config import config
# At the top of tpms_decoder.py
import logging
import sys
from pathlib import Path

# Setup logging
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)

# Configure root logger to WARNING to suppress DEBUG from other libraries
logging.basicConfig(
    level=logging.WARNING,  # Changed from DEBUG
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
)

# Set our logger to INFO
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add file handler for our logger only
file_handler = logging.FileHandler(log_dir / "tpms_decoder.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Also log to console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

logger.info("TPMS Decoder initialized")

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
        self.learning_engine = None
        self._init_protocol_patterns()

    def _init_protocol_patterns(self):
        """Initialize known TPMS protocol patterns matching Maurader"""
        self.protocol_patterns = {
            'Schrader_FSK': {
                'preamble': [0x55, 0x55],  # Alternating pattern
                'packet_length': 10,
                'modulation': 'FSK',
                'symbol_rate': 19200,
                'deviation': 38400,
                'min_packet_bits': 64
            },
            'Schrader_OOK_8k192': {
                'preamble': [0xAA, 0xAA],
                'packet_length': 8,
                'modulation': 'OOK',
                'symbol_rate': 8192,
                'min_packet_bits': 64
            },
            'Schrader_OOK_8k4': {
                'preamble': [0xAA, 0xAA],
                'packet_length': 8,
                'modulation': 'OOK',
                'symbol_rate': 8400,
                'min_packet_bits': 64
            },
            'Toyota': {
                'preamble': [0x55, 0x55, 0x55],
                'packet_length': 10,
                'modulation': 'FSK',
                'symbol_rate': 10000,
                'deviation': 20000,
                'min_packet_bits': 80
            }
        }

    def set_learning_engine(self, learning_engine):
        """Set reference to learning engine for adaptive decoding"""
        self.learning_engine = learning_engine

    def process_samples(self, iq_samples: np.ndarray, frequency: float) -> List[TPMSSignal]:
        """Process IQ samples and decode TPMS signals"""
        signals = []

        # Calculate signal power and SNR
        power = np.abs(iq_samples) ** 2
        avg_power = np.mean(power)
        signal_strength = 10 * np.log10(avg_power + 1e-10)
        snr = self._calculate_snr(iq_samples)

        # Check if signal is strong enough
        if signal_strength < config.SIGNAL_THRESHOLD:
            return signals

        # Try each protocol in order of likelihood
        protocol_order = ['Schrader_FSK', 'Schrader_OOK_8k192', 'Schrader_OOK_8k4', 'Toyota']

        for protocol_name in protocol_order:
            pattern = self.protocol_patterns[protocol_name]
            decoded = self._try_decode_protocol(
                iq_samples, protocol_name, pattern, frequency, signal_strength, snr
            )

            if decoded:
                signals.append(decoded)
                # Only log successful decodes
                print(f"✅ {protocol_name}: {decoded.tpms_id} | "
                      f"{decoded.pressure_psi:.1f} PSI | "
                      f"{signal_strength:.1f} dBm", flush=True)

                # Learn from successful decode
                if self.learning_engine:
                    self.learning_engine.learn_from_signal(
                        {
                            'frequency': frequency,
                            'power': signal_strength,
                            'snr': snr,
                            'modulation': pattern['modulation'],
                            'baud_rate': pattern['symbol_rate'],
                            'characteristics': {}
                        },
                        decoded=True,
                        protocol=protocol_name
                    )

                return signals

            # If no protocol matched, only log occasionally
            if not hasattr(self, '_failed_count'):
                self._failed_count = 0
            self._failed_count += 1

            # Log every 100th failure
            if self._failed_count % 100 == 0:
                print(f"⚠️  {self._failed_count} signals failed to decode", flush=True)

            # Analyze as unknown (but don't spam logs)
            if config.PROTOCOL_DETECTION_ENABLED:
                unknown = self._analyze_unknown_signal(iq_samples, frequency, signal_strength)
                if unknown:
                    self.unknown_signals.append(unknown)

                    if self.learning_engine:
                        self.learning_engine.learn_from_signal(
                            {
                                'frequency': frequency,
                                'power': signal_strength,
                                'snr': snr,
                                'modulation': unknown.modulation_type,
                                'baud_rate': unknown.baud_rate or 0,
                                'characteristics': {}
                            },
                            decoded=False,
                            protocol=None
                        )

                return signals

    def _try_decode_protocol(self, iq_samples: np.ndarray, protocol_name: str,
                            pattern: dict, frequency: float, signal_strength: float,
                            snr: float) -> Optional[TPMSSignal]:
        """Attempt to decode signal with specific protocol"""
        try:
            # Demodulate based on modulation type
            if pattern['modulation'] == 'FSK':
                bits = self._demodulate_fsk(iq_samples, pattern['symbol_rate'],
                                           pattern.get('deviation', pattern['symbol_rate'] * 2))
            elif pattern['modulation'] == 'OOK':
                bits = self._demodulate_ook(iq_samples, pattern['symbol_rate'])
            else:
                return None

            if bits is None or len(bits) < pattern['min_packet_bits']:
                return None

            # Look for preamble
            preamble_bits = self._bytes_to_bits(pattern['preamble'])
            preamble_pos = self._find_preamble(bits, preamble_bits)

            if preamble_pos == -1:
                # Try with inverted bits
                bits = 1 - bits
                preamble_pos = self._find_preamble(bits, preamble_bits)
                if preamble_pos == -1:
                    return None

            # Extract packet
            packet_start = preamble_pos + len(preamble_bits)
            packet_bits = bits[packet_start:packet_start + pattern['packet_length'] * 8]

            if len(packet_bits) < pattern['min_packet_bits']:
                return None

            # Convert to bytes
            packet_bytes = self._bits_to_bytes(packet_bits)

            # Validate and decode packet
            if not self._validate_packet(packet_bytes, protocol_name):
                return None

            decoded = self._decode_packet(packet_bytes, protocol_name)

            if decoded:
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
            print(f"⚠️  Error decoding {protocol_name}: {e}")
            return None

    def _demodulate_fsk(self, iq_samples: np.ndarray, symbol_rate: int,
                       deviation: int) -> Optional[np.ndarray]:
        """
        FSK demodulation matching Maurader implementation
        Uses instantaneous frequency detection
        """
        if len(iq_samples) < 100:
            return None

        # Calculate instantaneous frequency from phase
        phase = np.angle(iq_samples)
        unwrapped_phase = np.unwrap(phase)
        inst_freq = np.diff(unwrapped_phase) * self.sample_rate / (2 * np.pi)

        # Smooth the frequency
        samples_per_symbol = int(self.sample_rate / symbol_rate)
        if samples_per_symbol < 1:
            samples_per_symbol = 1

        window_size = max(1, samples_per_symbol // 4)
        if window_size > 1:
            kernel = np.ones(window_size) / window_size
            inst_freq = np.convolve(inst_freq, kernel, mode='same')

        # Resample to symbol rate
        num_symbols = len(inst_freq) // samples_per_symbol
        if num_symbols < 10:
            return None

        resampled = np.zeros(num_symbols)
        for i in range(num_symbols):
            start = i * samples_per_symbol
            end = start + samples_per_symbol
            if end <= len(inst_freq):
                resampled[i] = np.mean(inst_freq[start:end])

        # Threshold detection (positive freq = 1, negative = 0)
        threshold = np.median(resampled)
        bits = (resampled > threshold).astype(int)

        return bits

    def _demodulate_ook(self, iq_samples: np.ndarray, symbol_rate: int) -> Optional[np.ndarray]:
        """
        OOK (On-Off Keying) demodulation matching Maurader
        Uses envelope detection
        """
        if len(iq_samples) < 100:
            return None

        # Calculate amplitude envelope
        amplitude = np.abs(iq_samples)

        # Smooth the amplitude
        samples_per_symbol = int(self.sample_rate / symbol_rate)
        if samples_per_symbol < 1:
            samples_per_symbol = 1

        window_size = max(1, samples_per_symbol // 2)
        if window_size > 1:
            kernel = np.ones(window_size) / window_size
            amplitude = np.convolve(amplitude, kernel, mode='same')

        # Resample to symbol rate
        num_symbols = len(amplitude) // samples_per_symbol
        if num_symbols < 10:
            return None

        resampled = np.zeros(num_symbols)
        for i in range(num_symbols):
            start = i * samples_per_symbol
            end = start + samples_per_symbol
            if end <= len(amplitude):
                resampled[i] = np.mean(amplitude[start:end])

        # Adaptive threshold (Otsu's method approximation)
        hist, bin_edges = np.histogram(resampled, bins=50)
        threshold = self._otsu_threshold(resampled, hist, bin_edges)

        bits = (resampled > threshold).astype(int)

        return bits

    def _otsu_threshold(self, data: np.ndarray, hist: np.ndarray,
                       bin_edges: np.ndarray) -> float:
        """Calculate optimal threshold using Otsu's method"""
        total = len(data)
        current_max = 0
        threshold = 0
        sum_total = np.sum(data)
        sum_background = 0
        weight_background = 0

        for i in range(len(hist)):
            weight_background += hist[i]
            if weight_background == 0:
                continue

            weight_foreground = total - weight_background
            if weight_foreground == 0:
                break

            sum_background += bin_edges[i] * hist[i]
            mean_background = sum_background / weight_background
            mean_foreground = (sum_total - sum_background) / weight_foreground

            variance_between = weight_background * weight_foreground * \
                             (mean_background - mean_foreground) ** 2

            if variance_between > current_max:
                current_max = variance_between
                threshold = bin_edges[i]

        return threshold

    def _validate_packet(self, packet: bytes, protocol: str) -> bool:
         """Validate packet structure and checksum"""
         if len(packet) < 4:
             return False

         # Basic validation: check if packet has reasonable values
         # Most TPMS IDs are non-zero and non-0xFF
         if packet[0] == 0x00 and packet[1] == 0x00:
             return False
         if packet[0] == 0xFF and packet[1] == 0xFF:
             return False

         # Check for pressure sanity (byte 4 in Schrader)
         if 'Schrader' in protocol and len(packet) >= 5:
             pressure_raw = packet[4]
             # Pressure byte should be in reasonable range
             # Typical: 20-60 (which converts to ~27-82 PSI)
             if pressure_raw < 10 or pressure_raw > 100:
                 return False

         return True

    def _decode_packet(self, packet: bytes, protocol: str) -> Optional[Dict]:
        """Decode packet based on protocol"""
        if len(packet) < 4:
            return None

        try:
            # Extract ID (first 4 bytes for most protocols)
            tpms_id = ''.join(f'{b:02X}' for b in packet[:4])

            pressure = None
            temperature = None
            battery_low = False

            # Schrader protocol decoding
            if 'Schrader' in protocol:
                if len(packet) >= 8:
                    # Pressure: byte 4-5, typically in kPa * 4
                    pressure_raw = packet[4]
                    if pressure_raw > 0 and pressure_raw < 255:
                        # To one of these (try each):
                        pressure = pressure_raw * 0.25      # Option 1: Quarter PSI
                        #pressure = (pressure_raw - 40) * 0.5  # Option 2: Offset and half PSI
                        #pressure = pressure_raw / 4.0       # Option 3: Divide by 4

                    # Temperature: byte 6, offset by 40°C
                    temp_raw = packet[5]
                    if temp_raw > 0 and temp_raw < 255:
                        temperature = temp_raw - 40

                    # Status flags: byte 7
                    if len(packet) > 6:
                        flags = packet[6]
                        battery_low = bool(flags & 0x80)

            # Toyota protocol decoding
            elif 'Toyota' in protocol:
                if len(packet) >= 10:
                    # Toyota uses different byte positions
                    pressure_raw = packet[6]
                    if pressure_raw > 0 and pressure_raw < 255:
                        pressure = pressure_raw * 0.25  # Different scaling

                    temp_raw = packet[7]
                    if temp_raw > 0 and temp_raw < 255:
                        temperature = temp_raw - 40

                    flags = packet[8]
                    battery_low = bool(flags & 0x40)

            return {
                'id': tpms_id,
                'pressure': pressure,
                'temperature': temperature,
                'battery_low': battery_low,
                'confidence': 0.85
            }

        except Exception as e:
            print(f"⚠️  Packet decode error: {e}")
            return None

    def _bytes_to_bits(self, bytes_data: List[int]) -> np.ndarray:
        """Convert bytes to bit array (MSB first)"""
        bits = []
        for byte in bytes_data:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        return np.array(bits)

    def _bits_to_bytes(self, bits: np.ndarray) -> bytes:
        """Convert bit array to bytes (MSB first)"""
        # Pad to multiple of 8
        remainder = len(bits) % 8
        if remainder != 0:
            bits = np.pad(bits, (0, 8 - remainder), 'constant')

        bytes_data = []
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                if i + j < len(bits):
                    byte = (byte << 1) | int(bits[i + j])
            bytes_data.append(byte)
        return bytes(bytes_data)

    def _find_preamble(self, bits: np.ndarray, preamble: np.ndarray,
                      max_errors: int = 2) -> int:
        """Find preamble in bit stream with error tolerance"""
        preamble_len = len(preamble)

        for i in range(len(bits) - preamble_len):
            errors = np.sum(bits[i:i+preamble_len] != preamble)
            if errors <= max_errors:
                return i

        return -1

    def _calculate_snr(self, iq_samples: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio"""
        power = np.abs(iq_samples) ** 2

        # Use top 10% as signal, bottom 50% as noise
        sorted_power = np.sort(power)
        signal_power = np.mean(sorted_power[-len(sorted_power)//10:])
        noise_power = np.mean(sorted_power[:len(sorted_power)//2])

        if noise_power == 0:
            return 0

        snr = 10 * np.log10(signal_power / noise_power)
        return max(0, snr)

    def _analyze_unknown_signal(self, iq_samples: np.ndarray, frequency: float,
                                signal_strength: float) -> Optional[UnknownSignal]:
        """Analyze unknown signal characteristics"""
        try:
            modulation = self._detect_modulation(iq_samples)
            baud_rate = self._estimate_baud_rate(iq_samples)
            packet_length = len(iq_samples) // (self.sample_rate // (baud_rate or 10000))
            pattern_sig = self._create_pattern_signature(iq_samples)

            return UnknownSignal(
                timestamp=time.time(),
                frequency=frequency,
                signal_strength=signal_strength,
                modulation_type=modulation,
                baud_rate=baud_rate,
                packet_length=packet_length,
                pattern_signature=pattern_sig,
                raw_samples=iq_samples[:1000]
            )

        except Exception as e:
            print(f"⚠️  Error analyzing unknown signal: {e}")
            return None

    def _detect_modulation(self, iq_samples: np.ndarray) -> str:
        """Detect modulation type"""
        phase = np.angle(iq_samples)
        phase_diff = np.diff(np.unwrap(phase))
        phase_var = np.var(phase_diff)

        amplitude = np.abs(iq_samples)
        amp_var = np.var(amplitude) / (np.mean(amplitude) + 1e-10)

        if amp_var > 0.3:
            return "OOK/ASK"
        elif phase_var > 0.5:
            return "FSK/PSK"
        else:
            return "Unknown"

    def _estimate_baud_rate(self, iq_samples: np.ndarray) -> Optional[int]:
        """Estimate symbol/baud rate using autocorrelation"""
        try:
            amplitude = np.abs(iq_samples)

            # Normalize
            amplitude = amplitude - np.mean(amplitude)

            # Autocorrelation
            autocorr = np.correlate(amplitude, amplitude, mode='full')
            autocorr = autocorr[len(autocorr)//2:]

            # Find first significant peak after zero lag
            threshold = 0.5 * np.max(autocorr[10:])
            peaks, _ = scipy_signal.find_peaks(autocorr[10:], height=threshold, distance=5)

            if len(peaks) > 0:
                # First peak indicates symbol period
                symbol_period = peaks[0] + 10
                baud_rate = int(self.sample_rate / symbol_period)

                # Round to common rates
                common_rates = [8192, 8400, 9600, 10000, 19200, 38400]
                closest = min(common_rates, key=lambda x: abs(x - baud_rate))

                if abs(closest - baud_rate) < baud_rate * 0.1:  # Within 10%
                    return closest

                return baud_rate

        except Exception:
            pass

        return None

    def _create_pattern_signature(self, iq_samples: np.ndarray) -> str:
        """Create a signature for pattern matching"""
        amplitude = np.abs(iq_samples[:100])
        amplitude = (amplitude - np.min(amplitude)) / (np.max(amplitude) - np.min(amplitude) + 1e-10)
        quantized = (amplitude * 3).astype(int)
        return ''.join(map(str, quantized))

    def get_unknown_signals(self, max_age: float = 60.0) -> List[UnknownSignal]:
        """Get recent unknown signals"""
        current_time = time.time()
        return [s for s in self.unknown_signals if current_time - s.timestamp < max_age]

    def get_protocol_statistics(self) -> Dict:
        """Get statistics on detected protocols"""
        recent_unknown = self.get_unknown_signals(300)

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
