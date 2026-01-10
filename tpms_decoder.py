"""
TPMS Signal Decoder with Intelligent Reprocessing and Protocol Discovery
Enhanced with adaptive learning, error correction, and self-evolution
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.signal import hilbert
from sklearn.cluster import DBSCAN
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import time
from config import config
import logging
import sys
from pathlib import Path
from collections import defaultdict
import json
import threading

# Setup logging
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    file_handler = logging.FileHandler(log_dir / "tpms_decoder.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

logger.info("TPMS Decoder initialized with intelligent reprocessing and self-learning")


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
    is_groomed: bool = False


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
    retry_count: int = 0
    discovered_protocol: Optional[str] = None
    # Enhanced features for learning
    deviation: float = 0.0
    kurtosis: float = 0.0
    center_freq_offset: float = 0.0


@dataclass
class DiscoveredProtocol:
    name: str
    modulation: str
    symbol_rate: int
    deviation: Optional[int]
    preamble: List[int]
    packet_length: int
    id_offset: int
    pressure_offset: Optional[int]
    pressure_scale: float
    temp_offset: Optional[int]
    temp_correction: int
    confidence: float
    sample_count: int


class IntelligentReprocessor:
    """Handles intelligent reprocessing of failed signals with forensic grooming"""

    def __init__(self, db_connection):
        self.db = db_connection
        self.retry_queue = []
        self.max_retries = 5
        self.retry_strategies = [
            self._retry_with_relaxed_thresholds,
            self._retry_with_inverted_bits,
            self._retry_with_phase_correction,
            self._retry_with_different_symbol_rates,
            self._retry_with_adaptive_filtering,
            self._retry_with_frequency_offset_correction,
            self._retry_with_bit_flip_correction
        ]
        self.lock = threading.Lock()

    def queue_for_reprocessing(self, unknown_signal: UnknownSignal):
        """Add signal to reprocessing queue"""
        with self.lock:
            if unknown_signal.retry_count < self.max_retries:
                self.retry_queue.append(unknown_signal)

                logger.info(
                    f"🔄 Queued signal for reprocessing: {unknown_signal.modulation_type} @ {unknown_signal.frequency / 1e6:.2f} MHz")

                # Save to database for persistence
                if self.db:
                    try:
                        signal_id = self.db.insert_unknown_signal({
                            'timestamp': unknown_signal.timestamp,
                            'frequency': unknown_signal.frequency,
                            'signal_strength': unknown_signal.signal_strength,
                            'snr': 0,
                            'modulation_type': unknown_signal.modulation_type,
                            'baud_rate': unknown_signal.baud_rate,
                            'packet_length': unknown_signal.packet_length,
                            'pattern_signature': unknown_signal.pattern_signature,
                            'raw_samples': unknown_signal.raw_samples.tobytes()
                        })
                        logger.info(f"✅ Saved unknown signal to database with ID: {signal_id}")
                    except Exception as e:
                        logger.error(f"❌ Error saving unknown signal: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                else:
                    logger.warning("⚠️  No database connection - unknown signal not saved")

    def process_queue(self, decoder) -> List[TPMSSignal]:
        """Process queued signals with different strategies"""
        decoded_signals = []

        with self.lock:
            remaining_queue = []

            for signal in self.retry_queue:
                strategy_idx = signal.retry_count % len(self.retry_strategies)
                strategy = self.retry_strategies[strategy_idx]

                result = strategy(signal, decoder)

                if result:
                    decoded_signals.append(result)
                    logger.info(f"🔄 Reprocessing success: {result.tpms_id} using strategy {strategy_idx + 1}")

                    if self.db:
                        try:
                            self.db.insert_reprocessing_result({
                                'unknown_signal_id': 0,
                                'strategy_used': strategy.__name__,
                                'retry_attempt': signal.retry_count,
                                'timestamp': time.time(),
                                'success': 1,
                                'protocol_found': result.protocol,
                                'tpms_id': result.tpms_id,
                                'confidence': result.confidence,
                                'notes': 'Groomed' if result.is_groomed else 'Standard'
                            })
                        except Exception as e:
                            logger.debug(f"Error saving reprocessing result: {e}")
                else:
                    signal.retry_count += 1
                    if signal.retry_count < self.max_retries:
                        remaining_queue.append(signal)

            self.retry_queue = remaining_queue

        return decoded_signals

    def _retry_with_relaxed_thresholds(self, signal: UnknownSignal, decoder) -> Optional[TPMSSignal]:
        """Retry with more permissive thresholds"""
        original_threshold = config.SIGNAL_THRESHOLD
        config.SIGNAL_THRESHOLD = original_threshold - 5
        result = decoder._reprocess_samples(signal.raw_samples, signal.frequency, relaxed=True)
        config.SIGNAL_THRESHOLD = original_threshold
        return result

    def _retry_with_inverted_bits(self, signal: UnknownSignal, decoder) -> Optional[TPMSSignal]:
        """Retry with bit inversion"""
        inverted = -signal.raw_samples
        return decoder._reprocess_samples(inverted, signal.frequency)

    def _retry_with_phase_correction(self, signal: UnknownSignal, decoder) -> Optional[TPMSSignal]:
        """Retry with phase correction"""
        phase_corrected = signal.raw_samples * np.exp(-1j * np.angle(signal.raw_samples[0]))
        return decoder._reprocess_samples(phase_corrected, signal.frequency)

    def _retry_with_different_symbol_rates(self, signal: UnknownSignal, decoder) -> Optional[TPMSSignal]:
        """Retry with alternative symbol rates"""
        alternative_rates = [8000, 8192, 8400, 9600, 10000, 19200, 20000, 38400]
        for rate in alternative_rates:
            if rate == signal.baud_rate:
                continue
            result = decoder._reprocess_with_symbol_rate(signal.raw_samples, signal.frequency, rate)
            if result:
                result.is_groomed = True
                return result
        return None

    def _retry_with_adaptive_filtering(self, signal: UnknownSignal, decoder) -> Optional[TPMSSignal]:
        """Retry with adaptive filtering to clean signal"""
        nyquist = decoder.sample_rate / 2
        low = max(1000, signal.frequency - 50000) / nyquist
        high = min(nyquist - 1000, signal.frequency + 50000) / nyquist
        try:
            b, a = scipy_signal.butter(4, [low, high], btype='band')
            filtered = scipy_signal.filtfilt(b, a, signal.raw_samples)
            result = decoder._reprocess_samples(filtered, signal.frequency)
            if result:
                result.is_groomed = True
            return result
        except:
            return None

    def _retry_with_frequency_offset_correction(self, signal: UnknownSignal, decoder) -> Optional[TPMSSignal]:
        """Retry with Doppler/LO drift correction"""
        offsets = [-10000, -5000, 5000, 10000, -2500, 2500]
        for offset in offsets:
            t = np.arange(len(signal.raw_samples)) / decoder.sample_rate
            shifted_iq = signal.raw_samples * np.exp(-1j * 2 * np.pi * offset * t)
            result = decoder._reprocess_samples(shifted_iq, signal.frequency)
            if result:
                result.is_groomed = True
                result.confidence *= 0.9
                logger.debug(f"Frequency offset correction: {offset} Hz")
                return result
        return None

    def _retry_with_bit_flip_correction(self, signal: UnknownSignal, decoder) -> Optional[TPMSSignal]:
        """Retry with single-bit error correction"""
        if signal.baud_rate and abs(signal.baud_rate - 19200) < 2000:
            bits = decoder._extract_bits_from_samples(signal.raw_samples, signal.baud_rate or 19200)
            if bits is not None and len(bits) > 64:
                for flip_pos in range(min(len(bits), 100)):
                    flipped_bits = bits.copy()
                    flipped_bits[flip_pos] = 1 - flipped_bits[flip_pos]
                    packet = decoder._bits_to_bytes(flipped_bits)
                    decoded = decoder._decode_packet(packet, 'Schrader_FSK')
                    if decoded:
                        result = TPMSSignal(
                            tpms_id=decoded['id'],
                            timestamp=time.time(),
                            frequency=signal.frequency,
                            signal_strength=signal.signal_strength,
                            snr=0,
                            pressure_psi=decoded.get('pressure'),
                            temperature_c=decoded.get('temperature'),
                            battery_low=decoded.get('battery_low', False),
                            protocol='Schrader_FSK',
                            raw_data=packet,
                            confidence=0.6,
                            is_groomed=True
                        )
                        logger.debug(f"Bit-flip correction: flipped bit {flip_pos}")
                        return result
        return None


class ProtocolDiscoveryEngine:
    """Discovers new TPMS protocols from unknown signals"""

    def __init__(self, db_connection):
        self.db = db_connection
        self.candidate_protocols = []
        self.pattern_clusters = defaultdict(list)
        self.feature_vectors = []
        self.min_samples_for_discovery = 10
        self.lock = threading.Lock()

    def analyze_unknown_signal(self, signal: UnknownSignal):
        """Analyze and cluster unknown signals"""
        with self.lock:
            self.pattern_clusters[signal.pattern_signature].append(signal)
            if signal.baud_rate and signal.deviation:
                self.feature_vectors.append({
                    'signal': signal,
                    'features': [signal.baud_rate, signal.deviation, signal.kurtosis]
                })
            if len(self.pattern_clusters[signal.pattern_signature]) >= self.min_samples_for_discovery:
                self._attempt_protocol_discovery(signal.pattern_signature)
            if len(self.feature_vectors) >= 50:
                self._attempt_physics_based_discovery()

    def _attempt_protocol_discovery(self, pattern_sig: str):
        """Attempt to discover protocol from clustered signals"""
        signals = self.pattern_clusters[pattern_sig]
        modulation = self._determine_modulation(signals)
        symbol_rate = self._determine_symbol_rate(signals)
        preamble = self._discover_preamble(signals)
        packet_structure = self._analyze_packet_structure(signals)

        if preamble and packet_structure:
            protocol = DiscoveredProtocol(
                name=f"Discovered_{modulation}_{symbol_rate}_{int(time.time())}",
                modulation=modulation,
                symbol_rate=symbol_rate,
                deviation=symbol_rate * 2 if modulation == 'FSK' else None,
                preamble=preamble,
                packet_length=packet_structure['length'],
                id_offset=packet_structure['id_offset'],
                pressure_offset=packet_structure.get('pressure_offset'),
                pressure_scale=packet_structure.get('pressure_scale', 0.25),
                temp_offset=packet_structure.get('temp_offset'),
                temp_correction=packet_structure.get('temp_correction', -40),
                confidence=0.6,
                sample_count=len(signals)
            )
            self.candidate_protocols.append(protocol)
            logger.info(f"🔍 Discovered potential protocol: {protocol.name} from {len(signals)} samples")
            self._save_discovered_protocol(protocol)
            del self.pattern_clusters[pattern_sig]

    def _attempt_physics_based_discovery(self):
        """Use DBSCAN clustering on signal physics"""
        try:
            data = np.array([fv['features'] for fv in self.feature_vectors])
            data_normalized = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-10)
            clustering = DBSCAN(eps=0.5, min_samples=10).fit(data_normalized)
            labels = clustering.labels_

            for k in set(labels):
                if k == -1:
                    continue
                cluster_mask = (labels == k)
                cluster_signals = [self.feature_vectors[i]['signal'] for i, mask in enumerate(cluster_mask) if mask]

                if len(cluster_signals) >= self.min_samples_for_discovery:
                    cluster_data = data[cluster_mask]
                    center_baud = int(cluster_data[:, 0].mean())
                    center_dev = int(cluster_data[:, 1].mean())
                    avg_kurtosis = cluster_data[:, 2].mean()
                    modulation = 'OOK' if avg_kurtosis > 1.0 else 'FSK'
                    protocol_name = f"AutoLearned_{modulation}_B{center_baud}_D{center_dev}"

                    logger.info(f"💡 PHYSICS DISCOVERY: {protocol_name} from {len(cluster_signals)} signals")

                    preamble = self._discover_preamble(cluster_signals)
                    packet_structure = self._analyze_packet_structure(cluster_signals)

                    if preamble and packet_structure:
                        protocol = DiscoveredProtocol(
                            name=protocol_name,
                            modulation=modulation,
                            symbol_rate=center_baud,
                            deviation=center_dev if modulation == 'FSK' else None,
                            preamble=preamble,
                            packet_length=packet_structure['length'],
                            id_offset=0,
                            pressure_offset=4,
                            pressure_scale=0.25,
                            temp_offset=5,
                            temp_correction=-40,
                            confidence=0.7,
                            sample_count=len(cluster_signals)
                        )
                        self.candidate_protocols.append(protocol)
                        self._save_discovered_protocol(protocol)

            self.feature_vectors = []
        except Exception as e:
            logger.error(f"Physics-based discovery error: {e}")

    def _determine_modulation(self, signals: List[UnknownSignal]) -> str:
        mod_counts = defaultdict(int)
        for sig in signals:
            mod_counts[sig.modulation_type] += 1
        return max(mod_counts.items(), key=lambda x: x[1])[0].split('/')[0]

    def _determine_symbol_rate(self, signals: List[UnknownSignal]) -> int:
        rates = [s.baud_rate for s in signals if s.baud_rate]
        return int(np.median(rates)) if rates else 10000

    def _discover_preamble(self, signals: List[UnknownSignal]) -> Optional[List[int]]:
        common_patterns = defaultdict(int)
        for sig in signals:
            amplitude = np.abs(sig.raw_samples[:200])
            threshold = np.median(amplitude)
            bits = (amplitude > threshold).astype(int)
            for length in [8, 16, 24]:
                pattern = tuple(bits[:length])
                common_patterns[pattern] += 1

        if common_patterns:
            most_common = max(common_patterns.items(), key=lambda x: x[1])
            if most_common[1] >= len(signals) * 0.5:
                bits = list(most_common[0])
                bytes_list = []
                for i in range(0, len(bits), 8):
                    byte = 0
                    for j in range(8):
                        if i + j < len(bits):
                            byte = (byte << 1) | bits[i + j]
                    bytes_list.append(byte)
                return bytes_list
        return None

    def _analyze_packet_structure(self, signals: List[UnknownSignal]) -> Optional[Dict]:
        avg_length = int(np.mean([s.packet_length for s in signals]))
        return {
            'length': max(8, avg_length),
            'id_offset': 0,
            'pressure_offset': 4,
            'pressure_scale': 0.25,
            'temp_offset': 5,
            'temp_correction': -40
        }

    def _save_discovered_protocol(self, protocol: DiscoveredProtocol):
        if not self.db:
            return
        try:
            self.db.save_discovered_protocol({
                'name': protocol.name,
                'modulation': protocol.modulation,
                'symbol_rate': protocol.symbol_rate,
                'deviation': protocol.deviation,
                'preamble': protocol.preamble,
                'packet_length': protocol.packet_length,
                'confidence': protocol.confidence,
                'sample_count': protocol.sample_count,
                'discovered_at': time.time(),
                'metadata': {
                    'id_offset': protocol.id_offset,
                    'pressure_offset': protocol.pressure_offset,
                    'pressure_scale': protocol.pressure_scale,
                    'temp_offset': protocol.temp_offset,
                    'temp_correction': protocol.temp_correction
                }
            })
        except Exception as e:
            logger.error(f"Error saving discovered protocol: {e}")

    def get_candidate_protocols(self) -> List[DiscoveredProtocol]:
        return [p for p in self.candidate_protocols if p.confidence > 0.5]


class TPMSDecoder:
    def __init__(self, sample_rate: int, db_connection=None):
        self.sample_rate = sample_rate
        self.unknown_signals = []
        self.protocol_patterns = {}
        self.learning_engine = None
        self._failed_count = 0
        self._success_count = 0
        self._reprocessing_success = 0
        self._discovery_success = 0

        # 🔥 FIX: Accept either a connection or database object
        if db_connection:
            # Check if it's a database object with methods we need
            if hasattr(db_connection, 'insert_unknown_signal'):
                # It's a database object
                self.db = db_connection
            else:
                # It's a connection object - wrap it or store for later
                self.db = db_connection
                logger.warning("Received connection object instead of database object - some features may not work")
        else:
            self.db = None

        # Initialize intelligent components
        self.reprocessor = IntelligentReprocessor(self.db) if self.db else None
        self.discovery_engine = ProtocolDiscoveryEngine(self.db) if self.db else None

        self._init_protocol_patterns()
        self._load_discovered_protocols(self.db)

        if self.db:
            self._start_evolution_loop()

    def _init_protocol_patterns(self):
        """Initialize known TPMS protocol patterns"""
        self.protocol_patterns = {
            'Schrader_FSK': {
                'preamble': [0x55, 0x55],
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

    def _load_discovered_protocols(self, db_connection):
        """Load previously discovered protocols from database"""
        if not db_connection:
            return
        try:
            protocols = db_connection.get_discovered_protocols(min_confidence=0.6)
            for protocol in protocols:
                self.protocol_patterns[protocol['name']] = {
                    'preamble': protocol['preamble'],
                    'packet_length': protocol['packet_length'],
                    'modulation': protocol['modulation'],
                    'symbol_rate': protocol['symbol_rate'],
                    'deviation': protocol.get('deviation', protocol['symbol_rate'] * 2),
                    'min_packet_bits': 64,
                    'discovered': True,
                    'confidence': protocol['confidence']
                }
                logger.info(f"📚 Loaded discovered protocol: {protocol['name']}")
        except Exception as e:
            logger.debug(f"No discovered protocols table or error loading: {e}")

    def _start_evolution_loop(self):
        """Start background thread for continuous learning AND reprocessing"""

        def evolution_worker():
            logger.info("🧠 Evolution engine started with reprocessing...")
            reprocess_counter = 0

            while True:
                try:
                    # 🔥 REPROCESS EVERY 30 SECONDS
                    if self.reprocessor and reprocess_counter % 3 == 0:  # Every 90 seconds
                        queue_size = len(self.reprocessor.retry_queue)
                        if queue_size > 0:
                            logger.info(f"🔄 Reprocessing {queue_size} queued signals...")
                            recovered = self.reprocessor.process_queue(self)
                            if recovered:
                                logger.info(f"✅ Recovered {len(recovered)} signals!")

                    # Protocol discovery (existing code)
                    if self.discovery_engine:
                        candidates = self.discovery_engine.get_candidate_protocols()
                        for candidate in candidates:
                            if candidate.name not in self.protocol_patterns:
                                self.protocol_patterns[candidate.name] = {
                                    'preamble': candidate.preamble,
                                    'packet_length': candidate.packet_length,
                                    'modulation': candidate.modulation,
                                    'symbol_rate': candidate.symbol_rate,
                                    'deviation': candidate.deviation,
                                    'min_packet_bits': 64,
                                    'discovered': True,
                                    'confidence': candidate.confidence
                                }
                                logger.info(f"🆕 Added new protocol to decoder: {candidate.name}")

                    reprocess_counter += 1
                    time.sleep(30)  # Check every 30 seconds

                except Exception as e:
                    logger.error(f"Evolution loop error: {e}")
                    time.sleep(60)

        evolution_thread = threading.Thread(target=evolution_worker, daemon=True)
        evolution_thread.start()

    def set_learning_engine(self, learning_engine):
        """Set reference to learning engine for adaptive decoding"""
        self.learning_engine = learning_engine

    def process_reprocessing_queue(self) -> List[TPMSSignal]:
        """Process queued signals for reprocessing"""
        if not self.reprocessor:
            return []

        decoded = self.reprocessor.process_queue(self)
        self._reprocessing_success += len(decoded)

        if decoded:
            logger.info(f"🔄 Reprocessing recovered {len(decoded)} signals (total: {self._reprocessing_success})")

        return decoded

    def process_samples(self, iq_samples: np.ndarray, frequency: float) -> List[TPMSSignal]:
        """
        The "Brain" of the operation.
        Flow: Known Proto -> Grooming -> Blind Discovery -> Database Update
        """
        signals: List[TPMSSignal] = []

        power = np.abs(iq_samples) ** 2
        avg_power = float(np.mean(power))
        signal_strength = 10 * np.log10(avg_power + 1e-10)
        snr = float(self._calculate_snr(iq_samples))

        if signal_strength < config.SIGNAL_THRESHOLD:
            return signals

        # Try known protocols first (Fast Path)
        protocol_order = list(self.protocol_patterns.keys())
        protocol_order.sort(key=lambda p: self.protocol_patterns[p].get('confidence', 0.5), reverse=True)

        decoded_any = None
        for protocol_name in protocol_order:
            pattern = self.protocol_patterns[protocol_name]
            decoded = self._try_decode_protocol(
                iq_samples, protocol_name, pattern, frequency, signal_strength, snr
            )
            if decoded:
                decoded_any = decoded
                signals.append(decoded)
                self._success_count += 1

                p = decoded.pressure_psi
                p_str = f"{p:.1f} PSI" if p is not None else "PSI=?"
                t = decoded.temperature_c
                t_str = f"{t:.1f}°C" if t is not None else "T=?"

                if self._success_count % 10 == 0:
                    logger.info(
                        f"✅ {protocol_name}: {decoded.tpms_id} | {p_str} | {t_str} | {signal_strength:.1f} dB | Total: {self._success_count}")

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
                break

        # If decoding failed, queue for intelligent reprocessing (Forensic Path)
        if decoded_any is None:
            self._failed_count += 1

            # 🔥 WRAP EVERYTHING IN TRY-EXCEPT TO CATCH THE ERROR
            try:
                if config.PROTOCOL_DETECTION_ENABLED:
                    # Blind feature extraction for deep analysis
                    unknown = self._blind_feature_extraction(iq_samples, frequency, signal_strength)

                    if unknown:
                        self.unknown_signals.append(unknown)

                        # Queue for reprocessing
                        if self.reprocessor:
                            try:
                                self.reprocessor.queue_for_reprocessing(unknown)
                            except Exception as e:
                                logger.error(f"Error queuing for reprocessing: {e}")

                        # Analyze for protocol discovery (Learning Path)
                        if self.discovery_engine:
                            try:
                                self.discovery_engine.analyze_unknown_signal(unknown)
                            except Exception as e:
                                logger.error(f"Error in protocol discovery: {e}")

                        if self.learning_engine:
                            try:
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
                            except Exception as e:
                                logger.error(f"Error in learning engine: {e}")

            except Exception as e:
                logger.error(f"Error processing unknown signal: {e}")
                import traceback
                logger.error(traceback.format_exc())

            if self._failed_count % 100 == 0:
                success_rate = self._success_count / (self._success_count + self._failed_count) * 100
                logger.info(
                    f"⚠️  {self._failed_count} signals failed | Success rate: {success_rate:.1f}% | Reprocessed: {self._reprocessing_success}")

        return signals

    def _blind_feature_extraction(self, iq_samples: np.ndarray, frequency: float,
                                  signal_strength: float) -> Optional[UnknownSignal]:
        """
        Extracts physics parameters without knowing the protocol.
        Uses Hilbert Transform for Instantaneous Frequency analysis.
        """
        try:
            # Analytic signal via Hilbert
            analytic = hilbert(iq_samples)
            inst_phase = np.unwrap(np.angle(analytic))
            inst_freq = np.diff(inst_phase) / (2.0 * np.pi) * self.sample_rate

            # 1. Estimate Deviation (FSK spread)
            hist, bin_edges = np.histogram(inst_freq, bins=50)
            peaks, _ = scipy_signal.find_peaks(hist, prominence=max(hist) * 0.2)
            if len(peaks) >= 2:
                freqs = bin_edges[peaks]
                deviation = (max(freqs) - min(freqs)) / 2
            else:
                deviation = 0

            # 2. Estimate Baud Rate (Pulse Width Analysis)
            if deviation > 5000:  # FSK
                bits = (inst_freq > 0).astype(int)
            else:  # OOK (Amplitude based)
                env = np.abs(analytic)
                threshold = np.mean(env)
                bits = (env > threshold).astype(int)

            # Count run lengths
            changes = np.diff(bits)
            change_indices = np.where(changes != 0)[0]
            if len(change_indices) > 5:
                run_lengths = np.diff(change_indices)
                median_run = np.median(run_lengths)
                baud_rate = int(self.sample_rate / median_run)
            else:
                baud_rate = None

            # 3. Modulation Classification via Kurtosis
            amp_kurtosis = scipy_signal.kurtosis(np.abs(iq_samples))
            mod_type = 'OOK' if amp_kurtosis > 1.0 else 'FSK'

            # 4. Create pattern signature
            pattern_sig = self._create_pattern_signature(iq_samples)

            # 5. Estimate packet length
            packet_length = len(iq_samples) // (self.sample_rate // (baud_rate or 10000))

            return UnknownSignal(
                timestamp=time.time(),
                frequency=frequency,
                signal_strength=signal_strength,
                modulation_type=mod_type,
                baud_rate=baud_rate,
                packet_length=packet_length,
                pattern_signature=pattern_sig,
                raw_samples=iq_samples,
                deviation=deviation,
                kurtosis=amp_kurtosis,
                center_freq_offset=0
            )

        except Exception as e:
            logger.debug(f"Error in blind feature extraction: {e}")
            return None

    def _extract_bits_from_samples(self, iq_samples: np.ndarray, symbol_rate: int) -> Optional[np.ndarray]:
        """Helper to extract bits for bit-flip correction"""
        try:
            # Try FSK first
            bits = self._demodulate_fsk(iq_samples, symbol_rate, symbol_rate * 2)
            if bits is not None:
                return bits
            # Try OOK
            bits = self._demodulate_ook(iq_samples, symbol_rate)
            return bits
        except:
            return None

    def reprocess_all_unknown_signals(self, db_connection, batch_size: int = 100) -> Dict:
        """
        Reprocess ALL unknown signals from the database

        Args:
            db_connection: Database connection
            batch_size: Number of signals to process at once

        Returns:
            Statistics dictionary with results
        """
        if not self.reprocessor:
            logger.warning("No reprocessor available")
            return {'error': 'No reprocessor'}

        stats = {
            'total_processed': 0,
            'total_recovered': 0,
            'by_strategy': {},
            'start_time': time.time()
        }

        logger.info("🔄 Starting comprehensive reprocessing of ALL unknown signals...")

        # Get all unknown signals (no retry limit for initial reprocessing)
        try:
            cursor = db_connection.cursor()
            cursor.execute("""
                           SELECT id,
                                  timestamp,
                                  frequency,
                                  signal_strength,
                                  snr,
                                  modulation_type,
                                  baud_rate,
                                  packet_length,
                                  pattern_signature,
                                  raw_samples,
                                  retry_count
                           FROM unknown_signals
                           WHERE decoded = 0
                           ORDER BY timestamp DESC
                           """)

            all_unknown = cursor.fetchall()
            total_count = len(all_unknown)

            logger.info(f"📊 Found {total_count} unknown signals to reprocess")

            if total_count == 0:
                return stats

            # Process in batches
            batch_num = 0
            for i in range(0, total_count, batch_size):
                batch = all_unknown[i:i + batch_size]
                batch_num += 1

                logger.info(
                    f"🔄 Processing batch {batch_num}/{(total_count + batch_size - 1) // batch_size} ({len(batch)} signals)")

                for row in batch:
                    signal_id, timestamp, freq, strength, snr, mod_type, baud, pkt_len, pattern, raw_bytes, retry_count = row

                    try:
                        # Reconstruct IQ samples
                        if raw_bytes:
                            raw_samples = np.frombuffer(raw_bytes, dtype=np.complex64)
                        else:
                            logger.debug(f"Signal {signal_id} has no raw samples, skipping")
                            continue

                        # Try all strategies
                        recovered = False
                        for strategy_idx, strategy in enumerate(self.reprocessor.retry_strategies):
                            unknown_signal = UnknownSignal(
                                timestamp=timestamp,
                                frequency=freq,
                                signal_strength=strength,
                                modulation_type=mod_type or 'Unknown',
                                baud_rate=baud,
                                packet_length=pkt_len or 0,
                                pattern_signature=pattern or '',
                                raw_samples=raw_samples,
                                retry_count=retry_count or 0
                            )

                            result = strategy(unknown_signal, self)

                            if result:
                                # Success!
                                recovered = True
                                strategy_name = strategy.__name__
                                stats['by_strategy'][strategy_name] = stats['by_strategy'].get(strategy_name, 0) + 1
                                stats['total_recovered'] += 1

                                logger.info(f"✅ Recovered signal {signal_id}: {result.tpms_id} using {strategy_name}")

                                # Save recovered signal to main database
                                cursor.execute("""
                                               INSERT INTO tpms_signals
                                               (tpms_id, timestamp, frequency, signal_strength, snr,
                                                pressure_psi, temperature_c, battery_low, protocol,
                                                raw_data, confidence, reprocessed, reprocess_count)
                                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                                               """, (
                                    result.tpms_id,
                                    result.timestamp,
                                    result.frequency,
                                    result.signal_strength,
                                    result.snr,
                                    result.pressure_psi,
                                    result.temperature_c,
                                    result.battery_low,
                                    result.protocol,
                                    result.raw_data,
                                    result.confidence,
                                    retry_count + 1
                                               ))

                                # Mark as decoded in unknown_signals
                                cursor.execute("""
                                               UPDATE unknown_signals
                                               SET decoded          = 1,
                                                   decoded_protocol = ?,
                                                   decoded_at       = ?
                                               WHERE id = ?
                                               """, (result.protocol, time.time(), signal_id))

                                # Log reprocessing result
                                cursor.execute("""
                                               INSERT INTO reprocessing_results
                                               (unknown_signal_id, strategy_used, retry_attempt, timestamp,
                                                success, protocol_found, tpms_id, confidence, notes)
                                               VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?)
                                               """, (
                                                   signal_id,
                                                   strategy_name,
                                                   retry_count + 1,
                                                   time.time(),
                                                   result.protocol,
                                                   result.tpms_id,
                                                   result.confidence,
                                                   'Comprehensive reprocessing'
                                               ))

                                db_connection.commit()
                                break  # Stop trying other strategies

                        if not recovered:
                            # Update retry count
                            cursor.execute("""
                                           UPDATE unknown_signals
                                           SET retry_count = retry_count + 1,
                                               last_retry  = ?
                                           WHERE id = ?
                                           """, (time.time(), signal_id))

                        stats['total_processed'] += 1

                    except Exception as e:
                        logger.error(f"Error reprocessing signal {signal_id}: {e}")
                        continue

                # Commit after each batch
                db_connection.commit()

                # Progress update
                progress = (i + len(batch)) / total_count * 100
                logger.info(f"📈 Progress: {progress:.1f}% ({i + len(batch)}/{total_count})")

            stats['end_time'] = time.time()
            stats['duration_seconds'] = stats['end_time'] - stats['start_time']
            stats['success_rate'] = (stats['total_recovered'] / stats['total_processed'] * 100) if stats[
                                                                                                       'total_processed'] > 0 else 0

            logger.info(f"""
    🎉 Comprehensive Reprocessing Complete!
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    📊 Total Processed: {stats['total_processed']}
    ✅ Total Recovered: {stats['total_recovered']}
    📈 Success Rate: {stats['success_rate']:.1f}%
    ⏱️  Duration: {stats['duration_seconds']:.1f} seconds
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Strategy Breakdown:
    """)

            for strategy, count in sorted(stats['by_strategy'].items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {strategy}: {count} signals")

            return stats

        except Exception as e:
            logger.error(f"Error in comprehensive reprocessing: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'error': str(e)}

    def _reprocess_samples(self, iq_samples: np.ndarray, frequency: float, relaxed: bool = False) -> Optional[
        TPMSSignal]:
        """Reprocess samples with current protocols"""
        power = np.abs(iq_samples) ** 2
        avg_power = float(np.mean(power))
        signal_strength = 10 * np.log10(avg_power + 1e-10)
        snr = float(self._calculate_snr(iq_samples))

        for protocol_name, pattern in self.protocol_patterns.items():
            decoded = self._try_decode_protocol(
                iq_samples, protocol_name, pattern, frequency, signal_strength, snr
            )
            if decoded:
                return decoded

        return None

    def _reprocess_with_symbol_rate(self, iq_samples: np.ndarray, frequency: float, symbol_rate: int) -> Optional[
        TPMSSignal]:
        """Reprocess with specific symbol rate"""
        temp_pattern = {
            'preamble': [0x55, 0x55],
            'packet_length': 10,
            'modulation': 'FSK',
            'symbol_rate': symbol_rate,
            'deviation': symbol_rate * 2,
            'min_packet_bits': 64
        }

        power = np.abs(iq_samples) ** 2
        avg_power = float(np.mean(power))
        signal_strength = 10 * np.log10(avg_power + 1e-10)
        snr = float(self._calculate_snr(iq_samples))

        return self._try_decode_protocol(
            iq_samples, f"Adaptive_{symbol_rate}", temp_pattern, frequency, signal_strength, snr
        )

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

            # Debug logging
            if self._success_count % 100 == 0:
                debug_info = self.debug_packet_decode(packet_bytes, protocol_name)
                logger.info(f"📦 Packet debug: {debug_info}")

            # Validate packet structure
            if not self._validate_packet_structure(packet_bytes, protocol_name):
                return None

            # Decode packet
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
                    confidence=decoded.get('confidence', 0.8),
                    is_groomed=False
                )

        except Exception as e:
            logger.debug(f"Error decoding {protocol_name}: {e}")
            return None

    def _validate_packet_structure(self, packet: bytes, protocol: str) -> bool:
        """Validate basic packet structure"""
        if len(packet) < 4:
            return False
        if all(b == 0x00 for b in packet[:4]):
            return False
        if all(b == 0xFF for b in packet[:4]):
            return False
        return True

    def _decode_schrader_pressure_psi(self, pressure_raw: int) -> Optional[float]:
        """Decode Schrader pressure with permissive range"""
        if pressure_raw == 0 or pressure_raw == 255:
            return None
        return float(pressure_raw) * 0.25

    def _decode_packet(self, packet: bytes, protocol: str) -> Optional[Dict]:
        """Decode packet based on protocol - extract whatever data is available"""
        if len(packet) < 4:
            return None

        try:
            # Extract ID (first 4 bytes for most protocols)
            tpms_id = ''.join(f'{b:02X}' for b in packet[:4])

            pressure = None
            temperature = None
            battery_low = False
            confidence = 0.7

            # Schrader protocol decoding
            if 'Schrader' in protocol:
                if len(packet) >= 5:
                    pressure_raw = packet[4]
                    pressure = self._decode_schrader_pressure_psi(pressure_raw)
                    if pressure is not None and 10.0 <= pressure <= 80.0:
                        confidence += 0.1
                    else:
                        pressure = None

                if len(packet) >= 6:
                    temp_raw = packet[5]
                    if 20 < temp_raw < 120:
                        temperature = temp_raw - 40
                        confidence += 0.1
                    else:
                        temperature = None

                if len(packet) >= 7:
                    flags = packet[6]
                    battery_low = bool(flags & 0x80)

            # Toyota protocol decoding
            elif 'Toyota' in protocol:
                if len(packet) >= 7:
                    pressure_raw = packet[6]
                    if 0 < pressure_raw < 255:
                        pressure = pressure_raw * 0.25
                        if not (10.0 <= pressure <= 80.0):
                            pressure = None
                        else:
                            confidence += 0.1

                if len(packet) >= 8:
                    temp_raw = packet[7]
                    if 20 < temp_raw < 120:
                        temperature = temp_raw - 40
                        confidence += 0.1
                    else:
                        temperature = None

                if len(packet) >= 9:
                    flags = packet[8]
                    battery_low = bool(flags & 0x40)

            # Discovered protocols - use generic decoding
            elif 'Discovered' in protocol or 'AutoLearned' in protocol:
                if len(packet) >= 5:
                    pressure_raw = packet[4]
                    if 0 < pressure_raw < 255:
                        pressure = pressure_raw * 0.25
                        if 10.0 <= pressure <= 80.0:
                            confidence += 0.1
                        else:
                            pressure = None

                if len(packet) >= 6:
                    temp_raw = packet[5]
                    if 20 < temp_raw < 120:
                        temperature = temp_raw - 40
                        confidence += 0.1
                    else:
                        temperature = None

            # Don't return packets with no valid data
            if pressure is None and temperature is None:
                return None

            return {
                'id': tpms_id,
                'pressure': pressure,
                'temperature': temperature,
                'battery_low': battery_low,
                'confidence': min(confidence, 1.0)
            }

        except Exception as e:
            logger.debug(f"Packet decode error: {e}")
            return None

    def debug_packet_decode(self, packet: bytes, protocol: str) -> Dict:
        """Debug helper to see raw packet bytes and decoded values"""
        result = {
            'raw_hex': ' '.join(f'{b:02X}' for b in packet),
            'raw_bytes': list(packet),
            'length': len(packet),
            'protocol': protocol
        }

        if 'Schrader' in protocol and len(packet) >= 7:
            result['id_hex'] = ''.join(f'{b:02X}' for b in packet[:4])
            result['pressure_raw'] = packet[4]
            result['pressure_psi'] = packet[4] * 0.25
            result['temp_byte5_raw'] = packet[5]
            result['temp_byte5_minus40'] = packet[5] - 40
            result['temp_byte5_minus50'] = packet[5] - 50

            if len(packet) >= 8:
                result['temp_byte6_raw'] = packet[6]
                result['temp_byte6_minus40'] = packet[6] - 40

            result['flags'] = f'{packet[6]:08b}' if len(packet) > 6 else 'N/A'

        return result

    def _demodulate_fsk(self, iq_samples: np.ndarray, symbol_rate: int,
                        deviation: int) -> Optional[np.ndarray]:
        """FSK demodulation using instantaneous frequency"""
        if len(iq_samples) < 100:
            return None

        try:
            phase = np.angle(iq_samples)
            unwrapped_phase = np.unwrap(phase)
            inst_freq = np.diff(unwrapped_phase) * self.sample_rate / (2 * np.pi)

            samples_per_symbol = int(self.sample_rate / symbol_rate)
            if samples_per_symbol < 1:
                samples_per_symbol = 1

            window_size = max(1, samples_per_symbol // 4)
            if window_size > 1:
                kernel = np.ones(window_size) / window_size
                inst_freq = np.convolve(inst_freq, kernel, mode='same')

            num_symbols = len(inst_freq) // samples_per_symbol
            if num_symbols < 10:
                return None

            resampled = np.zeros(num_symbols)
            for i in range(num_symbols):
                start = i * samples_per_symbol
                end = start + samples_per_symbol
                if end <= len(inst_freq):
                    resampled[i] = np.mean(inst_freq[start:end])

            threshold = np.median(resampled)
            bits = (resampled > threshold).astype(int)

            return bits

        except Exception as e:
            logger.debug(f"FSK demodulation error: {e}")
            return None

    def _demodulate_ook(self, iq_samples: np.ndarray, symbol_rate: int) -> Optional[np.ndarray]:
        """OOK (On-Off Keying) demodulation"""
        if len(iq_samples) < 100:
            return None

        try:
            amplitude = np.abs(iq_samples)

            samples_per_symbol = int(self.sample_rate / symbol_rate)
            if samples_per_symbol < 1:
                samples_per_symbol = 1

            window_size = max(1, samples_per_symbol // 2)
            if window_size > 1:
                kernel = np.ones(window_size) / window_size
                amplitude = np.convolve(amplitude, kernel, mode='same')

            num_symbols = len(amplitude) // samples_per_symbol
            if num_symbols < 10:
                return None

            resampled = np.zeros(num_symbols)
            for i in range(num_symbols):
                start = i * samples_per_symbol
                end = start + samples_per_symbol
                if end <= len(amplitude):
                    resampled[i] = np.mean(amplitude[start:end])

            hist, bin_edges = np.histogram(resampled, bins=50)
            threshold = self._otsu_threshold(resampled, hist, bin_edges)

            bits = (resampled > threshold).astype(int)

            return bits

        except Exception as e:
            logger.debug(f"OOK demodulation error: {e}")
            return None

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

    def _bytes_to_bits(self, bytes_data: List[int]) -> np.ndarray:
        """Convert bytes to bit array (MSB first)"""
        bits = []
        for byte in bytes_data:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        return np.array(bits)

    def _bits_to_bytes(self, bits: np.ndarray) -> bytes:
        """Convert bit array to bytes (MSB first)"""
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
            errors = np.sum(bits[i:i + preamble_len] != preamble)
            if errors <= max_errors:
                return i

        return -1

    def _calculate_snr(self, iq_samples: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio"""
        try:
            power = np.abs(iq_samples) ** 2

            sorted_power = np.sort(power)
            signal_power = np.mean(sorted_power[-len(sorted_power) // 10:])
            noise_power = np.mean(sorted_power[:len(sorted_power) // 2])

            if noise_power == 0:
                return 0

            snr = 10 * np.log10(signal_power / noise_power)
            return max(0, snr)

        except Exception as e:
            logger.debug(f"SNR calculation error: {e}")
            return 0

    def _analyze_unknown_signal(self, iq_samples: np.ndarray, frequency: float,
                                signal_strength: float) -> Optional[UnknownSignal]:
        """Analyze unknown signal characteristics (legacy fallback)"""
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
                raw_samples=iq_samples[:10000]
            )

        except Exception as e:
            logger.debug(f"Error analyzing unknown signal: {e}")
            return None

    def _detect_modulation(self, iq_samples: np.ndarray) -> str:
        """Detect modulation type"""
        try:
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

        except Exception as e:
            logger.debug(f"Modulation detection error: {e}")
            return "Unknown"

    def _estimate_baud_rate(self, iq_samples: np.ndarray) -> Optional[int]:
        """Estimate symbol/baud rate using autocorrelation"""
        try:
            amplitude = np.abs(iq_samples)
            amplitude = amplitude - np.mean(amplitude)

            autocorr = np.correlate(amplitude, amplitude, mode='full')
            autocorr = autocorr[len(autocorr) // 2:]

            threshold = 0.5 * np.max(autocorr[10:])
            peaks, _ = scipy_signal.find_peaks(autocorr[10:], height=threshold, distance=5)

            if len(peaks) > 0:
                symbol_period = peaks[0] + 10
                baud_rate = int(self.sample_rate / symbol_period)

                common_rates = [8192, 8400, 9600, 10000, 19200, 38400]
                closest = min(common_rates, key=lambda x: abs(x - baud_rate))

                if abs(closest - baud_rate) < baud_rate * 0.1:
                    return closest

                return baud_rate

        except Exception:
            pass

        return None

    def _create_pattern_signature(self, iq_samples: np.ndarray) -> str:
        """Create a signature for pattern matching"""
        try:
            amplitude = np.abs(iq_samples[:100])
            amplitude = (amplitude - np.min(amplitude)) / (np.max(amplitude) - np.min(amplitude) + 1e-10)
            quantized = (amplitude * 3).astype(int)
            return ''.join(map(str, quantized))
        except Exception:
            return ""

    def get_unknown_signals(self, max_age: float = 60.0) -> List[UnknownSignal]:
        """Get recent unknown signals"""
        current_time = time.time()
        return [s for s in self.unknown_signals if current_time - s.timestamp < max_age]

    def get_statistics(self) -> Dict:
        """Get comprehensive decoder statistics"""
        stats = self.get_protocol_statistics()
        stats['reprocessing_success'] = self._reprocessing_success
        stats['discovery_count'] = len(self.discovery_engine.candidate_protocols) if self.discovery_engine else 0
        stats['queued_for_retry'] = len(self.reprocessor.retry_queue) if self.reprocessor else 0

        return stats

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
            'total_successful': self._success_count,
            'total_failed': self._failed_count,
            'success_rate': self._success_count / (self._success_count + self._failed_count) if (
                                                                                                            self._success_count + self._failed_count) > 0 else 0,
            'modulation_types': modulation_counts,
            'common_baud_rates': list(set(baud_rates)) if baud_rates else [],
            'avg_signal_strength': np.mean([s.signal_strength for s in recent_unknown]) if recent_unknown else 0
        }


# --- FUNCTIONS OUTSIDE CLASS (to avoid forward reference errors) ---

def start_evolution_worker(db_connection, decoder: 'TPMSDecoder', check_interval: int = 60):
    """
    Optional: Start a separate evolution worker thread for continuous learning
    This can run independently to analyze accumulated unknown signals
    """

    def worker():
        logger.info("🧠 Standalone evolution worker started...")
        while True:
            try:
                if hasattr(db_connection, 'get_unknown_signals_for_reprocessing'):
                    unknown_signals = db_connection.get_unknown_signals_for_reprocessing(
                        max_retries=5,
                        limit=100
                    )

                    if unknown_signals:
                        logger.info(f"Evolution worker processing {len(unknown_signals)} signals...")

                        for sig_data in unknown_signals:
                            unknown = UnknownSignal(
                                timestamp=sig_data['timestamp'],
                                frequency=sig_data['frequency'],
                                signal_strength=sig_data['signal_strength'],
                                modulation_type=sig_data.get('modulation_type', 'Unknown'),
                                baud_rate=sig_data.get('baud_rate'),
                                packet_length=sig_data.get('packet_length', 0),
                                pattern_signature=sig_data.get('pattern_signature', ''),
                                raw_samples=np.frombuffer(sig_data['raw_samples'], dtype=np.complex64) if sig_data.get(
                                    'raw_samples') else np.array([]),
                                retry_count=sig_data.get('retry_count', 0)
                            )

                            if decoder.discovery_engine:
                                decoder.discovery_engine.analyze_unknown_signal(unknown)

                if decoder.discovery_engine:
                    candidates = decoder.discovery_engine.get_candidate_protocols()
                    for candidate in candidates:
                        if candidate.name not in decoder.protocol_patterns:
                            decoder.protocol_patterns[candidate.name] = {
                                'preamble': candidate.preamble,
                                'packet_length': candidate.packet_length,
                                'modulation': candidate.modulation,
                                'symbol_rate': candidate.symbol_rate,
                                'deviation': candidate.deviation,
                                'min_packet_bits': 64,
                                'discovered': True,
                                'confidence': candidate.confidence
                            }
                            logger.info(f"🆕 Evolution worker added protocol: {candidate.name}")

                time.sleep(check_interval)

            except Exception as e:
                logger.error(f"Evolution worker error: {e}")
                time.sleep(check_interval)

    evolution_thread = threading.Thread(target=worker, daemon=True)
    evolution_thread.start()
    return evolution_thread


def create_decoder_with_learning(sample_rate: int, db_path: str) -> 'TPMSDecoder':
    """
    Factory function to create a fully-configured decoder with learning enabled
    """
    from database import TPMSDatabase

    db = TPMSDatabase(db_path)
    decoder = TPMSDecoder(sample_rate, db._connect())

    logger.info(f"Created intelligent decoder with learning (sample_rate={sample_rate})")
    logger.info(f"Loaded {len(decoder.protocol_patterns)} protocols")

    return decoder


if __name__ == "__main__":
    # Demo/Test code
    logger.info("TPMS Intelligent Decoder - Standalone Test")

    # Create test decoder
    decoder = TPMSDecoder(2000000)

    # Generate fake IQ samples
    fake_iq = np.random.randn(20000) + 1j * np.random.randn(20000)

    # Process
    signals = decoder.process_samples(fake_iq, 315e6)

    # Print stats
    stats = decoder.get_statistics()
    logger.info(f"Statistics: {json.dumps(stats, indent=2)}")

    logger.info("Test complete - decoder is ready for integration")
