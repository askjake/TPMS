"""
RTL-SDR Interface for TPMS Detection
Mirrors the HackRF interface API for drop-in compatibility
"""

import logging
import time
import numpy as np
from typing import Callable, Optional
import threading
import queue

logger = logging.getLogger(__name__)

# Try to import RTL-SDR library
try:
    from rtlsdr import RtlSdr

    RTLSDR_AVAILABLE = True
except ImportError:
    RTLSDR_AVAILABLE = False
    RtlSdr = None
    logger.warning("RTL-SDR library not available. Install with: pip install pyrtlsdr")


class RTLSDRInterface:
    """RTL-SDR interface compatible with HackRF interface API"""

    def __init__(self, device_index: int = 0):
        logger.info("RTLSDRInterface.__init__ called")

        # Initialize all attributes first (prevents __del__ errors)
        self.device = None
        self.is_running = False
        self.callback = None
        self.device_index = device_index
        self.frequency = 314.9e6
        self.sample_rate = 2_400_000
        self.gain = 40

        # Pipeline for decode→DB
        self._pipeline_db = None
        self._pipeline_decoder = None
        self._pipeline_on_signal = None
        self._pipeline_q: "queue.Queue[tuple]" = queue.Queue(maxsize=200)
        self._pipeline_stop = threading.Event()
        self._pipeline_thread: Optional[threading.Thread] = None

        # RX thread
        self._rx_thread: Optional[threading.Thread] = None
        self._rx_stop = threading.Event()

        if not RTLSDR_AVAILABLE:
            logger.warning("RTL-SDR library not available - simulation mode")
            return

        # Try to open device
        try:
            self.device = RtlSdr(device_index=self.device_index)
            logger.info("✅ RTL-SDR device opened successfully")
            self._configure_device()
        except Exception as e:
            logger.error(f"❌ Error initializing RTL-SDR: {e}")
            self.device = None

    def _configure_device(self):
        """Configure device with default settings"""
        if not self.device:
            return

        try:
            self.device.sample_rate = self.sample_rate
            self.device.center_freq = int(self.frequency)
            self.device.gain = self.gain
            logger.info(f"RTL-SDR configured: {self.sample_rate / 1e6:.2f} MHz sample rate, "
                        f"{self.frequency / 1e6:.2f} MHz center freq, {self.gain} dB gain")
        except Exception as e:
            logger.error(f"Configuration error: {e}")

    def attach_pipeline(self, db, decoder, on_signal=None):
        """Attach TPMS decode→DB pipeline"""
        self._pipeline_db = db
        self._pipeline_decoder = decoder
        self._pipeline_on_signal = on_signal
        logger.info("Pipeline attached to RTL-SDR interface")
        return True

    def _pipeline_worker(self):
        """Worker thread for decode→DB pipeline"""
        while not self._pipeline_stop.is_set():
            try:
                iq_complex, rssi, freq_hz, ts = self._pipeline_q.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                dec = self._pipeline_decoder
                db = self._pipeline_db
                if dec is None or db is None:
                    continue

                signals = dec.process_samples(iq_complex, freq_hz)
                if not signals:
                    continue

                rows = []
                now_ts = ts or time.time()
                for s in signals:
                    try:
                        s.signal_strength = rssi
                    except Exception:
                        pass

                    rows.append({
                        "tpms_id": getattr(s, "tpms_id", None),
                        "timestamp": getattr(s, "timestamp", None) or now_ts,
                        "frequency": getattr(s, "frequency", None) or freq_hz,
                        "signal_strength": getattr(s, "signal_strength", None) or rssi,
                        "snr": getattr(s, "snr", None),
                        "pressure_psi": getattr(s, "pressure_psi", None),
                        "temperature_c": getattr(s, "temperature_c", None),
                        "battery_low": getattr(s, "battery_low", None),
                        "protocol": getattr(s, "protocol", None),
                        "raw_data": getattr(s, "raw_data", None),
                    })

                # Insert batch
                if hasattr(db, "insert_signals_batch"):
                    db.insert_signals_batch(rows)
                else:
                    for r in rows:
                        db.insert_signal(r)

                # Optional hook
                if self._pipeline_on_signal:
                    for r in rows:
                        try:
                            self._pipeline_on_signal(r)
                        except Exception:
                            pass

            except Exception as e:
                logger.error(f"Pipeline worker error: {e}")

    def _start_pipeline(self):
        """Start pipeline worker thread"""
        if self._pipeline_thread and self._pipeline_thread.is_alive():
            return
        self._pipeline_stop.clear()
        self._pipeline_thread = threading.Thread(target=self._pipeline_worker, daemon=True)
        self._pipeline_thread.start()
        logger.info("Pipeline worker started")

    def _stop_pipeline(self):
        """Stop pipeline worker thread"""
        self._pipeline_stop.set()
        if self._pipeline_thread:
            self._pipeline_thread.join(timeout=2.0)

    def _rx_worker(self):
        """RX worker thread - reads samples from RTL-SDR"""
        logger.info("RX worker thread started")

        while not self._rx_stop.is_set():
            try:
                if not self.device:
                    break

                # Read samples (RTL-SDR returns complex samples directly)
                samples = self.device.read_samples(256 * 1024)  # 256K samples

                # Convert to complex64
                iq_complex = np.array(samples, dtype=np.complex64)

                # Calculate RSSI
                power = float(np.mean(np.abs(iq_complex) ** 2))
                rssi = 10 * np.log10(power + 1e-10) - 50

                # Call callback
                if self.callback:
                    self.callback(iq_complex, rssi, self.frequency)

            except Exception as e:
                if self._rx_stop.is_set():
                    break
                logger.error(f"RX worker error: {e}")
                time.sleep(0.1)

        logger.info("RX worker thread stopped")

    def start(self, callback: Optional[Callable] = None):
        """Start receiving"""
        logger.info("start() called")

        if not self.device:
            logger.error("❌ Device not opened")
            return False

        if self.is_running:
            logger.warning("Already running")
            return True

        # Set up callback
        if callback is None:
            if self._pipeline_db is None or self._pipeline_decoder is None:
                raise ValueError(
                    "start() called with no callback, but no pipeline attached. "
                    "Call attach_pipeline(db, decoder) first, or pass a callback."
                )

            self._start_pipeline()

            def _enqueue(iq_complex, rssi, freq_hz):
                try:
                    self._pipeline_q.put_nowait((iq_complex, rssi, freq_hz, time.time()))
                except queue.Full:
                    pass

            self.callback = _enqueue
        else:
            self.callback = callback

        # Start RX thread
        self._rx_stop.clear()
        self._rx_thread = threading.Thread(target=self._rx_worker, daemon=True)
        self._rx_thread.start()

        self.is_running = True
        logger.info("✅ RTL-SDR RX started successfully")
        return True

    def stop(self):
        """Stop receiving"""
        if not self.is_running:
            return

        logger.info("Stopping RTL-SDR RX...")

        # Stop RX thread
        self._rx_stop.set()
        if self._rx_thread:
            self._rx_thread.join(timeout=2.0)

        # Stop pipeline
        self._stop_pipeline()

        self.is_running = False
        self.callback = None
        logger.info("RTL-SDR RX stopped")

    def change_frequency(self, freq_hz: float):
        """Change center frequency"""
        self.frequency = freq_hz
        if self.device:
            try:
                self.device.center_freq = int(freq_hz)
                logger.info(f"Changed frequency to {freq_hz / 1e6:.1f} MHz")
            except Exception as e:
                logger.error(f"Failed to change frequency: {e}")

    def set_gain(self, gain_db: float):
        """Set gain"""
        self.gain = gain_db
        if self.device:
            try:
                self.device.gain = gain_db
                logger.info(f"Set gain to {gain_db} dB")
            except Exception as e:
                logger.error(f"Failed to set gain: {e}")

    def set_frequency_hopping(self, enabled: bool):
        """Frequency hopping not implemented"""
        return False

    def set_hop_interval(self, interval: float):
        """Frequency hopping not implemented"""
        return False

    def increment_detection(self, frequency: float):
        """Track detections per frequency"""
        pass

    def get_status(self):
        """Get current status"""
        return {
            'frequency': self.frequency / 1e6,
            'is_streaming': self.is_running,
            'sample_rate': self.sample_rate,
            'gain': self.gain,
            'frequency_hopping': False,
            'hop_interval': 30.0,
            'frequency_stats': {}
        }

    def get_statistics(self):
        """Get statistics"""
        return {
            'is_streaming': self.is_running,
            'samples_received': 0,
            'errors': 0,
            'buffer_size': 0,
            'sample_rate': self.sample_rate
        }

    def __del__(self):
        """Cleanup"""
        try:
            if hasattr(self, 'device') and self.device:
                self.stop()
                try:
                    self.device.close()
                except Exception:
                    pass
        except Exception:
            pass


class SimulatedRTLSDR:
    """Simulated RTL-SDR for testing"""

    def __init__(self):
        logger.info("SimulatedRTLSDR initialized")
        self.is_running = False
        self.callback = None
        self.frequency = 315_000_000
        self.sample_rate = 2_400_000
        self.gain = 40
        self.thread = None

    def attach_pipeline(self, db, decoder, on_signal=None):
        return True

    def start(self, callback: Optional[Callable] = None):
        if self.is_running:
            return False

        self.callback = callback
        self.is_running = True
        self.thread = threading.Thread(target=self._simulate, daemon=True)
        self.thread.start()
        logger.info("✅ RTL-SDR simulation started")
        return True

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        logger.info("RTL-SDR simulation stopped")

    def _simulate(self):
        """Generate simulated TPMS signals"""
        while self.is_running:
            # Generate noise
            noise = (np.random.randn(256 * 1024) +
                     1j * np.random.randn(256 * 1024)) * 0.1

            # Occasionally add a signal
            if np.random.random() < 0.05:
                t = np.arange(len(noise)) / self.sample_rate
                carrier = 50000
                symbol_rate = 19200

                num_bits = int(len(t) * symbol_rate / self.sample_rate)
                bits = np.random.randint(0, 2, num_bits)
                samples_per_bit = self.sample_rate // symbol_rate
                bit_signal = np.repeat(bits, samples_per_bit)[:len(t)]

                freq_dev = 20000
                inst_freq = carrier + (bit_signal - 0.5) * 2 * freq_dev
                phase = 2 * np.pi * np.cumsum(inst_freq) / self.sample_rate
                signal = 0.5 * np.exp(1j * phase)

                noise += signal

            if self.callback:
                power = np.abs(noise) ** 2
                rssi = 10 * np.log10(np.mean(power) + 1e-10)
                self.callback(noise, rssi, self.frequency)

            time.sleep(0.5)

    def change_frequency(self, freq):
        self.frequency = freq
        return True

    def set_gain(self, gain_db):
        self.gain = gain_db
        return True

    def set_frequency_hopping(self, enabled):
        return False

    def set_hop_interval(self, interval):
        return False

    def increment_detection(self, freq):
        pass

    def get_status(self):
        return {
            'frequency': self.frequency / 1e6,
            'is_streaming': self.is_running,
            'sample_rate': self.sample_rate,
            'gain': self.gain
        }

    def get_statistics(self):
        return {
            'is_streaming': self.is_running,
            'samples_received': 0
        }


def create_rtlsdr_interface(use_simulation=False):
    """Factory function"""
    if use_simulation or not RTLSDR_AVAILABLE:
        return SimulatedRTLSDR()
    return RTLSDRInterface()


# Backward compatibility
RTLSDRScanner = RTLSDRInterface
Scanner = RTLSDRInterface
