### hackrf_interface
import logging
import time
import numpy as np
from typing import Callable, Optional
import threading
import queue

# Import our wrapper
try:
    from hackrf_wrapper import HackRFDevice, is_available

    HACKRF_AVAILABLE = is_available()
except Exception as e:
    logging.error(f"Failed to import HackRF wrapper: {e}")
    HACKRF_AVAILABLE = False
    HackRFDevice = None

logger = logging.getLogger(__name__)


class HackRFInterface:
    def __init__(self):
        logger.info("HackRFInterface.__init__ called")

        self.device = None
        self.is_running = False
        self.callback = None
        self.frequency = 314.9e6
        self.sample_rate = 2_457_600
        self.lna_gain = 32
        self.vga_gain = 40
        # --- Optional decode→DB pipeline (used when start() is called with no callback) ---
        self._pipeline_db = None
        self._pipeline_decoder = None
        self._pipeline_on_signal = None  # optional callable(signal_dict)
        self._pipeline_q: "queue.Queue[tuple]" = queue.Queue(maxsize=200)
        self._pipeline_stop = threading.Event()
        self._pipeline_thread: Optional[threading.Thread] = None

        if not HACKRF_AVAILABLE:
            logger.warning("HackRF library not available - simulation mode")
            return

        # Try to open device
        try:
            self.device = HackRFDevice()
            if self.device.open():
                logger.info("✅ HackRF device opened successfully")
                self._configure_device()
            else:
                logger.error("❌ Failed to open HackRF device")
                self.device = None
        except Exception as e:
            logger.error(f"❌ Error initializing HackRF: {e}")
            self.device = None

    def _configure_device(self):
        """Configure device with default settings"""
        if not self.device:
            return

        self.device.set_freq(int(self.frequency))
        self.device.set_sample_rate(self.sample_rate)
        self.device.set_lna_gain(self.lna_gain)
        self.device.set_vga_gain(self.vga_gain)
        self.device.set_amp_enable(True)

    def attach_pipeline(self, db, decoder, on_signal=None):
        """
        Attach a TPMS decode→DB pipeline.

        If start() is called with callback=None, RX will enqueue IQ chunks and a worker
        thread will decode frames + insert into the DB.
        """
        self._pipeline_db = db
        self._pipeline_decoder = decoder
        self._pipeline_on_signal = on_signal
        return True

    def _pipeline_worker(self):
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
                    # best-effort mapping
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

                # Insert (batch if available)
                if hasattr(db, "insert_signals_batch"):
                    db.insert_signals_batch(rows)
                else:
                    for r in rows:
                        db.insert_signal(r)

                # Optional hook (UI/debug)
                if self._pipeline_on_signal:
                    for r in rows:
                        try:
                            self._pipeline_on_signal(r)
                        except Exception:
                            pass

            except Exception as e:
                logger.error(f"Pipeline worker error: {e}")

    def _start_pipeline(self):
        if self._pipeline_thread and self._pipeline_thread.is_alive():
            return
        self._pipeline_stop.clear()
        self._pipeline_thread = threading.Thread(target=self._pipeline_worker, daemon=True)
        self._pipeline_thread.start()

    def _stop_pipeline(self):
        self._pipeline_stop.set()

    def start(self, callback: Optional[Callable] = None):
        """Start receiving.

        If callback is provided: callback(iq_complex, rssi_dbm, freq_hz) is called.
        If callback is None: uses attached decode→DB pipeline (requires attach_pipeline()).
        """
        logger.info("start() called")

        if not self.device:
            logger.error("❌ Device not opened")
            return False

        # If user didn't provide callback, use built-in pipeline
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
                    # drop if we can't keep up
                    pass

            self.callback = _enqueue
        else:
            self.callback = callback

        def rx_callback(iq_data):
            try:
                iq_complex = (iq_data[::2] + 1j * iq_data[1::2]).astype(np.complex64) / 128.0
                power = float(np.mean(np.abs(iq_complex) ** 2))
                rssi = 10 * np.log10(power + 1e-10) - 50  # rough calibration

                if self.callback:
                    self.callback(iq_complex, rssi, self.frequency)
            except Exception as e:
                logger.error(f"Callback error: {e}")

        if self.device.start_rx(rx_callback):
            self.is_running = True
            logger.info("✅ RX started successfully")
            return True

        logger.error("❌ Failed to start RX")
        return False

    def stop(self):
        """Stop receiving"""
        # stop RX
        if self.device and self.is_running:
            try:
                self.device.stop_rx()
            except Exception:
                pass
            self.is_running = False
            logger.info("RX stopped")

        # stop pipeline worker
        self._stop_pipeline()

    def change_frequency(self, freq_hz: float):
        """Change center frequency"""
        self.frequency = freq_hz
        if self.device:
            self.device.set_freq(int(freq_hz))
            logger.info(f"Changed frequency to {freq_hz / 1e6:.1f} MHz")

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
            'lna_gain': self.lna_gain,
            'vga_gain': self.vga_gain,
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
        if self.device:
            self.stop()
            self.device.close()


class SimulatedHackRF:
    """Simulated HackRF for testing"""

    def __init__(self):
        logger.info("SimulatedHackRF initialized")
        self.is_running = False
        self.callback = None
        self.frequency = 315_000_000
        self.thread = None

    def start(self, callback: Callable):
        if self.is_running:
            return False

        self.callback = callback
        self.is_running = True
        self.thread = threading.Thread(target=self._simulate, daemon=True)
        self.thread.start()
        logger.info("✅ Simulation started")
        return True

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        logger.info("Simulation stopped")

    def _simulate(self):
        """Generate simulated TPMS signals"""
        from config import config

        while self.is_running:
            # Generate noise
            noise = (np.random.randn(config.SAMPLES_PER_SCAN) +
                     1j * np.random.randn(config.SAMPLES_PER_SCAN)) * 0.1

            # Occasionally add a signal
            if np.random.random() < 0.05:
                t = np.arange(config.SAMPLES_PER_SCAN) / config.SAMPLE_RATE
                carrier = 50000
                symbol_rate = 19200

                num_bits = int(len(t) * symbol_rate / config.SAMPLE_RATE)
                bits = np.random.randint(0, 2, num_bits)
                samples_per_bit = config.SAMPLE_RATE // symbol_rate
                bit_signal = np.repeat(bits, samples_per_bit)[:len(t)]

                freq_dev = 20000
                inst_freq = carrier + (bit_signal - 0.5) * 2 * freq_dev
                phase = 2 * np.pi * np.cumsum(inst_freq) / config.SAMPLE_RATE
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

    def set_frequency_hopping(self, enabled):
        return False

    def set_hop_interval(self, interval):
        return False

    def increment_detection(self, freq):
        pass

    def get_status(self):
        return {'frequency': self.frequency / 1e6, 'is_streaming': self.is_running}

    def get_statistics(self):
        return {'is_streaming': self.is_running, 'samples_received': 0}


def create_hackrf_interface(use_simulation=False):
    """Factory function"""
    if use_simulation or not HACKRF_AVAILABLE:
        return SimulatedHackRF()
    return HackRFInterface()


# -------------------------------------------------------------------
# Backward-compatibility shims (some apps expect HackRFScanner/Scanner)
# -------------------------------------------------------------------

class HackRFScanner(HackRFInterface):
    """Compatibility alias for older app code expecting HackRFScanner/Scanner."""

    def set_frequency(self, freq_mhz: float) -> None:
        # app code often uses MHz; HackRFInterface uses Hz
        self.change_frequency(freq_mhz * 1e6)


# Some code checks for Scanner
HackRFScanner = HackRFInterface
Scanner = HackRFScanner

