# hackrf_interface.py
"""
HackRF interface wrapper used by the Streamlit TPMS tracker.

Key behavior:
- Provides HackRFInterface.start(callback=None).
  * If callback is provided: callback(iq_complex, rssi_dbm, freq_hz) is invoked for each RX chunk.
  * If callback is None: a decode→DB pipeline is used (requires attach_pipeline(db, decoder)).
- Exposes HackRFScanner and Scanner classes for backward compatibility with older app wiring.
"""

from __future__ import annotations

import logging
import time
import threading
import queue
from typing import Callable, Optional, Any

import numpy as np

# Import our wrapper
try:
    from hackrf_wrapper import HackRFDevice, is_available

    HACKRF_AVAILABLE = is_available()
except Exception as e:
    logging.getLogger(__name__).error(f"Failed to import HackRF wrapper: {e}")
    HACKRF_AVAILABLE = False
    HackRFDevice = None  # type: ignore

logger = logging.getLogger(__name__)


class HackRFInterface:
    """
    Thin wrapper around HackRFDevice.

    Notes:
    - `frequency` is stored in Hz.
    - `start()` is safe to call with no args if you previously called `attach_pipeline(db, decoder)`.
    """

    def __init__(self):
        logger.info("HackRFInterface.__init__ called")

        self.device = None
        self.is_running = False
        self.callback: Optional[Callable[[np.ndarray, float, float], None]] = None

        # RF settings
        self.frequency: float = 314.9e6
        self.sample_rate: float = 2_457_600
        self.lna_gain: int = 32
        self.vga_gain: int = 40

        # --- Optional decode→DB pipeline (used when start() is called with no callback) ---
        self._pipeline_db: Any = None
        self._pipeline_decoder: Any = None
        self._pipeline_on_signal: Optional[Callable[[dict], None]] = None
        self._pipeline_q: "queue.Queue[tuple[np.ndarray, float, float, float]]" = queue.Queue(maxsize=400)
        self._pipeline_stop = threading.Event()
        self._pipeline_thread: Optional[threading.Thread] = None

        if not HACKRF_AVAILABLE:
            logger.warning("HackRF library not available - simulation mode suggested")
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

    # ---------------------------
    # Device control
    # ---------------------------

    def _configure_device(self) -> None:
        """Configure device with current settings."""
        if not self.device:
            return
        self.device.set_freq(int(self.frequency))
        self.device.set_sample_rate(self.sample_rate)
        self.device.set_lna_gain(self.lna_gain)
        self.device.set_vga_gain(self.vga_gain)
        try:
            self.device.set_amp_enable(True)
        except Exception:
            # Some wrappers may not support this; ignore.
            pass

    def change_frequency(self, freq_hz: float) -> None:
        """Change center frequency (Hz)."""
        self.frequency = float(freq_hz)
        if self.device:
            self.device.set_freq(int(self.frequency))
        logger.info(f"Changed frequency to {self.frequency/1e6:.3f} MHz")

    # Convenience for older app code (MHz)
    def set_frequency(self, freq_mhz: float) -> None:
        self.change_frequency(float(freq_mhz) * 1e6)

    def set_frequency_hopping(self, enabled: bool) -> bool:
        """Frequency hopping not implemented in this wrapper."""
        return False

    def set_hop_interval(self, interval: float) -> bool:
        """Frequency hopping not implemented in this wrapper."""
        return False

    def increment_detection(self, frequency: float) -> None:
        """Optional per-frequency stats hook (not implemented)."""
        return

    def get_status(self) -> dict:
        """Get current status."""
        return {
            "frequency": self.frequency / 1e6,
            "is_streaming": self.is_running,
            "sample_rate": self.sample_rate,
            "lna_gain": self.lna_gain,
            "vga_gain": self.vga_gain,
            "frequency_hopping": False,
            "hop_interval": 30.0,
            "frequency_stats": {},
        }

    def get_statistics(self) -> dict:
        """Return lightweight stats (placeholder)."""
        return {
            "is_streaming": self.is_running,
            "samples_received": 0,
            "errors": 0,
            "buffer_size": 0,
            "sample_rate": self.sample_rate,
        }

    # ---------------------------
    # Optional decode→DB pipeline
    # ---------------------------

    def attach_pipeline(self, db: Any, decoder: Any, on_signal: Optional[Callable[[dict], None]] = None) -> bool:
        """
        Attach a TPMS decode→DB pipeline.

        If start() is called with callback=None, RX will enqueue IQ chunks and a worker thread will:
        - decoder.process_samples(iq_complex, freq_hz)
        - map each decoded signal to dict
        - insert into db (insert_signals_batch preferred, else insert_signal)
        """
        self._pipeline_db = db
        self._pipeline_decoder = decoder
        self._pipeline_on_signal = on_signal
        return True

    def _pipeline_worker(self) -> None:
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

                now_ts = ts or time.time()
                rows = []
                for s in signals:
                    # Best-effort attribute mapping
                    try:
                        s.signal_strength = rssi
                    except Exception:
                        pass

                    rows.append(
                        {
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
                        }
                    )

                # Insert (batch if available)
                if hasattr(db, "insert_signals_batch"):
                    db.insert_signals_batch(rows)
                else:
                    for r in rows:
                        db.insert_signal(r)

                if self._pipeline_on_signal:
                    for r in rows:
                        try:
                            self._pipeline_on_signal(r)
                        except Exception:
                            pass

            except Exception as e:
                logger.exception(f"Pipeline worker error: {e}")

    def _start_pipeline(self) -> None:
        if self._pipeline_thread and self._pipeline_thread.is_alive():
            return
        self._pipeline_stop.clear()
        self._pipeline_thread = threading.Thread(target=self._pipeline_worker, daemon=True)
        self._pipeline_thread.start()

    def _stop_pipeline(self) -> None:
        self._pipeline_stop.set()

    # ---------------------------
    # RX start/stop
    # ---------------------------

    def start(self, callback: Optional[Callable[[np.ndarray, float, float], None]] = None) -> bool:
        """
        Start receiving.

        If callback is provided: callback(iq_complex, rssi_dbm, freq_hz) is called.
        If callback is None: uses attached decode→DB pipeline (requires attach_pipeline()).
        """
        logger.info("start() called")

        if self.is_running:
            logger.info("RX already running")
            return True

        if not self.device:
            logger.error("❌ Device not opened (is HackRF connected / permissions OK?)")
            return False

        # If user didn't provide callback, use built-in pipeline
        if callback is None:
            if self._pipeline_db is None or self._pipeline_decoder is None:
                logger.error(
                    "start() called with no callback, but no pipeline attached. "
                    "Call attach_pipeline(db, decoder) first, or pass a callback."
                )
                return False

            self._start_pipeline()

            def _enqueue(iq_complex: np.ndarray, rssi_dbm: float, freq_hz: float) -> None:
                try:
                    self._pipeline_q.put_nowait((iq_complex, rssi_dbm, freq_hz, time.time()))
                except queue.Full:
                    # Drop if we can't keep up
                    pass

            self.callback = _enqueue
        else:
            self.callback = callback

        def rx_callback(iq_data: np.ndarray) -> None:
            """Internal callback wrapper called by HackRFDevice."""
            try:
                # iq_data from wrapper is int8 interleaved I,Q: [I0,Q0,I1,Q1,...]
                if not isinstance(iq_data, np.ndarray):
                    iq_data = np.asarray(iq_data, dtype=np.int8)

                # Safety: ensure even length
                if iq_data.size < 2:
                    return
                if iq_data.size % 2 == 1:
                    iq_data = iq_data[:-1]

                i = iq_data[::2].astype(np.float32)
                q = iq_data[1::2].astype(np.float32)
                iq_complex = (i + 1j * q).astype(np.complex64) / 128.0

                power = float(np.mean(np.abs(iq_complex) ** 2))
                rssi_dbm = 10.0 * np.log10(power + 1e-10) - 50.0  # rough calibration

                cb = self.callback
                if cb:
                    cb(iq_complex, rssi_dbm, float(self.frequency))
            except Exception as e:
                logger.exception(f"RX callback error: {e}")

        ok = bool(self.device.start_rx(rx_callback))
        if ok:
            self.is_running = True
            logger.info("✅ RX started successfully")
            return True

        logger.error("❌ Failed to start RX")
        return False

    def stop(self) -> None:
        """Stop receiving + stop pipeline worker."""
        if self.device and self.is_running:
            try:
                self.device.stop_rx()
            except Exception:
                pass
            self.is_running = False
            logger.info("RX stopped")

        self._stop_pipeline()

    def __del__(self):
        """Best-effort cleanup."""
        try:
            self.stop()
        except Exception:
            pass
        try:
            if self.device:
                self.device.close()
        except Exception:
            pass


class SimulatedHackRF(HackRFInterface):
    """
    Simulated HackRF that reuses HackRFInterface's pipeline logic.
    Useful when running without real hardware.
    """

    def __init__(self):
        # Don't call HackRFInterface.__init__ (it tries to open real hardware).
        self.device = None
        self.is_running = False
        self.callback = None
        self.frequency = 315_000_000.0
        self.sample_rate = 2_457_600.0
        self.lna_gain = 0
        self.vga_gain = 0

        self._pipeline_db = None
        self._pipeline_decoder = None
        self._pipeline_on_signal = None
        self._pipeline_q = queue.Queue(maxsize=400)
        self._pipeline_stop = threading.Event()
        self._pipeline_thread = None

        self._sim_thread: Optional[threading.Thread] = None

        logger.info("SimulatedHackRF initialized")

    def start(self, callback: Optional[Callable[[np.ndarray, float, float], None]] = None) -> bool:
        if self.is_running:
            return True

        # Mirror HackRFInterface.start behavior
        if callback is None:
            if self._pipeline_db is None or self._pipeline_decoder is None:
                logger.error(
                    "SimulatedHackRF.start() called with no callback, but no pipeline attached. "
                    "Call attach_pipeline(db, decoder) first, or pass a callback."
                )
                return False

            self._start_pipeline()

            def _enqueue(iq_complex: np.ndarray, rssi_dbm: float, freq_hz: float) -> None:
                try:
                    self._pipeline_q.put_nowait((iq_complex, rssi_dbm, freq_hz, time.time()))
                except queue.Full:
                    pass

            self.callback = _enqueue
        else:
            self.callback = callback

        self.is_running = True
        self._sim_thread = threading.Thread(target=self._simulate, daemon=True)
        self._sim_thread.start()
        logger.info("✅ Simulation started")
        return True

    def stop(self) -> None:
        self.is_running = False
        self._stop_pipeline()
        if self._sim_thread:
            self._sim_thread.join(timeout=1.0)
        logger.info("Simulation stopped")

    def _simulate(self) -> None:
        """Generate simulated IQ buffers periodically."""
        try:
            from config import config  # your project's config object
            n = int(getattr(config, "SAMPLES_PER_SCAN", 4096))
            sr = float(getattr(config, "SAMPLE_RATE", self.sample_rate))
        except Exception:
            n = 4096
            sr = self.sample_rate

        while self.is_running:
            # Generate noise
            iq = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64) * 0.1

            # Occasionally add a simple FSK-ish tone burst
            if np.random.random() < 0.08:
                t = np.arange(n, dtype=np.float32) / sr
                carrier = 50_000.0
                freq_dev = 20_000.0
                bits = np.random.randint(0, 2, size=64)
                bit_sig = np.repeat(bits, max(1, n // bits.size))[:n].astype(np.float32)
                inst_freq = carrier + (bit_sig - 0.5) * 2.0 * freq_dev
                phase = 2.0 * np.pi * np.cumsum(inst_freq) / sr
                iq += (0.5 * np.exp(1j * phase)).astype(np.complex64)

            power = float(np.mean(np.abs(iq) ** 2))
            rssi_dbm = 10.0 * np.log10(power + 1e-10) - 30.0
            cb = self.callback
            if cb:
                cb(iq, rssi_dbm, float(self.frequency))

            time.sleep(0.25)


def create_hackrf_interface(use_simulation: bool = False) -> HackRFInterface:
    """Factory function."""
    if use_simulation or not HACKRF_AVAILABLE:
        return SimulatedHackRF()
    return HackRFInterface()


# -------------------------------------------------------------------
# Backward-compatibility shims (some apps expect HackRFScanner/Scanner)
# -------------------------------------------------------------------

class HackRFScanner(HackRFInterface):
    """Compatibility alias for older app code expecting HackRFScanner/Scanner."""
    pass


# Some code checks for Scanner
Scanner = HackRFScanner
