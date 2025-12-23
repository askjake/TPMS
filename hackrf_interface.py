import subprocess
import numpy as np
from typing import Optional, Callable
import threading
import time
from queue import Queue
from config import config


class HackRFInterface:
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.current_frequency = config.FREQUENCIES[0]
        self.current_gain = 20  # Start mid-range
        self.is_running = False
        self.data_queue = Queue(maxsize=1000)
        self.callback: Optional[Callable] = None

        # Auto-tuning parameters
        self.signal_history = []
        self.max_history_size = 100
        self.auto_tune_enabled = True
        self.last_tune_time = 0
        self.tune_interval = 5  # seconds

    def start(self, frequency: float, callback: Callable):
        """Start HackRF reception"""
        if self.is_running:
            self.stop()

        self.current_frequency = frequency
        self.callback = callback
        self.is_running = True

        # Start hackrf_transfer process
        cmd = [
            'hackrf_transfer',
            '-r', '-',  # Read to stdout
            '-f', str(int(frequency * 1e6)),  # Frequency in Hz
            '-s', str(config.SAMPLE_RATE),
            '-g', str(self.current_gain),
            '-l', '32',  # LNA gain
            '-a', '1'  # Enable amp
        ]

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )

            # Start reading thread
            self.read_thread = threading.Thread(target=self._read_samples, daemon=True)
            self.read_thread.start()

            # Start auto-tune thread
            if self.auto_tune_enabled:
                self.tune_thread = threading.Thread(target=self._auto_tune, daemon=True)
                self.tune_thread.start()

            return True
        except Exception as e:
            print(f"Failed to start HackRF: {e}")
            return False

    def stop(self):
        """Stop HackRF reception"""
        self.is_running = False
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=5)
            self.process = None

    def _read_samples(self):
        """Read IQ samples from HackRF"""
        buffer_size = 262144  # 256KB chunks

        while self.is_running and self.process:
            try:
                data = self.process.stdout.read(buffer_size)
                if not data:
                    break

                # Convert bytes to IQ samples
                iq_data = np.frombuffer(data, dtype=np.int8)
                i_samples = iq_data[0::2].astype(np.float32) / 128.0
                q_samples = iq_data[1::2].astype(np.float32) / 128.0
                complex_samples = i_samples + 1j * q_samples

                # Calculate signal metrics
                power = np.mean(np.abs(complex_samples) ** 2)
                signal_strength_dbm = 10 * np.log10(power + 1e-10) - 60

                # Store metrics for auto-tuning
                self.signal_history.append(signal_strength_dbm)
                if len(self.signal_history) > self.max_history_size:
                    self.signal_history.pop(0)

                # Pass to callback
                if self.callback:
                    self.callback(complex_samples, signal_strength_dbm, self.current_frequency)

            except Exception as e:
                print(f"Error reading samples: {e}")
                break

    def _auto_tune(self):
        """Automatically adjust gain for optimal reception"""
        while self.is_running:
            time.sleep(self.tune_interval)

            if len(self.signal_history) < 10:
                continue

            current_time = time.time()
            if current_time - self.last_tune_time < self.tune_interval:
                continue

            avg_signal = np.mean(self.signal_history[-20:])

            # Adjust gain based on signal strength
            if avg_signal < config.MIN_SIGNAL_STRENGTH - 10:
                # Signal too weak, increase gain
                new_gain = min(self.current_gain + config.GAIN_STEP, config.GAIN_MAX)
                if new_gain != self.current_gain:
                    self.set_gain(new_gain)
                    print(f"Auto-tune: Increased gain to {new_gain} (signal: {avg_signal:.1f} dBm)")

            elif avg_signal > -30:
                # Signal too strong, decrease gain
                new_gain = max(self.current_gain - config.GAIN_STEP, config.GAIN_MIN)
                if new_gain != self.current_gain:
                    self.set_gain(new_gain)
                    print(f"Auto-tune: Decreased gain to {new_gain} (signal: {avg_signal:.1f} dBm)")

            self.last_tune_time = current_time

    def set_gain(self, gain: int):
        """Change receiver gain"""
        self.current_gain = gain
        # Restart with new gain
        if self.is_running:
            callback = self.callback
            freq = self.current_frequency
            self.stop()
            time.sleep(0.5)
            self.start(freq, callback)

    def set_frequency(self, frequency: float):
        """Change receiver frequency"""
        self.current_frequency = frequency
        if self.is_running:
            callback = self.callback
            self.stop()
            time.sleep(0.5)
            self.start(frequency, callback)

    def get_status(self) -> dict:
        """Get current receiver status"""
        return {
            'running': self.is_running,
            'frequency': self.current_frequency,
            'gain': self.current_gain,
            'avg_signal_strength': np.mean(self.signal_history[-10:]) if self.signal_history else None,
            'sample_rate': config.SAMPLE_RATE
        }
