
import os
import shutil
import subprocess
import threading
import time
from collections import deque
from typing import Callable, Optional, List, Tuple, Dict, Any

import numpy as np

from config import config


class HackRFInterface:
    """
    HackRF receiver wrapper around `hackrf_transfer` that streams IQ from stdout.

    Expected callback signature:
        callback(complex_samples: np.ndarray, signal_strength_dbm: float, frequency_mhz: float)

    This class is designed to survive Streamlit reruns:
      - threads live in the object stored in st.session_state
      - process restarts are protected by restart_lock
      - stderr is continuously drained to avoid deadlocks
    """

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.hackrf_transfer_path: str = self._find_hackrf_transfer()

        freqs = getattr(config, "FREQUENCIES", [315.0, 433.92])
        self.current_frequency: float = float(freqs[0]) if freqs else 315.0

        # Gains (hackrf_transfer: -g VGA, -l LNA, -a amp enable)
        self.current_gain: int = int(getattr(config, "DEFAULT_GAIN", 20))
        self.lna_gain: int = int(getattr(config, "DEFAULT_LNA", 32))
        self.amp_enable: int = int(getattr(config, "AMP_ENABLE", 1))

        # Gain bounds (VGA by default: 0..62 step 2)
        self.gain_min: int = int(getattr(config, "GAIN_MIN", 0))
        self.gain_max: int = int(getattr(config, "GAIN_MAX", 62))
        self.gain_step: int = int(getattr(config, "GAIN_STEP", 2))

        # Sample rate
        self.sample_rate: int = int(getattr(config, "SAMPLE_RATE", 2_000_000))

        # Frequency hopping controls
        self.frequency_hop_enabled: bool = bool(getattr(config, "FREQUENCY_HOP_ENABLED", True))
        self.hop_interval: float = float(getattr(config, "FREQUENCY_HOP_INTERVAL", 30.0))
        self.default_dwell: float = float(getattr(config, "FREQUENCY_HOP_DWELL_TIME", self.hop_interval))
        self.is_hopping: bool = False
        self.frequency_index: int = 0
        self.last_hop_time: float = 0.0
        self.hop_thread_started: bool = False

        # Run control
        self.is_running: bool = False
        self.callback: Optional[Callable] = None
        self.restart_lock = threading.Lock()

        # Threads
        self.read_thread: Optional[threading.Thread] = None
        self.stderr_thread: Optional[threading.Thread] = None
        self.hop_thread: Optional[threading.Thread] = None

        # Status + metrics
        self.last_read_time: float = 0.0
        self.read_blocks: int = 0

        self.stderr_tail = deque(maxlen=50)

        self.signal_history: List[float] = []
        self.max_history_size: int = int(getattr(config, "SIGNAL_HISTORY_SIZE", 1000))

        self.frequency_stats: Dict[float, Dict[str, Any]] = {
            float(f): {"samples": 0, "avg_strength": 0.0, "detections": 0}
            for f in freqs
        }

        # Optional learning engine for adaptive hopping
        self.learning_engine = None
        self.adaptive_hopping: bool = True

        # Auto-tuning (kept off during hopping unless you explicitly enable)
        self.auto_tune_enabled: bool = False
        self.last_tune_time: float = 0.0
        self.tune_interval: float = 60.0

    # -----------------------
    # Learning / hop schedule
    # -----------------------

    def set_learning_engine(self, learning_engine):
        """Attach learning engine (optional)."""
        self.learning_engine = learning_engine

    def _get_adaptive_hop_schedule(self) -> List[Tuple[float, float]]:
        """
        Returns list of (frequency_mhz, dwell_seconds).
        If a learning engine is attached and provides priorities, dwell is weighted.
        """
        freqs = [float(f) for f in getattr(config, "FREQUENCIES", [self.current_frequency])]
        base = float(self.default_dwell or self.hop_interval or 30.0)

        # If engine can advise skipping, we still keep them in schedule but with small dwell
        priorities = []
        if self.learning_engine and hasattr(self.learning_engine, "get_frequency_priority"):
            for f in freqs:
                try:
                    p = float(self.learning_engine.get_frequency_priority(f))
                except Exception:
                    p = 1.0
                priorities.append(max(0.05, p))
        else:
            priorities = [1.0 for _ in freqs]

        # Normalize around 1.0 average so base dwell remains meaningful
        avg = sum(priorities) / max(len(priorities), 1)
        schedule: List[Tuple[float, float]] = []
        for f, p in zip(freqs, priorities):
            dwell = base * (p / avg) if avg > 0 else base
            dwell = max(5.0, min(dwell, base * 3.0))
            schedule.append((f, dwell))

        return schedule

    # -----------------------
    # Process management
    # -----------------------

    def _find_hackrf_transfer(self) -> str:
        """
        Find hackrf_transfer. Prefers PATH, then common Windows install paths.
        """
        path = shutil.which("hackrf_transfer") or shutil.which("hackrf_transfer.exe")
        if path:
            print(f"✅ Found hackrf_transfer in PATH: {path}")
            return path

        # Common Windows locations
        candidates = [
            r"C:\HackRF\bin\hackrf_transfer.exe",
            r"C:\Program Files\HackRF\bin\hackrf_transfer.exe",
            r"C:\Program Files (x86)\HackRF\bin\hackrf_transfer.exe",
        ]
        for c in candidates:
            if os.path.exists(c):
                print(f"✅ Found hackrf_transfer: {c}")
                return c

        print("⚠️  hackrf_transfer not found in PATH; will try 'hackrf_transfer'")
        return "hackrf_transfer"

    def start(self, frequency: float, callback: Callable) -> bool:
        """Start HackRF reception on a frequency (MHz)."""
        with self.restart_lock:
            if self.is_running:
                self._stop_process_only()

            freqs = [float(f) for f in getattr(config, "FREQUENCIES", [frequency])]
            self.current_frequency = float(frequency)
            self.callback = callback
            self.is_running = True

            # Hopping state mirrors config
            self.frequency_hop_enabled = bool(getattr(config, "FREQUENCY_HOP_ENABLED", self.frequency_hop_enabled))
            self.is_hopping = self.frequency_hop_enabled
            self.last_hop_time = time.time()

            if self.frequency_hop_enabled and freqs:
                self.frequency_index = freqs.index(self.current_frequency) if self.current_frequency in freqs else 0
                self.current_frequency = freqs[self.frequency_index]

            # Clamp + snap gain
            self.current_gain = int(self.current_gain)
            self.current_gain = max(self.gain_min, min(self.current_gain, self.gain_max))
            # Snap to step (HackRF docs: VGA step 2 dB) citeturn0search0turn0search8
            self.current_gain = (self.current_gain // self.gain_step) * self.gain_step

            cmd = [
                self.hackrf_transfer_path,
                "-r", "-",
                "-f", str(int(self.current_frequency * 1e6)),
                "-s", str(int(self.sample_rate)),
                "-g", str(int(self.current_gain)),
                "-l", str(int(self.lna_gain)),
                "-a", str(int(self.amp_enable)),
            ]

            print(f"🚀 Starting HackRF on {self.current_frequency} MHz with gain {self.current_gain} dB")
            if self.frequency_hop_enabled:
                print(f"🔄 Frequency hopping enabled: {freqs} (interval: {self.hop_interval}s)")
            print(f"Command: {' '.join(cmd)}")

            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                )

                # Give it a moment to fail fast
                time.sleep(0.35)
                if self.process.poll() is not None:
                    err = b""
                    try:
                        err = self.process.stderr.read() if self.process.stderr else b""
                    except Exception:
                        pass
                    print(f"❌ HackRF failed to start: {err.decode(errors='ignore')}")
                    self.process = None
                    self.is_running = False
                    return False

                # Start threads for this process
                self.read_thread = threading.Thread(target=self._read_samples, daemon=True)
                self.read_thread.start()

                self.stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
                self.stderr_thread.start()

                # Start hopper thread once
                if self.frequency_hop_enabled and not self.hop_thread_started:
                    self.hop_thread = threading.Thread(target=self._frequency_hopper, daemon=True)
                    self.hop_thread.start()
                    self.hop_thread_started = True

                # Do not auto-tune while hopping unless user explicitly enabled and hopping disabled
                if self.auto_tune_enabled and not self.frequency_hop_enabled:
                    t = threading.Thread(target=self._auto_tune, daemon=True)
                    t.start()

                print("✅ HackRF started successfully")
                return True

            except FileNotFoundError:
                print(f"❌ hackrf_transfer not found at: {self.hackrf_transfer_path}")
                self.is_running = False
                self.process = None
                return False
            except Exception as e:
                print(f"❌ Failed to start HackRF: {e}")
                self.is_running = False
                self.process = None
                return False

    def _drain_stderr(self):
        """Continuously drain stderr so hackrf_transfer can't block."""
        proc = self.process
        if not proc or not proc.stderr:
            return
        try:
            while self.is_running and self.process is proc and proc.poll() is None:
                line = proc.stderr.readline()
                if not line:
                    time.sleep(0.01)
                    continue
                if isinstance(line, bytes):
                    line = line.decode(errors="ignore")
                line = line.strip()
                if line:
                    self.stderr_tail.append(line)
        except Exception:
            return

    def _stop_process_only(self):
        """Stop only the hackrf_transfer process (threads may continue and exit naturally)."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                try:
                    self.process.kill()
                    self.process.wait(timeout=1)
                except Exception:
                    pass
            self.process = None
        # Give device time to release
        time.sleep(0.35)

    def stop(self):
        """Stop HackRF reception completely."""
        self.is_running = False
        self.is_hopping = False
        with self.restart_lock:
            self._stop_process_only()

    # -----------------------
    # Hopping
    # -----------------------

    def _restart_on_frequency(self, new_frequency: float) -> bool:
        """
        Restart hackrf_transfer on new_frequency (MHz).
        Assumes restart_lock already held.
        """
        self._stop_process_only()
        self.current_frequency = float(new_frequency)

        cmd = [
            self.hackrf_transfer_path,
            "-r", "-",
            "-f", str(int(self.current_frequency * 1e6)),
            "-s", str(int(self.sample_rate)),
            "-g", str(int(self.current_gain)),
            "-l", str(int(self.lna_gain)),
            "-a", str(int(self.amp_enable)),
        ]

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        time.sleep(0.25)
        if self.process.poll() is not None:
            err = b""
            try:
                err = self.process.stderr.read() if self.process.stderr else b""
            except Exception:
                pass
            print(f"❌ Failed to restart on {self.current_frequency} MHz: {err.decode(errors='ignore')}")
            self.process = None
            return False

        # Start fresh reader + stderr drain for this process
        self.read_thread = threading.Thread(target=self._read_samples, daemon=True)
        self.read_thread.start()
        self.stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self.stderr_thread.start()
        return True

    def _frequency_hopper(self):
        """Automatically hop between frequencies (optionally adaptive)."""
        freqs = [float(f) for f in getattr(config, "FREQUENCIES", [self.current_frequency])]
        if not freqs:
            return

        print(f"🔄 Frequency hopper started (adaptive={bool(getattr(self, 'adaptive_hopping', False))})")

        while self.is_hopping:
            time.sleep(1.0)
            if not self.is_hopping or not self.is_running:
                break

            schedule = self._get_adaptive_hop_schedule() if getattr(self, "adaptive_hopping", False) else [(f, self.default_dwell) for f in freqs]
            # Determine current dwell
            current_dwell = next((d for f, d in schedule if abs(f - self.current_frequency) < 0.01), float(self.default_dwell))

            now = time.time()
            if now - self.last_hop_time < current_dwell:
                continue

            # Get next frequency in schedule
            try:
                curr_idx = next((i for i, (f, _) in enumerate(schedule) if abs(f - self.current_frequency) < 0.01), 0)
            except Exception:
                curr_idx = 0
            next_idx = (curr_idx + 1) % len(schedule)
            new_frequency, _ = schedule[next_idx]

            # Optional skip rule
            if self.learning_engine and hasattr(self.learning_engine, "should_skip_frequency"):
                try:
                    if bool(self.learning_engine.should_skip_frequency(new_frequency)):
                        # Skip but still advance hop timer so we don't spin
                        self.last_hop_time = now
                        continue
                except Exception:
                    pass

            if self.restart_lock.acquire(timeout=2.0):
                try:
                    if not self.is_running or not self.is_hopping:
                        break
                    ok = self._restart_on_frequency(new_frequency)
                    if ok:
                        self.last_hop_time = time.time()
                finally:
                    self.restart_lock.release()

    # -----------------------
    # Sample reading / metrics
    # -----------------------

    def _read_samples(self):
        """Read IQ samples from hackrf_transfer stdout and pass to callback."""
        proc = self.process
        if not proc or not proc.stdout:
            return

        buffer_size = int(getattr(config, "READ_BUFFER_SIZE", 262144))

        while self.is_running and self.process is proc and proc.poll() is None:
            try:
                data = proc.stdout.read(buffer_size)
                if not data:
                    break

                iq = np.frombuffer(data, dtype=np.int8)
                if iq.size < 4:
                    continue
                if (iq.size % 2) != 0:
                    iq = iq[:-1]

                i = iq[0::2].astype(np.float32) / 128.0
                q = iq[1::2].astype(np.float32) / 128.0
                n = min(i.size, q.size)
                if n <= 0:
                    continue
                complex_samples = i[:n] + 1j * q[:n]

                # Burst-friendly power estimate (95th percentile)
                magsq = (complex_samples.real * complex_samples.real) + (complex_samples.imag * complex_samples.imag)
                p95 = float(np.percentile(magsq, 95))
                signal_strength_dbm = 10.0 * np.log10(p95 + 1e-12) - 60.0

                self.last_read_time = time.time()
                self.read_blocks += 1

                self.signal_history.append(signal_strength_dbm)
                if len(self.signal_history) > self.max_history_size:
                    self.signal_history = self.signal_history[-self.max_history_size :]

                stats = self.frequency_stats.get(self.current_frequency)
                if stats is not None:
                    stats["samples"] += 1
                    n_s = stats["samples"]
                    prev = float(stats["avg_strength"])
                    stats["avg_strength"] = (prev * (n_s - 1) + signal_strength_dbm) / n_s

                if self.callback:
                    try:
                        self.callback(complex_samples, signal_strength_dbm, self.current_frequency)
                    except Exception:
                        # Don't kill reader if UI code throws
                        pass

            except Exception as e:
                if self.is_running:
                    print(f"Error reading samples: {e}")
                break

    # -----------------------
    # Optional auto-tuning
    # -----------------------

    def _auto_tune(self):
        """Very simple gain auto-tune loop (only when hopping disabled)."""
        min_sig = float(getattr(config, "MIN_SIGNAL_STRENGTH", -80.0))
        while self.is_running and not self.frequency_hop_enabled and self.auto_tune_enabled:
            time.sleep(2.0)
            if not self.signal_history:
                continue
            avg = float(np.mean(self.signal_history[-20:]))
            now = time.time()
            if now - self.last_tune_time < self.tune_interval:
                continue

            # If we're too low, increase gain; if very high / noisy, decrease.
            new_gain = self.current_gain
            if avg < min_sig:
                new_gain = min(self.gain_max, self.current_gain + self.gain_step)
            elif avg > (min_sig + 30):
                new_gain = max(self.gain_min, self.current_gain - self.gain_step)

            if new_gain != self.current_gain:
                self.current_gain = new_gain
                self.last_tune_time = now
                with self.restart_lock:
                    if self.is_running:
                        self._restart_on_frequency(self.current_frequency)

    # -----------------------
    # Public helpers used by app
    # -----------------------

    def set_frequency_hopping(self, enabled: bool):
        self.frequency_hop_enabled = bool(enabled)
        self.is_hopping = self.frequency_hop_enabled

    def set_hop_interval(self, interval: float):
        self.hop_interval = float(interval)
        self.default_dwell = self.hop_interval

    def get_frequency_stats(self) -> Dict[float, Dict[str, Any]]:
        return self.frequency_stats

    def increment_detection(self, frequency: float):
        f = float(frequency)
        if f in self.frequency_stats:
            self.frequency_stats[f]["detections"] += 1

    def get_status(self) -> dict:
        """Used by Streamlit UI."""
        avg = float(np.mean(self.signal_history[-10:])) if self.signal_history else None
        return {
            "running": bool(self.is_running and self.process and self.process.poll() is None),
            "frequency": self.current_frequency,
            "gain": self.current_gain,
            "lna": self.lna_gain,
            "amp": self.amp_enable,
            "avg_signal_strength": avg,
            "sample_rate": self.sample_rate,
            "frequency_hopping": self.frequency_hop_enabled,
            "hop_interval": self.hop_interval,
            "read_blocks": self.read_blocks,
            "last_read_time": self.last_read_time,
            "stderr_tail": list(self.stderr_tail),
            "frequency_stats": self.frequency_stats,
        }
