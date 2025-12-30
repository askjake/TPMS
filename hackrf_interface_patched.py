import subprocess
import numpy as np
from typing import Optional, Callable
import threading
import time
from queue import Queue
from collections import deque
from config import config
import os
import shutil

class HackRFInterface:
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.current_frequency = config.FREQUENCIES[0]
        self.current_gain = config.DEFAULT_GAIN
        self.is_running = False
        self.is_hopping = False  # NEW: separate flag for hopping
        self.data_queue = Queue(maxsize=1000)
        self.callback: Optional[Callable] = None
        
        # Thread safety
        self.restart_lock = threading.Lock()
        self.restart_pending = False
        
        # Find hackrf_transfer
        self.hackrf_transfer_path = self._find_hackrf_transfer()
        
        # Frequency hopping
        self.frequency_hop_enabled = config.FREQUENCY_HOP_ENABLED
        self.frequency_index = 0
        self.last_hop_time = 0
        self.hop_interval = config.FREQUENCY_HOP_INTERVAL
        
        # Signal metrics
        self.signal_history = []
        self.max_history_size = 100
        
        # Process/IO diagnostics
        self.stderr_tail = deque(maxlen=200)
        self.read_bytes = 0
        self.read_blocks = 0
        self.read_errors = 0
        self.last_read_time = 0.0
        self.last_signal_time = 0.0
        self.frequency_stats = {freq: {'samples': 0, 'avg_strength': 0, 'detections': 0} 
                               for freq in config.FREQUENCIES}
        
        # Auto-tuning parameters - DISABLED during frequency hopping
        self.auto_tune_enabled = False
        self.last_tune_time = 0
        self.tune_interval = 60

    # Add to HackRFInterface class after __init__

    def set_learning_engine(self, learning_engine):
        """Set reference to learning engine for adaptive hopping"""
        self.learning_engine = learning_engine
        self.adaptive_hopping = True

    def _get_adaptive_hop_schedule(self):
        """Calculate optimal frequency hopping schedule based on learning"""
        if not hasattr(self, 'learning_engine') or not self.learning_engine:
            # No learning engine, use default schedule
            return [(freq, self.hop_interval) for freq in config.FREQUENCIES]
    
        # Get statistics for each frequency
        schedule = []
    
        for freq in config.FREQUENCIES:
            stats = self.frequency_stats.get(freq, {'samples': 0, 'detections': 0})
        
            # Calculate detection rate
            if stats['samples'] > 100:
                detection_rate = stats['detections'] / stats['samples']
            else:
                detection_rate = 0.1  # Default for new frequencies
        
            # Get learned profile confidence
            optimal_params = self.learning_engine.get_optimal_scan_parameters(freq)
            profile_confidence = 0.5  # Default
        
            # Check if we have a learned profile
            for key, profile in self.learning_engine.signal_profiles.items():
                if abs(profile.frequency - freq) < 0.5:
                    profile_confidence = profile.confidence
                    break
            
            # Calculate dwell time based on detection rate and confidence
            # Higher detection rate = more time
            # Higher confidence = less time needed (we know what to look for)
            base_time = self.hop_interval
        
            if detection_rate > 0.05:  # Active frequency
                # Spend more time on productive frequencies
                multiplier = 1.0 + (detection_rate * 2.0)
                # But reduce if we're confident (know what we're looking for)
                multiplier = multiplier * (1.0 - (profile_confidence * 0.3))
                dwell_time = base_time * multiplier
            elif stats['samples'] > 500 and detection_rate < 0.001:
                # Dead frequency - skip or minimize time
                dwell_time = base_time * 0.2  # Only 20% of normal time
            else:
                dwell_time = base_time
        
            # Clamp to reasonable range
            dwell_time = max(5.0, min(60.0, dwell_time))
        
            schedule.append((freq, dwell_time))
    
        return schedule

    
    def _find_hackrf_transfer(self):
        """Find hackrf_transfer executable"""
        path = shutil.which('hackrf_transfer')
        if path:
            print(f"✅ Found hackrf_transfer in PATH: {path}")
            return 'hackrf_transfer'
        
        print("⚠️  hackrf_transfer not found, will try 'hackrf_transfer' command")
        return 'hackrf_transfer'
    
    def start(self, frequency: float, callback: Callable):
        """Start HackRF reception"""
        with self.restart_lock:
            if self.is_running:
                self._stop_process_only()  # Only stop process, not threads
            
            self.current_frequency = frequency
            self.callback = callback
            self.is_running = False  # set True after hackrf_transfer is confirmed running
            self.is_hopping = self.frequency_hop_enabled  # Set hopping flag
            self.last_hop_time = time.time()
            
            # Start on first frequency if hopping enabled
            if self.frequency_hop_enabled:
                self.frequency_index = config.FREQUENCIES.index(frequency) if frequency in config.FREQUENCIES else 0
                self.current_frequency = config.FREQUENCIES[self.frequency_index]
            
            # Ensure gain is valid (must be multiple of 2)
            self.current_gain = (self.current_gain // 2) * 2
            
            # Build command
            cmd = [
                self.hackrf_transfer_path,
                '-r', '-',
                '-f', str(int(self.current_frequency * 1e6)),
                '-s', str(config.SAMPLE_RATE),
                '-g', str(self.current_gain),
                '-l', '32',
                '-a', '1'
            ]
            
            print(f"🚀 Starting HackRF on {self.current_frequency} MHz with gain {self.current_gain} dB")
            if self.frequency_hop_enabled:
                print(f"🔄 Frequency hopping enabled: {config.FREQUENCIES} (interval: {self.hop_interval}s)")
            print(f"Command: {' '.join(cmd)}")
            
            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0
                )
                
                # Check if process started successfully
                time.sleep(0.5)
                if self.process.poll() is not None:
                    stderr = self.process.stderr.read().decode()
                    print(f"❌ HackRF failed to start: {stderr}")
                    return False
                
                
                # Process is alive
                self.is_running = True
                
                # Drain stderr so hackrf_transfer never blocks
                self.stderr_thread = threading.Thread(target=self._drain_stderr, args=(self.process,), daemon=True)
                self.stderr_thread.start()
                
                # Start reading thread
                self.read_thread = threading.Thread(target=self._read_samples, daemon=True)
                self.read_thread.start()
                
                # Start (or restart) frequency hopping thread
                if self.frequency_hop_enabled and (not hasattr(self, 'hop_thread') or not self.hop_thread.is_alive()):
                    self.hop_thread = threading.Thread(target=self._frequency_hopper, daemon=True)
                    self.hop_thread.start()
                
                # Don't start auto-tune if frequency hopping is enabled
                if self.auto_tune_enabled and not self.frequency_hop_enabled:
                    self.tune_thread = threading.Thread(target=self._auto_tune, daemon=True)
                    self.tune_thread.start()
                
                print("✅ HackRF started successfully")
                return True
                
            except FileNotFoundError:
                print(f"❌ hackrf_transfer not found at: {self.hackrf_transfer_path}")
                return False
                
            except Exception as e:
                print(f"❌ Failed to start HackRF: {e}")
                return False
    
    def _stop_process_only(self):
        """Stop only the HackRF process, keep threads running"""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except:
                try:
                    self.process.kill()
                    self.process.wait(timeout=1)
                except:
                    pass
            self.process = None
        # Give the device time to release
        time.sleep(0.5)
    
    def stop(self):
        """Stop HackRF reception completely"""
        self.is_running = False
        self.is_hopping = False
        with self.restart_lock:
            self._stop_process_only()
        print("⏹️  HackRF stopped")
    
    def _frequency_hopper(self):
        """Automatically hop between frequencies with adaptive timing"""
        print(f"🔄 Frequency hopper started (adaptive mode: {hasattr(self, 'adaptive_hopping')})")
    
        while self.is_hopping:
            time.sleep(1.0)  # Check every second
        
            if not self.is_hopping:
                break
        
            current_time = time.time()
        
            # Get current frequency's dwell time
            if hasattr(self, 'adaptive_hopping') and self.adaptive_hopping:
                schedule = self._get_adaptive_hop_schedule()
                current_dwell = next((dwell for freq, dwell in schedule 
                                    if abs(freq - self.current_frequency) < 0.01), 
                                   self.hop_interval)
            else:
                current_dwell = self.hop_interval
        
            # Check if it's time to hop
            if current_time - self.last_hop_time >= current_dwell:
                # Try to acquire lock with timeout
                if self.restart_lock.acquire(timeout=2.0):
                    try:
                        # Get next frequency from schedule
                        if hasattr(self, 'adaptive_hopping') and self.adaptive_hopping:
                            schedule = self._get_adaptive_hop_schedule()
                            # Find current index
                            current_idx = next((i for i, (freq, _) in enumerate(schedule) 
                                              if abs(freq - self.current_frequency) < 0.01), 0)
                            next_idx = (current_idx + 1) % len(schedule)
                            new_frequency, next_dwell = schedule[next_idx]
                        
                            print(f"🔄 Adaptive hop to {new_frequency} MHz (dwell: {next_dwell:.1f}s)")
                        else:
                            # Standard hopping
                            self.frequency_index = (self.frequency_index + 1) % len(config.FREQUENCIES)
                            new_frequency = config.FREQUENCIES[self.frequency_index]
                            print(f"🔄 Hopping to {new_frequency} MHz (from {self.current_frequency} MHz)")
                    
                        # Stop current reception
                        self._stop_process_only()
                    
                        # Wait for settling
                        time.sleep(config.FREQUENCY_HOP_DWELL_TIME)
                    
                        if not self.is_hopping:
                            print("⏹️  Hopping cancelled (disabled)")
                            break
                        
                        # Update frequency
                        self.current_frequency = new_frequency
                        self.last_hop_time = current_time
                    
                        # Get optimal gain from learning engine if available
                        if hasattr(self, 'learning_engine') and self.learning_engine:
                            optimal_params = self.learning_engine.get_optimal_scan_parameters(new_frequency)
                            self.current_gain = optimal_params['gain']
                        else:
                            # Ensure gain is valid
                            self.current_gain = (self.current_gain // 2) * 2
                        
                        # Build command
                        cmd = [
                            self.hackrf_transfer_path,
                            '-r', '-',
                            '-f', str(int(self.current_frequency * 1e6)),
                            '-s', str(config.SAMPLE_RATE),
                            '-g', str(self.current_gain),
                            '-l', '32',
                            '-a', '1'
                        ]
                    
                        print(f"▶️  Starting on {self.current_frequency} MHz (gain: {self.current_gain} dB)...")
                    
                        # Start new process
                        self.process = subprocess.Popen(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            bufsize=0
                        )
                    
                        # Check if started successfully
                        time.sleep(0.5)
                        if self.process.poll() is not None:
                            stderr = self.process.stderr.read().decode()
                            print(f"❌ Failed to restart on {new_frequency} MHz: {stderr}")
                            self.is_running = False
                            self.is_hopping = False
                            break
                    
                        # Restart reading thread
                        self.read_thread = threading.Thread(target=self._read_samples, daemon=True)
                        self.read_thread.start()
                    
                        print(f"✅ Now scanning {new_frequency} MHz")
                    
                    except Exception as e:
                        print(f"❌ Error during frequency hop: {e}")
                        import traceback
                        traceback.print_exc()
                    finally:
                        self.restart_lock.release()
                else:
                    # Lock timeout
                    print("⏭️  Skipping hop (device busy, lock timeout)")
                    self.last_hop_time = current_time  # Reset timer to avoid rapid retries
    
        print("🛑 Frequency hopper stopped")

    
    def _read_samples(self):
        """Read IQ samples from HackRF"""
        buffer_size = 262144
    
        while self.is_running and self.process:
            try:
                data = self.process.stdout.read(buffer_size)
                if not data:
                    break
                self.read_bytes += len(data)
                self.read_blocks += 1
                self.last_read_time = time.time()
            
                # Convert bytes to IQ samples
                iq_data = np.frombuffer(data, dtype=np.int8)
            
                # Ensure we have pairs of I/Q samples (even length)
                if len(iq_data) % 2 != 0:
                    iq_data = iq_data[:-1]  # Drop the last byte if odd
            
                if len(iq_data) < 2:  # Need at least one I/Q pair
                    continue
            
                i_samples = iq_data[0::2].astype(np.float32) / 128.0
                q_samples = iq_data[1::2].astype(np.float32) / 128.0
            
                # Double-check they're the same length
                min_len = min(len(i_samples), len(q_samples))
                i_samples = i_samples[:min_len]
                q_samples = q_samples[:min_len]
            
                complex_samples = i_samples + 1j * q_samples
            
                # Calculate signal metrics
                power = np.abs(complex_samples) ** 2
                p95 = np.percentile(power, 95)
                mean_power = float(np.mean(power))
                # Percentile-based strength is better for bursty TPMS than a whole-block mean
                signal_strength_dbm = 10 * np.log10(p95 + 1e-12) - 60
            
                # Store metrics
                self.signal_history.append(signal_strength_dbm)
                if len(self.signal_history) > self.max_history_size:
                    self.signal_history.pop(0)
            
                # Update frequency stats
                freq_key = self.current_frequency
                if freq_key in self.frequency_stats:
                    stats = self.frequency_stats[freq_key]
                    stats['samples'] += 1
                    # Running average
                    stats['avg_strength'] = (stats['avg_strength'] * (stats['samples'] - 1) + signal_strength_dbm) / stats['samples']
            
                # Pass to callback
                if self.callback:
                    self.last_signal_time = time.time()
                    self.callback(complex_samples, signal_strength_dbm, self.current_frequency)
            
            except Exception as e:
                if self.is_running:
                    print(f"Error reading samples: {e}")
                break

    
    def _auto_tune(self):
        """Automatically adjust gain for optimal reception - DISABLED during frequency hopping"""
        while self.is_running and not self.frequency_hop_enabled:
            time.sleep(self.tune_interval)
            
            if len(self.signal_history) < 10:
                continue
            
            current_time = time.time()
            if current_time - self.last_tune_time < self.tune_interval:
                continue
            
            avg_signal = np.mean(self.signal_history[-20:])
            
            # Only adjust if we're not frequency hopping
            if not self.frequency_hop_enabled and self.restart_lock.acquire(blocking=False):
                try:
                    # Adjust gain based on signal strength
                    if avg_signal < config.MIN_SIGNAL_STRENGTH - 10:
                        new_gain = min(self.current_gain + config.GAIN_STEP, config.GAIN_MAX)
                        new_gain = (new_gain // 2) * 2  # Ensure multiple of 2
                        if new_gain != self.current_gain:
                            self.current_gain = new_gain
                            print(f"🔧 Auto-tune: Increased gain to {new_gain} dB (signal: {avg_signal:.1f} dBm)")
                    
                    elif avg_signal > -30:
                        new_gain = max(self.current_gain - config.GAIN_STEP, config.GAIN_MIN)
                        new_gain = (new_gain // 2) * 2  # Ensure multiple of 2
                        if new_gain != self.current_gain:
                            self.current_gain = new_gain
                            print(f"🔧 Auto-tune: Decreased gain to {new_gain} dB (signal: {avg_signal:.1f} dBm)")
                    
                    self.last_tune_time = current_time
                finally:
                    self.restart_lock.release()
    
    def set_frequency_hopping(self, enabled: bool):
        """Enable or disable frequency hopping"""
        self.frequency_hop_enabled = enabled
        self.is_hopping = enabled
        if enabled:
            self.auto_tune_enabled = False  # Disable auto-tune when hopping
            print("🔄 Frequency hopping enabled")
            # Restart hopping thread if needed
            if not hasattr(self, 'hop_thread') or not self.hop_thread.is_alive():
                self.hop_thread = threading.Thread(target=self._frequency_hopper, daemon=True)
                self.hop_thread.start()
        else:
            print("⏸️  Frequency hopping disabled")
    
    def set_hop_interval(self, interval: float):
        """Set frequency hop interval in seconds"""
        self.hop_interval = max(10.0, interval)  # Minimum 10 seconds
        print(f"⏱️  Hop interval set to {self.hop_interval}s")
    
    def get_frequency_stats(self):
        """Get statistics for each frequency"""
        return self.frequency_stats
    
    def increment_detection(self, frequency: float):
        """Increment detection count for a frequency"""
        if frequency in self.frequency_stats:
            self.frequency_stats[frequency]['detections'] += 1
    
    def _drain_stderr(self, proc: subprocess.Popen):
        """Drain hackrf_transfer stderr continuously.

        hackrf_transfer writes periodic status lines to stderr. If stderr is piped but never
        read, the OS pipe buffer can fill and stall the process (stdout stops producing IQ).
        """
        try:
            if proc is None or proc.stderr is None:
                return
            while True:
                line = proc.stderr.readline()
                if not line:
                    break
                try:
                    txt = line.decode(errors='replace').strip()
                except Exception:
                    txt = str(line)
                if txt:
                    self.stderr_tail.append(txt)
        except Exception:
            # Never let stderr draining crash the receiver
            pass

    def get_status(self) -> dict:
        """Get current receiver status"""
        return {
            'running': self.is_running,
            'frequency': self.current_frequency,
            'gain': self.current_gain,
            'avg_signal_strength': np.mean(self.signal_history[-10:]) if self.signal_history else None,
            'sample_rate': config.SAMPLE_RATE,
            'frequency_hopping': self.frequency_hop_enabled,
            'hop_interval': self.hop_interval,
            'frequency_stats': self.frequency_stats
        }


