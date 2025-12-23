import subprocess
import numpy as np
from typing import Optional, Callable
import threading
import time
from queue import Queue
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
        self.frequency_stats = {freq: {'samples': 0, 'avg_strength': 0, 'detections': 0} 
                               for freq in config.FREQUENCIES}
        
        # Auto-tuning parameters - DISABLED during frequency hopping
        self.auto_tune_enabled = False
        self.last_tune_time = 0
        self.tune_interval = 60
    
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
            self.is_running = True
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
                
                # Start reading thread
                self.read_thread = threading.Thread(target=self._read_samples, daemon=True)
                self.read_thread.start()
                
                # Start frequency hopping thread (only once)
                if self.frequency_hop_enabled and not hasattr(self, 'hop_thread_started'):
                    self.hop_thread = threading.Thread(target=self._frequency_hopper, daemon=True)
                    self.hop_thread.start()
                    self.hop_thread_started = True
                
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
        """Automatically hop between frequencies"""
        print(f"🔄 Frequency hopper started (interval: {self.hop_interval}s)")
        
        while self.is_hopping:  # Use is_hopping instead of is_running
            time.sleep(1.0)  # Check every second
            
            if not self.is_hopping:
                break
            
            current_time = time.time()
            
            # Check if it's time to hop
            if current_time - self.last_hop_time >= self.hop_interval:
                # Try to acquire lock with timeout
                if self.restart_lock.acquire(timeout=2.0):
                    try:
                        # Move to next frequency
                        self.frequency_index = (self.frequency_index + 1) % len(config.FREQUENCIES)
                        new_frequency = config.FREQUENCIES[self.frequency_index]
                        
                        print(f"🔄 Hopping to {new_frequency} MHz (from {self.current_frequency} MHz)")
                        
                        # Stop current process only
                        self._stop_process_only()
                        
                        # Wait for settling
                        print(f"⏳ Settling for {config.FREQUENCY_HOP_DWELL_TIME}s...")
                        time.sleep(config.FREQUENCY_HOP_DWELL_TIME)
                        
                        if not self.is_hopping:
                            print("⏹️  Hopping cancelled (disabled)")
                            break
                        
                        # Update frequency
                        self.current_frequency = new_frequency
                        self.last_hop_time = current_time
                        
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
                        
                        print(f"▶️  Starting on {self.current_frequency} MHz...")
                        
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
                        
                        print(f"✅ Now scanning {new_frequency} MHz (gain: {self.current_gain} dB)")
                        
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
                
                # Convert bytes to IQ samples
                iq_data = np.frombuffer(data, dtype=np.int8)
                i_samples = iq_data[0::2].astype(np.float32) / 128.0
                q_samples = iq_data[1::2].astype(np.float32) / 128.0
                complex_samples = i_samples + 1j * q_samples
                
                # Calculate signal metrics
                power = np.mean(np.abs(complex_samples) ** 2)
                signal_strength_dbm = 10 * np.log10(power + 1e-10) - 60
                
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
