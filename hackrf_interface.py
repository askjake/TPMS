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
        self.data_queue = Queue(maxsize=1000)
        self.callback: Optional[Callable] = None
        
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
        
        # Auto-tuning parameters
        self.auto_tune_enabled = True
        self.last_tune_time = 0
        self.tune_interval = 10
    
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
        if self.is_running:
            self.stop()
        
        self.current_frequency = frequency
        self.callback = callback
        self.is_running = True
        self.last_hop_time = time.time()
        
        # Start on first frequency if hopping enabled
        if self.frequency_hop_enabled:
            self.frequency_index = 0
            self.current_frequency = config.FREQUENCIES[self.frequency_index]
        
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
            print(f"🔄 Frequency hopping enabled: {config.FREQUENCIES}")
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
            
            # Start frequency hopping thread
            if self.frequency_hop_enabled:
                self.hop_thread = threading.Thread(target=self._frequency_hopper, daemon=True)
                self.hop_thread.start()
            
            # Start auto-tune thread
            if self.auto_tune_enabled:
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
    
    def stop(self):
        """Stop HackRF reception"""
        self.is_running = False
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except:
                self.process.kill()
            self.process = None
        print("⏹️  HackRF stopped")
    
    def _frequency_hopper(self):
        """Automatically hop between frequencies"""
        while self.is_running:
            current_time = time.time()
            
            if current_time - self.last_hop_time >= self.hop_interval:
                # Move to next frequency
                self.frequency_index = (self.frequency_index + 1) % len(config.FREQUENCIES)
                new_frequency = config.FREQUENCIES[self.frequency_index]
                
                print(f"🔄 Hopping to {new_frequency} MHz")
                
                # Restart with new frequency
                callback = self.callback
                self.stop()
                time.sleep(config.FREQUENCY_HOP_DWELL_TIME)
                self.start(new_frequency, callback)
                
                self.last_hop_time = current_time
            
            time.sleep(0.1)
    
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
                new_gain = min(self.current_gain + config.GAIN_STEP, config.GAIN_MAX)
                if new_gain != self.current_gain:
                    self.current_gain = new_gain
                    print(f"🔧 Auto-tune: Increased gain to {new_gain} dB (signal: {avg_signal:.1f} dBm)")
            
            elif avg_signal > -30:
                new_gain = max(self.current_gain - config.GAIN_STEP, config.GAIN_MIN)
                if new_gain != self.current_gain:
                    self.current_gain = new_gain
                    print(f"🔧 Auto-tune: Decreased gain to {new_gain} dB (signal: {avg_signal:.1f} dBm)")
            
            self.last_tune_time = current_time
    
    def set_frequency_hopping(self, enabled: bool):
        """Enable or disable frequency hopping"""
        self.frequency_hop_enabled = enabled
        if enabled and self.is_running:
            print("🔄 Frequency hopping enabled")
        elif not enabled:
            print("⏸️  Frequency hopping disabled")
    
    def set_hop_interval(self, interval: float):
        """Set frequency hop interval in seconds"""
        self.hop_interval = max(1.0, interval)
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
