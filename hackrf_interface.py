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
    def __init__(self, frequency: float = 314.9e6, sample_rate: int = 2_457_600, gain: int = 20):
        """Initialize HackRF with settings matching native TPMS app"""
        self.frequency = frequency
        self.sample_rate = sample_rate  # Match native: 2,457,600 Hz
        self.gain = gain
        self.vga_gain = 20
        self.bandwidth = 1_750_000  # Match native: 1.75 MHz
        self.process = None
        self.running = False
        self.callback = None
    
        # DISABLE frequency hopping for continuous reception
        self.frequencies = [314.9e6]  # Stay on primary frequency
        self.current_freq_index = 0
        self.adaptive_hopping = False  # Disabled
    
        print(f"🔧 HackRF configured to match native TPMS app:")
        print(f"   Sample Rate: {self.sample_rate:,} Hz")
        print(f"   Bandwidth: {self.bandwidth:,} Hz")
        print(f"   Frequency: {self.frequency / 1e6:.1f} MHz (FIXED)")

    def start(self, callback):
        """Start HackRF with continuous reception (no hopping)"""
        if self.running:
            print("⚠️  HackRF already running")
            return False
    
        self.callback = callback
    
        # Build command with optimized parameters
        cmd = [
            'hackrf_transfer',
            '-r', '-',  # Read to stdout
            '-f', str(int(self.frequency)),
            '-s', str(self.sample_rate),
            '-g', str(self.gain),
            '-l', str(self.vga_gain),
            '-a', '1',  # Enable RF amp
            '-b', str(self.bandwidth),  # Set bandwidth explicitly
        ]
    
        print(f"🚀 Starting continuous reception on {self.frequency / 1e6:.1f} MHz")
        print(f"   Command: {' '.join(cmd)}")
    
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=config.SIGNAL_BUFFER_SIZE  # Larger buffer
            )
        
            self.running = True
        
            # Start reading thread
            self.read_thread = threading.Thread(target=self._read_samples, daemon=True)
            self.read_thread.start()
        
            print("✅ HackRF started in continuous mode")
            return True
        
        except Exception as e:
            print(f"❌ Failed to start HackRF: {e}")
            return False

    def _read_samples(self):
        """Read samples continuously without interruption"""
        print("📡 Sample reader thread started")
    
        while self.running and self.process:
            try:
                # Read larger chunks for efficiency
                chunk = self.process.stdout.read(config.SIGNAL_BUFFER_SIZE)
            
                if not chunk:
                    break
            
                if self.callback:
                    # Convert to complex samples
                    samples = np.frombuffer(chunk, dtype=np.uint8).astype(np.float32)
                    samples = (samples - 127.5) / 127.5
                
                    # Create I/Q pairs
                    if len(samples) % 2 == 0:
                        iq_samples = samples[::2] + 1j * samples[1::2]
                    
                        # Call decoder callback
                        self.callback({
                            'samples': iq_samples,
                            'frequency': self.frequency,
                            'sample_rate': self.sample_rate,
                            'timestamp': time.time()
                        })
                    
            except Exception as e:
                if self.running:
                    print(f"⚠️  Error reading samples: {e}")
                break
    
        print("📡 Sample reader thread stopped")

    
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





