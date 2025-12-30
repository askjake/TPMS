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
        """Initialize HackRF with multi-frequency scanning"""
        self.frequencies = [314.9e6, 315.0e6, 433.92e6]  # All TPMS frequencies
        self.current_freq_index = 0
        self.frequency = self.frequencies[self.current_freq_index]
        self.sample_rate = sample_rate
        self.gain = gain
        self.vga_gain = 20
        self.bandwidth = 1_750_000
        self.process = None
        self.running = False
        self.callback = None
        self.read_thread = None
    
        # Frequency hopping
        self.enable_freq_hopping = True
        self.hop_interval = 10.0  # seconds per frequency
        self.last_hop_time = time.time()
        
        # Signal tracking
        self.signal_history = []
        self.max_history_size = 100
        
        # Frequency stats
        self.current_frequency = frequency
        self.frequency_stats = {
            frequency: {
                'samples': 0,
                'avg_strength': 0.0,
                'detections': 0
            }
        }
        
        print(f"🔧 HackRF configured to match native TPMS app:")
        print(f"   Sample Rate: {self.sample_rate:,} Hz")
        print(f"   Bandwidth: {self.bandwidth:,} Hz")
        print(f"   Frequency: {self.frequency / 1e6:.1f} MHz (FIXED)")

    def start(self, callback):
        """Start HackRF with continuous reception"""
        if self.running:
            print("⚠️  HackRF already running")
            return False
        
        self.callback = callback
        
        cmd = [
            'hackrf_transfer',
            '-r', '-',
            '-f', str(int(self.frequency)),
            '-s', str(self.sample_rate),
            '-g', str(self.gain),
            '-l', str(self.vga_gain),
            '-a', '1',
            '-b', str(self.bandwidth),
        ]
        
        print(f"🚀 Starting continuous reception on {self.frequency / 1e6:.1f} MHz")
        print(f"   Command: {' '.join(cmd)}")
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=config.SIGNAL_BUFFER_SIZE
            )
            
            self.running = True
            
            self.read_thread = threading.Thread(target=self._read_samples, daemon=True)
            self.read_thread.start()
            
            print("✅ HackRF started in continuous mode")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start HackRF: {e}")
            return False

    def stop(self):
        """Stop HackRF reception"""
        if not self.running:
            return
        
        print("🛑 Stopping HackRF...")
        self.running = False
        
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
        
        if self.read_thread:
            self.read_thread.join(timeout=2)
        
        print("✅ HackRF stopped")

    def _read_samples(self):
        """Read samples with frequency hopping"""
        print("📡 Sample reader thread started")
        buffer_size = config.SIGNAL_BUFFER_SIZE
    
        while self.running and self.process:
            try:
                # Check if it's time to hop frequencies
                if self.enable_freq_hopping:
                    current_time = time.time()
                    if current_time - self.last_hop_time >= self.hop_interval:
                        self._hop_frequency()
                        self.last_hop_time = current_time
            
                # Read samples
                data = self.process.stdout.read(buffer_size)
                if not data:
                    break
                
                # Convert bytes to IQ samples
                iq_data = np.frombuffer(data, dtype=np.int8)
                
                # Ensure we have pairs of I/Q samples
                if len(iq_data) % 2 != 0:
                    iq_data = iq_data[:-1]
                
                if len(iq_data) < 2:
                    continue
                
                i_samples = iq_data[0::2].astype(np.float32) / 128.0
                q_samples = iq_data[1::2].astype(np.float32) / 128.0
                
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
                    stats['avg_strength'] = (
                        (stats['avg_strength'] * (stats['samples'] - 1) + signal_strength_dbm) 
                        / stats['samples']
                    )
                
                # Pass to callback
                if self.callback:
                    self.callback(complex_samples, signal_strength_dbm, self.current_frequency)
                
            except Exception as e:
                if self.running:
                    print(f"⚠️  Error reading samples: {e}")
                break
        
        print("📡 Sample reader thread stopped")

    def _hop_frequency(self):
        """Hop to next frequency"""
        self.current_freq_index = (self.current_freq_index + 1) % len(self.frequencies)
        new_freq = self.frequencies[self.current_freq_index]
    
        print(f"🔄 Hopping to {new_freq / 1e6:.1f} MHz")
    
        # Restart HackRF on new frequency
        self.stop()
        time.sleep(0.5)
        self.frequency = new_freq
        self.start(self.callback)

    def increment_detection(self, frequency: float):
        """Increment detection count for a frequency"""
        if frequency in self.frequency_stats:
            self.frequency_stats[frequency]['detections'] += 1

    def get_status(self) -> dict:
        """Get current receiver status"""
        return {
            'running': self.running,
            'frequency': self.current_frequency / 1e6,  # Convert to MHz
            'gain': self.gain,
            'avg_signal_strength': np.mean(self.signal_history[-10:]) if self.signal_history else None,
            'sample_rate': self.sample_rate,
            'frequency_hopping': False,  # Disabled
            'hop_interval': 0,
            'frequency_stats': {
                freq / 1e6: stats for freq, stats in self.frequency_stats.items()
            }
        }

