"""
HackRF Interface using polling instead of callbacks
This avoids the segfault issue with callbacks from C threads
"""
import numpy as np
from typing import Optional, Callable
import time
import threading
from collections import deque
from config import config

DEBUG = True

def debug_print(msg):
    if DEBUG:
        print(f"[{time.time():.3f}] {msg}", flush=True)

# Try to import HackRF library
HACKRF_AVAILABLE = False
HackRF = None

debug_print("🔍 Attempting to import HackRF library...")

try:
    import hackrf
    debug_print(f"✅ Found hackrf module")
    from hackrf import HackRF
    HACKRF_AVAILABLE = True
except ImportError as e:
    debug_print(f"❌ hackrf import failed: {e}")

class HackRFInterface:
    """HackRF interface using file-based capture instead of callbacks"""
    
    def __init__(self):
        debug_print("HackRFInterface.__init__ called")
        self.device: Optional[HackRF] = None
        self.is_streaming = False
        self.callback: Optional[Callable] = None
        self.capture_thread = None
        self._stop_requested = False
        self.samples_received = 0
        
        if HACKRF_AVAILABLE:
            debug_print("🔧 Attempting to open HackRF device...")
            success = self.open()
            if success:
                self.configure(
                    config.DEFAULT_FREQUENCY,
                    config.LNA_GAIN,
                    config.VGA_GAIN,
                    config.ENABLE_AMP
                )
                debug_print("✅ HackRF ready")
        
    def open(self) -> bool:
        """Open HackRF device"""
        if not HACKRF_AVAILABLE:
            return False
        
        try:
            self.device = HackRF()
            self.device.sample_rate = config.SAMPLE_RATE
            bandwidth = int(config.SAMPLE_RATE * 0.75)
            self.device.baseband_filter_bandwidth = bandwidth
            debug_print("✅ HackRF device opened")
            return True
        except Exception as e:
            debug_print(f"❌ Failed to open HackRF: {e}")
            return False
    
    def close(self):
        """Close HackRF device"""
        self.stop()
        if self.device:
            try:
                self.device.close()
                debug_print("📻 HackRF closed")
            except:
                pass
            self.device = None
    
    def configure(self, frequency: int, lna_gain: int = None, 
                  vga_gain: int = None, enable_amp: bool = None):
        """Configure HackRF parameters"""
        if not self.device:
            return False
        
        try:
            self.device.freq = frequency
            if lna_gain is not None:
                self.device.lna_gain = lna_gain
            if vga_gain is not None:
                self.device.vga_gain = vga_gain
            if enable_amp is not None:
                self.device.enable_amp = enable_amp
            return True
        except Exception as e:
            debug_print(f"❌ Configuration error: {e}")
            return False
    
    def start(self, callback: Callable):
        """Start capturing using hackrf_transfer command"""
        debug_print("start() called - using external capture")
        
        if self.is_streaming:
            debug_print("Already streaming")
            return False
        
        self.callback = callback
        self.is_streaming = True
        self._stop_requested = False
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        debug_print("✅ Started capture thread")
        return True
    
    def stop(self):
        """Stop capturing"""
        debug_print("stop() called")
        self._stop_requested = True
        self.is_streaming = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
            self.capture_thread = None
        
        debug_print("✅ Stopped")
    
    def _capture_loop(self):
        """Capture loop using hackrf_transfer command"""
        import subprocess
        import tempfile
        import os
    
        debug_print("Capture loop started")
        capture_count = 0
    
        while not self._stop_requested:
            try:
                capture_count += 1
            
                # Create temporary file for capture
                with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as tmp:
                    tmp_file = tmp.name
            
                # Capture data using hackrf_transfer
                freq_hz = int(config.DEFAULT_FREQUENCY)
                sample_rate = int(config.SAMPLE_RATE)
            
                cmd = [
                    'hackrf_transfer',
                    '-r', tmp_file,
                    '-f', str(freq_hz),
                    '-s', str(sample_rate),
                    '-n', str(config.SAMPLES_PER_SCAN * 2),  # *2 for I+Q bytes
                    '-a', '1' if config.ENABLE_AMP else '0',
                    '-l', str(config.LNA_GAIN),
                    '-g', str(config.VGA_GAIN),
                ]
            
                if capture_count <= 3 or capture_count % 10 == 0:
                    debug_print(f"Capture #{capture_count}: {' '.join(cmd)}")
            
                # Run capture with stderr captured
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    timeout=2.0,
                    text=True
                )
            
                # Check if file was created and has data
                if os.path.exists(tmp_file):
                    file_size = os.path.getsize(tmp_file)
                
                    if capture_count <= 3:
                        debug_print(f"  File created: {file_size} bytes")
                        if result.stderr:
                            debug_print(f"  stderr: {result.stderr[:200]}")
                
                    if file_size >= 2:
                        # Read captured data
                        with open(tmp_file, 'rb') as f:
                            raw_data = np.fromfile(f, dtype=np.int8)
                    
                        if capture_count <= 3:
                            debug_print(f"  Read {len(raw_data)} samples")
                    
                        if len(raw_data) >= 2:
                            # Convert to IQ samples
                            i_data = raw_data[0::2].astype(np.float32) / 128.0
                            q_data = raw_data[1::2].astype(np.float32) / 128.0
                            iq_samples = i_data + 1j * q_data
                        
                            self.samples_received += len(iq_samples)
                        
                            if capture_count <= 3:
                                debug_print(f"  Created {len(iq_samples)} IQ samples")
                        
                            # Calculate signal strength
                            power = np.abs(iq_samples) ** 2
                            signal_strength = 10 * np.log10(np.mean(power) + 1e-10)
                        
                            if capture_count <= 3:
                                debug_print(f"  Signal strength: {signal_strength:.1f} dBm")
                        
                            # Call callback
                            if self.callback and len(iq_samples) >= config.SAMPLES_PER_SCAN:
                                if capture_count <= 3:
                                    debug_print(f"  Calling callback...")
                            
                                try:
                                    self.callback(
                                        iq_samples[:config.SAMPLES_PER_SCAN],
                                        signal_strength,
                                        freq_hz
                                    )
                                    if capture_count <= 3:
                                        debug_print(f"  Callback completed")
                                except Exception as e:
                                    debug_print(f"  ❌ Callback error: {e}")
                        else:
                            if capture_count <= 3:
                                debug_print(f"  ⚠️  Not enough raw data: {len(raw_data)}")
                    else:
                        if capture_count <= 3:
                            debug_print(f"  ⚠️  File too small: {file_size} bytes")
                else:
                    if capture_count <= 3:
                        debug_print(f"  ⚠️  File not created")
                        debug_print(f"  Return code: {result.returncode}")
                        if result.stderr:
                            debug_print(f"  stderr: {result.stderr}")
            
                # Clean up
                try:
                    os.unlink(tmp_file)
                except:
                    pass
            
            except subprocess.TimeoutExpired:
                if capture_count <= 3:
                    debug_print("⚠️  Capture timeout")
            except Exception as e:
                if capture_count <= 3:
                    debug_print(f"⚠️  Capture error: {e}")
                    import traceback
                    traceback.print_exc()
        
            # Small delay between captures
            time.sleep(0.1)
    
        debug_print(f"Capture loop ended (total captures: {capture_count})")

    
    def get_status(self) -> dict:
        return {
            'is_streaming': self.is_streaming,
            'frequency': config.DEFAULT_FREQUENCY / 1e6,
            'frequency_hopping': False,
            'hop_interval': 30.0,
            'frequency_stats': {}
        }
    
    def change_frequency(self, frequency: int):
        if self.device:
            try:
                self.device.freq = frequency
                return True
            except:
                return False
        return False
    
    def set_frequency_hopping(self, enabled: bool):
        return False
    
    def set_hop_interval(self, interval: float):
        return False
    
    def increment_detection(self, frequency: float):
        pass
    
    def get_statistics(self) -> dict:
        return {
            'is_streaming': self.is_streaming,
            'samples_received': self.samples_received,
            'errors': 0,
            'buffer_size': 0,
            'sample_rate': config.SAMPLE_RATE,
        }

class SimulatedHackRF:
    """Simulated HackRF"""
    def __init__(self):
        self.is_streaming = False
        self.callback = None
        self.thread = None
        
    def open(self):
        return True
    
    def close(self):
        self.stop()
    
    def configure(self, *args, **kwargs):
        return True
    
    def start(self, callback):
        self.callback = callback
        self.is_streaming = True
        self.thread = threading.Thread(target=self._sim_loop, daemon=True)
        self.thread.start()
        return True
    
    def stop(self):
        self.is_streaming = False
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def _sim_loop(self):
        while self.is_streaming:
            noise = (np.random.randn(config.SAMPLES_PER_SCAN) + 
                    1j * np.random.randn(config.SAMPLES_PER_SCAN)) * 0.1
            if self.callback:
                self.callback(noise, -60.0, 315000000)
            time.sleep(0.5)
    
    def get_status(self):
        return {'is_streaming': self.is_streaming, 'frequency': 315.0}
    
    def change_frequency(self, freq):
        return True
    
    def set_frequency_hopping(self, enabled):
        return False
    
    def set_hop_interval(self, interval):
        return False
    
    def increment_detection(self, freq):
        pass
    
    def get_statistics(self):
        return {'is_streaming': self.is_streaming, 'samples_received': 0}

def create_hackrf_interface(use_simulation=False):
    if use_simulation or not HACKRF_AVAILABLE:
        return SimulatedHackRF()
    return HackRFInterface()
