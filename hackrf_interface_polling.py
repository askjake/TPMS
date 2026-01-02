"""
HackRF Interface using polling with hackrf_transfer command
This avoids Python library issues entirely
"""
import numpy as np
from typing import Optional, Callable
import time
import threading
import subprocess
from config import config
import sys

DEBUG = True

def debug_print(msg):
    if DEBUG:
        print(f"[{time.time():.3f}] {msg}", file=sys.stderr, flush=True)

# Check if hackrf_transfer command is available
HACKRF_AVAILABLE = False

debug_print("🔍 Checking for hackrf_transfer command...")

try:
    result = subprocess.run(['hackrf_info'], capture_output=True, timeout=2.0, text=True)
    debug_print(f"   hackrf_info return code: {result.returncode}")
    if result.stdout:
        debug_print(f"   stdout: {result.stdout[:100]}")
    if result.stderr:
        debug_print(f"   stderr: {result.stderr[:100]}")
    
    if result.returncode == 0:
        HACKRF_AVAILABLE = True
        debug_print("✅ hackrf_transfer command available")
    else:
        debug_print(f"⚠️  hackrf_info failed with code {result.returncode}")
except FileNotFoundError:
    debug_print("❌ hackrf_info command not found")
except subprocess.TimeoutExpired:
    debug_print("⚠️  hackrf_info timeout")
except Exception as e:
    debug_print(f"⚠️  hackrf_info error: {e}")

debug_print(f"HACKRF_AVAILABLE = {HACKRF_AVAILABLE}")
class HackRFInterface:
    """HackRF interface using hackrf_transfer command-line tool"""
    
    def __init__(self):
        debug_print("HackRFInterface.__init__ called")
        self.is_streaming = False
        self.callback: Optional[Callable] = None
        self.capture_thread = None
        self._stop_requested = False
        self.samples_received = 0
        self.current_frequency = config.DEFAULT_FREQUENCY
        self.lna_gain = config.LNA_GAIN
        self.vga_gain = config.VGA_GAIN
        self.enable_amp = config.ENABLE_AMP
        
        if HACKRF_AVAILABLE:
            debug_print("✅ HackRF command-line tools ready")
        else:
            debug_print("⚠️  HackRF command-line tools not available")
    
    def open(self) -> bool:
        """Open is not needed for command-line approach"""
        return HACKRF_AVAILABLE
    
    def close(self):
        """Close HackRF device"""
        self.stop()
    
    def configure(self, frequency: int, lna_gain: int = None, 
                  vga_gain: int = None, enable_amp: bool = None):
        """Configure HackRF parameters"""
        debug_print(f"configure() called: freq={frequency}, lna={lna_gain}, vga={vga_gain}, amp={enable_amp}")
        
        self.current_frequency = frequency
        if lna_gain is not None:
            self.lna_gain = lna_gain
        if vga_gain is not None:
            self.vga_gain = vga_gain
        if enable_amp is not None:
            self.enable_amp = enable_amp
        
        debug_print(f"✅ Configured: {frequency/1e6:.1f} MHz, LNA={self.lna_gain}, VGA={self.vga_gain}, Amp={'ON' if self.enable_amp else 'OFF'}")
        return True
    
    def start_rx(self, callback: Callable):
        """Start capturing using hackrf_transfer command"""
        debug_print("start_rx() called")
        
        if not HACKRF_AVAILABLE:
            debug_print("❌ HackRF tools not available")
            return False
        
        if self.is_streaming:
            debug_print("⚠️  Already streaming")
            return False
        
        self.callback = callback
        self.is_streaming = True
        self._stop_requested = False
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        debug_print("✅ Started capture thread")
        return True
    
    def stop_rx(self):
        """Stop capturing"""
        debug_print("stop_rx() called")
        self._stop_requested = True
        self.is_streaming = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
            self.capture_thread = None
        
        debug_print("✅ Stopped")
    
    def start(self, callback: Callable):
        """Start scanning (wrapper)"""
        return self.start_rx(callback)
    
    def stop(self):
        """Stop scanning (wrapper)"""
        return self.stop_rx()
    
    def _capture_loop(self):
        """Capture loop using hackrf_transfer command"""
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
                freq_hz = int(self.current_frequency)
                sample_rate = int(config.SAMPLE_RATE)
                
                cmd = [
                    'hackrf_transfer',
                    '-r', tmp_file,
                    '-f', str(freq_hz),
                    '-s', str(sample_rate),
                    '-n', str(config.SAMPLES_PER_SCAN * 2),  # *2 for I+Q bytes
                    '-a', '1' if self.enable_amp else '0',
                    '-l', str(self.lna_gain),
                    '-g', str(self.vga_gain),
                ]
                
                if capture_count <= 3 or capture_count % 10 == 0:
                    debug_print(f"Capture #{capture_count}")
                
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
                                        debug_print(f"  ✅ Callback completed")
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
                        if result.stderr:
                            debug_print(f"  stderr: {result.stderr[:200]}")
                
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
        """Get current status"""
        return {
            'is_streaming': self.is_streaming,
            'frequency': self.current_frequency / 1e6,
            'frequency_hopping': False,
            'hop_interval': 30.0,
            'frequency_stats': {}
        }
    
    def change_frequency(self, frequency: int):
        """Change frequency"""
        self.current_frequency = frequency
        debug_print(f"📡 Changed to {frequency / 1e6:.3f} MHz")
        return True
    
    def set_frequency_hopping(self, enabled: bool):
        return False
    
    def set_hop_interval(self, interval: float):
        return False
    
    def increment_detection(self, frequency: float):
        pass
    
    def get_statistics(self) -> dict:
        """Get interface statistics"""
        return {
            'is_streaming': self.is_streaming,
            'samples_received': self.samples_received,
            'errors': 0,
            'buffer_size': 0,
            'sample_rate': config.SAMPLE_RATE,
        }

class SimulatedHackRF:
    """Simulated HackRF for testing"""
    
    def __init__(self):
        debug_print("SimulatedHackRF.__init__ called")
        self.is_streaming = False
        self.callback = None
        self.thread = None
        self.current_frequency = 315_000_000
        
    def open(self):
        return True
    
    def close(self):
        self.stop()
    
    def configure(self, frequency: int, *args, **kwargs):
        self.current_frequency = frequency
        return True
    
    def start_rx(self, callback):
        debug_print("SimulatedHackRF.start_rx() called")
        self.callback = callback
        self.is_streaming = True
        self.thread = threading.Thread(target=self._sim_loop, daemon=True)
        self.thread.start()
        debug_print("✅ Simulation started")
        return True
    
    def stop_rx(self):
        self.is_streaming = False
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def start(self, callback):
        return self.start_rx(callback)
    
    def stop(self):
        return self.stop_rx()
    
    def _sim_loop(self):
        """Simulate signal capture"""
        count = 0
        while self.is_streaming:
            count += 1
            noise = (np.random.randn(config.SAMPLES_PER_SCAN) + 
                    1j * np.random.randn(config.SAMPLES_PER_SCAN)) * 0.1
            
            if count <= 3:
                debug_print(f"Simulation #{count}: generating samples")
            
            if self.callback:
                self.callback(noise, -60.0, self.current_frequency)
            
            time.sleep(0.5)
    
    def get_status(self):
        return {
            'is_streaming': self.is_streaming, 
            'frequency': self.current_frequency / 1e6
        }
    
    def change_frequency(self, freq):
        self.current_frequency = freq
        return True
    
    def set_frequency_hopping(self, enabled):
        return False
    
    def set_hop_interval(self, interval):
        return False
    
    def increment_detection(self, freq):
        pass
    
    def get_statistics(self):
        return {
            'is_streaming': self.is_streaming, 
            'samples_received': 0,
            'errors': 0,
            'buffer_size': 0,
            'sample_rate': config.SAMPLE_RATE
        }

def create_hackrf_interface(use_simulation=False):
    """Create HackRF interface"""
    if use_simulation or not HACKRF_AVAILABLE:
        debug_print("Creating SimulatedHackRF")
        return SimulatedHackRF()
    else:
        debug_print("Creating HackRFInterface")
        return HackRFInterface()
