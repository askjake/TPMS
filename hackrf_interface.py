"""
HackRF Interface for TPMS Signal Capture
Modern Python (3.10+) compatible - with extensive debugging
"""
import numpy as np
from typing import Optional, Callable
import time
import threading
from collections import deque
from config import config
import sys
import traceback

# Debug mode
DEBUG = True

def debug_print(msg):
    """Print debug message with timestamp"""
    if DEBUG:
        print(f"[{time.time():.3f}] {msg}", flush=True)

# Try to import HackRF library
HACKRF_AVAILABLE = False
HackRF = None
HackRFError = Exception

debug_print("🔍 Attempting to import HackRF library...")

try:
    import hackrf
    debug_print(f"✅ Found hackrf module at: {hackrf.__file__}")
    
    from hackrf import HackRF
    
    try:
        from hackrf import HackRfError as HackRFError
        debug_print("   Using HackRfError")
    except ImportError:
        debug_print("   No specific error class found")
        HackRFError = Exception
    
    HACKRF_AVAILABLE = True
    debug_print("✅ HackRF library loaded successfully")
    
except ImportError as e:
    debug_print(f"❌ hackrf import failed: {e}")

class HackRFInterface:
    def __init__(self):
        debug_print("HackRFInterface.__init__ called")
        self.device: Optional[HackRF] = None
        self.is_streaming = False
        self.callback: Optional[Callable] = None
        self.sample_buffer = deque(maxlen=config.NUM_BUFFERS)
        self.lock = threading.Lock()
        self.samples_received = 0
        self.errors = 0
        self._stop_requested = False
        self.callback_count = 0
        self.last_callback_time = 0
        
        # Try to open device on init if library is available
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
                debug_print("✅ HackRF ready (not streaming)")
        
    def open(self) -> bool:
        """Open HackRF device"""
        debug_print("open() called")
        if not HACKRF_AVAILABLE:
            debug_print("⚠️  HackRF library not available")
            return False
        
        try:
            debug_print("Creating HackRF() instance...")
            self.device = HackRF()
            debug_print("✅ HackRF device opened")
            
            debug_print(f"Setting sample rate to {config.SAMPLE_RATE}...")
            self.device.sample_rate = config.SAMPLE_RATE
            debug_print(f"📊 Sample rate: {config.SAMPLE_RATE / 1e6:.3f} MHz")
            
            bandwidth = int(config.SAMPLE_RATE * 0.75)
            debug_print(f"Setting bandwidth to {bandwidth}...")
            self.device.baseband_filter_bandwidth = bandwidth
            debug_print(f"🔧 Baseband filter: {bandwidth / 1e6:.3f} MHz")
            
            return True
            
        except Exception as e:
            debug_print(f"❌ Failed to open HackRF: {e}")
            traceback.print_exc()
            return False
    
    def close(self):
        """Close HackRF device"""
        debug_print("close() called")
        self.stop_rx()
        
        if self.device:
            try:
                self.device.close()
                debug_print("📻 HackRF closed")
            except Exception as e:
                debug_print(f"⚠️  Error closing HackRF: {e}")
            finally:
                self.device = None
    
    def configure(self, frequency: int, lna_gain: int = None, 
                  vga_gain: int = None, enable_amp: bool = None):
        """Configure HackRF parameters"""
        debug_print(f"configure() called: freq={frequency}, lna={lna_gain}, vga={vga_gain}, amp={enable_amp}")
        if not self.device:
            debug_print("⚠️  No device to configure")
            return False
        
        try:
            self.device.freq = frequency
            debug_print(f"📡 Frequency: {frequency / 1e6:.3f} MHz")
            
            if lna_gain is not None:
                self.device.lna_gain = lna_gain
                debug_print(f"🔊 LNA Gain: {lna_gain} dB")
            
            if vga_gain is not None:
                self.device.vga_gain = vga_gain
                debug_print(f"🔊 VGA Gain: {vga_gain} dB")
            
            if enable_amp is not None:
                self.device.enable_amp = enable_amp
                debug_print(f"📶 RF Amp: {'ON' if enable_amp else 'OFF'}")
            
            return True
            
        except Exception as e:
            debug_print(f"❌ Configuration error: {e}")
            traceback.print_exc()
            return False
    
    def start_rx(self, callback: Callable):
        """Start receiving samples"""
        debug_print("start_rx() called")
        if not self.device:
            debug_print("❌ Device not opened")
            return False
        
        if self.is_streaming:
            debug_print("⚠️  Already streaming")
            return False
        
        try:
            debug_print(f"Setting callback: {callback}")
            self.callback = callback
            self.is_streaming = True
            self._stop_requested = False
            self.samples_received = 0
            self.errors = 0
            self.callback_count = 0
            
            debug_print("🎯 Calling device.start_rx()...")
            self.device.start_rx(self._rx_callback)
            debug_print("✅ RX streaming started")
            
            return True
            
        except Exception as e:
            debug_print(f"❌ Failed to start RX: {e}")
            traceback.print_exc()
            self.is_streaming = False
            return False
    
    def stop_rx(self):
        """Stop receiving samples"""
        debug_print("stop_rx() called")
        if not self.is_streaming:
            debug_print("Not streaming, nothing to stop")
            return
        
        debug_print("🛑 Stopping RX...")
        self._stop_requested = True
        self.is_streaming = False
        
        # Give callbacks time to finish
        debug_print("Waiting for callbacks to finish...")
        time.sleep(0.2)
        
        if self.device:
            try:
                debug_print("Calling device.stop_rx()...")
                self.device.stop_rx()
                debug_print(f"⏹️  Stopped RX (callbacks={self.callback_count}, samples={self.samples_received}, errors={self.errors})")
            except Exception as e:
                debug_print(f"⚠️  Error stopping RX: {e}")
                traceback.print_exc()
        
        # Clear buffer
        with self.lock:
            self.sample_buffer.clear()
        debug_print("Buffer cleared")
    
    def start(self, callback: Callable):
        """Start scanning (wrapper)"""
        return self.start_rx(callback)
    
    def stop(self):
        """Stop scanning (wrapper)"""
        return self.stop_rx()
    
    def get_status(self) -> dict:
        """Get current status"""
        freq = config.DEFAULT_FREQUENCY
        if self.device:
            try:
                freq = self.device.freq
            except:
                pass
        
        return {
            'is_streaming': self.is_streaming,
            'frequency': freq / 1e6,
            'frequency_hopping': False,
            'hop_interval': 30.0,
            'frequency_stats': {}
        }
    
    def change_frequency(self, frequency: int):
        """Change frequency"""
        if self.device:
            try:
                self.device.freq = frequency
                debug_print(f"📡 Changed to {frequency / 1e6:.3f} MHz")
                return True
            except Exception as e:
                debug_print(f"❌ Failed to change frequency: {e}")
                return False
        return False
    
    def set_frequency_hopping(self, enabled: bool):
        return False
    
    def set_hop_interval(self, interval: float):
        return False
    
    def increment_detection(self, frequency: float):
        pass
    
    def _rx_callback(self, hackrf_transfer):
        """Internal callback for HackRF data - MUST be fast and safe"""
        self.callback_count += 1
        current_time = time.time()
        
        # Log first few callbacks and then periodically
        if self.callback_count <= 5 or self.callback_count % 100 == 0:
            debug_print(f"_rx_callback #{self.callback_count} called (dt={current_time - self.last_callback_time:.3f}s)")
        
        self.last_callback_time = current_time
        
        # Quick exit if stop requested
        if self._stop_requested or not self.is_streaming:
            debug_print("Stop requested, returning -1")
            return -1
        
        try:
            # Get buffer safely
            if self.callback_count <= 3:
                debug_print(f"  Getting buffer from transfer...")
            
            buffer = hackrf_transfer.buffer
            if buffer is None:
                if self.callback_count <= 3:
                    debug_print(f"  Buffer is None!")
                return 0
            
            buffer_len = len(buffer)
            if self.callback_count <= 3:
                debug_print(f"  Buffer length: {buffer_len}")
            
            if buffer_len < 2:
                if self.callback_count <= 3:
                    debug_print(f"  Buffer too small: {buffer_len}")
                return 0
            
            # Convert bytes to numpy array
            if self.callback_count <= 3:
                debug_print(f"  Converting to numpy array...")
            
            raw_data = np.frombuffer(buffer, dtype=np.int8)
            
            if len(raw_data) < 2:
                return 0
            
            # Separate I and Q
            if self.callback_count <= 3:
                debug_print(f"  Separating I/Q...")
            
            i_data = raw_data[0::2].astype(np.float32) / 128.0
            q_data = raw_data[1::2].astype(np.float32) / 128.0
            
            # Create complex samples
            iq_samples = i_data + 1j * q_data
            
            self.samples_received += len(iq_samples)
            
            if self.callback_count <= 3:
                debug_print(f"  Created {len(iq_samples)} IQ samples")
            
            # Store in buffer with copy
            if self.callback_count <= 3:
                debug_print(f"  Storing in buffer...")
            
            with self.lock:
                self.sample_buffer.append(iq_samples.copy())
            
            # Call user callback if we have enough samples
            if len(iq_samples) >= config.SAMPLES_PER_SCAN and self.callback:
                if self.callback_count <= 3:
                    debug_print(f"  Calling user callback...")
                
                try:
                    # Calculate signal strength
                    power = np.abs(iq_samples[:config.SAMPLES_PER_SCAN]) ** 2
                    signal_strength = 10 * np.log10(np.mean(power) + 1e-10)
                    
                    # Get frequency
                    freq = config.DEFAULT_FREQUENCY
                    try:
                        if self.device:
                            freq = self.device.freq
                    except:
                        pass
                    
                    if self.callback_count <= 3:
                        debug_print(f"  Invoking callback with strength={signal_strength:.1f}, freq={freq}")
                    
                    # Call callback with copy - THIS IS WHERE IT MIGHT CRASH
                    self.callback(
                        iq_samples[:config.SAMPLES_PER_SCAN].copy(),
                        signal_strength,
                        freq
                    )
                    
                    if self.callback_count <= 3:
                        debug_print(f"  Callback completed successfully")
                    
                except Exception as e:
                    self.errors += 1
                    if self.errors <= 5:
                        debug_print(f"❌ Callback error #{self.errors}: {e}")
                        traceback.print_exc()
            
            if self.callback_count <= 3:
                debug_print(f"  Returning 0 (continue)")
            
            return 0
            
        except Exception as e:
            self.errors += 1
            if self.errors <= 5:
                debug_print(f"❌ RX callback error #{self.errors}: {e}")
                traceback.print_exc()
            if self.errors > 1000:
                debug_print("⚠️  Too many errors, stopping")
                return -1
            return 0
    
    def capture_samples(self, num_samples: int = None, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Capture samples (blocking)"""
        if num_samples is None:
            num_samples = config.SAMPLES_PER_SCAN
        
        if not self.is_streaming:
            return None
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.lock:
                if len(self.sample_buffer) > 0:
                    samples = self.sample_buffer.popleft()
                    if len(samples) >= num_samples:
                        return samples[:num_samples]
            time.sleep(0.01)
        
        return None
    
    def get_statistics(self) -> dict:
        """Get interface statistics"""
        return {
            'is_streaming': self.is_streaming,
            'samples_received': self.samples_received,
            'errors': self.errors,
            'buffer_size': len(self.sample_buffer),
            'sample_rate': config.SAMPLE_RATE,
            'callback_count': self.callback_count,
        }



class SimulatedHackRF:
    """Simulated HackRF for testing"""
    
    def __init__(self):
        self.is_streaming = False
        self.callback = None
        self.frequency = 315_000_000
        self.thread = None
        
    def open(self) -> bool:
        print("🔧 Using simulated HackRF")
        return True
    
    def close(self):
        self.stop_rx()
    
    def configure(self, frequency: int, **kwargs):
        self.frequency = frequency
        return True
    
    def start_rx(self, callback: Callable):
        if self.is_streaming:
            return False
        
        self.callback = callback
        self.is_streaming = True
        
        self.thread = threading.Thread(target=self._simulate_samples, daemon=True)
        self.thread.start()
        
        print("🎯 Started simulated RX")
        return True
    
    def stop_rx(self):
        self.is_streaming = False
        if self.thread:
            self.thread.join(timeout=1.0)
        print("⏹️  Stopped simulated RX")
    
    def start(self, callback: Callable):
        return self.start_rx(callback)
    
    def stop(self):
        return self.stop_rx()
    
    def get_status(self) -> dict:
        return {
            'is_streaming': self.is_streaming,
            'frequency': self.frequency / 1e6,
            'frequency_hopping': False,
            'hop_interval': 30.0,
            'frequency_stats': {}
        }
    
    def change_frequency(self, frequency: int):
        self.frequency = frequency
        print(f"📡 Simulated: {frequency / 1e6:.3f} MHz")
        return True
    
    def set_frequency_hopping(self, enabled: bool):
        return False
    
    def set_hop_interval(self, interval: float):
        return False
    
    def increment_detection(self, frequency: float):
        pass
    
    def _simulate_samples(self):
        while self.is_streaming:
            noise = (np.random.randn(config.SAMPLES_PER_SCAN) + 
                    1j * np.random.randn(config.SAMPLES_PER_SCAN)) * 0.1
            
            if np.random.random() < 0.1:
                t = np.arange(config.SAMPLES_PER_SCAN) / config.SAMPLE_RATE
                carrier_freq = 50000
                symbol_rate = 19200
                
                num_bits = int(len(t) * symbol_rate / config.SAMPLE_RATE)
                bits = np.random.randint(0, 2, num_bits)
                
                samples_per_bit = config.SAMPLE_RATE // symbol_rate
                bit_signal = np.repeat(bits, samples_per_bit)[:len(t)]
                
                freq_deviation = 20000
                inst_freq = carrier_freq + (bit_signal - 0.5) * 2 * freq_deviation
                phase = 2 * np.pi * np.cumsum(inst_freq) / config.SAMPLE_RATE
                signal = 0.5 * np.exp(1j * phase)
                
                noise += signal
            
            if self.callback:
                power = np.abs(noise) ** 2
                signal_strength = 10 * np.log10(np.mean(power) + 1e-10)
                self.callback(noise, signal_strength, self.frequency)
            
            time.sleep(config.SCAN_DWELL_TIME)
    
    def capture_samples(self, num_samples: int = None, timeout: float = 1.0):
        if num_samples is None:
            num_samples = config.SAMPLES_PER_SCAN
        
        noise = (np.random.randn(num_samples) + 
                1j * np.random.randn(num_samples)) * 0.1
        return noise
    
    def get_statistics(self):
        return {
            'is_streaming': self.is_streaming,
            'samples_received': 0,
            'errors': 0,
            'buffer_size': 0,
            'sample_rate': config.SAMPLE_RATE,
        }

def create_hackrf_interface(use_simulation: bool = False):
    """Create HackRF interface"""
    if use_simulation or not HACKRF_AVAILABLE:
        return SimulatedHackRF()
    else:
        return HackRFInterface()


