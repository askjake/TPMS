"""
HackRF Interface for TPMS Signal Capture
Modern Python (3.10+) compatible
"""
import numpy as np
from typing import Optional, Callable
import time
import threading
from collections import deque
from config import config

# Try to import HackRF library
HACKRF_AVAILABLE = False
HackRF = None
HackRFError = Exception  # Default fallback

print("🔍 Attempting to import HackRF library...")

try:
    # pyhackrf package installs as 'hackrf' module
    import hackrf
    print(f"✅ Found hackrf module at: {hackrf.__file__}")
    
    from hackrf import HackRF
    
    # Try different error class names
    try:
        from hackrf import HackRfError as HackRFError
        print("   Using HackRfError")
    except ImportError:
        try:
            from hackrf import HackRFError
            print("   Using HackRFError")
        except ImportError:
            print("   No specific error class found, using Exception")
            HackRFError = Exception
    
    HACKRF_AVAILABLE = True
    print("✅ HackRF library loaded successfully")
    
except ImportError as e:
    print(f"❌ hackrf import failed: {e}")
    print("⚠️  HackRF library not available - using simulation mode")
    print("   Install with: pip install pyhackrf")

class HackRFInterface:
    def __init__(self):
        self.device: Optional[HackRF] = None
        self.is_streaming = False
        self.callback: Optional[Callable] = None
        self.sample_buffer = deque(maxlen=config.NUM_BUFFERS)
        self.lock = threading.Lock()
        self.samples_received = 0
        self.errors = 0
        self._stop_requested = False
        
        # Try to open device on init if library is available
        if HACKRF_AVAILABLE:
            print("🔧 Attempting to open HackRF device...")
            success = self.open()
            if success:
                # Configure with defaults
                self.configure(
                    config.DEFAULT_FREQUENCY,
                    config.LNA_GAIN,
                    config.VGA_GAIN,
                    config.ENABLE_AMP
                )
                print("✅ HackRF ready (not streaming)")
        
    def open(self) -> bool:
        """Open HackRF device"""
        if not HACKRF_AVAILABLE:
            print("⚠️  HackRF library not available")
            return False
        
        try:
            self.device = HackRF()
            print("✅ HackRF device opened")
            
            # Set sample rate
            self.device.sample_rate = config.SAMPLE_RATE
            print(f"📊 Sample rate: {config.SAMPLE_RATE / 1e6:.3f} MHz")
            
            # Set baseband filter bandwidth
            bandwidth = int(config.SAMPLE_RATE * 0.75)
            self.device.baseband_filter_bandwidth = bandwidth
            print(f"🔧 Baseband filter: {bandwidth / 1e6:.3f} MHz")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to open HackRF: {e}")
            return False
    
    def close(self):
        """Close HackRF device"""
        self.stop_rx()
        
        if self.device:
            try:
                self.device.close()
                print("📻 HackRF closed")
            except Exception as e:
                print(f"⚠️  Error closing HackRF: {e}")
            finally:
                self.device = None
    
    def configure(self, frequency: int, lna_gain: int = None, 
                  vga_gain: int = None, enable_amp: bool = None):
        """Configure HackRF parameters"""
        if not self.device:
            print("⚠️  No device to configure")
            return False
        
        try:
            # Set frequency
            self.device.freq = frequency
            print(f"📡 Frequency: {frequency / 1e6:.3f} MHz")
            
            # Set gains
            if lna_gain is not None:
                self.device.lna_gain = lna_gain
                print(f"🔊 LNA Gain: {lna_gain} dB")
            
            if vga_gain is not None:
                self.device.vga_gain = vga_gain
                print(f"🔊 VGA Gain: {vga_gain} dB")
            
            # Set RF amp
            if enable_amp is not None:
                self.device.enable_amp = enable_amp
                print(f"📶 RF Amp: {'ON' if enable_amp else 'OFF'}")
            
            return True
            
        except Exception as e:
            print(f"❌ Configuration error: {e}")
            return False
    
    def start_rx(self, callback: Callable):
        """Start receiving samples"""
        if not self.device:
            print("❌ Device not opened")
            return False
        
        if self.is_streaming:
            print("⚠️  Already streaming")
            return False
        
        try:
            self.callback = callback
            self.is_streaming = True
            self._stop_requested = False
            self.samples_received = 0
            self.errors = 0
            
            print("🎯 Starting RX streaming...")
            self.device.start_rx(self._rx_callback)
            print("✅ RX streaming started")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start RX: {e}")
            import traceback
            traceback.print_exc()
            self.is_streaming = False
            return False
    
    def stop_rx(self):
        """Stop receiving samples"""
        if not self.is_streaming:
            return
        
        print("🛑 Stopping RX...")
        self._stop_requested = True
        self.is_streaming = False
        
        # Give callbacks time to finish
        time.sleep(0.2)
        
        if self.device:
            try:
                self.device.stop_rx()
                print(f"⏹️  Stopped RX (received {self.samples_received} samples, {self.errors} errors)")
            except Exception as e:
                print(f"⚠️  Error stopping RX: {e}")
        
        # Clear buffer
        with self.lock:
            self.sample_buffer.clear()
    
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
                print(f"📡 Changed to {frequency / 1e6:.3f} MHz")
                return True
            except Exception as e:
                print(f"❌ Failed to change frequency: {e}")
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
        # Quick exit if stop requested
        if self._stop_requested or not self.is_streaming:
            return -1
        
        try:
            # Get buffer safely
            buffer = hackrf_transfer.buffer
            if buffer is None or len(buffer) < 2:
                return 0
            
            # Convert bytes to numpy array
            raw_data = np.frombuffer(buffer, dtype=np.int8)
            
            if len(raw_data) < 2:
                return 0
            
            # Separate I and Q
            i_data = raw_data[0::2].astype(np.float32) / 128.0
            q_data = raw_data[1::2].astype(np.float32) / 128.0
            
            # Create complex samples
            iq_samples = i_data + 1j * q_data
            
            self.samples_received += len(iq_samples)
            
            # Store in buffer with copy
            with self.lock:
                self.sample_buffer.append(iq_samples.copy())
            
            # Call user callback if we have enough samples
            if len(iq_samples) >= config.SAMPLES_PER_SCAN and self.callback:
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
                    
                    # Call callback with copy
                    self.callback(
                        iq_samples[:config.SAMPLES_PER_SCAN].copy(),
                        signal_strength,
                        freq
                    )
                except Exception as e:
                    self.errors += 1
                    if self.errors % 100 == 0:
                        print(f"⚠️  Callback errors: {self.errors}")
            
            return 0
            
        except Exception as e:
            self.errors += 1
            if self.errors < 5:  # Only print first few errors
                print(f"❌ RX callback error: {e}")
            if self.errors > 1000:
                print("⚠️  Too many errors, stopping")
                return -1
            return 0
    
    
    def get_statistics(self) -> dict:
        """Get interface statistics"""
        return {
            'is_streaming': self.is_streaming,
            'samples_received': self.samples_received,
            'errors': self.errors,
            'buffer_size': len(self.sample_buffer),
            'sample_rate': config.SAMPLE_RATE,
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

