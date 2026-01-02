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
HackRFError = None

try:
    from pyhackrf import HackRF, HackRFError
    HACKRF_AVAILABLE = True
    print("✅ PyHackRF library loaded")
except ImportError:
    try:
        from hackrf import HackRF, HackRFError
        HACKRF_AVAILABLE = True
        print("✅ HackRF library loaded")
    except ImportError:
        print("⚠️  HackRF library not available - using simulation mode")

class HackRFInterface:
    def __init__(self):
        self.device: Optional[HackRF] = None
        self.is_streaming = False
        self.callback: Optional[Callable] = None
        self.sample_buffer = deque(maxlen=config.NUM_BUFFERS)
        self.lock = threading.Lock()
        self.samples_received = 0
        self.errors = 0
        
    def open(self) -> bool:
        """Open HackRF device"""
        if not HACKRF_AVAILABLE:
            return False
        
        try:
            self.device = HackRF()
            
            try:
                info = self.device.board_id_read()
                print(f"📻 HackRF detected: {info}")
            except:
                print(f"📻 HackRF detected")
            
            self.device.sample_rate = config.SAMPLE_RATE
            print(f"📊 Sample rate: {config.SAMPLE_RATE / 1e6:.3f} MHz")
            
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
            return False
        
        try:
            self.device.freq = frequency
            print(f"📡 Frequency: {frequency / 1e6:.3f} MHz")
            
            if lna_gain is not None:
                self.device.lna_gain = lna_gain
            else:
                self.device.lna_gain = config.LNA_GAIN
            
            if vga_gain is not None:
                self.device.vga_gain = vga_gain
            else:
                self.device.vga_gain = config.VGA_GAIN
            
            if enable_amp is not None:
                self.device.enable_amp = enable_amp
            else:
                self.device.enable_amp = config.ENABLE_AMP
            
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
            return False
        
        try:
            self.callback = callback
            self.is_streaming = True
            self.samples_received = 0
            self.errors = 0
            
            self.device.start_rx(self._rx_callback)
            print("🎯 Started RX streaming")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start RX: {e}")
            self.is_streaming = False
            return False
    
    def stop_rx(self):
        """Stop receiving samples"""
        if not self.device or not self.is_streaming:
            return
        
        try:
            self.is_streaming = False
            self.device.stop_rx()
            print(f"⏹️  Stopped RX")
            
        except Exception as e:
            print(f"⚠️  Error stopping RX: {e}")
    
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
        """Enable/disable frequency hopping"""
        return False
    
    def set_hop_interval(self, interval: float):
        """Set hop interval"""
        return False
    
    def increment_detection(self, frequency: float):
        """Increment detection count"""
        pass
    
    def _rx_callback(self, hackrf_transfer):
        """Internal callback for HackRF data"""
        if not self.is_streaming:
            return -1
        
        try:
            raw_data = np.frombuffer(hackrf_transfer.buffer, dtype=np.int8)
            
            i_data = raw_data[0::2].astype(np.float32) / 128.0
            q_data = raw_data[1::2].astype(np.float32) / 128.0
            
            iq_samples = i_data + 1j * q_data
            
            self.samples_received += len(iq_samples)
            
            with self.lock:
                self.sample_buffer.append(iq_samples)
            
            if len(iq_samples) >= config.SAMPLES_PER_SCAN:
                if self.callback:
                    try:
                        self.callback(iq_samples[:config.SAMPLES_PER_SCAN])
                    except Exception as e:
                        print(f"⚠️  Callback error: {e}")
                        self.errors += 1
            
            return 0
            
        except Exception as e:
            print(f"❌ RX callback error: {e}")
            self.errors += 1
            return -1
    
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
        """Start scanning (wrapper)"""
        return self.start_rx(callback)
    
    def stop(self):
        """Stop scanning (wrapper)"""
        return self.stop_rx()
    
    def get_status(self) -> dict:
        """Get current status"""
        return {
            'is_streaming': self.is_streaming,
            'frequency': self.frequency / 1e6,
            'frequency_hopping': False,
            'hop_interval': 30.0,
            'frequency_stats': {}
        }
    
    def change_frequency(self, frequency: int):
        """Change frequency"""
        self.frequency = frequency
        print(f"📡 Simulated: {frequency / 1e6:.3f} MHz")
        return True
    
    def set_frequency_hopping(self, enabled: bool):
        """Enable/disable frequency hopping"""
        return False
    
    def set_hop_interval(self, interval: float):
        """Set hop interval"""
        return False
    
    def increment_detection(self, frequency: float):
        """Increment detection count"""
        pass
    
    def _simulate_samples(self):
        """Generate simulated signals"""
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
                self.callback(noise)
            
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
