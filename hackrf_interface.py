"""
HackRF Interface for TPMS Signal Capture
Optimized for reliable sample capture
"""
import numpy as np
from typing import Optional, Callable
import time
import threading
from collections import deque
from config import config

try:
    from hackrf import HackRF, HackRFError
    HACKRF_AVAILABLE = True
except ImportError:
    HACKRF_AVAILABLE = False
    print("⚠️  PyHackRF not available - using simulation mode")

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
            print("⚠️  HackRF library not available")
            return False
        
        try:
            self.device = HackRF()
            
            # Get device info
            info = self.device.board_id_read()
            print(f"📻 HackRF detected: {info}")
            
            # Set sample rate
            self.device.sample_rate = config.SAMPLE_RATE
            print(f"📊 Sample rate: {config.SAMPLE_RATE / 1e6:.3f} MHz")
            
            # Set baseband filter bandwidth (slightly wider than sample rate)
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
            # Set frequency
            self.device.freq = frequency
            print(f"📡 Frequency: {frequency / 1e6:.3f} MHz")
            
            # Set gains
            if lna_gain is not None:
                self.device.lna_gain = lna_gain
                print(f"🔊 LNA Gain: {lna_gain} dB")
            else:
                self.device.lna_gain = config.LNA_GAIN
            
            if vga_gain is not None:
                self.device.vga_gain = vga_gain
                print(f"🔊 VGA Gain: {vga_gain} dB")
            else:
                self.device.vga_gain = config.VGA_GAIN
            
            # Set RF amp
            if enable_amp is not None:
                self.device.enable_amp = enable_amp
                print(f"📶 RF Amp: {'ON' if enable_amp else 'OFF'}")
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
            print("⚠️  Already streaming")
            return False
        
        try:
            self.callback = callback
            self.is_streaming = True
            self.samples_received = 0
            self.errors = 0
            
            # Start streaming with callback
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
            print(f"⏹️  Stopped RX (received {self.samples_received} samples, {self.errors} errors)")
            
        except Exception as e:
            print(f"⚠️  Error stopping RX: {e}")
    
    def _rx_callback(self, hackrf_transfer):
        """Internal callback for HackRF data"""
        if not self.is_streaming:
            return -1  # Stop streaming
        
        try:
            # Convert bytes to numpy array
            # HackRF provides interleaved I/Q as signed 8-bit integers
            raw_data = np.frombuffer(hackrf_transfer.buffer, dtype=np.int8)
            
            # Separate I and Q
            i_data = raw_data[0::2].astype(np.float32) / 128.0
            q_data = raw_data[1::2].astype(np.float32) / 128.0
            
            # Create complex samples
            iq_samples = i_data + 1j * q_data
            
            self.samples_received += len(iq_samples)
            
            # Store in buffer
            with self.lock:
                self.sample_buffer.append(iq_samples)
            
            # Call user callback if we have enough samples
            if len(iq_samples) >= config.SAMPLES_PER_SCAN:
                if self.callback:
                    try:
                        self.callback(iq_samples[:config.SAMPLES_PER_SCAN])
                    except Exception as e:
                        print(f"⚠️  Callback error: {e}")
                        self.errors += 1
            
            return 0  # Continue streaming
            
        except Exception as e:
            print(f"❌ RX callback error: {e}")
            self.errors += 1
            return -1  # Stop on error
    
    def capture_samples(self, num_samples: int = None, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Capture a specific number of samples (blocking)"""
        if num_samples is None:
            num_samples = config.SAMPLES_PER_SCAN
        
        if not self.is_streaming:
            print("⚠️  Not streaming - call start_rx() first")
            return None
        
        # Wait for samples with timeout
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.lock:
                if len(self.sample_buffer) > 0:
                    samples = self.sample_buffer.popleft()
                    if len(samples) >= num_samples:
                        return samples[:num_samples]
            time.sleep(0.01)
        
        print("⚠️  Capture timeout")
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
    """Simulated HackRF for testing without hardware"""
    
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
        print(f"📡 Simulated frequency: {frequency / 1e6:.3f} MHz")
        return True
    
    def start_rx(self, callback: Callable):
        if self.is_streaming:
            return False
        
        self.callback = callback
        self.is_streaming = True
        
        # Start simulation thread
        self.thread = threading.Thread(target=self._simulate_samples, daemon=True)
        self.thread.start()
        
        print("🎯 Started simulated RX")
        return True
    
    def stop_rx(self):
        self.is_streaming = False
        if self.thread:
            self.thread.join(timeout=1.0)
        print("⏹️  Stopped simulated RX")
    
    def _simulate_samples(self):
        """Generate simulated TPMS-like signals"""
        while self.is_streaming:
            # Generate noise
            noise = (np.random.randn(config.SAMPLES_PER_SCAN) + 
                    1j * np.random.randn(config.SAMPLES_PER_SCAN)) * 0.1
            
            # Occasionally add a signal
            if np.random.random() < 0.1:  # 10% chance
                # Simulate FSK signal
                t = np.arange(config.SAMPLES_PER_SCAN) / config.SAMPLE_RATE
                carrier_freq = 50000  # 50 kHz offset
                symbol_rate = 19200
                
                # Generate random bits
                num_bits = int(len(t) * symbol_rate / config.SAMPLE_RATE)
                bits = np.random.randint(0, 2, num_bits)
                
                # Upsample bits
                samples_per_bit = config.SAMPLE_RATE // symbol_rate
                bit_signal = np.repeat(bits, samples_per_bit)[:len(t)]
                
                # FSK modulation
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
        
        # Generate and return samples immediately
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

# Factory function
def create_hackrf_interface(use_simulation: bool = False) -> HackRFInterface:
    """Create HackRF interface (real or simulated)"""
    if use_simulation or not HACKRF_AVAILABLE:
        interface = SimulatedHackRF()
    else:
        interface = HackRFInterface()
    
    return interface
