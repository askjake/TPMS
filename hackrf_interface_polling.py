### hackrf_interface_polling.py
import logging
import time
import numpy as np
from typing import Callable, Optional
 import threading

# Import our wrapper instead of non-existent 'hackrf' module
try:
    from hackrf_wrapper import HackRFDevice, is_available
    HACKRF_AVAILABLE = is_available()
except Exception as e:
    logging.error(f"Failed to import HackRF wrapper: {e}")
    HACKRF_AVAILABLE = False
    HackRFDevice = None

logger = logging.getLogger(__name__)

class HackRFInterface:
    def __init__(self):
        logger.info("HackRFInterface.__init__ called")
        
        self.device = None
        self.is_running = False
        self.callback = None
        self.frequency = 314.9e6
        self.sample_rate = 2_457_600
        self.lna_gain = 32
        self.vga_gain = 40
        
        if not HACKRF_AVAILABLE:
            logger.warning("HackRF library not available - simulation mode")
            return
        
        # Try to open device
        try:
            self.device = HackRFDevice()
            if self.device.open():
                logger.info("✅ HackRF device opened successfully")
                self._configure_device()
            else:
                logger.error("❌ Failed to open HackRF device")
                self.device = None
        except Exception as e:
            logger.error(f"❌ Error initializing HackRF: {e}")
            self.device = None
    
    def _configure_device(self):
        """Configure device with default settings"""
        if not self.device:
            return
        
        self.device.set_freq(int(self.frequency))
        self.device.set_sample_rate(self.sample_rate)
        self.device.set_lna_gain(self.lna_gain)
        self.device.set_vga_gain(self.vga_gain)
        self.device.set_amp_enable(True)
    
    def start(self, callback: Callable):
        """Start receiving"""
        logger.info("start_rx() called")
        
        if not self.device:
            logger.error("❌ Device not opened")
            return False
        
        self.callback = callback
        
        def rx_callback(iq_data):
            """Internal callback wrapper"""
            try:
                # Convert int8 to complex float
                iq_complex = (iq_data[::2] + 1j * iq_data[1::2]).astype(np.complex64) / 128.0
                
                # Calculate RSSI
                power = np.mean(np.abs(iq_complex) ** 2)
                rssi = 10 * np.log10(power + 1e-10) - 50  # Rough calibration
                
                # Call user callback
                if self.callback:
                    self.callback(iq_complex, rssi, self.frequency)
            except Exception as e:
                logger.error(f"Callback error: {e}")
        
        if self.device.start_rx(rx_callback):
            self.is_running = True
            logger.info("✅ RX started successfully")
            return True
        else:
            logger.error("❌ Failed to start RX")
            return False
    
    def stop(self):
        """Stop receiving"""
        if self.device and self.is_running:
            self.device.stop_rx()
            self.is_running = False
            logger.info("RX stopped")
    
    def change_frequency(self, freq_hz: float):
        """Change center frequency"""
        self.frequency = freq_hz
        if self.device:
            self.device.set_freq(int(freq_hz))
    
    def get_status(self):
        """Get current status"""
        return {
            'frequency': self.frequency / 1e6,
            'is_running': self.is_running,
            'sample_rate': self.sample_rate,
            'lna_gain': self.lna_gain,
            'vga_gain': self.vga_gain
        }
    
    def __del__(self):
        """Cleanup"""
        if self.device:
            self.stop()
            self.device.close()
