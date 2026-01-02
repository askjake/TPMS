"""
HackRF Library Wrapper using ctypes
Direct interface to libhackrf.so without external dependencies
"""

import ctypes
import ctypes.util
import numpy as np
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)

# Find libhackrf
_libhackrf_path = ctypes.util.find_library('hackrf')
if not _libhackrf_path:
    # Try common paths
    for path in ['/usr/lib/x86_64-linux-gnu/libhackrf.so',
                 '/usr/lib/x86_64-linux-gnu/libhackrf.so.0',
                 '/usr/local/lib/libhackrf.so',
                 '/usr/lib/libhackrf.so']:
        try:
            _libhackrf = ctypes.CDLL(path)
            _libhackrf_path = path
            logger.info(f"✅ Loaded libhackrf from {path}")
            break
        except:
            continue
    
    if not _libhackrf_path:
        logger.error("❌ libhackrf not found!")
        _libhackrf = None
else:
    _libhackrf = ctypes.CDLL(_libhackrf_path)
    logger.info(f"✅ Loaded libhackrf from {_libhackrf_path}")

# HackRF constants
HACKRF_SUCCESS = 0
HACKRF_TRUE = 1
HACKRF_ERROR_INVALID_PARAM = -2
HACKRF_ERROR_NOT_FOUND = -5
HACKRF_ERROR_LIBUSB = -1000

# Transfer structure (matches hackrf.h)
class hackrf_transfer(ctypes.Structure):
    _fields_ = [
        ("device", ctypes.c_void_p),
        ("buffer", ctypes.POINTER(ctypes.c_uint8)),
        ("buffer_length", ctypes.c_int),
        ("valid_length", ctypes.c_int),
        ("rx_ctx", ctypes.c_void_p),
        ("tx_ctx", ctypes.c_void_p)
    ]

# Callback type - takes pointer to hackrf_transfer struct
hackrf_sample_block_cb_fn = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(hackrf_transfer)
)


class HackRFDevice:
    """Python wrapper for HackRF device"""
    
    def __init__(self):
        self.device = None
        self.callback = None
        self.is_streaming = False
        self.python_callback = None
        
        if _libhackrf is None:
            raise RuntimeError("libhackrf not available")
    
    def open(self) -> bool:
        """Open HackRF device"""
        try:
            # Initialize library
            result = _libhackrf.hackrf_init()
            if result != HACKRF_SUCCESS:
                logger.error(f"hackrf_init failed: {result}")
                return False
            
            # Open device
            device_ptr = ctypes.c_void_p()
            result = _libhackrf.hackrf_open(ctypes.byref(device_ptr))
            
            if result != HACKRF_SUCCESS:
                logger.error(f"hackrf_open failed: {result}")
                return False
            
            self.device = device_ptr
            logger.info("✅ HackRF device opened successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to open HackRF: {e}")
            return False
    
    def close(self):
        """Close HackRF device"""
        if self.device:
            try:
                if self.is_streaming:
                    self.stop_rx()
                _libhackrf.hackrf_close(self.device)
                _libhackrf.hackrf_exit()
                self.device = None
                logger.info("HackRF device closed")
            except Exception as e:
                logger.error(f"Error closing HackRF: {e}")
    
    def set_freq(self, freq_hz: int) -> bool:
        """Set center frequency in Hz"""
        if not self.device:
            return False
        
        result = _libhackrf.hackrf_set_freq(self.device, ctypes.c_uint64(freq_hz))
        if result == HACKRF_SUCCESS:
            logger.info(f"Frequency set to {freq_hz / 1e6:.2f} MHz")
            return True
        else:
            logger.error(f"Failed to set frequency: {result}")
            return False
    
    def set_sample_rate(self, rate_hz: int) -> bool:
        """Set sample rate in Hz"""
        if not self.device:
            return False
        
        result = _libhackrf.hackrf_set_sample_rate(self.device, ctypes.c_double(rate_hz))
        if result == HACKRF_SUCCESS:
            logger.info(f"Sample rate set to {rate_hz / 1e6:.2f} MS/s")
            return True
        else:
            logger.error(f"Failed to set sample rate: {result}")
            return False
    
    def set_lna_gain(self, gain_db: int) -> bool:
        """Set LNA gain (0-40 dB, 8 dB steps)"""
        if not self.device:
            return False
        
        # Clamp to valid range and round to nearest 8
        gain_db = max(0, min(40, gain_db))
        gain_db = (gain_db // 8) * 8
        
        result = _libhackrf.hackrf_set_lna_gain(self.device, ctypes.c_uint32(gain_db))
        if result == HACKRF_SUCCESS:
            logger.info(f"LNA gain set to {gain_db} dB")
            return True
        else:
            logger.error(f"Failed to set LNA gain: {result}")
            return False
    
    def set_vga_gain(self, gain_db: int) -> bool:
        """Set VGA gain (0-62 dB, 2 dB steps)"""
        if not self.device:
            return False
        
        # Clamp to valid range and round to nearest 2
        gain_db = max(0, min(62, gain_db))
        gain_db = (gain_db // 2) * 2
        
        result = _libhackrf.hackrf_set_vga_gain(self.device, ctypes.c_uint32(gain_db))
        if result == HACKRF_SUCCESS:
            logger.info(f"VGA gain set to {gain_db} dB")
            return True
        else:
            logger.error(f"Failed to set VGA gain: {result}")
            return False
    
    def set_amp_enable(self, enable: bool) -> bool:
        """Enable/disable RF amplifier"""
        if not self.device:
            return False
        
        result = _libhackrf.hackrf_set_amp_enable(
            self.device,
            ctypes.c_uint8(1 if enable else 0)
        )
        if result == HACKRF_SUCCESS:
            logger.info(f"RF amp {'enabled' if enable else 'disabled'}")
            return True
        else:
            logger.error(f"Failed to set amp enable: {result}")
            return False
    
    def start_rx(self, callback: Callable) -> bool:
        """Start receiving with callback"""
        if not self.device:
            logger.error("Device not opened")
            return False
        
        # Store Python callback
        self.python_callback = callback
        
        # Create C callback wrapper
        def c_callback(transfer_ptr):
            try:
                # Dereference the transfer structure
                transfer = transfer_ptr.contents
                
                # Get buffer length
                buffer_length = transfer.valid_length
                
                if buffer_length <= 0:
                    logger.warning(f"Empty buffer: valid_length={buffer_length}")
                    return 0
                
                # Extract data from buffer
                # Data is interleaved I/Q as signed 8-bit integers
                buffer_array = ctypes.cast(
                    transfer.buffer,
                    ctypes.POINTER(ctypes.c_int8 * buffer_length)
                ).contents
                
                # Convert to numpy array
                iq_data = np.frombuffer(buffer_array, dtype=np.int8)
                
                # Call Python callback
                if self.python_callback:
                    self.python_callback(iq_data)
                
                return 0  # Success
                
            except Exception as e:
                logger.error(f"Callback error: {e}", exc_info=True)
                return -1
        
        # Store callback to prevent garbage collection
        self.callback = hackrf_sample_block_cb_fn(c_callback)
        
        # Start RX
        result = _libhackrf.hackrf_start_rx(
            self.device,
            self.callback,
            None  # user data (not used)
        )
        
        if result == HACKRF_SUCCESS:
            self.is_streaming = True
            logger.info("✅ RX streaming started")
            return True
        else:
            logger.error(f"Failed to start RX: {result}")
            return False
    
    def stop_rx(self) -> bool:
        """Stop receiving"""
        if not self.device or not self.is_streaming:
            return False
        
        result = _libhackrf.hackrf_stop_rx(self.device)
        if result == HACKRF_SUCCESS:
            self.is_streaming = False
            logger.info("RX streaming stopped")
            return True
        else:
            logger.error(f"Failed to stop RX: {result}")
            return False
    
    def is_streaming_rx(self) -> bool:
        """Check if currently receiving"""
        if not self.device:
            return False
        
        result = _libhackrf.hackrf_is_streaming(self.device)
        return result == HACKRF_TRUE


# Test if library is available
def is_available() -> bool:
    """Check if libhackrf is available"""
    return _libhackrf is not None


def get_version() -> Optional[str]:
    """Get library version"""
    if not _libhackrf:
        return None
    
    try:
        version = ctypes.create_string_buffer(128)
        _libhackrf.hackrf_library_version(version)
        return version.value.decode('utf-8')
    except:
        return "unknown"

