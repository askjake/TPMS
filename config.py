"""
Configuration settings for TPMS Scanner
Optimized for Maurader TPMSRX compatibility
"""
from dataclasses import dataclass
from typing import List

@dataclass
class Config:
    # HackRF Settings - matching Maurader baseband
    SAMPLE_RATE: int = 2_457_600  # 2.4576 MHz - exact Maurader rate
    CENTER_FREQ: int = 315_000_000  # 315 MHz default
    
    # Gain Settings (conservative defaults)
    LNA_GAIN: int = 32  # 0-40 dB (start moderate)
    VGA_GAIN: int = 30  # 0-62 dB (start moderate)
    ENABLE_AMP: bool = True  # RF amp on
    
    # TPMS Frequency Bands
    TPMS_FREQUENCIES: List[int] = [
        314_900_000,  # 314.9 MHz (US)
        315_000_000,  # 315.0 MHz (US)
        433_920_000,  # 433.92 MHz (EU)
    ]
    
    # Signal Detection Thresholds
    SIGNAL_THRESHOLD: float = -70.0  # dBm - lowered for better sensitivity
    MIN_SNR: float = 6.0  # dB - minimum SNR for valid decode
    
    # Symbol Rates (matching Maurader exactly)
    SYMBOL_RATE_FSK: int = 19_200  # Schrader FSK
    SYMBOL_RATE_OOK: int = 8_192   # Schrader OOK primary
    SYMBOL_RATE_OOK_ALT: int = 8_400  # Schrader OOK alternate
    
    # FSK Deviation
    FSK_DEVIATION: int = 38_400  # 2x symbol rate for Schrader
    
    # Capture Settings
    SAMPLES_PER_SCAN: int = 262_144  # ~107ms at 2.4576 MHz
    SCAN_DWELL_TIME: float = 0.15  # seconds per frequency
    SCAN_INTERVAL: float = 0.5  # seconds between scans
    
    # Buffer Settings
    BUFFER_SIZE: int = 262_144  # Must match SAMPLES_PER_SCAN
    NUM_BUFFERS: int = 4
    
    # Protocol Detection
    PROTOCOL_DETECTION_ENABLED: bool = True
    MAX_UNKNOWN_SIGNALS: int = 100
    
    # Packet Validation
    MIN_PACKET_LENGTH: int = 64  # bits
    MAX_PACKET_LENGTH: int = 128  # bits
    MAX_PREAMBLE_ERRORS: int = 2  # bit errors allowed in preamble
    
    # Learning Engine
    LEARNING_ENABLED: bool = True
    LEARNING_WINDOW: int = 100  # signals to analyze
    ADAPTATION_THRESHOLD: float = 0.7  # confidence threshold
    
    # Display Settings
    MAX_DISPLAY_SENSORS: int = 20
    SENSOR_TIMEOUT: float = 300.0  # 5 minutes
    
    # Logging
    LOG_ENABLED: bool = True
    LOG_UNKNOWN_SIGNALS: bool = True
    LOG_RAW_SAMPLES: bool = False  # Warning: creates large files
    
    # Performance
    USE_MULTIPROCESSING: bool = True
    MAX_WORKERS: int = 2  # CPU cores for processing

# Global config instance
config = Config()

# Frequency presets with names
FREQUENCY_PRESETS = {
    'US_314.9': 314_900_000,
    'US_315.0': 315_000_000,
    'EU_433.92': 433_920_000,
}

# Protocol-specific parameters
PROTOCOL_PARAMS = {
    'Schrader_FSK': {
        'symbol_rate': 19_200,
        'deviation': 38_400,
        'preamble': [0x55, 0x55],
        'packet_bytes': 10,
    },
    'Schrader_OOK_8k192': {
        'symbol_rate': 8_192,
        'preamble': [0xAA, 0xAA],
        'packet_bytes': 8,
    },
    'Schrader_OOK_8k4': {
        'symbol_rate': 8_400,
        'preamble': [0xAA, 0xAA],
        'packet_bytes': 8,
    },
    'Toyota': {
        'symbol_rate': 10_000,
        'deviation': 20_000,
        'preamble': [0x55, 0x55, 0x55],
        'packet_bytes': 10,
    },
}
