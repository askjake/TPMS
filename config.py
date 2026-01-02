"""
Configuration settings for TPMS Scanner
Optimized for Maurader TPMSRX compatibility
"""
from dataclasses import dataclass, field
from typing import List
from pathlib import Path

@dataclass
class Config:
    # Add to Config class
    HISTOGRAM_BINS: int = 50  # Number of bins for signal histogram
    DEFAULT_FREQUENCY: int = 315_000_000  # Default frequency
    BANDWIDTH: int = 1_750_000  # 1.75 MHz bandwidth
    DEFAULT_GAIN: int = 32  # Default LNA gain

    # HackRF Settings - matching Maurader baseband
    SAMPLE_RATE: int = 2_457_600  # 2.4576 MHz - exact Maurader rate
    CENTER_FREQ: int = 315_000_000  # 315 MHz default
    
    # Gain Settings (conservative defaults)
    LNA_GAIN: int = 32  # 0-40 dB (start moderate)
    VGA_GAIN: int = 30  # 0-62 dB (start moderate)
    ENABLE_AMP: bool = True  # RF amp on
    
    # TPMS Frequency Bands - using default_factory for mutable default
    TPMS_FREQUENCIES: List[int] = field(default_factory=lambda: [
        314_900_000,  # 314.9 MHz (US)
        315_000_000,  # 315.0 MHz (US)
        433_920_000,  # 433.92 MHz (EU)
    ])
    
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
    SIGNAL_HISTORY_SIZE: int = 1000  # Number of signals to keep in history
    
    # Database Settings
    DB_PATH: str = "tpms_tracker.db"  # SQLite database path
    DB_BACKUP_ENABLED: bool = True
    DB_BACKUP_INTERVAL: int = 3600  # seconds (1 hour)
    
    # Logging
    LOG_ENABLED: bool = True
    LOG_UNKNOWN_SIGNALS: bool = True
    LOG_RAW_SAMPLES: bool = False  # Warning: creates large files
    LOG_DIR: str = "logs"
    LOG_FILE: str = "tpms_scanner.log"
    LOG_MAX_SIZE: int = 10_485_760  # 10 MB
    LOG_BACKUP_COUNT: int = 5
    
    # Performance
    USE_MULTIPROCESSING: bool = True
    MAX_WORKERS: int = 2  # CPU cores for processing
    
    # UI Settings
    REFRESH_RATE: float = 1.0  # seconds
    PLOT_HISTORY_SECONDS: int = 60  # seconds of history to plot
    ENABLE_ANIMATIONS: bool = True
    
    # Advanced Settings
    DEBUG_MODE: bool = False
    SIMULATION_MODE: bool = False  # Force simulation mode
    SAVE_RAW_IQ: bool = False  # Save raw IQ samples
    IQ_SAVE_DIR: str = "iq_samples"
    
    # ESP32 Trigger Settings (if using ESP32 trigger)
    ESP32_ENABLED: bool = False
    ESP32_PORT: str = "/dev/ttyUSB0"
    ESP32_BAUD: int = 115200
    ESP32_TRIGGER_DURATION: float = 0.5  # seconds
    
    # Alert Settings
    ENABLE_ALERTS: bool = False
    ALERT_ON_NEW_SENSOR: bool = True
    ALERT_SOUND: bool = False
    
    # Export Settings
    EXPORT_FORMAT: str = "csv"  # csv, json, or both
    EXPORT_DIR: str = "exports"
    AUTO_EXPORT: bool = False
    AUTO_EXPORT_INTERVAL: int = 3600  # seconds

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

# Ensure directories exist
def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        config.LOG_DIR,
        config.IQ_SAVE_DIR,
        config.EXPORT_DIR,
        "data",
        "models",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

# Call on import
ensure_directories()

