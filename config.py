
"""
Configuration for TPMS Tracker
Optimized to match native HackRF TPMS app performance
"""
from pathlib import Path

class Config:
    # Database
    DB_PATH = "tpms_tracker.db"
    
    # HackRF Settings - MATCH NATIVE APP
    SAMPLE_RATE = 2_457_600  # Match HackRF native (was 2_000_000)
    BANDWIDTH = 1_750_000    # Match HackRF native (was not set)
    DEFAULT_GAIN = 20        # LNA gain
    VGA_GAIN = 20           # VGA gain
    RF_AMP = 0              # RF amp (0=off, 1=on)
    
    # Gain Settings
    GAIN_MIN = 0
    GAIN_MAX = 47
    GAIN_STEP = 4
    
    # Frequency Configuration - MATCH NATIVE APP
    DEFAULT_FREQUENCY = 314.9e6  # 314.9 MHz (primary)
    FREQUENCY_LIST = [
        314.9e6,  # 314.9 MHz - most common
        315.0e6,  # 315.0 MHz - alternate
        433.92e6  # 433.92 MHz - European
    ]
    FREQUENCIES = [314.9, 433.92, 315.0]  # MHz format for compatibility
    
    # Frequency Hopping - DISABLED for better reception
    ENABLE_FREQUENCY_HOPPING = False  # Changed from True
    FREQUENCY_HOP_ENABLED = False     # Duplicate for compatibility
    FREQUENCY_HOP_INTERVAL = 60       # seconds (if enabled)
    FREQUENCY_HOP_DWELL_TIME = 1.1    # settling time after hop
    ADAPTIVE_HOPPING = False          # Changed from True
    
    # Signal Detection - LOWERED based on your captures
    MIN_SIGNAL_STRENGTH = -95    # dBm (very sensitive)
    SIGNAL_THRESHOLD = -95       # dBm
    DETECTION_THRESHOLD = -95    # dBm
    PEAK_POWER_THRESHOLD = -95   # dBm
    
    # Demodulation Settings - MATCH NATIVE APP
    FSK_DEVIATION = 19200       # Hz - for FSK signals
    SYMBOL_RATE_FSK = 19200     # baud - FSK Schrader
    SYMBOL_RATE_OOK = 8192      # baud - OOK Schrader
    
    # Vehicle Clustering - Allow single sensor detection
    MIN_SENSORS_FOR_VEHICLE = 1  # Show even single sensors
    CLUSTER_TIME_WINDOW = 30     # seconds
    MIN_SIGNALS_PER_VEHICLE = 1
    MAX_SIGNAL_AGE = 300         # seconds
    
    # Processing
    MAX_QUEUE_SIZE = 1000
    SIGNAL_BUFFER_SIZE = 262144  # Larger buffer for continuous reception
    
    # Signal Processing
    FFT_SIZE = 2048
    DECIMATION = 4
    
    # Protocol Detection
    PROTOCOL_DETECTION_ENABLED = True
    PROTOCOL_MIN_SAMPLES = 100  # Minimum samples before attempting protocol detection
    KNOWN_PROTOCOLS = [
        'Toyota/Lexus',
        'Schrader',
        'Continental',
        'TRW',
        'Pacific',
        'Beru',
        'Huf/Beru',
        'Jansite'
    ]
    
    # Histogram Settings
    HISTOGRAM_BINS = 50
    HISTOGRAM_WINDOW = 300  # seconds of data to display
    SIGNAL_HISTORY_SIZE = 10000  # Max samples to keep
    
    # ML Settings
    MIN_ENCOUNTERS_FOR_PREDICTION = 5
    PREDICTION_CONFIDENCE_THRESHOLD = 0.6
    
    # Trigger Settings
    LF_TRIGGER_FREQUENCY = 125000  # 125 kHz
    TRIGGER_PULSE_WIDTH = 0.1      # 100ms
    TRIGGER_TX_GAIN = 47           # Maximum TX gain
    ENABLE_TRIGGER = True          # Enable trigger functionality


# Create singleton instance
config = Config()
