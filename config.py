"""
Configuration settings for TPMS Tracker
"""
from pathlib import Path

class Config:
    # Database
    DB_PATH = "tpms_tracker.db"
    
    # HackRF Settings
    SAMPLE_RATE = 2_000_000  # 2 MSPS
    
    # TPMS Frequencies (MHz)
    FREQUENCIES = [314.9, 433.92, 315.0]  # Common TPMS frequencies
    
    # Frequency Hopping Settings
    FREQUENCY_HOP_ENABLED = True
    FREQUENCY_HOP_INTERVAL = 30.0  # seconds per frequency
    FREQUENCY_HOP_DWELL_TIME = 1.1  # settling time after hop
    
    # Signal Detection - Based on successful captures
    MIN_SIGNAL_STRENGTH = -90  # dBm (lowered from -70)
    SIGNAL_THRESHOLD = -90  # dBm for detection
    DETECTION_THRESHOLD = -90  # dBm - captures worked at -86 dBm
    PEAK_POWER_THRESHOLD = -90  # dBm (lowered from -85)

    # Vehicle Clustering - Allow single sensor detection
    MIN_SENSORS_FOR_VEHICLE = 1  # Show even single sensors
    CLUSTER_TIME_WINDOW = 30  # seconds
    MIN_SIGNALS_PER_VEHICLE = 1

    
    # Gain Settings
    GAIN_MIN = 0
    GAIN_MAX = 47
    GAIN_STEP = 4
    DEFAULT_GAIN = 20
    
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
    
    # Vehicle Clustering
    CLUSTER_TIME_WINDOW = 30  # seconds
    MIN_SIGNALS_PER_VEHICLE = 4
    MAX_SIGNAL_AGE = 300  # seconds
    
    # ML Settings
    MIN_ENCOUNTERS_FOR_PREDICTION = 5
    PREDICTION_CONFIDENCE_THRESHOLD = 0.6

config = Config()

