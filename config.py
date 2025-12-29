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
    
    # Signal Detection
    MIN_SIGNAL_STRENGTH = -70  # dBm
    SIGNAL_THRESHOLD = -60  # dBm for detection
    
    # Gain Settings
    GAIN_MIN = 0
    GAIN_MAX = 47
    GAIN_STEP = 4
    DEFAULT_GAIN = 20
    
    # Signal Processing
    FFT_SIZE = 2048
    DECIMATION = 4
    
    # Vehicle Clustering
    MIN_SENSORS_FOR_VEHICLE = 1  # Lowered from 4 for testing
    CLUSTER_TIME_WINDOW = 30  # seconds
    MIN_SIGNALS_PER_VEHICLE = 1  # Lowered from 4
    MAX_SIGNAL_AGE = 300  # seconds

    
    # Protocol Detection
    PROTOCOL_DETECTION_ENABLED = True
    PROTOCOL_MIN_SAMPLES = 10  # Minimum samples before attempting protocol detection
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

