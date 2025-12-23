import os
from dataclasses import dataclass
from typing import List


@dataclass
class TPMSConfig:
    # Frequencies to scan (in MHz)
    FREQUENCIES: List[float] = None

    # Signal processing
    SAMPLE_RATE: int = 2_000_000  # 2 MHz
    GAIN_MIN: int = 0
    GAIN_MAX: int = 47
    GAIN_STEP: int = 2

    # Detection thresholds
    SNR_THRESHOLD: float = 10.0  # dB
    MIN_SIGNAL_STRENGTH: float = -80  # dBm

    # Vehicle clustering
    TPMS_PER_VEHICLE: int = 4
    CLUSTER_TIME_WINDOW: int = 30  # seconds
    CLUSTER_DISTANCE_THRESHOLD: float = 0.1  # km

    # Database
    DB_PATH: str = "tpms_tracker.db"

    # ML Parameters
    MIN_ENCOUNTERS_FOR_PATTERN: int = 3
    SIMILARITY_THRESHOLD: float = 0.85

    def __post_init__(self):
        if self.FREQUENCIES is None:
            self.FREQUENCIES = [314.9, 315.0, 433.92]  # MHz


config = TPMSConfig()
