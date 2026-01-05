"""
Multi-Scanner Manager
Manages multiple SDR devices scanning different frequencies simultaneously
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ScannerConfig:
    """Configuration for a single scanner"""
    scanner_id: str
    hardware_type: str  # 'hackrf', 'rtlsdr', 'simulation'
    frequency: float  # in Hz
    scanner: any  # The actual scanner object
    is_running: bool = False


class MultiScannerManager:
    """Manages multiple SDR scanners simultaneously"""

    def __init__(self, db, decoder_class, sample_rate: int = 2_457_600):
        self.db = db
        self.decoder_class = decoder_class
        self.sample_rate = sample_rate

        # Dictionary of scanner_id -> ScannerConfig
        self.scanners: Dict[str, ScannerConfig] = {}

        # Lock for thread-safe operations
        self._lock = threading.Lock()

        logger.info("MultiScannerManager initialized")

    def add_scanner(self, scanner_id: str, hardware_type: str,
                    frequency: float, scanner) -> bool:
        """
        Add a scanner to the manager

        Args:
            scanner_id: Unique identifier (e.g., "scanner_1", "hackrf_314.9")
            hardware_type: Type of hardware ('hackrf', 'rtlsdr', 'simulation')
            frequency: Frequency in Hz
            scanner: The scanner object

        Returns:
            True if added successfully
        """
        with self._lock:
            if scanner_id in self.scanners:
                logger.warning(f"Scanner {scanner_id} already exists")
                return False

            # Create decoder instance for this scanner
            decoder = self.decoder_class(self.sample_rate)

            # Attach pipeline
            scanner.attach_pipeline(db=self.db, decoder=decoder)

            # Set frequency
            scanner.change_frequency(frequency)

            config = ScannerConfig(
                scanner_id=scanner_id,
                hardware_type=hardware_type,
                frequency=frequency,
                scanner=scanner,
                is_running=False
            )

            self.scanners[scanner_id] = config
            logger.info(f"Added scanner {scanner_id}: {hardware_type} @ {frequency / 1e6:.2f} MHz")
            return True

    def remove_scanner(self, scanner_id: str) -> bool:
        """Remove a scanner from the manager"""
        with self._lock:
            if scanner_id not in self.scanners:
                return False

            config = self.scanners[scanner_id]

            # Stop if running
            if config.is_running:
                try:
                    config.scanner.stop()
                except Exception as e:
                    logger.error(f"Error stopping scanner {scanner_id}: {e}")

            del self.scanners[scanner_id]
            logger.info(f"Removed scanner {scanner_id}")
            return True

    def start_scanner(self, scanner_id: str) -> bool:
        """Start a specific scanner"""
        with self._lock:
            if scanner_id not in self.scanners:
                logger.error(f"Scanner {scanner_id} not found")
                return False

            config = self.scanners[scanner_id]

            if config.is_running:
                logger.warning(f"Scanner {scanner_id} already running")
                return True

            try:
                if config.scanner.start():
                    config.is_running = True
                    logger.info(f"Started scanner {scanner_id}")
                    return True
                else:
                    logger.error(f"Failed to start scanner {scanner_id}")
                    return False
            except Exception as e:
                logger.error(f"Error starting scanner {scanner_id}: {e}")
                return False

    def stop_scanner(self, scanner_id: str) -> bool:
        """Stop a specific scanner"""
        with self._lock:
            if scanner_id not in self.scanners:
                return False

            config = self.scanners[scanner_id]

            if not config.is_running:
                return True

            try:
                config.scanner.stop()
                config.is_running = False
                logger.info(f"Stopped scanner {scanner_id}")
                return True
            except Exception as e:
                logger.error(f"Error stopping scanner {scanner_id}: {e}")
                return False

    def start_all(self) -> Dict[str, bool]:
        """Start all scanners"""
        results = {}
        for scanner_id in list(self.scanners.keys()):
            results[scanner_id] = self.start_scanner(scanner_id)
        return results

    def stop_all(self) -> Dict[str, bool]:
        """Stop all scanners"""
        results = {}
        for scanner_id in list(self.scanners.keys()):
            results[scanner_id] = self.stop_scanner(scanner_id)
        return results

    def change_frequency(self, scanner_id: str, frequency: float) -> bool:
        """Change frequency for a specific scanner"""
        with self._lock:
            if scanner_id not in self.scanners:
                return False

            config = self.scanners[scanner_id]

            try:
                config.scanner.change_frequency(frequency)
                config.frequency = frequency
                logger.info(f"Changed frequency for {scanner_id} to {frequency / 1e6:.2f} MHz")
                return True
            except Exception as e:
                logger.error(f"Error changing frequency for {scanner_id}: {e}")
                return False

    def get_status(self) -> Dict[str, Dict]:
        """Get status of all scanners"""
        with self._lock:
            status = {}
            for scanner_id, config in self.scanners.items():
                try:
                    scanner_status = config.scanner.get_status()
                    status[scanner_id] = {
                        'hardware_type': config.hardware_type,
                        'frequency': config.frequency / 1e6,  # MHz
                        'is_running': config.is_running,
                        'status': scanner_status
                    }
                except Exception as e:
                    status[scanner_id] = {
                        'hardware_type': config.hardware_type,
                        'frequency': config.frequency / 1e6,
                        'is_running': config.is_running,
                        'error': str(e)
                    }
            return status

    def get_scanner_count(self) -> int:
        """Get number of active scanners"""
        return len(self.scanners)

    def get_running_count(self) -> int:
        """Get number of running scanners"""
        with self._lock:
            return sum(1 for config in self.scanners.values() if config.is_running)

    def cleanup(self):
        """Stop all scanners and cleanup"""
        logger.info("Cleaning up MultiScannerManager")
        self.stop_all()
        with self._lock:
            self.scanners.clear()

