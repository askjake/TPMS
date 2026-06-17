"""
Multi-Scanner Manager
Manages multiple SDR devices scanning different frequencies simultaneously
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Tuple, Callable, Type
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ScannerConfig:
    """Configuration for a single scanner"""
    scanner_id: str
    hardware_type: str  # 'hackrf', 'rtlsdr', 'simulation'
    frequency: float  # in Hz
    scanner: any  # The actual scanner object
    decoder: any  # The decoder instance for this scanner
    is_running: bool = False


class MultiScannerManager:
    """Manages multiple SDR scanners simultaneously"""

    def __init__(self, db, decoder_class, sample_rate: int = 2_457_600, gps_manager=None):
        """
        Initialize MultiScannerManager

        Args:
            db: Database instance (TPMSDatabase object)
            decoder_class: TPMS decoder class
            sample_rate: Sample rate in Hz
            gps_manager: Optional GPS manager instance
        """
        self.db = db  # Store the database OBJECT, not connection
        self.decoder_class = decoder_class
        self.sample_rate = sample_rate
        self.gps_manager = gps_manager

        # Dictionary of scanner_id -> ScannerConfig
        self.scanners: Dict[str, ScannerConfig] = {}

        # Lock for thread-safe operations
        self._lock = threading.Lock()

        logger.info("MultiScannerManager initialized")
        if self.gps_manager:
            logger.info("GPS manager attached - location data will be added to signals")

    def add_scanner(self, scanner_id: str, hardware_type: str,
                    frequency: float, scanner) -> bool:
        """Add a scanner to the manager"""
        with self._lock:
            if scanner_id in self.scanners:
                logger.warning(f"Scanner {scanner_id} already exists")
                return False

            try:
                # 🔥 Pass the database OBJECT, not a connection
                decoder = self.decoder_class(self.sample_rate, self.db)

                logger.info(f"Created decoder instance for {scanner_id}")
            except Exception as e:
                logger.error(f"Failed to create decoder for {scanner_id}: {e}")
                return False

            # Create signal callback that adds GPS coordinates
            def on_signal_with_gps(signal_dict):
                """Callback that enriches signal data with GPS coordinates before saving"""
                try:
                    # Add GPS coordinates if GPS manager is available
                    if self.gps_manager:
                        gps_status = self.gps_manager.get_status()

                        # Only add coordinates if we have a valid GPS fix
                        if gps_status.get('active') and gps_status.get('has_fix'):
                            signal_dict['latitude'] = gps_status['latitude']
                            signal_dict['longitude'] = gps_status['longitude']
                            logger.debug(
                                f"Added GPS coordinates to signal: {gps_status['latitude']:.6f}, {gps_status['longitude']:.6f}")
                        else:
                            signal_dict['latitude'] = None
                            signal_dict['longitude'] = None
                            if gps_status.get('active'):
                                logger.debug("GPS active but no fix - signal saved without coordinates")
                    else:
                        signal_dict['latitude'] = None
                        signal_dict['longitude'] = None

                    # Insert signal into database (now with GPS data if available)
                    self.db.insert_signal(signal_dict)

                except Exception as e:
                    logger.error(f"Error processing signal with GPS: {e}")
                    # Still try to save the signal without GPS
                    try:
                        signal_dict['latitude'] = None
                        signal_dict['longitude'] = None
                        self.db.insert_signal(signal_dict)
                    except Exception as e2:
                        logger.error(f"Failed to save signal even without GPS: {e2}")

            # Attach pipeline with GPS-enhanced callback
            try:
                scanner.attach_pipeline(db=self.db, decoder=decoder, on_signal=on_signal_with_gps)
                logger.info(f"Attached pipeline to scanner {scanner_id}")
            except Exception as e:
                logger.error(f"Failed to attach pipeline to {scanner_id}: {e}")
                return False

            # Set frequency
            try:
                scanner.change_frequency(frequency)
                logger.info(f"Set frequency for {scanner_id} to {frequency / 1e6:.2f} MHz")
            except Exception as e:
                logger.error(f"Failed to set frequency for {scanner_id}: {e}")
                return False

            config = ScannerConfig(
                scanner_id=scanner_id,
                hardware_type=hardware_type,
                frequency=frequency,
                scanner=scanner,
                decoder=decoder,
                is_running=False
            )

            self.scanners[scanner_id] = config
            logger.info(f"✅ Added scanner {scanner_id}: {hardware_type} @ {frequency / 1e6:.2f} MHz")
            return True

    def set_gps_manager(self, gps_manager):
        """
        Set or update the GPS manager

        Args:
            gps_manager: GPS manager instance
        """
        self.gps_manager = gps_manager
        logger.info("GPS manager updated")

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

                    # Get decoder statistics if available
                    decoder_stats = {}
                    if hasattr(config.decoder, 'get_statistics'):
                        try:
                            decoder_stats = config.decoder.get_statistics()
                        except Exception:
                            pass

                    status[scanner_id] = {
                        'hardware_type': config.hardware_type,
                        'frequency': config.frequency / 1e6,  # MHz
                        'is_running': config.is_running,
                        'status': scanner_status,
                        'decoder_stats': decoder_stats
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

    def get_gps_status(self) -> Dict:
        """Get current GPS status"""
        if self.gps_manager:
            return self.gps_manager.get_status()
        return {'active': False, 'has_fix': False}

    def get_decoder_statistics(self) -> Dict:
        """Get aggregated decoder statistics from all scanners"""
        stats = {
            'total_success': 0,
            'total_failed': 0,
            'total_reprocessed': 0,
            'protocols_discovered': 0,
            'by_scanner': {}
        }

        with self._lock:
            for scanner_id, config in self.scanners.items():
                if hasattr(config.decoder, 'get_statistics'):
                    try:
                        scanner_stats = config.decoder.get_statistics()
                        stats['by_scanner'][scanner_id] = scanner_stats
                        stats['total_success'] += scanner_stats.get('total_successful', 0)
                        stats['total_failed'] += scanner_stats.get('total_failed', 0)
                        stats['total_reprocessed'] += scanner_stats.get('reprocessing_success', 0)
                        stats['protocols_discovered'] += scanner_stats.get('discovery_count', 0)
                    except Exception as e:
                        logger.debug(f"Error getting stats from {scanner_id}: {e}")

        return stats

    def cleanup(self):
        """Stop all scanners and cleanup"""
        logger.info("Cleaning up MultiScannerManager")
        self.stop_all()
        with self._lock:
            self.scanners.clear()
