"""
GPS Manager
Supports multiple GPS sources: built-in, Bluetooth, USB, GPSD
"""

import logging
import threading
import time
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class GPSData:
    """GPS data container"""
    latitude: float
    longitude: float
    altitude: Optional[float] = None
    speed: Optional[float] = None
    heading: Optional[float] = None
    timestamp: float = 0.0
    satellites: int = 0
    fix_quality: int = 0  # 0=no fix, 1=GPS, 2=DGPS


class GPSManager:
    """Unified GPS manager supporting multiple sources"""

    def __init__(self):
        self.current_location: Optional[GPSData] = None
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Try to detect available GPS sources
        self.gps_source = None
        self.gps_client = None

        logger.info("GPSManager initialized")

    def detect_gps_sources(self) -> Dict[str, bool]:
        """Detect available GPS sources"""
        sources = {
            'gpsd': self._check_gpsd(),
            'android_bluetooth': False,  # Requires manual pairing
            'usb_gps': self._check_usb_gps(),
            'geoclue': self._check_geoclue(),
        }
        return sources

    def _check_gpsd(self) -> bool:
        """Check if GPSD is available"""
        try:
            import gps
            client = gps.gps(mode=gps.WATCH_ENABLE | gps.WATCH_NEWSTYLE)
            client.close()
            logger.info("✅ GPSD available")
            return True
        except Exception as e:
            logger.debug(f"GPSD not available: {e}")
            return False

    def _check_usb_gps(self) -> bool:
        """Check for USB GPS devices"""
        import os
        import glob

        # Common USB GPS device paths
        gps_devices = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')

        if gps_devices:
            logger.info(f"✅ Found potential USB GPS devices: {gps_devices}")
            return True
        return False

    def _check_geoclue(self) -> bool:
        """Check if GeoClue (Linux location service) is available"""
        try:
            import subprocess
            result = subprocess.run(['which', 'geoclue'], capture_output=True)
            if result.returncode == 0:
                logger.info("✅ GeoClue available")
                return True
        except Exception:
            pass
        return False

    def start_gpsd(self, host: str = 'localhost', port: int = 2947):
        """Start GPS using GPSD"""
        try:
            import gps

            self.gps_client = gps.gps(host=host, port=port, mode=gps.WATCH_ENABLE | gps.WATCH_NEWSTYLE)
            self.gps_source = 'gpsd'
            self.is_running = True

            self._thread = threading.Thread(target=self._gpsd_worker, daemon=True)
            self._thread.start()

            logger.info(f"✅ Started GPS using GPSD ({host}:{port})")
            return True
        except Exception as e:
            logger.error(f"Failed to start GPSD: {e}")
            return False

    def start_android_bluetooth(self, device_address: str):
        """Start GPS using Android phone via Bluetooth

        Requires:
        1. Phone paired via Bluetooth
        2. GPS sharing app on phone (e.g., "Bluetooth GPS Provider")
        3. BlueZ on Linux
        """
        try:
            import serial

            # Android Bluetooth GPS typically creates a serial port
            # after pairing with an app like "Bluetooth GPS Provider"
            rfcomm_port = f'/dev/rfcomm0'  # May vary

            self.gps_client = serial.Serial(rfcomm_port, baudrate=9600, timeout=1)
            self.gps_source = 'android_bluetooth'
            self.is_running = True

            self._thread = threading.Thread(target=self._nmea_worker, daemon=True)
            self._thread.start()

            logger.info(f"✅ Started GPS using Android Bluetooth ({device_address})")
            return True
        except Exception as e:
            logger.error(f"Failed to start Android Bluetooth GPS: {e}")
            return False

    def start_usb_gps(self, device_path: str = '/dev/ttyUSB0'):
        """Start GPS using USB GPS dongle"""
        try:
            import serial

            self.gps_client = serial.Serial(device_path, baudrate=9600, timeout=1)
            self.gps_source = 'usb_gps'
            self.is_running = True

            self._thread = threading.Thread(target=self._nmea_worker, daemon=True)
            self._thread.start()

            logger.info(f"✅ Started GPS using USB device ({device_path})")
            return True
        except Exception as e:
            logger.error(f"Failed to start USB GPS: {e}")
            return False

    def start_mock_gps(self, lat: float = 40.7128, lon: float = -74.0060):
        """Start mock GPS for testing (defaults to New York City)"""
        self.current_location = GPSData(
            latitude=lat,
            longitude=lon,
            altitude=10.0,
            timestamp=time.time(),
            satellites=8,
            fix_quality=1
        )
        self.gps_source = 'mock'
        self.is_running = True

        self._thread = threading.Thread(target=self._mock_worker, daemon=True)
        self._thread.start()

        logger.info(f"✅ Started mock GPS at ({lat}, {lon})")
        return True

    def _gpsd_worker(self):
        """Worker thread for GPSD"""
        import gps

        while not self._stop_event.is_set() and self.is_running:
            try:
                report = self.gps_client.next()

                if report['class'] == 'TPV':
                    if hasattr(report, 'lat') and hasattr(report, 'lon'):
                        self.current_location = GPSData(
                            latitude=report.lat,
                            longitude=report.lon,
                            altitude=getattr(report, 'alt', None),
                            speed=getattr(report, 'speed', None),
                            heading=getattr(report, 'track', None),
                            timestamp=time.time(),
                            satellites=getattr(report, 'satellites_used', 0),
                            fix_quality=getattr(report, 'mode', 0)
                        )
                        logger.debug(
                            f"GPS: {self.current_location.latitude:.6f}, {self.current_location.longitude:.6f}")
            except StopIteration:
                logger.warning("GPSD connection lost")
                break
            except Exception as e:
                logger.error(f"GPSD worker error: {e}")
                time.sleep(1)

    def _nmea_worker(self):
        """Worker thread for NMEA-based GPS (USB/Bluetooth)"""
        import pynmea2

        while not self._stop_event.is_set() and self.is_running:
            try:
                line = self.gps_client.readline().decode('ascii', errors='ignore').strip()

                if line.startswith('$GPGGA') or line.startswith('$GNGGA'):
                    msg = pynmea2.parse(line)
                    if msg.latitude and msg.longitude:
                        self.current_location = GPSData(
                            latitude=msg.latitude,
                            longitude=msg.longitude,
                            altitude=msg.altitude if hasattr(msg, 'altitude') else None,
                            timestamp=time.time(),
                            satellites=int(msg.num_sats) if hasattr(msg, 'num_sats') else 0,
                            fix_quality=int(msg.gps_qual) if hasattr(msg, 'gps_qual') else 0
                        )
                        logger.debug(
                            f"GPS: {self.current_location.latitude:.6f}, {self.current_location.longitude:.6f}")
            except Exception as e:
                logger.error(f"NMEA worker error: {e}")
                time.sleep(1)

    def _mock_worker(self):
        """Worker thread for mock GPS (simulates movement)"""
        while not self._stop_event.is_set() and self.is_running:
            if self.current_location:
                # Simulate slight movement (random walk)
                import random
                self.current_location.latitude += random.uniform(-0.0001, 0.0001)
                self.current_location.longitude += random.uniform(-0.0001, 0.0001)
                self.current_location.timestamp = time.time()

            time.sleep(1)

    def stop(self):
        """Stop GPS"""
        self.is_running = False
        self._stop_event.set()

        if self._thread:
            self._thread.join(timeout=2.0)

        if self.gps_client:
            try:
                if self.gps_source == 'gpsd':
                    self.gps_client.close()
                elif self.gps_source in ('usb_gps', 'android_bluetooth'):
                    self.gps_client.close()
            except Exception:
                pass

        logger.info("GPS stopped")

    def get_location(self) -> Optional[Tuple[float, float]]:
        """Get current location as (lat, lon) tuple"""
        if self.current_location:
            return (self.current_location.latitude, self.current_location.longitude)
        return None

    def get_full_data(self) -> Optional[GPSData]:
        """Get full GPS data"""
        return self.current_location

    def get_status(self) -> Dict:
        """Get GPS status"""
        if not self.current_location:
            return {
                'active': self.is_running,
                'source': self.gps_source,
                'has_fix': False
            }

        return {
            'active': self.is_running,
            'source': self.gps_source,
            'has_fix': True,
            'latitude': self.current_location.latitude,
            'longitude': self.current_location.longitude,
            'altitude': self.current_location.altitude,
            'satellites': self.current_location.satellites,
            'fix_quality': self.current_location.fix_quality,
            'age_seconds': time.time() - self.current_location.timestamp
        }
