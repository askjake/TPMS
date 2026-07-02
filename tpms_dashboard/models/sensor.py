"""
Sensor data model — dataclass representation.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class SensorObservation:
    """A single TPMS sensor observation record."""
    sensor_id: str
    protocol: str
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    pressure_psi: Optional[float] = None
    pressure_kpa: Optional[float] = None
    temp_c: Optional[float] = None
    temp_f: Optional[float] = None
    battery_low: bool = False
    rssi_dbm: Optional[float] = None
    packet_count: int = 0
    flags: str = "0x00"
    magic_ok: bool = True
    suspect_temp: bool = False

    @property
    def is_known_protocol(self) -> bool:
        """Return True if protocol is a recognized named protocol."""
        from tpms_dashboard.utils.protocol import KNOWN_PROTOCOLS
        return self.protocol in KNOWN_PROTOCOLS

    @property
    def normalized_protocol(self) -> str:
        """Return display-friendly protocol name."""
        from tpms_dashboard.utils.protocol import normalize_protocol
        return normalize_protocol(self.protocol)
