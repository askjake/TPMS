"""
TPMS Dashboard Configuration
=============================
All filesystem paths, display names, and app metadata in one place.
Paths are sourced from environment variables with sensible defaults.
UI MUST use display names — never render raw paths.
"""
import os
from pathlib import Path

# --- App Identity ---
APP_NAME = "TPMS Dashboard"
APP_VERSION = "2.1.0"
APP_SUBTITLE = "BLE/RF Sensor Fleet Monitor"
PAGE_ICON = "\U0001f6de"

# --- Filesystem Paths (from env vars, never shown in UI) ---
_BASE_DIR = Path(os.getenv("TPMS_BASE_DIR", str(Path(__file__).parent.parent)))
EXPORT_DIR = Path(os.getenv("TPMS_EXPORT_DIR", str(_BASE_DIR / "exports")))
ARCHIVE_DIR = Path(os.getenv("TPMS_ARCHIVE_DIR", str(EXPORT_DIR / "archive")))

# --- Display Names (shown in UI instead of paths) ---
EXPORT_DISPLAY_NAME = "Daily Exports"
ARCHIVE_DISPLAY_NAME = "Archive Store"

# --- Performance Tuning ---
DEFAULT_PAGE_SIZE = 100
MAX_CHART_POINTS = 5000
LARGE_FILE_THRESHOLD_MB = 5

# --- Views ---
VIEWS = {
    "Daily Sync Overview": {
        "icon": "\U0001f4cb",
        "description": "Summary of recent daily export files \u2014 best for quick checks",
    },
    "Archive Explorer": {
        "icon": "\U0001f5c4\ufe0f",
        "description": "Browse and query large historical tracker archives",
    },
    "Sensor History": {
        "icon": "\U0001f4e1",
        "description": "Per-sensor observation timeline and pressure history",
    },
    "Observation Patterns": {
        "icon": "\U0001f50d",
        "description": "Time-of-day and frequency patterns across all sensors",
    },
    "Sensor Search": {
        "icon": "\U0001f50e",
        "description": "Search by sensor ID, protocol, or pressure range",
    },
    "Trends & Charts": {
        "icon": "\U0001f4c8",
        "description": "Fleet-wide pressure trends and protocol distribution over time",
    },
}
