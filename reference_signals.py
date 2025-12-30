"""
Reference 'Happy Path' Signals
These are known good TPMS captures that anchor the ML learning
"""

REFERENCE_SIGNALS = {
    "toyota_lexus_5920441A": {
        "tpms_id": "5920441A",
        "protocol": "Toyota/Lexus",
        "frequency": 314.9,
        "signal_strength": -86.6,
        "modulation": "FSK",
        "bit_pattern": [0x59, 0x20, 0x44, 0x1A],
        "timestamp": "2025-12-29 22:12:31",
        "notes": "First successful capture - reference signal"
    },
    "schrader_A42D124A": {
        "tpms_id": "A42D124A",
        "protocol": "Schrader",
        "frequency": 314.9,
        "signal_strength": -86.2,
        "modulation": "PSK",
        "bit_pattern": [0xA4, 0x2D, 0x12, 0x4A],
        "timestamp": "2025-12-30 15:20:26",
        "notes": "Second capture - Schrader protocol reference"
    }
}

def get_reference_characteristics():
    """Get the common characteristics from reference signals"""
    return {
        "frequency_range": (314.8, 315.0),
        "min_signal_strength": -90.0,  # dBm
        "typical_strength": -86.0,
        "modulation_types": ["FSK", "PSK", "FSK/PSK"],
        "symbol_rates": [2184, 2520, 2730],  # Common rates from captures
        "protocols": ["Toyota/Lexus", "Schrader"]
    }
