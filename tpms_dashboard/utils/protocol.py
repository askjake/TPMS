"""
Protocol normalization utilities.
Raw hex codes are preserved in data; display layer collapses unknowns.
"""
import re
from typing import Set

KNOWN_PROTOCOLS: Set[str] = {"Schrader", "Schrader-EV1", "Schrader-Alt"}

_UNKNOWN_HEX_RE = re.compile(r"^Unknown\(([0-9a-fA-F]+)\)$")


def normalize_protocol(protocol_str: str) -> str:
    """Collapse unknown hex protocols into a single bucket for display.

    Known protocols are returned as-is.
    'Unknown-Zero' becomes 'Unknown (Zero Reading)'.
    All 'Unknown(xxxxxx)' variants become 'Unknown Protocol'.
    """
    if not protocol_str:
        return "Unknown Protocol"
    if protocol_str in KNOWN_PROTOCOLS:
        return protocol_str
    if protocol_str == "Unknown-Zero":
        return "Unknown (Zero Reading)"
    if _UNKNOWN_HEX_RE.match(protocol_str):
        return "Unknown Protocol"
    # Catch-all for any other unexpected format
    if protocol_str.startswith("Unknown"):
        return "Unknown Protocol"
    return protocol_str


def classify_protocol(protocol_str: str) -> str:
    """Return 'known' or 'unknown' classification."""
    if protocol_str in KNOWN_PROTOCOLS:
        return "known"
    return "unknown"


def extract_hex_code(protocol_str: str) -> str:
    """Extract hex from 'Unknown(xxxxxx)' or return empty string."""
    m = _UNKNOWN_HEX_RE.match(protocol_str)
    return m.group(1) if m else ""
