"""
Unit tests for protocol normalization.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tpms_dashboard.utils.protocol import (
    normalize_protocol,
    classify_protocol,
    extract_hex_code,
    KNOWN_PROTOCOLS,
)


def test_known_protocols_pass_through():
    """Known protocols should be returned as-is."""
    assert normalize_protocol("Schrader") == "Schrader"
    assert normalize_protocol("Schrader-EV1") == "Schrader-EV1"
    assert normalize_protocol("Schrader-Alt") == "Schrader-Alt"


def test_unknown_hex_collapsed():
    """All Unknown(hex) variants collapse into single bucket."""
    assert normalize_protocol("Unknown(8d0020)") == "Unknown Protocol"
    assert normalize_protocol("Unknown(204653)") == "Unknown Protocol"
    assert normalize_protocol("Unknown(AABBCC)") == "Unknown Protocol"
    assert normalize_protocol("Unknown(000000)") == "Unknown Protocol"
    assert normalize_protocol("Unknown(ff1234)") == "Unknown Protocol"


def test_unknown_zero():
    """Unknown-Zero gets its own label."""
    assert normalize_protocol("Unknown-Zero") == "Unknown (Zero Reading)"


def test_empty_string():
    """Empty protocol returns Unknown Protocol."""
    assert normalize_protocol("") == "Unknown Protocol"


def test_none_like():
    """None-like inputs handled gracefully."""
    assert normalize_protocol("Unknown") == "Unknown Protocol"
    assert normalize_protocol("Unknown(abc)") == "Unknown Protocol"


def test_classify_known():
    """classify_protocol correctly identifies known protocols."""
    assert classify_protocol("Schrader") == "known"
    assert classify_protocol("Schrader-EV1") == "known"
    assert classify_protocol("Unknown(abc123)") == "unknown"


def test_extract_hex_code():
    """extract_hex_code pulls hex from Unknown(xxx) format."""
    assert extract_hex_code("Unknown(8d0020)") == "8d0020"
    assert extract_hex_code("Unknown(AABBCC)") == "AABBCC"
    assert extract_hex_code("Schrader") == ""
    assert extract_hex_code("") == ""


def test_max_legend_entries():
    """After normalization, there should be at most 5 distinct protocol labels."""
    test_inputs = [
        "Schrader", "Schrader-EV1", "Schrader-Alt",
        "Unknown(8d0020)", "Unknown(204653)", "Unknown(abc123)",
        "Unknown(ff0000)", "Unknown(112233)", "Unknown-Zero",
    ]
    results = set(normalize_protocol(p) for p in test_inputs)
    # Should be: Schrader, Schrader-EV1, Schrader-Alt, Unknown Protocol, Unknown (Zero Reading)
    assert len(results) <= 5
    assert "Unknown Protocol" in results
    assert "Schrader" in results


if __name__ == "__main__":
    test_known_protocols_pass_through()
    test_unknown_hex_collapsed()
    test_unknown_zero()
    test_empty_string()
    test_none_like()
    test_classify_known()
    test_extract_hex_code()
    test_max_legend_entries()
    print("\u2705 All protocol tests passed!")
