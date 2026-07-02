"""
Unit tests for KPI calculation correctness.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tpms_dashboard.utils.formatting import relative_time, format_ratio, format_date_range
from datetime import datetime, timedelta


def test_relative_time_just_now():
    """Less than 60 seconds should be 'Just now'."""
    dt = datetime.now() - timedelta(seconds=30)
    assert relative_time(dt) == "Just now"


def test_relative_time_minutes():
    """A few minutes ago."""
    dt = datetime.now() - timedelta(minutes=5)
    result = relative_time(dt)
    assert "5 minutes ago" == result


def test_relative_time_hours():
    """A few hours ago."""
    dt = datetime.now() - timedelta(hours=3)
    result = relative_time(dt)
    assert "3 hours ago" == result


def test_relative_time_yesterday():
    """Yesterday."""
    dt = datetime.now() - timedelta(days=1)
    result = relative_time(dt)
    assert result == "Yesterday"


def test_relative_time_days():
    """Multiple days ago."""
    dt = datetime.now() - timedelta(days=7)
    result = relative_time(dt)
    assert "7 days ago" == result


def test_format_ratio_normal():
    """Normal ratio calculation."""
    assert format_ratio(1299, 1221) == "1.06"


def test_format_ratio_zero_denominator():
    """Zero denominator returns N/A."""
    assert format_ratio(100, 0) == "N/A"


def test_format_date_range():
    """Date range formatting."""
    start = datetime(2026, 2, 20)
    end = datetime(2026, 7, 2)
    result = format_date_range(start, end)
    assert "Feb 20" in result
    assert "Jul 02" in result


if __name__ == "__main__":
    test_relative_time_just_now()
    test_relative_time_minutes()
    test_relative_time_hours()
    test_relative_time_yesterday()
    test_relative_time_days()
    test_format_ratio_normal()
    test_format_ratio_zero_denominator()
    test_format_date_range()
    print("\u2705 All KPI tests passed!")
