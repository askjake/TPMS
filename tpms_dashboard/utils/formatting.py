"""
Human-readable formatting utilities.
"""
import os
from datetime import datetime
from pathlib import Path


def relative_time(dt: datetime) -> str:
    """Return human-friendly relative time string."""
    now = datetime.now()
    delta = now - dt
    if delta.total_seconds() < 60:
        return "Just now"
    elif delta.total_seconds() < 3600:
        mins = int(delta.total_seconds() // 60)
        return f"{mins} minute{'s' if mins != 1 else ''} ago"
    elif delta.days == 0:
        hours = int(delta.total_seconds() // 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif delta.days == 1:
        return "Yesterday"
    else:
        return f"{delta.days} days ago"


def get_last_modified(path) -> str:
    """Return freshness string for a directory or file."""
    try:
        p = Path(path)
        if p.is_dir():
            # Find the most recent file modification in the directory
            files = list(p.glob("*.json"))
            if not files:
                return "No data files found"
            latest = max(files, key=lambda f: f.stat().st_mtime)
            ts = latest.stat().st_mtime
        else:
            ts = p.stat().st_mtime
        dt = datetime.fromtimestamp(ts)
        rel = relative_time(dt)
        absolute = dt.strftime("%Y-%m-%d %H:%M")
        return f"{rel} ({absolute})"
    except Exception:
        return "Last update time unavailable"


def format_ratio(numerator: int, denominator: int) -> str:
    """Format a ratio like '1.06 / sensor'."""
    if denominator == 0:
        return "N/A"
    ratio = numerator / denominator
    return f"{ratio:.2f}"


def format_date_range(start_date, end_date) -> str:
    """Format a date range like 'Feb 20 \u2013 Jul 2'."""
    try:
        start_str = start_date.strftime("%b %d")
        end_str = end_date.strftime("%b %d")
        return f"{start_str} \u2013 {end_str}"
    except Exception:
        return "Unknown range"
