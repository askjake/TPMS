"""
Data loading utilities for export/archive files.
Handles caching, lazy loading, pagination, and downsampling.
"""
import re
import json
import math
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from tpms_dashboard.config import EXPORT_DIR, ARCHIVE_DIR, LARGE_FILE_THRESHOLD_MB, MAX_CHART_POINTS


def exports_fingerprint(directory: Path = EXPORT_DIR) -> str:
    """Generate a fingerprint of the export directory for cache invalidation."""
    files = sorted(directory.glob("*.json"))
    parts = [f"{f.stat().st_mtime}:{f.stat().st_size}" for f in files]
    return hashlib.md5("|".join(parts).encode()).hexdigest()


@st.cache_data(show_spinner="Loading daily exports...", ttl=120)
def load_daily_exports(fingerprint: str) -> pd.DataFrame:
    """Load only small daily sync exports (< threshold), merge them."""
    frames = []
    for fpath in sorted(EXPORT_DIR.glob("*.json")):
        size_mb = fpath.stat().st_size / 1024 / 1024
        if size_mb > LARGE_FILE_THRESHOLD_MB:
            continue
        m = re.search(r"(\d{8})_(\d{6})", fpath.stem)
        if m:
            export_dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        else:
            export_dt = datetime.fromtimestamp(fpath.stat().st_mtime)
        try:
            with open(fpath) as f:
                records = json.load(f)
            df = pd.DataFrame(records)
            df["export_file"] = fpath.name
            df["export_dt"] = export_dt
            frames.append(df)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    for col in ("first_seen", "last_seen"):
        if col in combined.columns:
            combined[col] = pd.to_datetime(combined[col], errors="coerce")
    for col in ("pressure_psi", "pressure_kpa", "temp_c", "temp_f", "rssi_dbm", "packet_count"):
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    if "battery_low" in combined.columns:
        combined["battery_low"] = combined["battery_low"].astype(bool)
    if "magic_ok" in combined.columns:
        combined["magic_ok"] = combined["magic_ok"].astype(bool)
    combined["export_date"] = combined["export_dt"].dt.date
    combined["session_label"] = combined["export_dt"].dt.strftime("%b %d %H:%M")
    return combined


@st.cache_data(show_spinner="Scanning archive...", ttl=300)
def get_large_file_summary(filepath: str) -> dict:
    """Get summary stats for a large file without holding all records in a DataFrame."""
    fpath = Path(filepath)
    size_mb = fpath.stat().st_size / 1024 / 1024
    with open(fpath) as f:
        data = json.load(f)
    total = len(data)
    protocols = {}
    pressure_vals = []
    temp_vals = []
    for rec in data:
        proto = rec.get("protocol", "Unknown")
        protocols[proto] = protocols.get(proto, 0) + 1
        p = rec.get("pressure_psi")
        if p is not None:
            pressure_vals.append(float(p))
        t = rec.get("temp_c")
        if t is not None:
            temp_vals.append(float(t))
    return {
        "total_sensors": total,
        "size_mb": size_mb,
        "protocols": protocols,
        "pressure_stats": {
            "count": len(pressure_vals),
            "mean": float(np.mean(pressure_vals)) if pressure_vals else None,
            "median": float(np.median(pressure_vals)) if pressure_vals else None,
            "min": min(pressure_vals) if pressure_vals else None,
            "max": max(pressure_vals) if pressure_vals else None,
        },
        "temp_stats": {
            "count": len(temp_vals),
            "mean": float(np.mean(temp_vals)) if temp_vals else None,
            "min": min(temp_vals) if temp_vals else None,
            "max": max(temp_vals) if temp_vals else None,
        },
    }


@st.cache_data(show_spinner="Loading page...", ttl=60)
def load_large_file_page(filepath: str, offset: int = 0, limit: int = 100,
                         sort_by: str = "packet_count", ascending: bool = False,
                         protocol_filter: Optional[str] = None) -> Tuple[pd.DataFrame, int]:
    """Load a page of records from a large file with sort/filter."""
    with open(Path(filepath)) as f:
        data = json.load(f)
    if protocol_filter:
        data = [r for r in data if r.get("protocol") == protocol_filter]
    total = len(data)
    reverse = not ascending
    try:
        data.sort(key=lambda r: (r.get(sort_by) is None, r.get(sort_by, 0)), reverse=reverse)
    except Exception:
        pass
    page_data = data[offset:offset + limit]
    df = pd.DataFrame(page_data)
    if not df.empty:
        for col in ("first_seen", "last_seen"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        for col in ("pressure_psi", "pressure_kpa", "temp_c", "temp_f", "rssi_dbm", "packet_count"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return df, total


def downsample_for_chart(df: pd.DataFrame, max_pts: int = MAX_CHART_POINTS) -> pd.DataFrame:
    """Downsample a DataFrame for chart rendering performance."""
    if len(df) <= max_pts:
        return df
    step = len(df) // max_pts
    return df.iloc[::step].copy()


@st.cache_data(show_spinner="Building sensor timelines...", ttl=120)
def build_sensor_timelines(fingerprint: str) -> pd.DataFrame:
    """
    For every sensor across all daily exports, build one row per
    (sensor_id, export_session) so we can track history over time.
    """
    frames = []
    for fpath in sorted(EXPORT_DIR.glob("*.json")):
        size_mb = fpath.stat().st_size / 1024 / 1024
        if size_mb > LARGE_FILE_THRESHOLD_MB:
            continue
        m = re.search(r"(\d{8})_(\d{6})", fpath.stem)
        if m:
            export_dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        else:
            export_dt = datetime.fromtimestamp(fpath.stat().st_mtime)
        try:
            with open(fpath) as f:
                records = json.load(f)
            df = pd.DataFrame(records)
            df["export_dt"] = export_dt
            df["export_file"] = fpath.name
            frames.append(df)
        except Exception:
            pass

    if not frames:
        return pd.DataFrame()

    tl = pd.concat(frames, ignore_index=True)

    for col in ("pressure_psi", "pressure_kpa", "temp_c", "temp_f", "rssi_dbm", "packet_count"):
        if col in tl.columns:
            tl[col] = pd.to_numeric(tl[col], errors="coerce")
    if "battery_low" in tl.columns:
        tl["battery_low"] = tl["battery_low"].astype(bool)

    tl["hour"] = tl["export_dt"].dt.hour
    tl["weekday_num"] = tl["export_dt"].dt.weekday
    tl["weekday"] = tl["export_dt"].dt.strftime("%a")
    tl["date"] = tl["export_dt"].dt.date
    tl["week"] = tl["export_dt"].dt.isocalendar().week.astype(int)
    tl["session_label"] = tl["export_dt"].dt.strftime("%b %d %H:%M")

    if "last_seen" in tl.columns:
        tl["last_seen"] = pd.to_datetime(tl["last_seen"], errors="coerce")
        tl["capture_label"] = tl["last_seen"].dt.strftime("%b %d %H:%M")
    else:
        tl["capture_label"] = tl["session_label"]

    session_counts = tl.groupby("sensor_id")["export_file"].nunique()
    tl["session_count"] = tl["sensor_id"].map(session_counts)
    tl["is_recurring"] = tl["session_count"] > 1

    return tl


@st.cache_data(show_spinner="Computing observation patterns...", ttl=120)
def build_observation_heatmap(fingerprint: str) -> tuple:
    """
    Returns (heatmap_df, hourly_df, weekly_df) for the Observation Patterns view.
    """
    tl = build_sensor_timelines(fingerprint)
    if tl.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    DAYS_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    hm = (
        tl.groupby(["weekday", "weekday_num", "hour"])["sensor_id"]
        .nunique()
        .reset_index(name="unique_sensors")
    )
    heatmap_df = hm.pivot_table(
        index="weekday", columns="hour", values="unique_sensors",
        aggfunc="sum", fill_value=0
    )
    heatmap_df = heatmap_df.reindex(
        [d for d in DAYS_ORDER if d in heatmap_df.index]
    )
    for h in range(24):
        if h not in heatmap_df.columns:
            heatmap_df[h] = 0
    heatmap_df = heatmap_df[sorted(heatmap_df.columns)]

    all_hours = pd.DataFrame({"hour": range(24)})
    hourly_raw = (
        tl.groupby("hour")["sensor_id"]
        .nunique()
        .reset_index(name="unique_sensors")
        .sort_values("hour")
    )
    hourly_df = all_hours.merge(hourly_raw, on="hour", how="left").fillna(0)

    weekly_df = (
        tl.groupby(["week", "date"])
        .agg(unique_sensors=("sensor_id", "nunique"), total_captures=("sensor_id", "count"))
        .reset_index()
        .sort_values("date")
    )

    return heatmap_df, hourly_df, weekly_df
