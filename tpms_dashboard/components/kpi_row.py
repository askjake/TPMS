"""
Enhanced KPI tiles with context, ratios, and freshness.
"""
import streamlit as st
import pandas as pd
from datetime import datetime

from tpms_dashboard.config import EXPORT_DIR
from tpms_dashboard.utils.formatting import get_last_modified, relative_time, format_date_range
from tpms_dashboard.utils.protocol import normalize_protocol, KNOWN_PROTOCOLS


def render_kpi_row(df: pd.DataFrame):
    """Render the enhanced KPI row with context-rich metrics."""
    if df.empty:
        st.warning("No data available for KPI display.")
        return

    sessions = df["export_file"].nunique()
    unique_sensors = df["sensor_id"].nunique() if "sensor_id" in df.columns else 0
    records = len(df)
    records_per_sensor = records / unique_sensors if unique_sensors > 0 else 0

    # Date range
    earliest = None
    latest = None
    if "first_seen" in df.columns and df["first_seen"].notna().any():
        earliest = df["first_seen"].min()
    if "last_seen" in df.columns and df["last_seen"].notna().any():
        latest = df["last_seen"].max()
    span_days = (latest - earliest).days if earliest and latest else 0

    # Protocol stats
    known_count = 0
    unknown_count = 0
    if "protocol" in df.columns:
        proto_normalized = df["protocol"].apply(normalize_protocol)
        known_count = proto_normalized.isin(KNOWN_PROTOCOLS).sum()
        unknown_count = len(proto_normalized) - known_count
    known_pct = (known_count / records * 100) if records > 0 else 0
    unknown_pct = 100 - known_pct

    # Row 1: Core metrics
    row1 = st.columns(3)
    row1[0].metric(
        "\U0001f4cb Sessions",
        sessions,
        help="Total number of daily sync export files processed",
    )
    row1[1].metric(
        "\U0001f4e1 Unique Sensors",
        f"{unique_sensors:,}",
        help="Distinct sensor IDs seen across all sessions",
    )
    row1[2].metric(
        "\U0001f4c4 Records",
        f"{records:,}",
        delta=f"{records_per_sensor:.2f} per sensor",
        delta_color="off",
        help="Total observation records; delta shows records/sensor ratio",
    )

    # Row 2: Context metrics
    row2 = st.columns(3)

    date_range_str = format_date_range(earliest, latest) if earliest and latest else "N/A"
    row2[0].metric(
        "\U0001f4c5 Date Range",
        f"{span_days} days",
        delta=date_range_str,
        delta_color="off",
        help="Time span from earliest to latest observation",
    )

    freshness = get_last_modified(EXPORT_DIR)
    row2[1].metric(
        "\U0001f504 Last Sync",
        freshness.split(" (")[0] if " (" in freshness else freshness,
        delta=freshness.split("(")[1].rstrip(")") if "(" in freshness else "",
        delta_color="off",
        help="When the most recent export file was written",
    )

    row2[2].metric(
        "\u2705 Known Protocols",
        f"{len(KNOWN_PROTOCOLS)} / {unique_sensors:,}",
        delta=f"{unknown_pct:.1f}% unresolved",
        delta_color="inverse",
        help=f"Only {known_pct:.1f}% of observations use named protocols (Schrader variants). "
             f"The rest are Unknown hex identifiers.",
    )
