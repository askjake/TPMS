"""
Daily Sync Overview page.
"""
import pandas as pd
import streamlit as st

from tpms_dashboard.components.kpi_row import render_kpi_row
from tpms_dashboard.components.protocol_chart import render_pressure_chart
from tpms_dashboard.components.freshness_banner import render_freshness_banner
from tpms_dashboard.utils.data_loader import exports_fingerprint, load_daily_exports, downsample_for_chart
from tpms_dashboard.config import EXPORT_DIR


def render():
    """Render the Daily Sync Overview page."""
    st.header("\U0001f4cb Daily Sync Overview")
    st.caption("Shows data from daily sync exports (small files). For large tracker archives, use Archive Explorer.")

    render_freshness_banner()

    fp = exports_fingerprint(EXPORT_DIR)
    df = load_daily_exports(fp)
    if df.empty:
        st.warning("No daily export files found.")
        st.stop()

    # Enhanced KPI row
    render_kpi_row(df)

    st.divider()

    # Tabs for Sessions and Latest Export
    tab_sessions, tab_sensors = st.tabs([
        "\U0001f4cb Sessions \u2014 Daily sync file list",
        "\U0001f4e1 Latest Export \u2014 Sensors seen in most recent sync",
    ])

    with tab_sessions:
        if "export_dt" in df.columns and "sensor_id" in df.columns:
            session_stats = df.groupby("export_file").agg(
                sensors=("sensor_id", "nunique"),
                records=("sensor_id", "count"),
                export_date=("export_dt", "first"),
                avg_psi=("pressure_psi", "mean"),
            ).sort_values("export_date", ascending=False).reset_index()
            st.dataframe(session_stats, use_container_width=True, hide_index=True)

    with tab_sensors:
        latest_file = df.sort_values("export_dt", ascending=False)["export_file"].iloc[0]
        latest_df = df[df["export_file"] == latest_file].copy()
        display_cols = [c for c in ["sensor_id", "protocol", "pressure_psi", "temp_f",
                                    "rssi_dbm", "packet_count", "first_seen", "last_seen",
                                    "battery_low"] if c in latest_df.columns]
        st.dataframe(latest_df[display_cols].sort_values("packet_count", ascending=False),
                     use_container_width=True, hide_index=True, height=400)

    # Pressure distribution chart (grouped legend)
    if "pressure_psi" in df.columns:
        st.divider()
        chart_df = downsample_for_chart(df)
        render_pressure_chart(chart_df)
