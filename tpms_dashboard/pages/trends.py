"""
Trends & Charts page — cross-session analysis with downsampled charts.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from tpms_dashboard.config import EXPORT_DIR
from tpms_dashboard.components.freshness_banner import render_freshness_banner
from tpms_dashboard.utils.data_loader import exports_fingerprint, load_daily_exports, downsample_for_chart
from tpms_dashboard.utils.protocol import normalize_protocol


def render():
    """Render the Trends & Charts page."""
    st.header("\U0001f4c8 Trends & Charts")
    st.caption("Cross-session analysis (daily exports). Charts are downsampled for performance.")
    render_freshness_banner()

    fp = exports_fingerprint(EXPORT_DIR)
    df = load_daily_exports(fp)
    if df.empty:
        st.warning("No daily exports found.")
        st.stop()

    if "export_dt" in df.columns and "sensor_id" in df.columns:
        st.markdown("### Unique Sensors Per Session")
        session_counts = df.groupby("session_label").agg(
            sensors=("sensor_id", "nunique"),
            avg_psi=("pressure_psi", "mean"),
            export_time=("export_dt", "first"),
        ).sort_values("export_time").reset_index()
        fig = px.bar(session_counts, x="session_label", y="sensors",
                    title="Unique Sensors Detected Per Session",
                    labels={"session_label": "Session", "sensors": "Unique Sensors"})
        st.plotly_chart(fig, use_container_width=True)

    if "sensor_id" in df.columns and "pressure_psi" in df.columns:
        st.markdown("### Pressure Trends (Recurring Sensors)")
        sensor_freq = df.groupby("sensor_id")["export_file"].nunique()
        recurring = sensor_freq[sensor_freq > 1].index.tolist()
        if recurring:
            top_recurring = recurring[:20]
            selected = st.multiselect(
                "Select sensors to track",
                top_recurring,
                default=top_recurring[:min(3, len(top_recurring))],
                help="Choose recurring sensors to plot pressure trends",
            )
            if selected:
                trend_df = df[df["sensor_id"].isin(selected)].copy()
                trend_df = downsample_for_chart(trend_df)
                fig = px.line(trend_df.sort_values("export_dt"),
                            x="export_dt", y="pressure_psi", color="sensor_id",
                            title="Pressure Over Time (Recurring Sensors)",
                            labels={"export_dt": "Date", "pressure_psi": "Pressure (PSI)", "sensor_id": "Sensor"})
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No sensors found in multiple sessions yet.")

    if "protocol" in df.columns and "export_dt" in df.columns:
        st.markdown("### Protocol Mix Over Time")
        # Use normalized protocols for this chart too
        proto_df = df.copy()
        proto_df["protocol_display"] = proto_df["protocol"].apply(normalize_protocol)
        proto_time = proto_df.groupby(["session_label", "protocol_display"]).size().reset_index(name="count")
        fig = px.bar(proto_time, x="session_label", y="count", color="protocol_display",
                    title="Protocol Distribution By Session (Grouped)",
                    labels={"session_label": "Session", "count": "Sensors", "protocol_display": "Protocol"})
        st.plotly_chart(fig, use_container_width=True)
