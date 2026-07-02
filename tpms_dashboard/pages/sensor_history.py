"""
Sensor History page — per-sensor timeline, pressure/RSSI charts, heatmap.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from tpms_dashboard.config import EXPORT_DIR
from tpms_dashboard.components.freshness_banner import render_freshness_banner
from tpms_dashboard.utils.data_loader import exports_fingerprint, build_sensor_timelines


def render():
    """Render the Sensor History page."""
    st.header("\U0001f4e1 Sensor Capture History")
    st.caption(
        "Per-sensor timeline across all daily export sessions. "
        "Tracks pressure, RSSI, and packet-count changes over time."
    )
    render_freshness_banner()

    fp = exports_fingerprint(EXPORT_DIR)
    tl = build_sensor_timelines(fp)

    if tl.empty or "sensor_id" not in tl.columns:
        st.warning("No timeline data available yet.")
        st.stop()

    with st.sidebar:
        st.header("History Filters")
        show_only_recurring = st.checkbox("Recurring sensors only", value=True,
                                          help="Show only sensors seen in multiple sessions")
        max_sessions = int(tl["session_count"].max()) if tl["session_count"].max() > 1 else 2
        min_sessions = st.slider("Min sessions seen", 1, max_sessions, 2,
                                 help="Minimum number of sessions a sensor must appear in")
        proto_list = sorted(tl["protocol"].dropna().unique().tolist()) if "protocol" in tl.columns else []
        proto_filter_h = st.multiselect("Protocol", proto_list, default=[], key="hist_proto",
                                        help="Filter by protocol type")

    cands = tl.copy()
    if show_only_recurring:
        cands = cands[cands["is_recurring"]]
    cands = cands[cands["session_count"] >= min_sessions]
    if proto_filter_h:
        cands = cands[cands["protocol"].isin(proto_filter_h)]

    if cands.empty:
        st.info("No sensors match the current filters. Try loosening the criteria in the sidebar.")
        st.stop()

    st.markdown("### Recurring Sensors \u2014 Ranked by Observation Count")
    summary = (
        cands.sort_values("export_dt")
        .groupby("sensor_id")
        .agg(
            protocol=("protocol", "first"),
            sessions=("export_file", "nunique"),
            first_seen_dt=("export_dt", "min"),
            last_seen_dt=("export_dt", "max"),
            avg_psi=("pressure_psi", "mean"),
            last_psi=("pressure_psi", "last"),
            avg_rssi=("rssi_dbm", "mean"),
            total_pkts=("packet_count", "sum"),
            battery_ever=("battery_low", "max"),
        )
        .reset_index()
    )
    summary["span_days"] = (summary["last_seen_dt"] - summary["first_seen_dt"]).dt.days
    summary["avg_psi"] = summary["avg_psi"].round(1)
    summary["avg_rssi"] = summary["avg_rssi"].round(1)
    summary = summary.sort_values("sessions", ascending=False)

    disp = summary[["sensor_id", "protocol", "sessions", "span_days",
                     "avg_psi", "last_psi", "avg_rssi", "total_pkts",
                     "battery_ever", "first_seen_dt", "last_seen_dt"]].copy()
    disp.columns = ["Sensor ID", "Protocol", "Sessions", "Span (days)",
                    "Avg PSI", "Last PSI", "Avg RSSI", "Total Pkts",
                    "Batt Low", "First Seen", "Last Seen"]
    st.dataframe(disp, use_container_width=True, hide_index=True, height=300)

    st.markdown("### Sensor Detail")
    sensor_choices = summary["sensor_id"].tolist()
    selected_sensor = st.selectbox(
        "Select sensor",
        sensor_choices,
        format_func=lambda s: (
            f"{s}  ({summary.loc[summary['sensor_id']==s,'sessions'].values[0]} sessions)"
        ),
        help="Choose a sensor to view detailed history",
    )

    sdf = cands[cands["sensor_id"] == selected_sensor].sort_values("export_dt").copy()

    k = st.columns(5)
    k[0].metric("Sessions", int(sdf["export_file"].nunique()))
    k[1].metric("Span", f"{(sdf['export_dt'].max() - sdf['export_dt'].min()).days}d")
    if sdf["pressure_psi"].notna().any():
        k[2].metric("Avg PSI", f"{sdf['pressure_psi'].mean():.1f}")
        psi_delta = (sdf["pressure_psi"].iloc[-1] - sdf["pressure_psi"].iloc[0]) if len(sdf) > 1 else 0
        k[3].metric("PSI delta (first to last)", f"{psi_delta:+.1f}")
    if "rssi_dbm" in sdf.columns and sdf["rssi_dbm"].notna().any():
        k[4].metric("Avg RSSI", f"{sdf['rssi_dbm'].mean():.1f} dBm")

    st.markdown("#### Capture-by-capture history")
    raw_cols = [c for c in ["capture_label", "pressure_psi", "pressure_kpa",
                             "temp_f", "temp_c", "rssi_dbm", "packet_count",
                             "battery_low", "export_file"]
                if c in sdf.columns]
    st.dataframe(
        sdf[raw_cols].rename(columns={"capture_label": "Captured"}),
        use_container_width=True, hide_index=True
    )

    if len(sdf) > 1:
        st.markdown("#### Pressure over time")
        fig_psi = go.Figure()
        fig_psi.add_trace(go.Scatter(
            x=sdf["export_dt"], y=sdf["pressure_psi"],
            mode="lines+markers+text",
            text=[f"{v:.0f}" if pd.notna(v) else "" for v in sdf["pressure_psi"]],
            textposition="top center",
            name="PSI",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=10),
        ))
        fig_psi.add_hrect(y0=0, y1=25, fillcolor="red", opacity=0.07, line_width=0,
                          annotation_text="Critical (<25 PSI)", annotation_position="top left")
        fig_psi.add_hrect(y0=25, y1=32, fillcolor="orange", opacity=0.07, line_width=0,
                          annotation_text="Low (25-32)")
        fig_psi.add_hrect(y0=32, y1=50, fillcolor="green", opacity=0.05, line_width=0,
                          annotation_text="Normal (32-45)")
        fig_psi.update_layout(title=f"Pressure history \u2014 {selected_sensor}",
                               xaxis_title="Capture Date", yaxis_title="PSI",
                               hovermode="x unified")
        st.plotly_chart(fig_psi, use_container_width=True)

        if "rssi_dbm" in sdf.columns and sdf["rssi_dbm"].notna().any():
            st.markdown("#### Signal strength (RSSI) over time")
            fig_rssi = go.Figure()
            fig_rssi.add_trace(go.Scatter(
                x=sdf["export_dt"], y=sdf["rssi_dbm"],
                mode="lines+markers", name="RSSI (dBm)",
                line=dict(color="#ff7f0e", width=2), marker=dict(size=9),
            ))
            fig_rssi.add_hrect(y0=-120, y1=-85, fillcolor="red", opacity=0.06, line_width=0, annotation_text="Weak")
            fig_rssi.add_hrect(y0=-85, y1=-60, fillcolor="yellow", opacity=0.06, line_width=0, annotation_text="OK")
            fig_rssi.add_hrect(y0=-60, y1=0, fillcolor="green", opacity=0.06, line_width=0, annotation_text="Strong")
            fig_rssi.update_layout(title=f"RSSI history \u2014 {selected_sensor}",
                                    xaxis_title="Capture Date", yaxis_title="dBm",
                                    hovermode="x unified")
            st.plotly_chart(fig_rssi, use_container_width=True)

        if "packet_count" in sdf.columns and sdf["packet_count"].notna().any():
            st.markdown("#### Packet count per capture")
            fig_pkts = px.bar(sdf, x="export_dt", y="packet_count",
                              title=f"Packets per capture \u2014 {selected_sensor}",
                              labels={"export_dt": "Capture Date", "packet_count": "Packets"})
            st.plotly_chart(fig_pkts, use_container_width=True)

        st.markdown("#### Inter-capture gaps")
        if "last_seen" in sdf.columns and sdf["last_seen"].notna().any():
            gap_series = sdf["last_seen"].sort_values()
        else:
            gap_series = sdf["export_dt"].sort_values()
        gaps_h = gap_series.diff().dropna().dt.total_seconds().div(3600).round(1).values
        gap_df = pd.DataFrame({
            "From": sdf["capture_label"].iloc[:-1].values,
            "To": sdf["capture_label"].iloc[1:].values,
            "Gap (h)": gaps_h,
            "Gap (days)": (gaps_h / 24).round(1),
        })
        st.dataframe(gap_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### All Recurring Sensors \u2014 Pressure Heatmap")
    st.caption("Each row = one recurring sensor, each column = one capture session. Colour = PSI.")
    recurring_only = cands[cands["is_recurring"]].copy()
    if not recurring_only.empty:
        pivot = recurring_only.pivot_table(
            index="sensor_id", columns="session_label",
            values="pressure_psi", aggfunc="mean"
        )
        col_order = (
            recurring_only[["session_label", "export_dt"]]
            .drop_duplicates("session_label")
            .sort_values("export_dt")["session_label"]
            .tolist()
        )
        pivot = pivot[[c for c in col_order if c in pivot.columns]]
        pivot = pivot.loc[pivot.notna().sum(axis=1).sort_values(ascending=False).index]
        fig_hm = px.imshow(
            pivot,
            color_continuous_scale="RdYlGn",
            range_color=[0, 60],
            aspect="auto",
            title="Pressure (PSI) \u2014 Sensor \u00d7 Session",
            labels={"color": "PSI", "x": "Session", "y": "Sensor ID"},
        )
        fig_hm.update_layout(height=max(300, len(pivot) * 28 + 100))
        st.plotly_chart(fig_hm, use_container_width=True)
