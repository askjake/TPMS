"""
TPMS Sensor Statistics Dashboard
=================================
Streamlit app that loads all exports from ~/TPMS/exports/,
merges them, and presents interactive Plotly charts showing
histories, patterns, and per-sensor statistics.

Run:  streamlit run tpms_dashboard.py --server.address 0.0.0.0 --server.port 8501
Auto-sync: Watches exports/ for new files and reloads automatically via st.rerun().
"""

import os
import re
import glob
import json
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ─── CONFIG ──────────────────────────────────────────────────────────────────
EXPORTS_DIR = Path(__file__).parent / "exports"
REFRESH_INTERVAL_SECS = 30          # auto-refresh polling interval
PAGE_TITLE = "TPMS Sensor Dashboard"

# ─── PAGE SETUP ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="🛞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def exports_fingerprint() -> str:
    """Hash mtime+size of all export files to detect changes."""
    files = sorted(EXPORTS_DIR.glob("*.json"))
    parts = [f"{f.stat().st_mtime}:{f.stat().st_size}" for f in files]
    return hashlib.md5("|".join(parts).encode()).hexdigest()


@st.cache_data(show_spinner="Loading export files…")
def load_all_exports(fingerprint: str) -> pd.DataFrame:
    """Load & merge all JSON exports; fingerprint busts cache on file changes."""
    frames = []
    for fpath in sorted(EXPORTS_DIR.glob("*.json")):
        # Extract datetime from filename  tpms_sync_YYYYMMDD_HHMMSS_…
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
        except Exception as e:
            st.warning(f"Skipped {fpath.name}: {e}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Type coercions
    for col in ("first_seen", "last_seen"):
        combined[col] = pd.to_datetime(combined[col], errors="coerce")

    for col in ("pressure_psi", "pressure_kpa", "temp_c", "temp_f",
                "rssi_dbm", "packet_count"):
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    combined["battery_low"] = combined["battery_low"].astype(bool)
    combined["magic_ok"]    = combined["magic_ok"].astype(bool)

    # Convenience columns
    combined["export_date"] = combined["export_dt"].dt.date
    combined["session_label"] = combined["export_dt"].dt.strftime("%b %d %H:%M")

    return combined


def pressure_alert_color(psi):
    """Traffic-light color for PSI."""
    if pd.isna(psi):
        return "gray"
    if psi < 25:
        return "red"
    if psi < 32:
        return "orange"
    return "green"


# ─── DATA LOAD WITH AUTO-REFRESH ─────────────────────────────────────────────
if "last_fingerprint" not in st.session_state:
    st.session_state["last_fingerprint"] = ""

fp = exports_fingerprint()
if fp != st.session_state["last_fingerprint"]:
    st.cache_data.clear()
    st.session_state["last_fingerprint"] = fp

df_all = load_all_exports(fp)

if df_all.empty:
    st.error(f"No JSON export files found in {EXPORTS_DIR}")
    st.stop()

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/emoji/96/tire.png", width=64)
st.sidebar.title("🛞 TPMS Dashboard")
st.sidebar.markdown("---")

# File selector
all_files = sorted(df_all["export_file"].unique())
selected_files = st.sidebar.multiselect(
    "📂 Export Sessions",
    options=all_files,
    default=all_files,
    help="Choose which export sessions to include",
)

# Sensor filter
all_sensors = sorted(df_all["sensor_id"].unique())
selected_sensors = st.sidebar.multiselect(
    "📡 Sensors",
    options=all_sensors,
    default=all_sensors,
    help="Filter to specific sensor IDs",
)

# Protocol filter
all_protocols = sorted(df_all["protocol"].unique())
selected_protocols = st.sidebar.multiselect(
    "🔌 Protocols",
    options=all_protocols,
    default=all_protocols,
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Display Options")
pressure_unit = st.sidebar.radio("Pressure Unit", ["PSI", "kPa"], horizontal=True)
temp_unit     = st.sidebar.radio("Temperature Unit", ["°C", "°F"],  horizontal=True)
show_suspects = st.sidebar.checkbox("Hide suspect temps", value=True)
show_magic_ok = st.sidebar.checkbox("Only magic_ok packets", value=False)

st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox(f"🔄 Auto-refresh ({REFRESH_INTERVAL_SECS}s)", value=True)

# ─── FILTER DATA ─────────────────────────────────────────────────────────────
df = df_all[
    df_all["export_file"].isin(selected_files) &
    df_all["sensor_id"].isin(selected_sensors) &
    df_all["protocol"].isin(selected_protocols)
].copy()

if show_suspects and "suspect_temp" in df.columns:
    df = df[~df["suspect_temp"].astype(bool)]

if show_magic_ok:
    df = df[df["magic_ok"]]

pressure_col = "pressure_psi" if pressure_unit == "PSI" else "pressure_kpa"
temp_col     = "temp_c"       if temp_unit == "°C"  else "temp_f"
pressure_label = f"Pressure ({pressure_unit})"
temp_label     = f"Temperature ({temp_unit})"

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.title("🛞 TPMS Sensor Statistics Dashboard")
st.caption(
    f"Loaded **{len(all_files)}** export sessions · "
    f"**{df_all['sensor_id'].nunique()}** unique sensors · "
    f"Last updated: **{datetime.now().strftime('%H:%M:%S')}**"
)

# ─── KPI ROW ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Sessions",           len(selected_files))
k2.metric("Sensors (filtered)", df["sensor_id"].nunique())
k3.metric("Avg Pressure",       f"{df[pressure_col].mean():.1f} {pressure_unit}" if not df.empty else "–")
k4.metric("Avg Temp",           f"{df[temp_col].mean():.1f}{temp_unit}"          if not df.empty else "–")
k5.metric("Total Packets",      int(df["packet_count"].sum())                     if not df.empty else 0)

st.markdown("---")

if df.empty:
    st.warning("No data matches current filters.")
    st.stop()

# ─── TAB LAYOUT ──────────────────────────────────────────────────────────────
tab_overview, tab_history, tab_sensors, tab_patterns, tab_raw = st.tabs([
    "📊 Overview", "📈 History", "📡 Per-Sensor", "🔍 Patterns", "🗄️ Raw Data"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.subheader("Snapshot — Latest Reading per Sensor")

    # Latest reading per sensor across all selected sessions
    df_latest = (
        df.sort_values("export_dt")
          .groupby("sensor_id", as_index=False)
          .last()
    )

    # Pressure bar chart coloured by level
    df_latest["alert"] = df_latest["pressure_psi"].apply(pressure_alert_color)
    fig_pressure = px.bar(
        df_latest.sort_values(pressure_col, ascending=False),
        x="sensor_id", y=pressure_col,
        color="alert",
        color_discrete_map={"red": "#ef4444", "orange": "#f97316", "green": "#22c55e", "gray": "#94a3b8"},
        labels={"sensor_id": "Sensor ID", pressure_col: pressure_label},
        title="Latest Pressure per Sensor",
        hover_data=["protocol", temp_col, "packet_count", "rssi_dbm"],
    )
    fig_pressure.update_layout(showlegend=False, xaxis_tickangle=-45)
    st.plotly_chart(fig_pressure, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        # Protocol distribution
        proto_counts = df["protocol"].value_counts().reset_index()
        proto_counts.columns = ["protocol", "count"]
        fig_proto = px.pie(
            proto_counts, names="protocol", values="count",
            title="Protocol Distribution",
            hole=0.4,
        )
        st.plotly_chart(fig_proto, use_container_width=True)

    with col_b:
        # RSSI distribution
        fig_rssi = px.histogram(
            df[df["rssi_dbm"] != 0],
            x="rssi_dbm", nbins=30,
            color="protocol",
            title="RSSI Signal Strength Distribution",
            labels={"rssi_dbm": "RSSI (dBm)"},
        )
        st.plotly_chart(fig_rssi, use_container_width=True)

    # Scatter: Pressure vs Temperature coloured by protocol
    fig_scatter = px.scatter(
        df_latest,
        x=pressure_col, y=temp_col,
        color="protocol",
        text="sensor_id",
        title=f"Pressure vs Temperature — Latest Reading",
        labels={pressure_col: pressure_label, temp_col: temp_label},
        size="packet_count",
        size_max=20,
    )
    fig_scatter.update_traces(textposition="top center")
    st.plotly_chart(fig_scatter, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – HISTORY
# ══════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.subheader("Sensor Readings Across Sessions Over Time")

    # Only sensors seen in ≥ 2 sessions are interesting for history
    session_counts = df.groupby("sensor_id")["export_file"].nunique()
    recurring = session_counts[session_counts >= 2].index.tolist()

    if not recurring:
        st.info("No sensors appear in more than one session in the current selection. "
                "Showing all sensors with single-session data instead.")
        recurring = df["sensor_id"].unique().tolist()

    focus_sensors = st.multiselect(
        "Select sensors to plot history",
        options=sorted(df["sensor_id"].unique()),
        default=sorted(recurring)[:10],
        key="history_sensors",
    )

    if focus_sensors:
        df_hist = df[df["sensor_id"].isin(focus_sensors)].sort_values("export_dt")

        fig_ph = px.line(
            df_hist, x="export_dt", y=pressure_col,
            color="sensor_id", markers=True,
            title=f"Pressure History by Sensor",
            labels={"export_dt": "Export Time", pressure_col: pressure_label},
        )
        st.plotly_chart(fig_ph, use_container_width=True)

        fig_th = px.line(
            df_hist, x="export_dt", y=temp_col,
            color="sensor_id", markers=True,
            title=f"Temperature History by Sensor",
            labels={"export_dt": "Export Time", temp_col: temp_label},
        )
        st.plotly_chart(fig_th, use_container_width=True)

        fig_rssi_h = px.line(
            df_hist[df_hist["rssi_dbm"] != 0],
            x="export_dt", y="rssi_dbm",
            color="sensor_id", markers=True,
            title="RSSI History by Sensor",
            labels={"export_dt": "Export Time", "rssi_dbm": "RSSI (dBm)"},
        )
        st.plotly_chart(fig_rssi_h, use_container_width=True)

    # Session-level aggregated trends
    st.markdown("#### Session Averages Over Time")
    df_session_agg = (
        df.groupby("session_label", as_index=False)
          .agg(
              avg_pressure=(pressure_col, "mean"),
              avg_temp=(temp_col, "mean"),
              sensor_count=("sensor_id", "nunique"),
              total_packets=("packet_count", "sum"),
              export_dt=("export_dt", "first"),
          )
          .sort_values("export_dt")
    )

    fig_agg = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Avg Pressure/Session", "Avg Temp/Session",
                        "Sensors Seen/Session", "Total Packets/Session"),
    )
    fig_agg.add_trace(go.Scatter(x=df_session_agg["session_label"], y=df_session_agg["avg_pressure"],
                                  mode="lines+markers", name="Avg Pressure"), row=1, col=1)
    fig_agg.add_trace(go.Scatter(x=df_session_agg["session_label"], y=df_session_agg["avg_temp"],
                                  mode="lines+markers", name="Avg Temp"), row=1, col=2)
    fig_agg.add_trace(go.Bar(x=df_session_agg["session_label"], y=df_session_agg["sensor_count"],
                              name="Sensors"), row=2, col=1)
    fig_agg.add_trace(go.Bar(x=df_session_agg["session_label"], y=df_session_agg["total_packets"],
                              name="Packets"), row=2, col=2)
    fig_agg.update_layout(height=600, showlegend=False, title_text="Per-Session Aggregates")
    fig_agg.update_xaxes(tickangle=-45)
    st.plotly_chart(fig_agg, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – PER-SENSOR DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════
with tab_sensors:
    st.subheader("Per-Sensor Deep Dive")

    chosen = st.selectbox("Choose a sensor:", options=sorted(df["sensor_id"].unique()))
    df_s = df[df["sensor_id"] == chosen].sort_values("export_dt")

    if df_s.empty:
        st.warning("No data for this sensor in current filters.")
    else:
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Sessions seen",   df_s["export_file"].nunique())
        sc2.metric("Total packets",   int(df_s["packet_count"].sum()))
        sc3.metric("Avg Pressure",    f"{df_s[pressure_col].mean():.1f} {pressure_unit}")
        sc4.metric("Protocol",        df_s["protocol"].iloc[-1])

        fig_sensor = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=(pressure_label, temp_label, "RSSI (dBm)"),
            vertical_spacing=0.08,
        )
        fig_sensor.add_trace(go.Scatter(
            x=df_s["export_dt"], y=df_s[pressure_col],
            mode="lines+markers", name=pressure_label,
            line=dict(color="#3b82f6"),
        ), row=1, col=1)
        fig_sensor.add_trace(go.Scatter(
            x=df_s["export_dt"], y=df_s[temp_col],
            mode="lines+markers", name=temp_label,
            line=dict(color="#f97316"),
        ), row=2, col=1)
        rssi_data = df_s[df_s["rssi_dbm"] != 0]
        fig_sensor.add_trace(go.Scatter(
            x=rssi_data["export_dt"], y=rssi_data["rssi_dbm"],
            mode="lines+markers", name="RSSI",
            line=dict(color="#8b5cf6"),
        ), row=3, col=1)
        fig_sensor.update_layout(height=600, title_text=f"Sensor {chosen} — Full History",
                                  showlegend=False)
        st.plotly_chart(fig_sensor, use_container_width=True)

        st.markdown("#### All readings for this sensor")
        display_cols = ["export_dt", "export_file", pressure_col, temp_col,
                        "rssi_dbm", "packet_count", "battery_low", "magic_ok"]
        existing = [c for c in display_cols if c in df_s.columns]
        st.dataframe(df_s[existing].sort_values("export_dt", ascending=False), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – PATTERNS & ANOMALIES
# ══════════════════════════════════════════════════════════════════════════════
with tab_patterns:
    st.subheader("Patterns & Anomalies")

    col_h1, col_h2 = st.columns(2)

    with col_h1:
        # Heatmap: sensor × session, pressure
        pivot = df.pivot_table(
            index="sensor_id", columns="session_label",
            values=pressure_col, aggfunc="mean"
        )
        if not pivot.empty:
            fig_heat = px.imshow(
                pivot,
                aspect="auto",
                color_continuous_scale="RdYlGn",
                title=f"Pressure Heatmap (sensor × session)",
                labels={"color": pressure_label},
            )
            st.plotly_chart(fig_heat, use_container_width=True)

    with col_h2:
        # Heatmap: sensor × session, temperature
        pivot_t = df.pivot_table(
            index="sensor_id", columns="session_label",
            values=temp_col, aggfunc="mean"
        )
        if not pivot_t.empty:
            fig_heat_t = px.imshow(
                pivot_t,
                aspect="auto",
                color_continuous_scale="thermal",
                title=f"Temperature Heatmap (sensor × session)",
                labels={"color": temp_label},
            )
            st.plotly_chart(fig_heat_t, use_container_width=True)

    # Low-pressure anomalies
    st.markdown("#### ⚠️ Low-Pressure Alerts (< 30 PSI)")
    df_low = df[df["pressure_psi"] < 30][
        ["sensor_id", "protocol", "export_dt", "pressure_psi", "pressure_kpa",
         temp_col, "packet_count"]
    ].sort_values("pressure_psi")
    if df_low.empty:
        st.success("No low-pressure readings in selected data.")
    else:
        st.dataframe(df_low, use_container_width=True)

    # Battery low events
    st.markdown("#### 🔋 Battery Low Events")
    df_bat = df[df["battery_low"] == True][
        ["sensor_id", "protocol", "export_dt", pressure_col, temp_col]
    ]
    if df_bat.empty:
        st.success("No battery-low flags in selected data.")
    else:
        st.dataframe(df_bat, use_container_width=True)

    # Packet count distribution
    st.markdown("#### 📦 Packet Count Distribution")
    fig_pkt = px.box(
        df, x="protocol", y="packet_count",
        color="protocol",
        title="Packet Count by Protocol",
        labels={"packet_count": "Packets per Reading"},
    )
    st.plotly_chart(fig_pkt, use_container_width=True)

    # New sensors per session
    st.markdown("#### 🆕 New Sensors per Session")
    seen_so_far = set()
    new_per_session = []
    for export_dt, grp in df.sort_values("export_dt").groupby("export_dt"):
        new_ids = set(grp["sensor_id"]) - seen_so_far
        seen_so_far |= set(grp["sensor_id"])
        new_per_session.append({
            "session_label": grp["session_label"].iloc[0],
            "new_sensors": len(new_ids),
            "cumulative_unique": len(seen_so_far),
        })
    df_new = pd.DataFrame(new_per_session)
    fig_new = make_subplots(specs=[[{"secondary_y": True}]])
    fig_new.add_trace(go.Bar(x=df_new["session_label"], y=df_new["new_sensors"],
                              name="New Sensors"), secondary_y=False)
    fig_new.add_trace(go.Scatter(x=df_new["session_label"], y=df_new["cumulative_unique"],
                                  mode="lines+markers", name="Cumulative Unique"), secondary_y=True)
    fig_new.update_layout(title_text="New vs Cumulative Sensors per Session")
    fig_new.update_xaxes(tickangle=-45)
    st.plotly_chart(fig_new, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 – RAW DATA
# ══════════════════════════════════════════════════════════════════════════════
with tab_raw:
    st.subheader("Raw Combined Data")
    st.caption(f"{len(df):,} rows after filtering")

    search = st.text_input("🔎 Search sensor ID or protocol", "")
    df_view = df.copy()
    if search:
        mask = df_view.apply(lambda r: search.lower() in str(r).lower(), axis=1)
        df_view = df_view[mask]

    st.dataframe(df_view.sort_values("export_dt", ascending=False), use_container_width=True)

    csv_data = df_view.to_csv(index=False)
    st.download_button(
        label="⬇️ Download filtered data as CSV",
        data=csv_data,
        file_name=f"tpms_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

# ─── FOOTER / AUTO-REFRESH ───────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "🛞 TPMS Dashboard · Auto-refresh polls export directory for new files · "
    f"Exports directory: `{EXPORTS_DIR}`"
)

if auto_refresh:
    time.sleep(REFRESH_INTERVAL_SECS)
    st.rerun()
