"""
TPMS Sensor Statistics Dashboard v2 - Performance Optimized
=============================================================
Key improvements over v1:
  1. Lazy loading: Only loads files needed for current view
  2. Pagination: Sensor list is paginated (configurable page size)
  3. Large file detection: Archives auto-summarized without full load
  4. Sampling for charts: Max 5K points for Plotly (no browser hang)
  5. Session-aware caching

Run:  streamlit run tpms_dashboard_v2.py --server.address 0.0.0.0 --server.port 8569
"""

import os
import re
import json
import math
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- CONFIG ---
EXPORTS_DIR = Path(__file__).parent / "exports"
ARCHIVE_DIR = EXPORTS_DIR / "archive"
PAGE_TITLE = "TPMS Sensor Dashboard v2"
DEFAULT_PAGE_SIZE = 100
MAX_CHART_POINTS = 5000
LARGE_FILE_THRESHOLD_MB = 5

# --- PAGE SETUP ---
st.set_page_config(page_title=PAGE_TITLE, page_icon="\U0001f6de", layout="wide",
                   initial_sidebar_state="expanded")

# --- HELPERS ---

def exports_fingerprint(directory: Path) -> str:
    files = sorted(directory.glob("*.json"))
    parts = [f"{f.stat().st_mtime}:{f.stat().st_size}" for f in files]
    return hashlib.md5("|".join(parts).encode()).hexdigest()


@st.cache_data(show_spinner="Loading daily exports...", ttl=120)
def load_daily_exports(fingerprint: str) -> pd.DataFrame:
    """Load only small daily sync exports (< threshold), merge them."""
    frames = []
    for fpath in sorted(EXPORTS_DIR.glob("*.json")):
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
    if len(df) <= max_pts:
        return df
    step = len(df) // max_pts
    return df.iloc[::step].copy()




@st.cache_data(show_spinner="Building sensor timelines...", ttl=120)
def build_sensor_timelines(fingerprint: str) -> pd.DataFrame:
    """
    For every sensor across all daily exports, build one row per
    (sensor_id, export_session) so we can track history over time.
    Also computes derived time columns used by both new views.
    """
    frames = []
    for fpath in sorted(EXPORTS_DIR.glob("*.json")):
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

    tl["hour"]          = tl["export_dt"].dt.hour
    tl["weekday_num"]   = tl["export_dt"].dt.weekday   # 0=Mon
    tl["weekday"]       = tl["export_dt"].dt.strftime("%a")
    tl["date"]          = tl["export_dt"].dt.date
    tl["week"]          = tl["export_dt"].dt.isocalendar().week.astype(int)
    tl["session_label"] = tl["export_dt"].dt.strftime("%b %d %H:%M")

    session_counts      = tl.groupby("sensor_id")["export_file"].nunique()
    tl["session_count"] = tl["sensor_id"].map(session_counts)
    tl["is_recurring"]  = tl["session_count"] > 1

    return tl


@st.cache_data(show_spinner="Computing observation patterns...", ttl=120)
def build_observation_heatmap(fingerprint: str) -> tuple:
    """
    Returns (heatmap_df, hourly_df, weekly_df) for the Observation Patterns view.
    heatmap_df  – rows=weekday, cols=hour, values=unique sensor count
    hourly_df   – hour-of-day totals
    weekly_df   – captures per calendar date
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

    all_hours  = pd.DataFrame({"hour": range(24)})
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


# === SIDEBAR ===
with st.sidebar:
    st.header("TPMS Dashboard v2")
    st.caption("Performance-optimized")
    view = st.radio("View", [
        "Daily Sync Overview",
        "Archive Explorer",
        "Sensor History",
        "Observation Patterns",
        "Sensor Search",
        "Trends & Charts",
    ])
    st.divider()
    st.caption(f"Exports: {EXPORTS_DIR}")
    if ARCHIVE_DIR.exists():
        st.caption(f"Archive: {ARCHIVE_DIR}")


# === VIEW: Daily Sync Overview ===
if view == "Daily Sync Overview":
    st.header("Daily Sync Overview")
    st.caption("Shows data from daily sync exports (small files). For large tracker archives, use Archive Explorer.")
    fp = exports_fingerprint(EXPORTS_DIR)
    df = load_daily_exports(fp)
    if df.empty:
        st.warning("No daily export files found.")
        st.stop()

    k = st.columns(5)
    k[0].metric("Sessions", df["export_file"].nunique())
    k[1].metric("Unique Sensors", df["sensor_id"].nunique() if "sensor_id" in df.columns else 0)
    k[2].metric("Records", f"{len(df):,}")
    if "first_seen" in df.columns and df["first_seen"].notna().any():
        k[3].metric("Earliest", str(df["first_seen"].min().date()))
    if "last_seen" in df.columns and df["last_seen"].notna().any():
        k[4].metric("Latest", str(df["last_seen"].max().date()))

    st.markdown("### Sessions")
    if "export_dt" in df.columns and "sensor_id" in df.columns:
        session_stats = df.groupby("export_file").agg(
            sensors=("sensor_id", "nunique"),
            records=("sensor_id", "count"),
            export_date=("export_dt", "first"),
            avg_psi=("pressure_psi", "mean"),
        ).sort_values("export_date", ascending=False).reset_index()
        st.dataframe(session_stats, use_container_width=True, hide_index=True)

    st.markdown("### Latest Export Sensors")
    latest_file = df.sort_values("export_dt", ascending=False)["export_file"].iloc[0]
    latest_df = df[df["export_file"] == latest_file].copy()
    display_cols = [c for c in ["sensor_id", "protocol", "pressure_psi", "temp_f",
                                "rssi_dbm", "packet_count", "first_seen", "last_seen",
                                "battery_low"] if c in latest_df.columns]
    st.dataframe(latest_df[display_cols].sort_values("packet_count", ascending=False),
                 use_container_width=True, hide_index=True, height=400)

    if "pressure_psi" in df.columns:
        st.markdown("### Pressure Distribution (All Daily Sessions)")
        chart_df = downsample_for_chart(df)
        fig = px.histogram(chart_df, x="pressure_psi", nbins=40,
                          color="protocol" if "protocol" in chart_df.columns else None,
                          title="Pressure Distribution (PSI)")
        st.plotly_chart(fig, use_container_width=True)


# === VIEW: Archive Explorer ===
elif view == "Archive Explorer":
    st.header("Archive Explorer")
    st.caption("Browse large tracker database exports with pagination - no browser hang")

    all_files = []
    for directory in [EXPORTS_DIR, ARCHIVE_DIR]:
        if directory.exists():
            for fpath in sorted(directory.glob("*.json")):
                size_mb = fpath.stat().st_size / 1024 / 1024
                if size_mb > LARGE_FILE_THRESHOLD_MB:
                    all_files.append({"path": str(fpath), "name": fpath.name, "size_mb": size_mb})

    if not all_files:
        st.info("No large archive files found. Large exports (>5MB) appear here automatically.")
        st.stop()

    file_options = [f"{f['name']} ({f['size_mb']:.1f} MB)" for f in all_files]
    selected_idx = st.selectbox("Archive file", range(len(file_options)),
                                format_func=lambda i: file_options[i])
    selected_file = all_files[selected_idx]["path"]

    st.markdown("### Summary")
    summary = get_large_file_summary(selected_file)
    s_cols = st.columns(4)
    s_cols[0].metric("Total Sensors", f"{summary['total_sensors']:,}")
    s_cols[1].metric("File Size", f"{summary['size_mb']:.1f} MB")
    s_cols[2].metric("Protocols", len(summary['protocols']))
    if summary['pressure_stats']['mean']:
        s_cols[3].metric("Avg Pressure", f"{summary['pressure_stats']['mean']:.1f} PSI")

    st.markdown("### Protocols")
    proto_df = pd.DataFrame([
        {"Protocol": k, "Count": v, "Pct": f"{v/summary['total_sensors']*100:.1f}%"}
        for k, v in sorted(summary['protocols'].items(), key=lambda x: -x[1])
    ])
    st.dataframe(proto_df, use_container_width=True, hide_index=True)

    st.markdown("### Sensor Browser (Paginated)")
    filter_col, sort_col, page_col = st.columns(3)
    with filter_col:
        proto_options = ["All"] + sorted(summary['protocols'].keys())
        proto_filter = st.selectbox("Protocol filter", proto_options)
    with sort_col:
        sort_by = st.selectbox("Sort by", ["packet_count", "pressure_psi", "sensor_id", "first_seen"])
    with page_col:
        page_size = st.selectbox("Per page", [50, 100, 250, 500], index=1)

    proto_arg = proto_filter if proto_filter != "All" else None
    page_df, total_filtered = load_large_file_page(
        selected_file, offset=0, limit=page_size,
        sort_by=sort_by, ascending=(sort_by == "sensor_id"),
        protocol_filter=proto_arg
    )
    total_pages = max(1, math.ceil(total_filtered / page_size))
    nav_cols = st.columns([1, 3, 1])
    nav_cols[0].metric("Filtered", f"{total_filtered:,}")
    current_page = nav_cols[1].number_input("Page", min_value=1, max_value=total_pages, value=1)
    nav_cols[2].metric("Pages", total_pages)

    offset = (current_page - 1) * page_size
    if offset > 0:
        page_df, _ = load_large_file_page(
            selected_file, offset=offset, limit=page_size,
            sort_by=sort_by, ascending=(sort_by == "sensor_id"),
            protocol_filter=proto_arg
        )
    if not page_df.empty:
        display_cols = [c for c in ["sensor_id", "protocol", "packet_count", "pressure_psi",
                                    "pressure_kpa", "temp_c", "temp_f", "rssi_dbm",
                                    "first_seen", "last_seen", "battery_low"]
                       if c in page_df.columns]
        st.dataframe(page_df[display_cols], use_container_width=True, hide_index=True, height=450)


# === VIEW: Sensor Search ===
elif view == "Sensor Search":
    st.header("Sensor Search")
    st.caption("Search for a specific sensor ID across all exports (daily + archive)")
    search_id = st.text_input("Sensor ID (partial match)", placeholder="e.g. 00F14B3B")

    if search_id and len(search_id) >= 3:
        results = []
        search_upper = search_id.upper()

        fp = exports_fingerprint(EXPORTS_DIR)
        daily_df = load_daily_exports(fp)
        if not daily_df.empty and "sensor_id" in daily_df.columns:
            matches = daily_df[daily_df["sensor_id"].str.upper().str.contains(search_upper, na=False)]
            if not matches.empty:
                for _, row in matches.iterrows():
                    results.append({**row.to_dict(), "source": "daily"})

        for directory in [EXPORTS_DIR, ARCHIVE_DIR]:
            if not directory.exists():
                continue
            for fpath in sorted(directory.glob("*.json")):
                if fpath.stat().st_size / 1024 / 1024 <= LARGE_FILE_THRESHOLD_MB:
                    continue
                with open(fpath) as f:
                    data = json.load(f)
                for rec in data:
                    sid = rec.get("sensor_id", "")
                    if search_upper in sid.upper():
                        rec["source"] = fpath.name
                        results.append(rec)

        if results:
            st.success(f"Found {len(results)} match(es)")
            result_df = pd.DataFrame(results)
            display_cols = [c for c in ["sensor_id", "protocol", "packet_count",
                                        "pressure_psi", "temp_f", "rssi_dbm",
                                        "first_seen", "last_seen", "source"]
                          if c in result_df.columns]
            st.dataframe(result_df[display_cols], use_container_width=True, hide_index=True)
        else:
            st.warning(f"No sensors matching '{search_id}' found.")
    elif search_id:
        st.info("Enter at least 3 characters to search.")


# === VIEW: Sensor History ===
elif view == "Sensor History":
    st.header("Sensor Capture History")
    st.caption(
        "Per-sensor timeline across all daily export sessions. "
        "Tracks pressure, RSSI, and packet-count changes over time."
    )

    fp = exports_fingerprint(EXPORTS_DIR)
    tl = build_sensor_timelines(fp)

    if tl.empty or "sensor_id" not in tl.columns:
        st.warning("No timeline data available yet.")
        st.stop()

    with st.sidebar:
        st.header("History Filters")
        show_only_recurring = st.checkbox("Recurring sensors only", value=True)
        max_sessions = int(tl["session_count"].max()) if tl["session_count"].max() > 1 else 2
        min_sessions = st.slider("Min sessions seen", 1, max_sessions, 2)
        proto_list = sorted(tl["protocol"].dropna().unique().tolist()) if "protocol" in tl.columns else []
        proto_filter_h = st.multiselect("Protocol", proto_list, default=[], key="hist_proto")

    cands = tl.copy()
    if show_only_recurring:
        cands = cands[cands["is_recurring"]]
    cands = cands[cands["session_count"] >= min_sessions]
    if proto_filter_h:
        cands = cands[cands["protocol"].isin(proto_filter_h)]

    if cands.empty:
        st.info("No sensors match the current filters. Try loosening the criteria in the sidebar.")
        st.stop()

    st.markdown("### Recurring Sensors — Ranked by Observation Count")
    summary = (
        cands.sort_values("export_dt")
        .groupby("sensor_id")
        .agg(
            protocol      = ("protocol", "first"),
            sessions      = ("export_file", "nunique"),
            first_seen_dt = ("export_dt", "min"),
            last_seen_dt  = ("export_dt", "max"),
            avg_psi       = ("pressure_psi", "mean"),
            last_psi      = ("pressure_psi", "last"),
            avg_rssi      = ("rssi_dbm", "mean"),
            total_pkts    = ("packet_count", "sum"),
            battery_ever  = ("battery_low", "max"),
        )
        .reset_index()
    )
    summary["span_days"] = (summary["last_seen_dt"] - summary["first_seen_dt"]).dt.days
    summary["avg_psi"]   = summary["avg_psi"].round(1)
    summary["avg_rssi"]  = summary["avg_rssi"].round(1)
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
        "Select sensor", sensor_choices,
        format_func=lambda s: (
            f"{s}  ({summary.loc[summary['sensor_id']==s,'sessions'].values[0]} sessions)"
        )
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
    raw_cols = [c for c in ["session_label", "pressure_psi", "pressure_kpa",
                             "temp_f", "temp_c", "rssi_dbm", "packet_count",
                             "battery_low", "export_file"]
                if c in sdf.columns]
    st.dataframe(
        sdf[raw_cols].rename(columns={"session_label": "Session"}),
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
        fig_psi.add_hrect(y0=0,  y1=25, fillcolor="red",    opacity=0.07, line_width=0,
                          annotation_text="Critical (<25 PSI)", annotation_position="top left")
        fig_psi.add_hrect(y0=25, y1=32, fillcolor="orange",  opacity=0.07, line_width=0,
                          annotation_text="Low (25-32)")
        fig_psi.add_hrect(y0=32, y1=50, fillcolor="green",   opacity=0.05, line_width=0,
                          annotation_text="Normal (32-45)")
        fig_psi.update_layout(title=f"Pressure history — {selected_sensor}",
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
            fig_rssi.add_hrect(y0=-120, y1=-85, fillcolor="red",    opacity=0.06, line_width=0, annotation_text="Weak")
            fig_rssi.add_hrect(y0=-85,  y1=-60, fillcolor="yellow", opacity=0.06, line_width=0, annotation_text="OK")
            fig_rssi.add_hrect(y0=-60,  y1=0,   fillcolor="green",  opacity=0.06, line_width=0, annotation_text="Strong")
            fig_rssi.update_layout(title=f"RSSI history — {selected_sensor}",
                                    xaxis_title="Capture Date", yaxis_title="dBm",
                                    hovermode="x unified")
            st.plotly_chart(fig_rssi, use_container_width=True)

        if "packet_count" in sdf.columns and sdf["packet_count"].notna().any():
            st.markdown("#### Packet count per capture")
            fig_pkts = px.bar(sdf, x="export_dt", y="packet_count",
                              title=f"Packets per capture — {selected_sensor}",
                              labels={"export_dt": "Capture Date", "packet_count": "Packets"})
            st.plotly_chart(fig_pkts, use_container_width=True)

        st.markdown("#### Inter-capture gaps")
        gaps_h = sdf["export_dt"].sort_values().diff().dropna().dt.total_seconds().div(3600).round(1).values
        gap_df = pd.DataFrame({
            "From": sdf["session_label"].iloc[:-1].values,
            "To":   sdf["session_label"].iloc[1:].values,
            "Gap (h)": gaps_h,
            "Gap (days)": (gaps_h / 24).round(1),
        })
        st.dataframe(gap_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### All Recurring Sensors — Pressure Heatmap")
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
            title="Pressure (PSI) — Sensor × Session",
            labels={"color": "PSI", "x": "Session", "y": "Sensor ID"},
        )
        fig_hm.update_layout(height=max(300, len(pivot) * 28 + 100))
        st.plotly_chart(fig_hm, use_container_width=True)


# === VIEW: Observation Patterns ===
elif view == "Observation Patterns":
    st.header("Observation Patterns")
    st.caption(
        "When are sensors captured? Reveals time-of-day and day-of-week rhythms — "
        "useful for understanding receiver location and vehicle traffic patterns."
    )

    fp = exports_fingerprint(EXPORTS_DIR)
    heatmap_df, hourly_df, weekly_df = build_observation_heatmap(fp)
    tl = build_sensor_timelines(fp)

    if tl.empty:
        st.warning("No export data found.")
        st.stop()

    total_captures = len(tl)
    unique_sensors  = tl["sensor_id"].nunique()
    total_sessions  = tl["export_file"].nunique()
    span_days       = (tl["export_dt"].max() - tl["export_dt"].min()).days

    k = st.columns(4)
    k[0].metric("Total Captures", f"{total_captures:,}")
    k[1].metric("Unique Sensors", f"{unique_sensors:,}")
    k[2].metric("Sessions", total_sessions)
    k[3].metric("Observation Span", f"{span_days} days")

    st.markdown("---")

    # 1. Weekday x Hour heatmap
    st.markdown("### Capture Density: Day of Week vs Hour of Day")
    st.caption("Colour = unique sensors observed in that weekday/hour bucket, summed across all sessions.")
    if not heatmap_df.empty:
        fig_heat = px.imshow(
            heatmap_df,
            color_continuous_scale="Blues",
            aspect="auto",
            title="When are sensors observed? (Day x Hour)",
            labels={"color": "Unique Sensors", "x": "Hour of Day", "y": "Weekday"},
        )
        fig_heat.update_xaxes(
            dtick=1, tickvals=list(range(24)),
            ticktext=[f"{h:02d}:00" for h in range(24)]
        )
        fig_heat.update_layout(height=320)
        st.plotly_chart(fig_heat, use_container_width=True)

    # 2. Hour-of-day bar
    st.markdown("### Volume by Hour of Day")
    if not hourly_df.empty:
        hourly_df["hour_label"] = hourly_df["hour"].apply(lambda h: f"{h:02d}:00")
        fig_hour = px.bar(
            hourly_df, x="hour_label", y="unique_sensors",
            title="Unique Sensors Observed per Hour of Day (all sessions)",
            labels={"hour_label": "Hour", "unique_sensors": "Unique Sensors"},
            color="unique_sensors", color_continuous_scale="Blues",
        )
        fig_hour.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_hour, use_container_width=True)

    # 3. Day-of-week bar
    st.markdown("### Volume by Day of Week")
    DAYS_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    day_agg = (
        tl.groupby("weekday")["sensor_id"]
        .nunique()
        .reindex(DAYS_ORDER, fill_value=0)
        .reset_index()
    )
    day_agg.columns = ["Weekday", "Unique Sensors"]
    fig_day = px.bar(
        day_agg, x="Weekday", y="Unique Sensors",
        title="Unique Sensors Observed per Day of Week",
        color="Unique Sensors", color_continuous_scale="Greens",
    )
    fig_day.update_layout(showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig_day, use_container_width=True)

    # 4. Weekly cadence
    if not weekly_df.empty and len(weekly_df) > 1:
        st.markdown("### Capture Cadence Over Time")
        fig_wk = go.Figure()
        fig_wk.add_trace(go.Bar(
            x=weekly_df["date"].astype(str), y=weekly_df["unique_sensors"],
            name="Unique Sensors", marker_color="#1f77b4",
        ))
        fig_wk.add_trace(go.Scatter(
            x=weekly_df["date"].astype(str), y=weekly_df["total_captures"],
            name="Total Captures", mode="lines+markers",
            line=dict(color="orange", width=2), yaxis="y2",
        ))
        fig_wk.update_layout(
            title="Observation Cadence Over Calendar Time",
            xaxis_title="Date",
            yaxis=dict(title="Unique Sensors"),
            yaxis2=dict(title="Total Captures", overlaying="y", side="right"),
            hovermode="x unified",
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig_wk, use_container_width=True)

    # 5. Recurring vs new per session
    st.markdown("### Recurring vs. New Sensors per Session")
    st.caption("Are the same vehicles being seen repeatedly, or mostly new ones each session?")
    if "is_recurring" in tl.columns:
        recur_by_session = (
            tl.groupby(["session_label", "export_dt", "is_recurring"])["sensor_id"]
            .nunique()
            .reset_index(name="count")
            .sort_values("export_dt")
        )
        recur_by_session["type"] = recur_by_session["is_recurring"].map(
            {True: "Recurring", False: "First-time"}
        )
        fig_stack = px.bar(
            recur_by_session, x="session_label", y="count", color="type",
            title="Recurring vs. First-time Sensors per Session",
            labels={"session_label": "Session", "count": "Unique Sensors", "type": ""},
            color_discrete_map={"Recurring": "#1f77b4", "First-time": "#aec7e8"},
            barmode="stack",
        )
        st.plotly_chart(fig_stack, use_container_width=True)

    # 6. Interpretation callout
    st.markdown("---")
    st.markdown("### Interpretation")
    peak_hour = int(hourly_df.loc[hourly_df["unique_sensors"].idxmax(), "hour"]) if not hourly_df.empty else 0
    active_days = day_agg[day_agg["Unique Sensors"] > 0]["Weekday"].tolist()
    zero_days   = day_agg[day_agg["Unique Sensors"] == 0]["Weekday"].tolist()
    recurring_pct = float(tl["is_recurring"].mean() * 100) if "is_recurring" in tl.columns else 0.0

    col1, col2 = st.columns(2)
    with col1:
        st.info(
            f"**Peak capture hour:** {peak_hour:02d}:00\n\n"
            f"**Active days:** {', '.join(active_days)}\n\n"
            f"**No-capture days:** {', '.join(zero_days) if zero_days else 'None'}"
        )
    with col2:
        if zero_days:
            st.warning(
                f"No captures on {', '.join(zero_days)} — suggests the receiver "
                f"is at a location that is unoccupied on those days "
                f"(e.g. an office parking lot or workplace)."
            )
        st.success(
            f"**{recurring_pct:.0f}%** of all sensor observations are from recurring vehicles "
            f"(seen in more than one session)."
        )


# === VIEW: Trends & Charts ===
elif view == "Trends & Charts":
    st.header("Trends & Charts")
    st.caption("Cross-session analysis (daily exports). Charts are downsampled for performance.")
    fp = exports_fingerprint(EXPORTS_DIR)
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
                    title="Unique Sensors Detected Per Session")
        st.plotly_chart(fig, use_container_width=True)

    if "sensor_id" in df.columns and "pressure_psi" in df.columns:
        st.markdown("### Pressure Trends (Recurring Sensors)")
        sensor_freq = df.groupby("sensor_id")["export_file"].nunique()
        recurring = sensor_freq[sensor_freq > 1].index.tolist()
        if recurring:
            top_recurring = recurring[:20]
            selected = st.multiselect("Select sensors to track", top_recurring,
                                     default=top_recurring[:min(3, len(top_recurring))])
            if selected:
                trend_df = df[df["sensor_id"].isin(selected)].copy()
                trend_df = downsample_for_chart(trend_df)
                fig = px.line(trend_df.sort_values("export_dt"),
                            x="export_dt", y="pressure_psi", color="sensor_id",
                            title="Pressure Over Time (Recurring Sensors)")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No sensors found in multiple sessions yet.")

    if "protocol" in df.columns and "export_dt" in df.columns:
        st.markdown("### Protocol Mix Over Time")
        proto_time = df.groupby(["session_label", "protocol"]).size().reset_index(name="count")
        fig = px.bar(proto_time, x="session_label", y="count", color="protocol",
                    title="Protocol Distribution By Session")
        st.plotly_chart(fig, use_container_width=True)
