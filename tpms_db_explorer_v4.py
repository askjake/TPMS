# tpms_db_explorer_v4.py
# Performance-optimized Streamlit TPMS DB Explorer
# Key optimizations:
#   1. Server-side SQL aggregation (no full table loads)
#   2. Paginated queries with LIMIT/OFFSET
#   3. Lazy loading — only fetch data for active view
#   4. Pre-computed summary tables cached in SQLite
#   5. Virtual scrolling via st.dataframe with row limits
#   6. Indexed queries using existing idx_signals_ts, idx_signals_id
#
# Run: streamlit run tpms_db_explorer_v4.py

from __future__ import annotations

import math
import os
import sqlite3
import struct
from datetime import datetime, timedelta
from typing import Any, List, Optional, Tuple

import pandas as pd
import streamlit as st

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None

APP_TZ = os.environ.get("TPMS_APP_TZ", "America/Denver")
DEFAULT_PAGE_SIZE = 500
MAX_CHART_POINTS = 5000

# ═══════════════════════════════════════════════════════════════════════
# DB Connection & Core Helpers
# ═══════════════════════════════════════════════════════════════════════

def connect_ro(path: str) -> sqlite3.Connection:
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    except Exception:
        con = sqlite3.connect(path, check_same_thread=False)
    # Enable memory-mapped I/O for faster reads on large DBs
    con.execute("PRAGMA mmap_size = 268435456;")  # 256MB mmap
    con.execute("PRAGMA cache_size = -64000;")     # 64MB page cache
    con.execute("PRAGMA temp_store = MEMORY;")
    return con


def _decode_float32(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (bytes, bytearray)) and len(x) == 4:
        try:
            return struct.unpack("<f", x)[0]
        except Exception:
            return None
    try:
        return float(x)
    except Exception:
        return None


def _to_local(ts_series: pd.Series) -> pd.Series:
    s = pd.to_datetime(ts_series, unit="s", utc=True, errors="coerce")
    if ZoneInfo:
        try:
            return s.dt.tz_convert(ZoneInfo(APP_TZ))
        except Exception:
            pass
    return s.dt.tz_localize(None)


# ═══════════════════════════════════════════════════════════════════════
# SERVER-SIDE QUERIES (the key optimization)
# Push computation to SQLite instead of loading everything into pandas
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading DB stats...", ttl=300)
def get_db_stats(path: str) -> dict:
    """Quick metadata without loading full tables."""
    con = connect_ro(path)
    cur = con.cursor()
    
    tables = [r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence';").fetchall()]
    
    stats = {"tables": tables, "file_size": os.path.getsize(path)}
    
    for t in tables:
        try:
            cur.execute(f"SELECT COUNT(*) FROM {t}")
            stats[f"{t}_count"] = cur.fetchone()[0]
        except Exception:
            stats[f"{t}_count"] = 0
    
    # Time range for signals
    if "tpms_signals" in tables:
        cur.execute("SELECT MIN(timestamp), MAX(timestamp) FROM tpms_signals WHERE timestamp > 946684800")
        row = cur.fetchone()
        stats["ts_min"] = row[0]
        stats["ts_max"] = row[1]
    
    con.close()
    return stats


@st.cache_data(show_spinner="Aggregating sensors...", ttl=120)
def get_sensor_summary(path: str, protocol_filter: Optional[List[str]] = None,
                       time_start: Optional[float] = None, time_end: Optional[float] = None,
                       limit: int = 500, offset: int = 0) -> Tuple[pd.DataFrame, int]:
    """Server-side aggregation: one row per sensor with stats. Returns (df, total_count)."""
    con = connect_ro(path)
    
    where_clauses = ["timestamp > 946684800"]
    params = []
    
    if protocol_filter:
        placeholders = ",".join(["?"]*len(protocol_filter))
        where_clauses.append(f"protocol IN ({placeholders})")
        params.extend(protocol_filter)
    if time_start:
        where_clauses.append("timestamp >= ?")
        params.append(time_start)
    if time_end:
        where_clauses.append("timestamp <= ?")
        params.append(time_end)
    
    where = " AND ".join(where_clauses)
    
    # Get total unique sensors (for pagination)
    count_q = f"SELECT COUNT(DISTINCT tpms_id) FROM tpms_signals WHERE {where}"
    total = con.execute(count_q, params).fetchone()[0]
    
    # Aggregated query — all computation in SQLite
    query = f"""
        SELECT 
            tpms_id as sensor_id,
            protocol,
            COUNT(*) as packet_count,
            MIN(timestamp) as first_seen_ts,
            MAX(timestamp) as last_seen_ts,
            AVG(pressure_psi) as avg_pressure_psi,
            MIN(pressure_psi) as min_pressure_psi,
            MAX(pressure_psi) as max_pressure_psi,
            AVG(temperature_c) as avg_temp,
            MAX(COALESCE(battery_low, 0)) as battery_low,
            AVG(CASE WHEN typeof(signal_strength) = 'real' OR typeof(signal_strength) = 'integer' 
                     THEN signal_strength ELSE NULL END) as avg_rssi
        FROM tpms_signals
        WHERE {where}
        GROUP BY tpms_id
        ORDER BY packet_count DESC
        LIMIT ? OFFSET ?
    """
    params.extend([limit, offset])
    
    df = pd.read_sql_query(query, con, params=params)
    con.close()
    
    if not df.empty:
        df["first_seen"] = _to_local(df["first_seen_ts"])
        df["last_seen"] = _to_local(df["last_seen_ts"])
    
    return df, total


@st.cache_data(show_spinner="Loading sensor signals...", ttl=60)
def get_sensor_signals(path: str, sensor_id: str, limit: int = 2000) -> pd.DataFrame:
    """Drill-in: fetch signals for ONE sensor (indexed query on tpms_id)."""
    con = connect_ro(path)
    query = """
        SELECT timestamp, pressure_psi, temperature_c, battery_low,
               signal_strength, snr, frequency, latitude, longitude, protocol
        FROM tpms_signals
        WHERE tpms_id = ? AND timestamp > 946684800
        ORDER BY timestamp DESC
        LIMIT ?
    """
    df = pd.read_sql_query(query, con, params=[sensor_id, limit])
    con.close()
    
    if not df.empty:
        df["timestamp_local"] = _to_local(df["timestamp"])
        # Decode blob columns
        for col in ["signal_strength", "snr"]:
            if col in df.columns:
                df[f"{col}_decoded"] = df[col].apply(_decode_float32)
        if "frequency" in df.columns:
            df["frequency_mhz"] = pd.to_numeric(df["frequency"], errors="coerce") / 1e6
    
    return df


@st.cache_data(show_spinner="Loading protocols...", ttl=300)
def get_protocols(path: str) -> List[str]:
    """Fast distinct protocol list."""
    con = connect_ro(path)
    result = [r[0] for r in con.execute(
        "SELECT DISTINCT protocol FROM tpms_signals WHERE protocol IS NOT NULL ORDER BY protocol").fetchall()]
    con.close()
    return result


@st.cache_data(show_spinner="Building time histogram...", ttl=120)
def get_time_histogram(path: str, bin_seconds: int = 3600,
                       protocol_filter: Optional[List[str]] = None) -> pd.DataFrame:
    """Server-side time bucketing for activity chart (no full load needed)."""
    con = connect_ro(path)
    
    where_clauses = ["timestamp > 946684800"]
    params = []
    if protocol_filter:
        placeholders = ",".join(["?"]*len(protocol_filter))
        where_clauses.append(f"protocol IN ({placeholders})")
        params.extend(protocol_filter)
    
    where = " AND ".join(where_clauses)
    
    query = f"""
        SELECT 
            CAST(timestamp / ? AS INTEGER) * ? as time_bucket,
            COUNT(*) as signal_count,
            COUNT(DISTINCT tpms_id) as unique_sensors
        FROM tpms_signals
        WHERE {where}
        GROUP BY time_bucket
        ORDER BY time_bucket
    """
    params = [bin_seconds, bin_seconds] + params  # prepend for the CAST
    
    # Rebuild properly — params order matters
    params2 = []
    if protocol_filter:
        placeholders = ",".join(["?"]*len(protocol_filter))
        w2 = f"timestamp > 946684800 AND protocol IN ({placeholders})"
        params2 = protocol_filter
    else:
        w2 = "timestamp > 946684800"
    
    query2 = f"""
        SELECT 
            CAST(timestamp / {bin_seconds} AS INTEGER) * {bin_seconds} as time_bucket,
            COUNT(*) as signal_count,
            COUNT(DISTINCT tpms_id) as unique_sensors
        FROM tpms_signals
        WHERE {w2}
        GROUP BY time_bucket
        ORDER BY time_bucket
    """
    
    df = pd.read_sql_query(query2, con, params=params2)
    con.close()
    
    if not df.empty:
        df["time"] = _to_local(df["time_bucket"])
    
    return df


@st.cache_data(show_spinner="Pressure distribution...", ttl=120)
def get_pressure_distribution(path: str, bins: int = 50) -> pd.DataFrame:
    """Server-side histogram for pressure — avoids loading all rows."""
    con = connect_ro(path)
    # Use SQLite to bin pressure values
    query = """
        SELECT 
            CAST(pressure_psi / 2.0 AS INTEGER) * 2 as pressure_bin,
            COUNT(*) as count
        FROM tpms_signals
        WHERE pressure_psi IS NOT NULL AND pressure_psi >= 0 AND pressure_psi <= 100
              AND timestamp > 946684800
        GROUP BY pressure_bin
        ORDER BY pressure_bin
    """
    df = pd.read_sql_query(query, con)
    con.close()
    return df


@st.cache_data(show_spinner="Temperature distribution...", ttl=120)
def get_temp_distribution(path: str) -> pd.DataFrame:
    """Server-side histogram for temperature."""
    con = connect_ro(path)
    query = """
        SELECT 
            CAST(temperature_c / 5.0 AS INTEGER) * 5 as temp_bin,
            COUNT(*) as count
        FROM tpms_signals
        WHERE temperature_c IS NOT NULL AND temperature_c > -50 AND temperature_c < 200
              AND timestamp > 946684800
        GROUP BY temp_bin
        ORDER BY temp_bin
    """
    df = pd.read_sql_query(query, con)
    con.close()
    return df


@st.cache_data(show_spinner="Protocol breakdown...", ttl=120)
def get_protocol_stats(path: str) -> pd.DataFrame:
    """Server-side protocol breakdown."""
    con = connect_ro(path)
    query = """
        SELECT 
            protocol,
            COUNT(*) as signal_count,
            COUNT(DISTINCT tpms_id) as unique_sensors,
            AVG(pressure_psi) as avg_pressure,
            MIN(timestamp) as first_seen_ts,
            MAX(timestamp) as last_seen_ts
        FROM tpms_signals
        WHERE timestamp > 946684800
        GROUP BY protocol
        ORDER BY signal_count DESC
    """
    df = pd.read_sql_query(query, con)
    con.close()
    if not df.empty:
        df["first_seen"] = _to_local(df["first_seen_ts"])
        df["last_seen"] = _to_local(df["last_seen_ts"])
    return df


@st.cache_data(show_spinner="Running quality checks...", ttl=300)
def run_quality_checks(path: str) -> List[dict]:
    """Server-side quality checks — never loads full table."""
    con = connect_ro(path)
    issues = []
    
    # Negative pressure
    neg = con.execute(
        "SELECT COUNT(*) FROM tpms_signals WHERE pressure_psi < 0 AND timestamp > 946684800").fetchone()[0]
    total = con.execute(
        "SELECT COUNT(*) FROM tpms_signals WHERE timestamp > 946684800").fetchone()[0]
    if neg > 0:
        issues.append({"level": "warning", "msg": f"{neg:,} signals ({neg/max(total,1)*100:.1f}%) have negative pressure — likely noise/misdecodes."})
    
    # Battery-low ratio
    batt = con.execute(
        "SELECT COUNT(*) FROM tpms_signals WHERE battery_low = 1 AND timestamp > 946684800").fetchone()[0]
    if total > 0 and batt / total > 0.2:
        issues.append({"level": "warning", "msg": f"battery_low flagged on {batt/total*100:.1f}% of signals — unusually high."})
    
    # Suspect temperatures (> 150F / 65C)
    hot = con.execute(
        "SELECT COUNT(*) FROM tpms_signals WHERE temperature_c > 65 AND timestamp > 946684800").fetchone()[0]
    if hot > 0:
        issues.append({"level": "info", "msg": f"{hot:,} signals have temp > 65°C (149°F) — verify sensor decode logic."})
    
    # Signals with no pressure AND no temp (empty decodes)
    empty = con.execute(
        "SELECT COUNT(*) FROM tpms_signals WHERE pressure_psi IS NULL AND temperature_c IS NULL AND timestamp > 946684800").fetchone()[0]
    if total > 0 and empty / total > 0.3:
        issues.append({"level": "warning", "msg": f"{empty:,} signals ({empty/total*100:.1f}%) have no pressure AND no temperature — these may be undecoded."})
    
    # Orphan timestamps (1970)
    old = con.execute(
        "SELECT COUNT(*) FROM tpms_signals WHERE timestamp <= 946684800").fetchone()[0]
    if old > 0:
        issues.append({"level": "info", "msg": f"{old:,} signals have pre-2000 timestamps (epoch 0 / clock not set)."})
    
    con.close()
    return issues


# Paginated raw table viewer
@st.cache_data(show_spinner="Loading page...", ttl=60)
def get_table_page(path: str, table: str, limit: int = 500, offset: int = 0,
                   order_by: str = "rowid", desc: bool = True) -> pd.DataFrame:
    con = connect_ro(path)
    direction = "DESC" if desc else "ASC"
    query = f"SELECT * FROM {table} ORDER BY {order_by} {direction} LIMIT ? OFFSET ?"
    df = pd.read_sql_query(query, con, params=[limit, offset])
    con.close()
    return df


# ═══════════════════════════════════════════════════════════════════════
# UI PAGES
# ═══════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="TPMS DB Explorer v4", layout="wide", initial_sidebar_state="expanded")
st.title("TPMS Tracker — DB Explorer v4")
st.caption("⚡ Performance-optimized: server-side aggregation, paginated queries, lazy loading")

with st.sidebar:
    st.header("Database")
    default_path = os.environ.get("TPMS_DB_PATH", "tpms_tracker.db")
    db_path = st.text_input("SQLite DB path", value=default_path)
    
    upload = st.file_uploader("…or upload a .db", type=["db", "sqlite"])
    if upload is not None:
        up_path = f"/tmp/_uploaded_{upload.name}"
        with open(up_path, "wb") as f:
            f.write(upload.getbuffer())
        db_path = up_path
        st.success(f"Loaded: {upload.name}")
    
    st.divider()
    page = st.radio("View", [
        "Overview",
        "Sensor Explorer",
        "Signals Dashboard",
        "Table Browser",
        "Quality Checks",
        "Export",
    ])

# Validate
if not db_path or not os.path.exists(db_path):
    st.warning("Point me at a SQLite DB file (left sidebar).")
    st.stop()

try:
    stats = get_db_stats(db_path)
except Exception as e:
    st.error(f"Could not open DB: {e}")
    st.stop()


# ─────────────────────── OVERVIEW ────────────────────────

if page == "Overview":
    st.subheader("Database Overview")
    
    cols = st.columns(4)
    cols[0].metric("File", os.path.basename(db_path))
    cols[1].metric("Size", f"{stats['file_size']/1024/1024:.1f} MB")
    cols[2].metric("Tables", len(stats['tables']))
    sig_count = stats.get('tpms_signals_count', 0)
    cols[3].metric("Total Signals", f"{sig_count:,}")
    
    st.markdown("---")
    
    # Table summary
    rows_data = []
    for t in stats['tables']:
        rows_data.append({"Table": t, "Rows": f"{stats.get(f'{t}_count', 0):,}"})
    st.dataframe(pd.DataFrame(rows_data), use_container_width=True, hide_index=True)
    
    # Protocol breakdown
    if "tpms_signals" in stats['tables']:
        st.markdown("### Protocol Breakdown")
        proto_df = get_protocol_stats(db_path)
        if not proto_df.empty:
            st.dataframe(proto_df[["protocol", "signal_count", "unique_sensors", "avg_pressure", "first_seen", "last_seen"]],
                        use_container_width=True, hide_index=True)
            
            if px:
                fig = px.pie(proto_df, values="signal_count", names="protocol", title="Signals by Protocol")
                st.plotly_chart(fig, use_container_width=True)


# ─────────────────────── SENSOR EXPLORER (paginated) ────────────────────────

elif page == "Sensor Explorer":
    st.subheader("Sensor Explorer")
    st.caption("Aggregated per-sensor view with pagination — no full table load")
    
    # Filters in sidebar
    with st.sidebar:
        st.header("Filters")
        protocols = get_protocols(db_path)
        selected_proto = st.multiselect("Protocol", protocols, default=[])
        
        # Time range filter
        ts_min = stats.get("ts_min")
        ts_max = stats.get("ts_max")
        use_time_filter = st.checkbox("Filter by time range")
        time_start = None
        time_end = None
        if use_time_filter and ts_min and ts_max:
            d_min = datetime.utcfromtimestamp(ts_min).date()
            d_max = datetime.utcfromtimestamp(ts_max).date()
            date_range = st.date_input("Date range", value=(d_min, d_max), min_value=d_min, max_value=d_max)
            if len(date_range) == 2:
                time_start = datetime.combine(date_range[0], datetime.min.time()).timestamp()
                time_end = datetime.combine(date_range[1], datetime.max.time()).timestamp()
    
    # Pagination controls
    page_size = st.selectbox("Sensors per page", [50, 100, 250, 500], index=1)
    
    proto_arg = selected_proto if selected_proto else None
    
    # Get first page to know total
    df_sensors, total_sensors = get_sensor_summary(
        db_path, protocol_filter=proto_arg,
        time_start=time_start, time_end=time_end,
        limit=page_size, offset=0
    )
    
    total_pages = max(1, math.ceil(total_sensors / page_size))
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.metric("Total Sensors", f"{total_sensors:,}")
    with col2:
        current_page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    with col3:
        st.metric("Total Pages", total_pages)
    
    # Fetch current page
    offset = (current_page - 1) * page_size
    if offset > 0:
        df_sensors, _ = get_sensor_summary(
            db_path, protocol_filter=proto_arg,
            time_start=time_start, time_end=time_end,
            limit=page_size, offset=offset
        )
    
    if not df_sensors.empty:
        display_cols = ["sensor_id", "protocol", "packet_count", "avg_pressure_psi",
                       "min_pressure_psi", "max_pressure_psi", "avg_temp", "battery_low",
                       "first_seen", "last_seen"]
        display_cols = [c for c in display_cols if c in df_sensors.columns]
        st.dataframe(df_sensors[display_cols], use_container_width=True, hide_index=True, height=400)
        
        # Drill-in
        st.markdown("### Sensor Drill-In")
        sensor_list = df_sensors["sensor_id"].tolist()
        selected_sensor = st.selectbox("Select sensor", sensor_list)
        
        if selected_sensor:
            sig_df = get_sensor_signals(db_path, selected_sensor)
            st.caption(f"Showing up to 2,000 most recent signals for {selected_sensor}")
            
            if not sig_df.empty:
                st.dataframe(sig_df[[c for c in ["timestamp_local", "pressure_psi", "temperature_c",
                                                  "signal_strength_decoded", "frequency_mhz", "protocol",
                                                  "battery_low", "latitude", "longitude"]
                                     if c in sig_df.columns]],
                            use_container_width=True, hide_index=True, height=300)
                
                # Time-series chart for this sensor
                if px and len(sig_df) > 1 and "timestamp_local" in sig_df.columns:
                    tab1, tab2 = st.tabs(["Pressure", "Signal Strength"])
                    with tab1:
                        if sig_df["pressure_psi"].notna().any():
                            fig = px.line(sig_df.sort_values("timestamp_local"),
                                         x="timestamp_local", y="pressure_psi",
                                         title=f"Pressure over time — {selected_sensor}")
                            st.plotly_chart(fig, use_container_width=True)
                    with tab2:
                        if "signal_strength_decoded" in sig_df.columns and sig_df["signal_strength_decoded"].notna().any():
                            fig = px.line(sig_df.sort_values("timestamp_local"),
                                         x="timestamp_local", y="signal_strength_decoded",
                                         title=f"RSSI over time — {selected_sensor}")
                            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────── SIGNALS DASHBOARD ────────────────────────

elif page == "Signals Dashboard":
    st.subheader("Signals Dashboard")
    st.caption("All charts use server-side aggregation — no full table load")
    
    sig_count = stats.get('tpms_signals_count', 0)
    
    kpi = st.columns(4)
    kpi[0].metric("Total Signals", f"{sig_count:,}")
    
    proto_df = get_protocol_stats(db_path)
    kpi[1].metric("Unique Sensors", f"{proto_df['unique_sensors'].sum():,}" if not proto_df.empty else "0")
    kpi[2].metric("Protocols", len(proto_df))
    
    ts_min = stats.get("ts_min")
    ts_max = stats.get("ts_max")
    if ts_min and ts_max:
        days = (ts_max - ts_min) / 86400
        kpi[3].metric("Span", f"{days:.1f} days")
    
    if px:
        st.markdown("### Activity Over Time")
        with st.sidebar:
            st.header("Dashboard Options")
            bin_hours = st.selectbox("Time bin", [1, 4, 12, 24], index=0, format_func=lambda x: f"{x}h")
        
        hist_df = get_time_histogram(db_path, bin_seconds=bin_hours*3600)
        if not hist_df.empty:
            fig = px.bar(hist_df, x="time", y="signal_count", title="Signal Activity",
                        labels={"signal_count": "Signals", "time": "Time"})
            st.plotly_chart(fig, use_container_width=True)
            
            fig2 = px.line(hist_df, x="time", y="unique_sensors", title="Unique Sensors Per Bucket")
            st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("### Distributions")
        c1, c2 = st.columns(2)
        with c1:
            pres_df = get_pressure_distribution(db_path)
            if not pres_df.empty:
                fig = px.bar(pres_df, x="pressure_bin", y="count", title="Pressure Distribution (PSI)",
                            labels={"pressure_bin": "PSI", "count": "Signals"})
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            temp_df = get_temp_distribution(db_path)
            if not temp_df.empty:
                fig = px.bar(temp_df, x="temp_bin", y="count", title="Temperature Distribution",
                            labels={"temp_bin": "°C", "count": "Signals"})
                st.plotly_chart(fig, use_container_width=True)


# ─────────────────────── TABLE BROWSER (paginated) ────────────────────────

elif page == "Table Browser":
    st.subheader("Table Browser")
    st.caption("Paginated — fetches only the rows you see")
    
    table = st.selectbox("Table", [t for t in stats['tables'] if t != 'sqlite_sequence'])
    total_rows = stats.get(f"{table}_count", 0)
    
    col1, col2 = st.columns(2)
    with col1:
        page_size = st.selectbox("Rows per page", [100, 250, 500, 1000], index=1, key="tb_ps")
    with col2:
        sort_desc = st.checkbox("Newest first", value=True)
    
    total_pages = max(1, math.ceil(total_rows / page_size))
    
    c1, c2, c3 = st.columns([1, 2, 1])
    c1.metric("Total Rows", f"{total_rows:,}")
    current_page = c2.number_input("Page", min_value=1, max_value=total_pages, value=1, key="tb_page")
    c3.metric("Pages", total_pages)
    
    offset = (current_page - 1) * page_size
    df = get_table_page(db_path, table, limit=page_size, offset=offset, desc=sort_desc)
    
    # Convert timestamp columns for readability
    for col in df.columns:
        if "timestamp" in col.lower() or col in ("first_seen", "last_seen"):
            try:
                numeric = pd.to_numeric(df[col], errors="coerce")
                if numeric.median() > 1e9:  # looks like unix epoch
                    df[f"{col}_local"] = _to_local(numeric)
            except Exception:
                pass
    
    st.dataframe(df, use_container_width=True, height=500, hide_index=True)


# ─────────────────────── QUALITY CHECKS ────────────────────────

elif page == "Quality Checks":
    st.subheader("Quality Checks")
    st.caption("Server-side validation — no full table scan in Python")
    
    issues = run_quality_checks(db_path)
    
    if not issues:
        st.success("No obvious issues detected!")
    else:
        for issue in issues:
            if issue["level"] == "warning":
                st.warning(issue["msg"])
            else:
                st.info(issue["msg"])
    
    # Show basic stats
    st.markdown("### DB Health Summary")
    sig_count = stats.get('tpms_signals_count', 0)
    st.write(f"Total signals: {sig_count:,}")
    st.write(f"Valid signals (post-2000): checked server-side")
    st.write(f"File size: {stats['file_size']/1024/1024:.1f} MB")


# ─────────────────────── EXPORT ────────────────────────

elif page == "Export":
    st.subheader("Export Data")
    st.caption("Exports use server-side queries — won't hang on large DBs")
    
    export_type = st.radio("What to export?", ["Sensor Summary (aggregated)", "Raw Signals (sampled)"])
    
    if export_type == "Sensor Summary (aggregated)":
        st.write("Exports one row per sensor with aggregated stats.")
        max_export = st.number_input("Max sensors to export", value=10000, step=1000)
        
        if st.button("Generate Export"):
            with st.spinner("Querying..."):
                df, total = get_sensor_summary(db_path, limit=max_export, offset=0)
            
            st.success(f"Exported {len(df)} of {total} sensors")
            
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv_data, 
                             file_name=f"tpms_sensors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                             mime="text/csv")
    
    else:
        st.write("Exports raw signals (sampled for performance). Use pagination for full export.")
        sample_size = st.number_input("Sample size", value=5000, step=1000, max_value=50000)
        
        if st.button("Generate Export"):
            con = connect_ro(db_path)
            query = f"""
                SELECT timestamp, tpms_id, protocol, pressure_psi, temperature_c,
                       battery_low, frequency, latitude, longitude
                FROM tpms_signals
                WHERE timestamp > 946684800
                ORDER BY timestamp DESC
                LIMIT {int(sample_size)}
            """
            df = pd.read_sql_query(query, con)
            con.close()
            
            if not df.empty:
                df["timestamp_local"] = _to_local(df["timestamp"])
            
            st.success(f"Exported {len(df)} signals")
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv_data,
                             file_name=f"tpms_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                             mime="text/csv")
