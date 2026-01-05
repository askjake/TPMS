#!/usr/bin/env python3
# TPMS Tracker - Streamlit UI (refactored for speed + online learning)
#
# Key fixes vs the original:
# - Replaces st.tabs with a horizontal segmented control (st.radio) so only ONE page executes per rerun.
# - Sidebar "Statistics" is live: rate/hr from last minute, signals last hour, repeated signals last hour.
# - Adds online pattern learning (time/location/sensor/pressure/temp) with incremental DB reads.
# - Adds WAL/busy_timeout + extra indexes (handled in database.py) to reduce sluggishness & lockups.

from __future__ import annotations

import time
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from database import TPMSDatabase
from ml_engine import VehicleClusteringEngine, OnlinePatternLearner

# -----------------------------
# App configuration
# -----------------------------

APP_TITLE = "TPMS Tracker - Intelligent Vehicle Pattern Recognition"
DEFAULT_PORT = 8507

# A safe default list; if your config.py defines FREQUENCIES_MHZ, we use it.
DEFAULT_FREQUENCIES = [314.9, 315.0, 433.92]

try:
    import config  # type: ignore

    FREQUENCIES = getattr(config, "FREQUENCIES_MHZ", DEFAULT_FREQUENCIES)
    DB_PATH = getattr(config, "DB_PATH", "tpms_tracker.db")
    TIMEZONE_LABEL = getattr(config, "TIMEZONE_LABEL", "Mountain Time (MT)")
except Exception:
    FREQUENCIES = DEFAULT_FREQUENCIES
    DB_PATH = "tpms_tracker.db"
    TIMEZONE_LABEL = "Mountain Time (MT)"


# -----------------------------
# Helpers
# -----------------------------

@st.cache_resource
def get_db(db_path: str) -> TPMSDatabase:
    db = TPMSDatabase(db_path)
    # Ensure extra indexes exist (fast dashboard queries)
    try:
        db.ensure_performance_indexes()
    except Exception:
        pass
    return db


def fmt_ts(ts: float) -> str:
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "—"


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


@st.cache_data(ttl=2.0)
def cached_realtime_stats(db_path: str) -> Dict[str, Any]:
    db = get_db(db_path)
    return db.get_realtime_stats()


def ensure_models(db_path: str):
    """Ensure expensive models are created once per session."""
    # Session-state defaults (avoid KeyError/AttributeError on first run or after hot-reload)
    if "last_model_update" not in st.session_state:
        st.session_state.last_model_update = None
    if "last_rowid" not in st.session_state:
        st.session_state.last_rowid = 0

    if "online_learner" not in st.session_state:
        st.session_state.online_learner = OnlinePatternLearner()
    if "vehicle_engine" not in st.session_state:
        db = get_db(db_path)
        st.session_state.vehicle_engine = VehicleClusteringEngine(db)

def process_online_learning(db_path: str, limit: int = 5000) -> int:
    """Pull new rows from DB and update online learner + vehicle co-occurrence."""
    ensure_models(db_path)
    db = get_db(db_path)

    rows = db.get_signals_since_rowid(last_rowid=int(st.session_state.last_rowid), limit=int(limit))
    if not rows:
        return 0

    st.session_state.online_learner.update_rows(rows)

    # Feed vehicle clustering engine (best-effort; depends on your engine's API)
    # VehicleClusteringEngine in your repo may support update_signal(...) or update(...)
    ve = st.session_state.vehicle_engine
    for r in rows:
        try:
            ve.update_signal(
                tpms_id=str(r.get("tpms_id")),
                timestamp=float(r.get("timestamp") or 0.0),
                pressure=safe_float(r.get("pressure_psi")),
                temperature=safe_float(r.get("temperature_c")),
                latitude=safe_float(r.get("latitude")),
                longitude=safe_float(r.get("longitude")),
            )
        except Exception:
            # Don't break UI if the engine has a different signature
            pass

    # Advance checkpoint
    st.session_state.last_rowid = max(st.session_state.last_rowid, int(st.session_state.online_learner.last_rowid))
    st.session_state.last_model_update = time.time()
    return len(rows)


def sidebar_control_panel(db_path: str):
    st.sidebar.markdown("## ⚙️ Control Panel")
    st.sidebar.caption(f"🕒 Displaying times in {TIMEZONE_LABEL}")

    # Frequency selector
    if "selected_freq" not in st.session_state:
        st.session_state.selected_freq = float(FREQUENCIES[0]) if FREQUENCIES else 314.9

    st.sidebar.markdown("### 📡 Scanner Control")
    st.session_state.selected_freq = st.sidebar.selectbox(
        "Select Frequency (MHz)",
        options=[float(x) for x in FREQUENCIES],
        index=max(0, [float(x) for x in FREQUENCIES].index(float(st.session_state.selected_freq))) if FREQUENCIES else 0,
        key="freq_select",
    )

    colA, colB = st.sidebar.columns(2)
    with colA:
        if st.button("Set Frequency"):
            # Hook for your scanner object (if present)
            scanner = st.session_state.get("scanner")
            if scanner and hasattr(scanner, "set_frequency"):
                try:
                    scanner.set_frequency(st.session_state.selected_freq)
                    st.success(f"Set frequency to {st.session_state.selected_freq} MHz")
                except Exception as e:
                    st.error(f"Could not set frequency: {e}")
            else:
                st.info("Scanner object not attached in this build (Live Detection page wires it up).")

    st.sidebar.markdown("---")

    # Start / stop scan controls (hooks)
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("▶️ Start Scan"):
            scanner = st.session_state.get("scanner")
            if scanner and hasattr(scanner, "start"):
                try:
                    scanner.start()
                    st.session_state.scan_status = "Active"
                except Exception as e:
                    st.error(f"Start failed: {e}")
            else:
                st.info("Scanner object not attached in this build (Live Detection page wires it up).")
    with col2:
        if st.button("⏹ Stop Scan"):
            scanner = st.session_state.get("scanner")
            if scanner and hasattr(scanner, "stop"):
                try:
                    scanner.stop()
                    st.session_state.scan_status = "Inactive"
                except Exception as e:
                    st.error(f"Stop failed: {e}")
            else:
                st.info("Scanner object not attached in this build (Live Detection page wires it up).")

    status = st.session_state.get("scan_status", "Inactive")
    st.sidebar.markdown("### 🟢 Status")
    st.sidebar.write(f"**{status}**")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Statistics (live)")

    stats = cached_realtime_stats(db_path)

    if isinstance(stats, dict) and stats.get("ok") is False:
        st.sidebar.warning(f"Stats error: {stats.get('error', 'unknown')}")



    # Requested stats
    st.sidebar.metric("Rate (signals/hr)", int(stats.get("rate_per_hour_last_min", 0)))
    st.sidebar.metric("Signals (last hour)", int(stats.get("n_last_hour", stats.get("signals_last_hour", 0))))
    st.sidebar.metric("Repeated signals (last hour)", int(stats.get("repeats_last_hour", stats.get("repeated_signals_last_hour", 0))))

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧾 Inventory")
    st.sidebar.metric("Known Vehicles", int(stats.get("known_vehicles", 0)))
    st.sidebar.metric("Known Sensors", int(stats.get("known_sensors", 0)))

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧠 Learning")
    ensure_models(db_path)
    st.sidebar.checkbox("Auto-learn (small batch each rerun)", value=True, key="auto_learn")
    st.sidebar.write(f"Last model update: **{fmt_ts(st.session_state.last_model_update)}**")
    if st.sidebar.button("Update model now"):
        n = process_online_learning(db_path, limit=10000)
        if n:
            st.sidebar.success(f"Learned from {n} new signals")
        else:
            st.sidebar.info("No new signals to learn from")


# -----------------------------
# Pages
# -----------------------------

def page_live_detection(db_path: str):
    st.header("🎯 Live Detection")

    st.info(
        "This page is where your HackRF/RTL-SDR scanning integration usually lives. "
        "To keep Analytics/ML/Maintenance fast, hardware modules are imported lazily here."
    )

    # Initialize scanner in session state if not present
    if 'scanner' not in st.session_state:
        st.session_state.scanner = None
    if 'scanner_attached' not in st.session_state:
        st.session_state.scanner_attached = False

    # Attempt to import scanner modules only when needed
    with st.expander("Scanner wiring (optional)", expanded=not st.session_state.scanner_attached):
        st.write("If your repo includes hackrf_interface / tpms_decoder, this will attach them to session_state.")
        if st.button("Attach scanner modules"):
            try:
                # Local imports – won't slow down the rest of the app
                import hackrf_interface  # type: ignore
                import tpms_decoder  # type: ignore
                from config import config  # type: ignore

                st.sidebar.caption(f"hackrf_interface loaded from: {hackrf_interface.__file__}")

                # Your project may have a scanner/controller class. We try a couple names.
                scanner = None
                if hasattr(hackrf_interface, "HackRFScanner"):
                    scanner = hackrf_interface.HackRFScanner()
                elif hasattr(hackrf_interface, "Scanner"):
                    scanner = hackrf_interface.Scanner()
                elif hasattr(hackrf_interface, "HackRFInterface"):
                    scanner = hackrf_interface.HackRFInterface()

                if scanner is not None:
                    # Get database and decoder
                    db = get_db(db_path)

                    DecoderCls = getattr(tpms_decoder, "TPMSDecoder", None)
                    if DecoderCls is None:
                        st.error("tpms_decoder.TPMSDecoder not found")
                        return

                    # Pick your actual sample rate source
                    sample_rate = getattr(config, "SAMPLE_RATE", 2_457_600)
                    decoder = DecoderCls(sample_rate)

                    # Attach the pipeline
                    scanner.attach_pipeline(db=db, decoder=decoder)

                    # Store in session state
                    st.session_state.scanner = scanner
                    st.session_state.scanner_attached = True
                    st.success("Scanner attached and pipeline configured!")
                else:
                    st.warning("Couldn't find HackRFScanner/Scanner/HackRFInterface class in hackrf_interface.")
            except Exception as e:
                st.error(f"Attach failed: {e}")
                import traceback
                st.code(traceback.format_exc())

    # Control buttons for scanner
    if st.session_state.scanner_attached and st.session_state.scanner:
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("▶️ Start Scanning"):
                try:
                    # Start without callback - uses attached pipeline
                    if st.session_state.scanner.start():
                        st.success("Scanning started!")
                    else:
                        st.error("Failed to start scanning")
                except Exception as e:
                    st.error(f"Start failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        with col2:
            if st.button("⏹️ Stop Scanning"):
                try:
                    st.session_state.scanner.stop()
                    st.info("Scanning stopped")
                except Exception as e:
                    st.error(f"Stop failed: {e}")

        with col3:
            status = st.session_state.scanner.get_status()
            if status.get('is_streaming'):
                st.success("🟢 Active")
            else:
                st.info("⚪ Idle")

    # Show most recent signals (quick)
    db = get_db(db_path)
    recent = db.get_recent_signals(time_window=120)
    df = pd.DataFrame(recent)

    if df.empty:
        st.write("No recent signals in the last 2 minutes.")
        return

    df["time"] = df["timestamp"].apply(fmt_ts)
    show_cols = [c for c in ["time", "tpms_id", "protocol", "pressure_psi", "temperature_c",
                             "battery_low", "signal_strength", "snr", "frequency"]
                 if c in df.columns]
    st.dataframe(df[show_cols].sort_values("time", ascending=False),
                 use_container_width=True, height=420)


def page_sensor_database(db_path: str):
    st.header("📡 Sensor Database")

    db = get_db(db_path)

    lookback_h = st.slider("Look back window (hours)", 1, 72, 24)
    since = time.time() - lookback_h * 3600

    # Fast top sensors query
    conn = db._connect(read_only=True)  # intentionally using the helper for speed
    cur = conn.cursor()
    cur.execute(
        """
        SELECT tpms_id,
               COUNT(*)             AS signals,
               MIN(timestamp)       AS first_ts,
               MAX(timestamp)       AS last_ts,
               AVG(signal_strength) AS avg_rssi,
               AVG(pressure_psi)    AS avg_pressure,
               AVG(temperature_c)   AS avg_temp,
               AVG(frequency)       AS avg_freq
        FROM tpms_signals
        WHERE timestamp >= ?
        GROUP BY tpms_id
        ORDER BY signals DESC
        LIMIT 1000
        """,
        (since,),
    )
    rows = cur.fetchall()
    conn.close()

    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        st.write("No sensors found in that window.")
        return

    df["first_seen"] = df["first_ts"].apply(fmt_ts)
    df["last_seen"] = df["last_ts"].apply(fmt_ts)

    display_cols = ["tpms_id", "signals", "first_seen", "last_seen", "avg_rssi",
                    "avg_pressure", "avg_temp", "avg_freq"]
    st.dataframe(df[display_cols], use_container_width=True, height=560)


def page_vehicle_database(db_path: str):
    st.header("🚗 Vehicle Database")

    db = get_db(db_path)

    st.caption("Vehicles are inferred from sensor co-occurrence. This page is lightweight by default.")
    min_enc = st.slider("Min encounters (filter)", 0, 20, 3)
    vehicles = db.get_all_vehicles(min_encounters=int(min_enc))

    if not vehicles:
        st.write("No vehicles found yet.")
        return

    df = pd.DataFrame(vehicles)
    show_cols = [c for c in ["vehicle_hash", "sensor_count", "encounter_count", "first_seen", "last_seen", "confidence_score"] if c in df.columns]
    if "first_seen" in df.columns:
        df["first_seen"] = df["first_seen"].apply(fmt_ts)
    if "last_seen" in df.columns:
        df["last_seen"] = df["last_seen"].apply(fmt_ts)

    st.dataframe(df[show_cols], use_container_width=True, height=520)

    with st.expander("Show sensors for a vehicle"):
        vh = st.selectbox("Vehicle hash", options=df["vehicle_hash"].tolist())
        details = db.get_vehicle_details(vh)
        st.json(details)


def page_analytics(db_path: str):
    st.header("📈 Analytics")

    db = get_db(db_path)

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        lookback_h = st.selectbox("Window", [1, 6, 12, 24, 48, 72], index=3)
    with col2:
        bucket = st.selectbox("Bucket", ["1 min", "5 min", "15 min", "1 hour"], index=1)
    with col3:
        st.caption("All charts are computed from a time-windowed subset (fast).")

    now = time.time()
    since = now - int(lookback_h) * 3600

    bucket_s = {"1 min": 60, "5 min": 300, "15 min": 900, "1 hour": 3600}[bucket]

    # Pull just timestamps (fast) and bucket in pandas
    conn = db._connect(read_only=True)
    cur = conn.cursor()
    cur.execute("SELECT timestamp FROM tpms_signals WHERE timestamp >= ? ORDER BY timestamp ASC", (since,))
    ts = [float(r[0]) for r in cur.fetchall()]
    conn.close()

    if not ts:
        st.write("No signals in that window.")
        return

    s = pd.Series(1, index=pd.to_datetime(ts, unit="s"))
    series = s.resample(f"{bucket_s}S").sum()

    st.line_chart(series)

    # Top sensors
    conn = db._connect(read_only=True)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT tpms_id, COUNT(*) AS n
        FROM tpms_signals
        WHERE timestamp >= ?
        GROUP BY tpms_id
        ORDER BY n DESC
        LIMIT 20
        """,
        (since,),
    )
    top = cur.fetchall()
    conn.close()

    top_df = pd.DataFrame([{"tpms_id": r[0], "signals": int(r[1])} for r in top])
    st.subheader("Top sensors (by signals)")
    st.dataframe(top_df, use_container_width=True, height=360)

    # Pressure/Temp distributions (sample)
    st.subheader("Pressure / Temperature (sample)")
    conn = db._connect(read_only=True)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT pressure_psi, temperature_c
        FROM tpms_signals
        WHERE timestamp >= ?
          AND pressure_psi IS NOT NULL
          AND temperature_c IS NOT NULL
        ORDER BY timestamp DESC
        LIMIT 5000
        """,
        (since,),
    )
    rows = cur.fetchall()
    conn.close()

    if rows:
        df = pd.DataFrame(rows, columns=["pressure_psi", "temperature_c"])
        st.scatter_chart(df, x="pressure_psi", y="temperature_c")


def page_maintenance(db_path: str):
    st.header("🛠 Maintenance")

    db = get_db(db_path)

    st.caption(
        "Maintenance analysis can get expensive if you run it over the whole DB. "
        "This page defaults to a time window and computes alerts on-demand."
    )

    lookback_days = st.slider("Analyze lookback (days)", 1, 30, 7)
    min_vehicle_enc = st.slider("Vehicles: min encounters", 0, 20, 3)

    vehicles = db.get_all_vehicles(min_encounters=int(min_vehicle_enc))
    if not vehicles:
        st.write("No vehicles found yet.")
        return

    vdf = pd.DataFrame(vehicles)
    if "vehicle_hash" not in vdf.columns:
        st.write("Vehicle data missing vehicle_hash column.")
        return

    selected = st.selectbox("Select a vehicle", options=vdf["vehicle_hash"].tolist())
    details = db.get_vehicle_details(selected)

    st.subheader("Vehicle details")
    st.json(details)

    if st.button("Run maintenance alert scan for this vehicle"):
        # Best-effort: analyze only sensors in this vehicle
        sensors = details.get("tpms_ids") or details.get("sensors") or []
        if isinstance(sensors, str):
            sensors = [s.strip() for s in sensors.split(",") if s.strip()]

        if not sensors:
            st.warning("No sensors found for this vehicle.")
            return

        since = time.time() - float(lookback_days) * 86400.0
        conn = db._connect(read_only=True)
        cur = conn.cursor()

        alerts = []
        for sid in sensors:
            cur.execute(
                """
                SELECT timestamp, pressure_psi, temperature_c, battery_low
                FROM tpms_signals
                WHERE tpms_id = ?
                  AND timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT 2000
                """,
                (sid, since),
            )
            rows = cur.fetchall()
            if not rows:
                continue
            sdf = pd.DataFrame(rows, columns=["timestamp", "pressure_psi", "temperature_c", "battery_low"])

            # simple heuristics
            p = pd.to_numeric(sdf["pressure_psi"], errors="coerce").dropna()
            t = pd.to_numeric(sdf["temperature_c"], errors="coerce").dropna()

            if len(p) >= 10:
                drift = float(p.iloc[0] - p.iloc[-1])
                if abs(drift) >= 2.5:
                    alerts.append({"tpms_id": sid, "type": "Pressure drift", "detail": f"{drift:+.1f} PSI over window"})

                if float(p.min()) <= 18:
                    alerts.append({"tpms_id": sid, "type": "Low pressure", "detail": f"min {float(p.min()):.1f} PSI"})

            if len(t) >= 10:
                if float(t.max()) >= 85:
                    alerts.append({"tpms_id": sid, "type": "High temp", "detail": f"max {float(t.max()):.1f}°C"})

            # battery low flags
            if "battery_low" in sdf.columns and sdf["battery_low"].fillna(0).astype(int).sum() > 0:
                alerts.append({"tpms_id": sid, "type": "Battery low", "detail": "battery_low observed"})

        conn.close()

        if not alerts:
            st.success("No alerts triggered by heuristics.")
        else:
            adf = pd.DataFrame(alerts)
            st.dataframe(adf, use_container_width=True, height=420)


def page_ml_insights(db_path: str):
    st.header("🧠 ML Insights")
    ensure_models(db_path)

    db = get_db(db_path)

    st.caption(
        "This page uses mixed learning: online incremental updates + batch analysis from a selected window.\n\n"
        "Online model updates are fast (only new DB rows). Batch analysis is optional and windowed."
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("Learn from new signals"):
            n = process_online_learning(db_path, limit=20000)
            if n:
                st.success(f"Updated models from {n} new signals")
            else:
                st.info("No new signals.")
    with col2:
        st.write(
            f"Online learner updates: **{st.session_state.online_learner.total_updates}**, "
            f"last rowid: **{st.session_state.online_learner.last_rowid}**"
        )

    # Choose a sensor
    lookback_h = st.slider("Sensor picker lookback (hours)", 1, 72, 24)
    since = time.time() - float(lookback_h) * 3600.0

    conn = db._connect(read_only=True)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT tpms_id, COUNT(*) AS n
        FROM tpms_signals
        WHERE timestamp >= ?
        GROUP BY tpms_id
        ORDER BY n DESC
        LIMIT 500
        """,
        (since,),
    )
    sensors = [r[0] for r in cur.fetchall()]
    conn.close()

    if not sensors:
        st.write("No sensors in that window.")
        return

    sensor = st.selectbox("Sensor ID", options=sensors)

    # Latest reading for sensor
    conn = db._connect(read_only=True)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT timestamp, pressure_psi, temperature_c, latitude, longitude
        FROM tpms_signals
        WHERE tpms_id = ?
        ORDER BY timestamp DESC
        LIMIT 1
        """,
        (sensor,),
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        st.write("No data for sensor.")
        return

    ts, p, t, lat, lon = float(row[0]), row[1], row[2], row[3], row[4]
    st.subheader("Latest reading")
    st.write(
        f"Time: **{fmt_ts(ts)}**  |  Pressure: **{p} PSI**  |  Temp: **{t} °C**  "
        f"|  Location: **({lat}, {lon})**"
    )

    # Prediction + anomaly
    pred = st.session_state.online_learner.predict(sensor_id=sensor, ts=ts, lat=safe_float(lat), lon=safe_float(lon))
    if pred.get("ok"):
        st.subheader("Predicted baseline (online)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pred Pressure", f"{pred['pressure_pred']:.1f} PSI")
        c2.metric("Pred Temp", f"{pred['temp_pred']:.1f} °C")
        c3.metric("Pressure conf", f"{pred['pressure_conf']:.2f}")
        c4.metric("Temp conf", f"{pred['temp_conf']:.2f}")

        an = st.session_state.online_learner.anomaly_scores(sensor, safe_float(p), safe_float(t), ts=ts)
        if an.get("ok"):
            z_p = an.get("z_pressure")
            z_t = an.get("z_temp")
            st.subheader("Anomaly score (z)")
            c1, c2 = st.columns(2)
            c1.metric("z(Pressure)", "—" if z_p is None else f"{z_p:+.2f}")
            c2.metric("z(Temp)", "—" if z_t is None else f"{z_t:+.2f}")

    else:
        st.info(pred.get("reason", "No prediction available."))

    # Batch pattern slice
    with st.expander("Batch patterns (windowed)"):
        w_h = st.slider("Batch window (hours)", 1, 168, 48)
        since2 = time.time() - float(w_h) * 3600.0

        conn = db._connect(read_only=True)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT timestamp, pressure_psi, temperature_c
            FROM tpms_signals
            WHERE tpms_id = ?
              AND timestamp >= ?
            ORDER BY timestamp ASC
            LIMIT 20000
            """,
            (sensor, since2),
        )
        rows = cur.fetchall()
        conn.close()

        if rows:
            df = pd.DataFrame(rows, columns=["timestamp", "pressure_psi", "temperature_c"])
            df["dt"] = pd.to_datetime(df["timestamp"], unit="s")
            df = df.set_index("dt").drop(columns=["timestamp"])
            st.line_chart(df.resample("15min").mean(numeric_only=True))

            # Hour-of-day profile
            df2 = df.copy()
            df2["hour"] = df2.index.hour
            prof = df2.groupby("hour")[["pressure_psi", "temperature_c"]].mean(numeric_only=True).reset_index()
            st.dataframe(prof, use_container_width=True)

def page_sensor_trigger(db_path: str):
    st.header("🧲 Sensor Trigger")

    st.info("Placeholder UI for your ESP32 / GPIO / SDR trigger controls.")
    st.write("If your project provides esp32_trigger_controller, wire it up here with lazy imports (same pattern as Live Detection).")


# -----------------------------
# Main
# -----------------------------

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    # Header
    st.markdown(f"# 🚗 {APP_TITLE}")
    st.caption(f"DB: {DB_PATH}")

    # Sidebar
    sidebar_control_panel(DB_PATH)

    # Page selection (segmented control)
    pages = [
        ("Live Detection", page_live_detection),
        ("Vehicle Database", page_vehicle_database),
        ("Sensor Database", page_sensor_database),
        ("Analytics", page_analytics),
        ("Maintenance", page_maintenance),
        ("ML Insights", page_ml_insights),
        ("Sensor Trigger", page_sensor_trigger),
    ]

    labels = [p[0] for p in pages]
    st.markdown(
        """
        <style>
        div[role="radiogroup"] > label {
            background: rgba(127,127,127,0.08);
            padding: 6px 10px;
            border-radius: 999px;
            margin-right: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    choice = st.radio("Navigation", labels, horizontal=True, label_visibility="collapsed")
    page_fn = dict(pages)[choice]

    # Opportunistic lightweight online learning on each run (small batch)
    # Keeps predictions "fresh" without making the UI sluggish.
    if st.session_state.get("auto_learn", True):
        try:
            process_online_learning(DB_PATH, limit=2000)
        except Exception:
            pass

    page_fn(DB_PATH)


if __name__ == "__main__":
    main()
