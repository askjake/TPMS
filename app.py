#!/usr/bin/env python3
# TPMS Tracker - Streamlit UI (refactored for speed + online learning)
#
# Key fixes:
# - Replaced deprecated use_container_width with width='stretch'/'content'.
# - Analytics now converts UTC to US/Mountain time.
# - Added "Repeated Sensor Analysis" (Map + Timeline) to Analytics.

from __future__ import annotations

import time
import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
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

# Hardcoded for Denver based on your location context
# Pandas uses pytz format usually
PYTZ_TIMEZONE = "US/Mountain"


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

def create_sensor_link(tpms_id: str) -> str:
    """Create a clickable sensor ID that navigates to detail page"""
    # Store in session state for navigation
    if st.button(f"🔍 {tpms_id}", key=f"link_{tpms_id}_{id(tpms_id)}", help=f"View details for {tpms_id}"):
        st.session_state.previous_page = st.session_state.get('current_page', 'Live Detection')
        st.session_state.current_page = 'Sensor Detail'
        st.session_state.selected_sensor = tpms_id
        st.rerun()
    return tpms_id


def fmt_ts(ts: float) -> str:
    try:
        # Convert to local time for display string
        dt = datetime.fromtimestamp(float(ts))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "—"


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v):
            return None
        v = v
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
        index=max(0,
                  [float(x) for x in FREQUENCIES].index(float(st.session_state.selected_freq))) if FREQUENCIES else 0,
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
    st.sidebar.metric("Repeated signals (last hour)",
                      int(stats.get("repeats_last_hour", stats.get("repeated_signals_last_hour", 0))))

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
    st.header("🎯 Live Detection - Multi-Scanner")

    st.info(
        "Scan multiple frequencies simultaneously with multiple SDR devices. "
        "Perfect for determining which frequency your car uses!"
    )

    # Initialize GPS manager first (before multi-scanner)
    if 'gps_manager' not in st.session_state:
        from gps_manager import GPSManager
        st.session_state.gps_manager = GPSManager()
        # Note: logger might not be defined in app.py, so let's use st.info instead
        st.info("GPS manager initialized")

    # Initialize multi-scanner manager with GPS
    if 'multi_scanner' not in st.session_state:
        st.session_state.multi_scanner = None
    if 'scanner_counter' not in st.session_state:
        st.session_state.scanner_counter = 0

    # Initialize multi-scanner manager
    if st.session_state.multi_scanner is None:
        try:
            import tpms_decoder
            from config import config

            db = get_db(db_path)
            DecoderCls = getattr(tpms_decoder, "TPMSDecoder", None)
            if DecoderCls is None:
                st.error("tpms_decoder.TPMSDecoder not found")
                return

            sample_rate = getattr(config, "SAMPLE_RATE", 2_457_600)

            from multi_scanner_manager import MultiScannerManager

            # 🔥 PASS GPS MANAGER TO MULTI-SCANNER
            st.session_state.multi_scanner = MultiScannerManager(
                db=db,
                decoder_class=DecoderCls,
                sample_rate=sample_rate,
                gps_manager=st.session_state.gps_manager  # ← GPS MANAGER ATTACHED HERE
            )
            st.success("Multi-scanner manager initialized with GPS support!")
        except Exception as e:
            st.error(f"Failed to initialize multi-scanner manager: {e}")
            import traceback
            st.code(traceback.format_exc())
            return

    manager = st.session_state.multi_scanner

    # 🔥 GPS STATUS INDICATOR
    gps_status = st.session_state.gps_manager.get_status()
    col_gps1, col_gps2, col_gps3 = st.columns([1, 2, 1])

    with col_gps1:
        if gps_status['active'] and gps_status['has_fix']:
            st.success("🛰️ GPS: Active (Fix)")
        elif gps_status['active']:
            st.warning("🛰️ GPS: Active (No Fix)")
        else:
            st.info("🛰️ GPS: Inactive")

    with col_gps2:
        if gps_status['has_fix']:
            st.caption(
                f"📍 Location: {gps_status['latitude']:.6f}, {gps_status['longitude']:.6f} | Sats: {gps_status['satellites']}")
        else:
            st.caption("📍 No GPS fix - signals will be saved without location data")

    with col_gps3:
        # Note: st.switch_page might not work in your Streamlit version
        # Alternative: just show a message or use st.page_link if available
        if st.button("⚙️ GPS Setup"):
            st.info("Navigate to GPS Setup page from the main menu")
            # Or if you have page navigation working:
            # st.switch_page("GPS Setup")

    st.markdown("---")

    # Add new scanner section
    with st.expander("➕ Add New Scanner", expanded=manager.get_scanner_count() == 0):
        st.write("Configure and add a new SDR scanner")

        col1, col2, col3 = st.columns(3)

        with col1:
            hw_type = st.selectbox(
                "Hardware Type",
                options=["Auto-detect", "HackRF", "RTL-SDR", "Simulation"],
                key="new_hw_type"
            )

        with col2:
            freq_mhz = st.selectbox(
                "Frequency (MHz)",
                options=FREQUENCIES,
                key="new_freq"
            )

        with col3:
            scanner_name = st.text_input(
                "Scanner Name (optional)",
                placeholder=f"Scanner {st.session_state.scanner_counter + 1}",
                key="new_scanner_name"
            )

        if st.button("🔌 Add Scanner", type="primary"):
            try:
                import hardware_manager

                hw_map = {
                    "Auto-detect": None,
                    "HackRF": "hackrf",
                    "RTL-SDR": "rtlsdr",
                    "Simulation": "simulation"
                }

                preferred = hw_map[hw_type]
                hw_manager = hardware_manager.HardwareManager(preferred=preferred)
                scanner = hw_manager.get_interface()
                actual_hw_type = hw_manager.get_hardware_type()

                if scanner is None or actual_hw_type is None:
                    st.error(f"❌ Hardware initialization failed for {hw_type}.")
                    st.info(
                        "If HackRF shows as plugged in but cannot be opened, select RTL-SDR or Simulation from the dropdown.")
                    return

                # Generate scanner ID
                if scanner_name and scanner_name.strip():
                    scanner_id = scanner_name.strip()
                else:
                    st.session_state.scanner_counter += 1
                    scanner_id = f"scanner_{st.session_state.scanner_counter}"

                # Add to manager
                freq_hz = float(freq_mhz) * 1e6
                if manager.add_scanner(scanner_id, actual_hw_type, freq_hz, scanner):
                    st.success(f"✅ Added {scanner_id}: {actual_hw_type.upper()} @ {freq_mhz} MHz")
                    st.rerun()
                else:
                    st.error(f"Failed to add scanner (ID may already exist)")

            except Exception as e:
                st.error(f"Error adding scanner: {e}")
                import traceback
                st.code(traceback.format_exc())

    # Display active scanners
    st.markdown("---")
    st.subheader("📡 Active Scanners")

    status = manager.get_status()

    if not status:
        st.info("No scanners configured. Add a scanner above to get started!")
        return

    # Create a row for each scanner
    for scanner_id, info in status.items():
        with st.container():
            col1, col2, col3, col4, col5, col6 = st.columns([2, 1.5, 1, 1, 1, 1])

            with col1:
                # Safe hardware type handling
                hw_type_str = info.get('hardware_type') or 'unknown'
                hw_icon = {"hackrf": "🔴", "rtlsdr": "📻", "simulation": "🎮"}.get(hw_type_str, "❓")
                st.write(f"{hw_icon} **{scanner_id}**")
                st.caption(f"{hw_type_str.upper()}")

            with col2:
                freq = info.get('frequency', 0)
                st.metric("Frequency", f"{freq:.2f} MHz")

            with col3:
                is_running = info.get('is_running', False)
                if is_running:
                    st.success("🟢 Active")
                else:
                    st.info("⚪ Idle")

            with col4:
                if is_running:
                    if st.button("⏹️ Stop", key=f"stop_{scanner_id}"):
                        if manager.stop_scanner(scanner_id):
                            st.success(f"Stopped {scanner_id}")
                            st.rerun()
                else:
                    if st.button("▶️ Start", key=f"start_{scanner_id}"):
                        if manager.start_scanner(scanner_id):
                            st.success(f"Started {scanner_id}")
                            st.rerun()

            with col5:
                # Change frequency
                freq = info.get('frequency', FREQUENCIES[0])
                try:
                    current_idx = FREQUENCIES.index(freq) if freq in FREQUENCIES else 0
                except (ValueError, IndexError):
                    current_idx = 0

                new_freq = st.selectbox(
                    "Freq",
                    options=FREQUENCIES,
                    index=current_idx,
                    key=f"freq_{scanner_id}",
                    label_visibility="collapsed"
                )
                if new_freq != freq:
                    if manager.change_frequency(scanner_id, float(new_freq) * 1e6):
                        st.rerun()

            with col6:
                if st.button("🗑️", key=f"remove_{scanner_id}", help="Remove scanner"):
                    if manager.remove_scanner(scanner_id):
                        st.success(f"Removed {scanner_id}")
                        st.rerun()

            # Show error if present
            if 'error' in info:
                st.error(f"Error: {info['error']}")

            st.markdown("---")

    # Global controls
    st.subheader("🎛️ Global Controls")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("▶️ Start All", type="primary"):
            results = manager.start_all()
            success_count = sum(1 for v in results.values() if v)
            st.success(f"Started {success_count}/{len(results)} scanners")
            st.rerun()

    with col2:
        if st.button("⏹️ Stop All"):
            results = manager.stop_all()
            st.info("All scanners stopped")
            st.rerun()

    with col3:
        st.metric("Total Scanners", manager.get_scanner_count())

    with col4:
        st.metric("Running", manager.get_running_count())

    # Show recent signals grouped by frequency
    st.markdown("---")
    st.subheader("📊 Recent Detections (Last 2 Minutes)")

    db = get_db(db_path)
    recent = db.get_recent_signals(time_window=120)
    df = pd.DataFrame(recent)

    if df.empty:
        st.write("No recent signals detected.")
        return

    df["time"] = df["timestamp"].apply(fmt_ts)

    # Group by frequency
    if "frequency" in df.columns:
        for freq in sorted(df["frequency"].unique()):
            freq_df = df[df["frequency"] == freq].copy()

            with st.expander(f"📡 {freq / 1e6:.2f} MHz ({len(freq_df)} signals)", expanded=True):
                # Display with clickable sensor IDs
                for idx, row in freq_df.sort_values("time", ascending=False).head(20).iterrows():
                    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])

                    with col1:
                        if st.button(f"🔍 {row['tpms_id']}", key=f"sensor_live_{idx}"):
                            st.session_state.previous_page = 'Live Detection'
                            st.session_state.current_page = 'Sensor Detail'
                            st.session_state.selected_sensor = row['tpms_id']
                            st.rerun()

                    with col2:
                        st.write(f"{row.get('pressure_psi', 'N/A')} PSI")

                    with col3:
                        st.write(f"{row.get('temperature_c', 'N/A')}°C")

                    with col4:
                        st.write(f"{row.get('signal_strength', 'N/A'):.1f} dB")

                    with col5:
                        st.caption(row['time'])

    else:
        show_cols = [c for c in ["time", "tpms_id", "protocol", "pressure_psi", "temperature_c",
                                 "battery_low", "signal_strength", "snr", "frequency", "latitude", "longitude"]
                     if c in df.columns]
        st.dataframe(df[show_cols].sort_values("time", ascending=False),
                     width="stretch", height=420)

    # 🔥 REMOVED DUPLICATE CODE THAT WAS AT THE BOTTOM


# In app.py or a background service
def reprocess_database_signals(db_path: str, decoder):
    """Periodically reprocess unknown signals from database"""
    db = TPMSDatabase(db_path)

    # Get unknown signals from database
    unknown_signals = db.get_unknown_signals_for_reprocessing(max_retries=5, limit=100)

    if unknown_signals:
        logger.info(f"Reprocessing {len(unknown_signals)} signals from database...")

        for sig_data in unknown_signals:
            # Reconstruct UnknownSignal
            unknown = UnknownSignal(
                timestamp=sig_data['timestamp'],
                frequency=sig_data['frequency'],
                signal_strength=sig_data['signal_strength'],
                modulation_type=sig_data.get('modulation_type', 'Unknown'),
                baud_rate=sig_data.get('baud_rate'),
                packet_length=sig_data.get('packet_length', 0),
                pattern_signature=sig_data.get('pattern_signature', ''),
                raw_samples=np.frombuffer(sig_data['raw_samples'], dtype=np.complex64) if sig_data.get(
                    'raw_samples') else np.array([]),
                retry_count=sig_data.get('retry_count', 0)
            )

            # Queue for reprocessing
            decoder.reprocessor.queue_for_reprocessing(unknown)

        # Process the queue
        recovered = decoder.process_reprocessing_queue()

        if recovered:
            logger.info(f"✅ Recovered {len(recovered)} signals from database!")

            # Save recovered signals back to database
            for signal in recovered:
                db.insert_signal({
                    'tpms_id': signal.tpms_id,
                    'timestamp': signal.timestamp,
                    'frequency': signal.frequency,
                    'signal_strength': signal.signal_strength,
                    'snr': signal.snr,
                    'pressure_psi': signal.pressure_psi,
                    'temperature_c': signal.temperature_c,
                    'battery_low': signal.battery_low,
                    'protocol': signal.protocol,
                    'raw_data': signal.raw_data,
                    'confidence': signal.confidence,
                    'reprocessed': 1  # Mark as reprocessed
                })


# Run periodically (e.g., every 5 minutes)
import threading
import time


def start_background_reprocessing(db_path: str, decoder, interval: int = 300):
    """Start background reprocessing thread"""

    def worker():
        while True:
            try:
                reprocess_database_signals(db_path, decoder)
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Background reprocessing error: {e}")
                time.sleep(interval)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread


def page_sensor_database(db_path: str):
    st.header("📡 Sensor Database & Intelligence Report")

    db = get_db(db_path)

    # === COMPREHENSIVE REPROCESSING SECTION ===
    st.subheader("🔄 Comprehensive Signal Reprocessing")

    with st.expander("🚀 Reprocess ALL Unknown Signals", expanded=False):
        st.info("""
        **What this does:**
        - Reprocesses EVERY unknown signal in the database
        - Tries all 7 reprocessing strategies on each signal
        - Can recover signals that failed initial decoding
        - May take several minutes depending on database size
        """)

        # Get count of unknown signals
        try:
            conn = db._connect(read_only=True)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM unknown_signals WHERE decoded = 0")
            unknown_count = cur.fetchone()[0]
            conn.close()

            st.metric("Unknown Signals Ready for Reprocessing", unknown_count)

            if unknown_count > 0:
                col1, col2 = st.columns(2)

                with col1:
                    batch_size = st.number_input("Batch Size", min_value=10, max_value=1000, value=100)

                with col2:
                    st.write("")  # Spacing

                if st.button("🚀 Start Comprehensive Reprocessing", type="primary"):
                    # Import decoder
                    try:
                        import tpms_decoder
                        from config import config

                        # Create decoder instance
                        sample_rate = getattr(config, "SAMPLE_RATE", 2_457_600)
                        decoder = tpms_decoder.TPMSDecoder(sample_rate, db._connect())

                        # Run reprocessing with progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        status_text.text("Starting comprehensive reprocessing...")

                        # Run reprocessing
                        stats = decoder.reprocess_all_unknown_signals(db._connect(), batch_size=batch_size)

                        progress_bar.progress(100)

                        if 'error' in stats:
                            st.error(f"Error: {stats['error']}")
                        else:
                            st.success(f"""
                            ✅ Reprocessing Complete!

                            - **Processed:** {stats['total_processed']} signals
                            - **Recovered:** {stats['total_recovered']} signals
                            - **Success Rate:** {stats['success_rate']:.1f}%
                            - **Duration:** {stats['duration_seconds']:.1f} seconds
                            """)

                            if stats['by_strategy']:
                                st.write("**Strategy Breakdown:**")
                                strategy_df = pd.DataFrame([
                                    {'Strategy': k, 'Recovered': v}
                                    for k, v in stats['by_strategy'].items()
                                ])
                                st.dataframe(strategy_df, width="stretch")

                            st.balloons()

                    except Exception as e:
                        st.error(f"Failed to run reprocessing: {e}")
                        import traceback
                        st.code(traceback.format_exc())
            else:
                st.success("✅ No unknown signals to reprocess!")

        except Exception as e:
            st.error(f"Error checking unknown signals: {e}")

    st.markdown("---")
    
    # Get decoder statistics
    try:
        decoder_stats = db.get_reprocessing_statistics()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            success_rate = (decoder_stats.get('successfully_decoded', 0) /
                            max(decoder_stats.get('total_unknown', 1), 1) * 100)
            st.metric(
                "Recovery Success Rate",
                f"{success_rate:.1f}%",
                help="Percentage of failed signals successfully recovered through intelligent reprocessing"
            )

        with col2:
            st.metric(
                "Signals Recovered",
                decoder_stats.get('successfully_decoded', 0),
                help="Total signals decoded after initial failure"
            )

        with col3:
            st.metric(
                "Protocols Discovered",
                decoder_stats.get('discovered_protocols', 0),
                help="New TPMS protocols learned automatically"
            )

        with col4:
            st.metric(
                "Pending Analysis",
                decoder_stats.get('pending_reprocessing', 0),
                help="Unknown signals queued for reprocessing"
            )

        # Detailed intelligence breakdown
        with st.expander("🔬 Detailed Intelligence Report", expanded=True):
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Success Breakdown",
                "🔄 Reprocessing Strategies",
                "🔍 Discovered Protocols",
                "⚠️ Failed Signals"
            ])

            with tab1:
                st.markdown("### What Went Well")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**✅ Successful Decodes**")

                    # Get protocol breakdown
                    conn = db._connect(read_only=True)
                    cur = conn.cursor()
                    cur.execute("""
                                SELECT protocol, COUNT(*) as count, AVG(confidence) as avg_conf
                                FROM tpms_signals
                                WHERE timestamp >= ?
                                GROUP BY protocol
                                ORDER BY count DESC
                                LIMIT 10
                                """, (time.time() - 86400,))  # Last 24 hours

                    protocol_stats = cur.fetchall()
                    conn.close()

                    if protocol_stats:
                        for proto, count, conf in protocol_stats:
                            confidence_icon = "🟢" if conf and conf > 0.8 else "🟡" if conf and conf > 0.6 else "🔴"
                            st.write(f"{confidence_icon} **{proto}**: {count} signals (avg conf: {conf:.2f})")
                    else:
                        st.info("No successful decodes in last 24h")

                with col2:
                    st.markdown("**🔧 Recovered Signals**")

                    # Get reprocessed signals
                    conn = db._connect(read_only=True)
                    cur = conn.cursor()
                    cur.execute("""
                                SELECT COUNT(*) as count, AVG(confidence) as avg_conf
                                FROM tpms_signals
                                WHERE reprocessed = 1
                                  AND timestamp >= ?
                                """, (time.time() - 86400,))

                    reprocessed = cur.fetchone()
                    conn.close()

                    if reprocessed and reprocessed[0] > 0:
                        st.success(f"🎯 {reprocessed[0]} signals recovered!")
                        st.write(f"Average confidence: {reprocessed[1]:.2f}")
                        st.caption("These signals would have been lost without intelligent reprocessing")
                    else:
                        st.info("No reprocessed signals in last 24h")

            with tab2:
                st.markdown("### 🔄 Reprocessing Strategy Performance")

                success_by_strategy = decoder_stats.get('success_by_strategy', {})

                if success_by_strategy:
                    # Create DataFrame for visualization
                    strategy_df = pd.DataFrame([
                        {'Strategy': k, 'Successes': v}
                        for k, v in success_by_strategy.items()
                    ]).sort_values('Successes', ascending=False)

                    st.bar_chart(strategy_df.set_index('Strategy'))

                    st.markdown("**Strategy Descriptions:**")
                    strategy_help = {
                        '_retry_with_relaxed_thresholds': '🎚️ Lowered signal quality requirements',
                        '_retry_with_inverted_bits': '🔄 Tried inverted bit patterns',
                        '_retry_with_phase_correction': '📐 Corrected phase alignment',
                        '_retry_with_different_symbol_rates': '📊 Tested alternative baud rates',
                        '_retry_with_adaptive_filtering': '🔧 Applied noise filtering',
                        '_retry_with_frequency_offset_correction': '📡 Corrected Doppler shift',
                        '_retry_with_bit_flip_correction': '🩹 Fixed single-bit errors'
                    }

                    for strategy, count in strategy_df.itertuples(index=False):
                        help_text = strategy_help.get(strategy, '❓ Unknown strategy')
                        st.write(f"**{count}x** {help_text}")
                else:
                    st.info("No reprocessing attempts yet")

            with tab3:
                st.markdown("### 🔍 Discovered Protocols")

                discovered = db.get_discovered_protocols(min_confidence=0.5)

                if discovered:
                    st.success(f"🎉 {len(discovered)} protocols discovered through machine learning!")

                    for proto in discovered:
                        with st.expander(f"📡 {proto['name']} (Confidence: {proto['confidence']:.0%})"):
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.write(f"**Modulation:** {proto['modulation']}")
                                st.write(f"**Symbol Rate:** {proto['symbol_rate']} baud")

                            with col2:
                                st.write(f"**Packet Length:** {proto['packet_length']} bytes")
                                st.write(f"**Sample Count:** {proto['sample_count']}")

                            with col3:
                                st.write(f"**Success Count:** {proto.get('success_count', 0)}")
                                discovered_date = fmt_ts(proto['discovered_at'])
                                st.write(f"**Discovered:** {discovered_date}")

                            if proto.get('metadata'):
                                st.json(proto['metadata'])
                else:
                    st.info("No protocols discovered yet. The system learns from repeated unknown patterns.")

            with tab4:
                st.markdown("### ⚠️ What Could Be Better")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**🔴 Exhausted Retries**")
                    exhausted = decoder_stats.get('exhausted_retries', 0)

                    if exhausted > 0:
                        st.warning(f"{exhausted} signals failed all retry attempts")
                        st.caption("These may be from unknown protocols or severely corrupted")
                    else:
                        st.success("No signals have exhausted all retry attempts!")

                with col2:
                    st.markdown("**📊 Unknown Signal Analysis**")

                    # Get unknown signal patterns
                    conn = db._connect(read_only=True)
                    cur = conn.cursor()
                    cur.execute("""
                                SELECT modulation_type, COUNT(*) as count
                                FROM unknown_signals
                                WHERE decoded = 0
                                GROUP BY modulation_type
                                ORDER BY count DESC
                                """)

                    unknown_patterns = cur.fetchall()
                    conn.close()

                    if unknown_patterns:
                        st.write("**Unknown modulation types detected:**")
                        for mod_type, count in unknown_patterns:
                            st.write(f"• {mod_type}: {count} signals")
                        st.caption("System is analyzing these for pattern discovery")
                    else:
                        st.success("All signal types recognized!")

        st.markdown("---")

    except Exception as e:
        st.warning(f"Intelligence stats unavailable: {e}")

    # === SENSOR DATABASE ===
    st.subheader("📊 Sensor Database")

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        lookback_h = st.slider("Look back window (hours)", 1, 144, 24)

    with col2:
        filter_type = st.selectbox(
            "Filter by",
            ["All Sensors", "High Confidence Only", "Reprocessed Only", "Low Confidence"]
        )

    with col3:
        sort_by = st.selectbox("Sort by", ["Signals", "Last Seen", "Confidence"])

    since = time.time() - lookback_h * 3600

    # Build query based on filters
    where_clause = "WHERE timestamp >= ?"
    params = [since]

    if filter_type == "High Confidence Only":
        where_clause += " AND confidence >= 0.8"
    elif filter_type == "Reprocessed Only":
        where_clause += " AND reprocessed = 1"
    elif filter_type == "Low Confidence":
        where_clause += " AND confidence < 0.7"

    order_clause = {
        "Signals": "signals DESC",
        "Last Seen": "last_ts DESC",
        "Confidence": "avg_confidence DESC"
    }[sort_by]

    # Query sensors
    conn = db._connect(read_only=True)
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT tpms_id,
               COUNT(*)             AS signals,
               MIN(timestamp)       AS first_ts,
               MAX(timestamp)       AS last_ts,
               AVG(signal_strength) AS avg_rssi,
               AVG(pressure_psi)    AS avg_pressure,
               AVG(temperature_c)   AS avg_temp,
               AVG(frequency)       AS avg_freq,
               AVG(confidence)      AS avg_confidence,
               SUM(reprocessed)     AS reprocessed_count,
               MAX(protocol)        AS protocol
        FROM tpms_signals
        {where_clause}
        GROUP BY tpms_id
        ORDER BY {order_clause}
        LIMIT 1000
        """,
        params,
    )
    rows = cur.fetchall()
    conn.close()

    df = pd.DataFrame([dict(zip([
        'tpms_id', 'signals', 'first_ts', 'last_ts', 'avg_rssi',
        'avg_pressure', 'avg_temp', 'avg_freq', 'avg_confidence',
        'reprocessed_count', 'protocol'
    ], r)) for r in rows])

    if df.empty:
        st.write("No sensors found matching the current filters.")
        return

    df["first_seen"] = df["first_ts"].apply(fmt_ts)
    df["last_seen"] = df["last_ts"].apply(fmt_ts)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Sensors", len(df))

    with col2:
        high_conf = len(df[df['avg_confidence'] >= 0.8])
        st.metric("High Confidence", high_conf,
                  delta=f"{high_conf / len(df) * 100:.0f}%" if len(df) > 0 else "0%")

    with col3:
        reprocessed = df['reprocessed_count'].sum()
        st.metric("Recovered Signals", int(reprocessed))

    with col4:
        avg_conf = df['avg_confidence'].mean()
        st.metric("Avg Confidence", f"{avg_conf:.2f}")

    # Display sensors with enhanced information
    st.write(f"**Displaying {len(df)} sensors**")

    for idx, row in df.iterrows():
        with st.container():
            # Determine confidence color
            conf = row['avg_confidence']
            if conf >= 0.8:
                conf_badge = "🟢"
                conf_text = "High"
            elif conf >= 0.6:
                conf_badge = "🟡"
                conf_text = "Medium"
            else:
                conf_badge = "🔴"
                conf_text = "Low"

            # Check if reprocessed
            reprocessed_badge = " 🔧" if row['reprocessed_count'] > 0 else ""

            col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 1, 1, 1, 2])

            with col1:
                if st.button(
                        f"🔍 {row['tpms_id']}{reprocessed_badge}",
                        key=f"sensor_db_{idx}",
                        help=f"Confidence: {conf:.2f} | Click for details"
                ):
                    st.session_state.previous_page = 'Sensor Database'
                    st.session_state.current_page = 'Sensor Detail'
                    st.session_state.selected_sensor = row['tpms_id']
                    st.rerun()

                st.caption(f"{conf_badge} {conf_text} confidence | {row['protocol']}")

            with col2:
                st.metric("Signals", row['signals'])
                if row['reprocessed_count'] > 0:
                    st.caption(f"🔧 {int(row['reprocessed_count'])} recovered")

            with col3:
                rssi_val = row['avg_rssi']
                if rssi_val:
                    rssi_color = "🟢" if rssi_val > -15 else "🟡" if rssi_val > -20 else "🔴"
                    st.metric("Avg RSSI", f"{rssi_val:.1f} dB")
                    st.caption(f"{rssi_color} Signal quality")
                else:
                    st.metric("Avg RSSI", "N/A")

            with col4:
                pressure_val = row['avg_pressure']
                if pressure_val:
                    # Check if pressure is in normal range
                    if 28 <= pressure_val <= 40:
                        pressure_icon = "🟢"
                    elif 25 <= pressure_val <= 45:
                        pressure_icon = "🟡"
                    else:
                        pressure_icon = "🔴"
                    st.metric("Avg Pressure", f"{pressure_val:.1f} PSI")
                    st.caption(f"{pressure_icon}")
                else:
                    st.metric("Avg Pressure", "N/A")

            with col5:
                temp_val = row['avg_temp']
                if temp_val:
                    st.metric("Avg Temp", f"{temp_val:.1f}°C")
                else:
                    st.metric("Avg Temp", "N/A")

            with col6:
                st.caption(f"**First:** {row['first_seen']}")
                st.caption(f"**Last:** {row['last_seen']}")

            if idx < len(df) - 1:
                st.markdown("---")

    # Export options
    st.markdown("---")

    col1, col2 = st.columns([3, 1])

    with col1:
        st.caption(f"💡 **Tip:** Sensors marked with 🔧 were recovered through intelligent reprocessing")

    with col2:
        if st.button("📥 Export to CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                "Download",
                data=csv.encode('utf-8'),
                file_name=f"sensor_database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )


def page_find_my_car(db_path: str):
    st.header("🔍 Find My Car (Capture Frequency Analysis)")

    st.info("""
    **How this works:**
    - Your car's 4 sensors transmit every 30-60 seconds while driving
    - On a 30-minute drive, each sensor should appear 30-60 times
    - Random passing cars appear 1-3 times
    - This tool filters for sensors with HIGH capture counts
    """)

    db = get_db(db_path)

    # Time window selector
    col1, col2 = st.columns(2)
    with col1:
        lookback_hours = st.slider("Analysis window (hours)", 1, 72, 24)
    with col2:
        min_captures = st.slider("Minimum captures", 5, 50, 10)

    since = time.time() - (lookback_hours * 3600)

    # Query for repeated sensors
    conn = db._connect(read_only=True)
    query = """
            SELECT tpms_id, \
                   COUNT(*)                          as captures, \
                   MIN(timestamp)                    as first_ts, \
                   MAX(timestamp)                    as last_ts, \
                   (MAX(timestamp) - MIN(timestamp)) as time_span_sec, \
                   AVG(pressure_psi)                 as avg_pressure, \
                   AVG(temperature_c)                as avg_temp, \
                   GROUP_CONCAT(timestamp)           as all_timestamps
            FROM tpms_signals
            WHERE timestamp >= ?
            GROUP BY tpms_id
            HAVING captures >= ?
            ORDER BY captures DESC \
            """

    cur = conn.cursor()
    cur.execute(query, (since, min_captures))
    rows = cur.fetchall()
    conn.close()

    if not rows:
        st.warning(f"No sensors found with {min_captures}+ captures in the last {lookback_hours} hours.")
        st.info("Try: 1) Driving around more, 2) Lowering 'Minimum captures', or 3) Increasing the time window")
        return

    # Build analysis dataframe
    candidates = []
    for row in rows:
        tpms_id, captures, first, last, span, avg_p, avg_t, ts_str = row

        # Parse timestamps
        timestamps = [float(t) for t in ts_str.split(',')]

        # Calculate intervals
        if len(timestamps) > 1:
            timestamps.sort()
            intervals = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
            avg_interval = np.mean(intervals)
            std_interval = np.std(intervals)
        else:
            avg_interval = 0
            std_interval = 0

        # Classify likelihood
        if captures >= 20 and 20 <= avg_interval <= 90:
            likelihood = "🟢 Very Likely YOUR CAR"
        elif captures >= 10 and avg_interval <= 120:
            likelihood = "🟡 Possibly Your Car"
        elif captures >= 5:
            likelihood = "🔵 Nearby Parked Car"
        else:
            likelihood = "⚪ Random"

        candidates.append({
            'tpms_id': tpms_id,
            'captures': captures,
            'likelihood': likelihood,
            'time_span_min': span / 60,
            'avg_interval_sec': avg_interval,
            'interval_std': std_interval,
            'avg_pressure': avg_p,
            'avg_temp': avg_t,
            'first_seen': fmt_ts(first),
            'last_seen': fmt_ts(last)
        })

    df = pd.DataFrame(candidates)

    # Summary
    st.subheader(f"📊 Found {len(df)} Sensors with {min_captures}+ Captures")

    col1, col2, col3 = st.columns(3)
    very_likely = len(df[df['likelihood'].str.contains('Very Likely')])
    possibly = len(df[df['likelihood'].str.contains('Possibly')])
    col1.metric("Very Likely Your Car", very_likely)
    col2.metric("Possibly Your Car", possibly)
    col3.metric("Total Candidates", len(df))

    # Display table
    st.dataframe(df, width="stretch", height=400)

    # Auto-select top candidates
    top_candidates = df[df['captures'] >= 15].head(6)

    if len(top_candidates) >= 3:
        st.success(f"✅ Found {len(top_candidates)} sensors that are very likely YOUR CAR!")

        st.subheader("🚗 Suggested Vehicle Sensors")
        st.dataframe(top_candidates[['tpms_id', 'captures', 'avg_interval_sec', 'avg_pressure']], width="stretch")

        # Manual selection
        st.markdown("---")
        st.subheader("Create Vehicle from Selected Sensors")

        selected = st.multiselect(
            "Confirm or adjust sensor selection",
            options=df['tpms_id'].tolist(),
            default=top_candidates['tpms_id'].tolist()[:4]  # Default to top 4
        )

        if len(selected) >= 3:
            vehicle_name = st.text_input("Vehicle nickname", placeholder="My Car")

            if st.button("✅ Create Vehicle", type="primary"):
                vehicle_id = db.upsert_vehicle(selected, time.time())
                if vehicle_name:
                    db.update_vehicle_nickname(vehicle_id, vehicle_name)

                st.success(f"🎉 Created vehicle '{vehicle_name or 'Vehicle'}' with {len(selected)} sensors!")
                st.balloons()

                # Show what was created
                st.json({
                    "vehicle_id": vehicle_id,
                    "sensors": selected,
                    "name": vehicle_name
                })
    else:
        st.warning("Not enough high-frequency sensors detected. Try:")
        st.markdown("""
        1. **Drive around for 20-30 minutes** with the scanner running
        2. **Lower the 'Minimum captures' slider** to 5
        3. **Check if your scanner is actually running** (look for new signals in Live Detection)
        """)

    # Export for analysis
    st.markdown("---")
    if st.button("📥 Download Full Analysis CSV"):
        csv = df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            data=csv.encode('utf-8'),
            file_name=f"tpms_candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


def page_analytics(db_path: str):
    st.header("📈 Analytics")
    db = get_db(db_path)

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        lookback_h = st.selectbox("Window", [1, 6, 12, 24, 48, 72, 300], index=3)
    with col2:
        bucket = st.selectbox("Bucket", ["1 min", "5 min", "15 min", "1 hour"], index=1)
    with col3:
        st.caption("All charts are computed from a time-windowed subset.")

    now = time.time()
    since = now - int(lookback_h) * 3600

    bucket_s = {"1 min": 60, "5 min": 300, "15 min": 900, "1 hour": 3600}[bucket]

    # Pull just timestamps (fast) and bucket in pandas
    conn = db._connect(read_only=True)
    cur = conn.cursor()
    # Pull data for general charts
    cur.execute("SELECT timestamp FROM tpms_signals WHERE timestamp >= ? ORDER BY timestamp ASC", (since,))
    ts = [float(r[0]) for r in cur.fetchall()]
    conn.close()

    if not ts:
        st.write("No signals in that window.")
        return

    # --- Activity Chart (Time Zone Fixed) ---
    # Create index in UTC, then convert to User Timezone (Denver/Mountain)
    s = pd.Series(1, index=pd.to_datetime(ts, unit="s").tz_localize("UTC").tz_convert(PYTZ_TIMEZONE))
    series = s.resample(f"{bucket_s}s").sum()
    st.line_chart(series)

    st.markdown("---")

    # --- All Sensors Geography & Timeline ---
    st.subheader("📍 Sensor Analysis")
    st.caption("Displaying all sensors detected in the selected window.")

    # Get detailed data for analysis
    conn = db._connect(read_only=True)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT tpms_id, timestamp, latitude, longitude, pressure_psi
        FROM tpms_signals
        WHERE timestamp >= ?
        """,
        (since,)
    )
    all_signals = cur.fetchall()
    conn.close()

    if all_signals:
        sig_df = pd.DataFrame(all_signals, columns=["tpms_id", "timestamp", "lat", "lon", "pressure"])

        # Calculate statistics
        counts = sig_df["tpms_id"].value_counts()
        repeated_ids = counts[counts > 1].index.tolist()

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Unique Sensors", len(counts))
        c2.metric("Repeated Sensors", len(repeated_ids))
        c3.metric("Total Signals", len(sig_df))

        # Work with ALL sensors, not just repeats
        sig_df["Local Time"] = pd.to_datetime(sig_df["timestamp"], unit="s").dt.tz_localize(
            "UTC").dt.tz_convert(PYTZ_TIMEZONE)

        # 1. Timeline Chart (All Sensors)
        st.markdown("#### ⏳ Sighting Timeline (All Sensors)")
        st.scatter_chart(
            sig_df,
            x="Local Time",
            y="tpms_id",
            color="tpms_id",
            height=400,
            width="stretch"
        )

        # 2. Geography Map (All Sensors)
        st.markdown("#### 🗺️ Sighting Geography (All Sensors)")
        # Filter for valid lat/lon
        map_df = sig_df.dropna(subset=["lat", "lon"])
        if not map_df.empty:
            # Rename for st.map
            map_df = map_df.rename(columns={"lat": "latitude", "lon": "longitude"})
            st.map(map_df, size=20, color="#FF4B4B")
        else:
            st.info("No GPS data found for sensors in this window.")

        # 3. List with filter option
        with st.expander("View Sensor Data"):
            show_filter = st.radio("Show:", ["All Sensors", "Repeated Only"], horizontal=True)
            if show_filter == "Repeated Only" and repeated_ids:
                display_df = sig_df[sig_df["tpms_id"].isin(repeated_ids)]
            else:
                display_df = sig_df

            st.dataframe(display_df[["Local Time", "tpms_id", "pressure", "lat", "lon"]], width="stretch")

    st.markdown("---")

    # --- Top Sensors ---
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
    st.dataframe(top_df, width="stretch", height=360)

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

    st.subheader("🔍 Debug: Most Repeated Sensors")

    query = """
            SELECT tpms_id,
                   COUNT(*)          as count,
                   AVG(pressure_psi) as avg_pressure,
                   MIN(timestamp)    as first,
                   MAX(timestamp)    as last
            FROM tpms_signals
            WHERE timestamp >= ?
            GROUP BY tpms_id
            ORDER BY count DESC
            LIMIT 20
            """

    since = time.time() - 3600  # Last hour
    conn = db._connect(read_only=True)
    df = pd.read_sql_query(query, conn, params=(since,))
    conn.close()

    st.dataframe(df)


def page_maintenance(db_path: str):
    st.header("🛠 Maintenance")

    db = get_db(db_path)

    st.caption(
        "Maintenance analysis can get expensive if you run it over the whole DB. "
        "This page defaults to a time window and computes alerts on-demand."
    )

    lookback_days = st.slider("Analyze lookback (days)", 1, 360, 7)
    min_vehicle_enc = st.slider("Vehicles: min encounters", 0, 20, 2)

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
            st.dataframe(adf, width="stretch", height=420)


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
        f"Time: **{fmt_ts(ts)}** |  Pressure: **{p} PSI** |  Temp: **{t} °C** "
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
            st.dataframe(prof, width="stretch")


def page_sensor_trigger(db_path: str):
    st.header("🧲 Sensor Trigger")

    st.info("Placeholder UI for your ESP32 / GPIO / SDR trigger controls.")
    st.write(
        "If your project provides esp32_trigger_controller, wire it up here with lazy imports (same pattern as Live Detection).")


def page_gps_setup(db_path: str):
    st.header("🛰️ GPS Setup")

    st.info("Configure GPS to add location data to TPMS signals")

    # Initialize GPS manager in session state
    if 'gps_manager' not in st.session_state:
        from GPS_manager import GPSManager
        st.session_state.gps_manager = GPSManager()

    gps = st.session_state.gps_manager

    # Detect available sources
    with st.expander("🔍 Detect GPS Sources", expanded=True):
        if st.button("Scan for GPS"):
            sources = gps.detect_gps_sources()

            st.write("**Available GPS Sources:**")
            for source, available in sources.items():
                icon = "✅" if available else "❌"
                st.write(f"{icon} {source}")

    # GPS source selection
    st.subheader("Select GPS Source")

    gps_type = st.radio(
        "GPS Type",
        options=["GPSD (Recommended)", "Android Bluetooth", "USB GPS", "Mock GPS (Testing)"],
        horizontal=True
    )

    if gps_type == "GPSD (Recommended)":
        st.write("**GPSD** - Universal GPS daemon (works with most GPS devices)")

        col1, col2 = st.columns(2)
        with col1:
            gpsd_host = st.text_input("GPSD Host", value="localhost")
        with col2:
            gpsd_port = st.number_input("GPSD Port", value=2947, min_value=1, max_value=65535)

        st.caption("Install GPSD: `sudo apt install gpsd gpsd-clients`")

        if st.button("Connect to GPSD"):
            if gps.start_gpsd(host=gpsd_host, port=gpsd_port):
                st.success("✅ Connected to GPSD!")
                st.rerun()
            else:
                st.error("❌ Failed to connect to GPSD")

    elif gps_type == "Android Bluetooth":
        st.write("**Android Bluetooth GPS** - Use your phone's GPS")

        st.info(
            "**Setup Instructions:**\n\n"
            "1. Install 'Bluetooth GPS Provider' app on your Android phone\n"
            "2. Pair your phone with your laptop via Bluetooth\n"
            "3. Start GPS sharing in the app\n"
            "4. Note the Bluetooth device address"
        )

        device_addr = st.text_input("Bluetooth Device Address", placeholder="XX:XX:XX:XX:XX:XX")

        if st.button("Connect to Android"):
            if device_addr:
                if gps.start_android_bluetooth(device_addr):
                    st.success("✅ Connected to Android GPS!")
                    st.rerun()
                else:
                    st.error("❌ Failed to connect")
            else:
                st.warning("Please enter device address")

    elif gps_type == "USB GPS":
        st.write("**USB GPS Dongle**")

        device_path = st.text_input("Device Path", value="/dev/ttyACM0")
        st.caption("Common paths: /dev/ttyUSB0, /dev/ttyACM0")

        if st.button("Connect to USB GPS"):
            if gps.start_usb_gps(device_path):
                st.success("✅ Connected to USB GPS!")
                st.rerun()
            else:
                st.error("❌ Failed to connect")

    elif gps_type == "Mock GPS (Testing)":
        st.write("**Mock GPS** - Simulated location for testing")

        col1, col2 = st.columns(2)
        with col1:
            mock_lat = st.number_input("Latitude", value=39.576324, format="%.6f")
        with col2:
            mock_lon = st.number_input("Longitude", value=-104.866249, format="%.6f")

        if st.button("Start Mock GPS"):
            if gps.start_mock_gps(lat=mock_lat, lon=mock_lon):
                st.success("✅ Mock GPS started!")
                st.rerun()
            else:
                st.error("❌ Failed to start mock GPS")

    # GPS Status
    st.markdown("---")
    st.subheader("📍 GPS Status")

    status = gps.get_status()

    if status['active']:
        col1, col2, col3 = st.columns(3)

        with col1:
            if status['has_fix']:
                st.success("🟢 GPS Active (Fix)")
            else:
                st.warning("🟡 GPS Active (No Fix)")

        with col2:
            st.metric("Source", status['source'])

        with col3:
            if st.button("Stop GPS"):
                gps.stop()
                st.info("GPS stopped")
                st.rerun()

        if status['has_fix']:
            st.write(f"**Location:** {status['latitude']:.6f}, {status['longitude']:.6f}")
            st.write(f"**Satellites:** {status['satellites']}")
            st.write(f"**Altitude:** {status.get('altitude', 'N/A')} m")
            st.write(f"**Data age:** {status['age_seconds']:.1f} seconds")

            # Show on map
            import pandas as pd
            map_df = pd.DataFrame({
                'lat': [status['latitude']],
                'lon': [status['longitude']]
            })
            st.map(map_df, zoom=15)
    else:
        st.info("⚪ GPS not active")


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

    # Display vehicle information with clickable sensor IDs
    for idx, vehicle in enumerate(vehicles):
        with st.expander(
                f"🚗 Vehicle {vehicle.get('nickname') or vehicle.get('vehicle_hash', 'Unknown')} "
                f"(ID: {vehicle['id']}) - {vehicle.get('encounter_count', 0)} encounters",
                expanded=idx < 3  # Expand first 3
        ):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write(f"**First Seen:** {fmt_ts(vehicle.get('first_seen', 0))}")
            with col2:
                st.write(f"**Last Seen:** {fmt_ts(vehicle.get('last_seen', 0))}")
            with col3:
                st.write(f"**Encounters:** {vehicle.get('encounter_count', 0)}")

            st.markdown("**Sensors:**")

            # Display sensors as clickable buttons
            sensor_ids = vehicle.get('tpms_ids', [])
            if sensor_ids:
                cols = st.columns(min(len(sensor_ids), 4))  # Max 4 columns
                for i, sensor_id in enumerate(sensor_ids):
                    with cols[i % 4]:
                        if st.button(f"🔍 {sensor_id}", key=f"vehicle_{vehicle['id']}_sensor_{i}"):
                            st.session_state.previous_page = 'Vehicle Database'
                            st.session_state.current_page = 'Sensor Detail'
                            st.session_state.selected_sensor = sensor_id
                            st.rerun()
            else:
                st.write("No sensors associated")

            # Show metadata if available
            if vehicle.get('metadata'):
                with st.expander("View Metadata"):
                    st.json(vehicle['metadata'])


def page_sensor_detail(db_path: str, tpms_id: str):
    """Comprehensive sensor detail page showing all history and associations"""
    st.header(f"🔍 Sensor Detail: {tpms_id}")

    db = get_db(db_path)

    # Back button
    if st.button("← Back"):
        if 'previous_page' in st.session_state:
            st.session_state.current_page = st.session_state.previous_page
            st.rerun()

    st.markdown("---")

    # === OVERVIEW SECTION ===
    st.subheader("📊 Overview")

    stats = db.get_sensor_statistics(tpms_id)

    if not stats or stats['total_signals'] == 0:
        st.warning(f"No data found for sensor {tpms_id}")
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Signals", stats['total_signals'])
    with col2:
        st.metric("First Seen", fmt_ts(stats['first_seen']))
    with col3:
        st.metric("Last Seen", fmt_ts(stats['last_seen']))
    with col4:
        time_span_days = (stats['last_seen'] - stats['first_seen']) / 86400
        st.metric("Active Period", f"{time_span_days:.1f} days")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if stats['avg_pressure']:
            st.metric("Avg Pressure", f"{stats['avg_pressure']:.1f} PSI")
        else:
            st.metric("Avg Pressure", "N/A")

    with col2:
        if stats['avg_temp']:
            st.metric("Avg Temperature", f"{stats['avg_temp']:.1f}°C")
        else:
            st.metric("Avg Temperature", "N/A")

    with col3:
        if stats['avg_rssi']:
            st.metric("Avg Signal", f"{stats['avg_rssi']:.1f} dB")
        else:
            st.metric("Avg Signal", "N/A")

    with col4:
        st.metric("Protocol", stats.get('protocol', 'Unknown'))

    st.markdown("---")

    # === VEHICLE ASSOCIATIONS ===
    st.subheader("🚗 Vehicle Associations")

    # Find which vehicles this sensor belongs to
    vehicles = db.get_all_vehicles(min_encounters=1)
    associated_vehicles = []

    for v in vehicles:
        if tpms_id in v.get('tpms_ids', []):
            associated_vehicles.append(v)

    if associated_vehicles:
        for v in associated_vehicles:
            with st.expander(f"Vehicle: {v.get('nickname') or v.get('vehicle_hash', 'Unknown')} (ID: {v['id']})",
                             expanded=True):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write(f"**Encounters:** {v.get('encounter_count', 0)}")
                with col2:
                    st.write(f"**First Seen:** {fmt_ts(v.get('first_seen', 0))}")
                with col3:
                    st.write(f"**Last Seen:** {fmt_ts(v.get('last_seen', 0))}")

                st.write(f"**All Sensors:** {', '.join(v.get('tpms_ids', []))}")
    else:
        st.info("This sensor is not associated with any vehicle yet.")

        # Suggest potential associations
        st.write("**Potential Vehicle Associations:**")

        # Find sensors that appear together with this one
        conn = db._connect(read_only=True)
        cur = conn.cursor()

        # Find co-occurring sensors (within 60 seconds)
        cur.execute("""
                    WITH sensor_times AS (SELECT timestamp
                                          FROM tpms_signals
                                          WHERE tpms_id = ?)
                    SELECT DISTINCT s.tpms_id, COUNT(*) as co_occurrences
                    FROM tpms_signals s
                             JOIN sensor_times st ON ABS(s.timestamp - st.timestamp) < 60
                    WHERE s.tpms_id != ?
                    GROUP BY s.tpms_id
                    ORDER BY co_occurrences DESC
                    LIMIT 10
                    """, (tpms_id, tpms_id))

        co_occurring = cur.fetchall()
        conn.close()

        if co_occurring:
            st.write("Sensors frequently seen together:")
            for sensor, count in co_occurring:
                st.write(f"- `{sensor}`: {count} co-occurrences")
        else:
            st.write("No frequently co-occurring sensors found.")

    st.markdown("---")

    # === LOCATION HISTORY ===
    st.subheader("🗺️ Location History")

    # Get all signals with GPS data
    conn = db._connect(read_only=True)
    cur = conn.cursor()
    cur.execute("""
                SELECT timestamp, latitude, longitude, pressure_psi, temperature_c
                FROM tpms_signals
                WHERE tpms_id = ?
                  AND latitude IS NOT NULL
                  AND longitude IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT 1000
                """, (tpms_id,))

    gps_data = cur.fetchall()
    conn.close()

    if gps_data:
        map_df = pd.DataFrame(gps_data, columns=['timestamp', 'latitude', 'longitude', 'pressure', 'temperature'])

        col1, col2 = st.columns([2, 1])

        with col1:
            st.write(f"**{len(map_df)} signals with GPS data**")
            st.map(map_df[['latitude', 'longitude']], size=15, color="#FF4B4B")

        with col2:
            st.write("**Location Statistics:**")
            st.write(f"Lat range: {map_df['latitude'].min():.6f} to {map_df['latitude'].max():.6f}")
            st.write(f"Lon range: {map_df['longitude'].min():.6f} to {map_df['longitude'].max():.6f}")

            # Calculate approximate travel distance (rough estimate)
            if len(map_df) > 1:
                from math import radians, cos, sin, asin, sqrt

                def haversine(lon1, lat1, lon2, lat2):
                    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
                    dlon = lon2 - lon1
                    dlat = lat2 - lat1
                    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
                    c = 2 * asin(sqrt(a))
                    km = 6371 * c
                    return km

                total_distance = 0
                for i in range(len(map_df) - 1):
                    d = haversine(
                        map_df.iloc[i]['longitude'], map_df.iloc[i]['latitude'],
                        map_df.iloc[i + 1]['longitude'], map_df.iloc[i + 1]['latitude']
                    )
                    total_distance += d

                st.write(f"**Approx. distance:** {total_distance:.1f} km")
    else:
        st.info("No GPS data available for this sensor.")

    st.markdown("---")

    # === SIGNAL HISTORY TIMELINE ===
    st.subheader("📈 Signal History")

    # Time range selector
    col1, col2 = st.columns(2)
    with col1:
        lookback_days = st.selectbox("Time Range", [1, 7, 30, 90, 365, "All Time"], index=2)
    with col2:
        chart_type = st.selectbox("Chart Type", ["Pressure", "Temperature", "Signal Strength", "All"])

    # Get historical data
    if lookback_days == "All Time":
        since = 0
    else:
        since = time.time() - (lookback_days * 86400)

    history_df = db.get_sensor_history(tpms_id)
    history_df = history_df[history_df['timestamp'] >= since]

    if not history_df.empty:
        # Convert timestamp to datetime
        history_df['datetime'] = pd.to_datetime(history_df['timestamp'], unit='s').dt.tz_localize('UTC').dt.tz_convert(
            PYTZ_TIMEZONE)
        history_df = history_df.set_index('datetime')

        # Plot based on selection
        if chart_type == "Pressure" and 'pressure_psi' in history_df.columns:
            st.line_chart(history_df['pressure_psi'].dropna())
        elif chart_type == "Temperature" and 'temperature_c' in history_df.columns:
            st.line_chart(history_df['temperature_c'].dropna())
        elif chart_type == "Signal Strength" and 'signal_strength' in history_df.columns:
            st.line_chart(history_df['signal_strength'].dropna())
        elif chart_type == "All":
            chart_data = history_df[['pressure_psi', 'temperature_c', 'signal_strength']].dropna()
            if not chart_data.empty:
                st.line_chart(chart_data)

        # Statistics table
        st.write("**Statistics for selected period:**")

        stats_data = []
        if 'pressure_psi' in history_df.columns:
            p = history_df['pressure_psi'].dropna()
            if len(p) > 0:
                stats_data.append({
                    'Metric': 'Pressure (PSI)',
                    'Min': f"{p.min():.1f}",
                    'Max': f"{p.max():.1f}",
                    'Avg': f"{p.mean():.1f}",
                    'Std Dev': f"{p.std():.2f}"
                })

        if 'temperature_c' in history_df.columns:
            t = history_df['temperature_c'].dropna()
            if len(t) > 0:
                stats_data.append({
                    'Metric': 'Temperature (°C)',
                    'Min': f"{t.min():.1f}",
                    'Max': f"{t.max():.1f}",
                    'Avg': f"{t.mean():.1f}",
                    'Std Dev': f"{t.std():.2f}"
                })

        if 'signal_strength' in history_df.columns:
            s = history_df['signal_strength'].dropna()
            if len(s) > 0:
                stats_data.append({
                    'Metric': 'Signal Strength (dB)',
                    'Min': f"{s.min():.1f}",
                    'Max': f"{s.max():.1f}",
                    'Avg': f"{s.mean():.1f}",
                    'Std Dev': f"{s.std():.2f}"
                })

        if stats_data:
            st.table(pd.DataFrame(stats_data))
    else:
        st.info("No data available for the selected time range.")

    st.markdown("---")

    # === RECENT READINGS ===
    st.subheader("📋 Recent Readings")

    recent_limit = st.slider("Number of readings to show", 10, 100, 25)

    recent_df = history_df.head(recent_limit).reset_index()

    if not recent_df.empty:
        display_cols = ['datetime', 'pressure_psi', 'temperature_c', 'signal_strength', 'snr',
                        'battery_low', 'latitude', 'longitude']
        display_cols = [c for c in display_cols if c in recent_df.columns]

        st.dataframe(
            recent_df[display_cols],
            width="stretch",
            height=400
        )

        # Download button
        csv = recent_df.to_csv(index=False)
        st.download_button(
            "📥 Download Full History (CSV)",
            data=csv.encode('utf-8'),
            file_name=f"sensor_{tpms_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    st.markdown("---")

    # === ANOMALY DETECTION ===
    st.subheader("⚠️ Anomaly Detection")

    ensure_models(db_path)

    # Get latest reading
    if not history_df.empty:
        latest = history_df.iloc[0]

        # Check for anomalies using ML engine
        learner = st.session_state.online_learner

        anomaly = learner.anomaly_scores(
            tpms_id,
            safe_float(latest.get('pressure_psi')),
            safe_float(latest.get('temperature_c'))
        )

        if anomaly.get('ok'):
            col1, col2 = st.columns(2)

            with col1:
                z_p = anomaly.get('z_pressure')
                if z_p is not None:
                    if abs(z_p) > 3:
                        st.error(f"🔴 Pressure Anomaly Detected! Z-score: {z_p:+.2f}")
                    elif abs(z_p) > 2:
                        st.warning(f"🟡 Pressure Unusual: Z-score: {z_p:+.2f}")
                    else:
                        st.success(f"🟢 Pressure Normal: Z-score: {z_p:+.2f}")
                else:
                    st.info("Pressure: No baseline yet")

            with col2:
                z_t = anomaly.get('z_temp')
                if z_t is not None:
                    if abs(z_t) > 3:
                        st.error(f"🔴 Temperature Anomaly Detected! Z-score: {z_t:+.2f}")
                    elif abs(z_t) > 2:
                        st.warning(f"🟡 Temperature Unusual: Z-score: {z_t:+.2f}")
                    else:
                        st.success(f"🟢 Temperature Normal: Z-score: {z_t:+.2f}")
                else:
                    st.info("Temperature: No baseline yet")
        else:
            st.info("Not enough data for anomaly detection yet.")


# -----------------------------
# Main
# -----------------------------

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    # Header
    st.markdown(f"# 🚗 {APP_TITLE}")
    st.caption(f"DB: {DB_PATH}")

    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Live Detection'
    if 'selected_sensor' not in st.session_state:
        st.session_state.selected_sensor = None

    # Sidebar
    sidebar_control_panel(DB_PATH)

    # Check if we should show sensor detail page
    if st.session_state.current_page == 'Sensor Detail' and st.session_state.selected_sensor:
        page_sensor_detail(DB_PATH, st.session_state.selected_sensor)
        return

    # Page selection (segmented control)
    pages = [
        ("Live Detection", page_live_detection),
        ("GPS Setup", page_gps_setup),
        ("Find My Car", page_find_my_car),
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

    # Update current page from radio selection
    choice = st.radio("Navigation", labels, horizontal=True, label_visibility="collapsed",
                      index=labels.index(
                          st.session_state.current_page) if st.session_state.current_page in labels else 0)

    if choice != st.session_state.current_page:
        st.session_state.current_page = choice
        st.rerun()

    page_fn = dict(pages)[choice]

    # Opportunistic lightweight online learning
    if st.session_state.get("auto_learn", True):
        try:
            process_online_learning(DB_PATH, limit=2000)
        except Exception:
            pass

    page_fn(DB_PATH)


if __name__ == "__main__":
    main()

