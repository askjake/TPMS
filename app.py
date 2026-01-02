import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import threading
import queue
import numpy as np
from collections import deque
from esp32_trigger_controller import ESP32TriggerController
import pytz

# Timezone configuration
MOUNTAIN_TZ = pytz.timezone('America/Denver')  # Mountain Time with DST support
# At top of app.py, after imports
_processing_thread = None
_should_process = False

def processing_loop():
    """Background thread to process samples"""
    global _should_process
    
    print(f"[{time.time():.3f}] Processing thread started", flush=True)
    
    while _should_process:
        try:
            # Add periodic heartbeat
            if not hasattr(processing_loop, 'count'):
                processing_loop.count = 0
            processing_loop.count += 1
            
            if processing_loop.count % 50 == 0:  # Every 5 seconds
                print(f"[{time.time():.3f}] Processing thread alive, count={processing_loop.count}", flush=True)
            
            process_signal_buffer()
            time.sleep(0.1)  # Process every 100ms
        except Exception as e:
            print(f"Processing thread error: {e}", flush=True)
            import traceback
            traceback.print_exc()


def start_processing():
    """Start background processing thread"""
    global _processing_thread, _should_process
    
    if _processing_thread and _processing_thread.is_alive():
        return  # Already running
    
    _should_process = True
    _processing_thread = threading.Thread(target=processing_loop, daemon=True)
    _processing_thread.start()
    print("✅ Started background processing thread", flush=True)

def stop_processing():
    """Stop background processing thread"""
    global _should_process
    _should_process = False
    print("⏹️ Stopped background processing thread", flush=True)

def format_timestamp(timestamp, format_str="%Y-%m-%d %H:%M:%S", include_timezone=False):
    """Convert Unix timestamp to Mountain Time formatted string"""
    if timestamp is None:
        return "Never"
    
    # Convert from UTC to Mountain Time
    utc_dt = datetime.fromtimestamp(timestamp, tz=pytz.UTC)
    mountain_dt = utc_dt.astimezone(MOUNTAIN_TZ)
    
    if include_timezone:
        return mountain_dt.strftime(format_str + " %Z")
    return mountain_dt.strftime(format_str)

def timestamp_to_mountain(timestamp):
    """Convert Unix timestamp to Mountain Time datetime object"""
    if timestamp is None:
        return None
    utc_dt = datetime.fromtimestamp(timestamp, tz=pytz.UTC)
    return utc_dt.astimezone(MOUNTAIN_TZ)

def pandas_timestamp_to_mountain(series):
    """Convert pandas timestamp series to Mountain Time"""
    return pd.to_datetime(series, unit='s', utc=True).dt.tz_convert(MOUNTAIN_TZ)

# Add to session state
if 'esp32_trigger' not in st.session_state:
    st.session_state.esp32_trigger = ESP32TriggerController("192.168.4.1")

# Update show_trigger_controls()
def show_trigger_controls():
    """ESP32 LF Trigger Controls"""
    st.header("📡 ESP32 LF Trigger Control")
    
    # Connection status
    if st.session_state.esp32_trigger.connected:
        st.success("✅ ESP32 Connected")
    else:
        st.error("❌ ESP32 Not Connected")
        if st.button("🔄 Reconnect"):
            st.session_state.esp32_trigger.check_connection()
            st.rerun()
    
    if not st.session_state.esp32_trigger.connected:
        st.info("Connect to WiFi network: **TPMS_Trigger** (password: tpms12345)")
        return
    
    # Rest of trigger controls...

# Import config and modules
from config import config
from database import TPMSDatabase
from hackrf_interface import HackRFInterface
from tpms_decoder import TPMSDecoder
from ml_engine import VehicleClusteringEngine
from esp32_trigger_controller import ESP32TriggerController
import pytz

# Timezone configuration
MOUNTAIN_TZ = pytz.timezone('America/Denver')  # Mountain Time with DST support

def format_timestamp(timestamp, format_str="%Y-%m-%d %H:%M:%S", include_timezone=False):
    """Convert Unix timestamp to Mountain Time formatted string"""
    if timestamp is None:
        return "Never"
    
    # Convert from UTC to Mountain Time
    utc_dt = datetime.fromtimestamp(timestamp, tz=pytz.UTC)
    mountain_dt = utc_dt.astimezone(MOUNTAIN_TZ)
    
    if include_timezone:
        return mountain_dt.strftime(format_str + " %Z")
    return mountain_dt.strftime(format_str)

def timestamp_to_mountain(timestamp):
    """Convert Unix timestamp to Mountain Time datetime object"""
    if timestamp is None:
        return None
    utc_dt = datetime.fromtimestamp(timestamp, tz=pytz.UTC)
    return utc_dt.astimezone(MOUNTAIN_TZ)

def pandas_timestamp_to_mountain(series):
    """Convert pandas timestamp series to Mountain Time"""
    return pd.to_datetime(series, unit='s', utc=True).dt.tz_convert(MOUNTAIN_TZ)

# Global queues for thread-safe communication
signal_queue = queue.Queue(maxsize=1000)
signal_history_queue = deque(maxlen=config.SIGNAL_HISTORY_SIZE)

# Page config
st.set_page_config(
    page_title="TPMS Tracker",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

def init_globals():
    """Initialize global references from session state"""
    global _decoder, _db, _ml_engine
    
    if 'decoder' in st.session_state:
        _decoder = st.session_state.decoder
        _db = st.session_state.db
        _ml_engine = st.session_state.ml_engine
        
        # Verify it worked
        #if _decoder is not None:
            #print(f"✅ Globals initialized: decoder={type(_decoder).__name__}", flush=True)
        #else:
            #print("⚠️  init_globals() called but decoder is still None!", flush=True)
    else:
        print("⚠️  init_globals() called but 'decoder' not in session_state yet!", flush=True)


# Initialize session state
if 'db' not in st.session_state:
    st.session_state.db = TPMSDatabase(config.DB_PATH)
    st.session_state.hackrf = HackRFInterface()
    st.session_state.decoder = TPMSDecoder(config.SAMPLE_RATE)
    st.session_state.ml_engine = VehicleClusteringEngine(st.session_state.db)
    st.session_state.is_scanning = False
    st.session_state.signal_buffer = []
    st.session_state.recent_detections = []
    
    # Initialize global references for background thread
    # init_globals()  # <-- REMOVE THIS LINE FROM HERE

if 'esp32_trigger' not in st.session_state:
    st.session_state.esp32_trigger = ESP32TriggerController("192.168.4.1")

# ADD THIS LINE HERE (outside the if block):
init_globals()  # <-- Always refresh global references on every script run


# Global buffer for thread-safe data passing
_sample_buffer = []
_buffer_lock = threading.Lock()
_decoder = None
_db = None
_ml_engine = None
_signal_buffer_list = []

def signal_callback(iq_samples, signal_strength, frequency):
    """Callback for processing HackRF samples"""
    global _sample_buffer
    
    try:
        if not hasattr(signal_callback, 'count'):
            signal_callback.count = 0
            signal_callback.last_log = time.time()
        
        signal_callback.count += 1
        
        # Only log every 100 callbacks or every 30 seconds
        now = time.time()
        if signal_callback.count % 100 == 0 or (now - signal_callback.last_log) > 30.0:
            #print(f"📊 Processed {signal_callback.count} samples | "
                  #f"Buffer: {len(_sample_buffer)}", flush=True)
            signal_callback.last_log = now
        
        with _buffer_lock:
            _sample_buffer.append({
                'iq_samples': iq_samples.copy(),
                'signal_strength': signal_strength,
                'frequency': frequency,
                'timestamp': time.time()
            })
            
            if len(_sample_buffer) > 100:
                _sample_buffer = _sample_buffer[-100:]
                
    except Exception as e:
        print(f"❌ Callback error: {e}", flush=True)


def process_signal_buffer():
    """Process signals from buffer"""
    global _sample_buffer, _decoder, _db, _ml_engine, _signal_buffer_list
    
    if not hasattr(process_signal_buffer, 'call_count'):
        process_signal_buffer.call_count = 0
    process_signal_buffer.call_count += 1
    
    # Only log every 500 calls
    if process_signal_buffer.call_count % 500 == 0:
        print(f"🔄 Processing loop: {process_signal_buffer.call_count} iterations", flush=True)
    
    if _decoder is None:
        return
    
    with _buffer_lock:
        if not _sample_buffer:
            return
        samples_to_process = _sample_buffer.copy()
        _sample_buffer.clear()
    
    # Process each sample (decoder will log successful decodes)
    for data in samples_to_process:
        try:
            signal_history_queue.append(data['signal_strength'])
            signals = _decoder.process_samples(data['iq_samples'], data['frequency'])
            
            for signal in signals:
                signal.signal_strength = data['signal_strength']
                
                signal_dict = {
                    'tpms_id': signal.tpms_id,
                    'timestamp': signal.timestamp,
                    'frequency': signal.frequency,
                    'signal_strength': signal.signal_strength,
                    'snr': signal.snr,
                    'pressure_psi': signal.pressure_psi,
                    'temperature_c': signal.temperature_c,
                    'battery_low': signal.battery_low,
                    'protocol': signal.protocol,
                    'raw_data': signal.raw_data
                }
                
                _db.insert_signal(signal_dict)
                _signal_buffer_list.append(signal_dict)
            
            # Vehicle clustering
            if len(_signal_buffer_list) > 10:
                vehicle_ids = _ml_engine.process_signals(_signal_buffer_list)
                for vehicle_id in vehicle_ids:
                    vehicle_info = _db.get_vehicle_history(vehicle_id)
                    print(f"🚗 Vehicle {vehicle_id} detected", flush=True)
                _signal_buffer_list = []
                
        except Exception as e:
            print(f"❌ Processing error: {e}", flush=True)


def show_signal_histogram():
    """Display real-time signal strength histogram"""
    st.subheader("📊 Signal Strength Distribution")
    
    if len(signal_history_queue) < 10:
        st.info("Collecting signal data...")
        return
    
    # Convert to numpy array
    signal_data = np.array(list(signal_history_queue))
    
    # Create histogram
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=signal_data,
        nbinsx=config.HISTOGRAM_BINS,
        name='Signal Strength',
        marker_color='lightblue'
    ))
    
    # Add threshold line
    fig.add_vline(
        x=config.SIGNAL_THRESHOLD,
        line_dash="dash",
        line_color="red",
        annotation_text="Detection Threshold"
    )
    
    fig.update_layout(
        title='Signal Strength Distribution (dBm)',
        xaxis_title='Signal Strength (dBm)',
        yaxis_title='Count',
        showlegend=True,
        height=300
    )
    
    st.plotly_chart(fig, width="stretch")
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean", f"{np.mean(signal_data):.1f} dBm")
    with col2:
        st.metric("Median", f"{np.median(signal_data):.1f} dBm")
    with col3:
        st.metric("Max", f"{np.max(signal_data):.1f} dBm")
    with col4:
        above_threshold = np.sum(signal_data > config.SIGNAL_THRESHOLD)
        st.metric("Above Threshold", f"{above_threshold} ({above_threshold/len(signal_data)*100:.1f}%)")
        
def show_reference_signals():
    """Display reference 'happy path' signals"""
    from reference_signals import REFERENCE_SIGNALS, get_reference_characteristics
    
    st.subheader("📌 Reference Signals (Happy Path)")
    st.caption("These known-good captures anchor the ML learning system")
    
    for ref_name, ref_data in REFERENCE_SIGNALS.items():
        with st.expander(f"🎯 {ref_data['protocol']} - {ref_data['tpms_id']}"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Frequency", f"{ref_data['frequency']} MHz")
                st.metric("Signal Strength", f"{ref_data['signal_strength']} dBm")
            with col2:
                st.metric("Protocol", ref_data['protocol'])
                st.metric("Modulation", ref_data['modulation'])
            with col3:
                st.metric("TPMS ID", ref_data['tpms_id'])
                st.caption(ref_data['timestamp'])
            
            st.info(f"📝 {ref_data['notes']}")
    
    # Show reference characteristics
    ref_chars = get_reference_characteristics()
    st.divider()
    st.subheader("🎯 Detection Parameters (from references)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Frequency Range", f"{ref_chars['frequency_range'][0]}-{ref_chars['frequency_range'][1]} MHz")
    with col2:
        st.metric("Min Signal Strength", f"{ref_chars['min_signal_strength']} dBm")
    with col3:
        st.metric("Typical Strength", f"{ref_chars['typical_strength']} dBm")


def show_live_detection():
    """Live detection tab"""
    # ALWAYS process queue, even if not on this tab
    if st.session_state.is_scanning:
        process_signal_buffer() 
    
    st.header("Live TPMS Detection")
    
    # Add frequency status at top
    show_frequency_status()
    
    st.divider()
    
    # Signal histogram
    show_signal_histogram()
    
    st.divider()
    
    # Protocol monitoring
    show_protocol_monitoring()
    
    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Recent Vehicle Detections")


        if st.session_state.recent_detections:
            recent = st.session_state.recent_detections[-10:][::-1]

            for detection in recent:
                vehicle = detection['vehicle']
                dt = timestamp_to_mountain(detection['timestamp'])

                with st.container():
                    col_a, col_b, col_c = st.columns([2, 2, 1])

                    with col_a:
                        nickname = vehicle.get('nickname', 'Unknown Vehicle')
                        st.markdown(f"**{nickname}**")
                        st.caption(f"ID: {vehicle['id']}")

                    with col_b:
                        st.text(dt.strftime("%H:%M:%S"))
                        st.caption(f"Seen {vehicle['encounter_count']} times")

                    with col_c:
                        if st.button("View", key=f"view_{detection['vehicle_id']}_{detection['timestamp']}"):
                            st.session_state.selected_vehicle = vehicle['id']

                    st.divider()
        else:
            st.info("No vehicles detected yet. Start scanning to begin detection.")

    with col2:
        st.subheader("Live Signal Stream")
        recent_signals = st.session_state.db.get_recent_signals(10)

        if recent_signals:
            signal_df = pd.DataFrame(recent_signals)
            signal_df['timestamp'] = pandas_timestamp_to_mountain(signal_df['timestamp'])

            st.dataframe(
                signal_df[['tpms_id', 'timestamp', 'signal_strength', 'frequency']],
                hide_index=True,
                width="stretch"
            )
        else:
            st.info("Waiting for signals...")

        if st.session_state.is_scanning:
            time.sleep(1)
            st.rerun()

def show_protocol_monitoring():
    """Display detected protocols and unknown signals"""
    st.subheader("🔍 Protocol Detection")
    
    if not hasattr(st.session_state, 'decoder'):
        return
    
    # Get protocol statistics
    stats = st.session_state.decoder.get_protocol_statistics()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Unknown Signals Detected**")
        st.metric("Total Unknown", stats['total_unknown'])
        
        if stats['modulation_types']:
            st.write("**Modulation Types:**")
            for mod_type, count in stats['modulation_types'].items():
                st.write(f"- {mod_type}: {count}")
        else:
            st.info("No unknown signals detected yet")
    
    with col2:
        st.write("**Signal Characteristics**")
        
        if stats['common_baud_rates']:
            st.write("**Detected Baud Rates:**")
            for rate in stats['common_baud_rates']:
                st.write(f"- {rate:,} bps")
        
        if stats['avg_signal_strength'] > 0:
            st.metric("Avg Signal Strength", f"{stats['avg_signal_strength']:.1f} dBm")
    
    # Show recent unknown signals
    unknown_signals = st.session_state.decoder.get_unknown_signals(60)
    
    if unknown_signals:
        st.write("**Recent Unknown Signals (last 60s):**")
        
        unknown_df = pd.DataFrame([
            {
                'Time': datetime.fromtimestamp(s.timestamp).strftime('%H:%M:%S'),
                'Frequency': f"{s.frequency:.2f} MHz",
                'Strength': f"{s.signal_strength:.1f} dBm",
                'Modulation': s.modulation_type,
                'Baud Rate': f"{s.baud_rate:,}" if s.baud_rate else "Unknown",
                'Length': s.packet_length
            }
            for s in unknown_signals[-10:]
        ])
        
        st.dataframe(unknown_df, width="stretch", hide_index=True)

def show_vehicle_database():
    """Vehicle database tab"""
    st.header("Vehicle Database")

    vehicles = st.session_state.db.get_all_vehicles()

    if not vehicles:
        st.info("No vehicles in database yet.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        min_encounters = st.slider("Min Encounters", 1, 50, 1)
    with col2:
        sort_by = st.selectbox("Sort By", ["Last Seen", "First Seen", "Encounter Count"])
    with col3:
        search = st.text_input("Search TPMS ID")

    filtered = [v for v in vehicles if v['encounter_count'] >= min_encounters]

    if search:
        filtered = [v for v in filtered if search.upper() in str(v['tpms_ids'])]

    if sort_by == "Last Seen":
        filtered.sort(key=lambda x: x['last_seen'], reverse=True)
    elif sort_by == "First Seen":
        filtered.sort(key=lambda x: x['first_seen'], reverse=True)
    else:
        filtered.sort(key=lambda x: x['encounter_count'], reverse=True)

    for vehicle in filtered:
        with st.expander(
                f"🚗 {vehicle.get('nickname', 'Vehicle')} #{vehicle['id']} - "
                f"Seen {vehicle['encounter_count']} times"
        ):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.write("**TPMS IDs:**")
                for tpms_id in vehicle['tpms_ids']:
                    st.code(tpms_id, language=None)

                st.write("**Timeline:**")
                st.write(f"First seen: {format_timestamp(vehicle['first_seen'], '%Y-%m-%d %H:%M')}")
                st.write(f"Last seen: {format_timestamp(vehicle['last_seen'], '%Y-%m-%d %H:%M')}")

                current_nickname = vehicle.get('nickname', '')
                new_nickname = st.text_input(
                    "Nickname",
                    value=current_nickname,
                    key=f"nickname_{vehicle['id']}"
                )
                if new_nickname != current_nickname:
                    if st.button("Save Nickname", key=f"save_{vehicle['id']}"):
                        st.session_state.db.update_vehicle_nickname(vehicle['id'], new_nickname)
                        st.success("Nickname updated!")
                        st.rerun()

            with col2:
                history = st.session_state.db.get_vehicle_history(vehicle['id'])
                st.metric("Total Encounters", len(history['encounters']))

                if history['maintenance']:
                    avg_pressure = sum(m['avg_pressure'] for m in history['maintenance'] if m['avg_pressure']) / len(
                        history['maintenance'])
                    st.metric("Avg Pressure", f"{avg_pressure:.1f} PSI")

                prediction = st.session_state.ml_engine.predict_next_encounter(vehicle['id'])
                if prediction['prediction'] == 'estimated':
                    st.write("**Next encounter predicted:**")
                    st.write(prediction['predicted_datetime'].strftime('%Y-%m-%d %H:%M'))
                    st.progress(prediction['confidence'])
                    st.caption(f"Confidence: {prediction['confidence'] * 100:.0f}%")
                    
def safe_float_convert(value, default=0.0):
    """Safely convert various types to float"""
    if value is None:
        return default
    
    # If it's bytes, try to decode and convert
    if isinstance(value, bytes):
        try:
            # Try to decode as UTF-8 string first
            value = value.decode('utf-8')
        except:
            try:
                # Try to interpret as raw float bytes
                import struct
                if len(value) == 4:
                    return struct.unpack('f', value)[0]
                elif len(value) == 8:
                    return struct.unpack('d', value)[0]
            except:
                return default
    
    # Try direct conversion
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def render_sensor_database_tab(db):
    """Render the sensor database and management interface"""
    st.header("🔍 TPMS Sensor Database")

    # Get all sensors
    sensors_df = db.get_all_unique_sensors()

    if sensors_df.empty:
        st.warning("No TPMS sensors detected yet. Start scanning to collect sensor data.")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Sensors", len(sensors_df))
    with col2:
        st.metric("Total Signals", int(sensors_df['signal_count'].sum()))
    with col3:
        active_sensors = len(sensors_df[sensors_df['last_seen'] > time.time() - 3600])
        st.metric("Active (1hr)", active_sensors)
    with col4:
        orphaned = db.get_orphaned_sensors()
        st.metric("Unassigned", len(orphaned))

    # Tabs for different views
    sensor_tab1, sensor_tab2, sensor_tab3 = st.tabs([
        "📋 All Sensors",
        "🔍 Sensor Details",
        "⚠️ Unassigned Sensors"
    ])

    with sensor_tab1:
        st.subheader("All Detected Sensors")

        # Format the dataframe
        display_df = sensors_df.copy()
        display_df['first_seen'] = pandas_timestamp_to_mountain(display_df['first_seen'])
        display_df['last_seen'] = pandas_timestamp_to_mountain(display_df['last_seen'])
        
        # Convert to float and round
        display_df['avg_rssi'] = pd.to_numeric(display_df['avg_rssi'], errors='coerce').round(1)
        display_df['avg_pressure'] = pd.to_numeric(display_df['avg_pressure'], errors='coerce').round(1)
        display_df['avg_temperature'] = pd.to_numeric(display_df['avg_temperature'], errors='coerce').round(1)
        display_df['frequency'] = pd.to_numeric(display_df['frequency'], errors='coerce') / 1e6
        display_df['frequency'] = display_df['frequency'].round(2)

        # Rename columns for display
        display_df = display_df.rename(columns={
            'tpms_id': 'Sensor ID',
            'protocol': 'Protocol',
            'signal_count': 'Signals',
            'first_seen': 'First Seen',
            'last_seen': 'Last Seen',
            'avg_rssi': 'Avg RSSI (dBm)',
            'avg_pressure': 'Avg Pressure (PSI)',
            'avg_temperature': 'Avg Temp (°C)',
            'frequency': 'Frequency (MHz)'
        })

        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True
        )

        # Export option
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Export Sensor List (CSV)",
            data=csv,
            file_name=f"tpms_sensors_{int(time.time())}.csv",
            mime="text/csv"
        )

    with sensor_tab2:
        st.subheader("Detailed Sensor Analysis")

        # Sensor selector
        sensor_ids = sensors_df['tpms_id'].tolist()
        selected_sensor = st.selectbox(
            "Select Sensor ID",
            options=sensor_ids,
            format_func=lambda x: f"{x} ({sensors_df[sensors_df['tpms_id'] == x]['protocol'].iloc[0]})"
        )

        if selected_sensor:
            # Get statistics
            stats = db.get_sensor_statistics(selected_sensor)
            history_df = db.get_sensor_history(selected_sensor)

            # Display statistics
            st.markdown("### Sensor Statistics")
            stat_col1, stat_col2, stat_col3 = st.columns(3)

            with stat_col1:
                st.metric("Total Signals", stats['total_signals'])
                st.metric("Protocol", stats['protocol'])
                st.metric("First Seen", format_timestamp(stats['first_seen'], "%Y-%m-%d %H:%M"))

            with stat_col2:
                # Convert to float safely
                avg_rssi = safe_float_convert(stats['avg_rssi'])
                min_rssi = safe_float_convert(stats['min_rssi'])
                max_rssi = safe_float_convert(stats['max_rssi'])
    
                st.metric("Avg RSSI", f"{avg_rssi:.1f} dBm")
                st.metric("RSSI Range", f"{min_rssi:.1f} to {max_rssi:.1f} dBm")
    
                if stats.get('avg_pressure'):
                    avg_pressure = safe_float_convert(stats['avg_pressure'])
                    st.metric("Avg Pressure", f"{avg_pressure:.1f} PSI")

            with stat_col3:
                st.metric("Last Seen", format_timestamp(stats['last_seen'], "%Y-%m-%d %H:%M"))
                age_minutes = (time.time() - stats['last_seen']) / 60
                st.metric("Age", f"{age_minutes:.0f} min ago")
    
                if stats.get('avg_temp'):
                    avg_temp = safe_float_convert(stats['avg_temp'])
                    st.metric("Avg Temperature", f"{avg_temp:.1f} °C")


            # Signal strength over time
            st.markdown("### Signal Strength History")
            if not history_df.empty:
                chart_df = history_df[['timestamp', 'signal_strength']].copy()
                chart_df['timestamp'] = pandas_timestamp_to_mountain(chart_df['timestamp'])
                chart_df['signal_strength'] = pd.to_numeric(chart_df['signal_strength'], errors='coerce')

                fig = px.line(
                    chart_df,
                    x='timestamp',
                    y='signal_strength',
                    title=f"RSSI Over Time - {selected_sensor}",
                    labels={'timestamp': 'Time', 'signal_strength': 'RSSI (dBm)'}
                )
                st.plotly_chart(fig, width="stretch")

            # Pressure/Temperature over time (if available)
            if 'pressure_psi' in history_df.columns and history_df['pressure_psi'].notna().any():
                st.markdown("### Pressure History")
                pressure_df = history_df[history_df['pressure_psi'].notna()][['timestamp', 'pressure_psi']].copy()
                pressure_df['timestamp'] = pandas_timestamp_to_mountain(pressure_df['timestamp'])
                pressure_df['pressure_psi'] = pd.to_numeric(pressure_df['pressure_psi'], errors='coerce')

                fig = px.line(
                    pressure_df,
                    x='timestamp',
                    y='pressure_psi',
                    title=f"Tire Pressure Over Time - {selected_sensor}",
                    labels={'timestamp': 'Time', 'pressure_psi': 'Pressure (PSI)'}
                )
                st.plotly_chart(fig, width="stretch")

            # Raw signal history table
            with st.expander("📊 View Raw Signal History"):
                display_history = history_df.copy()
                display_history['timestamp'] = pandas_timestamp_to_mountain(display_history['timestamp'])
                st.dataframe(display_history, width="stretch")

    with sensor_tab3:
        st.subheader("Unassigned Sensors")
        st.info("These sensors have been detected but are not assigned to any vehicle.")

        orphaned_df = db.get_orphaned_sensors()

        if orphaned_df.empty:
            st.success("All sensors are assigned to vehicles!")
        else:
            # Display orphaned sensors
            display_orphaned = orphaned_df.copy()
            display_orphaned['last_seen'] = pandas_timestamp_to_mountain(display_orphaned['last_seen'])
            st.dataframe(display_orphaned, width="stretch")

            # Manual assignment interface
            st.markdown("### Assign Sensor to Vehicle")
            vehicles_list = db.get_all_vehicles()

            if vehicles_list:
                # Convert to DataFrame for easier handling
                vehicles_df = pd.DataFrame(vehicles_list)

                assign_col1, assign_col2, assign_col3 = st.columns([2, 2, 1])

                with assign_col1:
                    sensor_to_assign = st.selectbox(
                        "Select Sensor",
                        options=orphaned_df['tpms_id'].tolist()
                    )

                with assign_col2:
                    vehicle_to_assign = st.selectbox(
                        "Select Vehicle",
                        options=vehicles_df['id'].tolist(),
                        format_func=lambda x: (
                            vehicles_df[vehicles_df['id'] == x]['nickname'].iloc[0]
                            if vehicles_df[vehicles_df['id'] == x]['nickname'].iloc[0]
                            else f"Vehicle {x}"
                        )
                    )

                with assign_col3:
                    if st.button("Assign", type="primary"):
                        if db.assign_sensor_to_vehicle(sensor_to_assign, vehicle_to_assign):
                            st.success(f"Assigned {sensor_to_assign} to vehicle!")
                            st.rerun()
                        else:
                            st.error("Assignment failed")
            else:
                st.info("📝 No vehicles in database yet. Vehicles will appear here once detected during scanning.")


def show_frequency_status():
    """Display frequency hopping status and statistics"""
    st.subheader("📡 Frequency Status")
    
    if not st.session_state.is_scanning:
        st.info("Start scanning to see frequency statistics")
        return
    
    status = st.session_state.hackrf.get_status()
    freq_stats = status.get('frequency_stats', {})
    
    # Current frequency
    st.write(f"**Current Frequency:** {status['frequency']:.2f} MHz")
    
    # Initialize session state for controls if not exists
    if 'freq_hop_enabled' not in st.session_state:
        st.session_state.freq_hop_enabled = status.get('frequency_hopping', True)
    if 'hop_interval' not in st.session_state:
        st.session_state.hop_interval = status.get('hop_interval', 30.0)
    
    # Frequency hopping control
    col1, col2 = st.columns(2)
    with col1:
        hop_enabled = st.checkbox(
            "Enable Frequency Hopping",
            value=st.session_state.freq_hop_enabled,
            key="freq_hop_toggle"
        )
        # Only update if changed
        if hop_enabled != st.session_state.freq_hop_enabled:
            st.session_state.freq_hop_enabled = hop_enabled
            st.session_state.hackrf.set_frequency_hopping(hop_enabled)
            st.rerun()
    
    with col2:
        if st.session_state.freq_hop_enabled:
            hop_interval = st.slider(
                "Hop Interval (seconds)",
                min_value=10.0,
                max_value=60.0,
                value=st.session_state.hop_interval,
                step=5.0,
                key="hop_interval_slider"
            )
            # Only update if changed
            if abs(hop_interval - st.session_state.hop_interval) > 0.1:
                st.session_state.hop_interval = hop_interval
                st.session_state.hackrf.set_hop_interval(hop_interval)
    
    # Frequency statistics table
    if freq_stats:
        st.write("**Frequency Statistics:**")
        
        freq_df = pd.DataFrame([
            {
                'Frequency': f"{freq:.2f} MHz",
                'Samples': stats['samples'],
                'Avg Strength': f"{stats['avg_strength']:.1f} dBm",
                'Detections': stats['detections']
            }
            for freq, stats in freq_stats.items()
        ])
        
        st.dataframe(freq_df, width="stretch", hide_index=True)
        
        # Bar chart of detections per frequency
        if any(stats['detections'] > 0 for stats in freq_stats.values()):
            fig = go.Figure(data=[
                go.Bar(
                    x=[f"{freq:.2f}" for freq in freq_stats.keys()],
                    y=[stats['detections'] for stats in freq_stats.values()],
                    marker_color='lightgreen'
                )
            ])
            fig.update_layout(
                title='TPMS Detections by Frequency',
                xaxis_title='Frequency (MHz)',
                yaxis_title='Detections',
                height=300
            )
            st.plotly_chart(fig, width="stretch")


def show_analytics():
    """Analytics tab"""
    st.header("Analytics & Patterns")

    vehicles = st.session_state.db.get_all_vehicles(min_encounters=2)

    if not vehicles:
        st.info("Not enough data for analytics yet. Keep scanning!")
        return

    days = st.slider("Analysis Period (days)", 1, 90, 30)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Encounter Frequency")

        encounter_data = []
        for vehicle in vehicles:
            history = st.session_state.db.get_vehicle_history(vehicle['id'])
            for encounter in history['encounters']:
                mt_timestamp = timestamp_to_mountain(encounter['timestamp'])
                encounter_data.append({
                    'vehicle_id': vehicle['id'],
                    'nickname': vehicle.get('nickname', 'Vehicle ' + str(vehicle['id'])),
                    'timestamp': mt_timestamp,
                    'date': mt_timestamp.date()
                })

        if encounter_data:
            df = pd.DataFrame(encounter_data)

            daily_counts = df.groupby('date').size().reset_index(name='encounters')
            fig = px.line(
                daily_counts,
                x='date',
                y='encounters',
                title='Daily Vehicle Encounters',
                labels={'date': 'Date', 'encounters': 'Number of Vehicles'}
            )
            st.plotly_chart(fig, width="stretch")

            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.day_name()

            heatmap_data = df.groupby(['day_of_week', 'hour']).size().reset_index(name='count')

            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            heatmap_pivot = heatmap_data.pivot(index='day_of_week', columns='hour', values='count').fillna(0)
            heatmap_pivot = heatmap_pivot.reindex(days_order)

            fig_heatmap = go.Figure(data=go.Heatmap(
                z=heatmap_pivot.values,
                x=heatmap_pivot.columns,
                y=heatmap_pivot.index,
                colorscale='YlOrRd',
                text=heatmap_pivot.values,
                texttemplate='%{text}',
                textfont={"size": 10}
            ))
            fig_heatmap.update_layout(
                title='Encounter Patterns by Day and Hour',
                xaxis_title='Hour of Day',
                yaxis_title='Day of Week'
            )
            st.plotly_chart(fig_heatmap, width="stretch")

    with col2:
        st.subheader("Top Vehicles")

        top_vehicles = sorted(vehicles, key=lambda x: x['encounter_count'], reverse=True)[:10]

        top_df = pd.DataFrame([
            {
                'Vehicle': v.get('nickname', 'Vehicle ' + str(v['id'])),
                'Encounters': v['encounter_count'],
                'Last Seen': format_timestamp(v['last_seen'], '%Y-%m-%d')
            }
            for v in top_vehicles
        ])

        fig_bar = px.bar(
            top_df,
            x='Encounters',
            y='Vehicle',
            orientation='h',
            title='Most Frequently Encountered Vehicles',
            color='Encounters',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_bar, width="stretch")

        st.subheader("Recent Activity")
        recent_vehicles = sorted(vehicles, key=lambda x: x['last_seen'], reverse=True)[:5]

        for v in recent_vehicles:
            now_mountain = datetime.now(MOUNTAIN_TZ)
            last_seen_mountain = timestamp_to_mountain(v['last_seen'])
            time_ago = now_mountain - last_seen_mountain
            hours_ago = time_ago.total_seconds() / 3600

            if hours_ago < 1:
                time_str = f"{int(time_ago.total_seconds() / 60)} minutes ago"
            elif hours_ago < 24:
                time_str = f"{int(hours_ago)} hours ago"
            else:
                time_str = f"{int(hours_ago / 24)} days ago"

            nickname = v.get('nickname', 'Vehicle ' + str(v['id']))
            st.write(f"**{nickname}** - {time_str}")


def show_maintenance():
    """Maintenance tracking tab"""
    st.header("🔧 Tire Maintenance Monitoring")

    vehicles = st.session_state.db.get_all_vehicles(min_encounters=3)

    if not vehicles:
        st.info("Not enough data for maintenance analysis yet.")
        return

    vehicle_options = {
        f"{v.get('nickname', 'Vehicle ' + str(v['id']))} (ID: {v['id']})": v['id']
        for v in vehicles
    }

    selected_name = st.selectbox("Select Vehicle", list(vehicle_options.keys()))
    vehicle_id = vehicle_options[selected_name]

    days = st.slider("Analysis Period (days)", 7, 90, 30, key="maintenance_days")

    analysis = st.session_state.db.analyze_maintenance(vehicle_id, days)

    if not analysis:
        st.warning("No maintenance data available for this vehicle.")
        return

    st.subheader("Tire Health Overview")

    cols = st.columns(4)
    for idx, (tpms_id, data) in enumerate(analysis.items()):
        with cols[idx % 4]:
            st.write(f"**Tire {idx + 1}**")
            st.caption(f"ID: {tpms_id[:8]}...")

            if data['avg_pressure']:
                pressure = data['avg_pressure']

                if 30 <= pressure <= 35:
                    color = "🟢"
                    status = "Good"
                elif 28 <= pressure < 30 or 35 < pressure <= 38:
                    color = "🟡"
                    status = "Warning"
                else:
                    color = "🔴"
                    status = "Alert"

                st.metric(
                    "Avg Pressure",
                    f"{pressure:.1f} PSI",
                    delta=f"{data['pressure_std']:.1f} std"
                )
                st.write(f"{color} {status}")

                if data['avg_temp']:
                    st.metric("Avg Temp", f"{data['avg_temp']:.1f}°C")

                if data['alerts']:
                    st.warning("⚠️ " + ", ".join(data['alerts']))


def show_ml_insights():
    """ML insights tab"""
    st.header("🤖 Machine Learning Insights")

    patterns = st.session_state.ml_engine.find_patterns(days=30)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Frequent Commuters")

        if patterns['frequent_vehicles']:
            for vehicle_info in patterns['frequent_vehicles'][:10]:
                with st.container():
                    st.write(f"**{vehicle_info['nickname']}**")

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Encounters", vehicle_info['encounter_count'])
                    with col_b:
                        days_known = (vehicle_info['last_seen'] - vehicle_info['first_seen']).days
                        if days_known > 0:
                            freq = vehicle_info['encounter_count'] / days_known
                            st.metric("Frequency", f"{freq:.1f}/day")

                    st.divider()
        else:
            st.info("Not enough data to identify patterns yet.")

    with col2:
        st.subheader("Detection Patterns")
        st.info("Pattern analysis will appear here as data accumulates.")

def process_signal_queue():
    """Process signals from the queue - runs in main Streamlit thread"""
    processed = 0
    max_batch = 10
    
    # Debug: Log when this function is called
    if not hasattr(process_signal_queue, 'call_count'):
        process_signal_queue.call_count = 0
    process_signal_queue.call_count += 1
    
    if process_signal_queue.call_count <= 3 or process_signal_queue.call_count % 50 == 0:
        print(f"[{time.time():.3f}] process_signal_queue() called #{process_signal_queue.call_count}, "
              f"queue_size={signal_queue.qsize()}", flush=True)
    
    while not signal_queue.empty() and processed < max_batch:
        try:
            data = signal_queue.get_nowait()
            
            if processed == 0:  # Log first item in batch
                print(f"  Processing batch: strength={data['signal_strength']:.1f} dBm", flush=True)
            
            # Add to signal history for histogram
            signal_history_queue.append(data['signal_strength'])
            
            # Process with decoder
            signals = st.session_state.decoder.process_samples(
                data['iq_samples'], 
                data['frequency']
            )
            
            if signals:
                print(f"  ✅ Decoder found {len(signals)} TPMS signals!", flush=True)

            for signal in signals:
                signal.signal_strength = data['signal_strength']
                
                print(f"  📡 TPMS Decoded: ID={signal.tpms_id}, "
                      f"Protocol={signal.protocol}, "
                      f"Pressure={signal.pressure_psi} PSI", flush=True)
                
                # Increment detection count for this frequency
                st.session_state.hackrf.increment_detection(data['frequency'])

                signal_dict = {
                    'tpms_id': signal.tpms_id,
                    'timestamp': signal.timestamp,
                    'frequency': signal.frequency,
                    'signal_strength': signal.signal_strength,
                    'snr': signal.snr,
                    'pressure_psi': signal.pressure_psi,
                    'temperature_c': signal.temperature_c,
                    'battery_low': signal.battery_low,
                    'protocol': signal.protocol,
                    'raw_data': signal.raw_data
                }
                
                st.session_state.db.insert_signal(signal_dict)
                st.session_state.signal_buffer.append(signal_dict)

            # Process for vehicle clustering
            if len(st.session_state.signal_buffer) > 10:
                vehicle_ids = st.session_state.ml_engine.process_signals(st.session_state.signal_buffer)

                for vehicle_id in vehicle_ids:
                    vehicle_info = st.session_state.db.get_vehicle_history(vehicle_id)
                    st.session_state.recent_detections.append({
                        'vehicle_id': vehicle_id,
                        'timestamp': time.time(),
                        'vehicle': vehicle_info['vehicle']
                    })

                st.session_state.signal_buffer = []
            
            processed += 1
            
        except queue.Empty:
            break
        except Exception as e:
            print(f"  ❌ Error processing signal: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue
    
    #if processed > 0:
        #print(f"  Processed {processed} items from queue", flush=True)


# Update show_trigger_controls()
def show_trigger_controls():
    """ESP32 LF Trigger Controls"""
    st.header("📡 ESP32 LF Trigger Control")
    
    # Connection status
    if st.session_state.esp32_trigger.connected:
        st.success("✅ ESP32 Connected")
    else:
        st.error("❌ ESP32 Not Connected")
        if st.button("🔄 Reconnect"):
            st.session_state.esp32_trigger.check_connection()
            st.rerun()
    
    if not st.session_state.esp32_trigger.connected:
        st.info("Connect to WiFi network: **TPMS_Trigger** (password: tpms12345)")
        return
    
    st.info("⚠️  **Warning:** Transmitting requires proper licensing and should only be used in controlled environments.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔧 Sensor Activation")
        
        protocol = st.selectbox(
            "Trigger Protocol",
            list(st.session_state.trigger.trigger_patterns.keys()),
            key="trigger_protocol_select"
        )
        
        if st.button("📡 Send Single Trigger", key="send_trigger_btn"):
            with st.spinner("Sending trigger..."):
                st.session_state.trigger.send_trigger(protocol)
                st.success(f"✅ {protocol.title()} trigger sent!")
        
        st.divider()
        
        st.subheader("🔄 Continuous Triggering")
        
        trigger_interval = st.slider(
            "Trigger Interval (seconds)",
            min_value=0.5,
            max_value=5.0,
            value=1.0,
            step=0.5,
            key="trigger_interval_slider"
        )
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("▶️ Start Continuous", 
                        disabled=st.session_state.trigger.is_transmitting,
                        key="start_continuous_trigger_btn"):
                st.session_state.trigger.start_continuous_trigger(protocol, trigger_interval)
                st.success("Continuous trigger started!")
        
        with col_b:
            if st.button("⏹️ Stop Continuous",
                        disabled=not st.session_state.trigger.is_transmitting,
                        key="stop_continuous_trigger_btn"):
                st.session_state.trigger.stop_continuous_trigger()
                st.info("Continuous trigger stopped")
    
    with col2:
        st.subheader("🎯 Active Scan")
        
        st.write("**Trigger and Listen Mode**")
        st.caption("Send trigger, then listen for sensor responses")
        
        listen_duration = st.slider(
            "Listen Duration (seconds)",
            min_value=1.0,
            max_value=10.0,
            value=5.0,
            step=1.0,
            key="listen_duration_slider"
        )
        
        if st.button("🔍 Trigger & Listen", key="trigger_listen_btn"):
            with st.spinner(f"Triggering and listening for {listen_duration}s..."):
                st.session_state.dual_mode.trigger_and_listen(protocol, listen_duration)
                st.success("Scan complete! Check Live Detection tab for results.")
        
        st.divider()
        
        st.write("**Multi-Protocol Scan**")
        st.caption("Try all protocols sequentially")
        
        if st.button("🔍 Scan All Protocols", key="scan_all_btn"):
            with st.spinner("Scanning all protocols..."):
                st.session_state.dual_mode.scan_and_activate()
                st.success("Multi-protocol scan complete!")
    
    st.divider()
    
    # Trigger status
    st.subheader("📊 Trigger Status")
    
    status = st.session_state.trigger.get_status()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("LF Frequency", f"{status['lf_frequency'] / 1000:.1f} kHz")
    
    with col2:
        st.metric("Status", "🟢 Active" if status['transmitting'] else "🔴 Idle")
    
    with col3:
        if status['transmitting']:
            st.metric("Trigger Interval", f"{status['trigger_interval']:.1f}s")
    
    # Protocol details
    with st.expander("📋 Available Trigger Protocols"):
        for proto_name, proto_config in st.session_state.trigger.trigger_patterns.items():
            st.write(f"**{proto_name.title()}**")
            st.write(f"- Frequency: {proto_config['frequency'] / 1000:.1f} kHz")
            st.write(f"- Pulse Width: {proto_config['pulse_width'] * 1000:.0f}ms")
            st.write(f"- Pulse Count: {proto_config['pulse_count']}")
            st.divider()


def main():
    st.title("🚗 TPMS Tracker - Intelligent Vehicle Pattern Recognition")
    st.caption(f"🕐 Displaying times in Mountain Time (MT)")
    
    if st.session_state.is_scanning:
        process_signal_buffer() 

    with st.sidebar:
        st.header("⚙️ Control Panel")
        st.subheader("Scanner Control")

        # Manual frequency selection
        if not st.session_state.is_scanning:
            freq_mhz = st.selectbox(
                "Select Frequency",
                [314.9, 315.0, 433.92],
                index=0,
                key="freq_select"
            )
            
            if st.button("Set Frequency", key="set_freq_btn"):
                st.session_state.hackrf.change_frequency(freq_mhz * 1e6)
                st.success(f"Frequency set to {freq_mhz} MHz")
        else:
            # Show current frequency when scanning
            status = st.session_state.hackrf.get_status()
            st.info(f"📡 Scanning: {status['frequency']:.1f} MHz")
            
            # Allow frequency change during scan
            if st.button("Change Frequency", key="change_freq_btn"):
                st.session_state.show_freq_change = True
            
            if st.session_state.get('show_freq_change', False):
                new_freq = st.selectbox(
                    "New Frequency",
                    [314.9, 315.0, 433.92],
                    key="new_freq_select"
                )
                
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Apply", key="apply_freq_btn"):
                        st.session_state.hackrf.change_frequency(new_freq * 1e6)
                        st.session_state.show_freq_change = False
                        st.rerun()
                with col_b:
                    if st.button("Cancel", key="cancel_freq_btn"):
                        st.session_state.show_freq_change = False
                        st.rerun()

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start Scan", disabled=st.session_state.is_scanning, key="main_start_btn"):
                # First, ensure globals are initialized
                    init_globals()
    
                    # Then start everything
                    st.session_state.is_scanning = True
                    st.session_state.hackrf.start(signal_callback)
                    start_processing()  # Start processing thread AFTER everything is ready
                    st.success("Scanning started!")
                    st.rerun()

        with col2:
            if st.button("⏹️ Stop Scan", disabled=not st.session_state.is_scanning, key="main_stop_btn"):
                st.session_state.is_scanning = False
                st.session_state.hackrf.stop()
                stop_processing()
                st.info("Scanning stopped")
                st.rerun()



        # Scanner status display
        if st.session_state.is_scanning:
            st.divider()
            st.metric("Status", "🟢 Active")
            st.metric("Frequency", f"{config.DEFAULT_FREQUENCY / 1e6:.1f} MHz")
            st.metric("Sample Rate", f"{config.SAMPLE_RATE / 1e6:.2f} MS/s")
            st.metric("Bandwidth", f"{config.BANDWIDTH / 1e6:.2f} MHz")
            st.caption("🔒 Continuous reception mode (no hopping)")
            
            # Advanced settings
            with st.expander("⚙️ Advanced Settings"):
                new_gain = st.slider(
                    "LNA Gain (dB)",
                    min_value=0,
                    max_value=40,
                    value=config.DEFAULT_GAIN,
                    step=8,
                    help="Adjust receiver gain",
                    key="main_lna_gain_slider"  # ADD THIS
                )
                
                new_vga = st.slider(
                    "VGA Gain (dB)",
                    min_value=0,
                    max_value=62,
                    value=config.VGA_GAIN,
                    step=2,
                    help="Adjust VGA gain",
                    key="main_vga_gain_slider"  # ADD THIS
                )
                
                if st.button("Apply Settings", key="main_apply_settings_btn"):
                    # Restart with new settings
                    st.session_state.hackrf.stop()
                    st.session_state.hackrf.gain = new_gain
                    st.session_state.hackrf.vga_gain = new_vga
                    st.session_state.hackrf.start(signal_callback)
                    st.success("Settings applied!")
        else:
            st.metric("Status", "🔴 Inactive")

        st.divider()

        st.subheader("📊 Statistics")
        vehicles = st.session_state.db.get_all_vehicles()
        st.metric("Known Vehicles", len(vehicles))

        recent_signals = st.session_state.db.get_recent_signals(3600)
        st.metric("Signals (1hr)", len(recent_signals))
        
        # Signals per hour
        if recent_signals:
            st.metric("Signals/Hour", len(recent_signals))

    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🎯 Live Detection",
        "🚗 Vehicle Database",
        "🔍 Sensor Database",
        "📈 Analytics",
        "🔧 Maintenance",
        "🤖 ML Insights",
        "📡 Sensor Trigger" 
    ])

    with tab1:
        show_live_detection()
    
    with tab2:
        show_vehicle_database()

    with tab3:
        render_sensor_database_tab(st.session_state.db)

    with tab4:
        show_analytics()

    with tab5:
        show_maintenance()

    with tab6:
        show_ml_insights()
        
    with tab7:
        show_trigger_controls()


if __name__ == "__main__":
    main()
