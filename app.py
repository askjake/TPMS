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

# Import config and modules
from config import config
from database import TPMSDatabase
from hackrf_interface import HackRFInterface
from tpms_decoder import TPMSDecoder
from ml_engine import VehicleClusteringEngine

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

# Initialize session state
if 'db' not in st.session_state:
    st.session_state.db = TPMSDatabase(config.DB_PATH)
    st.session_state.hackrf = HackRFInterface()
    st.session_state.decoder = TPMSDecoder(config.SAMPLE_RATE)
    st.session_state.ml_engine = VehicleClusteringEngine(st.session_state.db)
    st.session_state.is_scanning = False
    st.session_state.signal_buffer = []
    st.session_state.recent_detections = []


def signal_callback(iq_samples, signal_strength, frequency):
    """Callback for processing HackRF samples - queue-based"""
    try:
        # Put data in queue instead of processing directly
        signal_queue.put({
            'iq_samples': iq_samples,
            'signal_strength': signal_strength,
            'frequency': frequency,
            'timestamp': time.time()
        }, block=False)
    except queue.Full:
        pass  # Drop samples if queue is full

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
    
    st.plotly_chart(fig, use_container_width=True)
    
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
    if st.session_state.is_scanning:
        process_signal_queue()
    
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
                dt = datetime.fromtimestamp(detection['timestamp'])

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
            signal_df['timestamp'] = pd.to_datetime(signal_df['timestamp'], unit='s')

            st.dataframe(
                signal_df[['tpms_id', 'timestamp', 'signal_strength', 'frequency']],
                hide_index=True,
                use_container_width=True
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
        
        st.dataframe(unknown_df, use_container_width=True, hide_index=True)

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
                st.write(f"First seen: {datetime.fromtimestamp(vehicle['first_seen']).strftime('%Y-%m-%d %H:%M')}")
                st.write(f"Last seen: {datetime.fromtimestamp(vehicle['last_seen']).strftime('%Y-%m-%d %H:%M')}")

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
        
        st.dataframe(freq_df, use_container_width=True, hide_index=True)
        
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
            st.plotly_chart(fig, use_container_width=True)


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
                encounter_data.append({
                    'vehicle_id': vehicle['id'],
                    'nickname': vehicle.get('nickname', 'Vehicle ' + str(vehicle['id'])),
                    'timestamp': datetime.fromtimestamp(encounter['timestamp']),
                    'date': datetime.fromtimestamp(encounter['timestamp']).date()
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
            st.plotly_chart(fig, use_container_width=True)

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
            st.plotly_chart(fig_heatmap, use_container_width=True)

    with col2:
        st.subheader("Top Vehicles")

        top_vehicles = sorted(vehicles, key=lambda x: x['encounter_count'], reverse=True)[:10]

        top_df = pd.DataFrame([
            {
                'Vehicle': v.get('nickname', 'Vehicle ' + str(v['id'])),
                'Encounters': v['encounter_count'],
                'Last Seen': datetime.fromtimestamp(v['last_seen']).strftime('%Y-%m-%d')
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
        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Recent Activity")
        recent_vehicles = sorted(vehicles, key=lambda x: x['last_seen'], reverse=True)[:5]

        for v in recent_vehicles:
            time_ago = datetime.now() - datetime.fromtimestamp(v['last_seen'])
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
    
    while not signal_queue.empty() and processed < max_batch:
        try:
            data = signal_queue.get_nowait()
            
            # Add to signal history for histogram
            signal_history_queue.append(data['signal_strength'])
            
            # Process with decoder
            signals = st.session_state.decoder.process_samples(
                data['iq_samples'], 
                data['frequency']
            )

            for signal in signals:
                signal.signal_strength = data['signal_strength']
                
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
            print(f"Error processing signal: {e}")
            continue



def main():
    st.title("🚗 TPMS Tracker - Intelligent Vehicle Pattern Recognition")

    with st.sidebar:
        st.header("⚙️ Control Panel")
        st.subheader("Scanner Control")

        frequency = st.selectbox(
            "Frequency (MHz)",
            config.FREQUENCIES,
            index=0,
            key="main_frequency_select"  # ADD THIS
        )
        
        # Convert MHz to Hz for HackRF
        frequency_hz = frequency * 1e6

        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start Scan", disabled=st.session_state.is_scanning, key="main_start_btn"):
                st.session_state.is_scanning = True
                # Use the correct start() method signature
                st.session_state.hackrf.start(signal_callback)
                st.success("Scanning started!")

        with col2:
            if st.button("⏹️ Stop Scan", disabled=not st.session_state.is_scanning, key="main_stop_btn"):
                st.session_state.is_scanning = False
                st.session_state.hackrf.stop()
                st.info("Scanning stopped")

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
    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Live Detection",
        "📌 Reference Signals",
        "🚗 Vehicle Database",
        "📈 Analytics",
        "🔧 Maintenance",
        "🤖 ML Insights"
    ])

    with tab0:
        show_live_detection()
    
    with tab1:
        show_reference_signals()

    with tab2:
        show_vehicle_database()

    with tab3:
        show_analytics()

    with tab4:
        show_maintenance()

    with tab5:
        show_ml_insights()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
