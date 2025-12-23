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
from debug_tools import DebugTools, SpectrumPeak, ModulationAnalysis

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

def show_debug_tools():
    """Debug and diagnostic tools tab"""
    st.header("🔧 Debug & Diagnostic Tools")
    
    st.warning("⚠️ Running diagnostics will stop any active scanning!")
    
    # Initialize debug tools
    if 'debug_tools' not in st.session_state:
        st.session_state.debug_tools = DebugTools()
    
    # Hardware Info Section
    st.subheader("📡 Hardware Information")
    
    if st.button("🔍 Check Hardware", type="primary"):
        with st.spinner("Checking hardware..."):
            hw_info = st.session_state.debug_tools.get_hardware_info()
            
            if hw_info.device_found:
                col1, col2 = st.columns(2)
                with col1:
                    st.success("✅ HackRF One Detected")
                    st.metric("Serial Number", hw_info.serial_number)
                    st.metric("Board ID", hw_info.board_id)
                with col2:
                    st.metric("Firmware Version", hw_info.firmware_version)
                    st.metric("Part ID", hw_info.part_id)
                    if hw_info.operacake_detected:
                        st.info("🎛️ Opera Cake detected")
            else:
                st.error("❌ HackRF not found or not responding")
    
    st.divider()
    
    # Spectrum Scan Section
    st.subheader("📊 Spectrum Scanner")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        scan_start = st.number_input("Start Freq (MHz)", value=310.0, step=1.0)
    with col2:
        scan_end = st.number_input("End Freq (MHz)", value=320.0, step=1.0)
    with col3:
        scan_step = st.number_input("Step (MHz)", value=0.5, step=0.1, min_value=0.1)
    
    if st.button("🔍 Run Spectrum Scan"):
        # Stop any active scanning
        if st.session_state.is_scanning:
            st.session_state.hackrf.stop()
            st.session_state.is_scanning = False
            time.sleep(1)
    
        with st.spinner(f"Scanning {scan_start} - {scan_end} MHz..."):
            peaks, raw_spectrum = st.session_state.debug_tools.spectrum_scan(
                scan_start, scan_end, step=scan_step, duration=0.5
            )
        
            # Always show the full spectrum plot
            st.subheader("📊 Full Spectrum")
        
            if raw_spectrum:
                freqs = [f for f, p in raw_spectrum]
                powers = [p for f, p in raw_spectrum]
            
                fig = go.Figure()
            
                # Plot all measurements
                fig.add_trace(go.Scatter(
                    x=freqs,
                    y=powers,
                    mode='lines+markers',
                    name='Measured Power',
                    line=dict(color='lightblue', width=2),
                    marker=dict(size=4)
                ))
            
                # Add threshold line
                fig.add_hline(
                    y=-80, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text="Detection Threshold (-80 dBm)"
                )
            
                # Highlight peaks
                if peaks:
                    peak_freqs = [p.frequency for p in peaks]
                    peak_powers = [p.power for p in peaks]
                    fig.add_trace(go.Scatter(
                        x=peak_freqs,
                        y=peak_powers,
                        mode='markers',
                        name='Detected Peaks',
                        marker=dict(size=12, color='red', symbol='star')
                    ))
            
                fig.update_layout(
                    title=f'Spectrum Scan: {scan_start} - {scan_end} MHz',
                    xaxis_title='Frequency (MHz)',
                    yaxis_title='Power (dBm)',
                    height=500,
                    showlegend=True,
                    hovermode='x unified'
                )
            
                st.plotly_chart(fig, use_container_width=True)
            
                # Statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Frequencies Scanned", len(raw_spectrum))
                with col2:
                    st.metric("Peaks Found", len(peaks))
                with col3:
                    st.metric("Avg Power", f"{np.mean(powers):.1f} dBm")
                with col4:
                st.    metric("Max Power", f"{np.max(powers):.1f} dBm")
        
            # Show peaks table if any
            if peaks:
                st.subheader("🎯 Detected Peaks")
                peaks_df = pd.DataFrame([
                    {
                        'Frequency (MHz)': f"{p.frequency:.2f}",
                        'Power (dBm)': f"{p.power:.1f}",
                        'SNR (dB)': f"{p.snr:.1f}",
                        'Bandwidth (MHz)': f"{p.bandwidth:.2f}"
                    }
                    for p in peaks
                ])
            
                st.dataframe(peaks_df, use_container_width=True, hide_index=True)
            else:
                st.info("ℹ️ No peaks detected above -80 dBm threshold")

    
    st.divider()
    
    # Modulation Analysis Section
    st.subheader("🔬 Modulation Analyzer")
    
    analyze_freq = st.number_input("Frequency to Analyze (MHz)", value=314.9, step=0.1)
    analyze_duration = st.slider("Capture Duration (seconds)", 0.5, 5.0, 2.0, 0.5)
    
    if st.button("🔬 Analyze Modulation"):
        # Stop any active scanning
        if st.session_state.is_scanning:
            st.session_state.hackrf.stop()
            st.session_state.is_scanning = False
            time.sleep(1)
        
        with st.spinner(f"Analyzing {analyze_freq} MHz..."):
            analysis = st.session_state.debug_tools.analyze_modulation(
                analyze_freq, duration=analyze_duration
            )
            
            st.success("✅ Analysis complete")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Modulation Type", analysis.modulation_type)
                st.metric("Confidence", f"{analysis.confidence*100:.0f}%")
            with col2:
                st.metric("Baud Rate", f"{analysis.baud_rate:,} bps" if analysis.baud_rate > 0 else "Unknown")
                st.metric("Bandwidth", f"{analysis.bandwidth:.2f} MHz")
            with col3:
                st.metric("Frequency", f"{analysis.frequency:.2f} MHz")
            
            # Show characteristics
            if analysis.characteristics:
                with st.expander("📈 Signal Characteristics"):
                    char_df = pd.DataFrame([
                        {'Parameter': k, 'Value': f"{v:.4f}"}
                        for k, v in analysis.characteristics.items()
                    ])
                    st.dataframe(char_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Full Diagnostic Suite
    st.subheader("🚀 Full Diagnostic Suite")
    
    st.info("This will run a complete diagnostic including hardware check, spectrum scan, and modulation analysis on all detected signals.")
    
    if st.button("🚀 Run Full Diagnostic", type="primary"):
        # Stop any active scanning
        if st.session_state.is_scanning:
            st.session_state.hackrf.stop()
            st.session_state.is_scanning = False
            time.sleep(1)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Starting diagnostic...")
        progress_bar.progress(10)
        
        results = st.session_state.debug_tools.run_full_diagnostic()
        
        progress_bar.progress(100)
        status_text.text("Complete!")
        
        # Display results
        st.success("✅ Full diagnostic complete!")
        
        # Hardware
        with st.expander("📡 Hardware Info", expanded=True):
            hw = results['hardware']
            if hw.device_found:
                st.write(f"**Serial:** {hw.serial_number}")
                st.write(f"**Board ID:** {hw.board_id}")
                st.write(f"**Firmware:** {hw.firmware_version}")
            else:
                st.error("Hardware not detected")
        
        # Spectrum
        if results['raw_spectrum']:
            with st.expander(f"📊 Spectrum Scan ({len(results['raw_spectrum'])} frequencies measured)", expanded=True):
                # Plot full spectrum
                freqs = [f for f, p in results['raw_spectrum']]
                powers = [p for f, p in results['raw_spectrum']]
        
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=freqs,
                    y=powers,
                    mode='lines+markers',
                    name='Power',
                    line=dict(color='lightblue', width=2),
                    marker=dict(size=3)
                ))
        
                # Mark TPMS frequencies
                for tpms_freq in config.FREQUENCIES:
                    fig.add_vline(
                        x=tpms_freq,
                        line_dash="dash",
                        line_color="green",
                        annotation_text=f"{tpms_freq} MHz"
                    )
        
                fig.add_hline(y=-80, line_dash="dash", line_color="red", annotation_text="Threshold")
        
                fig.update_layout(
                    title='Complete Spectrum Scan',
                    xaxis_title='Frequency (MHz)',
                    yaxis_title='Power (dBm)',
                    height=400
                )
        
                st.plotly_chart(fig, use_container_width=True)
        
                # Show top peaks
                if results['spectrum_scan']:
                    st.write("**Top Detected Peaks:**")
                    spectrum_df = pd.DataFrame([
                        {
                            'Frequency': f"{p.frequency:.2f} MHz",
                            'Power': f"{p.power:.1f} dBm",
                            'SNR': f"{p.snr:.1f} dB"
                        }
                        for p in results['spectrum_scan'][:10]
                    ])
                    st.dataframe(spectrum_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No peaks above -80 dBm detected")

        
        # Modulation
        if results['modulation_analysis']:
            with st.expander(f"🔬 Modulation Analysis ({len(results['modulation_analysis'])} signals)", expanded=True):
                for analysis in results['modulation_analysis']:
                    st.write(f"**{analysis.frequency:.2f} MHz:** {analysis.modulation_type} "
                            f"({analysis.confidence*100:.0f}% confidence, {analysis.baud_rate:,} baud)")


def main():
    st.title("🚗 TPMS Tracker - Intelligent Vehicle Pattern Recognition")

    with st.sidebar:
        st.header("⚙️ Control Panel")
        st.subheader("Scanner Control")

        frequency = st.selectbox(
            "Frequency (MHz)",
            config.FREQUENCIES,
            index=0
        )
        # In the sidebar, after frequency selection
        if st.session_state.is_scanning:
            st.divider()
    
            # Manual gain control
            with st.expander("⚙️ Advanced Settings"):
                current_gain = st.session_state.hackrf.current_gain
                new_gain = st.slider(
                    "Manual Gain (dB)",
                    min_value=0,
                    max_value=47,
                    value=current_gain,
                    step=2,
                    help="Adjust receiver gain (must be even number)"
                )
        
                if new_gain != current_gain and st.button("Apply Gain"):
                    st.session_state.hackrf.current_gain = new_gain
                    st.success(f"Gain set to {new_gain} dB (will apply on next hop)")


        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start Scan", disabled=st.session_state.is_scanning):
                st.session_state.is_scanning = True
                st.session_state.hackrf.start(frequency, signal_callback)
                st.success("Scanning started!")

        with col2:
            if st.button("⏹️ Stop Scan", disabled=not st.session_state.is_scanning):
                st.session_state.is_scanning = False
                st.session_state.hackrf.stop()
                st.info("Scanning stopped")

        if st.session_state.is_scanning:
            status = st.session_state.hackrf.get_status()
            st.metric("Status", "🟢 Active")
            st.metric("Frequency", f"{status['frequency']:.2f} MHz")
            st.metric("Gain", f"{status['gain']} dB")
            if status['avg_signal_strength']:
                st.metric("Signal", f"{status['avg_signal_strength']:.1f} dBm")
        else:
            st.metric("Status", "🔴 Inactive")

        st.divider()

        st.subheader("📊 Statistics")
        vehicles = st.session_state.db.get_all_vehicles()
        st.metric("Known Vehicles", len(vehicles))

        recent_signals = st.session_state.db.get_recent_signals(3600)
        st.metric("Signals (1hr)", len(recent_signals))

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Live Detection",
        "🚗 Vehicle Database",
        "📈 Analytics",
        "🔧 Maintenance",
        "🤖 ML Insights",
        "🔧 Debug Tools"
    ])


    with tab1:
        show_live_detection()

    with tab2:
        show_vehicle_database()

    with tab3:
        show_analytics()

    with tab4:
        show_maintenance()

    with tab5:
        show_ml_insights()
    
    with tab6:
        show_debug_tools()



if __name__ == "__main__":
    main()
