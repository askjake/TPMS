import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import threading
from queue import Queue
import numpy as np

from config import config
from database import TPMSDatabase
from hackrf_interface import HackRFInterface
from tpms_decoder import TPMSDecoder
from ml_engine import VehicleClusteringEngine

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
    """Callback for processing HackRF samples"""
    # Decode TPMS signals
    signals = st.session_state.decoder.process_samples(iq_samples, frequency)

    for signal in signals:
        # Add signal strength
        signal.signal_strength = signal_strength

        # Store in database
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

    # Process signals for vehicle clustering
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


def show_live_detection():
    """Live detection tab"""
    st.header("Live TPMS Detection")

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
                    'nickname': vehicle.get('nickname', f"Vehicle {vehicle['id']}"),
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
                'Vehicle': v.get('nickname', f"Vehicle {v['id']}"),
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

            st.write(f"**{v.get('nickname', f'Vehicle {v['id']}')}** - {time_str}")


def show_maintenance():
    """Maintenance tracking tab"""
    st.header("🔧 Tire Maintenance Monitoring")

    vehicles = st.session_state.db.get_all_vehicles(min_encounters=3)

    if not vehicles:
        st.info("Not enough data for maintenance analysis yet.")
        return

    vehicle_options = {
        f"{v.get('nickname', f'Vehicle {v['id']}')} (ID: {v['id']})": v['id']
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

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Live Detection",
        "🚗 Vehicle Database",
        "📈 Analytics",
        "🔧 Maintenance",
        "🤖 ML Insights"
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


if __name__ == "__main__":
    main()
