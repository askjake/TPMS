"""
TPMS Dashboard v2.1.0 — BLE/RF Sensor Fleet Monitor
=====================================================
Refactored entry point. Addresses all usability audit findings:
  - No raw filesystem paths exposed in UI
  - Protocol chart legend collapsed (max 4-5 entries)
  - Navigation with icons, descriptions, active state
  - KPI row with context ratios and freshness
  - All buttons/inputs labeled for accessibility
  - Descriptive subtitle, consistent versioning

Run:
    streamlit run tpms_dashboard/app.py --server.address 0.0.0.0 --server.port 8569
"""
import sys
from pathlib import Path

# Ensure the parent directory is on sys.path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from tpms_dashboard.config import APP_NAME, APP_VERSION, APP_SUBTITLE, PAGE_ICON
from tpms_dashboard.components.sidebar import render_sidebar


# --- PAGE SETUP ---
st.set_page_config(
    page_title=f"{APP_NAME} v{APP_VERSION}",
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- ROUTING ---
selected_view = render_sidebar()

if selected_view == "Daily Sync Overview":
    from tpms_dashboard.pages.daily_overview import render
    render()
elif selected_view == "Archive Explorer":
    from tpms_dashboard.pages.archive_explorer import render
    render()
elif selected_view == "Sensor History":
    from tpms_dashboard.pages.sensor_history import render
    render()
elif selected_view == "Observation Patterns":
    from tpms_dashboard.pages.observation_patterns import render
    render()
elif selected_view == "Sensor Search":
    from tpms_dashboard.pages.sensor_search import render
    render()
elif selected_view == "Trends & Charts":
    from tpms_dashboard.pages.trends import render
    render()
