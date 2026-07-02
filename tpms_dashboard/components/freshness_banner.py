"""
Data freshness indicator component.
"""
import streamlit as st
from tpms_dashboard.config import EXPORT_DIR
from tpms_dashboard.utils.formatting import get_last_modified


def render_freshness_banner():
    """Display a data freshness info banner."""
    freshness = get_last_modified(EXPORT_DIR)
    st.info(f"\U0001f504 Data freshness: {freshness}")
