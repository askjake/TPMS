"""
Sidebar navigation component with icons, descriptions, and active state.
"""
import streamlit as st
from tpms_dashboard.config import APP_NAME, APP_VERSION, APP_SUBTITLE, VIEWS
from tpms_dashboard.config import EXPORT_DISPLAY_NAME, ARCHIVE_DISPLAY_NAME


def render_sidebar() -> str:
    """Render the sidebar and return the selected view name."""
    with st.sidebar:
        st.header(f"\U0001f4e1 {APP_NAME} v{APP_VERSION}")
        st.caption(APP_SUBTITLE)
        st.divider()

        selected_view = st.radio(
            "Navigation",
            options=list(VIEWS.keys()),
            format_func=lambda k: f"{VIEWS[k]['icon']} {k}",
            label_visibility="visible",
            help="Select a dashboard view",
        )

        # Show descriptions for the selected view
        st.caption(f"*{VIEWS[selected_view]['description']}*")

        st.divider()
        st.caption(f"\U0001f4c1 {EXPORT_DISPLAY_NAME}")
        st.caption(f"\U0001f5c4\ufe0f {ARCHIVE_DISPLAY_NAME}")

    return selected_view
