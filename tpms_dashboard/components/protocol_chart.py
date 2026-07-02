"""
Pressure distribution chart with grouped protocol legend.
Collapses 100+ Unknown(hex) entries into a single 'Unknown Protocol' bucket.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from tpms_dashboard.utils.protocol import normalize_protocol, KNOWN_PROTOCOLS, extract_hex_code


# Fixed color map for consistency
PROTOCOL_COLORS = {
    "Schrader": "#1f77b4",
    "Schrader-EV1": "#ff7f0e",
    "Schrader-Alt": "#2ca02c",
    "Unknown Protocol": "#d62728",
    "Unknown (Zero Reading)": "#9467bd",
}


def render_pressure_chart(df: pd.DataFrame):
    """Render the pressure distribution chart with grouped legend (max 4-5 entries)."""
    if "pressure_psi" not in df.columns or df["pressure_psi"].isna().all():
        st.info("No pressure data available for chart.")
        return

    chart_df = df.copy()

    # Normalize protocols for display
    if "protocol" in chart_df.columns:
        chart_df["protocol_raw"] = chart_df["protocol"]
        chart_df["protocol_display"] = chart_df["protocol"].apply(normalize_protocol)
    else:
        chart_df["protocol_display"] = "Unknown Protocol"
        chart_df["protocol_raw"] = ""

    # Determine date range for title
    date_min = ""
    date_max = ""
    if "first_seen" in chart_df.columns and chart_df["first_seen"].notna().any():
        date_min = str(chart_df["first_seen"].min().date())
    if "last_seen" in chart_df.columns and chart_df["last_seen"].notna().any():
        date_max = str(chart_df["last_seen"].max().date())
    date_range_str = f" ({date_min} to {date_max})" if date_min and date_max else ""

    # Build chart
    fig = px.histogram(
        chart_df,
        x="pressure_psi",
        color="protocol_display",
        nbins=40,
        title=f"Pressure Distribution by Protocol{date_range_str}",
        labels={"pressure_psi": "Pressure (PSI)", "count": "Sensor Count", "protocol_display": "Protocol"},
        color_discrete_map=PROTOCOL_COLORS,
        category_orders={"protocol_display": list(KNOWN_PROTOCOLS) + ["Unknown Protocol", "Unknown (Zero Reading)"]},
    )
    fig.update_layout(
        xaxis_title="Pressure (PSI)",
        yaxis_title="Sensor Count",
        legend_title="Protocol",
        bargap=0.05,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Collapsible detail: raw hex codes
    with st.expander("\U0001f50d Raw Protocol Hex Codes (for debugging)", expanded=False):
        if "protocol_raw" in chart_df.columns:
            raw_protocols = chart_df["protocol_raw"].value_counts().reset_index()
            raw_protocols.columns = ["Protocol (Raw)", "Count"]
            raw_protocols["Hex Code"] = raw_protocols["Protocol (Raw)"].apply(extract_hex_code)
            raw_protocols["Classification"] = raw_protocols["Protocol (Raw)"].apply(
                lambda p: "Known" if p in KNOWN_PROTOCOLS else "Unknown"
            )
            st.dataframe(raw_protocols, use_container_width=True, hide_index=True)
            st.caption(
                f"Total unique protocol strings: {len(raw_protocols)}. "
                f"Known: {(raw_protocols['Classification'] == 'Known').sum()}. "
                f"Unknown: {(raw_protocols['Classification'] == 'Unknown').sum()}."
            )
