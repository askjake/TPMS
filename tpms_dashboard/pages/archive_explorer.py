"""
Archive Explorer page — browse large tracker exports with pagination.
"""
import math
import pandas as pd
import streamlit as st

from tpms_dashboard.config import EXPORT_DIR, ARCHIVE_DIR, LARGE_FILE_THRESHOLD_MB
from tpms_dashboard.components.freshness_banner import render_freshness_banner
from tpms_dashboard.utils.data_loader import get_large_file_summary, load_large_file_page


def render():
    """Render the Archive Explorer page."""
    st.header("\U0001f5c4\ufe0f Archive Explorer")
    st.caption("Browse large tracker database exports with pagination \u2014 no browser hang")

    render_freshness_banner()

    all_files = []
    for directory in [EXPORT_DIR, ARCHIVE_DIR]:
        if directory.exists():
            for fpath in sorted(directory.glob("*.json")):
                size_mb = fpath.stat().st_size / 1024 / 1024
                if size_mb > LARGE_FILE_THRESHOLD_MB:
                    all_files.append({"path": str(fpath), "name": fpath.name, "size_mb": size_mb})

    if not all_files:
        st.info("No large archive files found. Large exports (>5 MB) appear here automatically.")
        st.stop()

    file_options = [f"{f['name']} ({f['size_mb']:.1f} MB)" for f in all_files]
    selected_idx = st.selectbox(
        "Archive file",
        range(len(file_options)),
        format_func=lambda i: file_options[i],
        help="Select a large archive file to browse",
    )
    selected_file = all_files[selected_idx]["path"]

    st.markdown("### Summary")
    summary = get_large_file_summary(selected_file)
    s_cols = st.columns(4)
    s_cols[0].metric("Total Sensors", f"{summary['total_sensors']:,}")
    s_cols[1].metric("File Size", f"{summary['size_mb']:.1f} MB")
    s_cols[2].metric("Protocols", len(summary['protocols']))
    if summary['pressure_stats']['mean']:
        s_cols[3].metric("Avg Pressure", f"{summary['pressure_stats']['mean']:.1f} PSI")

    st.markdown("### Protocols")
    proto_df = pd.DataFrame([
        {"Protocol": k, "Count": v, "Pct": f"{v/summary['total_sensors']*100:.1f}%"}
        for k, v in sorted(summary['protocols'].items(), key=lambda x: -x[1])
    ])
    st.dataframe(proto_df, use_container_width=True, hide_index=True)

    st.markdown("### Sensor Browser (Paginated)")
    filter_col, sort_col, page_col = st.columns(3)
    with filter_col:
        proto_options = ["All"] + sorted(summary['protocols'].keys())
        proto_filter = st.selectbox("Protocol filter", proto_options, help="Filter by protocol type")
    with sort_col:
        sort_by = st.selectbox("Sort by", ["packet_count", "pressure_psi", "sensor_id", "first_seen"],
                               help="Column to sort results by")
    with page_col:
        page_size = st.selectbox("Per page", [50, 100, 250, 500], index=1,
                                 help="Number of records per page")

    proto_arg = proto_filter if proto_filter != "All" else None
    page_df, total_filtered = load_large_file_page(
        selected_file, offset=0, limit=page_size,
        sort_by=sort_by, ascending=(sort_by == "sensor_id"),
        protocol_filter=proto_arg
    )
    total_pages = max(1, math.ceil(total_filtered / page_size))
    nav_cols = st.columns([1, 3, 1])
    nav_cols[0].metric("Filtered", f"{total_filtered:,}")
    current_page = nav_cols[1].number_input("Page", min_value=1, max_value=total_pages, value=1,
                                            help="Navigate between pages")
    nav_cols[2].metric("Pages", total_pages)

    offset = (current_page - 1) * page_size
    if offset > 0:
        page_df, _ = load_large_file_page(
            selected_file, offset=offset, limit=page_size,
            sort_by=sort_by, ascending=(sort_by == "sensor_id"),
            protocol_filter=proto_arg
        )
    if not page_df.empty:
        display_cols = [c for c in ["sensor_id", "protocol", "packet_count", "pressure_psi",
                                    "pressure_kpa", "temp_c", "temp_f", "rssi_dbm",
                                    "first_seen", "last_seen", "battery_low"]
                       if c in page_df.columns]
        st.dataframe(page_df[display_cols], use_container_width=True, hide_index=True, height=450)
