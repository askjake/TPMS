"""
Sensor Search page — search by sensor ID across all exports and archives.
"""
import json
import pandas as pd
import streamlit as st

from tpms_dashboard.config import EXPORT_DIR, ARCHIVE_DIR, LARGE_FILE_THRESHOLD_MB
from tpms_dashboard.components.freshness_banner import render_freshness_banner
from tpms_dashboard.utils.data_loader import exports_fingerprint, load_daily_exports


def render():
    """Render the Sensor Search page."""
    st.header("\U0001f50e Sensor Search")
    st.caption("Search for a specific sensor ID across all exports (daily + archive)")
    render_freshness_banner()

    search_id = st.text_input(
        "Sensor ID (partial match)",
        placeholder="e.g. 00F14B3B",
        help="Enter at least 3 characters of a sensor ID to search across all data sources",
    )

    if search_id and len(search_id) >= 3:
        results = []
        search_upper = search_id.upper()

        fp = exports_fingerprint(EXPORT_DIR)
        daily_df = load_daily_exports(fp)
        if not daily_df.empty and "sensor_id" in daily_df.columns:
            matches = daily_df[daily_df["sensor_id"].str.upper().str.contains(search_upper, na=False)]
            if not matches.empty:
                for _, row in matches.iterrows():
                    results.append({**row.to_dict(), "source": "daily"})

        for directory in [EXPORT_DIR, ARCHIVE_DIR]:
            if not directory.exists():
                continue
            for fpath in sorted(directory.glob("*.json")):
                if fpath.stat().st_size / 1024 / 1024 <= LARGE_FILE_THRESHOLD_MB:
                    continue
                with open(fpath) as f:
                    data = json.load(f)
                for rec in data:
                    sid = rec.get("sensor_id", "")
                    if search_upper in sid.upper():
                        rec["source"] = fpath.name
                        results.append(rec)

        if results:
            st.success(f"Found {len(results)} match(es)")
            result_df = pd.DataFrame(results)
            display_cols = [c for c in ["sensor_id", "protocol", "packet_count",
                                        "pressure_psi", "temp_f", "rssi_dbm",
                                        "first_seen", "last_seen", "source"]
                          if c in result_df.columns]
            st.dataframe(result_df[display_cols], use_container_width=True, hide_index=True)
        else:
            st.warning(f"No sensors matching '{search_id}' found.")
    elif search_id:
        st.info("Enter at least 3 characters to search.")
