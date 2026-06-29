"""
tpms_vehicle_explorer.py
========================
Streamlit dashboard for TPMS vehicle pattern analysis and learning.

Views:
  1. Live Overview    - summary stats and recent activity
  2. Sensor Timeline  - per-sensor capture history with burst context
  3. Vehicle Profiles - learned vehicle fingerprints and their sightings
  4. Burst Viewer     - drill into any burst event
  5. Map View         - geographic sighting map

Run: streamlit run tpms_vehicle_explorer.py --server.port 8570
"""

import sys, os, json, time, math, sqlite3
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── paths ────────────────────────────────────────────────────────────────────
THIS_DIR = Path(__file__).parent

# Search order: script dir, common remote mounts, home-relative paths
_SEARCH_ROOTS = [
    THIS_DIR,
    Path("/home/diship-test/TPMS"),
    Path("/home/montjac/TPMS"),
    Path.home() / "TPMS",
    Path("/opt/TPMS"),
]

def _find_db(name: str) -> str:
    """Return the first existing path for a named DB file across search roots."""
    for root in _SEARCH_ROOTS:
        p = root / name
        if p.exists():
            return str(p)
    return str(THIS_DIR / name)   # fallback (will show 'not found' in UI)

def _discover_source_dbs() -> dict:
    """Auto-discover all tpms_tracker*.db files across search roots."""
    found = {}
    seen = set()
    for root in _SEARCH_ROOTS:
        if not root.exists():
            continue
        for p in sorted(root.glob("tpms_tracker*.db")):
            if str(p) not in seen and p.stat().st_size > 1024:
                label = f"{p.name}  [{root}]"
                found[label] = str(p)
                seen.add(str(p))
    if not found:
        # Fallback entries so the selectbox always has something
        found["tracker4 (not found)"] = _find_db("tpms_tracker4.db")
        found["tracker (not found)"]  = _find_db("tpms_tracker.db")
    return found

SOURCE_DB_OPTIONS = _discover_source_dbs()
PROFILE_DB = _find_db("tpms_vehicle_profiles.db")

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TPMS Vehicle Intelligence",
    page_icon="\U0001f697",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── ensure engine is importable ──────────────────────────────────────────────
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    from tpms_vehicle_engine import TPMSVehicleProcessor
    ENGINE_AVAILABLE = True
except ImportError as e:
    ENGINE_AVAILABLE = False
    ENGINE_ERROR = str(e)

# ── helpers ──────────────────────────────────────────────────────────────────

def ts_fmt(ts):
    if ts is None or ts == 0:
        return "—"
    return datetime.utcfromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")


def pdb_query(sql, params=(), db=PROFILE_DB):
    if not os.path.exists(db):
        return []
    con = sqlite3.connect(db)
    try:
        rows = con.execute(sql, params).fetchall()
        return rows
    except Exception:
        return []
    finally:
        con.close()


def pdb_df(sql, params=(), db=PROFILE_DB) -> pd.DataFrame:
    rows = pdb_query(sql, params, db)
    if not rows:
        return pd.DataFrame()
    cur = sqlite3.connect(db).execute(sql, params)
    cols = [d[0] for d in cur.description]
    cur.close()
    return pd.DataFrame(rows, columns=cols)


def source_df(sql, params=(), source=None):
    if source is None:
        return pd.DataFrame()
    con = sqlite3.connect(source)
    try:
        cur = con.execute(sql, params)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        return pd.DataFrame(rows, columns=cols)
    except Exception:
        return pd.DataFrame()
    finally:
        con.close()


def confidence_color(conf):
    if conf is None:
        return "gray"
    if conf >= 0.7:
        return "green"
    if conf >= 0.4:
        return "orange"
    return "red"


# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("\U0001f697 TPMS Vehicle Intel")
    st.caption("Pattern recognition & learning")

    view = st.radio("View", [
        "Overview",
        "Sensor Timeline",
        "Vehicle Profiles",
        "Burst Viewer",
        "Map View",
        "Run Processor",
    ])

    st.divider()
    selected_source_name = st.selectbox("Source DB", list(SOURCE_DB_OPTIONS.keys()))
    selected_source = SOURCE_DB_OPTIONS[selected_source_name]

    # Manual path override
    with st.expander("Override path", expanded=not os.path.exists(selected_source)):
        custom_path = st.text_input("Custom DB path", value=selected_source,
                                    help="Paste full path to tpms_tracker*.db")
        if custom_path and custom_path != selected_source:
            selected_source = custom_path.strip()
        custom_profile = st.text_input("Profile DB path", value=PROFILE_DB,
                                       help="Where to store/read vehicle profiles")
        if custom_profile and custom_profile.strip() != PROFILE_DB:
            PROFILE_DB = custom_profile.strip()

    db_exists = os.path.exists(selected_source)
    profile_exists = os.path.exists(PROFILE_DB)
    st.caption(f"Source: {'✅' if db_exists else '❌ NOT FOUND'} {Path(selected_source).name}")
    st.caption(f"Profiles: {'✅' if profile_exists else '⚠ Not run yet'}")
    if not db_exists:
        st.warning(f"DB not found:\n`{selected_source}`\nUse override above.")


# ════════════════════════════════════════════════════════════════════════════
# VIEW: OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if view == "Overview":
    st.header("\U0001f4ca Vehicle Intelligence Overview")

    # Engine stats
    col1, col2 = st.columns([2, 1])

    with col1:
        if profile_exists:
            st.subheader("Learning Progress")
            stats_row = pdb_query("SELECT COUNT(*) FROM burst_events")[0][0] if profile_exists else 0
            stats = {
                "Bursts Analyzed":    pdb_query("SELECT COUNT(*) FROM burst_events"),
                "Vehicle Clusters":   pdb_query("SELECT COUNT(*) FROM vehicle_clusters"),
                "Vehicle Profiles":   pdb_query("SELECT COUNT(*) FROM vehicle_profiles"),
                "Confirmed (3+)":     pdb_query("SELECT COUNT(*) FROM vehicle_profiles WHERE encounter_count>=3"),
                "Processing Runs":    pdb_query("SELECT COUNT(*) FROM processing_log"),
            }
            ks = st.columns(5)
            labels = list(stats.keys())
            for i, (label, val) in enumerate(stats.items()):
                v = val[0][0] if val else 0
                ks[i].metric(label, f"{v:,}")

            # Method breakdown
            method_rows = pdb_query("SELECT method, COUNT(*), AVG(confidence) FROM vehicle_clusters GROUP BY method ORDER BY COUNT(*) DESC")
            if method_rows:
                st.markdown("### Clustering Methods")
                mdf = pd.DataFrame(method_rows, columns=["Method", "Clusters", "Avg Confidence"])
                mdf["Avg Confidence"] = mdf["Avg Confidence"].round(3)
                fig = px.bar(mdf, x="Method", y="Clusters", color="Avg Confidence",
                             color_continuous_scale="Viridis",
                             title="Clusters by Detection Method")
                st.plotly_chart(fig, use_container_width=True)

            # Processing history
            log_df = pdb_df("SELECT processed_at, signals_processed, bursts_detected, clusters_found, new_profiles, known_matches, elapsed_s FROM processing_log ORDER BY processed_at DESC LIMIT 10")
            if not log_df.empty:
                st.markdown("### Processing History")
                log_df["processed_at"] = log_df["processed_at"].apply(ts_fmt)
                st.dataframe(log_df, use_container_width=True, hide_index=True)

        else:
            st.info("\U0001f4dd No profile database yet. Go to **Run Processor** to analyze the data.")

    with col2:
        st.subheader("\U0001f50e Quick Stats")
        if db_exists:
            sig_count = source_df("SELECT COUNT(*), COUNT(DISTINCT tpms_id), MIN(timestamp), MAX(timestamp) FROM tpms_signals WHERE timestamp > 946684800", source=selected_source)
            if not sig_count.empty:
                row = sig_count.iloc[0]
                st.metric("Total Signals", f"{int(row[0]):,}")
                st.metric("Unique Sensors", f"{int(row[1]):,}")
                st.metric("First Seen", ts_fmt(row[2])[:10])
                st.metric("Last Seen", ts_fmt(row[3])[:10])
                days = (row[3] - row[2]) / 86400 if row[2] and row[3] else 0
                st.metric("Span", f"{days:.1f} days")

        if profile_exists:
            top_v = pdb_query("SELECT nickname, encounter_count, avg_confidence FROM vehicle_profiles ORDER BY encounter_count DESC LIMIT 5")
            if top_v:
                st.markdown("**Top Recurring Vehicles**")
                for r in top_v:
                    color = confidence_color(r[2])
                    st.markdown(f"- `{r[0]}` — {r[1]} sightings")

    # Encounter count distribution
    if profile_exists:
        enc_dist = pdb_query("SELECT encounter_count, COUNT(*) FROM vehicle_profiles GROUP BY encounter_count ORDER BY encounter_count")
        if enc_dist:
            st.markdown("### Vehicle Encounter Distribution")
            edf = pd.DataFrame(enc_dist, columns=["Encounters", "Vehicles"])
            fig2 = px.bar(edf, x="Encounters", y="Vehicles",
                         title="How many times has each vehicle been seen?",
                         labels={"Encounters": "Times Seen", "Vehicles": "# Vehicle Profiles"})
            st.plotly_chart(fig2, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# VIEW: SENSOR TIMELINE
# ════════════════════════════════════════════════════════════════════════════
elif view == "Sensor Timeline":
    st.header("\U0001f4f6 Sensor Capture Timeline")
    st.caption("See when each sensor was captured and what burst it belonged to")

    if not db_exists:
        st.error("Source database not found.")
        st.stop()

    tab1, tab2 = st.tabs(["Search Sensor", "Burst Timeline"])

    with tab1:
        sensor_id = st.text_input("Sensor ID (partial or full)", placeholder="e.g. A89B2FC2")
        if sensor_id and len(sensor_id) >= 4:
            sig_df = source_df(
                "SELECT timestamp, tpms_id, protocol, pressure_psi, temperature_c, "
                "signal_strength, battery_low, latitude, longitude "
                "FROM tpms_signals WHERE tpms_id LIKE ? AND timestamp > 946684800 "
                "ORDER BY timestamp",
                [f"%{sensor_id.upper()}%"], source=selected_source
            )
            if sig_df.empty:
                st.warning("No signals found for that sensor ID.")
            else:
                st.success(f"Found {len(sig_df)} signal(s) for sensors matching '{sensor_id}'")
                sig_df["datetime"] = sig_df["timestamp"].apply(ts_fmt)
                display_cols = [c for c in ["datetime","tpms_id","protocol","pressure_psi",
                                            "temperature_c","signal_strength","battery_low",
                                            "latitude","longitude"] if c in sig_df.columns]
                st.dataframe(sig_df[display_cols], use_container_width=True, hide_index=True)

                if len(sig_df) > 1 and "pressure_psi" in sig_df.columns:
                    fig = px.scatter(sig_df, x="datetime", y="pressure_psi", color="tpms_id",
                                   title="Pressure Over Time", size_max=10)
                    st.plotly_chart(fig, use_container_width=True)

                # Show what vehicle profile this sensor belongs to (if any)
                if profile_exists:
                    for _, row in sig_df.iterrows():
                        sid = row["tpms_id"]
                        vh_rows = pdb_query(
                            "SELECT vehicle_hash, nickname, encounter_count, psi_fingerprint "
                            "FROM vehicle_profiles WHERE sensor_ids LIKE ?",
                            [f"%{sid}%"]
                        )
                        if vh_rows:
                            st.info(f"\U0001f697 Sensor `{sid}` is linked to vehicle profile: "
                                   f"**{vh_rows[0][1]}** (seen {vh_rows[0][2]}x, "
                                   f"PSI={json.loads(vh_rows[0][3])})")

    with tab2:
        st.markdown("### Burst Timeline")
        st.caption("Each row is a burst event — a group of sensors captured together")

        burst_df = pdb_df(
            "SELECT burst_id, start_ts, end_ts, duration_s, center_lat, center_lon, "
            "signal_count, pressured_count, clusters_found FROM burst_events ORDER BY start_ts DESC LIMIT 200"
        )
        if burst_df.empty:
            st.info("Run the processor first to populate burst data.")
        else:
            burst_df["start_time"] = burst_df["start_ts"].apply(ts_fmt)
            burst_df["end_time"] = burst_df["end_ts"].apply(ts_fmt)
            burst_df["duration_s"] = burst_df["duration_s"].round(1)
            burst_df["coverage_pct"] = (burst_df["clusters_found"] * 4 / burst_df["pressured_count"].clip(lower=1) * 100).round(1)
            display = ["start_time","duration_s","signal_count","pressured_count","clusters_found","coverage_pct","center_lat","center_lon"]
            st.dataframe(burst_df[display], use_container_width=True, hide_index=True)

            # Signal count over time
            burst_df["date"] = pd.to_datetime(burst_df["start_ts"], unit="s").dt.date
            daily = burst_df.groupby("date").agg(
                total_signals=("signal_count","sum"),
                bursts=("burst_id","count"),
                clusters=("clusters_found","sum")
            ).reset_index()
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=daily["date"], y=daily["total_signals"], name="Signals"), secondary_y=False)
            fig.add_trace(go.Scatter(x=daily["date"], y=daily["clusters"], name="Clusters", mode="lines+markers"), secondary_y=True)
            fig.update_layout(title="Daily Signal Volume and Vehicle Clusters Found")
            st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# VIEW: VEHICLE PROFILES
# ════════════════════════════════════════════════════════════════════════════
elif view == "Vehicle Profiles":
    st.header("\U0001f697 Learned Vehicle Profiles")

    if not profile_exists:
        st.info("No profiles yet. Run the processor first.")
        st.stop()

    filt_col, sort_col, min_enc_col = st.columns(3)
    with min_enc_col:
        min_enc = st.number_input("Min encounters", min_value=1, max_value=20, value=2)
    with sort_col:
        sort_by = st.selectbox("Sort by", ["encounter_count", "avg_confidence", "last_seen", "first_seen"])
    with filt_col:
        search_vh = st.text_input("Search by nickname or hash", "")

    profiles_df = pdb_df(
        f"SELECT vehicle_hash, nickname, psi_fingerprint, psi_tolerance, encounter_count, "
        f"avg_confidence, first_seen, last_seen, sensor_ids, protocol, encounter_lats, encounter_lons "
        f"FROM vehicle_profiles WHERE encounter_count >= ? ORDER BY {sort_by} DESC",
        [min_enc]
    )

    if profiles_df.empty:
        st.warning(f"No profiles with {min_enc}+ encounters yet.")
        st.stop()

    if search_vh:
        profiles_df = profiles_df[
            profiles_df["nickname"].str.contains(search_vh, case=False, na=False) |
            profiles_df["vehicle_hash"].str.contains(search_vh, case=False, na=False)
        ]

    st.metric("Profiles shown", len(profiles_df))

    # Table view
    disp = profiles_df.copy()
    disp["first_seen"] = disp["first_seen"].apply(ts_fmt)
    disp["last_seen"]  = disp["last_seen"].apply(ts_fmt)
    disp["psi_fingerprint"] = disp["psi_fingerprint"].apply(
        lambda x: str(json.loads(x)) if x else "")
    disp["sensor_count"] = disp["sensor_ids"].apply(
        lambda x: len(json.loads(x)) if x else 0)
    disp["avg_confidence"] = disp["avg_confidence"].round(3)

    show_cols = ["nickname","psi_fingerprint","encounter_count","avg_confidence",
                 "sensor_count","first_seen","last_seen","protocol"]
    st.dataframe(disp[show_cols], use_container_width=True, hide_index=True, height=300)

    # Drill-down
    st.markdown("### Drill Down")
    selected_nick = st.selectbox("Select vehicle to inspect",
                                  ["—"] + disp["nickname"].tolist())
    if selected_nick != "—":
        row = profiles_df[profiles_df["nickname"] == selected_nick].iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Encounters", int(row["encounter_count"]))
        c2.metric("Confidence", f"{row['avg_confidence']:.2f}")
        c3.metric("Unique Sensors Seen", len(json.loads(row["sensor_ids"])))

        fp = json.loads(row["psi_fingerprint"])
        st.markdown(f"**PSI Fingerprint:** `{fp}`")

        # PSI bar chart
        fig_psi = go.Figure(go.Bar(
            x=[f"Tire {i+1}" for i in range(len(fp))],
            y=fp,
            marker_color=["#636EFA","#EF553B","#00CC96","#AB63FA","#FFA15A"][:len(fp)],
            text=[f"{p} PSI" for p in fp],
            textposition="outside"
        ))
        fig_psi.update_layout(title=f"PSI Fingerprint — {selected_nick}",
                              yaxis_title="Pressure (PSI)", height=300)
        st.plotly_chart(fig_psi, use_container_width=True)

        # Sensor list
        sensor_ids = json.loads(row["sensor_ids"])
        st.markdown(f"**All sensor IDs seen ({len(sensor_ids)}):**")
        sensor_cols = st.columns(4)
        for i, sid in enumerate(sensor_ids):
            sensor_cols[i % 4].code(sid)

        # Sighting history from clusters
        sightings = pdb_df(
            "SELECT timestamp, burst_id, psi_values, confidence, method, latitude, longitude "
            "FROM vehicle_clusters WHERE vehicle_hash=? ORDER BY timestamp",
            [row["vehicle_hash"]]
        )
        if not sightings.empty:
            sightings["datetime"] = sightings["timestamp"].apply(ts_fmt)
            st.markdown("### Sighting History")
            disp_s = sightings[["datetime","method","confidence","latitude","longitude"]].copy()
            disp_s["confidence"] = disp_s["confidence"].round(3)
            st.dataframe(disp_s, use_container_width=True, hide_index=True)

            # Confidence trend
            fig_conf = px.line(sightings, x="datetime", y="confidence", markers=True,
                              title=f"Confidence Over Time — {selected_nick}")
            st.plotly_chart(fig_conf, use_container_width=True)

            # PSI stability: show each sighting's PSI values
            psi_records = []
            for _, sr in sightings.iterrows():
                try:
                    psi_list = json.loads(sr["psi_values"])
                    for j, p in enumerate(psi_list):
                        psi_records.append({
                            "datetime": sr["datetime"],
                            "tire": f"Tire {j+1}",
                            "pressure_psi": p
                        })
                except Exception:
                    pass
            if psi_records:
                psi_df = pd.DataFrame(psi_records)
                fig_psi_trend = px.line(psi_df, x="datetime", y="pressure_psi",
                                       color="tire", markers=True,
                                       title=f"Tire Pressure Stability — {selected_nick}")
                st.plotly_chart(fig_psi_trend, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# VIEW: BURST VIEWER
# ════════════════════════════════════════════════════════════════════════════
elif view == "Burst Viewer":
    st.header("\U0001f4a5 Burst Event Viewer")
    st.caption("Inspect raw burst events and which vehicle clusters were found within them")

    if not profile_exists:
        st.info("No bursts analyzed yet. Run the processor first.")
        st.stop()

    burst_list = pdb_query(
        "SELECT burst_id, start_ts, signal_count, pressured_count, clusters_found "
        "FROM burst_events ORDER BY start_ts DESC LIMIT 100"
    )
    if not burst_list:
        st.warning("No bursts in database.")
        st.stop()

    burst_labels = [f"{ts_fmt(r[1])} — {r[2]} signals, {r[4]} clusters" for r in burst_list]
    sel_idx = st.selectbox("Select burst event", range(len(burst_labels)), format_func=lambda i: burst_labels[i])
    sel_burst = burst_list[sel_idx]
    burst_id = sel_burst[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Signals", sel_burst[2])
    c2.metric("With Pressure", sel_burst[3])
    c3.metric("Clusters Found", sel_burst[4])
    coverage = f"{sel_burst[4]*4/max(sel_burst[3],1)*100:.0f}%" if sel_burst[3] else "—"
    c4.metric("Coverage", coverage)

    # Clusters in this burst
    clusters_df = pdb_df(
        "SELECT cluster_id, sensor_ids, psi_fingerprint, psi_values, confidence, "
        "method, vehicle_hash FROM vehicle_clusters WHERE burst_id=? ORDER BY confidence DESC",
        [burst_id]
    )

    if clusters_df.empty:
        st.info("No vehicle clusters found in this burst.")
    else:
        st.markdown(f"### {len(clusters_df)} Vehicle Clusters")
        for _, crow in clusters_df.iterrows():
            fp = json.loads(crow["psi_fingerprint"])
            psi_vals = json.loads(crow["psi_values"])
            sids = json.loads(crow["sensor_ids"])
            vh = crow["vehicle_hash"] or "unlinked"
            conf = crow["confidence"]

            # Look up nickname
            vh_name = ""
            if vh and vh != "unlinked":
                vh_rows = pdb_query("SELECT nickname, encounter_count FROM vehicle_profiles WHERE vehicle_hash=?", [vh])
                if vh_rows:
                    vh_name = f"{vh_rows[0][0]} (seen {vh_rows[0][1]}x)"

            with st.expander(
                f"Cluster {crow['cluster_id'][:8]} | {len(sids)} sensors | "
                f"conf={conf:.2f} | method={crow['method']} | {vh_name or vh[:8]}"
            ):
                psi_col, sid_col = st.columns(2)
                with psi_col:
                    fig = go.Figure(go.Bar(
                        x=[f"T{i+1}" for i in range(len(fp))],
                        y=fp,
                        text=[f"{p}" for p in fp],
                        textposition="outside",
                        marker_color="#636EFA"
                    ))
                    fig.update_layout(height=200, margin=dict(l=10,r=10,t=10,b=10),
                                      yaxis_title="PSI")
                    st.plotly_chart(fig, use_container_width=True)
                with sid_col:
                    st.markdown("**Sensor IDs:**")
                    for sid in sids:
                        st.code(sid, language=None)

    # Also show raw signals in this burst from the source DB
    if db_exists:
        burst_info = pdb_query("SELECT start_ts, end_ts, center_lat, center_lon FROM burst_events WHERE burst_id=?", [burst_id])
        if burst_info:
            binfo = burst_info[0]
            raw_df = source_df(
                "SELECT timestamp, tpms_id, protocol, pressure_psi, temperature_c, "
                "signal_strength, battery_low FROM tpms_signals "
                "WHERE timestamp BETWEEN ? AND ? ORDER BY pressure_psi",
                [binfo[0] - 1, binfo[1] + 1], source=selected_source
            )
            if not raw_df.empty:
                st.markdown("### Raw Signals in Burst (sorted by PSI)")
                raw_df["timestamp"] = raw_df["timestamp"].apply(lambda t: f"+{t-binfo[0]:.1f}s")
                raw_df["signal_strength"] = raw_df["signal_strength"].round(1) if raw_df["signal_strength"].dtype == float else raw_df["signal_strength"]

                # Highlight which sensors are in clusters
                clustered_sensors = set()
                if not clusters_df.empty:
                    for sids_json in clusters_df["sensor_ids"]:
                        clustered_sensors.update(json.loads(sids_json))

                raw_df["in_cluster"] = raw_df["tpms_id"].isin(clustered_sensors)

                def highlight_clustered(row):
                    color = "background-color: #e6ffe6" if row["in_cluster"] else ""
                    return [color] * len(row)

                st.dataframe(
                    raw_df.style.apply(highlight_clustered, axis=1),
                    use_container_width=True, hide_index=True, height=350
                )

                # PSI histogram of this burst
                if "pressure_psi" in raw_df.columns:
                    fig_h = px.histogram(raw_df, x="pressure_psi", nbins=30,
                                        color="in_cluster",
                                        title="PSI Distribution in Burst (green = in a cluster)",
                                        color_discrete_map={True: "green", False: "#aaa"})
                    st.plotly_chart(fig_h, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# VIEW: MAP VIEW
# ════════════════════════════════════════════════════════════════════════════
elif view == "Map View":
    st.header("\U0001f5fa Geographic Sighting Map")

    if not profile_exists:
        st.info("Run the processor first to generate sighting data.")
        st.stop()

    # Vehicle sightings as lat/lon points
    sighting_rows = pdb_query(
        "SELECT vc.latitude, vc.longitude, vc.timestamp, vc.confidence, "
        "vp.nickname, vp.encounter_count, vc.method "
        "FROM vehicle_clusters vc "
        "LEFT JOIN vehicle_profiles vp ON vc.vehicle_hash = vp.vehicle_hash "
        "WHERE vc.latitude IS NOT NULL AND vc.longitude IS NOT NULL"
    )

    if not sighting_rows:
        st.warning("No GPS data available in clusters.")
        # Fall back to burst centroids
        burst_rows = pdb_query(
            "SELECT center_lat, center_lon, start_ts, signal_count FROM burst_events "
            "WHERE center_lat IS NOT NULL AND center_lon IS NOT NULL"
        )
        if burst_rows:
            bdf = pd.DataFrame(burst_rows, columns=["lat","lon","timestamp","signal_count"])
            bdf["datetime"] = bdf["timestamp"].apply(ts_fmt)
            fig_map = px.scatter_mapbox(bdf, lat="lat", lon="lon", size="signal_count",
                                        hover_name="datetime", hover_data=["signal_count"],
                                        title="Burst Event Locations",
                                        mapbox_style="open-street-map", zoom=11)
            st.plotly_chart(fig_map, use_container_width=True)
        st.stop()

    sdf = pd.DataFrame(sighting_rows, columns=["lat","lon","timestamp","confidence","nickname","encounters","method"])
    sdf["datetime"] = sdf["timestamp"].apply(ts_fmt)
    sdf["nickname"] = sdf["nickname"].fillna("Unknown")
    sdf["confidence"] = sdf["confidence"].round(2)

    # Filter
    min_enc_map = st.slider("Min encounters for vehicle to show", 1, 20, 2)
    sdf_filt = sdf[sdf["encounters"] >= min_enc_map].copy() if "encounters" in sdf.columns else sdf

    if sdf_filt.empty:
        st.warning("No sightings match the filter.")
        st.stop()

    fig_map = px.scatter_mapbox(
        sdf_filt, lat="lat", lon="lon",
        color="nickname", size="confidence",
        hover_name="nickname",
        hover_data=["datetime","confidence","method","encounters"],
        title=f"Vehicle Sightings ({len(sdf_filt)} points, {sdf_filt['nickname'].nunique()} vehicles)",
        mapbox_style="open-street-map",
        zoom=12, height=600
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Heatmap of all sightings
    st.markdown("### Density Heatmap")
    fig_heat = px.density_mapbox(
        sdf, lat="lat", lon="lon", radius=20,
        title="Signal Density Map",
        mapbox_style="open-street-map", zoom=12, height=500
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# VIEW: RUN PROCESSOR
# ════════════════════════════════════════════════════════════════════════════
elif view == "Run Processor":
    st.header("\U0001f504 Run Vehicle Processor")
    st.caption("Process signals from the source DB and update the vehicle profile database.")

    if not ENGINE_AVAILABLE:
        st.error(f"Engine not available: {ENGINE_ERROR if 'ENGINE_ERROR' in dir() else 'import failed'}")
        st.stop()

    if not db_exists:
        st.error(f"Source database not found: {selected_source}")
        st.stop()

    c1, c2 = st.columns(2)
    with c1:
        limit = st.number_input("Max signals to process", 1000, 200000, 50000, step=5000)
        since_ts = st.number_input("Since timestamp (0 = all)", 0, int(time.time()), 946684800)
    with c2:
        reset_profiles = st.checkbox("Reset profile DB before run", False)
        st.caption("⚠ Reset clears ALL learned vehicle profiles")

    if st.button("\U0001f680 Run Processor", type="primary"):
        if reset_profiles and os.path.exists(PROFILE_DB):
            os.remove(PROFILE_DB)
            st.warning("Profile DB reset.")

        proc = TPMSVehicleProcessor(selected_source, PROFILE_DB)

        progress = st.progress(0, text="Initializing...")
        log_area = st.empty()

        with st.spinner("Processing..."):
            result = proc.process(since_ts=since_ts, limit=limit, verbose=False)
            progress.progress(100, text="Done!")

        st.success("Processing complete!")
        for k, v in result.items():
            if isinstance(v, dict):
                st.markdown(f"**{k}:**")
                for kk, vv in v.items():
                    st.write(f"  - {kk}: {vv}")
            else:
                st.write(f"**{k}:** {v}")

        st.balloons()

    # Show current profile DB summary
    if profile_exists:
        st.markdown("### Current Profile DB Status")
        row = pdb_query("SELECT COUNT(*), AVG(encounter_count), MAX(encounter_count) FROM vehicle_profiles")
        if row and row[0][0]:
            st.write(f"- Profiles: {row[0][0]:,}")
            st.write(f"- Avg encounters: {row[0][1]:.1f}")
            st.write(f"- Max encounters: {row[0][2]}")
