# tpms_db_explorer.py
# Streamlit app for exploring tpms_tracker SQLite databases.
# Run:  streamlit run tpms_db_explorer.py
#
# Optional:
#   export TPMS_DB_PATH=/path/to/tpms_tracker.db

from __future__ import annotations

import io
import json
import os
import sqlite3
import struct
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import streamlit as st

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

# Plotly is nicer for interactive exploration; fall back gracefully if unavailable.
try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    px = None  # type: ignore
    go = None  # type: ignore


APP_TZ = os.environ.get("TPMS_APP_TZ", "America/Denver")


# ----------------------------- Helpers -----------------------------

def _connect_ro(path: str) -> sqlite3.Connection:
    """Read-only sqlite connection (best-effort)."""
    # mode=ro works for file paths (not for :memory:). If it fails, fall back to regular open.
    try:
        return sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    except Exception:
        return sqlite3.connect(path, check_same_thread=False)


def _list_tables(con: sqlite3.Connection) -> List[str]:
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view') ORDER BY name;")
    return [r[0] for r in cur.fetchall()]


def _table_info(con: sqlite3.Connection, table: str) -> pd.DataFrame:
    df = pd.read_sql_query(f"PRAGMA table_info({table});", con)
    # columns: cid, name, type, notnull, dflt_value, pk
    return df


def _foreign_keys(con: sqlite3.Connection, table: str) -> pd.DataFrame:
    return pd.read_sql_query(f"PRAGMA foreign_key_list({table});", con)


def _row_count(con: sqlite3.Connection, table: str) -> int:
    try:
        cur = con.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {table};")
        return int(cur.fetchone()[0])
    except Exception:
        return 0


def _decode_float32_blob(x: Any) -> Optional[float]:
    """SQLite can store BLOBs in REAL columns. Your DB often stores float32-packed bytes."""
    if x is None:
        return None
    if isinstance(x, (bytes, bytearray)) and len(x) == 4:
        try:
            return struct.unpack("<f", x)[0]
        except Exception:
            return None
    # If it is already numeric (or string numeric), try casting.
    try:
        return float(x)
    except Exception:
        return None


def _bytes_to_hex(x: Any) -> Any:
    if isinstance(x, (bytes, bytearray)):
        return x.hex()
    return x


def _maybe_parse_json_array(s: Any) -> List[str]:
    if s is None:
        return []
    if isinstance(s, (list, tuple)):
        return list(map(str, s))
    if isinstance(s, str) and s.strip().startswith("["):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return [str(v) for v in arr]
        except Exception:
            return []
    return []


def _to_local_dt(series_seconds: pd.Series) -> pd.Series:
    # Treat as unix epoch seconds.
    s = pd.to_datetime(series_seconds, unit="s", utc=True, errors="coerce")
    if ZoneInfo is None:
        return s.dt.tz_localize(None)
    try:
        return s.dt.tz_convert(ZoneInfo(APP_TZ))
    except Exception:
        return s.dt.tz_convert("UTC")


def _is_blob_float_col(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return False
    sample = df[col].dropna().head(20)
    if sample.empty:
        return False
    # Count 4-byte blobs
    blob4 = sample.apply(lambda v: isinstance(v, (bytes, bytearray)) and len(v) == 4).sum()
    return blob4 >= max(1, int(0.5 * len(sample)))


@st.cache_data(show_spinner=False)
def load_table(path: str, table: str, limit: Optional[int] = None) -> pd.DataFrame:
    con = _connect_ro(path)
    q = f"SELECT * FROM {table}"
    if limit is not None:
        q += f" LIMIT {int(limit)}"
    df = pd.read_sql_query(q + ";", con)
    con.close()
    return df


@st.cache_data(show_spinner=False)
def load_signals_clean(path: str) -> pd.DataFrame:
    df = load_table(path, "tpms_signals", limit=None).copy()

    # decode float-packed blobs
    for c in ["signal_strength", "snr"]:
        if c in df.columns and _is_blob_float_col(df, c):
            df[c + "_decoded"] = df[c].apply(_decode_float32_blob)
        elif c in df.columns:
            df[c + "_decoded"] = df[c].apply(_decode_float32_blob)

    # convert timestamps
    if "timestamp" in df.columns:
        df["timestamp_local"] = _to_local_dt(df["timestamp"])

    # frequency convenience
    if "frequency" in df.columns:
        df["frequency_mhz"] = pd.to_numeric(df["frequency"], errors="coerce") / 1e6

    # hex-ify raw payload
    if "raw_data" in df.columns:
        df["raw_data_hex"] = df["raw_data"].apply(_bytes_to_hex)

    # "temperature_c" appears to be Fahrenheit in this dataset; offer both.
    if "temperature_c" in df.columns:
        t = pd.to_numeric(df["temperature_c"], errors="coerce")
        df["temp_f"] = t
        df["temp_c_from_f"] = (t - 32.0) * (5.0 / 9.0)

    return df


@st.cache_data(show_spinner=False)
def load_vehicles_clean(path: str) -> pd.DataFrame:
    df = load_table(path, "vehicles", limit=None).copy()
    if "tpms_ids" in df.columns:
        df["tpms_ids_list"] = df["tpms_ids"].apply(_maybe_parse_json_array)
        df["tpms_count"] = df["tpms_ids_list"].apply(len)
    if "first_seen" in df.columns:
        df["first_seen_local"] = _to_local_dt(df["first_seen"])
    if "last_seen" in df.columns:
        df["last_seen_local"] = _to_local_dt(df["last_seen"])
    return df


@st.cache_data(show_spinner=False)
def load_encounters_clean(path: str) -> pd.DataFrame:
    df = load_table(path, "encounters", limit=None).copy()
    if "timestamp" in df.columns:
        df["timestamp_local"] = _to_local_dt(df["timestamp"])
    return df


# ----------------------------- UI -----------------------------

st.set_page_config(page_title="TPMS DB Explorer", layout="wide")
st.title("TPMS Tracker — DB Explorer")

with st.sidebar:
    st.header("Database")
    default_path = os.environ.get("TPMS_DB_PATH", "tpms_tracker.db")
    db_path = st.text_input("SQLite DB path", value=default_path, help="Set TPMS_DB_PATH env var to avoid typing this every time.")
    upload = st.file_uploader("…or upload a .db file", type=["db", "sqlite", "bak", "bak1"])
    if upload is not None:
        # Save uploaded db into a temp file in the Streamlit working directory
        uploaded_path = os.path.join(st.session_state.get("_upload_dir", "."), f"_uploaded_{upload.name}")
        with open(uploaded_path, "wb") as f:
            f.write(upload.getbuffer())
        db_path = uploaded_path
        st.success(f"Loaded uploaded DB: {upload.name}")

    st.divider()
    page = st.radio(
        "View",
        ["How your app stores data", "DB overview", "Table explorer", "Signals dashboard", "Vehicles & encounters", "Quality checks", "Export / clean"],
        index=0,
    )

# Validate DB early
if not db_path or not os.path.exists(db_path):
    st.warning("Point me at a SQLite DB file (left sidebar).")
    st.stop()

try:
    con = _connect_ro(db_path)
    tables = _list_tables(con)
except Exception as e:
    st.error(f"Could not open DB: {e}")
    st.stop()


def section_how_data_is_stored():
    st.subheader("What your app is doing with the data (inferred from the DB)")

    st.markdown(
        """
This database has **four app tables** (plus SQLite internals):

- **`tpms_signals`** — raw TPMS transmissions (one row per decoded RF message).
  - Keeps the *sensor id*, *time*, optional *location*, *RF metrics* (signal strength, SNR), *decoded payload* (pressure/temperature/battery), and the *raw payload bytes*.
- **`vehicles`** — a **derived** entity: your app appears to **cluster multiple TPMS sensor IDs into a “vehicle”**.
  - `tpms_ids` is stored as a **JSON array** of sensor IDs.
  - `vehicle_hash` looks like a stable signature for the cluster.
  - `encounter_count`, `first_seen`, `last_seen` track how often and when the cluster was observed.
- **`encounters`** — an event log of each time a vehicle-cluster was “seen”.
  - References `vehicles.id` (foreign key).
  - Has optional fields for lat/lon, duration, signal_quality (currently often null).
- **`maintenance_history`** — looks like it’s meant to store **alerts / trends** (pressure variance, min/max, alert_type) per vehicle + TPMS sensor,
  but it’s empty in this DB snapshot.

**Important quirk (and why dashboards might look “weird”):**
SQLite is dynamically typed, and this DB stores `signal_strength` and `snr` values as **4‑byte float blobs** (packed float32),
not as numeric REALs. This app automatically decodes those so charts/filters behave like you’d expect.
        """
    )

    st.markdown("### Quick schema glance")
    cols = st.columns(2)
    with cols[0]:
        st.write("Tables:", ", ".join(tables))
    with cols[1]:
        st.write("DB file size (bytes):", os.path.getsize(db_path))

    st.markdown("---")
    st.markdown("### Recommended mental model")
    st.markdown(
        """
Think of the pipeline as:

**RF capture → decode → `tpms_signals` (ground truth) → clustering → `vehicles` → event log → `encounters` → (optional) alerting → `maintenance_history`**
        """
    )


def section_db_overview():
    st.subheader("DB overview")

    meta_cols = st.columns(4)
    meta_cols[0].metric("File", os.path.basename(db_path))
    meta_cols[1].metric("Size", f"{os.path.getsize(db_path):,} bytes")
    meta_cols[2].metric("Tables", len(tables))
    try:
        mtime = pd.to_datetime(os.path.getmtime(db_path), unit="s")
        meta_cols[3].metric("Modified", str(mtime))
    except Exception:
        pass

    overview_rows = []
    for t in tables:
        if t == "sqlite_sequence":
            continue
        overview_rows.append(
            {
                "table": t,
                "rows": _row_count(con, t),
                "columns": len(_table_info(con, t)),
            }
        )
    st.dataframe(pd.DataFrame(overview_rows).sort_values(["rows", "table"], ascending=[False, True]), use_container_width=True)

    with st.expander("Schema details (columns + foreign keys)"):
        for t in tables:
            if t == "sqlite_sequence":
                continue
            st.markdown(f"#### `{t}`")
            st.dataframe(_table_info(con, t), use_container_width=True)
            fk = _foreign_keys(con, t)
            if not fk.empty:
                st.caption("Foreign keys")
                st.dataframe(fk, use_container_width=True)


def section_table_explorer():
    st.subheader("Table explorer")

    t = st.selectbox("Table", [x for x in tables if x != "sqlite_sequence"])
    n = _row_count(con, t)
    st.caption(f"{n:,} rows")

    df = load_table(db_path, t, limit=None)

    # Make timestamps readable if present
    ts_cols = [c for c in df.columns if c in ("timestamp", "first_seen", "last_seen")]
    for c in ts_cols:
        df[c + "_local"] = _to_local_dt(df[c])

    # Show blobs as hex for readability
    for c in df.columns:
        if df[c].dtype == "object":
            sample = df[c].dropna().head(10)
            if not sample.empty and isinstance(sample.iloc[0], (bytes, bytearray)):
                df[c + "_hex"] = df[c].apply(_bytes_to_hex)

    st.dataframe(df, use_container_width=True, height=520)


def section_signals_dashboard():
    st.subheader("Signals dashboard (`tpms_signals`)")

    if "tpms_signals" not in tables:
        st.info("No tpms_signals table found.")
        return

    df = load_signals_clean(db_path)

    # Filters
    with st.sidebar:
        st.header("Signals filters")
        proto = sorted([p for p in df.get("protocol", pd.Series()).dropna().unique().tolist()])
        selected_proto = st.multiselect("Protocol", proto, default=proto)
        freq_min, freq_max = float(df["frequency_mhz"].min()), float(df["frequency_mhz"].max())
        if math.isfinite(freq_min) and math.isfinite(freq_max):
            selected_freq = st.slider("Frequency (MHz)", min_value=freq_min, max_value=freq_max, value=(freq_min, freq_max))
        else:
            selected_freq = (freq_min, freq_max)
        battery = st.multiselect("Battery low", [0, 1], default=[0, 1])

    # Apply filters
    fdf = df.copy()
    if "protocol" in fdf.columns and selected_proto:
        fdf = fdf[fdf["protocol"].isin(selected_proto)]
    if "frequency_mhz" in fdf.columns:
        fdf = fdf[(fdf["frequency_mhz"] >= selected_freq[0]) & (fdf["frequency_mhz"] <= selected_freq[1])]
    if "battery_low" in fdf.columns:
        fdf = fdf[fdf["battery_low"].isin(battery)]

    kpi = st.columns(5)
    kpi[0].metric("Signals", len(fdf))
    kpi[1].metric("Unique sensors", fdf["tpms_id"].nunique() if "tpms_id" in fdf.columns else 0)
    kpi[2].metric("Protocols", fdf["protocol"].nunique() if "protocol" in fdf.columns else 0)
    if "timestamp_local" in fdf.columns:
        kpi[3].metric("First", str(fdf["timestamp_local"].min()))
        kpi[4].metric("Last", str(fdf["timestamp_local"].max()))

    # Sensor list summary
    st.markdown("### Sensor summary")
    summary_cols = []
    if "signal_strength_decoded" in fdf.columns:
        summary_cols.append("signal_strength_decoded")
    if "snr_decoded" in fdf.columns:
        summary_cols.append("snr_decoded")
    if "pressure_psi" in fdf.columns:
        summary_cols.append("pressure_psi")
    if "temp_f" in fdf.columns:
        summary_cols.append("temp_f")

    if "tpms_id" in fdf.columns:
        agg = {c: "mean" for c in summary_cols}
        agg["timestamp_local"] = ["min", "max"] if "timestamp_local" in fdf.columns else "count"
        agg["tpms_id"] = "count"
        g = fdf.groupby("tpms_id").agg(agg)
        # flatten columns
        g.columns = ["_".join([x for x in col if x]) if isinstance(col, tuple) else str(col) for col in g.columns]
        g = g.rename(columns={"tpms_id_count": "signals"})
        g = g.reset_index().sort_values("signals", ascending=False)
        st.dataframe(g, use_container_width=True, height=320)

        # Drill into a single sensor
        st.markdown("### Drill-in")
        sel = st.selectbox("Pick a sensor (tpms_id)", g["tpms_id"].tolist())
        sdf = fdf[fdf["tpms_id"] == sel].copy()
        st.dataframe(sdf.sort_values("timestamp") if "timestamp" in sdf.columns else sdf, use_container_width=True, height=260)

        if px is not None and "timestamp_local" in sdf.columns and len(sdf) > 1:
            # Time series if multiple signals exist
            st.plotly_chart(px.line(sdf.sort_values("timestamp_local"), x="timestamp_local", y="pressure_psi", title="Pressure over time (psi)"), use_container_width=True)

    # Charts
    st.markdown("### Patterns")
    if px is None:
        st.info("Plotly isn't available in this environment. Install plotly for interactive charts: pip install plotly")
        return

    c1, c2 = st.columns(2)
    with c1:
        if "pressure_psi" in fdf.columns:
            st.plotly_chart(px.histogram(fdf, x="pressure_psi", nbins=40, title="Pressure distribution (psi)"), use_container_width=True)
    with c2:
        if "temp_f" in fdf.columns:
            st.plotly_chart(px.histogram(fdf, x="temp_f", nbins=40, title="Temperature distribution (°F)"), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        if "signal_strength_decoded" in fdf.columns:
            st.plotly_chart(px.histogram(fdf, x="signal_strength_decoded", nbins=40, title="Signal strength (decoded, dBm-ish)"), use_container_width=True)
    with c4:
        if "snr_decoded" in fdf.columns:
            st.plotly_chart(px.histogram(fdf, x="snr_decoded", nbins=40, title="SNR (decoded, dB-ish)"), use_container_width=True)

    if all(c in fdf.columns for c in ["pressure_psi", "signal_strength_decoded"]):
        st.plotly_chart(px.scatter(fdf, x="signal_strength_decoded", y="pressure_psi", hover_data=["tpms_id"], title="Pressure vs signal strength"), use_container_width=True)


def section_vehicles_encounters():
    st.subheader("Vehicles & encounters")

    if "vehicles" not in tables or "encounters" not in tables:
        st.info("This DB doesn't have both vehicles and encounters tables.")
        return

    vdf = load_vehicles_clean(db_path)
    edf = load_encounters_clean(db_path)

    # KPIs
    k = st.columns(5)
    k[0].metric("Vehicle clusters", len(vdf))
    k[1].metric("Encounters", len(edf))
    k[2].metric("Median sensors/cluster", int(vdf["tpms_count"].median()) if "tpms_count" in vdf.columns else 0)
    k[3].metric("Max sensors/cluster", int(vdf["tpms_count"].max()) if "tpms_count" in vdf.columns else 0)
    if "timestamp_local" in edf.columns:
        k[4].metric("Encounter window", f"{edf['timestamp_local'].min()} → {edf['timestamp_local'].max()}")

    if px is not None and "tpms_count" in vdf.columns:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.histogram(vdf, x="tpms_count", nbins=60, title="How many TPMS IDs per vehicle-cluster?"), use_container_width=True)
        with c2:
            if "encounter_count" in vdf.columns:
                st.plotly_chart(px.scatter(vdf, x="tpms_count", y="encounter_count", hover_data=["id", "nickname"], title="Cluster size vs encounter count"), use_container_width=True)

    st.markdown("### Browse clusters")
    show_cols = ["id", "nickname", "encounter_count", "tpms_count", "first_seen_local", "last_seen_local", "vehicle_hash"]
    show_cols = [c for c in show_cols if c in vdf.columns]
    st.dataframe(vdf[show_cols].sort_values(["encounter_count"], ascending=False), use_container_width=True, height=320)

    sel_id = st.selectbox("Select vehicle cluster (vehicles.id)", vdf["id"].tolist())
    one = vdf[vdf["id"] == sel_id].iloc[0].to_dict()

    st.markdown("### Cluster detail")
    left, right = st.columns([1, 1])
    with left:
        st.json({k: one.get(k) for k in ["id", "nickname", "vehicle_hash", "encounter_count", "tpms_count", "notes", "metadata"] if k in one})
    with right:
        # Encounters timeline for this vehicle
        v_enc = edf[edf["vehicle_id"] == sel_id].copy()
        if "timestamp_local" in v_enc.columns and px is not None and not v_enc.empty:
            ts = v_enc.sort_values("timestamp_local")
            ts["n"] = 1
            # Resample per minute for readability
            tmp = ts.set_index("timestamp_local")["n"].resample("1min").sum().reset_index()
            st.plotly_chart(px.line(tmp, x="timestamp_local", y="n", title="Encounters over time (per minute)"), use_container_width=True)

    # Join to signals (best-effort) using tpms_ids_list
    tpms_ids = one.get("tpms_ids_list", [])
    st.markdown(f"### TPMS IDs in this cluster ({len(tpms_ids)})")
    st.code(", ".join(tpms_ids[:50]) + (" ..." if len(tpms_ids) > 50 else ""))

    if "tpms_signals" in tables and tpms_ids:
        sdf = load_signals_clean(db_path)
        sdf = sdf[sdf["tpms_id"].isin(tpms_ids)].copy()
        st.dataframe(
            sdf[[
                c for c in ["tpms_id", "timestamp_local", "frequency_mhz", "signal_strength_decoded", "snr_decoded", "pressure_psi", "temp_f", "battery_low", "protocol"]
                if c in sdf.columns
            ]].sort_values("timestamp_local", ascending=False),
            use_container_width=True,
            height=320,
        )


def section_quality_checks():
    st.subheader("Quality checks (find bugs + weirdness fast)")

    issues = []

    # Signals checks
    if "tpms_signals" in tables:
        s = load_signals_clean(db_path)

        # Blob-in-real detection
        for c in ["signal_strength", "snr"]:
            if c in s.columns and _is_blob_float_col(s, c):
                issues.append(f"`tpms_signals.{c}` is stored as 4-byte float blobs. (This app decodes it automatically.)")

        # Temperature plausibility check
        if "temp_f" in s.columns:
            # If column is mislabeled C but looks like F, temps will cluster around typical Fahrenheit values.
            t = s["temp_f"].dropna()
            if not t.empty:
                # Heuristic: if median is > 60 and max < 250, it's probably Fahrenheit.
                if t.median() > 60 and t.max() < 260:
                    issues.append("`tpms_signals.temperature_c` looks like **°F** (not °C). This app shows °F and also converts to °C.")

        # Pressure plausibility
        if "pressure_psi" in s.columns:
            p = pd.to_numeric(s["pressure_psi"], errors="coerce")
            if (p < 0).any():
                issues.append("Some pressures are negative. Likely invalid decodes/noise; consider filtering `pressure_psi >= 0` for dashboards.")

        # Battery low ratio
        if "battery_low" in s.columns:
            vc = s["battery_low"].value_counts(dropna=False)
            if 0 in vc.index and 1 in vc.index:
                ratio = vc.get(1, 0) / max(1, (vc.get(0, 0) + vc.get(1, 0)))
                if ratio > 0.2:
                    issues.append(f"`battery_low` is flagged on {ratio*100:.1f}% of signals — unusually high; might be noise/misdecode or inverted bit.")

    # Vehicles checks
    if "vehicles" in tables:
        v = load_vehicles_clean(db_path)
        if "tpms_count" in v.columns:
            big = (v["tpms_count"] > 8).mean()
            if big > 0.1:
                issues.append(
                    f"{big*100:.1f}% of vehicle-clusters have >8 TPMS IDs. In dense environments, the clustering logic may be over-grouping."
                )

            # “sensor belongs to many vehicles” check
            id_lists = v.get("tpms_ids_list", pd.Series([[]]*len(v)))
            counts = {}
            for lst in id_lists:
                for tid in lst:
                    counts[tid] = counts.get(tid, 0) + 1
            if counts:
                max_membership = max(counts.values())
                if max_membership > 3:
                    issues.append(f"At least one TPMS ID appears in {max_membership} different vehicle-clusters — strong sign of over-grouping.")

    if issues:
        st.markdown("### Detected issues / quirks")
        for x in issues:
            st.warning(x)
    else:
        st.success("No obvious red flags detected.")

    st.markdown("---")
    st.markdown("### Raw distributions (so you can see the weirdness)")
    if "tpms_signals" in tables and px is not None:
        s = load_signals_clean(db_path)
        c1, c2 = st.columns(2)
        with c1:
            if "pressure_psi" in s.columns:
                st.plotly_chart(px.box(s, y="pressure_psi", points="outliers", title="Pressure (psi) — box"), use_container_width=True)
        with c2:
            if "temp_f" in s.columns:
                st.plotly_chart(px.box(s, y="temp_f", points="outliers", title="Temperature (°F) — box"), use_container_width=True)

    if "vehicles" in tables and px is not None:
        v = load_vehicles_clean(db_path)
        if "tpms_count" in v.columns:
            st.plotly_chart(px.box(v, y="tpms_count", points="outliers", title="TPMS IDs per vehicle-cluster — box"), use_container_width=True)


def section_export_clean():
    st.subheader("Export / clean")

    st.markdown(
        """
This does **not** modify your DB unless you explicitly export a cleaned copy.
You can export:
- a **cleaned CSV** for `tpms_signals` (decoded RSSI/SNR, readable timestamps, raw payload hex),
- or a **new SQLite DB** with corrected numeric types for easier SQL/dashboarding.
        """
    )

    if "tpms_signals" not in tables:
        st.info("No tpms_signals table found.")
        return

    s = load_signals_clean(db_path)

    export_cols = [c for c in s.columns if c not in ["signal_strength", "snr", "raw_data"]]
    cleaned = s[export_cols].copy()

    st.markdown("### Preview (cleaned signals)")
    st.dataframe(cleaned.head(50), use_container_width=True, height=320)

    # CSV download
    csv_bytes = cleaned.to_csv(index=False).encode("utf-8")
    st.download_button("Download cleaned tpms_signals.csv", data=csv_bytes, file_name="tpms_signals_cleaned.csv", mime="text/csv")

    st.markdown("---")
    st.markdown("### Create cleaned SQLite (optional)")
    out_name = st.text_input("Output DB file name", value="tpms_tracker.cleaned.db")
    if st.button("Build cleaned DB"):
        out_path = os.path.join(os.getcwd(), out_name)
        try:
            out = sqlite3.connect(out_path)
            cleaned.to_sql("tpms_signals_cleaned", out, if_exists="replace", index=False)

            # Also bring over vehicles + encounters (as-is), but add parsed tpms_count if possible.
            if "vehicles" in tables:
                v = load_vehicles_clean(db_path)
                v.to_sql("vehicles", out, if_exists="replace", index=False)
            if "encounters" in tables:
                e = load_encounters_clean(db_path)
                e.to_sql("encounters", out, if_exists="replace", index=False)

            out.close()
            st.success(f"Wrote: {out_path}")
            with open(out_path, "rb") as f:
                st.download_button("Download cleaned DB", data=f.read(), file_name=os.path.basename(out_path), mime="application/octet-stream")
        except Exception as e:
            st.error(f"Failed to build cleaned DB: {e}")


# ----------------------------- Router -----------------------------

if page == "How your app stores data":
    section_how_data_is_stored()
elif page == "DB overview":
    section_db_overview()
elif page == "Table explorer":
    section_table_explorer()
elif page == "Signals dashboard":
    section_signals_dashboard()
elif page == "Vehicles & encounters":
    section_vehicles_encounters()
elif page == "Quality checks":
    section_quality_checks()
elif page == "Export / clean":
    section_export_clean()

con.close()
