# tpms_db_explorer.py
# Streamlit app for exploring tpms_tracker SQLite databases + mate-graph analysis.
# Run:  streamlit run tpms_db_explorer.py
#
# Optional:
#   export TPMS_DB_PATH=/path/to/tpms_tracker.db
#   export TPMS_APP_TZ=America/Denver

from __future__ import annotations

import json
import math
import os
import sqlite3
import struct
import tempfile
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd
import streamlit as st

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

# Plotly is nicer for interactive exploration; fall back gracefully if unavailable.
try:
    import plotly.express as px
except Exception:  # pragma: no cover
    px = None  # type: ignore


APP_TZ = os.environ.get("TPMS_APP_TZ", "America/Denver")
DEFAULT_DB_PATH = os.environ.get("TPMS_DB_PATH", "tpms_tracker.db")


# =============================================================================
# SQLite helpers
# =============================================================================

def connect_ro(path: str) -> sqlite3.Connection:
    """Read-only sqlite connection (best-effort)."""
    try:
        return sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    except Exception:
        return sqlite3.connect(path, check_same_thread=False)


def list_tables(con: sqlite3.Connection) -> List[str]:
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view') ORDER BY name;")
    return [r[0] for r in cur.fetchall()]


def table_info(con: sqlite3.Connection, table: str) -> pd.DataFrame:
    return pd.read_sql_query(f"PRAGMA table_info({table});", con)


def foreign_keys(con: sqlite3.Connection, table: str) -> pd.DataFrame:
    return pd.read_sql_query(f"PRAGMA foreign_key_list({table});", con)


def row_count(con: sqlite3.Connection, table: str) -> int:
    try:
        cur = con.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {table};")
        return int(cur.fetchone()[0])
    except Exception:
        return 0


@st.cache_data(show_spinner=False)
def sqlite_quick_check(db_path: str) -> List[str]:
    """Return PRAGMA quick_check results. ['ok'] means healthy.

    If the DB is malformed/corrupted enough that PRAGMA itself throws, return a single
    'error: ...' string so callers can degrade gracefully instead of crashing.
    """
    con = connect_ro(db_path)
    try:
        cur = con.cursor()
        try:
            cur.execute("PRAGMA quick_check;")
            return [r[0] for r in cur.fetchall()]
        except sqlite3.DatabaseError as e:
            return [f"error: {e}"]
    finally:
        con.close()
@st.cache_data(show_spinner=False)
def best_effort_signal_bounds(
    db_path: str,
    scan_cap_rows: int = 2_000_000,
    chunk_size: int = 50_000,
) -> Dict[str, Optional[float]]:
    """Best-effort stats for tpms_signals that won't crash on partially corrupted DBs.

    Returns a dict with *both* the new key names (n, n_sensors) and the legacy ones
    (n_rows_scanned, n_sensors_scanned) so older UI code doesn't explode.

    Notes:
    - If the DB is corrupted, we stop at the first unreadable page and return partial stats.
    - Values are floats for cache/typing compatibility; UI should format/cast safely.
    """
    con = connect_ro(db_path)
    try:
        cur = con.cursor()

        # Fast path (can fail on malformed DB)
        try:
            cur.execute(
                "SELECT MIN(timestamp), MAX(timestamp), COUNT(*), COUNT(DISTINCT tpms_id) FROM tpms_signals;"
            )
            mn, mx, n, ns = cur.fetchone()
            n_f = float(n) if n is not None else None
            ns_f = float(ns) if ns is not None else None
            return {
                "min_ts": float(mn) if mn is not None else None,
                "max_ts": float(mx) if mx is not None else None,
                "n": n_f,
                "n_sensors": ns_f,
                "n_rows_scanned": n_f,
                "n_sensors_scanned": ns_f,
            }
        except sqlite3.DatabaseError:
            pass

        # Slow but robust: scan rows until corruption (or cap)
        try:
            cur.execute("SELECT tpms_id, timestamp FROM tpms_signals;")
        except sqlite3.DatabaseError:
            return {
                "min_ts": None,
                "max_ts": None,
                "n": None,
                "n_sensors": None,
                "n_rows_scanned": None,
                "n_sensors_scanned": None,
            }

        mn: Optional[float] = None
        mx: Optional[float] = None
        n_rows = 0
        sensors: Set[str] = set()

        while True:
            if n_rows >= scan_cap_rows:
                break
            try:
                rows = cur.fetchmany(chunk_size)
            except sqlite3.DatabaseError:
                break
            if not rows:
                break

            n_rows += len(rows)
            for sid, ts in rows:
                try:
                    if ts is not None:
                        t = float(ts)
                        mn = t if (mn is None or t < mn) else mn
                        mx = t if (mx is None or t > mx) else mx
                except Exception:
                    pass
                if sid is not None and len(sensors) < 200_000:
                    sensors.add(str(sid))

        n_f = float(n_rows) if n_rows else None
        ns_f = float(len(sensors)) if sensors else None
        return {
            "min_ts": mn,
            "max_ts": mx,
            "n": n_f,
            "n_sensors": ns_f,
            "n_rows_scanned": n_f,
            "n_sensors_scanned": ns_f,
        }
    finally:
        con.close()
def get_signal_bounds(con: sqlite3.Connection) -> Dict[str, Optional[float]]:
    """Back-compat wrapper used by older sections.

    Reads the DB path from the connection and delegates to best_effort_signal_bounds().
    Always returns keys: min_ts, max_ts, n, n_sensors, n_rows_scanned, n_sensors_scanned.
    """
    empty = {
        "min_ts": None,
        "max_ts": None,
        "n": None,
        "n_sensors": None,
        "n_rows_scanned": None,
        "n_sensors_scanned": None,
    }
    # We can’t reliably cache on a sqlite connection object; use the file path instead.
    try:
        db_path = con.execute("PRAGMA database_list;").fetchone()[2]
    except Exception:
        return empty

    try:
        out = best_effort_signal_bounds(db_path)
    except Exception:
        return empty

    # Ensure expected keys even if someone imports an older cached function.
    out = dict(empty, **(out or {}))
    if out.get("n") is None:
        out["n"] = out.get("n_rows_scanned")
    if out.get("n_sensors") is None:
        out["n_sensors"] = out.get("n_sensors_scanned")
    if out.get("n_rows_scanned") is None:
        out["n_rows_scanned"] = out.get("n")
    if out.get("n_sensors_scanned") is None:
        out["n_sensors_scanned"] = out.get("n_sensors")
    return out
def decode_float32_blob(x: Any) -> Optional[float]:
    """SQLite can store BLOBs in REAL columns. This DB often stores float32-packed bytes."""
    if x is None:
        return None
    if isinstance(x, (bytes, bytearray)) and len(x) == 4:
        try:
            return struct.unpack("<f", x)[0]
        except Exception:
            return None
    try:
        return float(x)
    except Exception:
        return None


def bytes_to_hex(x: Any) -> Any:
    return x.hex() if isinstance(x, (bytes, bytearray)) else x


def maybe_parse_json_array(s: Any) -> List[str]:
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


def to_local_dt(series_seconds: pd.Series) -> pd.Series:
    """Treat as unix epoch seconds."""
    s = pd.to_datetime(series_seconds, unit="s", utc=True, errors="coerce")
    if ZoneInfo is None:
        return s.dt.tz_localize(None)
    try:
        return s.dt.tz_convert(ZoneInfo(APP_TZ))
    except Exception:
        return s.dt.tz_convert("UTC")


def is_blob_float_col(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return False
    sample = df[col].dropna().head(25)
    if sample.empty:
        return False
    blob4 = sample.apply(lambda v: isinstance(v, (bytes, bytearray)) and len(v) == 4).sum()
    return blob4 >= max(1, int(0.5 * len(sample)))


# =============================================================================
# Cached loaders
# =============================================================================

@st.cache_data(show_spinner=False)
def load_table(db_path: str, table: str, limit: Optional[int] = None) -> pd.DataFrame:
    con = connect_ro(db_path)
    q = f"SELECT * FROM {table}"
    if limit is not None:
        q += f" LIMIT {int(limit)}"
    df = pd.read_sql_query(q + ";", con)
    con.close()
    return df


@st.cache_data(show_spinner=False)
def load_signals_clean(
    db_path: str,
    limit: int = 200_000,
    order: str = "ASC",
    start_ts: Optional[float] = None,
    end_ts: Optional[float] = None,
    order_by: str = "rowid",
) -> pd.DataFrame:
    """Load a *subset* of tpms_signals and add convenience/decoded columns.

    Why subset? Large tables are slow, and partially-corrupted DB files can fail on full scans.
    """
    order = (order or "ASC").upper()
    if order not in ("ASC", "DESC"):
        order = "ASC"

    order_by = (order_by or "rowid").lower()
    if order_by not in ("rowid", "timestamp", "id"):
        order_by = "rowid"

    con = connect_ro(db_path)
    try:
        where = []
        params: List[Any] = []

        if start_ts is not None:
            where.append("timestamp >= ?")
            params.append(float(start_ts))
        if end_ts is not None:
            where.append("timestamp <= ?")
            params.append(float(end_ts))

        sql = "SELECT * FROM tpms_signals"
        if where:
            sql += " WHERE " + " AND ".join(where)

        # NOTE: ORDER BY rowid tends to be the most robust on corrupted DBs (no index needed).
        if order_by == "rowid":
            sql += f" ORDER BY rowid {order}"
        else:
            sql += f" ORDER BY {order_by} {order}"

        sql += " LIMIT ?"
        params.append(int(limit))

        df = pd.read_sql_query(sql + ";", con, params=params).copy()
    finally:
        con.close()

    # decode float-packed blobs / numeric cast
    for c in ["signal_strength", "snr"]:
        if c in df.columns:
            df[c + "_decoded"] = df[c].apply(decode_float32_blob)

    # convert timestamps
    if "timestamp" in df.columns:
        df["timestamp_local"] = to_local_dt(df["timestamp"])

    # frequency convenience
    if "frequency" in df.columns:
        df["frequency_mhz"] = pd.to_numeric(df["frequency"], errors="coerce") / 1e6

    # hex-ify raw payload
    if "raw_data" in df.columns:
        df["raw_data_hex"] = df["raw_data"].apply(bytes_to_hex)

    # temperature convenience (many TPMS decoders store F but label it C)
    if "temperature_c" in df.columns:
        t = pd.to_numeric(df["temperature_c"], errors="coerce")
        df["temp_f"] = t
        df["temp_c_from_f"] = (t - 32.0) * (5.0 / 9.0)

    return df
def load_vehicles_clean(db_path: str) -> pd.DataFrame:
    df = load_table(db_path, "vehicles", limit=None).copy()
    if "tpms_ids" in df.columns:
        df["tpms_ids_list"] = df["tpms_ids"].apply(maybe_parse_json_array)
        df["tpms_count"] = df["tpms_ids_list"].apply(len)
    if "first_seen" in df.columns:
        df["first_seen_local"] = to_local_dt(df["first_seen"])
    if "last_seen" in df.columns:
        df["last_seen_local"] = to_local_dt(df["last_seen"])
    return df


@st.cache_data(show_spinner=False)
def load_encounters_clean(db_path: str) -> pd.DataFrame:
    df = load_table(db_path, "encounters", limit=None).copy()
    if "timestamp" in df.columns:
        df["timestamp_local"] = to_local_dt(df["timestamp"])
    return df


# =============================================================================
# Mate-graph analysis (DB replay)
# =============================================================================

@dataclass(frozen=True)
class MateGraphParams:
    base_mate_window_s: float = 1.0
    relaxed_mate_window_s: float = 3.0
    min_same_sensor_gap_s: float = 0.25
    suspect_score: float = 2.0
    link_score: float = 4.0
    decay_halflife_s: float = 180.0


class MateGraphAnalyzer:
    """
    Replays tpms_signals from the DB to build a "mate graph".

    Pair scoring rules (mirrors your runtime engine idea):
    - A and B detected within base_mate_window_s => score += 1
    - Once a pair score >= suspect_score, effective window relaxes to relaxed_mate_window_s
    - Pair scores decay exponentially with half-life decay_halflife_s
    """

    def __init__(self, params: MateGraphParams):
        self.p = params
        self.pair_scores: Dict[Tuple[str, str], float] = defaultdict(float)
        self._recent: deque[Tuple[str, float]] = deque()
        self._last_decay_ts: Optional[float] = None
        self._last_seen_sensor_ts: Dict[str, float] = {}

    @staticmethod
    def _pair_key(a: str, b: str) -> Tuple[str, str]:
        return (a, b) if a <= b else (b, a)

    def _decay(self, now: float):
        if self._last_decay_ts is None:
            self._last_decay_ts = now
            return
        dt = max(0.0, now - self._last_decay_ts)
        if dt <= 0:
            return
        decay_factor = 0.5 ** (dt / max(1e-9, self.p.decay_halflife_s))
        if decay_factor < 0.999:
            for k in list(self.pair_scores.keys()):
                self.pair_scores[k] *= decay_factor
                if self.pair_scores[k] < 0.05:
                    del self.pair_scores[k]
        self._last_decay_ts = now

    def _effective_window(self, a: str, b: str) -> float:
        s = self.pair_scores.get(self._pair_key(a, b), 0.0)
        return self.p.relaxed_mate_window_s if s >= self.p.suspect_score else self.p.base_mate_window_s

    def get_pair_likelihood(self, a: str, b: str) -> float:
        s = float(self.pair_scores.get(self._pair_key(a, b), 0.0))
        return 1.0 - math.exp(-s / max(1e-6, self.p.link_score))

    def ingest_event(self, sid: str, ts: float):
        if not sid:
            return

        # debounce same sensor spam
        last = self._last_seen_sensor_ts.get(sid)
        if last is not None and (ts - last) < self.p.min_same_sensor_gap_s:
            return
        self._last_seen_sensor_ts[sid] = ts

        self._decay(ts)

        # purge stale
        cutoff = ts - self.p.relaxed_mate_window_s
        while self._recent and self._recent[0][1] < cutoff:
            self._recent.popleft()

        # compare with recent (newest->oldest)
        for other_sid, other_ts in reversed(self._recent):
            dt = ts - other_ts
            if dt > self.p.relaxed_mate_window_s:
                break
            if other_sid == sid:
                continue
            if dt <= self._effective_window(sid, other_sid):
                self.pair_scores[self._pair_key(sid, other_sid)] += 1.0

        self._recent.append((sid, ts))

    def ingest_events(self, events: Sequence[Tuple[str, float]]):
        for sid, ts in events:
            self.ingest_event(str(sid), float(ts))

    def components(self, min_score: float) -> List[List[str]]:
        adj: Dict[str, Set[str]] = defaultdict(set)
        for (a, b), score in self.pair_scores.items():
            if score >= min_score:
                adj[a].add(b)
                adj[b].add(a)

        seen: Set[str] = set()
        comps: List[List[str]] = []

        for node in adj.keys():
            if node in seen:
                continue
            stack = [node]
            seen.add(node)
            comp: List[str] = []
            while stack:
                x = stack.pop()
                comp.append(x)
                for y in adj[x]:
                    if y not in seen:
                        seen.add(y)
                        stack.append(y)
            comps.append(sorted(comp))

        comps.sort(key=len, reverse=True)
        return comps


@st.cache_data(show_spinner=True)
def load_signal_events(
    db_path: str,
    start_ts: float,
    end_ts: float,
    max_events: int,
    sensor_filter: str = "",
) -> List[Tuple[str, float]]:
    """
    Load (tpms_id, timestamp) events in [start_ts, end_ts].
    Uses a direct SQL query so we don't have to load the entire signals table.
    """
    con = connect_ro(db_path)
    try:
        where = "WHERE timestamp >= ? AND timestamp <= ? AND tpms_id IS NOT NULL AND tpms_id != ''"
        params: List[Any] = [float(start_ts), float(end_ts)]
        if sensor_filter.strip():
            where += " AND tpms_id LIKE ?"
            params.append(f"%{sensor_filter.strip()}%")

        q = f"""
        SELECT tpms_id, timestamp
        FROM tpms_signals
        {where}
        ORDER BY timestamp ASC
        LIMIT ?;
        """
        params.append(int(max_events))
        df = pd.read_sql_query(q, con, params=params)
        return list(zip(df["tpms_id"].astype(str).tolist(), df["timestamp"].astype(float).tolist()))
    finally:
        con.close()


@st.cache_data(show_spinner=True)
def compute_mate_graph(
    db_path: str,
    start_ts: float,
    end_ts: float,
    max_events: int,
    sensor_filter: str,
    params: MateGraphParams,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      pairs_df: mate pairs with metrics (sorted desc)
      comps_df: strong components at params.link_score
    """
    events = load_signal_events(db_path, start_ts, end_ts, max_events=max_events, sensor_filter=sensor_filter)

    mg = MateGraphAnalyzer(params)
    mg.ingest_events(events)

    rows = []
    for (a, b), score in mg.pair_scores.items():
        eff_window = params.relaxed_mate_window_s if score >= params.suspect_score else params.base_mate_window_s
        rows.append({
            "sensor_a": a,
            "sensor_b": b,
            "raw_score": float(score),
            "likelihood": float(mg.get_pair_likelihood(a, b)),
            "status": "suspected" if score >= params.suspect_score else "candidate",
            "effective_window_s": float(eff_window),
        })

    pairs_df = pd.DataFrame(rows)
    if not pairs_df.empty:
        pairs_df = pairs_df.sort_values(["raw_score", "likelihood"], ascending=False)

    comps = mg.components(params.link_score)
    comps_df = pd.DataFrame([
        {"component_id": i, "size": len(c), "sensors": ", ".join(c)}
        for i, c in enumerate(comps)
    ])
    return pairs_df, comps_df


# =============================================================================
# UI sections
# =============================================================================

def section_how_data_is_stored(tables: List[str], db_path: str):
    st.subheader("What your app is doing with the data (inferred from the DB)")

    st.markdown(
        """
This database has **four app tables** (plus SQLite internals):

- **`tpms_signals`** — raw TPMS transmissions (one row per decoded RF message).
  - *sensor id*, *time*, optional *location*, *RF metrics* (RSSI/SNR), decoded payload (pressure/temp/battery), and raw payload bytes.
- **`vehicles`** — a derived entity: TPMS sensor IDs clustered into a “vehicle”.
  - `tpms_ids` is stored as a JSON array of sensor IDs.
  - `vehicle_hash` looks like a stable signature for the cluster.
  - `encounter_count`, `first_seen`, `last_seen` track how often/when the cluster was observed.
- **`encounters`** — event log of each time a vehicle-cluster was “seen”.
  - References `vehicles.id`.
- **`maintenance_history`** — intended for alerts/trends, but often empty in early snapshots.

**Important quirk:**
`signal_strength` and `snr` are often stored as 4‑byte float blobs (packed float32) instead of numeric REALs.
This app automatically decodes them so charts/filters behave normally.
        """
    )

    st.markdown("### Quick schema glance")
    c1, c2 = st.columns(2)
    with c1:
        st.write("Tables:", ", ".join(tables))
    with c2:
        st.write("DB file size (bytes):", os.path.getsize(db_path))

    st.markdown("---")
    st.markdown("### Pipeline mental model")
    st.markdown("**RF capture → decode → `tpms_signals` → clustering → `vehicles` → `encounters` → (optional) alerting**")


def section_db_overview(con: sqlite3.Connection, tables: List[str], db_path: str):
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
        overview_rows.append({"table": t, "rows": row_count(con, t), "columns": len(table_info(con, t))})
    st.dataframe(pd.DataFrame(overview_rows).sort_values(["rows", "table"], ascending=[False, True]), use_container_width=True)

    with st.expander("Schema details (columns + foreign keys)"):
        for t in tables:
            if t == "sqlite_sequence":
                continue
            st.markdown(f"#### `{t}`")
            st.dataframe(table_info(con, t), use_container_width=True)
            fk = foreign_keys(con, t)
            if not fk.empty:
                st.caption("Foreign keys")
                st.dataframe(fk, use_container_width=True)


def section_table_explorer(con: sqlite3.Connection, tables: List[str], db_path: str):
    st.subheader("Table explorer")

    t = st.selectbox("Table", [x for x in tables if x != "sqlite_sequence"])
    n = row_count(con, t)
    if n is not None:
        st.caption(f"{n:,} rows (COUNT may be partial/unknown if DB is corrupted)")
    else:
        st.caption("Row count unavailable (DB may be corrupted).")

    with st.sidebar:
        st.header("Table explorer — load")
        limit = int(st.number_input("Max rows to load", min_value=100, max_value=1_000_000, value=10_000, step=5_000))
        show_hex = st.checkbox("Show BLOB columns as hex", value=True)

    try:
        df = load_table(db_path, t, limit=limit)
    except sqlite3.DatabaseError as e:
        st.error(f"Failed to read table '{t}': {e}")
        st.info("Try lowering 'Max rows to load' in the sidebar, or run a DB recovery from the Export / Clean page.")
        return

    # Make timestamps readable if present
    ts_cols = [c for c in df.columns if c in ("timestamp", "first_seen", "last_seen")]
    for c in ts_cols:
        df[c + "_local"] = to_local_dt(df[c])

    # Show blobs as hex for readability
    if show_hex:
        for c in df.columns:
            if df[c].dtype == "object":
                sample = df[c].dropna().head(10)
                if not sample.empty and isinstance(sample.iloc[0], (bytes, bytearray)):
                    df[c + "_hex"] = df[c].apply(bytes_to_hex)

    st.dataframe(df, use_container_width=True, height=520)
def section_signals_dashboard(tables: List[str], db_path: str):
    st.subheader("Signals dashboard (`tpms_signals`)")

    if "tpms_signals" not in tables:
        st.info("No tpms_signals table found.")
        return

    # --- DB health ---
    qc = sqlite_quick_check(db_path)
    db_ok = (qc == ["ok"])

    with st.sidebar:
        st.header("Signals — load")
        if not db_ok:
            st.error("SQLite quick_check is NOT ok — this DB is partially corrupted. We'll load a safe subset and avoid full scans.")
            with st.expander("quick_check details"):
                st.write(qc)

        limit = int(st.number_input("Max rows to load", min_value=1_000, max_value=2_000_000, value=200_000, step=25_000))
        order_ui = st.radio("Order", ["Newest first", "Oldest first"], index=0 if db_ok else 1)
        order = "DESC" if order_ui == "Newest first" else "ASC"

        order_by = st.selectbox("Order by", ["timestamp", "rowid"], index=0 if db_ok else 1,
                                help="If the DB is corrupted, ordering by rowid is often the most robust.")

        use_time = st.checkbox("Filter by time window (epoch seconds)", value=False)
        start_ts = end_ts = None
        if use_time:
            st.caption("Tip: timestamps here are Unix epoch seconds (UTC).")
            start_ts = st.number_input("Start timestamp (>=)", value=0.0, step=1.0, format="%.0f")
            end_ts = st.number_input("End timestamp (<=)", value=0.0, step=1.0, format="%.0f")
            if end_ts <= 0:
                end_ts = None
            if start_ts <= 0:
                start_ts = None

    # --- Load subset safely ---
    load_note = ""
    try:
        df = load_signals_clean(
            db_path,
            limit=limit,
            order=order,
            order_by=order_by,
            start_ts=start_ts,
            end_ts=end_ts,
        )
    except sqlite3.DatabaseError as e:
        # Common on corrupted DBs when trying to read the tail.
        load_note = f"Load failed with: {e}. Falling back to oldest-first rowid scan."
        try:
            df = load_signals_clean(
                db_path,
                limit=limit,
                order="ASC",
                order_by="rowid",
                start_ts=start_ts,
                end_ts=end_ts,
            )
        except Exception as e2:
            st.error(f"Could not read tpms_signals even in safe mode: {e2}")
            return

    if df.empty:
        st.info("No rows matched the current load settings.")
        return

    if load_note:
        st.warning(load_note)

    st.caption(f"Showing **{len(df):,}** rows (subset). Increase 'Max rows to load' in the sidebar if needed.")

    # --- Filters (applied on the loaded subset) ---
    with st.sidebar:
        st.header("Signals — filters")
        proto = sorted([p for p in df.get('protocol', pd.Series(dtype=object)).dropna().unique().tolist()])
        selected_proto = st.multiselect("Protocol", proto, default=proto) if proto else []

        freq_min = float(df['frequency_mhz'].min()) if 'frequency_mhz' in df.columns else float('nan')
        freq_max = float(df['frequency_mhz'].max()) if 'frequency_mhz' in df.columns else float('nan')
        if math.isfinite(freq_min) and math.isfinite(freq_max) and freq_min < freq_max:
            selected_freq = st.slider("Frequency (MHz)", min_value=freq_min, max_value=freq_max, value=(freq_min, freq_max))
        else:
            selected_freq = None

        battery = st.multiselect("Battery low", [0, 1], default=[0, 1]) if "battery_low" in df.columns else None

    fdf = df.copy()
    if selected_proto and "protocol" in fdf.columns:
        fdf = fdf[fdf["protocol"].isin(selected_proto)]
    if selected_freq and "frequency_mhz" in fdf.columns:
        fdf = fdf[(fdf["frequency_mhz"] >= selected_freq[0]) & (fdf["frequency_mhz"] <= selected_freq[1])]
    if battery is not None and "battery_low" in fdf.columns:
        fdf = fdf[fdf["battery_low"].isin(battery)]

    # KPIs (subset-based)
    kpi = st.columns(5)
    kpi[0].metric("Signals (subset)", len(fdf))
    kpi[1].metric("Unique sensors (subset)", fdf["tpms_id"].nunique() if "tpms_id" in fdf.columns else 0)
    kpi[2].metric("Protocols (subset)", fdf["protocol"].nunique() if "protocol" in fdf.columns else 0)
    if "timestamp_local" in fdf.columns and not fdf["timestamp_local"].isna().all():
        kpi[3].metric("First (subset)", str(fdf["timestamp_local"].min()))
        kpi[4].metric("Last (subset)", str(fdf["timestamp_local"].max()))

    # Sensor summary
    st.markdown("### Sensor summary (subset)")
    summary_cols: List[str] = []
    for c in ["signal_strength_decoded", "snr_decoded", "pressure_psi", "temp_f"]:
        if c in fdf.columns:
            summary_cols.append(c)

    if "tpms_id" in fdf.columns:
        agg = {c: "mean" for c in summary_cols}
        if "timestamp_local" in fdf.columns:
            agg["timestamp_local"] = ["min", "max"]
        agg["tpms_id"] = "count"

        g = fdf.groupby("tpms_id").agg(agg)
        g.columns = ["_".join([x for x in col if x]) if isinstance(col, tuple) else str(col) for col in g.columns]
        g = g.rename(columns={"tpms_id_count": "signals"}).reset_index().sort_values("signals", ascending=False)
        st.dataframe(g, use_container_width=True, height=320)

        # Drill into a single sensor
        st.markdown("### Drill-in (subset)")
        sel = st.selectbox("Pick a sensor (tpms_id)", g["tpms_id"].tolist())
        sdf = fdf[fdf["tpms_id"] == sel].copy()
        sort_col = "timestamp_local" if "timestamp_local" in sdf.columns else None
        if sort_col:
            sdf = sdf.sort_values(sort_col)
        st.dataframe(sdf, use_container_width=True, height=260)

        if px is not None and "timestamp_local" in sdf.columns and len(sdf) > 1 and "pressure_psi" in sdf.columns:
            st.plotly_chart(
                px.line(sdf, x="timestamp_local", y="pressure_psi", title="Pressure over time (psi)"),
                use_container_width=True,
            )

    # Charts
    st.markdown("### Patterns (subset)")
    if px is None:
        st.info("Plotly isn't available. Install plotly for interactive charts: pip install plotly")
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
            st.plotly_chart(px.histogram(fdf, x="signal_strength_decoded", nbins=40, title="Signal strength (decoded)"), use_container_width=True)
    with c4:
        if "snr_decoded" in fdf.columns:
            st.plotly_chart(px.histogram(fdf, x="snr_decoded", nbins=40, title="SNR (decoded)"), use_container_width=True)

    if all(c in fdf.columns for c in ["pressure_psi", "signal_strength_decoded", "tpms_id"]):
        st.plotly_chart(
            px.scatter(fdf, x="signal_strength_decoded", y="pressure_psi", hover_data=["tpms_id"], title="Pressure vs signal strength"),
            use_container_width=True,
        )
def section_vehicles_encounters(tables: List[str], db_path: str):
    st.subheader("Vehicles & encounters")

    if "vehicles" not in tables or "encounters" not in tables:
        st.info("This DB doesn't have both vehicles and encounters tables.")
        return

    vdf = load_vehicles_clean(db_path)
    edf = load_encounters_clean(db_path)

    k = st.columns(5)
    k[0].metric("Vehicle clusters", len(vdf))
    k[1].metric("Encounters", len(edf))
    med = vdf["tpms_count"].median() if "tpms_count" in vdf.columns else None
    k[2].metric("Median sensors/cluster", int(med) if (med is not None and pd.notna(med)) else 0)
    mx = vdf["tpms_count"].max() if "tpms_count" in vdf.columns else None
    k[3].metric("Max sensors/cluster", int(mx) if (mx is not None and pd.notna(mx)) else 0)
    if "timestamp_local" in edf.columns and not edf.empty:
        k[4].metric("Encounter window", f"{edf['timestamp_local'].min()} → {edf['timestamp_local'].max()}")

    if px is not None and "tpms_count" in vdf.columns:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.histogram(vdf, x="tpms_count", nbins=40, title="TPMS IDs per vehicle-cluster"), use_container_width=True)
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
        v_enc = edf[edf["vehicle_id"] == sel_id].copy()
        if "timestamp_local" in v_enc.columns and px is not None and not v_enc.empty:
            ts = v_enc.sort_values("timestamp_local")
            ts["n"] = 1
            tmp = ts.set_index("timestamp_local")["n"].resample("1min").sum().reset_index()
            st.plotly_chart(px.line(tmp, x="timestamp_local", y="n", title="Encounters over time (per minute)"), use_container_width=True)

    tpms_ids = one.get("tpms_ids_list", [])
    st.markdown(f"### TPMS IDs in this cluster ({len(tpms_ids)})")
    st.code(", ".join(tpms_ids[:50]) + (" ..." if len(tpms_ids) > 50 else ""))

    if "tpms_signals" in tables and tpms_ids:
        sdf = load_signals_clean(db_path)
        sdf = sdf[sdf["tpms_id"].isin(tpms_ids)].copy()
        cols = [c for c in ["tpms_id", "timestamp_local", "frequency_mhz", "signal_strength_decoded", "snr_decoded", "pressure_psi", "temp_f", "battery_low", "protocol"] if c in sdf.columns]
        st.dataframe(sdf[cols].sort_values("timestamp_local", ascending=False), use_container_width=True, height=320)


def section_quality_checks(tables: List[str], db_path: str):
    st.subheader("Quality checks (find bugs + weirdness fast)")

    issues: List[str] = []

    if "tpms_signals" in tables:
        s = load_signals_clean(db_path)

        for c in ["signal_strength", "snr"]:
            if c in s.columns and is_blob_float_col(s, c):
                issues.append(f"`tpms_signals.{c}` is stored as 4-byte float blobs. (This app decodes it automatically.)")

        if "temp_f" in s.columns:
            t = s["temp_f"].dropna()
            if not t.empty and t.median() > 60 and t.max() < 260:
                issues.append("`tpms_signals.temperature_c` looks like **°F** (not °C). This app shows °F and converts to °C.")

        if "pressure_psi" in s.columns:
            p = pd.to_numeric(s["pressure_psi"], errors="coerce")
            if (p < 0).any():
                issues.append("Some pressures are negative. Likely invalid decodes/noise; consider filtering `pressure_psi >= 0`.")

        if "battery_low" in s.columns:
            vc = s["battery_low"].value_counts(dropna=False)
            if 0 in vc.index and 1 in vc.index:
                ratio = vc.get(1, 0) / max(1, (vc.get(0, 0) + vc.get(1, 0)))
                if ratio > 0.2:
                    issues.append(f"`battery_low` is flagged on {ratio*100:.1f}% of signals — unusually high; might be noise/misdecode or inverted bit.")

    if "vehicles" in tables:
        v = load_vehicles_clean(db_path)
        if "tpms_count" in v.columns:
            big = (v["tpms_count"] > 8).mean()
            if big > 0.1:
                issues.append(f"{big*100:.1f}% of vehicle-clusters have >8 TPMS IDs. In dense environments, clustering may be over-grouping.")

            id_lists = v.get("tpms_ids_list", pd.Series([[]] * len(v)))
            counts: Dict[str, int] = {}
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
    st.markdown("### Raw distributions")
    if "tpms_signals" in tables and px is not None:
        s = load_signals_clean(db_path)
        c1, c2 = st.columns(2)
        with c1:
            if "pressure_psi" in s.columns:
                st.plotly_chart(px.box(s, y="pressure_psi", points="outliers", title="Pressure (psi)"), use_container_width=True)
        with c2:
            if "temp_f" in s.columns:
                st.plotly_chart(px.box(s, y="temp_f", points="outliers", title="Temperature (°F)"), use_container_width=True)

    if "vehicles" in tables and px is not None and "tpms_count" in load_vehicles_clean(db_path).columns:
        v = load_vehicles_clean(db_path)
        st.plotly_chart(px.box(v, y="tpms_count", points="outliers", title="TPMS IDs per vehicle-cluster"), use_container_width=True)


def section_export_clean(tables: List[str], db_path: str):
    st.subheader("Export / clean")

    st.markdown(
        """
This does **not** modify your DB unless you explicitly export a cleaned copy.
You can export:
- a cleaned CSV for `tpms_signals` (decoded RSSI/SNR, readable timestamps, raw payload hex)
- or a new SQLite DB with corrected numeric types for easier SQL/dashboarding.
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
            if "vehicles" in tables:
                load_vehicles_clean(db_path).to_sql("vehicles", out, if_exists="replace", index=False)
            if "encounters" in tables:
                load_encounters_clean(db_path).to_sql("encounters", out, if_exists="replace", index=False)
            out.close()

            st.success(f"Wrote: {out_path}")
            with open(out_path, "rb") as f:
                st.download_button("Download cleaned DB", data=f.read(), file_name=os.path.basename(out_path), mime="application/octet-stream")
        except Exception as e:
            st.error(f"Failed to build cleaned DB: {e}")


def section_mate_graph(con: sqlite3.Connection, db_path: str):
    st.subheader("Mate graph (pair sensors to likely vehicles)")

    bounds = get_signal_bounds(con)

    n_total = bounds.get("n")
    if n_total is None:
        n_total = bounds.get("n_rows_scanned")
    n_sensors = bounds.get("n_sensors")
    if n_sensors is None:
        n_sensors = bounds.get("n_sensors_scanned")

    if not n_total or (isinstance(n_total, (int, float)) and n_total <= 0):
        st.info("No tpms_signals data available to build a mate graph.")
        return

    with st.sidebar:
        st.header("Mate graph parameters")
        base = st.slider("Base mate window (s)", 0.2, 2.0, 1.0, 0.1)
        relaxed = st.slider("Relaxed mate window (s)", 1.0, 10.0, 3.0, 0.5)
        suspect = st.slider("Suspect score", 1.0, 10.0, 2.0, 1.0)
        link = st.slider("Link score", 2.0, 20.0, 4.0, 1.0)
        half = st.slider("Decay half-life (s)", 30.0, 900.0, 180.0, 30.0)
        debounce = st.slider("Debounce same sensor (s)", 0.0, 1.0, 0.25, 0.05)

        st.header("Mate graph scope")
        sensor_filter = st.text_input("Optional sensor id contains", value="")
        max_events = st.number_input("Max events to replay", min_value=10_000, value=200_000, step=10_000)

    params = MateGraphParams(
        base_mate_window_s=float(base),
        relaxed_mate_window_s=float(relaxed),
        min_same_sensor_gap_s=float(debounce),
        suspect_score=float(suspect),
        link_score=float(link),
        decay_halflife_s=float(half),
    )

    min_ts, max_ts = bounds["min_ts"], bounds["max_ts"]
    if max_ts is None:
        st.info("No timestamps found.")
        return

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        lookback_min = st.number_input("Lookback (minutes)", min_value=1, value=60, step=15)
    with c2:
        top_n = st.number_input("Show top N pairs", min_value=10, value=250, step=10)
    with c3:
        min_score = st.number_input("Min raw score to display", min_value=0.0, value=float(params.suspect_score), step=1.0)

    end_ts = float(max_ts)
    start_ts = max(float(min_ts or 0.0), end_ts - float(lookback_min) * 60.0)

    st.caption(f"Replaying events from {pd.to_datetime(start_ts, unit='s')} → {pd.to_datetime(end_ts, unit='s')}  (max {int(max_events):,} events)")

    with st.spinner("Replaying signals to build mate graph…"):
        pairs_df, comps_df = compute_mate_graph(
            db_path=db_path,
            start_ts=start_ts,
            end_ts=end_ts,
            max_events=int(max_events),
            sensor_filter=str(sensor_filter),
            params=params,
        )

    k = st.columns(5)
    def _fmt_int(x: Optional[float]) -> str:
        try:
            if x is None:
                return "?"
            if isinstance(x, float) and math.isnan(x):
                return "?"
            return f"{int(x):,}"
        except Exception:
            return "?"

    k[0].metric("Signals in DB", _fmt_int(n_total))
    k[1].metric("Unique sensors", _fmt_int(n_sensors))
    k[2].metric("Pairs tracked", f"{len(pairs_df):,}" if not pairs_df.empty else "0")
    k[3].metric("Strong components", f"{len(comps_df):,}" if not comps_df.empty else "0")
    largest = comps_df["size"].max() if (not comps_df.empty and "size" in comps_df.columns) else None
    k[4].metric("Largest component", int(largest) if (largest is not None and pd.notna(largest)) else 0)

    st.markdown("### Top mate pairs")
    if pairs_df.empty:
        st.info("No mate pairs yet (try increasing lookback or lowering thresholds).")
    else:
        show = pairs_df[pairs_df["raw_score"] >= float(min_score)].head(int(top_n))
        st.dataframe(show, use_container_width=True, height=520)
        st.download_button(
            "Download shown mate pairs (CSV)",
            data=show.to_csv(index=False).encode("utf-8"),
            file_name="mate_pairs.csv",
            mime="text/csv",
        )

    st.markdown("### Strong components (likely vehicles)")
    if comps_df.empty:
        st.write("No strong components yet — you may need more confirmations (or reduce link score).")
    else:
        st.dataframe(comps_df.sort_values("size", ascending=False), use_container_width=True, height=320)
        st.download_button(
            "Download components (JSON)",
            data=json.dumps(comps_df.to_dict(orient="records"), indent=2).encode("utf-8"),
            file_name="mate_components.json",
            mime="application/json",
        )


# =============================================================================
# App
# =============================================================================

def choose_db() -> str:
    """Choose DB via sidebar text input or upload. Returns local filesystem path."""
    st.sidebar.header("Database")
    db_path = st.sidebar.text_input("SQLite DB path", value=DEFAULT_DB_PATH, help="Set TPMS_DB_PATH env var to avoid typing this every time.")
    upload = st.sidebar.file_uploader("…or upload a .db file", type=["db", "sqlite", "bak", "bak1"])

    if upload is not None:
        tmp_dir = st.session_state.get("_tpms_upload_dir")
        if not tmp_dir:
            tmp_dir = tempfile.mkdtemp(prefix="tpms_db_")
            st.session_state["_tpms_upload_dir"] = tmp_dir

        uploaded_path = os.path.join(tmp_dir, f"uploaded_{upload.name}")
        with open(uploaded_path, "wb") as f:
            f.write(upload.getbuffer())
        st.sidebar.success(f"Loaded uploaded DB: {upload.name}")
        db_path = uploaded_path

    return db_path


def main():
    st.set_page_config(page_title="TPMS DB Explorer", layout="wide")
    st.title("TPMS Tracker — DB Explorer")

    db_path = choose_db()

    if not db_path or not os.path.exists(db_path):
        st.warning("Point me at a SQLite DB file (left sidebar).")
        st.stop()

    try:
        con = connect_ro(db_path)
        tables = list_tables(con)
    except Exception as e:
        st.error(f"Could not open DB: {e}")
        st.stop()

    with st.sidebar:
        st.divider()
        page = st.radio(
            "View",
            [
                "How your app stores data",
                "DB overview",
                "Table explorer",
                "Signals dashboard",
                "Mate graph",
                "Vehicles & encounters",
                "Quality checks",
                "Export / clean",
            ],
            index=0,
        )

    try:
        if page == "How your app stores data":
            section_how_data_is_stored(tables, db_path)
        elif page == "DB overview":
            section_db_overview(con, tables, db_path)
        elif page == "Table explorer":
            section_table_explorer(con, tables, db_path)
        elif page == "Signals dashboard":
            section_signals_dashboard(tables, db_path)
        elif page == "Mate graph":
            section_mate_graph(con, db_path)
        elif page == "Vehicles & encounters":
            section_vehicles_encounters(tables, db_path)
        elif page == "Quality checks":
            section_quality_checks(tables, db_path)
        elif page == "Export / clean":
            section_export_clean(tables, db_path)
    finally:
        con.close()


if __name__ == "__main__":
    main()