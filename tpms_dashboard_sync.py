#!/usr/bin/env python3
"""
TPMS Dashboard Sync Watcher
============================
Watches ~/TPMS/exports/ for new/modified JSON files and:
  1. Logs additions to a changelog.
  2. Regenerates a static standalone HTML snapshot chart (no server needed).
  3. Can notify the Streamlit app to reload via a lightweight sentinel file.

Usage:
    python3 tpms_dashboard_sync.py [--exports-dir PATH] [--html-out PATH]

Run continuously as a background process or systemd service.
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── LOGGING ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SYNC] %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(__file__).parent / "tpms_dashboard_sync.log"),
    ],
)
log = logging.getLogger("tpms_sync")

# ─── DEFAULTS ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
EXPORTS_DIR  = SCRIPT_DIR / "exports"
HTML_OUT     = SCRIPT_DIR / "tpms_dashboard_snapshot.html"
SENTINEL     = SCRIPT_DIR / ".dashboard_reload_sentinel"
CHANGELOG    = SCRIPT_DIR / "tpms_sync_changelog.log"
POLL_SECS    = 15


# ─── HELPERS ─────────────────────────────────────────────────────────────────
def fingerprint_dir(directory: Path) -> str:
    files = sorted(directory.glob("*.json"))
    parts = [f"{p.name}:{p.stat().st_mtime}:{p.stat().st_size}" for p in files]
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def load_exports(directory: Path) -> pd.DataFrame:
    frames = []
    for fpath in sorted(directory.glob("*.json")):
        m = re.search(r"(\d{8})_(\d{6})", fpath.stem)
        export_dt = (
            datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
            if m else datetime.fromtimestamp(fpath.stat().st_mtime)
        )
        try:
            records = json.loads(fpath.read_text())
            df = pd.DataFrame(records)
            df["export_file"] = fpath.name
            df["export_dt"] = export_dt
            frames.append(df)
        except Exception as exc:
            log.warning(f"Could not parse {fpath.name}: {exc}")
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    for col in ("pressure_psi", "pressure_kpa", "temp_c", "temp_f", "rssi_dbm", "packet_count"):
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
    combined["export_dt"] = pd.to_datetime(combined["export_dt"])
    combined["session_label"] = combined["export_dt"].dt.strftime("%b %d %H:%M")
    return combined


def generate_snapshot_html(df: pd.DataFrame, out_path: Path) -> None:
    """Build a standalone, self-contained HTML file with key charts."""
    if df.empty:
        log.warning("No data — skipping HTML snapshot.")
        return

    df_latest = df.sort_values("export_dt").groupby("sensor_id", as_index=False).last()

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Latest Pressure (PSI) per Sensor",
            "Latest Temp (°C) per Sensor",
            "Pressure History Over Sessions",
            "Temperature History Over Sessions",
            "RSSI Distribution",
            "Packet Count by Protocol",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # Row 1: bar charts of latest values
    df_s = df_latest.sort_values("pressure_psi", ascending=False)
    colors = ["#ef4444" if p < 25 else "#f97316" if p < 32 else "#22c55e"
              for p in df_s["pressure_psi"].fillna(0)]
    fig.add_trace(go.Bar(x=df_s["sensor_id"], y=df_s["pressure_psi"],
                          marker_color=colors, name="PSI", showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=df_s["sensor_id"], y=df_s["temp_c"],
                          marker_color="#f97316", name="°C", showlegend=False), row=1, col=2)

    # Row 2: history lines
    df_agg = (
        df.groupby(["export_dt", "session_label"], as_index=False)
          .agg(avg_psi=("pressure_psi", "mean"), avg_temp=("temp_c", "mean"))
          .sort_values("export_dt")
    )
    fig.add_trace(go.Scatter(x=df_agg["session_label"], y=df_agg["avg_psi"],
                              mode="lines+markers", name="Avg PSI",
                              line=dict(color="#3b82f6")), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_agg["session_label"], y=df_agg["avg_temp"],
                              mode="lines+markers", name="Avg °C",
                              line=dict(color="#f97316")), row=2, col=2)

    # Row 3: RSSI histogram + packet box
    rssi_vals = df[df["rssi_dbm"] != 0]["rssi_dbm"].dropna()
    fig.add_trace(go.Histogram(x=rssi_vals, nbinsx=25,
                                name="RSSI", marker_color="#8b5cf6", showlegend=False), row=3, col=1)

    for proto, grp in df.groupby("protocol"):
        fig.add_trace(go.Box(y=grp["packet_count"], name=proto,
                              boxpoints="outliers"), row=3, col=2)

    fig.update_layout(
        height=1100,
        title_text=f"🛞 TPMS Dashboard Snapshot — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        title_font_size=18,
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font_color="#e2e8f0",
    )
    fig.update_xaxes(tickangle=-45, gridcolor="#334155")
    fig.update_yaxes(gridcolor="#334155")

    html_str = fig.to_html(full_html=True, include_plotlyjs="cdn")
    out_path.write_text(html_str)
    log.info(f"📄 Snapshot written → {out_path}  ({out_path.stat().st_size:,} bytes)")


def touch_sentinel(path: Path) -> None:
    """Notify Streamlit app that data changed."""
    path.write_text(datetime.now().isoformat())


def log_changelog(new_files: list[str], changelog: Path) -> None:
    with open(changelog, "a") as f:
        for fn in new_files:
            f.write(f"{datetime.now().isoformat()}  NEW_FILE  {fn}\n")


# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
def main(exports_dir: Path, html_out: Path) -> None:
    log.info(f"👀 Watching {exports_dir}  (poll every {POLL_SECS}s)")
    log.info(f"📄 HTML snapshot → {html_out}")

    known_fp = fingerprint_dir(exports_dir)
    known_files: set[str] = {p.name for p in exports_dir.glob("*.json")}

    # Initial snapshot
    df = load_exports(exports_dir)
    generate_snapshot_html(df, html_out)

    while True:
        time.sleep(POLL_SECS)
        current_fp = fingerprint_dir(exports_dir)
        if current_fp == known_fp:
            continue

        current_files = {p.name for p in exports_dir.glob("*.json")}
        new_files = sorted(current_files - known_files)

        if new_files:
            log.info(f"🆕 {len(new_files)} new export(s): {new_files}")
            log_changelog(new_files, CHANGELOG)

        modified = sorted(current_files & known_files)
        if modified:
            # Check which actually changed by re-hashing
            log.info(f"🔄 Detected changes in {len(current_files)} files — regenerating snapshot.")

        known_fp = current_fp
        known_files = current_files

        df = load_exports(exports_dir)
        generate_snapshot_html(df, html_out)
        touch_sentinel(SENTINEL)
        log.info("✅ Dashboard snapshot updated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TPMS Dashboard Sync Watcher")
    parser.add_argument("--exports-dir", type=Path, default=EXPORTS_DIR)
    parser.add_argument("--html-out",    type=Path, default=HTML_OUT)
    args = parser.parse_args()
    args.exports_dir.mkdir(parents=True, exist_ok=True)
    main(args.exports_dir, args.html_out)
