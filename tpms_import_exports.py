#!/usr/bin/env python3
"""
tpms_import_exports.py
======================
Imports tpms_sync_*.json files from ~/TPMS/exports/ into tpms_local.sqlite3.

Usage:
  python3 tpms_import_exports.py              # one-shot backfill
  python3 tpms_import_exports.py --watch      # continuous watcher
  python3 tpms_import_exports.py --status     # print DB state
  python3 tpms_import_exports.py --force      # reimport everything
"""

import argparse
import json
import logging
import sqlite3
import time
from datetime import datetime
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────
TPMS_HOME   = Path(__file__).resolve().parent
EXPORTS_DIR = TPMS_HOME / "exports"
SQLITE_DB   = TPMS_HOME / "tpms_local.sqlite3"
LOG_DIR     = TPMS_HOME / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ── logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [IMPORT] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "import_exports.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("tpms_import")

# ── schema migrations ─────────────────────────────────────────────────────
_MIGRATIONS = [
    "ALTER TABLE sensors ADD COLUMN magic_ok     INTEGER DEFAULT 1",
    "ALTER TABLE sensors ADD COLUMN suspect_temp INTEGER DEFAULT 0",
    """CREATE TABLE IF NOT EXISTS import_log (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        filename        TEXT UNIQUE NOT NULL,
        imported_at     TEXT NOT NULL,
        records_found   INTEGER DEFAULT 0,
        records_new     INTEGER DEFAULT 0,
        records_updated INTEGER DEFAULT 0,
        records_skipped INTEGER DEFAULT 0,
        source_tag      TEXT
    )""",
]


def open_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=15)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    for sql in _MIGRATIONS:
        try:
            conn.execute(sql)
        except sqlite3.OperationalError as e:
            if "duplicate column" not in str(e).lower() and "already exists" not in str(e).lower():
                raise
    conn.commit()
    return conn


def already_imported(conn: sqlite3.Connection, filename: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM import_log WHERE filename=?", [filename]
    ).fetchone() is not None


def import_json_file(conn: sqlite3.Connection, json_path: Path) -> dict:
    with open(json_path) as f:
        records = json.load(f)

    if not isinstance(records, list):
        log.warning("%s: unexpected JSON shape (not a list), skipping", json_path.name)
        return {"found": 0, "new": 0, "updated": 0, "skipped": 0}

    parts = json_path.stem.split("_")
    source_tag = f"json:{parts[2]}_{parts[3]}" if len(parts) >= 4 else f"json:{json_path.stem}"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_c = upd_c = skip_c = 0
    cur = conn.cursor()

    for rec in records:
        sid = str(rec.get("sensor_id", "")).strip().upper()
        if not sid:
            skip_c += 1
            continue

        protocol      = rec.get("protocol", "Unknown")
        first_seen    = rec.get("first_seen", "")
        last_seen     = rec.get("last_seen", "")
        first_seen_ts = int(rec.get("first_seen_ts") or 0)
        last_seen_ts  = int(rec.get("last_seen_ts")  or 0)
        pressure_psi  = rec.get("pressure_psi")
        pressure_kpa  = rec.get("pressure_kpa")
        temp_c        = rec.get("temp_c")
        temp_f        = rec.get("temp_f")
        battery_low   = 1 if rec.get("battery_low") else 0
        rssi_dbm      = rec.get("rssi_dbm")
        packet_count  = rec.get("packet_count", 0) or 0
        flags         = rec.get("flags", "0x00")
        magic_ok      = 1 if rec.get("magic_ok", True)    else 0
        suspect_temp  = 1 if rec.get("suspect_temp", False) else 0

        existing = cur.execute(
            "SELECT last_seen_ts, packet_count FROM sensors WHERE sensor_id=?", [sid]
        ).fetchone()

        if existing is None:
            cur.execute("""
                INSERT INTO sensors
                    (sensor_id, protocol, first_seen, last_seen,
                     first_seen_ts, last_seen_ts,
                     pressure_psi, pressure_kpa, temp_c, temp_f,
                     battery_low, rssi_dbm, packet_count, flags,
                     source, updated_at, magic_ok, suspect_temp)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (sid, protocol, first_seen, last_seen,
                  first_seen_ts, last_seen_ts,
                  pressure_psi, pressure_kpa, temp_c, temp_f,
                  battery_low, rssi_dbm, packet_count, flags,
                  source_tag, now, magic_ok, suspect_temp))
            new_c += 1

        elif last_seen_ts > existing[0]:
            extra_packets = max(0, packet_count - (existing[1] or 0))
            cur.execute("""
                UPDATE sensors SET
                    last_seen=?,    last_seen_ts=?,
                    pressure_psi=?, pressure_kpa=?,
                    temp_c=?,       temp_f=?,
                    battery_low=?,  rssi_dbm=?,
                    packet_count=packet_count + ?,
                    flags=?,        source=?,
                    updated_at=?,   magic_ok=?,  suspect_temp=?
                WHERE sensor_id=?
            """, (last_seen, last_seen_ts,
                  pressure_psi, pressure_kpa,
                  temp_c, temp_f,
                  battery_low, rssi_dbm,
                  extra_packets,
                  flags, source_tag,
                  now, magic_ok, suspect_temp,
                  sid))
            upd_c += 1
        else:
            skip_c += 1

    conn.commit()
    conn.execute("""
        INSERT OR REPLACE INTO import_log
            (filename, imported_at, records_found,
             records_new, records_updated, records_skipped, source_tag)
        VALUES (?,?,?,?,?,?,?)
    """, (json_path.name, now, len(records), new_c, upd_c, skip_c, source_tag))
    conn.commit()
    return {"found": len(records), "new": new_c, "updated": upd_c, "skipped": skip_c}


def run_backfill(db_path=SQLITE_DB, exports_dir=EXPORTS_DIR, force=False):
    conn = open_db(db_path)
    json_files = sorted(exports_dir.glob("tpms_sync_*.json"))

    if not json_files:
        log.warning("No tpms_sync_*.json files found in %s", exports_dir)
        conn.close()
        return {"files": 0, "total_new": 0, "total_updated": 0}

    total_new = total_upd = total_skip = 0
    imported = already_done = 0

    for jf in json_files:
        if not force and already_imported(conn, jf.name):
            already_done += 1
            continue
        log.info("Importing %s ...", jf.name)
        try:
            counts = import_json_file(conn, jf)
            log.info("  new=%-4d  updated=%-4d  skipped=%d  (of %d records)",
                     counts["new"], counts["updated"], counts["skipped"], counts["found"])
            total_new  += counts["new"]
            total_upd  += counts["updated"]
            total_skip += counts["skipped"]
            imported   += 1
        except Exception as e:
            log.error("  ERROR in %s: %s", jf.name, e, exc_info=True)

    total_sensors = conn.execute("SELECT COUNT(*) FROM sensors").fetchone()[0]
    conn.close()

    summary = {
        "files_imported":    imported,
        "files_already_done": already_done,
        "total_new":         total_new,
        "total_updated":     total_upd,
        "total_skipped":     total_skip,
        "sensors_in_db":     total_sensors,
    }
    log.info("Backfill complete: %s", summary)
    return summary


def watch_loop(db_path=SQLITE_DB, exports_dir=EXPORTS_DIR, poll_interval=15):
    log.info("Watch mode started — polling %s every %ds", exports_dir, poll_interval)
    conn = open_db(db_path)
    while True:
        for jf in sorted(exports_dir.glob("tpms_sync_*.json")):
            if not already_imported(conn, jf.name):
                log.info("New file: %s", jf.name)
                try:
                    counts = import_json_file(conn, jf)
                    log.info("  Imported: new=%d updated=%d skipped=%d",
                             counts["new"], counts["updated"], counts["skipped"])
                except Exception as e:
                    log.error("  ERROR: %s: %s", jf.name, e, exc_info=True)
        time.sleep(poll_interval)


def print_status(db_path=SQLITE_DB):
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return
    conn = open_db(db_path)
    total   = conn.execute("SELECT COUNT(*) FROM sensors").fetchone()[0]
    sources = conn.execute(
        "SELECT source, COUNT(*) FROM sensors GROUP BY source ORDER BY COUNT(*) DESC"
    ).fetchall()
    latest  = conn.execute(
        "SELECT sensor_id, protocol, pressure_psi, last_seen "
        "FROM sensors ORDER BY last_seen_ts DESC LIMIT 5"
    ).fetchall()
    log_rows = conn.execute(
        "SELECT filename, imported_at, records_new, records_updated "
        "FROM import_log ORDER BY imported_at DESC LIMIT 20"
    ).fetchall()
    conn.close()

    bar = "=" * 62
    print(f"\n{bar}")
    print(f"  DB: {db_path}")
    print(f"  Total sensors: {total:,}")
    print(f"\n  By source:")
    for src, cnt in sources:
        print(f"    {src or 'unknown':<45} {cnt:>5} sensors")
    print(f"\n  Latest 5 sensors by last_seen:")
    for r in latest:
        print(f"    {r[0]:<12} {r[1]:<25} {str(r[2]) + ' PSI':<10}  {r[3]}")
    if log_rows:
        print(f"\n  Import log (most recent {len(log_rows)} files):")
        for r in log_rows:
            print(f"    {r[1]}  {r[0]:<55} +{r[2]} new  ~{r[3]} upd")
    print(f"{bar}\n")


# ── entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Import TPMS JSON exports into SQLite")
    p.add_argument("--watch",    action="store_true", help="Poll for new files continuously")
    p.add_argument("--force",    action="store_true", help="Reimport all files")
    p.add_argument("--status",   action="store_true", help="Print DB status and exit")
    p.add_argument("--db",       default=str(SQLITE_DB),   help="Path to tpms_local.sqlite3")
    p.add_argument("--exports",  default=str(EXPORTS_DIR), help="Path to exports/ directory")
    p.add_argument("--interval", default=15, type=int,     help="Poll interval seconds (--watch)")
    args = p.parse_args()

    db_path      = Path(args.db)
    exports_path = Path(args.exports)

    if args.status:
        print_status(db_path)
    elif args.watch:
        run_backfill(db_path, exports_path, force=args.force)
        watch_loop(db_path, exports_path, args.interval)
    else:
        run_backfill(db_path, exports_path, force=args.force)
        print_status(db_path)
