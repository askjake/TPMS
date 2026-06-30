#!/usr/bin/env python3
"""
hackrf_sd_sync.py
=================
HackRF Portapack SD Card -> Local TPMS DB Sync Tool

Triggered automatically via udev when the Portapack SD card is mounted,
or run manually.

Pipeline:
  1. Detect HackRF Portapack SD card mount point (auto or forced)
  2. Copy sensors.db locally (timestamped backup)
  3. Decode all 36-byte SensorRecord binary structs
  4. Merge into local SQLite + binary sensors.db (dedup by sensor_id, keep freshest)
  5. Export CSV/JSON summary
  6. Clean sensors.db off the SD card (leaves README.txt intact)

SensorRecord layout — 36 bytes, all little-endian:
  Offset  Size  Type      Field
  0x00      4   uint32    sensor_id
  0x04      1   uint8     proto_len  (always 4)
  0x05      3   uint8[3]  proto_tag  ('rad'=Schrader, etc.)
  0x08      4   uint32    first_seen  (unix timestamp)
  0x0C      4   uint32    last_seen   (unix timestamp)
  0x10      1   uint8     pressure_raw  (PSI = raw * 0.25)
  0x11      1   uint8     temp_raw      (Celsius = raw - 40)
  0x12      1   uint8     flags         (bit0 = battery_low)
  0x13      1   uint8     reserved
  0x14      4   uint32    reserved2
  0x18      1   uint8     rssi_mag      (dBm = -rssi_mag)
  0x19      3   uint8[3]  extra
  0x1C      4   uint32    packet_count
  0x20      4   uint32    magic         (always 0x12C684C0)
  -------  36 bytes total

NOTE: The README.txt on the SD card says "48 bytes" — that is incorrect.
The actual binary is 36 bytes per record (verified empirically).

Usage:
  python3 hackrf_sd_sync.py                        # auto-detect SD + full sync
  python3 hackrf_sd_sync.py --mount /media/user/X  # force mount point
  python3 hackrf_sd_sync.py --decode-only FILE      # decode + print; no sync
  python3 hackrf_sd_sync.py --list-sensors          # read-only table view
  python3 hackrf_sd_sync.py --dry-run               # preview without writing
  python3 hackrf_sd_sync.py --no-cleanup            # keep sensors.db on SD

Author: montjac / askjake
Updated: 2026-06-05
"""

import os
import sys
import struct
import time
import shutil
import sqlite3
import json
import logging
import glob
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RECORD_SIZE   = 36           # empirically verified — 36 bytes per SensorRecord
RECORD_MAGIC  = 0x12C684C0
SD_TPMS_DIR   = "TPMS"
SD_DB_FILE    = "sensors.db"
SD_VOLUME_ID  = "0403-0201"  # FAT volume label on Portapack SD card
# USB identity of the Portapack mass-storage interface (verified via udevadm)
SD_USB_VENDOR  = "0781"
SD_USB_PRODUCT = "a7a8"
SD_MODEL_HINT  = "Portapack"

PROTO_NAMES = {
    b"rad":                        "Schrader",
    bytes([0x7F, 0x00, 0x20]):     "Schrader-EV1",
    bytes([0x88, 0x00, 0x20]):     "Schrader-Alt",
    bytes([0x00, 0x00, 0x00]):     "Unknown-Zero",
}

# Dirs relative to this script
TPMS_HOME  = Path(__file__).resolve().parent
LOCAL_DB   = TPMS_HOME / "sensors.db"
SQLITE_DB  = TPMS_HOME / "tpms_local.sqlite3"
BACKUP_DIR = TPMS_HOME / "data" / "sd_backups"
EXPORT_DIR = TPMS_HOME / "exports"
LOG_DIR    = TPMS_HOME / "logs"

SD_MOUNT_HINTS = ["/media/montjac", "/media", "/mnt", "/run/media"]

# ---------------------------------------------------------------------------
# Bootstrap dirs
# ---------------------------------------------------------------------------
for _d in (LOG_DIR, BACKUP_DIR, EXPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "hackrf_sd_sync.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("hackrf_sd_sync")


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------
@dataclass
class SensorRecord:
    sensor_id:    int
    proto_tag:    bytes
    first_seen:   int
    last_seen:    int
    pressure_raw: int
    temp_raw:     int
    flags:        int
    reserved:     int
    reserved2:    int
    rssi_mag:     int
    extra:        bytes
    packet_count: int
    magic:        int

    # ---- human-readable properties ----

    @property
    def protocol(self) -> str:
        return PROTO_NAMES.get(self.proto_tag,
                               f"Unknown({self.proto_tag.hex()})")

    @property
    def pressure_psi(self) -> float:
        return round(self.pressure_raw * 0.25, 2)

    @property
    def pressure_kpa(self) -> float:
        return round(self.pressure_psi * 6.89476, 2)

    @property
    def temp_c(self) -> int:
        return self.temp_raw - 40

    @property
    def temp_f(self) -> float:
        return round(self.temp_c * 9 / 5 + 32, 1)

    @property
    def battery_low(self) -> bool:
        return bool(self.flags & 0x01)

    @property
    def suspect_temp(self) -> bool:
        """True when temp_raw decodes outside the typical automotive range."""
        return not (-40 <= self.temp_c <= 125)

    @property
    def rssi_dbm(self) -> int:
        return -self.rssi_mag

    def _fmt_ts(self, ts: int) -> str:
        if 1_000_000_000 < ts < 2_100_000_000:
            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        return f"epoch:{ts}"

    @property
    def first_seen_dt(self) -> str:
        return self._fmt_ts(self.first_seen)

    @property
    def last_seen_dt(self) -> str:
        return self._fmt_ts(self.last_seen)

    @property
    def valid_magic(self) -> bool:
        return self.magic == RECORD_MAGIC

    def to_dict(self) -> dict:
        return {
            "sensor_id":    f"{self.sensor_id:08X}",
            "protocol":     self.protocol,
            "first_seen":   self.first_seen_dt,
            "last_seen":    self.last_seen_dt,
            "first_seen_ts": self.first_seen,
            "last_seen_ts": self.last_seen,
            "pressure_psi": self.pressure_psi,
            "pressure_kpa": self.pressure_kpa,
            "temp_c":       self.temp_c,
            "temp_f":       self.temp_f,
            "battery_low":  self.battery_low,
            "rssi_dbm":     self.rssi_dbm,
            "packet_count": self.packet_count,
            "flags":        f"0x{self.flags:02X}",
            "magic_ok":     self.valid_magic,
            "suspect_temp": self.suspect_temp,
        }

    def repack(self) -> bytes:
        """Serialize back to the canonical 36-byte binary representation."""
        pt  = (self.proto_tag + b'\x00\x00\x00')[:3]
        ex  = (self.extra      + b'\x00\x00\x00')[:3]
        return struct.pack(
            "<I"      # sensor_id           4
            "BBBB"    # proto_len + 3 tag   4
            "II"      # first/last_seen     8
            "BBBB"    # pressure,temp,flags,reserved  4
            "I"       # reserved2           4
            "BBBB"    # rssi_mag + 3 extra  4
            "I"       # packet_count        4
            "I",      # magic               4   => total 36
            self.sensor_id,
            4, pt[0], pt[1], pt[2],
            self.first_seen, self.last_seen,
            self.pressure_raw, self.temp_raw, self.flags, self.reserved,
            self.reserved2,
            self.rssi_mag, ex[0], ex[1], ex[2],
            self.packet_count,
            self.magic,
        )


# ---------------------------------------------------------------------------
# Binary decoder
# ---------------------------------------------------------------------------
_UNPACK = struct.Struct("<I BBBB II BBBB I BBBB I I")
assert _UNPACK.size == RECORD_SIZE, \
    f"BUG: struct size {_UNPACK.size} != RECORD_SIZE {RECORD_SIZE}"


def _decode_chunk(chunk: bytes) -> Optional[SensorRecord]:
    """Decode a single 36-byte chunk into a SensorRecord; None if bad magic."""
    if len(chunk) < RECORD_SIZE:
        return None
    (sensor_id,
     proto_len, p0, p1, p2,
     first_seen, last_seen,
     pressure_raw, temp_raw, flags, reserved,
     reserved2,
     rssi_mag, e0, e1, e2,
     packet_count,
     magic) = _UNPACK.unpack_from(chunk, 0)

    rec = SensorRecord(
        sensor_id    = sensor_id,
        proto_tag    = bytes([p0, p1, p2]),
        first_seen   = first_seen,
        last_seen    = last_seen,
        pressure_raw = pressure_raw,
        temp_raw     = temp_raw,
        flags        = flags,
        reserved     = reserved,
        reserved2    = reserved2,
        rssi_mag     = rssi_mag,
        extra        = bytes([e0, e1, e2]),
        packet_count = packet_count,
        magic        = magic,
    )
    return rec if rec.valid_magic else None


def decode_db(data: bytes) -> Tuple[List[SensorRecord], int]:
    """Decode raw bytes -> (valid_records, bad_count)."""
    records, bad = [], 0
    leftover = len(data) % RECORD_SIZE
    if leftover:
        log.warning(f"File size {len(data)} is not a multiple of {RECORD_SIZE}; "
                    f"{leftover} trailing bytes skipped.")
    total = len(data) // RECORD_SIZE
    for i in range(total):
        chunk = data[i * RECORD_SIZE : (i + 1) * RECORD_SIZE]
        rec = _decode_chunk(chunk)
        if rec:
            records.append(rec)
        else:
            bad += 1
    return records, bad


def decode_db_safe(path: Path) -> List[SensorRecord]:
    """Read and decode a sensors.db file; returns empty list on error."""
    try:
        data = path.read_bytes()
    except OSError as e:
        log.error(f"Cannot read {path}: {e}")
        return []
    if not data:
        log.warning(f"{path} is empty.")
        return []
    records, bad = decode_db(data)
    log.info(f"Decoded {len(records)} valid / {bad} bad-magic records from {path.name}")
    return records


# ---------------------------------------------------------------------------
# SD card detection
# ---------------------------------------------------------------------------
def find_sd_mount() -> Optional[Path]:
    """
    Locate the HackRF Portapack SD card.
    Priority:
      1. SD_CARD_PATH env var (set by udev/systemd)
      2. /proc/mounts scan for volume label
      3. Filesystem walk under mount hints
    """
    # 1. env var (udev path)
    env_path = os.environ.get("SD_CARD_PATH")
    if env_path:
        p = Path(env_path)
        if p.is_dir() and (p / SD_TPMS_DIR / SD_DB_FILE).exists():
            log.info(f"SD card from $SD_CARD_PATH: {p}")
            return p

    # 2. /proc/mounts
    try:
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    mnt = parts[1]
                    if SD_VOLUME_ID in mnt:
                        p = Path(mnt)
                        if (p / SD_TPMS_DIR / SD_DB_FILE).exists():
                            log.info(f"SD card via /proc/mounts: {p}")
                            return p
    except OSError as exc:
        log.debug(f"/proc/mounts scan error: {exc}")

    # 3. Filesystem walk
    for hint in SD_MOUNT_HINTS:
        pattern = f"{hint}/**/{SD_TPMS_DIR}/{SD_DB_FILE}"
        hits = glob.glob(pattern, recursive=True)
        if hits:
            mount = Path(hits[0]).parent.parent
            log.info(f"SD card via filesystem scan: {mount}")
            return mount

    log.error("HackRF SD card not found. Is it mounted?")
    return None


# ---------------------------------------------------------------------------
# Local SQLite DB
# ---------------------------------------------------------------------------
_SCHEMA = """
CREATE TABLE IF NOT EXISTS sensors (
    sensor_id       TEXT PRIMARY KEY,
    protocol        TEXT,
    first_seen      TEXT,
    last_seen       TEXT,
    first_seen_ts   INTEGER,
    last_seen_ts    INTEGER,
    pressure_psi    REAL,
    pressure_kpa    REAL,
    temp_c          INTEGER,
    temp_f          REAL,
    battery_low     INTEGER,
    rssi_dbm        INTEGER,
    packet_count    INTEGER,
    flags           TEXT,
    source          TEXT,
    updated_at      TEXT,
    magic_ok        INTEGER DEFAULT 1,
    suspect_temp    INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS sync_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    synced_at       TEXT,
    sd_path         TEXT,
    records_found   INTEGER,
    records_new     INTEGER,
    records_updated INTEGER,
    records_skipped INTEGER
);
"""


# Columns that may have been added after initial schema; used by _migrate_schema
_EXPECTED_COLUMNS = {
    "magic_ok":     "INTEGER DEFAULT 1",
    "suspect_temp": "INTEGER DEFAULT 0",
}


def _migrate_schema(conn: sqlite3.Connection):
    """Auto-merge schema discrepancies: add any missing columns to sensors table."""
    cur = conn.execute("PRAGMA table_info(sensors)")
    existing_cols = {row[1] for row in cur.fetchall()}
    for col_name, col_def in _EXPECTED_COLUMNS.items():
        if col_name not in existing_cols:
            stmt = f"ALTER TABLE sensors ADD COLUMN {col_name} {col_def}"
            conn.execute(stmt)
            log.info(f"Schema migration: added column '{col_name}' ({col_def})")
    conn.commit()


def open_local_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=10)
    conn.executescript(_SCHEMA)
    _migrate_schema(conn)
    conn.commit()
    return conn


def merge_records(conn: sqlite3.Connection,
                  records: List[SensorRecord],
                  source: str = "sd_card") -> Tuple[int, int, int]:
    """Upsert records into SQLite. Returns (new, updated, skipped)."""
    new_c = upd_c = skip_c = 0
    cur = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for rec in records:
        sid = f"{rec.sensor_id:08X}"
        cur.execute(
            "SELECT last_seen_ts, packet_count FROM sensors WHERE sensor_id=?",
            (sid,)
        )
        row = cur.fetchone()

        if row is None:
            cur.execute(
                "INSERT INTO sensors "
                "(sensor_id, protocol, first_seen, last_seen, "
                " first_seen_ts, last_seen_ts, pressure_psi, pressure_kpa, "
                " temp_c, temp_f, battery_low, rssi_dbm, "
                " packet_count, flags, source, updated_at, "
                " magic_ok, suspect_temp) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (sid, rec.protocol,
                 rec.first_seen_dt, rec.last_seen_dt,
                 rec.first_seen,    rec.last_seen,
                 rec.pressure_psi,  rec.pressure_kpa,
                 rec.temp_c,        rec.temp_f,
                 int(rec.battery_low), rec.rssi_dbm,
                 rec.packet_count,  f"0x{rec.flags:02X}",
                 source, now,
                 int(rec.valid_magic), int(rec.suspect_temp))
            )
            new_c += 1
        else:
            existing_ts = row[0]
            if rec.last_seen > existing_ts:
                cur.execute("""
                    UPDATE sensors SET
                        last_seen=?,     last_seen_ts=?,
                        pressure_psi=?,  pressure_kpa=?,
                        temp_c=?,        temp_f=?,
                        battery_low=?,   rssi_dbm=?,
                        packet_count=packet_count + ?,
                        flags=?,         source=?,   updated_at=?
                    WHERE sensor_id=?
                """, (rec.last_seen_dt, rec.last_seen,
                      rec.pressure_psi,  rec.pressure_kpa,
                      rec.temp_c,        rec.temp_f,
                      int(rec.battery_low), rec.rssi_dbm,
                      rec.packet_count,
                      f"0x{rec.flags:02X}", source, now,
                      sid))
                upd_c += 1
            else:
                skip_c += 1

    conn.commit()
    return new_c, upd_c, skip_c


def log_sync_event(conn: sqlite3.Connection, sd_path: str,
                   found: int, new: int, updated: int, skipped: int):
    conn.execute(
        "INSERT INTO sync_log "
        "(synced_at,sd_path,records_found,records_new,records_updated,records_skipped) "
        "VALUES (?,?,?,?,?,?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
         sd_path, found, new, updated, skipped)
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Binary merge
# ---------------------------------------------------------------------------
def merge_binary(local_path: Path,
                 new_records: List[SensorRecord],
                 dry_run: bool = False) -> Tuple[int, int]:
    """Merge new_records into local binary sensors.db. Returns (added, updated)."""
    if local_path.exists():
        raw = local_path.read_bytes()
        existing_list, _ = decode_db(raw)
        existing = {r.sensor_id: r for r in existing_list}
    else:
        log.info(f"Local binary db not found; will create: {local_path}")
        existing = {}

    added = updated = 0
    for rec in new_records:
        old = existing.get(rec.sensor_id)
        if old is None:
            existing[rec.sensor_id] = rec
            added += 1
        elif rec.last_seen > old.last_seen:
            existing[rec.sensor_id] = rec
            updated += 1

    if not dry_run:
        blob = bytearray()
        for rec in sorted(existing.values(), key=lambda r: r.sensor_id):
            packed = rec.repack()
            assert len(packed) == RECORD_SIZE, \
                f"BUG: repack() returned {len(packed)} bytes for {rec.sensor_id:08X}"
            blob += packed
        tmp = local_path.with_suffix(".tmp")
        tmp.write_bytes(bytes(blob))
        tmp.replace(local_path)          # atomic rename
        log.info(
            f"Binary db updated: {local_path.name}  "
            f"total={len(existing)}  +new={added}  updated={updated}"
        )
    else:
        log.info(f"[DRY-RUN] Binary merge: would +{added} new, ~{updated} updated")

    return added, updated


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export_records(records: List[SensorRecord], tag: str) -> Tuple[Path, Path]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = EXPORT_DIR / f"tpms_sync_{tag}_{ts}.json"
    json_path.write_text(
        json.dumps([r.to_dict() for r in records], indent=2), encoding="utf-8"
    )
    log.info(f"JSON export: {json_path}")

    csv_path = EXPORT_DIR / f"tpms_sync_{tag}_{ts}.csv"
    headers = ["sensor_id", "protocol", "first_seen", "last_seen",
               "pressure_psi", "pressure_kpa", "temp_c", "temp_f",
               "battery_low", "rssi_dbm", "packet_count", "flags", "magic_ok"]
    lines = [",".join(headers)]
    for r in records:
        d = r.to_dict()
        lines.append(",".join(str(d[h]) for h in headers))
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info(f"CSV  export: {csv_path}")

    return json_path, csv_path


# ---------------------------------------------------------------------------
# Pretty-print summary
# ---------------------------------------------------------------------------
def print_summary(records: List[SensorRecord]):
    W = 88
    print("\n" + "=" * W)
    print(f"  TPMS SENSOR SUMMARY  —  {len(records)} records  "
          f"(36-byte SensorRecord structs, magic=0x12C684C0)")
    print("=" * W)
    hdr = (f"  {'SensorID':>10}  {'Protocol':<16} {'Last Seen':<19}"
           f"  {'PSI':>6}  {'°C':>4}  {'°F':>5}  {'dBm':>4}  {'Pkts':>5}  Bat   Flags")
    print(hdr)
    print("  " + "-" * (W - 2))
    for r in sorted(records, key=lambda x: x.sensor_id):
        bat = "⚠ LOW" if r.battery_low else "OK"
        print(f"  {r.sensor_id:08X}  {r.protocol:<16} {r.last_seen_dt:<19}"
              f"  {r.pressure_psi:>5.1f}  {r.temp_c:>4d}  {r.temp_f:>5.1f}"
              f"  {r.rssi_dbm:>4d}  {r.packet_count:>5d}  {bat:<5} {hex(r.flags)}")
    print("=" * W + "\n")


# ---------------------------------------------------------------------------
# SD card cleanup
# ---------------------------------------------------------------------------
def cleanup_sd(sd_tpms_dir: Path, dry_run: bool = False) -> List[str]:
    """Remove sensors.db (and sensors.tmp) from SD card; preserves README.txt."""
    removed = []
    for fname in ("sensors.db", "sensors.tmp"):
        target = sd_tpms_dir / fname
        if target.exists():
            if dry_run:
                log.info(f"[DRY-RUN] Would delete: {target}")
            else:
                target.unlink()
                log.info(f"Deleted from SD card: {target}")
            removed.append(str(target))
        else:
            log.debug(f"Not found on SD (skip): {target}")
    return removed


# ---------------------------------------------------------------------------
# Main sync pipeline
# ---------------------------------------------------------------------------
def run_sync(
    dry_run:    bool = False,
    no_cleanup: bool = False,
    no_export:  bool = False,
    force_mount: Optional[str] = None,
) -> List[SensorRecord]:
    """Execute the full sync pipeline. Returns decoded records."""
    banner = "=" * 60
    log.info(banner)
    log.info("  HackRF Portapack SD Card — TPMS Sync")
    log.info(banner)

    # ── 1. Locate SD card ────────────────────────────────────────────────
    sd_root = Path(force_mount) if force_mount else find_sd_mount()
    if sd_root is None:
        log.error("Aborting: SD card not found or not mounted.")
        sys.exit(1)

    sd_tpms = sd_root / SD_TPMS_DIR
    sd_db   = sd_tpms / SD_DB_FILE

    if not sd_db.exists():
        log.error(f"sensors.db not found at: {sd_db}")
        sys.exit(1)

    log.info(f"SD root:     {sd_root}")
    log.info(f"sensors.db:  {sd_db}  ({sd_db.stat().st_size} bytes, "
             f"{sd_db.stat().st_size // RECORD_SIZE} max records)")

    # ── 2. Backup ─────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"sensors_sd_{ts}.db"
    if not dry_run:
        shutil.copy2(sd_db, backup_path)
        log.info(f"Backup:      {backup_path}")
    else:
        log.info(f"[DRY-RUN] Would backup to: {backup_path}")

    # ── 3. Decode ─────────────────────────────────────────────────────────
    records = decode_db_safe(sd_db)
    if not records:
        log.warning("No valid records decoded — nothing to sync.")
        sys.exit(0)

    print_summary(records)

    # ── 4. Merge into SQLite ──────────────────────────────────────────────
    if not dry_run:
        conn = open_local_db(SQLITE_DB)
        new, updated, skipped = merge_records(conn, records, source=f"sd:{ts}")
        log_sync_event(conn, str(sd_root), len(records), new, updated, skipped)
        conn.close()
        log.info(f"SQLite:      +{new} new  ~{updated} updated  ={skipped} unchanged")
    else:
        log.info("[DRY-RUN] Skipping SQLite merge.")

    # ── 5. Merge into binary sensors.db ──────────────────────────────────
    b_added, b_updated = merge_binary(LOCAL_DB, records, dry_run=dry_run)

    # ── 6. Export JSON / CSV ──────────────────────────────────────────────
    if not no_export and not dry_run:
        export_records(records, tag=ts)
    elif dry_run:
        log.info("[DRY-RUN] Skipping export.")

    # ── 7. Cleanup SD card ────────────────────────────────────────────────
    if not no_cleanup:
        removed = cleanup_sd(sd_tpms, dry_run=dry_run)
        if removed:
            log.info(f"SD cleanup:  removed {len(removed)} file(s)")
    else:
        log.info("SD cleanup:  skipped (--no-cleanup)")

    log.info(banner)
    log.info(f"  Sync complete — {len(records)} records processed")
    log.info(banner)
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="hackrf_sd_sync — HackRF Portapack SD card TPMS sync tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Preview all actions without writing or deleting anything",
    )
    ap.add_argument(
        "--no-cleanup", action="store_true",
        help="Keep sensors.db on SD card after sync",
    )
    ap.add_argument(
        "--no-export", action="store_true",
        help="Skip JSON/CSV export",
    )
    ap.add_argument(
        "--mount", metavar="PATH",
        help="Force SD card mount point (skip auto-detection)",
    )
    ap.add_argument(
        "--decode-only", metavar="FILE",
        help="Decode a sensors.db file and print summary; no sync performed",
    )
    ap.add_argument(
        "--list-sensors", action="store_true",
        help="Decode SD card sensors.db and show table; no modifications",
    )
    args = ap.parse_args()

    if args.decode_only:
        p = Path(args.decode_only)
        if not p.exists():
            sys.exit(f"ERROR: File not found: {p}")
        recs = decode_db_safe(p)
        print_summary(recs)
        sys.exit(0)

    if args.list_sensors:
        mount = Path(args.mount) if args.mount else find_sd_mount()
        if mount is None:
            sys.exit(1)
        recs = decode_db_safe(mount / SD_TPMS_DIR / SD_DB_FILE)
        print_summary(recs)
        sys.exit(0)

    run_sync(
        dry_run    = args.dry_run,
        no_cleanup = args.no_cleanup,
        no_export  = args.no_export,
        force_mount= args.mount,
    )


if __name__ == "__main__":
    main()
