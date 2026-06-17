#!/usr/bin/env python3
"""
TPMS DB Recovery (best-effort)

This helps when you see errors like:
  sqlite3.DatabaseError: database disk image is malformed

It streams rows from tpms_signals until SQLite hits a corrupt page, writes what it can
into a new DB, and recreates indexes.

Usage:
  python tpms_db_recover.py path\to\tpms_tracker.db
  python tpms_db_recover.py path\to\tpms_tracker.db --out tpms_tracker.recovered.db

Notes:
- This cannot magically restore rows that live inside corrupt pages. It salvages everything
  SQLite can still read.
- Always keep your original file. This script never modifies it.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from typing import List, Tuple


def quick_check(db_path: str) -> List[str]:
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute("PRAGMA quick_check;")
        return [r[0] for r in cur.fetchall()]
    finally:
        con.close()


def get_schema_tables(db_path: str) -> List[Tuple[str, str]]:
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL;"
        )
        return [(n, s) for (n, s) in cur.fetchall()]
    finally:
        con.close()


def recover_tpms_signals(src_path: str, dst_path: str, chunk_size: int = 5000) -> None:
    if os.path.exists(dst_path):
        raise SystemExit(f"Output already exists: {dst_path}")

    src = sqlite3.connect(src_path)
    dst = sqlite3.connect(dst_path)

    try:
        # Create tables
        for name, sql in get_schema_tables(src_path):
            dst.execute(sql)
        dst.commit()

        # Stream-copy tpms_signals (this is usually the big table)
        cur = src.cursor()
        cur.execute("SELECT * FROM tpms_signals;")
        col_count = len(cur.description)
        ins = f"INSERT INTO tpms_signals VALUES ({','.join(['?']*col_count)});"

        copied = 0
        t0 = time.time()
        while True:
            try:
                rows = cur.fetchmany(chunk_size)
            except sqlite3.DatabaseError as e:
                print(f"[WARN] Stopped at corrupt page after {copied:,} rows: {e}")
                break
            if not rows:
                break
            dst.executemany(ins, rows)
            copied += len(rows)
            if copied % 50_000 == 0:
                dst.commit()
                print(f"[OK] copied {copied:,} rows...")

        dst.commit()
        print(f"[OK] Finished copying {copied:,} rows in {time.time()-t0:.1f}s")

        # Copy small tables (if any)
        for t in ("vehicles", "encounters", "maintenance_history"):
            try:
                rows = src.execute(f"SELECT * FROM {t};").fetchall()
            except Exception:
                rows = []
            if rows:
                coln = len(rows[0])
                dst.executemany(f"INSERT INTO {t} VALUES ({','.join(['?']*coln)});", rows)
                dst.commit()

        # Recreate indexes (fresh)
        idx_sql = [
            "CREATE INDEX IF NOT EXISTS idx_tpms_id ON tpms_signals(tpms_id);",
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON tpms_signals(timestamp);",
            "CREATE INDEX IF NOT EXISTS idx_tpms_id_timestamp ON tpms_signals(tpms_id, timestamp);",
        ]
        for s in idx_sql:
            dst.execute(s)
        dst.commit()

    finally:
        dst.close()
        src.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("db", help="Source (possibly corrupted) SQLite DB")
    ap.add_argument("--out", default=None, help="Recovered DB output path")
    ap.add_argument("--chunk", type=int, default=5000, help="Rows per chunk (default: 5000)")
    args = ap.parse_args()

    src = args.db
    if not os.path.exists(src):
        raise SystemExit(f"Not found: {src}")

    out = args.out or (os.path.splitext(src)[0] + ".recovered.db")

    qc = quick_check(src)
    if qc == ["ok"]:
        print("[OK] quick_check: ok (DB looks healthy)")
    else:
        print("[WARN] quick_check is NOT ok:")
        for line in qc:
            print("  -", line)

    print(f"[INFO] Recovering -> {out}")
    recover_tpms_signals(src, out, chunk_size=args.chunk)

    qc2 = quick_check(out)
    if qc2 == ["ok"]:
        print("[OK] recovered quick_check: ok")
    else:
        print("[WARN] recovered quick_check not ok:")
        for line in qc2:
            print("  -", line)


if __name__ == "__main__":
    main()

