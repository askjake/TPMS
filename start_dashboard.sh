#!/usr/bin/env bash
# start_dashboard.sh  –  Start TPMS dashboard + sync watcher in the background
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/venv/bin"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "[TPMS] Starting sync watcher..."
nohup "$VENV/python3" "$SCRIPT_DIR/tpms_dashboard_sync.py" \
    > "$LOG_DIR/sync_watcher.log" 2>&1 &
echo "[TPMS] Sync watcher PID=$!"

echo "[TPMS] Starting Streamlit dashboard on 0.0.0.0:8501..."
exec "$VENV/streamlit" run "$SCRIPT_DIR/tpms_dashboard.py" \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --server.headless true \
    --server.fileWatcherType none \
    --browser.gatherUsageStats false
