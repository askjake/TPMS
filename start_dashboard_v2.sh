#!/usr/bin/env bash
# start_dashboard_v2.sh - Start the performance-optimized TPMS dashboard
# This version handles large tracker archives (100K+ sensors) without hanging.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/venv/bin"
PORT="${TPMS_PORT:-8569}"

if [[ ! -x "$VENV/streamlit" ]]; then
    echo "[ERROR] $VENV/streamlit not found. Run: python3 -m venv venv && venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi

# Kill any existing dashboard on this port
lsof -t -i:"$PORT" 2>/dev/null | xargs -r kill 2>/dev/null
sleep 1

echo "[TPMS v2] Starting optimized dashboard on port $PORT..."
echo "[TPMS v2] URL: http://$(hostname -I | awk '{print $1}'):$PORT"

if [[ "${1:-}" == "--bg" ]]; then
    nohup "$VENV/streamlit" run "$SCRIPT_DIR/tpms_dashboard_v2.py"         --server.address 0.0.0.0         --server.port "$PORT"         --server.headless true         --server.fileWatcherType none         --browser.gatherUsageStats false         > "$SCRIPT_DIR/logs/streamlit_v2.log" 2>&1 &
    echo "[TPMS v2] Background PID: $!"
else
    exec "$VENV/streamlit" run "$SCRIPT_DIR/tpms_dashboard_v2.py"         --server.address 0.0.0.0         --server.port "$PORT"         --server.headless true         --server.fileWatcherType none         --browser.gatherUsageStats false
fi
