#!/usr/bin/env bash
# start_dashboard.sh — Start TPMS dashboard + sync watcher
# Usage:  ./start_dashboard.sh          (foreground, Ctrl-C to stop)
#         ./start_dashboard.sh --bg     (background, writes PID files)
#         ./start_dashboard.sh --stop   (kills background processes)
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/venv/bin"
LOG_DIR="$SCRIPT_DIR/logs"
PID_DIR="$SCRIPT_DIR/.pids"
PORT="${TPMS_PORT:-8569}"

mkdir -p "$LOG_DIR" "$PID_DIR"

# ── Sanity checks ────────────────────────────────────────────────────────────
if [[ ! -x "$VENV/streamlit" ]]; then
    echo "[ERROR] $VENV/streamlit not found. Run: python3 -m venv venv && venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi
if [[ ! -f "$SCRIPT_DIR/tpms_dashboard.py" ]]; then
    echo "[ERROR] tpms_dashboard.py not found in $SCRIPT_DIR" >&2
    exit 1
fi

# ── Stop mode ────────────────────────────────────────────────────────────────
stop_services() {
    local stopped=0
    for pf in "$PID_DIR"/*.pid; do
        [[ -f "$pf" ]] || continue
        local pid
        pid=$(<"$pf")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null && echo "[TPMS] Stopped PID $pid ($(basename "$pf" .pid))"
            stopped=$((stopped + 1))
        fi
        rm -f "$pf"
    done
    # Also kill by port as fallback
    local port_pid
    port_pid=$(lsof -t -i:"$PORT" 2>/dev/null || true)
    if [[ -n "$port_pid" ]]; then
        kill "$port_pid" 2>/dev/null && echo "[TPMS] Stopped port $PORT occupant (PID $port_pid)"
        stopped=$((stopped + 1))
    fi
    [[ $stopped -eq 0 ]] && echo "[TPMS] Nothing running."
    return 0
}

if [[ "${1:-}" == "--stop" ]]; then
    stop_services
    exit 0
fi

# ── Kill previous instances on our port ───────────────────────────────────────
stop_services >/dev/null 2>&1 || true
sleep 0.5

# ── Start sync watcher ────────────────────────────────────────────────────────
echo "[TPMS] Starting sync watcher..."
nohup "$VENV/python3" "$SCRIPT_DIR/tpms_dashboard_sync.py"     > "$LOG_DIR/sync_watcher.log" 2>&1 &
SYNC_PID=$!
echo "$SYNC_PID" > "$PID_DIR/sync_watcher.pid"
echo "[TPMS] Sync watcher PID=$SYNC_PID (log: $LOG_DIR/sync_watcher.log)"

# ── Start Streamlit ───────────────────────────────────────────────────────────
STREAMLIT_CMD=(
    "$VENV/streamlit" run "$SCRIPT_DIR/tpms_dashboard.py"
    --server.address 0.0.0.0
    --server.port "$PORT"
    --server.headless true
    --server.fileWatcherType none
    --browser.gatherUsageStats false
)

if [[ "${1:-}" == "--bg" ]]; then
    echo "[TPMS] Starting Streamlit (background) on 0.0.0.0:$PORT..."
    nohup "${STREAMLIT_CMD[@]}" > "$LOG_DIR/streamlit.log" 2>&1 &
    ST_PID=$!
    echo "$ST_PID" > "$PID_DIR/streamlit.pid"
    echo "[TPMS] Streamlit PID=$ST_PID (log: $LOG_DIR/streamlit.log)"
    echo "[TPMS] Dashboard: http://$(hostname -I | awk '{print $1}'):$PORT"
    echo "[TPMS] Stop with: $0 --stop"
else
    echo "[TPMS] Starting Streamlit (foreground) on 0.0.0.0:$PORT..."
    echo "[TPMS] Press Ctrl-C to stop."
    echo "[TPMS] Dashboard: http://$(hostname -I | awk '{print $1}'):$PORT"
    # exec replaces shell — Ctrl-C will kill Streamlit and also the sync watcher
    trap "kill $SYNC_PID 2>/dev/null; exit 0" INT TERM
    exec "${STREAMLIT_CMD[@]}"
fi
