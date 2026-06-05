#!/usr/bin/env bash
# install_hackrf_autosync.sh
# ==========================
# Installs the udev rule and systemd service that make TPMS sync happen
# automatically whenever the HackRF Portapack SD card is plugged in.
#
# Run once with:  sudo bash install_hackrf_autosync.sh
# Uninstall  :    sudo bash install_hackrf_autosync.sh --uninstall

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYNC_SCRIPT="${SCRIPT_DIR}/hackrf_sd_sync.py"
UDEV_RULE_SRC="${SCRIPT_DIR}/80-hackrf-portapack-sd.rules"
SERVICE_SRC="${SCRIPT_DIR}/hackrf-tpms-sync@.service"

UDEV_DEST="/etc/udev/rules.d/80-hackrf-portapack-sd.rules"
SERVICE_DEST="/etc/systemd/system/hackrf-tpms-sync@.service"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC}  $*"; }
warn() { echo -e "${YELLOW}[!!]${NC}  $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

# ──────────────────────────────────────────
uninstall() {
    echo "Uninstalling HackRF TPMS auto-sync..."
    systemctl disable hackrf-tpms-sync@.service 2>/dev/null || true
    rm -f "${UDEV_DEST}" "${SERVICE_DEST}"
    udevadm control --reload-rules
    systemctl daemon-reload
    ok "Uninstalled."
    exit 0
}

[[ "${1:-}" == "--uninstall" ]] && uninstall

# ──────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════"
echo "  HackRF Portapack TPMS Auto-Sync Installer"
echo "══════════════════════════════════════════════════"

# Preflight checks
[[ $(id -u) -eq 0 ]] || fail "Must be run as root: sudo bash $0"
[[ -f "${SYNC_SCRIPT}" ]] || fail "hackrf_sd_sync.py not found at: ${SYNC_SCRIPT}"
[[ -f "${UDEV_RULE_SRC}" ]] || fail "udev rule not found at: ${UDEV_RULE_SRC}"
[[ -f "${SERVICE_SRC}" ]]   || fail "service file not found at: ${SERVICE_SRC}"

PYTHON=$(which python3)
[[ -x "${PYTHON}" ]] || fail "python3 not found in PATH"

ok "Preflight checks passed"
echo "  Sync script : ${SYNC_SCRIPT}"
echo "  Python      : ${PYTHON}"
echo ""

# ── Install udev rule ──────────────────────────────────────
cp -v "${UDEV_RULE_SRC}" "${UDEV_DEST}"
chmod 644 "${UDEV_DEST}"
ok "udev rule installed: ${UDEV_DEST}"

# ── Install systemd service ────────────────────────────────
# Patch the service to use the actual python3 path
sed "s|/usr/bin/python3|${PYTHON}|g" "${SERVICE_SRC}" > "${SERVICE_DEST}"
chmod 644 "${SERVICE_DEST}"
ok "systemd service installed: ${SERVICE_DEST}"

# ── Reload both ────────────────────────────────────────────
udevadm control --reload-rules
ok "udev rules reloaded"

systemctl daemon-reload
ok "systemd daemon reloaded"

# ── Summary ────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════"
echo "  Installation complete!"
echo "══════════════════════════════════════════════════"
echo ""
echo "  WHAT HAPPENS NEXT:"
echo "  • Plug in the HackRF Portapack (SD card side)"
echo "  • udev detects: idVendor=0781 idProduct=a7a8"
echo "  • systemd starts: hackrf-tpms-sync@<dev>.service"
echo "  • Sync runs:  hackrf_sd_sync.py"
echo "    - Decodes 36-byte SensorRecords"
echo "    - Merges into ~/TPMS/sensors.db + tpms_local.sqlite3"
echo "    - Exports CSV/JSON to ~/TPMS/exports/"
echo "    - Cleans sensors.db off the SD card"
echo ""
echo "  MONITOR:   journalctl -fu 'hackrf-tpms-sync@*'"
echo "  LOGS:      ~/TPMS/logs/hackrf_sd_sync.log"
echo "  MANUAL:    python3 ~/TPMS/hackrf_sd_sync.py"
echo "  DRY RUN:   python3 ~/TPMS/hackrf_sd_sync.py --dry-run"
echo "  UNINSTALL: sudo bash $0 --uninstall"
echo ""
