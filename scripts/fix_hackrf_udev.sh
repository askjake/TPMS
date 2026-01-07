#!/usr/bin/env bash
set -euo pipefail

VID="1d50"
PID="6089"   # HackRF One

RULES_FILE="/etc/udev/rules.d/53-hackrf.rules"

echo "Creating udev rules for HackRF (${VID}:${PID}) -> ${RULES_FILE}"
echo "You will be prompted for sudo."

sudo tee "${RULES_FILE}" >/dev/null <<'EOF'
# HackRF One
SUBSYSTEM=="usb", ATTR{idVendor}=="1d50", ATTR{idProduct}=="6089", MODE:="0666", GROUP:="plugdev"
EOF

echo "Reloading udev rules..."
sudo udevadm control --reload-rules
sudo udevadm trigger

echo "Ensuring user is in plugdev group..."
if getent group plugdev >/dev/null; then
  sudo usermod -aG plugdev "$USER" || true
  echo "Added $USER to plugdev. You must log out/in (or reboot) for group change to apply."
else
  echo "plugdev group not found on this system. Using 'users' group instead."
  sudo sed -i 's/GROUP:="plugdev"/GROUP:="users"/' "${RULES_FILE}"
  sudo udevadm control --reload-rules
  sudo udevadm trigger
fi

echo
echo "Now unplug/replug the HackRF and run:"
echo "  hackrf_info"
echo "If it only works with sudo, permissions still aren't right."
