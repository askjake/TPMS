#!/usr/bin/env bash
set -euo pipefail

# TPMS Tracker - Linux/macOS quick start
VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements.txt

streamlit run app.py --server.address 0.0.0.0 --server.port 8507
