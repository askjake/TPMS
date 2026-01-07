#!/bin/bash

echo ""
echo "========================================"
echo "🚀 TPMS Scanner"
echo "========================================"
echo ""

# Get absolute path to script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate
echo "✅ Using: $(python --version)"
echo ""

# Get full path to streamlit
STREAMLIT_PATH=$(which streamlit)
echo "📍 Streamlit path: $STREAMLIT_PATH"
echo ""

# Check for devices (with proper permissions)
echo "🔍 Checking for SDR devices..."

# Check HackRF
if sg plugdev -c "hackrf_info" &>/dev/null; then
    echo "✅ HackRF device detected"
else
    echo "⚠️  HackRF device not detected"
fi

# Check RTL-SDR
if sg plugdev -c "rtl_test -t" &>/dev/null; then
    echo "✅ RTL-SDR device detected"
else
    echo "⚠️  RTL-SDR device not detected"
fi

echo ""
echo "========================================"
echo "🌐 Starting web interface"
echo "========================================"
echo ""
echo "URL: http://localhost:8502"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Run Streamlit with plugdev group, preserving the venv python path
exec sg plugdev -c "cd '$SCRIPT_DIR' && source venv/bin/activate && streamlit run app.py --server.address 0.0.0.0"

