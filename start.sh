#!/bin/bash

echo ""
echo "========================================"
echo "🚀 TPMS Scanner"
echo "========================================"
echo ""

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate
echo "✅ Using: $(python --version)"
echo ""

# Check for devices (with proper permissions)
echo "🔍 Checking for SDR devices..."

# Check HackRF
if sg plugdev -c "hackrf_info" &>/dev/null; then
    echo "✅ HackRF device detected"
    HACKRF_FOUND=1
else
    echo "⚠️  HackRF device not detected"
    HACKRF_FOUND=0
fi

# Check RTL-SDR
if sg plugdev -c "rtl_test -t" &>/dev/null; then
    echo "✅ RTL-SDR device detected"
    RTLSDR_FOUND=1
else
    echo "⚠️  RTL-SDR device not detected"
    RTLSDR_FOUND=0
fi

echo ""

# Warn if no devices found
if [ $HACKRF_FOUND -eq 0 ] && [ $RTLSDR_FOUND -eq 0 ]; then
    echo "⚠️  WARNING: No SDR devices detected!"
    echo "   Make sure devices are plugged in and you have permissions."
    echo ""
fi

echo "========================================"
echo "🌐 Starting web interface"
echo "========================================"
echo ""
echo "URL: http://localhost:8502"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# IMPORTANT: Run Streamlit with plugdev group permissions
# This ensures Python can access USB devices
exec sg plugdev -c "streamlit run app.py --server.port 8502 --server.address 0.0.0.0"

