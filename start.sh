#!/bin/bash

# TPMS Scanner Startup Script
# Activates Python 3.10 venv and starts the application

set -e

echo "🚀 Starting TPMS Scanner"
echo "======================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run ./setup.sh first"
    exit 1
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Verify Python version
PYTHON_VERSION=$(python --version)
echo "✅ Using: $PYTHON_VERSION"

# Check if Python is 3.10+
PYTHON_MAJOR=$(python -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$(python -c 'import sys; print(sys.version_info.minor)')

if [ "$PYTHON_MAJOR" -lt 3 ] || [ "$PYTHON_MINOR" -lt 10 ]; then
    echo "❌ Python 3.10+ required, but found Python $PYTHON_MAJOR.$PYTHON_MINOR"
    echo "Please run ./setup.sh to install Python 3.10"
    exit 1
fi

# Check for HackRF device
echo "🔍 Checking for HackRF device..."
if command -v hackrf_info &> /dev/null; then
    if hackrf_info &> /dev/null; then
        echo "✅ HackRF device detected"
    else
        echo "⚠️  HackRF device not found - will run in simulation mode"
    fi
else
    echo "⚠️  hackrf_info not found - will run in simulation mode"
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Set environment variables
export PYTHONUNBUFFERED=1
export STREAMLIT_SERVER_PORT=8502
export STREAMLIT_SERVER_ADDRESS=localhost

# Start the Streamlit application
echo "🌐 Starting web interface on http://localhost:8502"
echo ""
streamlit run app.py
