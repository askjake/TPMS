#!/bin/bash

# TPMS Scanner Startup Script
# Modern Python (3.10-3.12)

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo "========================================"
echo "🚀 TPMS Scanner"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}❌ Virtual environment not found!${NC}"
    echo "Please run ./install.sh first"
    exit 1
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Verify Python version
PYTHON_VERSION=$(python --version)
PYTHON_MAJOR=$(python -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$(python -c 'import sys; print(sys.version_info.minor)')

echo -e "${GREEN}✅ Using: $PYTHON_VERSION${NC}"

if [ "$PYTHON_MAJOR" -lt 3 ] || [ "$PYTHON_MINOR" -lt 10 ]; then
    echo -e "${RED}❌ Python 3.10+ required${NC}"
    echo "Please run ./install.sh to set up the correct Python version"
    exit 1
fi

# Check for HackRF device
echo ""
echo "🔍 Checking for HackRF device..."
HACKRF_FOUND=false

if command -v hackrf_info &> /dev/null; then
    if timeout 2 hackrf_info &> /dev/null 2>&1; then
        echo -e "${GREEN}✅ HackRF device detected${NC}"
        HACKRF_FOUND=true
    fi
fi

if [ "$HACKRF_FOUND" = false ]; then
    echo -e "${YELLOW}⚠️  HackRF device not found${NC}"
    echo "   Running in SIMULATION MODE"
    echo ""
    echo "   If you have a HackRF connected:"
    echo "   - Check USB connection"
    echo "   - Verify permissions: groups | grep plugdev"
    echo "   - Test with: hackrf_info"
    echo ""
fi

# Create logs directory
mkdir -p logs

# Set environment variables
export PYTHONUNBUFFERED=1
export STREAMLIT_SERVER_PORT=8502
export STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Start the application
echo ""
echo "========================================"
echo "🌐 Starting web interface"
echo "========================================"
echo ""
echo "URL: http://localhost:8502"
echo ""
echo "Press Ctrl+C to stop"
echo ""

streamlit run app.py
