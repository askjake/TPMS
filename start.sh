#!/bin/bash
# TPMS Tracker - Linux/WSL Startup Script
# ========================================

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo "========================================"
echo "TPMS Tracker - Starting..."
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}[ERROR] Virtual environment not found!${NC}"
    echo "Please run ./install.sh first"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check for HackRF
echo -e "${BLUE}[1/3] Checking for HackRF One...${NC}"
if command -v hackrf_info &> /dev/null; then
    if hackrf_info &> /dev/null; then
        echo -e "${GREEN}[OK] HackRF One detected${NC}"
        hackrf_info | grep -E "(Serial|Board ID|Firmware)"
    else
        echo -e "${YELLOW}[WARNING] HackRF not responding${NC}"
        echo ""
        echo "Please ensure:"
        echo "1. HackRF One is plugged in"
        echo "2. USB permissions are correct"
        echo "3. You're in the 'plugdev' group"
        echo ""
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
else
    echo -e "${RED}[ERROR] hackrf_info not found${NC}"
    echo "Please install HackRF tools"
    exit 1
fi

# Check dependencies
echo ""
echo -e "${BLUE}[2/3] Checking dependencies...${NC}"
if python3 -c "import streamlit" 2>/dev/null; then
    echo -e "${GREEN}[OK] Dependencies found${NC}"
else
    echo -e "${RED}[ERROR] Dependencies not installed${NC}"
    echo "Please run ./install.sh first"
    exit 1
fi

# Start the application
echo ""
echo -e "${BLUE}[3/3] Starting TPMS Tracker...${NC}"
echo "========================================"
echo ""
echo -e "${GREEN}Opening browser at http://localhost:8501${NC}"
echo "Press Ctrl+C to stop the application"
echo ""

python3 start_tracker.py
