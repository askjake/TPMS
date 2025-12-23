#!/bin/bash
# TPMS Tracker - Ubuntu Startup Script
# =====================================

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
    echo "Please run: ./install.sh"
    exit 1
fi

# Check if activate script exists
if [ ! -f "venv/bin/activate" ]; then
    echo -e "${RED}[ERROR] Virtual environment is broken!${NC}"
    echo "Please run: rm -rf venv && ./install.sh"
    exit 1
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate

# Use Linux-specific startup script if it exists, otherwise fall back
if [ -f "start_tracker_linux.py" ]; then
    python3 start_tracker_linux.py
elif [ -f "start_tracker.py" ]; then
    echo -e "${YELLOW}[WARNING] Using original start_tracker.py (may have issues on Linux)${NC}"
    python3 start_tracker.py
else
    echo -e "${RED}[ERROR] No startup script found${NC}"
    exit 1
fi
