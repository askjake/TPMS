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

# Check for HackRF tools
echo ""
echo -e "${BLUE}[1/4] Checking for HackRF tools...${NC}"
if ! command -v hackrf_info &> /dev/null; then
    echo -e "${RED}[ERROR] hackrf_info not found${NC}"
    echo "Please run: ./install.sh"
    exit 1
fi
echo -e "${GREEN}[OK] HackRF tools found${NC}"

# Check for HackRF device
echo ""
echo -e "${BLUE}[2/4] Checking for HackRF One device...${NC}"
if hackrf_info &> /dev/null; then
    echo -e "${GREEN}[OK] HackRF One detected${NC}"
    echo ""
    hackrf_info 2>&1 | grep -E "(Serial|Board ID|Firmware|Part ID)" || true
else
    echo -e "${YELLOW}[WARNING] HackRF not responding${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "1. Check if device is detected: lsusb | grep -i hackrf"
    echo "2. Check permissions: ls -la /dev/bus/usb/*/*"
    echo "3. Check group membership: groups"
    echo "4. Try unplugging and replugging the device"
    echo ""
    
    # Check if device is visible via lsusb
    if lsusb | grep -qi "1d50:6089"; then
        echo -e "${YELLOW}Device is visible via lsusb but not accessible${NC}"
        echo "This is likely a permissions issue."
        echo ""
    fi
    
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Python dependencies
echo ""
echo -e "${BLUE}[3/4] Checking Python dependencies...${NC}"
MISSING_DEPS=()

for package in streamlit numpy scipy pandas; do
    if ! python3 -c "import $package" 2>/dev/null; then
        MISSING_DEPS+=($package)
    fi
done

if [ ${#MISSING_DEPS[@]} -eq 0 ]; then
    echo -e "${GREEN}[OK] All dependencies found${NC}"
else
    echo -e "${RED}[ERROR] Missing dependencies: ${MISSING_DEPS[*]}${NC}"
    echo "Please run: ./install.sh"
    exit 1
fi

# Start the application
echo ""
echo -e "${BLUE}[4/4] Starting TPMS Tracker...${NC}"
echo "========================================"
echo ""
echo -e "${GREEN}Application will open at http://localhost:8501${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the application${NC}"
echo ""

# Check if start_tracker.py exists
if [ ! -f "start_tracker.py" ]; then
    echo -e "${RED}[ERROR] start_tracker.py not found${NC}"
    exit 1
fi

# Start the application
python3 start_tracker.py
