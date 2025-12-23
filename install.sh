#!/bin/bash
# TPMS Tracker - Ubuntu Installation Script
# ==========================================

set -e  # Exit on error

echo ""
echo "========================================"
echo "TPMS Tracker - Installation Script"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Python is installed
echo "[1/7] Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}[ERROR] Python 3 is not installed${NC}"
    echo "Install with: sudo apt install python3 python3-pip python3-venv"
    exit 1
fi
echo -e "${GREEN}Python found:${NC} $(python3 --version)"

# Check if pip is installed
echo ""
echo "[2/7] Checking pip..."
if ! command -v pip3 &> /dev/null; then
    echo -e "${YELLOW}[WARNING] pip3 not found, installing...${NC}"
    sudo apt install -y python3-pip
fi
echo -e "${GREEN}pip found:${NC} $(pip3 --version)"

# Remove old venv if it exists and is broken
echo ""
echo "[3/7] Setting up virtual environment..."
if [ -d "venv" ]; then
    if [ ! -f "venv/bin/activate" ]; then
        echo -e "${YELLOW}Removing broken virtual environment...${NC}"
        rm -rf venv
    else
        echo "Virtual environment exists and looks good"
    fi
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating new virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}Virtual environment created successfully${NC}"
fi

# Activate virtual environment
echo ""
echo "[4/7] Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo ""
echo "[5/7] Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo -e "${GREEN}Dependencies installed successfully${NC}"
else
    echo -e "${RED}[ERROR] requirements.txt not found${NC}"
    exit 1
fi

# Install HackRF tools
echo ""
echo "[6/7] Installing HackRF tools..."
if ! command -v hackrf_info &> /dev/null; then
    echo "Installing HackRF tools from apt..."
    sudo apt update
    sudo apt install -y hackrf libhackrf-dev libhackrf0
    echo -e "${GREEN}HackRF tools installed${NC}"
else
    echo -e "${GREEN}HackRF tools already installed${NC}"
fi

# Test HackRF installation
echo ""
echo "Testing HackRF installation..."
if command -v hackrf_info &> /dev/null; then
    echo -e "${GREEN}hackrf_info found in PATH${NC}"
    hackrf_info --version 2>/dev/null || echo "Version info not available"
else
    echo -e "${RED}[ERROR] hackrf_info not found in PATH${NC}"
    exit 1
fi

# Create/update udev rules for HackRF
echo ""
echo "[7/7] Setting up udev rules for HackRF..."
UDEV_RULE='ATTR{idVendor}=="1d50", ATTR{idProduct}=="6089", SYMLINK+="hackrf-%k", MODE="0666", GROUP="plugdev"'

if [ ! -f "/etc/udev/rules.d/53-hackrf.rules" ]; then
    echo "Creating udev rules..."
    echo "$UDEV_RULE" | sudo tee /etc/udev/rules.d/53-hackrf.rules > /dev/null
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    echo -e "${GREEN}udev rules created${NC}"
else
    echo "udev rules already exist"
fi

# Add user to plugdev group
echo ""
echo "Checking group membership..."
if groups $USER | grep -q '\bplugdev\b'; then
    echo -e "${GREEN}User already in plugdev group${NC}"
else
    echo "Adding user to plugdev group..."
    sudo usermod -a -G plugdev $USER
    echo -e "${YELLOW}[IMPORTANT] You must log out and log back in for group changes to take effect${NC}"
fi

echo ""
echo "========================================"
echo -e "${GREEN}Installation Complete!${NC}"
echo "========================================"
echo ""
echo "NEXT STEPS:"
echo "1. If you just joined the plugdev group, log out and log back in"
echo "2. Unplug and replug your HackRF One"
echo "3. Test with: hackrf_info"
echo "4. Run: ./start.sh"
echo ""
