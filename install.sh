#!/bin/bash
# TPMS Tracker - Linux/WSL Installation Script
# ============================================

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
NC='\033[0m' # No Color

# Check if Python is installed
echo "[1/6] Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}[ERROR] Python 3 is not installed${NC}"
    echo "Install with: sudo apt install python3 python3-pip python3-venv"
    exit 1
fi
echo -e "${GREEN}Python found:${NC} $(python3 --version)"

# Check if git is installed
echo ""
echo "[2/6] Checking Git..."
if ! command -v git &> /dev/null; then
    echo -e "${YELLOW}[WARNING] Git not found - skipping repository check${NC}"
else
    echo -e "${GREEN}Git found:${NC} $(git --version)"
fi

# Create virtual environment
echo ""
echo "[3/6] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists, skipping..."
else
    python3 -m venv venv
    echo -e "${GREEN}Virtual environment created successfully${NC}"
fi

# Activate virtual environment
echo ""
echo "[4/6] Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
echo ""
echo "[5/6] Installing Python dependencies..."
pip install -r requirements.txt
echo -e "${GREEN}Dependencies installed successfully${NC}"

# Install HackRF tools (Linux)
echo ""
echo "[6/6] Installing HackRF tools..."
if command -v apt &> /dev/null; then
    echo "Detected Debian/Ubuntu system"
    sudo apt update
    sudo apt install -y hackrf libhackrf-dev
elif command -v yum &> /dev/null; then
    echo "Detected RedHat/CentOS system"
    sudo yum install -y hackrf libhackrf-devel
else
    echo -e "${YELLOW}[WARNING] Unknown package manager${NC}"
    echo "Please install hackrf manually"
fi

# Test HackRF installation
echo ""
echo "Testing HackRF installation..."
if command -v hackrf_info &> /dev/null; then
    echo -e "${GREEN}HackRF tools installed successfully${NC}"
else
    echo -e "${YELLOW}[WARNING] hackrf_info not found in PATH${NC}"
fi

# Create udev rules for HackRF
echo ""
echo "Setting up udev rules for HackRF..."
if [ ! -f "/etc/udev/rules.d/53-hackrf.rules" ]; then
    echo 'ATTR{idVendor}=="1d50", ATTR{idProduct}=="6089", SYMLINK+="hackrf-%k", MODE="0666", GROUP="plugdev"' | sudo tee /etc/udev/rules.d/53-hackrf.rules
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    echo -e "${GREEN}udev rules created${NC}"
else
    echo "udev rules already exist"
fi

# Add user to plugdev group
echo ""
echo "Adding user to plugdev group..."
sudo usermod -a -G plugdev $USER

echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "NEXT STEPS:"
echo "1. Log out and log back in (for group changes)"
echo "2. Plug in your HackRF One"
echo "3. Test with: hackrf_info"
echo "4. Run: ./start.sh"
echo ""
