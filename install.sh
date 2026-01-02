#!/bin/bash

# TPMS Tracker - Modern Python Installation Script
# Supports Python 3.10, 3.11, 3.12

set -e

echo ""
echo "========================================"
echo "TPMS Tracker - Installation Script"
echo "Modern Python (3.10-3.12)"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# [1/8] Check Python version
echo "[1/8] Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
    PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')
    
    print_status "Python found: $PYTHON_VERSION"
    
    if [ "$PYTHON_MAJOR" -lt 3 ] || [ "$PYTHON_MINOR" -lt 10 ]; then
        print_error "Python 3.10 or higher required, but found Python $PYTHON_MAJOR.$PYTHON_MINOR"
        echo ""
        echo "Please install Python 3.10, 3.11, or 3.12:"
        echo "  Ubuntu/Debian: sudo apt install python3.11 python3.11-venv python3.11-dev"
        echo "  Or use pyenv: https://github.com/pyenv/pyenv"
        exit 1
    fi
    
    if [ "$PYTHON_MINOR" -gt 12 ]; then
        print_warning "Python 3.$PYTHON_MINOR detected. Some packages may not be compatible yet."
        print_warning "Python 3.10-3.12 recommended."
    fi
else
    print_error "Python 3 not found!"
    exit 1
fi

# [2/8] Check pip
echo ""
echo "[2/8] Checking pip..."
if python3 -m pip --version &> /dev/null; then
    PIP_VERSION=$(python3 -m pip --version)
    print_status "pip found: $PIP_VERSION"
else
    print_error "pip not found! Installing..."
    python3 -m ensurepip --upgrade
fi

# [3/8] Setup virtual environment
echo ""
echo "[3/8] Setting up virtual environment..."
if [ -d "venv" ]; then
    print_warning "Virtual environment already exists"
    read -p "Remove and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        print_status "Removed old virtual environment"
        python3 -m venv venv
        print_status "Created new virtual environment"
    else
        print_status "Using existing virtual environment"
    fi
else
    python3 -m venv venv
    print_status "Created virtual environment"
fi

# [4/8] Activate virtual environment
echo ""
echo "[4/8] Activating virtual environment..."
source venv/bin/activate
print_status "Virtual environment activated"

# [5/8] Upgrade pip
echo ""
echo "[5/8] Upgrading pip and tools..."
pip install --upgrade pip setuptools wheel
print_status "pip upgraded"

# [6/8] Install Python dependencies
echo ""
echo "[6/8] Installing Python dependencies..."
pip install -r requirements.txt
print_status "Python packages installed"

# [7/8] Install HackRF support
echo ""
echo "[7/8] Installing HackRF support..."

# Check if libhackrf is installed
if ldconfig -p | grep -q libhackrf; then
    print_status "libhackrf found"
    
    # Get the library path
    LIBHACKRF_PATH=$(ldconfig -p | grep libhackrf | awk '{print $NF}' | head -1)
    print_status "Library path: $LIBHACKRF_PATH"
else
    print_error "libhackrf not found!"
    echo ""
    echo "Installing libhackrf..."
    sudo apt update
    sudo apt install -y libhackrf-dev libusb-1.0-0-dev hackrf
    print_status "libhackrf installed"
fi

# We'll use ctypes to call libhackrf directly - no Python wrapper needed
print_status "HackRF support configured (using ctypes)"
# [8/8] Setup permissions and directories
echo ""
echo "[8/8] Final setup..."

# Create necessary directories
mkdir -p logs
mkdir -p data
mkdir -p models
print_status "Created directories"

# Setup HackRF permissions (Linux only)
if [ "$(uname)" == "Linux" ]; then
    echo ""
    echo "Setting up HackRF permissions..."
    
    # Check if user is in plugdev group
    if ! groups | grep -q plugdev; then
        print_warning "User not in plugdev group"
        echo "Adding user to plugdev group..."
        sudo usermod -a -G plugdev $USER
        print_warning "You need to log out and back in for group changes to take effect"
    else
        print_status "User already in plugdev group"
    fi
    
    # Create udev rule if it doesn't exist
    if [ ! -f /etc/udev/rules.d/53-hackrf.rules ]; then
        echo "Creating udev rule for HackRF..."
        echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="1d50", ATTR{idProduct}=="6089", MODE="0666", GROUP="plugdev"' | sudo tee /etc/udev/rules.d/53-hackrf.rules > /dev/null
        sudo udevadm control --reload-rules
        sudo udevadm trigger
        print_status "udev rule created"
    else
        print_status "udev rule already exists"
    fi
fi

# Test HackRF connection
echo ""
echo "Testing HackRF connection..."
if command -v hackrf_info &> /dev/null; then
    if timeout 2 hackrf_info &> /dev/null; then
        print_status "HackRF device detected!"
    else
        print_warning "HackRF device not detected (will use simulation mode)"
    fi
else
    print_warning "hackrf_info not found (install hackrf package for hardware support)"
fi

# Summary
echo ""
echo "========================================"
echo -e "${GREEN}Installation Complete!${NC}"
echo "========================================"
echo ""
echo "Python version: $PYTHON_VERSION"
echo "Virtual environment: $(pwd)/venv"
echo ""
echo "To start the application:"
echo "  ./start.sh"
echo ""
if ! groups | grep -q plugdev; then
    echo -e "${YELLOW}⚠️  IMPORTANT: Log out and back in for HackRF permissions${NC}"
    echo ""
fi
