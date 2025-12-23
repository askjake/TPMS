1. Create requirements.txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
scikit-learn>=1.3.0
scipy>=1.11.0

2. Create setup.py (Optional but recommended)
from setuptools import setup, find_packages

setup(
    name="tpms-tracker",
    version="1.0.0",
    description="Intelligent TPMS Vehicle Pattern Recognition System",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        'streamlit>=1.28.0',
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'plotly>=5.17.0',
        'scikit-learn>=1.3.0',
        'scipy>=1.11.0',
    ],
    python_requires='>=3.9',
)

3. Create .gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/
.venv

# Database
*.db
*.sqlite
*.sqlite3
tpms_tracker.db

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
desktop.ini

# Streamlit
.streamlit/secrets.toml

# Logs
*.log

# Exports
*.csv
exports/

4. Create README.md
# 🚗 TPMS Tracker - Intelligent Vehicle Pattern Recognition

Track and analyze vehicles through their Tire Pressure Monitoring System (TPMS) signals using HackRF One.

## Features

- 🎯 Real-time TPMS signal detection on 314.9, 315, and 433.92 MHz
- 🤖 Automatic vehicle clustering (groups 4 tire sensors into vehicles)
- 📊 Pattern recognition and predictive analytics
- 🔧 Tire maintenance monitoring
- 📈 Historical data analysis and visualization
- 🔮 Predict when familiar vehicles will appear next

## Prerequisites

### Hardware
- **HackRF One** SDR device
- USB cable
- Windows 10/11 PC

### Software
- Python 3.9 or higher
- HackRF tools for Windows

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/tpms-tracker.git
cd tpms-tracker

2. Install Python Dependencies
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

3. Install HackRF Tools
Option A: Download Pre-built Release
Download from: https://github.com/greatscottgadgets/hackrf/releases
Extract to C:\hackrf
Add C:\hackrf\bin to your PATH
Option B: Use Automated Installer (Recommended)
python install_hackrf.py

4. Install HackRF Driver
Download Zadig: https://zadig.akeo.ie/
Plug in HackRF One
Run Zadig as Administrator
Options → List All Devices
Select “HackRF One”
Select “WinUSB” driver
Click “Replace Driver” or “Install Driver”
Usage
Quick Start
# Activate virtual environment (if not already active)
venv\Scripts\activate

# Run the application
streamlit run app.py

The app will automatically:

Detect your HackRF One
Open in your default browser at http://localhost:8501
Start scanning when you click “Start Scan”
Manual Start
python start_tracker.py

Configuration
Edit config.py to customize:

# Frequencies to scan (MHz)
FREQUENCIES = [314.9, 315.0, 433.92]

# Detection thresholds
SNR_THRESHOLD = 10.0
MIN_SIGNAL_STRENGTH = -80  # dBm

# Vehicle clustering
TPMS_PER_VEHICLE = 4
CLUSTER_TIME_WINDOW = 30  # seconds

Troubleshooting
HackRF Not Detected
Check Connection:

hackrf_info

Expected Output:

Found HackRF
Serial number: 0000000000000000...
Board ID Number: 2 (HackRF One)

If not found:

Verify USB connection
Check Device Manager for “HackRF One” under “Universal Serial Bus devices”
Reinstall WinUSB driver using Zadig
Try different USB port (USB 2.0 recommended)
Permission Errors
Run Command Prompt or PowerShell as Administrator:

# Right-click → Run as Administrator
cd path\to\tpms-tracker
venv\Scripts\activate
streamlit run app.py

No Signals Detected
Check Frequency: Ensure you’re scanning the correct frequency for your region

North America: 315 MHz
Europe: 433.92 MHz
Some vehicles: 314.9 MHz
Check Gain: Try increasing gain in the sidebar (20-40 dB recommended)

Check Location: Drive near traffic or park near a busy road

Verify HackRF: Test with known signal source

Database Issues
Reset database:

del tpms_tracker.db
python -c "from database import TPMSDatabase; TPMSDatabase('tpms_tracker.db')"

Project Structure
tpms-tracker/
├── app.py                      # Main Streamlit application
├── config.py                   # Configuration settings
├── database.py                 # Database management
├── hackrf_interface.py         # HackRF SDR interface
├── tpms_decoder.py            # TPMS signal decoder
├── ml_engine.py               # Machine learning engine
├── requirements.txt           # Python dependencies
├── install_hackrf.py          # HackRF tools installer
├── start_tracker.py           # Startup script
└── README.md                  # This file

Usage Tips
Let it run for at least a week to build good patterns
Drive your regular routes for best results
Name familiar vehicles for easier tracking
Check maintenance tab weekly for tire health
Export data regularly for backup
Privacy & Legal
⚠️ Important: This tool is for educational and research purposes only.

TPMS signals are unencrypted and broadcast publicly
Be aware of local laws regarding radio monitoring
Do not use for stalking or illegal surveillance
Respect privacy of others
Contributing
Contributions welcome! Please:

Fork the repository
Create a feature branch
Submit a pull request
License
MIT License - See LICENSE file for details

Support
For issues or questions:

Open an issue on GitHub
Check the troubleshooting section above
Acknowledgments
HackRF One by Great Scott Gadgets
TPMS protocol documentation from various sources
Streamlit for the amazing framework

## 5. Create `install_hackrf.py` (Automated Installer)

```python
"""
Automated HackRF Tools Installer for Windows
Downloads and installs HackRF tools and sets up PATH
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
import winreg
from pathlib import Path

HACKRF_VERSION = "2024.02.1"
HACKRF_URL = f"https://github.com/greatscottgadgets/hackrf/releases/download/v{HACKRF_VERSION}/hackrf-{HACKRF_VERSION}-win-x64.zip"
INSTALL_DIR = Path("C:/hackrf")

def is_admin():
    """Check if script is running with admin privileges"""
    try:
        return os.getuid() == 0
    except AttributeError:
        import ctypes
        return ctypes.windll.shell32.IsUserAnAdmin() != 0

def download_hackrf():
    """Download HackRF tools"""
    print(f"📥 Downloading HackRF {HACKRF_VERSION}...")
    
    zip_path = Path("hackrf.zip")
    
    try:
        urllib.request.urlretrieve(HACKRF_URL, zip_path)
        print("✅ Download complete!")
        return zip_path
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None

def extract_hackrf(zip_path):
    """Extract HackRF tools"""
    print(f"📦 Extracting to {INSTALL_DIR}...")
    
    try:
        INSTALL_DIR.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(INSTALL_DIR)
        
        print("✅ Extraction complete!")
        
        # Clean up
        zip_path.unlink()
        
        return True
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False

def add_to_path():
    """Add HackRF bin directory to system PATH"""
    bin_dir = str(INSTALL_DIR / "bin")
    
    print(f"🔧 Adding {bin_dir} to PATH...")
    
    try:
        # Get current PATH
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            'Environment',
            0,
            winreg.KEY_ALL_ACCESS
        )
        
        try:
            current_path, _ = winreg.QueryValueEx(key, 'Path')
        except WindowsError:
            current_path = ''
        
        # Add to PATH if not already there
        if bin_dir not in current_path:
            new_path = f"{current_path};{bin_dir}" if current_path else bin_dir
            winreg.SetValueEx(key, 'Path', 0, winreg.REG_EXPAND_SZ, new_path)
            print("✅ PATH updated!")
            print("⚠️  Please restart your terminal for PATH changes to take effect")
        else:
            print("ℹ️  Already in PATH")
        
        winreg.CloseKey(key)
        return True
        
    except Exception as e:
        print(f"❌ Failed to update PATH: {e}")
        print(f"Please manually add {bin_dir} to your PATH")
        return False

def test_installation():
    """Test if HackRF tools are working"""
    print("\n🧪 Testing installation...")
    
    try:
        result = subprocess.run(
            ['hackrf_info', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print("✅ HackRF tools installed successfully!")
            print(f"Version: {result.stdout.strip()}")
            return True
        else:
            print("⚠️  Installation complete but hackrf_info not responding")
            print("You may need to restart your terminal")
            return False
            
    except FileNotFoundError:
        print("⚠️  hackrf_info not found in PATH")
        print("Please restart your terminal and try again")
        return False
    except Exception as e:
        print(f"⚠️  Test failed: {e}")
        return False

def main():
    print("=" * 60)
    print("HackRF Tools Installer for Windows")
    print("=" * 60)
    print()
    
    # Check if already installed
    try:
        result = subprocess.run(
            ['hackrf_info', '--version'],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✅ HackRF tools already installed!")
            print("Run 'hackrf_info' to test your device")
            return
    except:
        pass
    
    # Download
    zip_path = download_hackrf()
    if not zip_path:
        sys.exit(1)
    
    # Extract
    if not extract_hackrf(zip_path):
        sys.exit(1)
    
    # Add to PATH
    add_to_path()
    
    # Test
    test_installation()
    
    print("\n" + "=" * 60)
    print("📋 Next Steps:")
    print("=" * 60)
    print("1. Restart your terminal/command prompt")
    print("2. Plug in your HackRF One")
    print("3. Install WinUSB driver using Zadig:")
    print("   - Download: https://zadig.akeo.ie/")
    print("   - Select 'HackRF One' device")
    print("   - Install WinUSB driver")
    print("4. Test with: hackrf_info")
    print("5. Run the app: streamlit run app.py")
    print()

if __name__ == "__main__":
    main()

6. Create start_tracker.py (Auto-start script)
"""
TPMS Tracker Startup Script
Checks for HackRF and starts the application
"""

import subprocess
import sys
import time
from pathlib import Path

def check_hackrf():
    """Check if HackRF is connected and working"""
    print("🔍 Checking for HackRF One...")
    
    try:
        result = subprocess.run(
            ['hackrf_info'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0 and 'Found HackRF' in result.stdout:
            print("✅ HackRF One detected!")
            print(result.stdout)
            return True
        else:
            print("❌ HackRF not found")
            print(result.stderr)
            return False
            
    except FileNotFoundError:
        print("❌ hackrf_info not found!")
        print("Please install HackRF tools:")
        print("  python install_hackrf.py")
        return False
    except subprocess.TimeoutExpired:
        print("⚠️  HackRF not responding (timeout)")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_dependencies():
    """Check if all Python dependencies are installed"""
    print("\n📦 Checking dependencies...")
    
    required = [
        'streamlit',
        'pandas',
        'numpy',
        'plotly',
        'sklearn',
        'scipy'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies installed")
    return True

def start_app():
    """Start the Streamlit application"""
    print("\n🚀 Starting TPMS Tracker...")
    print("=" * 60)
    
    try:
        subprocess.run([
            'streamlit',
            'run',
            'app.py',
            '--server.headless=true'
        ])
    except KeyboardInterrupt:
        print("\n\n👋 TPMS Tracker stopped")
    except Exception as e:
        print(f"\n❌ Error starting app: {e}")
        sys.exit(1)

def main():
    print("=" * 60)
    print("🚗 TPMS Tracker - Startup Script")
    print("=" * 60)
    print()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check HackRF
    if not check_hackrf():
        print("\n" + "=" * 60)
        print("⚠️  HackRF Setup Required")
        print("=" * 60)
        print("\n1. Connect HackRF One via USB")
        print("2. Install WinUSB driver with Zadig")
        print("3. Run: hackrf_info")
        print("\nTry again after setup is complete")
        sys.exit(1)
    
    # Start app
    start_app()

if __name__ == "__main__":
    main()

7. Create INSTALL.md (Quick Setup Guide)
# Quick Installation Guide

## 1. Clone Repository
```bash
git clone https://github.com/yourusername/tpms-tracker.git
cd tpms-tracker

2. Setup Python Environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

3. Install HackRF Tools
python install_hackrf.py

Then restart your terminal!

4. Install HackRF Driver
Download Zadig: https://zadig.akeo.ie/
Plug in HackRF
Run Zadig as Admin
Select “HackRF One”
Install “WinUSB” driver
5. Test HackRF
hackrf_info

6. Run Application
python start_tracker.py

Done! 🎉 Open http://localhost:8501 in your browser


## 8. GitHub Repository Setup

```bash
# Initialize git (if not already)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: TPMS Tracker v1.0"

# Add remote (replace with your GitHub repo URL)
git remote add origin https://github.com/yourusername/tpms-tracker.git

# Push
git push -u origin main

9. On Your Other Laptop
# Clone
git clone https://github.com/yourusername/tpms-tracker.git
cd tpms-tracker

# Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Install HackRF tools
python install_hackrf.py

# Restart terminal, then:
python start_tracker.py

Complete File Checklist
Make sure you have all these files in your repo:

✅ app.py
✅ config.py
✅ database.py
✅ hackrf_interface.py
✅ tpms_decoder.py
✅ ml_engine.py
✅ requirements.txt
✅ install_hackrf.py
✅ start_tracker.py
✅ README.md
✅ INSTALL.md
✅ .gitignore
✅ setup.py (optional)

Now you’re ready to go! Just clone on your other laptop, plug in the HackRF, and run python start_tracker.py 🚗📡