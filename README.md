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
git clone https://github.com/askjake/TPMS.git
cd tpms-tracker


Install Python Dependencies
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