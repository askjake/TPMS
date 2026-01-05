# TPMS Tracker - Intelligent Vehicle Pattern Recognition System

<div align="center">

![TPMS Tracker](https://img.shields.io/badge/TPMS-Tracker-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Active-success)

**A sophisticated software-defined radio (SDR) system for passive TPMS signal reception, decoding, and vehicle tracking with machine learning pattern recognition.**

[Features](#features) • [Hardware Requirements](#hardware-requirements) • [Installation](#installation) • [Usage](#usage) • [Documentation](#documentation) • [Troubleshooting](#troubleshooting)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
- [Understanding TPMS Signals](#understanding-tpms-signals)
- [ESP32 LF Trigger Setup](#esp32-lf-trigger-setup)
- [Database Schema](#database-schema)
- [Machine Learning Engine](#machine-learning-engine)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Legal & Safety](#legal--safety)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

TPMS Tracker is an advanced vehicle tracking and tire pressure monitoring system that uses software-defined radio (SDR) technology to passively receive, decode, and analyze Tire Pressure Monitoring System (TPMS) signals from nearby vehicles. The system employs machine learning algorithms to cluster sensors into vehicles, track encounter patterns, and predict future sightings.

### What Does It Do?

- **Passive Signal Reception**: Captures TPMS signals at 314.9, 315.0, and 433.92 MHz
- **Multi-Protocol Decoding**: Supports Toyota/Lexus, Schrader, Continental, and other TPMS protocols
- **Vehicle Clustering**: Automatically groups 4 sensors into vehicle profiles using ML
- **Pattern Recognition**: Learns commute patterns and predicts next encounters
- **Tire Health Monitoring**: Tracks pressure and temperature trends over time
- **Active Sensor Triggering**: Optional ESP32-based 125 kHz LF transmitter for sensor activation

### Use Cases

- **Research**: Study TPMS protocol implementations and signal characteristics
- **Vehicle Tracking**: Monitor recurring vehicles in parking lots or traffic patterns
- **Tire Maintenance**: Track tire pressure trends for fleet management
- **Education**: Learn about SDR, signal processing, and RF protocols
- **Security Research**: Analyze TPMS security and privacy implications

---

## ✨ Features

### Core Functionality

- ✅ **Real-Time Signal Processing**
  - Continuous reception on selected frequency
  - FSK/OOK demodulation
  - Multi-protocol packet decoding
  - Signal strength monitoring (-95 dBm sensitivity)

- ✅ **Protocol Support**
  - Toyota/Lexus (proprietary protocol)
  - Schrader EZ-Sensor family
  - Continental
  - Generic FSK/OOK protocols
  - Extensible decoder architecture

- ✅ **Vehicle Database**
  - Automatic sensor-to-vehicle clustering
  - Encounter history tracking
  - GPS location support (optional)
  - Custom vehicle nicknames and notes
  - Export to CSV

- ✅ **Sensor Database**
  - Individual sensor tracking
  - Signal strength history
  - Pressure/temperature trends
  - Orphaned sensor detection
  - Manual vehicle assignment

- ✅ **Machine Learning**
  - DBSCAN clustering for vehicle identification
  - Pattern recognition for commute analysis
  - Next encounter prediction
  - Anomaly detection

- ✅ **Analytics Dashboard**
  - Daily encounter frequency charts
  - Day/hour heatmaps
  - Top vehicles ranking
  - Recent activity feed
  - Tire health monitoring

- ✅ **ESP32 LF Trigger (Optional)**
  - 125 kHz LF signal generation
  - Multi-protocol trigger support
  - WiFi control interface
  - Continuous and single-shot modes
  - Trigger-and-listen workflow

### User Interface

- **Streamlit-based Web UI**: Modern, responsive interface
- **Live Signal Stream**: Real-time detection feed
- **Interactive Charts**: Plotly visualizations
- **Signal Histogram**: Distribution analysis
- **Protocol Monitoring**: Unknown signal detection
- **Timezone Support**: Mountain Time (MT) display with DST handling

---

### Component Overview

| Component | Purpose | Technology |
|-----------|---------|------------|
| **HackRF One** | RF reception | SDR hardware (1 MHz - 6 GHz) |
| **hackrf_interface.py** | Hardware control | Python + libhackrf |
| **tpms_decoder.py** | Signal processing | NumPy, SciPy, DSP algorithms |
| **ml_engine.py** | Vehicle clustering | scikit-learn (DBSCAN) |
| **database.py** | Data persistence | SQLite3 |
| **app.py** | User interface | Streamlit |
| **ESP32** | LF triggering | Arduino firmware + WiFi |

---

## 🔧 Hardware Requirements

### Required Hardware

#### 1. HackRF One SDR
- **Model**: Great Scott Gadgets HackRF One
- **Frequency Range**: 1 MHz - 6 GHz
- **Sample Rate**: Up to 20 MS/s
- **Interface**: USB 2.0
- **Antenna**: 300-500 MHz antenna (included or aftermarket)
- **Cost**: ~$300 USD

**Purchase Links**:
- [Great Scott Gadgets](https://greatscottgadgets.com/hackrf/)
- [Amazon](https://www.amazon.com/s?k=hackrf+one)
- [Adafruit](https://www.adafruit.com/product/3583)

#### 2. Computer
- **OS**: Ubuntu 20.04+ (recommended) or other Linux distro
- **CPU**: Intel i5 or better (for real-time processing)
- **RAM**: 4 GB minimum, 8 GB recommended
- **USB**: USB 2.0 or 3.0 port
- **Storage**: 10 GB free space

**Tested Configurations**:
- HP EliteBook 840 G4 (Ubuntu 20.04)
- Dell XPS 13 (Ubuntu 22.04)
- Raspberry Pi 4 8GB (Ubuntu Server 22.04) - works but slower

### Optional Hardware

#### 3. ESP32 Development Board (for LF Triggering)
- **Model**: ESP32 DevKit with CP2102 USB chip
- **Purpose**: Generate 125 kHz LF signals to activate TPMS sensors
- **Cost**: ~$10 USD

**Specifications**:
- ESP32-WROOM-32 module
- WiFi 802.11 b/g/n
- GPIO pins for antenna connection
- USB programming interface

#### 4. LF Antenna Components (for ESP32)
- 24 AWG magnet wire (50-100 feet)
- 10 cm diameter coil form (PVC pipe or 3D printed)
- 1 µF film capacitor
- 100 nF ceramic trimmer capacitor
- IRF540N or IRLZ44N MOSFET
- TC4420 or MCP1407 gate driver IC
- 10Ω 2W resistor
- Perfboard or breadboard
- 12V power supply (1A minimum)

**Total Cost**: ~$20-30 USD for all components

---

## 💻 Software Requirements

### Operating System
- **Linux**: Ubuntu 20.04+ (recommended), Debian, Fedora, Arch
- **macOS**: 10.14+ (experimental, limited HackRF support)
- **Windows**: WSL2 with Ubuntu (not recommended for real-time performance)

### Python Environment
- **Python**: 3.8 - 3.10 (3.10 recommended)
- **pip**: Latest version
- **venv**: For virtual environment

### System Dependencies

#### Ubuntu/Debian:
```bash
sudo apt update
sudo apt install -y \
    python3 python3-pip python3-venv \
    libhackrf-dev hackrf \
    libusb-1.0-0-dev \
    pkg-config \
    git
Fedora:
sudo dnf install -y \
    python3 python3-pip \
    hackrf hackrf-devel \
    libusb-devel \
    git

macOS (Homebrew):
brew install hackrf python@3.10 libusb

Python Dependencies
See requirements.txt for complete list. Key packages:

streamlit (1.28+): Web UI framework
pandas (1.5+): Data manipulation
numpy (1.24+): Numerical computing
scipy (1.10+): Signal processing
scikit-learn (1.3+): Machine learning
plotly (5.17+): Interactive charts
pytz: Timezone support
📦 Installation
Step 1: Clone Repository
cd ~
git clone git@git.dtc.dish.corp:montjac/TPMS_pattern_collector.git TPMS
cd TPMS

Step 2: Create Virtual Environment
python3 -m venv venv
source venv/bin/activate

Step 3: Install Python Dependencies
pip install --upgrade pip
pip install -r requirements.txt

requirements.txt (if not present, create it):

streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
plotly>=5.17.0
pytz>=2023.3
requests>=2.31.0

Step 4: Verify HackRF Installation
# Check HackRF is detected
hackrf_info

# Expected output:
# Found HackRF
# Serial number: 0000000000000000...
# Firmware Version: 2023.01.1

If hackrf_info fails:

# Check USB connection
lsusb | grep HackRF

# Add udev rules (if needed)
sudo cp /usr/share/hackrf/udev/53-hackrf.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger

# Reboot if necessary
sudo reboot

Step 5: Initialize Database
The database will be created automatically on first run, but you can initialize it manually:

python3 -c "from database import TPMSDatabase; TPMSDatabase('tpms_data.db')"

Step 6: Test Installation
streamlit run app.py

Open browser to http://localhost:8501

You should see the TPMS Tracker interface with:

✅ Control panel in sidebar
✅ Live Detection tab
✅ Vehicle/Sensor Database tabs
✅ Analytics dashboard
⚙️ Configuration
config.py
Main configuration file with all system parameters:

# config.py

class Config:
    # HackRF Settings
    DEFAULT_FREQUENCY = 314.9e6  # 314.9 MHz (North America)
    SAMPLE_RATE = 2_457_600      # 2.4576 MS/s (matches native TPMS app)
    BANDWIDTH = 1_750_000        # 1.75 MHz
    DEFAULT_GAIN = 32            # LNA gain (dB)
    VGA_GAIN = 40                # VGA gain (dB)
    
    # Signal Processing
    SIGNAL_THRESHOLD = -95       # Detection threshold (dBm)
    SIGNAL_HISTORY_SIZE = 10000  # Samples to keep in memory
    HISTOGRAM_BINS = 50          # Histogram resolution
    
    # TPMS Decoder
    FSK_SYMBOL_RATE = 19200      # Schrader baud rate
    OOK_SYMBOL_RATE = 8192       # OOK baud rate
    
    # Database
    DB_PATH = 'tpms_data.db'
    
    # Machine Learning
    CLUSTERING_EPS = 300         # DBSCAN epsilon (seconds)
    CLUSTERING_MIN_SAMPLES = 3   # Minimum signals for cluster
    
    # ESP32 Trigger
    ESP32_IP = '192.168.4.1'
    ESP32_SSID = 'TPMS_Trigger'
    ESP32_PASSWORD = 'tpms12345'
    LF_FREQUENCY = 125000        # 125 kHz

config = Config()

Frequency Configuration
Different regions use different TPMS frequencies:

Region	Frequency	Setting
North America	314.9 - 315.0 MHz	DEFAULT_FREQUENCY = 314.9e6
Europe	433.92 MHz	DEFAULT_FREQUENCY = 433.92e6
Japan	315.0 MHz	DEFAULT_FREQUENCY = 315.0e6
Timezone Configuration
Edit app.py to change timezone:

# Default: Mountain Time
MOUNTAIN_TZ = pytz.timezone('America/Denver')

# Other US timezones:
# EASTERN_TZ = pytz.timezone('America/New_York')
# CENTRAL_TZ = pytz.timezone('America/Chicago')
# PACIFIC_TZ = pytz.timezone('America/Los_Angeles')

📖 Usage Guide
Starting the Application
Activate virtual environment:
cd ~/TPMS
source venv/bin/activate

Launch Streamlit app:
streamlit run app.py

Open browser: Navigate to http://localhost:8501
Basic Workflow
1. Configure Frequency
In sidebar, select frequency (314.9, 315.0, or 433.92 MHz)
Click “Set Frequency”
System locks to this frequency (no hopping)
2. Start Scanning
Click “▶️ Start Scan” in sidebar
Status changes to “🟢 Active”
Signal histogram begins populating
3. Monitor Live Detection
Live Detection tab shows:
Real-time signal strength histogram
Protocol detection statistics
Recent vehicle detections
Live signal stream
4. View Detected Vehicles
Vehicle Database tab shows:
All detected vehicles
TPMS IDs per vehicle
Encounter count and timestamps
Add custom nicknames
5. Analyze Sensors
Sensor Database tab shows:
All unique sensors detected
Signal strength history
Pressure/temperature trends (if available)
Unassigned sensors
6. Review Analytics
Analytics tab shows:
Daily encounter frequency
Day/hour heatmaps
Top vehicles ranking
Recent activity
7. Monitor Tire Health
Maintenance tab shows:
Per-tire pressure trends
Temperature monitoring
Pressure variance alerts
Maintenance recommendations
Advanced Features
Manual Sensor Assignment
If the ML engine doesn’t automatically cluster sensors:

Go to Sensor Database → Unassigned Sensors
Select sensor from dropdown
Select vehicle to assign to
Click “Assign”
Export Data
Export sensor or vehicle data to CSV:

Navigate to desired tab
Click “📥 Export” button
Save CSV file
Frequency Switching During Scan
To change frequency while scanning:

Click “Change Frequency” in sidebar
Select new frequency
Click “Apply”
System retunes without stopping
📡 Understanding TPMS Signals
Signal Characteristics
Transmission Pattern
Frequency: Periodic (every 30-60 seconds while moving)
Duration: 5-20 ms burst
Power: -90 to -70 dBm at 10 meters
Modulation: FSK (most common) or OOK
Packet Structure
Typical TPMS Packet:

┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│ Preamble │ Sync     │ ID (32b) │ Pressure │ Temp     │ CRC      │
│ (8-16b)  │ (8-16b)  │          │ (8b)     │ (8b)     │ (8-16b)  │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘

Field Descriptions:

Preamble: Alternating 1010… pattern for sync
Sync Word: Protocol-specific synchronization
ID: Unique 32-bit sensor identifier
Pressure: Tire pressure (typically in kPa or PSI)
Temperature: Tire temperature (°C)
CRC: Checksum for error detection
Protocol Details
Toyota/Lexus Protocol
Frequency: 314.9 MHz
Modulation: FSK
Baud Rate: 10 kbps
ID Format: 32-bit hex (e.g., 5920441A)
Packet Length: 64 bits
Special Features: Proprietary encoding
Schrader EZ-Sensor
Frequency: 315.0 MHz (NA), 433.92 MHz (EU)
Modulation: FSK
Baud Rate: 19.2 kbps
ID Format: 32-bit hex (e.g., A42D124A)
Packet Length: 80 bits
Special Features: Multi-protocol support, programmable
Continental
Frequency: 433.92 MHz
Modulation: FSK
Baud Rate: 9.6 kbps
ID Format: 32-bit hex
Packet Length: 72 bits
Special Features: Enhanced security features
Signal Quality Indicators
RSSI (dBm)	Quality	Distance	Notes
-70 to -60	Excellent	< 5m	Very close, strong signal
-80 to -70	Good	5-15m	Normal detection range
-90 to -80	Fair	15-30m	Acceptable, may have errors
-95 to -90	Poor	30-50m	Threshold, unreliable
< -95	Noise	> 50m	Below detection threshold
🔌 ESP32 LF Trigger Setup
Hardware Assembly
Components Needed
ESP32 DevKit board
24 AWG magnet wire (50 turns)
10 cm diameter coil form
IRF540N MOSFET
TC4420 gate driver
1 µF capacitor
100 nF trimmer capacitor
10Ω resistor (2W)
12V power supply
Circuit Diagram
                    +12V
                     │
                     │
                    ┌┴┐
                    │ │ 10Ω (Current Limiting)
                    └┬┘
                     │
                ┌────┴────┐
                │         │
             D  │ IRF540N │  S
    ┌──────────┤         ├──────────┐
    │       G  │         │          │
    │          └─────────┘          │
    │               │               │
    │          ┌────┴────┐          │
    │          │ TC4420  │          │
    │    IN────┤  Gate   │          │
    │          │ Driver  │          │
    │          └─────────┘          │
    │                               │
    │          ┌─────────┐          │
    │          │   LC    │          │
    └──────────┤  Tank   ├──────────┘
               │ Circuit │
               └─────────┘
                    │
                   GND

ESP32 GPIO25 (PWM) ──────▶ TC4420 IN
ESP32 GND ───────────────▶ GND

Coil Specifications
Inductance: 1.5 mH (target)
Turns: 50 turns
Wire: 24 AWG magnet wire
Diameter: 10 cm
Core: Air core or ferrite rod (optional)
Tuning Capacitor
Resonance: 125 kHz
Calculation: C = 1 / (4π²f²L) ≈ 1.08 µF
Implementation: 1 µF fixed + 100 nF trimmer
Firmware Installation
1. Install Arduino IDE
# Download from arduino.cc or use snap
sudo snap install arduino

2. Add ESP32 Board Support
Open Arduino IDE
File → Preferences
Add to “Additional Board Manager URLs”:
https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json

Tools → Board → Boards Manager
Search “ESP32” and install
3. Upload Firmware
Open esp32_tpms_trigger.ino
Tools → Board → “ESP32 Dev Module”
Tools → Port → Select your ESP32 port
Click Upload
4. Verify Operation
ESP32 creates WiFi AP: “TPMS_Trigger”
Password: “tpms12345”
IP Address: 192.168.4.1
Using the LF Trigger
Connect to ESP32
Connect to WiFi network “TPMS_Trigger”
Password: “tpms12345”
Go to Sensor Trigger tab in app
Status should show “✅ ESP32 Connected”
Single Trigger
Select protocol (Schrader, Toyota, Continental, Generic)
Click “📡 Send Single Trigger”
ESP32 transmits 125 kHz pulse pattern
Check Live Detection tab for sensor responses
Continuous Triggering
Select protocol
Set trigger interval (0.5 - 5.0 seconds)
Click “▶️ Start Continuous”
ESP32 repeatedly triggers
Click “⏹️ Stop Continuous” to stop
Trigger & Listen Mode
Select protocol
Set listen duration (1-10 seconds)
Click “🔍 Trigger & Listen”
System triggers sensor and listens for response
Results appear in Live Detection tab
Multi-Protocol Scan
Click “🔍 Scan All Protocols”
System tries each protocol sequentially
Useful for identifying unknown sensor types
LF Trigger Safety
⚠️ Important Safety Notes:

FCC Compliance: 125 kHz is Part 15 unlicensed band
Power Limits: Keep field strength < 10 mG at 10 cm
Pacemaker Warning: Maintain 6-inch distance from medical devices
Interference: May trigger car locks - test away from vehicles
Sensor Damage: TPMS sensors are designed for repeated triggering - no damage risk
🗄️ Database Schema
Tables Overview
tpms_signals
Stores every raw TPMS signal received.

CREATE TABLE tpms_signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    tpms_id         TEXT NOT NULL,           -- Sensor ID (hex)
    timestamp       REAL NOT NULL,           -- Unix timestamp
    latitude        REAL,                    -- GPS latitude (optional)
    longitude       REAL,                    -- GPS longitude (optional)
    frequency       REAL,                    -- Frequency in Hz
    signal_strength REAL,                    -- RSSI in dBm
    snr             REAL,                    -- Signal-to-noise ratio
    pressure_psi    REAL,                    -- Tire pressure (PSI)
    temperature_c   REAL,                    -- Temperature (°C)
    battery_low     INTEGER,                 -- Battery status (0/1)
    protocol        TEXT,                    -- Protocol name
    raw_data        BLOB                     -- Raw packet bytes
);

vehicles
Stores clustered vehicles (groups of 4 sensors).

CREATE TABLE vehicles (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_hash    TEXT UNIQUE NOT NULL,    -- Hash of sorted TPMS IDs
    first_seen      REAL NOT NULL,           -- First encounter timestamp
    last_seen       REAL NOT NULL,           -- Most recent encounter
    encounter_count INTEGER DEFAULT 1,       -- Number of encounters
    tpms_ids        TEXT NOT NULL,           -- JSON array of sensor IDs
    nickname        TEXT,                    -- User-assigned name
    notes           TEXT,                    -- User notes
    metadata        TEXT                     -- JSON for additional data
);

encounters
Records each time a vehicle is detected.

CREATE TABLE encounters (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_id     INTEGER NOT NULL,         -- Foreign key to vehicles
    timestamp      REAL NOT NULL,            -- Encounter timestamp
    latitude       REAL,                     -- GPS location
    longitude      REAL,
    duration       REAL,                     -- Encounter duration (seconds)
    signal_quality REAL,                     -- Average RSSI
    FOREIGN KEY (vehicle_id) REFERENCES vehicles(id)
);

maintenance_history
Tracks tire health metrics over time.

CREATE TABLE maintenance_history (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_id        INTEGER NOT NULL,
    tpms_id           TEXT NOT NULL,
    timestamp         REAL NOT NULL,
    avg_pressure      REAL,                  -- Average pressure
    min_pressure      REAL,                  -- Minimum pressure
    max_pressure      REAL,                  -- Maximum pressure
    avg_temperature   REAL,                  -- Average temperature
    pressure_variance REAL,                  -- Pressure variance
    alert_type        TEXT,                  -- Alert type (if any)
    FOREIGN KEY (vehicle_id) REFERENCES vehicles(id)
);

🐛 Troubleshooting
Common Issues
1. HackRF Not Detected
Symptoms: hackrf_info returns “No HackRF boards found”

Solutions:

# Check USB connection
lsusb | grep HackRF

# Reinstall udev rules
sudo cp /usr/share/hackrf/udev/53-hackrf.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger

# Try different USB port
# Try different USB cable

# Check permissions
sudo usermod -a -G plugdev $USER
# Log out and back in

2. No Signals Detected
Symptoms: Scanning active but no signals appear

Common Causes:

Wrong frequency (try 314.9, 315.0, 433.92 MHz)
Antenna not connected
No vehicles nearby (sensors only transmit while moving)
Sensor battery dead
Interference from other devices
⚖️ Legal & Safety
Legal Considerations
United States (FCC Regulations)
Receiving TPMS Signals (Passive):

✅ Legal: Receiving radio signals is generally legal under FCC Part 15
✅ No License Required: Passive reception doesn’t require a license
⚠️ Privacy Concerns: TPMS IDs can potentially identify vehicles
⚠️ Use Restrictions: Don’t use for stalking, harassment, or illegal surveillance
Transmitting LF Signals (Active):

✅ Legal: 125 kHz is Part 15 unlicensed band
✅ Power Limits: Must comply with field strength limits (very low power)
⚠️ Interference: Must not cause harmful interference to licensed services
⚠️ Intent: Only use for research/educational purposes
Privacy & Ethics
Best Practices
Anonymize Data: Don’t link TPMS IDs to specific individuals
Secure Storage: Encrypt database if storing long-term
Responsible Disclosure: Report security issues to manufacturers
Transparency: Be open about what you’re doing
Respect Privacy: Don’t track specific individuals
What NOT To Do
❌ Track specific people without consent
❌ Sell or share collected data
❌ Use for stalking or harassment
❌ Interfere with vehicle operation
❌ Clone or spoof TPMS sensors
📄 License
This project is licensed under the MIT License.

📞 Support & Contact
Getting Help
Internal Dish Git: Report issues
Project Maintainer: montjac@dish.com
🙏 Acknowledgments
Credits
HackRF One: Great Scott Gadgets for excellent SDR hardware
Streamlit: For the amazing web framework
TPMS Research Community: For protocol documentation
Dish Network: For supporting this research project
Built with ❤️ for the RF research community


## Step 2: Save and commit

```bash
# Save the file (Ctrl+X, Y, Enter in nano)

# Add to git
git add README.md

# Commit
git commit -m "Add comprehensive README documentation"

# Push to internal Dish repo
git push origin main
