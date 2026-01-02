# 🚗 TPMS Tracker - Intelligent Vehicle Pattern Recognition

A real-time Tire Pressure Monitoring System (TPMS) signal decoder and vehicle tracking application using Software Defined Radio (SDR). Automatically detects, decodes, and tracks vehicles by their TPMS sensor signatures.

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-Linux-lightgrey)

## ✨ Features

- **Real-time TPMS Signal Decoding**
  - Schrader FSK/OOK protocols
  - Toyota/Lexus protocol support
  - Automatic protocol detection
  - Signal strength monitoring

- **Intelligent Vehicle Tracking**
  - Machine learning-based vehicle clustering
  - Automatic vehicle identification by 4-sensor patterns
  - Historical encounter tracking
  - Predictive analytics for repeat encounters

- **Comprehensive Database**
  - SQLite-based sensor and vehicle database
  - Full signal history with timestamps
  - Maintenance tracking and alerts
  - Tire pressure/temperature monitoring

- **Modern Web Interface**
  - Real-time signal visualization
  - Interactive charts and graphs
  - Vehicle database management
  - Analytics dashboard

## 📋 Requirements

### Hardware
- **HackRF One** SDR receiver
- USB cable and antenna (315 MHz or 433.92 MHz)
- Linux system (tested on Ubuntu/Debian)

### Software
- Python 3.9 or higher
- HackRF tools and libraries
- Virtual environment (recommended)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/askjake/TPMS.git
cd TPMS
