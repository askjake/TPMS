#!/usr/bin/env python3
"""
TPMS Tracker - Linux Startup Script
Checks dependencies and launches the Streamlit app
"""

import sys
import os
import subprocess
import shutil
from pathlib import Path

def print_header():
    """Print startup header"""
    print("\n" + "="*60)
    print("🚗 TPMS Tracker - Startup Script (Linux)")
    print("="*60 + "\n")

def check_dependencies():
    """Check if required Python packages are installed"""
    print("📦 Checking Python dependencies...")
    required = ['streamlit', 'numpy', 'scipy', 'pandas']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies installed")
    return True

def check_hackrf_tools():
    """Check if HackRF tools are installed and accessible"""
    print("🔍 Checking for HackRF tools...\n")
    
    # Check for hackrf_transfer (most important tool)
    hackrf_transfer = shutil.which('hackrf_transfer')
    hackrf_info = shutil.which('hackrf_info')
    
    if hackrf_transfer and hackrf_info:
        print(f"✅ HackRF tools found:")
        print(f"   hackrf_transfer: {hackrf_transfer}")
        print(f"   hackrf_info: {hackrf_info}")
        return True
    else:
        print("❌ HackRF tools not found in PATH")
        print("\nInstall with:")
        print("   sudo apt install hackrf libhackrf-dev")
        return False

def check_hackrf_device():
    """Check if HackRF device is connected and accessible"""
    print("\n🔌 Checking for HackRF device...\n")
    
    try:
        result = subprocess.run(
            ['hackrf_info'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print("✅ HackRF device detected:")
            # Print relevant info
            for line in result.stdout.split('\n'):
                if any(keyword in line for keyword in ['Serial', 'Board ID', 'Firmware']):
                    print(f"   {line.strip()}")
            return True
        else:
            print("❌ HackRF device not responding")
            print("\nTroubleshooting:")
            print("1. Check USB connection")
            print("2. Check permissions: groups (should include 'plugdev')")
            print("3. Try: sudo udevadm control --reload-rules && sudo udevadm trigger")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ HackRF device check timed out")
        return False
    except FileNotFoundError:
        print("❌ hackrf_info command not found")
        return False
    except Exception as e:
        print(f"❌ Error checking device: {e}")
        return False

def start_streamlit():
    """Launch the Streamlit application"""
    print("\n🚀 Starting TPMS Tracker application...\n")
    print("="*60)
    print("📱 Access the app at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    try:
        # Launch streamlit
        subprocess.run([
            'streamlit', 'run', 'app.py',
            '--server.address', 'localhost',
            '--server.port', '8501',
            '--browser.serverAddress', 'localhost'
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down TPMS Tracker...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        sys.exit(1)

def main():
    """Main startup sequence"""
    print_header()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed")
        sys.exit(1)
    
    # Check HackRF tools
    tools_ok = check_hackrf_tools()
    
    # Check HackRF device
    device_ok = check_hackrf_device()
    
    if not tools_ok:
        print("\n⚠️  Cannot continue without HackRF tools")
        sys.exit(1)
    
    if not device_ok:
        print("\n⚠️  Warning: HackRF device not detected")
        response = input("Continue anyway? (y/n): ").lower().strip()
        if response != 'y':
            print("Exiting...")
            sys.exit(1)
    
    # Everything looks good, start the app
    start_streamlit()

if __name__ == "__main__":
    main()
