"""
TPMS Tracker Startup Script
Checks for HackRF and starts the application
"""

import subprocess
import sys
import time
from pathlib import Path
import re

def check_hackrf_tools():
    """Check if HackRF tools are installed"""
    print("🔍 Checking for HackRF tools...")

    try:
        result = subprocess.run(
            ['hackrf_info', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )

        # Check if command exists (even if device not found)
        if 'hackrf' in result.stdout.lower() or 'hackrf' in result.stderr.lower():
            print("✅ HackRF tools installed")
            return True

        print("✅ HackRF tools found")
        return True

    except FileNotFoundError:
        print("❌ hackrf_info not found!")
        print("\nPlease install HackRF tools:")
        print("  python install_hackrf.py")
        print("\nOr download from:")
        print("  https://github.com/greatscottgadgets/hackrf/releases")
        return False
    except Exception as e:
        print(f"⚠️  Error checking tools: {e}")
        return False

def check_hackrf_device():
    """Check if HackRF device is connected"""
    print("\n🔍 Checking for HackRF One device...")

    try:
        result = subprocess.run(
            ['hackrf_info'],
            capture_output=True,
            text=True,
            timeout=10
        )

        output = result.stdout + result.stderr

        # Check for various success indicators
        if 'Found HackRF' in output:
            print("✅ HackRF One detected and working!")

            # Extract device info
            for line in output.split('\n'):
                if any(key in line for key in ['Serial', 'Board ID', 'Firmware', 'Part ID']):
                    print(f"   {line.strip()}")

            return True

        elif 'hackrf_open() failed' in output.lower():
            print("⚠️  HackRF detected but cannot open device")
            print("\n🔧 DRIVER ISSUE - Please install WinUSB driver:")
            print_driver_instructions()
            return False

        elif 'No HackRF boards found' in output or 'not find' in output.lower():
            print("⚠️  HackRF not detected")
            print("\n📋 Troubleshooting steps:")
            print("1. Check USB connection (try different port)")
            print("2. Check Device Manager:")
            print("   - Look for 'HackRF One' under 'Universal Serial Bus devices'")
            print("   - If you see 'Unknown device', install WinUSB driver")
            print("3. Try USB 2.0 port (not USB 3.0)")
            print("4. Restart computer with HackRF plugged in")
            print_driver_instructions()
            return False

        else:
            print("⚠️  Unexpected response from HackRF")
            print(f"Output: {output[:200]}")
            return False

    except subprocess.TimeoutExpired:
        print("⚠️  HackRF not responding (timeout)")
        print("\nThis usually means the device is connected but driver is not installed")
        print_driver_instructions()
        return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def print_driver_instructions():
    """Print WinUSB driver installation instructions"""
    print("\n" + "=" * 60)
    print("📋 WinUSB Driver Installation (Required)")
    print("=" * 60)
    print("\n1. Download Zadig:")
    print("   https://zadig.akeo.ie/")
    print("\n2. Run Zadig as Administrator")
    print("\n3. In Zadig:")
    print("   - Click 'Options' → 'List All Devices'")
    print("   - Select 'HackRF One' from dropdown")
    print("   - Ensure 'WinUSB' is selected as driver")
    print("   - Click 'Replace Driver' or 'Install Driver'")
    print("   - Wait for installation to complete")
    print("\n4. Unplug and replug HackRF")
    print("\n5. Run this script again")
    print("=" * 60)

def check_device_manager():
    """Check Windows Device Manager for HackRF"""
    print("\n🔍 Checking Windows Device Manager...")

    try:
        # Use PowerShell to check for HackRF in device manager
        ps_command = """
        Get-PnpDevice | Where-Object {
            $_.FriendlyName -like '*HackRF*' -or 
            $_.FriendlyName -like '*1d50:6089*' -or
            $_.HardwareID -like '*VID_1D50&PID_6089*'
        } | Select-Object Status, Class, FriendlyName, InstanceId | Format-List
        """

        result = subprocess.run(
            ['powershell', '-Command', ps_command],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.stdout.strip():
            print("Found in Device Manager:")
            print(result.stdout)

            if 'Status      : OK' in result.stdout:
                print("✅ Device status: OK")
                return True
            elif 'Status      : Error' in result.stdout or 'Status      : Unknown' in result.stdout:
                print("⚠️  Device has error status - driver issue")
                return False
            else:
                print("⚠️  Device found but status unclear")
                return False
        else:
            print("❌ HackRF not found in Device Manager")
            print("\nPlease check:")
            print("1. USB cable is connected")
            print("2. HackRF LED is lit")
            print("3. Try different USB port")
            return False

    except Exception as e:
        print(f"⚠️  Could not check Device Manager: {e}")
        return False

def check_dependencies():
    """Check if all Python dependencies are installed"""
    print("\n📦 Checking Python dependencies...")

    required = {
        'streamlit': 'streamlit',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'plotly': 'plotly',
        'sklearn': 'scikit-learn',
        'scipy': 'scipy'
    }

    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print("  pip install -r requirements.txt")
        return False

    print("✅ All dependencies installed")
    return True

def prompt_continue():
    """Ask user if they want to continue anyway"""
    print("\n" + "=" * 60)
    response = input("Continue without HackRF? (y/n): ").strip().lower()
    return response == 'y'

def start_app():
    """Start the Streamlit application"""
    print("\n🚀 Starting TPMS Tracker...")
    print("=" * 60)
    print("\n📱 The app will open in your browser at:")
    print("   http://localhost:8501")
    print("\n💡 Tips:")
    print("   - Click 'Start Scan' to begin detection")
    print("   - Adjust frequency if needed (315 MHz for North America)")
    print("   - Drive near traffic for best results")
    print("\n⏹️  Press Ctrl+C to stop the application")
    print("=" * 60)
    print()

    try:
        subprocess.run([
            sys.executable,
            '-m',
            'streamlit',
            'run',
            'app.py',
            '--server.headless=true',
            '--browser.gatherUsageStats=false'
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

    # Check Python dependencies first
    if not check_dependencies():
        sys.exit(1)

    # Check if HackRF tools are installed
    if not check_hackrf_tools():
        sys.exit(1)

    # Check for HackRF device
    device_ok = check_hackrf_device()

    # If device not detected, check Device Manager
    if not device_ok:
        device_manager_ok = check_device_manager()

        if not device_manager_ok:
            print("\n❌ HackRF setup incomplete")
            sys.exit(1)

        # Device in manager but not working - driver issue
        print("\n⚠️  HackRF found in Device Manager but not accessible")
        print("This is typically a driver issue.")

        if not prompt_continue():
            print("\nSetup HackRF and try again")
            sys.exit(1)

    # Start app
    print("\n✅ All checks passed!")
    start_app()

if __name__ == "__main__":
    main()
