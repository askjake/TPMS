"""
TPMS Tracker Startup Script
Checks for HackRF and starts the application
"""

import subprocess
import sys
import time
from pathlib import Path
import re
import os

def find_hackrf_tools():
    """Find HackRF tools in common locations"""
    print("🔍 Searching for HackRF tools...")
    
    # Common installation paths
    search_paths = [
        Path("C:/hackrf/bin"),
        Path("C:/Program Files/HackRF/bin"),
        Path("C:/Program Files (x86)/HackRF/bin"),
        Path(os.environ.get('PROGRAMFILES', 'C:/Program Files')) / "HackRF/bin",
        Path(os.environ.get('LOCALAPPDATA', '')) / "Programs/HackRF/bin",
        # Universal Radio Hacker might bundle it
        Path(os.environ.get('PROGRAMFILES', 'C:/Program Files')) / "Universal Radio Hacker",
        Path(os.environ.get('LOCALAPPDATA', '')) / "Programs/urh",
    ]
    
    # Also check PATH
    path_env = os.environ.get('PATH', '').split(';')
    for path_dir in path_env:
        if path_dir:
            search_paths.append(Path(path_dir))
    
    # Search for hackrf_info.exe or hackrf_transfer.exe
    for search_path in search_paths:
        if search_path.exists():
            hackrf_info = search_path / "hackrf_info.exe"
            hackrf_transfer = search_path / "hackrf_transfer.exe"
            
            if hackrf_info.exists() or hackrf_transfer.exists():
                print(f"✅ Found HackRF tools at: {search_path}")
                
                # Add to PATH for this session
                os.environ['PATH'] = f"{search_path};{os.environ['PATH']}"
                
                return search_path
    
    return None

def check_hackrf_via_usb():
    """Check for HackRF using Windows USB detection"""
    print("\n🔍 Checking USB devices...")
    
    try:
        # Use PowerShell to check for HackRF
        ps_command = """
        Get-PnpDevice | Where-Object {
            $_.FriendlyName -like '*HackRF*' -or 
            $_.HardwareID -like '*VID_1D50&PID_6089*'
        } | Select-Object Status, Class, FriendlyName | Format-List
        """
        
        result = subprocess.run(
            ['powershell', '-Command', ps_command],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        output = result.stdout
        
        if 'HackRF' in output or '1D50' in output:
            print("✅ HackRF One found in USB devices:")
            print(output)
            
            if 'Status      : OK' in output:
                print("✅ Device status: OK")
                return True
            else:
                print("⚠️  Device found but may have driver issue")
                return False
        else:
            print("❌ HackRF not found in USB devices")
            return False
            
    except Exception as e:
        print(f"⚠️  Could not check USB devices: {e}")
        return False

def check_urh_installation():
    """Check if Universal Radio Hacker is installed and can access HackRF"""
    print("\n🔍 Checking for Universal Radio Hacker...")
    
    urh_paths = [
        Path(os.environ.get('PROGRAMFILES', 'C:/Program Files')) / "Universal Radio Hacker",
        Path(os.environ.get('PROGRAMFILES(X86)', 'C:/Program Files (x86)')) / "Universal Radio Hacker",
        Path(os.environ.get('LOCALAPPDATA', '')) / "Programs/urh",
    ]
    
    for urh_path in urh_paths:
        if urh_path.exists():
            print(f"✅ Found Universal Radio Hacker at: {urh_path}")
            
            # Check if it has HackRF tools
            for file in ['hackrf_info.exe', 'hackrf_transfer.exe']:
                found_files = list(urh_path.rglob(file))
                if found_files:
                    tool_path = found_files[0].parent
                    print(f"✅ Found {file} at: {tool_path}")
                    
                    # Add to PATH
                    os.environ['PATH'] = f"{tool_path};{os.environ['PATH']}"
                    return tool_path
    
    return None

def test_hackrf_direct():
    """Try to run hackrf_info directly"""
    print("\n🧪 Testing HackRF access...")
    
    try:
        result = subprocess.run(
            ['hackrf_info'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        output = result.stdout + result.stderr
        print("Output:", output[:500])
        
        if 'Found HackRF' in output:
            return True
        
        return False
        
    except FileNotFoundError:
        return False
    except Exception as e:
        print(f"Error: {e}")
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
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    # Try to find HackRF tools
    tools_path = find_hackrf_tools()
    
    if not tools_path:
        # Try URH installation
        tools_path = check_urh_installation()
    
    if not tools_path:
        print("\n❌ HackRF tools not found in standard locations")
        print("\nSince Universal Radio Hacker works, you have two options:")
        print("\n1. Install HackRF tools separately:")
        print("   python install_hackrf.py")
        print("\n2. Use mock mode for testing (no hardware required)")
        
        response = input("\nUse mock mode? (y/n): ").strip().lower()
        if response == 'y':
            print("\n✅ Starting in MOCK MODE (simulated data)")
            # Create a flag file for mock mode
            with open('.mock_mode', 'w') as f:
                f.write('1')
            start_app()
            return
        else:
            input("\nPress Enter to exit...")
            sys.exit(1)
    
    # Check if device is accessible
    device_ok = check_hackrf_via_usb()
    
    if not device_ok:
        print("\n⚠️  HackRF detected but may not be accessible")
        response = input("\nTry to start anyway? (y/n): ").strip().lower()
        if response != 'y':
            input("\nPress Enter to exit...")
            sys.exit(1)
    
    # Test direct access
    if not test_hackrf_direct():
        print("\n⚠️  Could not communicate with HackRF")
        print("\nPossible issues:")
        print("1. WinUSB driver not installed (use Zadig)")
        print("2. Device in use by another application")
        print("3. USB connection issue")
        
        response = input("\nTry to start anyway? (y/n): ").strip().lower()
        if response != 'y':
            input("\nPress Enter to exit...")
            sys.exit(1)
    
    # Remove mock mode flag if it exists
    if os.path.exists('.mock_mode'):
        os.remove('.mock_mode')
    
    # Start app
    print("\n✅ All checks passed!")
    start_app()

if __name__ == "__main__":
    main()
