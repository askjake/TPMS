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
