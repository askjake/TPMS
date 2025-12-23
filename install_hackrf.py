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
