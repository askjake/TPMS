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
import json

# Latest release info
HACKRF_RELEASES_API = "https://api.github.com/repos/greatscottgadgets/hackrf/releases/latest"
HACKRF_FALLBACK_VERSION = "2024.02.1"
INSTALL_DIR = Path("C:/hackrf")

def get_latest_release():
    """Get the latest HackRF release info from GitHub API"""
    print("🔍 Checking for latest HackRF release...")

    try:
        req = urllib.request.Request(HACKRF_RELEASES_API)
        req.add_header('User-Agent', 'TPMS-Tracker-Installer')

        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())

            version = data['tag_name'].replace('v', '')

            # Find the Windows x64 asset
            for asset in data['assets']:
                if 'win-x64' in asset['name'] and asset['name'].endswith('.zip'):
                    return {
                        'version': version,
                        'url': asset['browser_download_url'],
                        'name': asset['name']
                    }

            print("⚠️  No Windows x64 release found in latest version")
            return None

    except Exception as e:
        print(f"⚠️  Could not fetch latest release: {e}")
        return None

def get_fallback_url():
    """Get fallback download URLs for known versions"""
    # These are direct links to known working releases
    fallback_urls = [
        {
            'version': '2024.02.1',
            'url': 'https://github.com/greatscottgadgets/hackrf/releases/download/v2024.02.1/hackrf-2024.02.1-win-x64.zip',
            'name': 'hackrf-2024.02.1-win-x64.zip'
        },
        {
            'version': '2023.01.1',
            'url': 'https://github.com/greatscottgadgets/hackrf/releases/download/v2023.01.1/hackrf-2023.01.1-win-x64.zip',
            'name': 'hackrf-2023.01.1-win-x64.zip'
        },
        {
            'version': '2022.09.1',
            'url': 'https://github.com/greatscottgadgets/hackrf/releases/download/v2022.09.1/hackrf-2022.09.1-win-x64.zip',
            'name': 'hackrf-2022.09.1-win-x64.zip'
        }
    ]

    return fallback_urls

def is_admin():
    """Check if script is running with admin privileges"""
    try:
        return os.getuid() == 0
    except AttributeError:
        import ctypes
        return ctypes.windll.shell32.IsUserAnAdmin() != 0

def download_hackrf(release_info):
    """Download HackRF tools"""
    print(f"📥 Downloading HackRF {release_info['version']}...")
    print(f"URL: {release_info['url']}")

    zip_path = Path("hackrf.zip")

    try:
        # Add headers to avoid 403 errors
        req = urllib.request.Request(release_info['url'])
        req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')

        with urllib.request.urlopen(req, timeout=60) as response:
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            downloaded = 0

            with open(zip_path, 'wb') as f:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break

                    downloaded += len(buffer)
                    f.write(buffer)

                    # Show progress
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='')

        print("\n✅ Download complete!")
        return zip_path

    except urllib.error.HTTPError as e:
        print(f"\n❌ HTTP Error {e.code}: {e.reason}")
        return None
    except urllib.error.URLError as e:
        print(f"\n❌ URL Error: {e.reason}")
        return None
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return None

def extract_hackrf(zip_path):
    """Extract HackRF tools"""
    print(f"📦 Extracting to {INSTALL_DIR}...")

    try:
        INSTALL_DIR.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # List contents
            file_list = zip_ref.namelist()
            print(f"Extracting {len(file_list)} files...")

            # Extract all files
            zip_ref.extractall(INSTALL_DIR)

        print("✅ Extraction complete!")

        # Clean up
        zip_path.unlink()

        # Find the bin directory (it might be in a subdirectory)
        bin_dirs = list(INSTALL_DIR.rglob("bin"))
        if bin_dirs:
            print(f"Found bin directory: {bin_dirs[0]}")
            return bin_dirs[0].parent
        else:
            print("⚠️  No bin directory found, using install root")
            return INSTALL_DIR

    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return None

def add_to_path(hackrf_dir):
    """Add HackRF bin directory to system PATH"""
    bin_dir = str(hackrf_dir / "bin")

    # Check if bin directory exists
    if not (hackrf_dir / "bin").exists():
        print(f"⚠️  Bin directory not found at {bin_dir}")
        # Try to find it
        possible_bins = list(hackrf_dir.rglob("hackrf_transfer.exe"))
        if possible_bins:
            bin_dir = str(possible_bins[0].parent)
            print(f"Found hackrf_transfer.exe at {bin_dir}")
        else:
            print("❌ Could not locate hackrf_transfer.exe")
            return False

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
        if bin_dir.lower() not in current_path.lower():
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

    # Refresh environment variables
    os.environ['PATH'] = os.popen('echo %PATH%').read().strip()

    try:
        result = subprocess.run(
            ['hackrf_info', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0 or 'hackrf' in result.stdout.lower():
            print("✅ HackRF tools installed successfully!")
            if result.stdout:
                print(f"Output: {result.stdout.strip()}")
            return True
        else:
            print("⚠️  Installation complete but hackrf_info not responding properly")
            print("You may need to restart your terminal")
            return False

    except FileNotFoundError:
        print("⚠️  hackrf_info not found in PATH")
        print("Please restart your terminal and try again")
        return False
    except Exception as e:
        print(f"⚠️  Test failed: {e}")
        return False

def manual_install_instructions():
    """Print manual installation instructions"""
    print("\n" + "=" * 60)
    print("📋 Manual Installation Instructions")
    print("=" * 60)
    print("\n1. Go to: https://github.com/greatscottgadgets/hackrf/releases")
    print("2. Download the latest 'hackrf-*-win-x64.zip' file")
    print("3. Extract to C:\\hackrf")
    print("4. Add C:\\hackrf\\bin to your PATH:")
    print("   - Search for 'Environment Variables' in Windows")
    print("   - Edit 'Path' under User variables")
    print("   - Add new entry: C:\\hackrf\\bin")
    print("5. Restart your terminal")
    print("6. Test with: hackrf_info")
    print()

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
        if result.returncode == 0 or 'hackrf' in result.stdout.decode().lower():
            print("✅ HackRF tools already installed!")
            print("Run 'hackrf_info' to test your device")
            return
    except:
        pass

    # Try to get latest release
    release_info = get_latest_release()

    # If that fails, try fallback URLs
    if not release_info:
        print("\n📋 Trying fallback URLs...")
        fallback_urls = get_fallback_url()

        for fb in fallback_urls:
            print(f"\nTrying version {fb['version']}...")
            zip_path = download_hackrf(fb)

            if zip_path:
                release_info = fb
                break

        if not release_info:
            print("\n❌ All download attempts failed")
            manual_install_instructions()
            sys.exit(1)
    else:
        # Download latest release
        zip_path = download_hackrf(release_info)

        if not zip_path:
            print("\n❌ Download failed")
            manual_install_instructions()
            sys.exit(1)

    # Extract
    hackrf_dir = extract_hackrf(zip_path)
    if not hackrf_dir:
        manual_install_instructions()
        sys.exit(1)

    # Add to PATH
    add_to_path(hackrf_dir)

    # Test
    test_installation()

    print("\n" + "=" * 60)
    print("📋 Next Steps:")
    print("=" * 60)
    print("1. RESTART your terminal/command prompt")
    print("2. Plug in your HackRF One")
    print("3. Install WinUSB driver using Zadig:")
    print("   - Download: https://zadig.akeo.ie/")
    print("   - Run as Administrator")
    print("   - Options → List All Devices")
    print("   - Select 'HackRF One' device")
    print("   - Select 'WinUSB' driver")
    print("   - Click 'Replace Driver' or 'Install Driver'")
    print("4. Test with: hackrf_info")
    print("5. Run the app: start.bat")
    print()

if __name__ == "__main__":
    main()
