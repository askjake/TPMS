#!/usr/bin/env python3
"""
Diagnostic script to troubleshoot TPMS signal reception
"""

import subprocess
import time
import numpy as np
from pathlib import Path


def check_hackrf_connection():
    """Verify HackRF is connected and working"""
    print("🔍 Checking HackRF connection...")
    try:
        result = subprocess.run(['hackrf_info'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ HackRF detected:")
            print(result.stdout)
            return True
        else:
            print("❌ HackRF not detected")
            return False
    except Exception as e:
        print(f"❌ Error checking HackRF: {e}")
        return False


def capture_raw_samples(frequency_mhz, duration_sec=30):
    """Capture raw IQ samples for analysis"""
    print(f"\n📡 Capturing {duration_sec}s at {frequency_mhz} MHz...")

    filename = f"diagnostic_capture_{frequency_mhz}mhz.bin"
    sample_rate = 2457600
    num_samples = sample_rate * duration_sec

    cmd = [
        'hackrf_transfer',
        '-r', filename,
        '-f', str(int(frequency_mhz * 1e6)),
        '-s', str(sample_rate),
        '-n', str(num_samples),
        '-a', '1',  # Enable amp
        '-l', '32',  # LNA gain
        '-g', '40'  # VGA gain
    ]

    try:
        subprocess.run(cmd, timeout=duration_sec + 10)

        # Check file size
        file_path = Path(filename)
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"✅ Captured {size_mb:.1f} MB")

            # Analyze samples
            analyze_capture(filename, sample_rate)
            return True
        else:
            print("❌ Capture file not created")
            return False

    except Exception as e:
        print(f"❌ Capture failed: {e}")
        return False


def analyze_capture(filename, sample_rate):
    """Basic analysis of captured samples"""
    print(f"\n📊 Analyzing {filename}...")

    try:
        # Read first 1 million samples
        samples = np.fromfile(filename, dtype=np.int8, count=2000000)

        if len(samples) == 0:
            print("❌ No samples in file")
            return

        # Convert to complex
        iq = samples[::2] + 1j * samples[1::2]

        # Calculate power
        power = np.abs(iq) ** 2
        avg_power = np.mean(power)
        max_power = np.max(power)

        # Convert to dBm (approximate)
        avg_dbm = 10 * np.log10(avg_power) - 50  # Rough calibration
        max_dbm = 10 * np.log10(max_power) - 50

        print(f"  Average power: {avg_dbm:.1f} dBm")
        print(f"  Peak power: {max_dbm:.1f} dBm")

        # Look for bursts (potential signals)
        threshold = avg_power * 10  # 10x above noise floor
        bursts = power > threshold
        burst_count = np.sum(np.diff(bursts.astype(int)) > 0)

        print(f"  Detected bursts: {burst_count}")

        if burst_count > 0:
            print("✅ Signal activity detected!")
        else:
            print("⚠️  No clear signal bursts - may be only noise")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")


def main():
    print("=" * 60)
    print("TPMS Reception Diagnostic Tool")
    print("=" * 60)

    # Check HackRF
    if not check_hackrf_connection():
        print("\n❌ Cannot proceed without HackRF")
        return

    # Test each frequency
    frequencies = [314.9, 315.0, 433.92]

    print("\n" + "=" * 60)
    print("Starting multi-frequency capture test")
    print("Drive around or park near vehicles during this test")
    print("=" * 60)

    for freq in frequencies:
        print(f"\n{'=' * 60}")
        capture_raw_samples(freq, duration_sec=30)
        print(f"{'=' * 60}")
        time.sleep(2)

    print("\n" + "=" * 60)
    print("Diagnostic complete!")
    print("=" * 60)
    print("\nResults:")
    print("  - Check for 'Signal activity detected' messages above")
    print("  - If no activity on any frequency, possible issues:")
    print("    1. Antenna not connected or damaged")
    print("    2. Sensors not transmitting (need movement/LF trigger)")
    print("    3. Frequency mismatch (some regions use different frequencies)")
    print("    4. Signal too weak (try closer to vehicles)")
    print("\nNext steps:")
    print("  - Review diagnostic_capture_*.bin files")
    print("  - Try inspecting with: inspectrum diagnostic_capture_315.0mhz.bin")
    print("  - Check antenna connection")
    print("  - Test with ESP32 LF trigger to force sensor transmission")


if __name__ == "__main__":
    main()
