"""
Debug and diagnostic tools for TPMS Tracker
"""
import numpy as np
from scipy import signal as scipy_signal
import subprocess
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass
from config import config
import shutil

@dataclass
class SpectrumPeak:
    frequency: float
    power: float
    bandwidth: float
    snr: float

@dataclass
class ModulationAnalysis:
    frequency: float
    modulation_type: str
    confidence: float
    baud_rate: int
    bandwidth: float
    characteristics: Dict

@dataclass
class HardwareInfo:
    device_found: bool
    serial_number: str
    board_id: str
    firmware_version: str
    part_id: str
    operacake_detected: bool

class DebugTools:
    def __init__(self):
        self.hackrf_path = shutil.which('hackrf_transfer') or 'hackrf_transfer'
        self.hackrf_info_path = shutil.which('hackrf_info') or 'hackrf_info'
    
    def get_hardware_info(self) -> HardwareInfo:
        """Get HackRF hardware information"""
        try:
            result = subprocess.run(
                [self.hackrf_info_path],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                return HardwareInfo(
                    device_found=False,
                    serial_number="N/A",
                    board_id="N/A",
                    firmware_version="N/A",
                    part_id="N/A",
                    operacake_detected=False
                )
            
            output = result.stdout
            
            # Parse output
            serial = "Unknown"
            board_id = "Unknown"
            firmware = "Unknown"
            part_id = "Unknown"
            operacake = False
            
            for line in output.split('\n'):
                if "Serial number:" in line:
                    serial = line.split(':')[1].strip()
                elif "Board ID Number:" in line:
                    board_id = line.split(':')[1].strip()
                elif "Firmware Version:" in line:
                    firmware = line.split(':')[1].strip()
                elif "Part ID Number:" in line:
                    part_id = line.split(':')[1].strip()
                elif "Operacake found" in line:
                    operacake = True
            
            return HardwareInfo(
                device_found=True,
                serial_number=serial,
                board_id=board_id,
                firmware_version=firmware,
                part_id=part_id,
                operacake_detected=operacake
            )
        
        except Exception as e:
            print(f"Error getting hardware info: {e}")
            return HardwareInfo(
                device_found=False,
                serial_number="Error",
                board_id="Error",
                firmware_version="Error",
                part_id="Error",
                operacake_detected=False
            )
    
    def spectrum_scan(self, start_freq: float, end_freq: float, 
                     step: float = 0.5, duration: float = 0.5) -> List[SpectrumPeak]:
        """
        Perform a spectrum scan across frequency range
        
        Args:
            start_freq: Start frequency in MHz
            end_freq: End frequency in MHz
            step: Step size in MHz
            duration: Duration to sample each frequency in seconds
        
        Returns:
            List of detected peaks
        """
        peaks = []
        num_steps = int((end_freq - start_freq) / step) + 1
        
        for i in range(num_steps):
            freq = start_freq + (i * step)
            
            # Capture samples at this frequency
            power, snr = self._measure_frequency(freq, duration)
            
            if power > -80:  # Only record significant signals
                # Estimate bandwidth
                bandwidth = self._estimate_bandwidth(freq, power)
                
                peaks.append(SpectrumPeak(
                    frequency=freq,
                    power=power,
                    bandwidth=bandwidth,
                    snr=snr
                ))
        
        # Sort by power
        peaks.sort(key=lambda x: x.power, reverse=True)
        
        return peaks
    
    def _measure_frequency(self, freq: float, duration: float) -> Tuple[float, float]:
        """Measure power and SNR at a specific frequency"""
        try:
            # Calculate number of samples
            num_samples = int(config.SAMPLE_RATE * duration)
            
            # Run hackrf_transfer to capture samples
            cmd = [
                self.hackrf_path,
                '-r', '-',
                '-f', str(int(freq * 1e6)),
                '-s', str(config.SAMPLE_RATE),
                '-g', '40',  # High gain for detection
                '-l', '32',
                '-a', '1',
                '-n', str(num_samples)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=duration + 2
            )
            
            if result.returncode != 0 or len(result.stdout) < 100:
                return -100, 0
            
            # Convert to IQ samples
            iq_data = np.frombuffer(result.stdout, dtype=np.int8)
            
            # Ensure even length
            if len(iq_data) % 2 != 0:
                iq_data = iq_data[:-1]
            
            i_samples = iq_data[0::2].astype(np.float32) / 128.0
            q_samples = iq_data[1::2].astype(np.float32) / 128.0
            
            min_len = min(len(i_samples), len(q_samples))
            complex_samples = i_samples[:min_len] + 1j * q_samples[:min_len]
            
            # Calculate power
            power = np.mean(np.abs(complex_samples) ** 2)
            power_dbm = 10 * np.log10(power + 1e-10) - 60
            
            # Calculate SNR
            signal_power = np.max(np.abs(complex_samples) ** 2)
            noise_power = np.median(np.abs(complex_samples) ** 2)
            snr = 10 * np.log10((signal_power / (noise_power + 1e-10)))
            
            return power_dbm, snr
        
        except Exception as e:
            print(f"Error measuring {freq} MHz: {e}")
            return -100, 0
    
    def _estimate_bandwidth(self, center_freq: float, center_power: float) -> float:
        """Estimate signal bandwidth"""
        # Simple bandwidth estimation by checking adjacent frequencies
        step = 0.1  # MHz
        bandwidth = 0
        
        # Check up to 1 MHz on each side
        for offset in [0.1, 0.2, 0.3, 0.4, 0.5]:
            power_high, _ = self._measure_frequency(center_freq + offset, 0.2)
            power_low, _ = self._measure_frequency(center_freq - offset, 0.2)
            
            # If power drops by more than 6dB, we've found the edge
            if power_high < center_power - 6 or power_low < center_power - 6:
                bandwidth = offset * 2
                break
        
        return bandwidth if bandwidth > 0 else 0.2  # Default to 200 kHz
    
    def analyze_modulation(self, frequency: float, duration: float = 2.0) -> ModulationAnalysis:
        """
        Analyze modulation type at a specific frequency
        
        Args:
            frequency: Frequency to analyze in MHz
            duration: Duration to capture in seconds
        
        Returns:
            ModulationAnalysis object
        """
        try:
            # Capture samples
            num_samples = int(config.SAMPLE_RATE * duration)
            
            cmd = [
                self.hackrf_path,
                '-r', '-',
                '-f', str(int(frequency * 1e6)),
                '-s', str(config.SAMPLE_RATE),
                '-g', '40',
                '-l', '32',
                '-a', '1',
                '-n', str(num_samples)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=duration + 2
            )
            
            if result.returncode != 0 or len(result.stdout) < 1000:
                return ModulationAnalysis(
                    frequency=frequency,
                    modulation_type="Unknown",
                    confidence=0.0,
                    baud_rate=0,
                    bandwidth=0,
                    characteristics={}
                )
            
            # Convert to IQ samples
            iq_data = np.frombuffer(result.stdout, dtype=np.int8)
            
            if len(iq_data) % 2 != 0:
                iq_data = iq_data[:-1]
            
            i_samples = iq_data[0::2].astype(np.float32) / 128.0
            q_samples = iq_data[1::2].astype(np.float32) / 128.0
            
            min_len = min(len(i_samples), len(q_samples))
            complex_samples = i_samples[:min_len] + 1j * q_samples[:min_len]
            
            # Analyze characteristics
            characteristics = self._analyze_signal_characteristics(complex_samples)
            
            # Determine modulation type
            mod_type, confidence = self._classify_modulation(characteristics)
            
            # Estimate baud rate
            baud_rate = self._estimate_baud_rate(complex_samples)
            
            # Estimate bandwidth
            bandwidth = self._estimate_bandwidth(frequency, characteristics['avg_power'])
            
            return ModulationAnalysis(
                frequency=frequency,
                modulation_type=mod_type,
                confidence=confidence,
                baud_rate=baud_rate,
                bandwidth=bandwidth,
                characteristics=characteristics
            )
        
        except Exception as e:
            print(f"Error analyzing modulation: {e}")
            return ModulationAnalysis(
                frequency=frequency,
                modulation_type="Error",
                confidence=0.0,
                baud_rate=0,
                bandwidth=0,
                characteristics={"error": str(e)}
            )
    
    def _analyze_signal_characteristics(self, samples: np.ndarray) -> Dict:
        """Analyze signal characteristics"""
        # Amplitude
        amplitude = np.abs(samples)
        amp_mean = np.mean(amplitude)
        amp_std = np.std(amplitude)
        amp_var = np.var(amplitude)
        
        # Phase
        phase = np.angle(samples)
        phase_diff = np.diff(phase)
        phase_var = np.var(phase_diff)
        
        # Frequency (instantaneous)
        inst_freq = np.diff(np.unwrap(phase))
        freq_var = np.var(inst_freq)
        
        # Power
        power = amplitude ** 2
        avg_power = 10 * np.log10(np.mean(power) + 1e-10) - 60
        
        return {
            'amp_mean': float(amp_mean),
            'amp_std': float(amp_std),
            'amp_var': float(amp_var),
            'phase_var': float(phase_var),
            'freq_var': float(freq_var),
            'avg_power': float(avg_power)
        }
    
    def _classify_modulation(self, characteristics: Dict) -> Tuple[str, float]:
        """Classify modulation type based on characteristics"""
        amp_var = characteristics['amp_var']
        phase_var = characteristics['phase_var']
        freq_var = characteristics['freq_var']
        
        # Decision tree based on variance
        if amp_var > 0.3:
            if phase_var < 0.2:
                return "ASK/OOK", 0.8
            else:
                return "QAM", 0.6
        
        elif phase_var > 0.5:
            if amp_var < 0.1:
                return "PSK/BPSK", 0.7
            else:
                return "QPSK", 0.6
        
        elif freq_var > 0.3:
            return "FSK", 0.75
        
        else:
            return "Unknown/CW", 0.5
    
    def _estimate_baud_rate(self, samples: np.ndarray) -> int:
        """Estimate symbol/baud rate"""
        try:
            # Use autocorrelation
            amplitude = np.abs(samples)
            
            # Normalize
            amplitude = amplitude - np.mean(amplitude)
            
            # Autocorrelation
            autocorr = np.correlate(amplitude, amplitude, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find peaks
            peaks, _ = scipy_signal.find_peaks(autocorr, distance=10, height=np.max(autocorr)*0.3)
            
            if len(peaks) > 1:
                # Average distance between peaks
                avg_distance = np.mean(np.diff(peaks[:5]))
                baud_rate = int(config.SAMPLE_RATE / avg_distance)
                
                # Round to common baud rates
                common_rates = [1200, 2400, 4800, 9600, 19200, 38400, 57600, 115200]
                closest = min(common_rates, key=lambda x: abs(x - baud_rate))
                
                if abs(closest - baud_rate) < baud_rate * 0.2:  # Within 20%
                    return closest
                
                return baud_rate
        
        except Exception:
            pass
        
        return 0
    
    def run_full_diagnostic(self) -> Dict:
        """Run complete diagnostic suite"""
        print("🔍 Starting full diagnostic...")
        
        results = {
            'timestamp': time.time(),
            'hardware': None,
            'spectrum_scan': [],
            'modulation_analysis': []
        }
        
        # 1. Hardware check
        print("📡 Checking hardware...")
        results['hardware'] = self.get_hardware_info()
        
        if not results['hardware'].device_found:
            print("❌ Hardware not found!")
            return results
        
        # 2. Spectrum scan on TPMS frequencies
        print("📊 Scanning TPMS frequencies...")
        for freq in config.FREQUENCIES:
            # Scan ±5 MHz around each TPMS frequency
            peaks = self.spectrum_scan(freq - 5, freq + 5, step=0.5, duration=0.3)
            results['spectrum_scan'].extend(peaks)
        
        # 3. Analyze top peaks
        print("🔬 Analyzing detected signals...")
        top_peaks = sorted(results['spectrum_scan'], key=lambda x: x.power, reverse=True)[:5]
        
        for peak in top_peaks:
            print(f"   Analyzing {peak.frequency:.2f} MHz...")
            analysis = self.analyze_modulation(peak.frequency, duration=1.0)
            results['modulation_analysis'].append(analysis)
        
        print("✅ Diagnostic complete!")
        
        return results
