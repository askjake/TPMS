"""
TPMS Trigger Module
Sends LF (Low Frequency) activation signals to wake up TPMS sensors
Mimics functionality of EZ-sensor programming tools
"""

import subprocess
import time
import threading
from typing import Optional, Callable
import numpy as np
from config import config

class TPMSTrigger:
    """
    LF Trigger for TPMS sensors
    Sends 125 kHz activation signal to wake up sensors for:
    - Programming
    - Diagnostics
    - Forced transmission
    """
    
    def __init__(self):
        self.is_transmitting = False
        self.trigger_thread = None
        
        # LF trigger parameters (matching Schrader specs)
        self.lf_frequency = 125000  # 125 kHz
        self.trigger_duration = 0.1  # 100ms pulse
        self.trigger_interval = 1.0  # 1 second between triggers
        
        # Trigger patterns for different manufacturers
        self.trigger_patterns = {
            'schrader': {
                'frequency': 125000,
                'pulse_width': 0.1,
                'pulse_count': 3,
                'pulse_interval': 0.05
            },
            'toyota': {
                'frequency': 125000,
                'pulse_width': 0.15,
                'pulse_count': 2,
                'pulse_interval': 0.1
            },
            'continental': {
                'frequency': 125000,
                'pulse_width': 0.08,
                'pulse_count': 4,
                'pulse_interval': 0.03
            },
            'generic': {
                'frequency': 125000,
                'pulse_width': 0.1,
                'pulse_count': 1,
                'pulse_interval': 0
            }
        }
        
        print("🔧 TPMS Trigger initialized")
        print(f"   LF Frequency: {self.lf_frequency / 1000:.1f} kHz")
    
    def send_trigger(self, protocol: str = 'generic', callback: Optional[Callable] = None):
        """
        Send LF trigger signal to activate TPMS sensors
        
        Args:
            protocol: Trigger pattern to use ('schrader', 'toyota', 'continental', 'generic')
            callback: Optional callback function when sensor responds
        """
        if protocol not in self.trigger_patterns:
            print(f"⚠️  Unknown protocol: {protocol}, using generic")
            protocol = 'generic'
        
        pattern = self.trigger_patterns[protocol]
        
        print(f"📡 Sending {protocol} trigger pattern...")
        print(f"   Pulses: {pattern['pulse_count']}")
        print(f"   Pulse width: {pattern['pulse_width']*1000:.0f}ms")
        
        # Send trigger pulses
        for i in range(pattern['pulse_count']):
            self._send_lf_pulse(
                pattern['frequency'],
                pattern['pulse_width']
            )
            
            if i < pattern['pulse_count'] - 1:
                time.sleep(pattern['pulse_interval'])
        
        print("✅ Trigger sent")
        
        if callback:
            callback()
    
    def _send_lf_pulse(self, frequency: int, duration: float):
        """
        Send a single LF pulse using HackRF
        
        Note: HackRF One can transmit at 125 kHz, but it's outside its optimal range.
        For production use, consider a dedicated LF transmitter module.
        """
        try:
            # Generate LF waveform
            sample_rate = 2_000_000  # 2 MS/s
            samples = int(sample_rate * duration)
            
            # Generate 125 kHz sine wave
            t = np.linspace(0, duration, samples, endpoint=False)
            waveform = np.sin(2 * np.pi * frequency * t)
            
            # Convert to I/Q format for HackRF
            i_samples = (waveform * 127).astype(np.int8)
            q_samples = np.zeros_like(i_samples)  # No phase shift
            
            # Interleave I and Q
            iq_data = np.empty(samples * 2, dtype=np.int8)
            iq_data[0::2] = i_samples
            iq_data[1::2] = q_samples
            
            # Transmit via HackRF
            cmd = [
                'hackrf_transfer',
                '-t', '-',  # Transmit from stdin
                '-f', str(frequency),
                '-s', str(sample_rate),
                '-x', '47',  # TX gain
                '-a', '1'   # Enable TX amp
            ]
            
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Send data
            process.stdin.write(iq_data.tobytes())
            process.stdin.close()
            
            # Wait for completion
            process.wait(timeout=1)
            
        except Exception as e:
            print(f"⚠️  Trigger transmission error: {e}")
    
    def start_continuous_trigger(self, protocol: str = 'generic', interval: float = 1.0):
        """
        Start continuous triggering (for sensor activation/diagnostics)
        
        Args:
            protocol: Trigger pattern to use
            interval: Time between trigger sequences (seconds)
        """
        if self.is_transmitting:
            print("⚠️  Trigger already running")
            return
        
        self.is_transmitting = True
        self.trigger_interval = interval
        
        def trigger_loop():
            print(f"🔄 Starting continuous trigger ({protocol}, {interval}s interval)")
            while self.is_transmitting:
                self.send_trigger(protocol)
                time.sleep(self.trigger_interval)
            print("⏹️  Continuous trigger stopped")
        
        self.trigger_thread = threading.Thread(target=trigger_loop, daemon=True)
        self.trigger_thread.start()
    
    def stop_continuous_trigger(self):
        """Stop continuous triggering"""
        if self.is_transmitting:
            print("🛑 Stopping continuous trigger...")
            self.is_transmitting = False
            if self.trigger_thread:
                self.trigger_thread.join(timeout=2)
    
    def program_sensor(self, sensor_id: str, protocol: str = 'schrader'):
        """
        Simulate sensor programming sequence
        
        Args:
            sensor_id: Target sensor ID to program
            protocol: Manufacturer protocol
        """
        print(f"🔧 Programming sensor: {sensor_id}")
        print(f"   Protocol: {protocol}")
        
        # Send activation trigger
        self.send_trigger(protocol)
        
        # Wait for sensor to wake up
        time.sleep(0.5)
        
        # In a real implementation, this would:
        # 1. Send programming command via LF
        # 2. Receive confirmation via RF (314.9/433 MHz)
        # 3. Verify sensor ID
        
        print("✅ Programming sequence complete")
        print("   (Note: Actual programming requires bidirectional communication)")
    
    def get_status(self) -> dict:
        """Get trigger status"""
        return {
            'transmitting': self.is_transmitting,
            'lf_frequency': self.lf_frequency,
            'trigger_interval': self.trigger_interval,
            'available_protocols': list(self.trigger_patterns.keys())
        }


class DualModeTPMS:
    """
    Dual-mode TPMS system: Receiver + Trigger
    Combines passive reception with active sensor triggering
    """
    
    def __init__(self, hackrf_rx, hackrf_tx=None):
        """
        Initialize dual-mode system
        
        Args:
            hackrf_rx: HackRF interface for reception (314.9/433 MHz)
            hackrf_tx: Optional second HackRF for LF transmission (125 kHz)
                      If None, will time-multiplex on single HackRF
        """
        self.receiver = hackrf_rx
        self.transmitter = hackrf_tx
        self.trigger = TPMSTrigger()
        
        self.dual_hackrf = hackrf_tx is not None
        
        if self.dual_hackrf:
            print("🎯 Dual HackRF mode: Simultaneous RX + TX")
        else:
            print("🎯 Single HackRF mode: Time-multiplexed RX/TX")
    
    def trigger_and_listen(self, protocol: str = 'generic', listen_duration: float = 5.0):
        """
        Send trigger and listen for sensor responses
        
        Args:
            protocol: Trigger protocol to use
            listen_duration: How long to listen after trigger (seconds)
        """
        print(f"📡 Trigger and listen sequence...")
        
        if not self.dual_hackrf:
            # Stop receiver
            self.receiver.stop()
            time.sleep(0.5)
        
        # Send trigger
        self.trigger.send_trigger(protocol)
        
        # Wait for sensor to respond
        time.sleep(0.2)
        
        if not self.dual_hackrf:
            # Restart receiver
            self.receiver.start(self.receiver.callback)
        
        print(f"👂 Listening for {listen_duration}s...")
        time.sleep(listen_duration)
        
        print("✅ Trigger and listen complete")
    
    def scan_and_activate(self, protocols: list = None):
        """
        Scan all frequencies while periodically sending triggers
        Maximizes sensor detection probability
        
        Args:
            protocols: List of protocols to try (None = all)
        """
        if protocols is None:
            protocols = list(self.trigger.trigger_patterns.keys())
        
        print(f"🔍 Active scan mode: {len(protocols)} protocols")
        
        for protocol in protocols:
            print(f"\n📡 Trying {protocol} protocol...")
            self.trigger_and_listen(protocol, listen_duration=3.0)
