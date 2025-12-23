import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass
from scipy import signal as scipy_signal
import struct


@dataclass
class TPMSSignal:
    tpms_id: str
    timestamp: float
    frequency: float
    signal_strength: float
    snr: float
    pressure_psi: Optional[float] = None
    temperature_c: Optional[float] = None
    battery_low: bool = False
    protocol: str = "unknown"
    raw_data: bytes = b''


class TPMSDecoder:
    """Decode TPMS signals from IQ samples"""

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.protocols = {
            'toyota': self._decode_toyota,
            'schrader': self._decode_schrader,
            'continental': self._decode_continental,
            'generic': self._decode_generic
        }

    def process_samples(self, iq_samples: np.ndarray, frequency: float) -> List[TPMSSignal]:
        """Process IQ samples and extract TPMS signals"""
        signals = []

        # Demodulate FSK/ASK
        demod_data = self._demodulate(iq_samples)

        # Find preambles and decode packets
        packets = self._find_packets(demod_data)

        for packet in packets:
            # Try each protocol
            for protocol_name, decoder in self.protocols.items():
                try:
                    tpms_signal = decoder(packet, frequency)
                    if tpms_signal:
                        signals.append(tpms_signal)
                        break
                except:
                    continue

        return signals

    def _demodulate(self, iq_samples: np.ndarray) -> np.ndarray:
        """Demodulate FSK/ASK signal"""
        # Calculate magnitude
        magnitude = np.abs(iq_samples)

        # Calculate instantaneous frequency
        phase = np.unwrap(np.angle(iq_samples))
        inst_freq = np.diff(phase) / (2 * np.pi) * self.sample_rate

        # Threshold for bit detection
        threshold = np.mean(magnitude)
        bits = (magnitude[:-1] > threshold).astype(int)

        return bits

    def _find_packets(self, bits: np.ndarray) -> List[bytes]:
        """Find packet preambles and extract data"""
        packets = []

        # Common TPMS preambles (varies by protocol)
        preambles = [
            [1, 0, 1, 0, 1, 0, 1, 0],  # Alternating pattern
            [1, 1, 1, 1, 0, 0, 0, 0],  # Block pattern
        ]

        for preamble in preambles:
            preamble_arr = np.array(preamble)

            # Find preamble using correlation
            correlation = np.correlate(bits, preamble_arr, mode='valid')
            peaks = np.where(correlation > len(preamble) * 0.8)[0]

            for peak in peaks:
                # Extract packet (typically 64-128 bits)
                packet_start = peak + len(preamble)
                packet_end = packet_start + 128

                if packet_end < len(bits):
                    packet_bits = bits[packet_start:packet_end]
                    packet_bytes = self._bits_to_bytes(packet_bits)
                    packets.append(packet_bytes)

        return packets

    def _bits_to_bytes(self, bits: np.ndarray) -> bytes:
        """Convert bit array to bytes"""
        # Pad to multiple of 8
        padding = (8 - len(bits) % 8) % 8
        bits = np.concatenate([bits, np.zeros(padding, dtype=int)])

        # Convert to bytes
        byte_array = []
        for i in range(0, len(bits), 8):
            byte_bits = bits[i:i + 8]
            byte_val = sum(bit << (7 - j) for j, bit in enumerate(byte_bits))
            byte_array.append(byte_val)

        return bytes(byte_array)

    def _decode_toyota(self, packet: bytes, frequency: float) -> Optional[TPMSSignal]:
        """Decode Toyota TPMS protocol"""
        if len(packet) < 8:
            return None

        # Toyota format (simplified):
        # Bytes 0-3: ID
        # Byte 4: Pressure
        # Byte 5: Temperature
        # Byte 6: Flags
        # Byte 7: Checksum

        tpms_id = packet[0:4].hex().upper()
        pressure_raw = packet[4]
        temp_raw = packet[5]
        flags = packet[6]

        # Convert to physical units
        pressure_psi = pressure_raw * 0.25  # Example conversion
        temperature_c = temp_raw - 40  # Example conversion
        battery_low = bool(flags & 0x80)

        return TPMSSignal(
            tpms_id=tpms_id,
            timestamp=time.time(),
            frequency=frequency,
            signal_strength=0,  # Will be filled by caller
            snr=0,  # Will be filled by caller
            pressure_psi=pressure_psi,
            temperature_c=temperature_c,
            battery_low=battery_low,
            protocol='toyota',
            raw_data=packet
        )

    def _decode_schrader(self, packet: bytes, frequency: float) -> Optional[TPMSSignal]:
        """Decode Schrader TPMS protocol"""
        if len(packet) < 10:
            return None

        # Schrader format varies, this is a simplified version
        tpms_id = packet[0:4].hex().upper()

        # Extract pressure and temperature (format varies)
        pressure_psi = struct.unpack('>H', packet[4:6])[0] * 0.01
        temperature_c = packet[6] - 50

        return TPMSSignal(
            tpms_id=tpms_id,
            timestamp=time.time(),
            frequency=frequency,
            signal_strength=0,
            snr=0,
            pressure_psi=pressure_psi,
            temperature_c=temperature_c,
            protocol='schrader',
            raw_data=packet
        )

    def _decode_continental(self, packet: bytes, frequency: float) -> Optional[TPMSSignal]:
        """Decode Continental TPMS protocol"""
        # Similar structure to above
        if len(packet) < 8:
            return None

        tpms_id = packet[0:4].hex().upper()

        return TPMSSignal(
            tpms_id=tpms_id,
            timestamp=time.time(),
            frequency=frequency,
            signal_strength=0,
            snr=0,
            protocol='continental',
            raw_data=packet
        )

    def _decode_generic(self, packet: bytes, frequency: float) -> Optional[TPMSSignal]:
        """Generic decoder for unknown protocols"""
        if len(packet) < 4:
            return None

        # Just extract ID
        tpms_id = packet[0:4].hex().upper()

        return TPMSSignal(
            tpms_id=tpms_id,
            timestamp=time.time(),
            frequency=frequency,
            signal_strength=0,
            snr=0,
            protocol='generic',
            raw_data=packet
        )


import time
