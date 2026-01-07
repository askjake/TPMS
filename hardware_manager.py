"""
Unified Hardware Manager
Automatically detects and manages HackRF or RTL-SDR hardware
"""

import logging
from typing import Optional, Literal
from enum import Enum

logger = logging.getLogger(__name__)

HardwareType = Literal["hackrf", "rtlsdr", "simulation"]

"""
Unified Hardware Manager
Automatically detects and manages HackRF or RTL-SDR hardware
"""

import logging
from typing import Optional, Literal
from enum import Enum

logger = logging.getLogger(__name__)

HardwareType = Literal["hackrf", "rtlsdr", "simulation"]


class HardwareManager:
    """Manages hardware detection and provides unified interface"""

    def __init__(self, preferred: Optional[HardwareType] = None):
        self.hardware_type: Optional[HardwareType] = None
        self.interface = None
        self.preferred = preferred

        self._detect_and_initialize()

    def _detect_and_initialize(self):
        """Detect available hardware and initialize"""

        # If user explicitly wants simulation, give it to them
        if self.preferred == "simulation":
            logger.info("Simulation mode explicitly requested")
            if self._try_init("simulation"):
                return
            logger.error("Failed to initialize simulation mode")
            return

        # If user specified a real hardware preference, try that first
        if self.preferred in ("hackrf", "rtlsdr"):
            if self._try_init(self.preferred):
                return
            logger.warning(f"Preferred hardware '{self.preferred}' not available, trying others...")

        # Auto-detect: Try HackRF first
        if self._try_init("hackrf"):
            return

        # Auto-detect: Try RTL-SDR second
        if self._try_init("rtlsdr"):
            return

        # No hardware found - DO NOT fall back to simulation
        logger.error("❌ No SDR hardware detected. Please connect HackRF or RTL-SDR.")
        logger.error("If you want to use simulation mode, explicitly select it from the UI.")
        self.hardware_type = None
        self.interface = None

    def _try_init(self, hw_type: HardwareType, device_index: int = 0) -> bool:
        """Try to initialize specific hardware type"""
        try:
            if hw_type == "hackrf":
                import hackrf_interface
                if hackrf_interface.HACKRF_AVAILABLE:
                    self.interface = hackrf_interface.HackRFInterface()
                    if self.interface.device:
                        self.hardware_type = "hackrf"
                        logger.info("✅ HackRF hardware initialized")
                        return True
                    else:
                        logger.debug("HackRF library available but no device found")
                else:
                    logger.debug("HackRF library not available")

            elif hw_type == "rtlsdr":
                import rtl_interface
                if rtl_interface.RTLSDR_AVAILABLE:
                    # Support multiple RTL-SDR devices by index
                    self.interface = rtl_interface.RTLSDRInterface(device_index=device_index)
                    if self.interface.device:
                        self.hardware_type = "rtlsdr"
                        logger.info(f"✅ RTL-SDR hardware initialized (device {device_index})")
                        return True
                    else:
                        logger.debug(f"RTL-SDR library available but no device found at index {device_index}")
                else:
                    logger.debug("RTL-SDR library not available")

            elif hw_type == "simulation":
                # Try HackRF simulation first, then RTL-SDR
                try:
                    import hackrf_interface
                    self.interface = hackrf_interface.SimulatedHackRF()
                    self.hardware_type = "simulation"
                    logger.info("✅ Simulation mode (HackRF)")
                    return True
                except Exception:
                    import rtl_interface
                    self.interface = rtl_interface.SimulatedRTLSDR()
                    self.hardware_type = "simulation"
                    logger.info("✅ Simulation mode (RTL-SDR)")
                    return True

        except Exception as e:
            logger.error(f"Failed to initialize {hw_type}: {e}")

        return False

    def get_interface(self):
        """Get the active hardware interface"""
        return self.interface

    def get_hardware_type(self) -> Optional[HardwareType]:
        """Get the active hardware type"""
        return self.hardware_type

    def get_hardware_info(self) -> dict:
        """Get information about active hardware"""
        if not self.interface:
            return {"type": None, "available": False}

        status = self.interface.get_status()
        return {
            "type": self.hardware_type,
            "available": True,
            "status": status
        }



def create_hardware_interface(preferred: Optional[HardwareType] = None):
    """
    Factory function to create appropriate hardware interface

    Args:
        preferred: Preferred hardware type ("hackrf", "rtlsdr", or None for auto-detect)

    Returns:
        Hardware interface object
    """
    manager = HardwareManager(preferred=preferred)
    return manager.get_interface()



