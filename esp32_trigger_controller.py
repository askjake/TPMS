"""
ESP32 TPMS Trigger Controller
Communicates with ESP32 via WiFi to control LF triggering
"""

import requests
import time
from typing import Optional, Dict
import threading

class ESP32TriggerController:
    """Control ESP32-based LF trigger via WiFi"""
    
    def __init__(self, esp32_ip: str = "192.168.4.1"):
        """
        Initialize ESP32 controller
        
        Args:
            esp32_ip: IP address of ESP32 (default is AP mode IP)
        """
        self.esp32_ip = esp32_ip
        self.base_url = f"http://{esp32_ip}"
        self.connected = False
        self.is_triggering = False
        
        # Try to connect
        self.check_connection()
    
    def check_connection(self) -> bool:
        """Check if ESP32 is reachable"""
        try:
            response = requests.get(f"{self.base_url}/status", timeout=2)
            if response.status_code == 200:
                self.connected = True
                print(f"✅ Connected to ESP32 at {self.esp32_ip}")
                return True
        except Exception as e:
            self.connected = False
            print(f"❌ Cannot connect to ESP32 at {self.esp32_ip}: {e}")
        
        return False
    
    def send_trigger(self, protocol: str = "generic") -> bool:
        """
        Send single trigger pulse
        
        Args:
            protocol: Trigger pattern ('schrader', 'toyota', 'continental', 'generic')
        
        Returns:
            True if successful
        """
        if not self.connected:
            print("⚠️  Not connected to ESP32")
            return False
        
        pattern_map = {
            'schrader': 0,
            'toyota': 1,
            'continental': 2,
            'generic': 3
        }
        
        pattern_id = pattern_map.get(protocol.lower(), 3)
        
        try:
            response = requests.post(
                f"{self.base_url}/trigger",
                data={'pattern': pattern_id},
                timeout=5
            )
            
            if response.status_code == 200:
                print(f"📡 Trigger sent: {protocol}")
                return True
            else:
                print(f"⚠️  Trigger failed: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Trigger error: {e}")
            return False
    
    def start_continuous(self, protocol: str = "generic", interval: float = 1.0) -> bool:
        """
        Start continuous triggering on ESP32
        
        Args:
            protocol: Trigger pattern
            interval: Time between triggers (seconds)
        
        Returns:
            True if successful
        """
        if not self.connected:
            print("⚠️  Not connected to ESP32")
            return False
        
        pattern_map = {
            'schrader': 0,
            'toyota': 1,
            'continental': 2,
            'generic': 3
        }
        
        pattern_id = pattern_map.get(protocol.lower(), 3)
        
        try:
            response = requests.post(
                f"{self.base_url}/start_continuous",
                data={
                    'pattern': pattern_id,
                    'interval': interval
                },
                timeout=5
            )
            
            if response.status_code == 200:
                self.is_triggering = True
                print(f"🔄 Continuous triggering started: {protocol} @ {interval}s")
                return True
            else:
                print(f"⚠️  Start failed: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Start error: {e}")
            return False
    
    def stop_continuous(self) -> bool:
        """Stop continuous triggering"""
        if not self.connected:
            return False
        
        try:
            response = requests.post(
                f"{self.base_url}/stop_continuous",
                timeout=5
            )
            
            if response.status_code == 200:
                self.is_triggering = False
                print("⏹️  Continuous triggering stopped")
                return True
            else:
                print(f"⚠️  Stop failed: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Stop error: {e}")
            return False
    
    def get_status(self) -> Optional[Dict]:
        """Get ESP32 trigger status"""
        if not self.connected:
            return None
        
        try:
            response = requests.get(f"{self.base_url}/status", timeout=2)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"⚠️  Status error: {e}")
        
        return None
