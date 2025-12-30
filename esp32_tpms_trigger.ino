/*
 * ESP32 TPMS LF Trigger
 * Generates 125 kHz LF signals to activate TPMS sensors
 * Controlled via WiFi/Bluetooth from main TPMS app
 * 
 * Hardware:
 * - ESP32 Dev Board (CP2102)
 * - LF Antenna coil (125 kHz resonant)
 * - MOSFET driver circuit
 * - Optional: Amplifier for stronger signal
 */

#include <WiFi.h>
#include <WebServer.h>
#include <ArduinoJson.h>

// Configuration
const char* ssid = "TPMS_Trigger";
const char* password = "tpms12345";

// LF Output Pin (use DAC for analog output)
#define LF_OUTPUT_PIN 25  // DAC1
#define LF_ENABLE_PIN 26  // Enable/disable trigger

// Trigger parameters
#define LF_FREQUENCY 125000  // 125 kHz
#define SAMPLE_RATE 1000000  // 1 MHz DAC sample rate

WebServer server(80);

// Trigger patterns for different manufacturers
struct TriggerPattern {
  int pulse_count;
  int pulse_width_ms;
  int pulse_interval_ms;
};

TriggerPattern patterns[] = {
  {3, 100, 50},   // Schrader
  {2, 150, 100},  // Toyota
  {4, 80, 30},    // Continental
  {1, 100, 0}     // Generic
};

const char* pattern_names[] = {"schrader", "toyota", "continental", "generic"};

bool is_triggering = false;
int current_pattern = 3;  // Default to generic
float trigger_interval = 1.0;  // seconds

// DAC waveform buffer
uint8_t sine_wave[80];  // One cycle of 125 kHz at 10 MS/s

void setup() {
  Serial.begin(115200);
  
  pinMode(LF_OUTPUT_PIN, OUTPUT);
  pinMode(LF_ENABLE_PIN, OUTPUT);
  digitalWrite(LF_ENABLE_PIN, LOW);
  
  // Generate sine wave lookup table
  generate_sine_wave();
  
  // Setup WiFi AP
  WiFi.softAP(ssid, password);
  IPAddress IP = WiFi.softAPIP();
  
  Serial.println("ESP32 TPMS Trigger Ready");
  Serial.print("AP IP address: ");
  Serial.println(IP);
  
  // Setup web server endpoints
  server.on("/", handleRoot);
  server.on("/trigger", HTTP_POST, handleTrigger);
  server.on("/start_continuous", HTTP_POST, handleStartContinuous);
  server.on("/stop_continuous", HTTP_POST, handleStopContinuous);
  server.on("/status", HTTP_GET, handleStatus);
  
  server.begin();
  Serial.println("HTTP server started");
}

void loop() {
  server.handleClient();
  
  // Continuous triggering loop
  if (is_triggering) {
    send_trigger_pattern(current_pattern);
    delay(trigger_interval * 1000);
  }
}

void generate_sine_wave() {
  // Generate one cycle of sine wave for 125 kHz
  for (int i = 0; i < 80; i++) {
    sine_wave[i] = (uint8_t)(127.5 + 127.5 * sin(2 * PI * i / 80));
  }
}

void send_lf_pulse(int duration_ms) {
  digitalWrite(LF_ENABLE_PIN, HIGH);
  
  unsigned long start_time = millis();
  unsigned long cycles = (LF_FREQUENCY * duration_ms) / 1000;
  
  // Output sine wave via DAC
  for (unsigned long c = 0; c < cycles; c++) {
    for (int i = 0; i < 80; i++) {
      dacWrite(LF_OUTPUT_PIN, sine_wave[i]);
      delayMicroseconds(1);  // ~1 MHz sample rate
    }
  }
  
  digitalWrite(LF_ENABLE_PIN, LOW);
  dacWrite(LF_OUTPUT_PIN, 0);
}

void send_trigger_pattern(int pattern_index) {
  if (pattern_index < 0 || pattern_index > 3) {
    pattern_index = 3;  // Default to generic
  }
  
  TriggerPattern pattern = patterns[pattern_index];
  
  Serial.print("Sending ");
  Serial.print(pattern_names[pattern_index]);
  Serial.println(" trigger pattern");
  
  for (int i = 0; i < pattern.pulse_count; i++) {
    send_lf_pulse(pattern.pulse_width_ms);
    
    if (i < pattern.pulse_count - 1) {
      delay(pattern.pulse_interval_ms);
    }
  }
  
  Serial.println("Trigger sent");
}

void handleRoot() {
  String html = "<html><body>";
  html += "<h1>ESP32 TPMS Trigger</h1>";
  html += "<p>Status: " + String(is_triggering ? "Triggering" : "Idle") + "</p>";
  html += "<p>Pattern: " + String(pattern_names[current_pattern]) + "</p>";
  html += "<form action='/trigger' method='POST'>";
  html += "<select name='pattern'>";
  for (int i = 0; i < 4; i++) {
    html += "<option value='" + String(i) + "'>" + String(pattern_names[i]) + "</option>";
  }
  html += "</select>";
  html += "<button type='submit'>Send Trigger</button>";
  html += "</form>";
  html += "</body></html>";
  
  server.send(200, "text/html", html);
}

void handleTrigger() {
  if (server.hasArg("pattern")) {
    int pattern = server.arg("pattern").toInt();
    send_trigger_pattern(pattern);
    server.send(200, "application/json", "{\"status\":\"success\",\"message\":\"Trigger sent\"}");
  } else {
    server.send(400, "application/json", "{\"status\":\"error\",\"message\":\"Missing pattern parameter\"}");
  }
}

void handleStartContinuous() {
  if (server.hasArg("pattern")) {
    current_pattern = server.arg("pattern").toInt();
  }
  if (server.hasArg("interval")) {
    trigger_interval = server.arg("interval").toFloat();
  }
  
  is_triggering = true;
  
  String response = "{\"status\":\"success\",\"message\":\"Continuous triggering started\",";
  response += "\"pattern\":\"" + String(pattern_names[current_pattern]) + "\",";
  response += "\"interval\":" + String(trigger_interval) + "}";
  
  server.send(200, "application/json", response);
}

void handleStopContinuous() {
  is_triggering = false;
  server.send(200, "application/json", "{\"status\":\"success\",\"message\":\"Continuous triggering stopped\"}");
}

void handleStatus() {
  String response = "{";
  response += "\"is_triggering\":" + String(is_triggering ? "true" : "false") + ",";
  response += "\"pattern\":\"" + String(pattern_names[current_pattern]) + "\",";
  response += "\"interval\":" + String(trigger_interval) + ",";
  response += "\"lf_frequency\":" + String(LF_FREQUENCY);
  response += "}";
  
  server.send(200, "application/json", response);
}
