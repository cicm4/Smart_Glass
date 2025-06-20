/*  esp32_cam_ap_echo.ino  ─ Soft-AP + camera stream + simple TCP echo  */

#include "esp_camera.h"
#include <WiFi.h>


#define CAMERA_MODEL_AI_THINKER

#include "camera_pins.h"

// ---------- Soft-AP credentials ----------
const char *AP_SSID = "ESP32_CAM_AP";
const char *AP_PASS = "12345678";            // ≥8 chars

// ---------- Ports ----------
const uint16_t ECHO_PORT = 3333;             // avoid clash with camera HTTP (80)

// ---------- Globals ----------
WiFiServer echoServer(ECHO_PORT);

// Forward declaration (implemented by ESP32-Cam example)
void startCameraServer();
void setupLedFlash(int pin);

// =========================================================
// SETUP
// =========================================================
void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println("\nBooting…");

  // ---- Camera configuration (unchanged from example) ----
  camera_config_t config;
  config.ledc_channel   = LEDC_CHANNEL_0;
  config.ledc_timer     = LEDC_TIMER_0;
  config.pin_d0         = Y2_GPIO_NUM;  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2         = Y4_GPIO_NUM;  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4         = Y6_GPIO_NUM;  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6         = Y8_GPIO_NUM;  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk       = XCLK_GPIO_NUM;
  config.pin_pclk       = PCLK_GPIO_NUM;
  config.pin_vsync      = VSYNC_GPIO_NUM;
  config.pin_href       = HREF_GPIO_NUM;
  config.pin_sccb_sda   = SIOD_GPIO_NUM;
  config.pin_sccb_scl   = SIOC_GPIO_NUM;
  config.pin_pwdn       = PWDN_GPIO_NUM;
  config.pin_reset      = RESET_GPIO_NUM;
  config.xclk_freq_hz   = 20000000;
  config.frame_size     = FRAMESIZE_UXGA;
  config.pixel_format   = PIXFORMAT_JPEG;      // streaming
  config.grab_mode      = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location    = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality   = 12;
  config.fb_count       = 1;

  if (config.pixel_format == PIXFORMAT_JPEG && psramFound()) {
    config.jpeg_quality = 10;
    config.fb_count     = 2;
    config.grab_mode    = CAMERA_GRAB_LATEST;
  } else if (!psramFound()) {
    config.frame_size   = FRAMESIZE_SVGA;
    config.fb_location  = CAMERA_FB_IN_DRAM;
  }

#if defined(CAMERA_MODEL_ESP_EYE)
  pinMode(13, INPUT_PULLUP);      // button pins on ESP-EYE
  pinMode(14, INPUT_PULLUP);
#endif

  // ---- Initialise camera ----
  if (esp_camera_init(&config) != ESP_OK) {
    Serial.println("Camera init failed");   while (true) delay(1000);
  }

  // ---- Start Soft-AP ----
  WiFi.mode(WIFI_AP);
  WiFi.softAP(AP_SSID, AP_PASS);
  WiFi.setSleep(false);                       // keep radio awake
  IPAddress ip = WiFi.softAPIP();             // default 192.168.4.1
  Serial.printf("Soft-AP started  ▶  %s\n", ip.toString().c_str());

  // ---- Start servers ----
  echoServer.begin();
  Serial.printf("Echo server on   ▶  tcp://%s:%u\n", ip.toString().c_str(),
                ECHO_PORT);

  startCameraServer();                        // HTTP stream on port 80
  Serial.printf("Camera stream on ▶  http://%s\n", ip.toString().c_str());
}

// =========================================================
// LOOP – handle echo traffic (camera runs in its own task)
// =========================================================
void loop() {
  WiFiClient client = echoServer.accept();      // non-blocking
  if (!client) return;

  Serial.println("TCP client connected");
  while (client.connected()) {
    // ---------- laptop → ESP32 ----------
    if (client.available()) {
      String line = client.readStringUntil('\n');
      Serial.print("rx: ");  Serial.println(line);
      client.print("echo: " + line + "\n");
    }
    // ---------- USB serial → laptop ----------
    if (Serial.available()) {
      client.write(Serial.read());
    }
    // yield to avoid watchdog reset
    delay(2);
  }
  Serial.println("Client disconnected");
}
