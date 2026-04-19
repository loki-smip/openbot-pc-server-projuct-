/*
 * ESP8266_Car.ino
 * 
 * Standalone firmware for controlling an L298N motor driver and a Servo 
 * over WebSockets via an ESP8266 (e.g., NodeMCU or Wemos D1 Mini).
 * 
 * Dependencies:
 *   - ArduinoWebsockets (Links2004)
 *   - ArduinoJson
 */

#include <ESP8266WiFi.h>
#include <ESP8266mDNS.h>
#include <WiFiUdp.h>
#include <ArduinoOTA.h>
#include <WebSocketsServer.h>
#include <ArduinoJson.h>
#include <Servo.h>

// ---------------------------------------------------------
// CONFIGURATION
// ---------------------------------------------------------

const char* ssid = "your_wifi_name";
const char* password = "your_wifi_password";

// Server Port
const int WS_PORT = 81;

// Motor Pins (NodeMCU D-pins)
// Left Motor
#define PIN_IN1 D5  // GPIO 14
#define PIN_IN2 D6  // GPIO 12

// Right Motor
#define PIN_IN3 D7  // GPIO 13
#define PIN_IN4 D8  // GPIO 15

// Servo Pin
#define PIN_SERVO D1 // GPIO 5
Servo camServo;

// ---------------------------------------------------------
// GLOBALS
// ---------------------------------------------------------

WebSocketsServer webSocket = WebSocketsServer(WS_PORT);

// ---------------------------------------------------------
// MOTOR FUNCTIONS
// ---------------------------------------------------------

void setMotorSpeed(int in1, int in2, int speed) {
    // Speed is -100 to 100
    // L298N ENA/ENB are assumed clamped to 5V (or high),
    // so we PWM the INx pins directly. ESP8266 analogWrite is 0-1023.
    
    if (speed > 100) speed = 100;
    if (speed < -100) speed = -100;

    int pwmVal = map(abs(speed), 0, 100, 0, 1023);

    if (speed == 0) {
        analogWrite(in1, 0);
        analogWrite(in2, 0);
    } 
    else if (speed > 0) {
        analogWrite(in1, pwmVal);
        analogWrite(in2, 0);
    } 
    else {
        analogWrite(in1, 0);
        analogWrite(in2, pwmVal);
    }
}

// ---------------------------------------------------------
// WEBSOCKET HANDLER
// ---------------------------------------------------------

void webSocketEvent(uint8_t num, WStype_t type, uint8_t *payload, size_t length) {
    switch (type) {
        case WStype_DISCONNECTED:
            Serial.printf("[%u] Disconnected!\n", num);
            // Stop car on disconnect for safety
            setMotorSpeed(PIN_IN1, PIN_IN2, 0);
            setMotorSpeed(PIN_IN3, PIN_IN4, 0);
            break;

        case WStype_CONNECTED:
            {
                IPAddress ip = webSocket.remoteIP(num);
                Serial.printf("[%u] Connected from %d.%d.%d.%d\n", num, ip[0], ip[1], ip[2], ip[3]);
            }
            break;

        case WStype_TEXT:
            {
                // Parse JSON {"left": X, "right": Y, "servo": Z}
                StaticJsonDocument<200> doc;
                DeserializationError error = deserializeJson(doc, payload, length);

                if (error) {
                    Serial.println("JSON Parse Error");
                    return;
                }

                if (doc.containsKey("left")) {
                    int leftSpeed = doc["left"];
                    setMotorSpeed(PIN_IN1, PIN_IN2, leftSpeed);
                }

                if (doc.containsKey("right")) {
                    int rightSpeed = doc["right"];
                    setMotorSpeed(PIN_IN3, PIN_IN4, rightSpeed);
                }

                if (doc.containsKey("servo")) {
                    int angle = doc["servo"];
                    if (angle < 0) angle = 0;
                    if (angle > 180) angle = 180;
                    camServo.write(angle);
                }
            }
            break;

        case WStype_BIN:
        case WStype_ERROR:
        case WStype_FRAGMENT_TEXT_START:
        case WStype_FRAGMENT_BIN_START:
        case WStype_FRAGMENT:
        case WStype_FRAGMENT_FIN:
            break;
    }
}

// ---------------------------------------------------------
// SETUP & LOOP
// ---------------------------------------------------------

void setup() {
    Serial.begin(115200);
    delay(500);

    // Init Motor Pins
    pinMode(PIN_IN1, OUTPUT);
    pinMode(PIN_IN2, OUTPUT);
    pinMode(PIN_IN3, OUTPUT);
    pinMode(PIN_IN4, OUTPUT);

    // Safety stop
    setMotorSpeed(PIN_IN1, PIN_IN2, 0);
    setMotorSpeed(PIN_IN3, PIN_IN4, 0);

    // Init Servo
    camServo.attach(PIN_SERVO);
    camServo.write(90);

    // Init Wi-Fi
    Serial.println();
    Serial.printf("Connecting to %s ...", ssid);
    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid, password);

    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }

    Serial.println("\nWiFi connected!");
    Serial.print("ESP8266 IP address: ");
    Serial.println(WiFi.localIP());

    // Init OTA
    ArduinoOTA.setHostname("esp8266-car-motor");
    ArduinoOTA.onStart([]() {
        Serial.println("Start updating OTA");
    });
    ArduinoOTA.onEnd([]() {
        Serial.println("\nEnd OTA");
    });
    ArduinoOTA.onProgress([](unsigned int progress, unsigned int total) {
        Serial.printf("Progress: %u%%\r", (progress / (total / 100)));
    });
    ArduinoOTA.onError([](ota_error_t error) {
        Serial.printf("Error[%u]: ", error);
        if (error == OTA_AUTH_ERROR) Serial.println("Auth Failed");
        else if (error == OTA_BEGIN_ERROR) Serial.println("Begin Failed");
        else if (error == OTA_CONNECT_ERROR) Serial.println("Connect Failed");
        else if (error == OTA_RECEIVE_ERROR) Serial.println("Receive Failed");
        else if (error == OTA_END_ERROR) Serial.println("End Failed");
    });
    ArduinoOTA.begin();

    // Init WebSocket Server
    webSocket.begin();
    webSocket.onEvent(webSocketEvent);
    webSocket.disableHeartbeat(); // Keeps Python websockets v14.x happy
    
    Serial.println("WebSocket Server started on port 81");
}

void loop() {
    ArduinoOTA.handle();
    webSocket.loop();
}
