
#define ACTUATOR_PIN   9    // PWM output to motor driver / servo
#define STATUS_LED     13   // On-board LED

#define SERIAL_BAUD    115200
#define CMD_TIMEOUT_MS 500  // If no command received in this time → idle

unsigned long lastCmdTime = 0;
int  currentPWM  = 0;
int  lastDepthMm = 0;
int  lastDistCm  = 0;
bool ledState    = false;

void setup() {
  Serial.begin(SERIAL_BAUD);
  pinMode(ACTUATOR_PIN, OUTPUT);
  pinMode(STATUS_LED,   OUTPUT);
  analogWrite(ACTUATOR_PIN, 0);
  digitalWrite(STATUS_LED, LOW);
  Serial.println("Adaptive Suspension Ready");
}


void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();

    if (line.length() > 1 && line.charAt(0) == 'S') {
      int idx1 = line.indexOf(',');
      int idx2 = line.indexOf(',', idx1 + 1);

      if (idx1 > 0 && idx2 > idx1) {
        int pwm      = line.substring(1,       idx1).toInt();
        int depthMm  = line.substring(idx1 + 1, idx2).toInt();
        int distCm   = line.substring(idx2 + 1).toInt();

        pwm = constrain(pwm, 0, 255);

        currentPWM  = pwm;
        lastDepthMm = depthMm;
        lastDistCm  = distCm;
        lastCmdTime = millis();

        analogWrite(ACTUATOR_PIN, pwm);
        digitalWrite(STATUS_LED, pwm > 0 ? HIGH : LOW);

        Serial.print("ACK PWM=");
        Serial.print(pwm);
        Serial.print(" DEPTH=");
        Serial.print(depthMm);
        Serial.print("mm DIST=");
        Serial.print(distCm);
        Serial.println("cm");
      }
    }
  }

  if (millis() - lastCmdTime > CMD_TIMEOUT_MS && currentPWM != 0) {
    currentPWM = 0;
    analogWrite(ACTUATOR_PIN, 0);
    digitalWrite(STATUS_LED, LOW);
    Serial.println("TIMEOUT – actuator idle");
  }
}