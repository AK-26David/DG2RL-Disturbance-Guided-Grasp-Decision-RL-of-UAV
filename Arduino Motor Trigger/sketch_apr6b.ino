#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

#define RIGHT 500
#define LEFT  200

void setup() {
  Serial.begin(9600);
  pwm.begin();
  pwm.setPWMFreq(50);
}

void loop() {
  if (Serial.available()) {
    char cmd = Serial.read();

    if (cmd == '1') {
      pwm.setPWM(0, 0, RIGHT);  // move servo right
    } 
    else if (cmd == '0') {
      pwm.setPWM(0, 0, LEFT);   // move servo left
    }
  }
}