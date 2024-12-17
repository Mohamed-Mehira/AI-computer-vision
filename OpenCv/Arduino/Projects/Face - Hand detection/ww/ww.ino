#include <cvzone.h>
#include <Servo.h>

Servo motor;
int led = 5;
int angle;

SerialData serialData(2, 1);
int valsRec[2];


void setup()
{
  serialData.begin();
  pinMode(led, OUTPUT);
  motor.attach(9);
}

void loop()
{
    serialData.Get(valsRec);
    digitalWrite(led, valsRec[0]);
    angle = 50*valsRec[1];
    motor.write(angle);
}
