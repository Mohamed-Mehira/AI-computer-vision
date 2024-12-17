#include <cvzone.h>

SerialData serialData(3,1);
int valsRec[3];

int red = 8;
int blue = 9;
int green = 10;

void setup()
{
  serialData.begin();
  pinMode(red, OUTPUT);
  pinMode(green, OUTPUT);
  pinMode(blue, OUTPUT);
}

void loop()
{
    serialData.Get(valsRec);
    digitalWrite(red, valsRec[0]);
    digitalWrite(green, valsRec[1]);
    digitalWrite(blue,valsRec[2]);
}
