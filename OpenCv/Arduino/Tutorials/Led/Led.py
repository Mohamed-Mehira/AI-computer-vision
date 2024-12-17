from cvzone.SerialModule import SerialObject
from time import sleep
import cv2


arduino = SerialObject()

imgLedOn = cv2.imread("../Resources/Pin13On.jpg")
imgLedOff = cv2.imread("../Resources/Pin13Off.jpg")


while True:

    arduino.sendData([1])
    cv2.imshow("Image", imgLedOn)
    cv2.waitKey(1000)    # sleep(1)

    arduino.sendData([0])
    cv2.imshow("Image", imgLedOff)
    cv2.waitKey(1000)
