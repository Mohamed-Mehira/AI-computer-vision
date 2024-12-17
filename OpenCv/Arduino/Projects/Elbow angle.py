import cv2
import numpy as np
from cvzone.SerialModule import SerialObject

arduino = SerialObject("COM5")


vid = cv2.VideoCapture(0)
bar = 650
per = 0
count = 0
dir = 0


while True:
    success, img = vid.read()
    img = cv2.resize(img, (1280, 720))

    img, lmList = pm.findPose(img, False)

    if len(lmList) != 0:

        # Right Arm
        angle = pm.findAngle(img, lmList, 12, 14, 16)

        # Left Arm
        # angle = pm.findAngle(img, lmList, 11, 13, 15, False)

        motorAngle = np.interp(angle, [45, 140], [0, 180])

        arduino.sendData([motorAngle])


    cv2.imshow("Image", img)
    cv2.waitKey(1)
