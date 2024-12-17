import cv2
import sys
from cvzone.SerialModule import SerialObject

sys.path.insert(1, 'C://Users//mohme//Desktop//P//ML//OpenCv//Projects//HandDetection//HTmodule.py')




arduino = SerialObject("COM5")

vid = cv2.VideoCapture(0)

while True:
    success, img = vid.read()

    imgFace, boundingBoxs = ftm.findFaces(img)
    imgHand = htm.findHands(img)
    lmLists = htm.findHandLmPos(imgHand)


    if boundingBoxs and lmLists != 0:
        if len(lmLists) > 1:
            arduino.sendData([1, 3])
        else:
            arduino.sendData([1, 1])

    elif boundingBoxs:
        arduino.sendData([1, 0])

    elif lmLists != 0:
        if len(lmLists) > 1:
            arduino.sendData([0, 3])
        else:
            arduino.sendData([0, 1])

    else:
        arduino.sendData([0, 0])


    cv2.imshow("Image", img)
    cv2.waitKey(1)
