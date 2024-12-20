import cv2
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.SerialModule import SerialObject

vid = cv2.VideoCapture(1)
detector = FaceDetector()
arduino = SerialObject()

while True:
    success, img = vid.read()
    img, boundingBoxs = detector.findFaces(img)

    if boundingBoxs:
        # (R,G,B)  "0" is on "1" is off
        arduino.sendData([1, 0, 1])
    else:
        arduino.sendData([0, 1, 1])

    cv2.imshow("Image", img)
    cv2.waitKey(1)
