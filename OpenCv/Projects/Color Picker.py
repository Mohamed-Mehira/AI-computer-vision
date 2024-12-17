import cv2
import numpy as np
from Tutorials.stackedImages import stackImages


def empty(a):
    pass

cv2.namedWindow("HSV")
cv2.resizeWindow("HSV", 640, 240)
cv2.createTrackbar("HUE Min", "HSV", 0, 179, empty)
cv2.createTrackbar("HUE Max", "HSV", 179, 179, empty)
cv2.createTrackbar("SAT Min", "HSV", 0, 255, empty)
cv2.createTrackbar("SAT Max", "HSV", 255, 255, empty)
cv2.createTrackbar("VALUE Min", "HSV", 0, 255, empty)
cv2.createTrackbar("VALUE Max", "HSV", 255, 255, empty)


# path = '../Resources/vid1.mp4'
vid = cv2.VideoCapture(0)
address = "http://192.168.1.2:4747/video"
vid.open(address)

frameWidth = 640
frameHeight = 480
vid.set(3, frameWidth)
vid.set(4, frameHeight)

frameCounter = 0

while True:

    frameCounter += 1
    if vid.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
        vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frameCounter = 0

    success, img = vid.read()
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("HUE Min", "HSV")
    h_max = cv2.getTrackbarPos("HUE Max", "HSV")
    s_min = cv2.getTrackbarPos("SAT Min", "HSV")
    s_max = cv2.getTrackbarPos("SAT Max", "HSV")
    v_min = cv2.getTrackbarPos("VALUE Min", "HSV")
    v_max = cv2.getTrackbarPos("VALUE Max", "HSV")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHsv, lower, upper)
    imgResult = cv2.bitwise_and(img, img, mask=mask)
    # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # hStack = np.hstack([img, mask, imgResult])
    imgStack = stackImages(0.4, ([img, mask, imgResult]))
    cv2.imshow('Stacked Images', imgStack)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


vid.release()
cv2.destroyAllWindows()