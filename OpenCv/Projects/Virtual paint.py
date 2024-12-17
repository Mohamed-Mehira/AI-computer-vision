import cv2
import numpy as np


vid = cv2.VideoCapture(0)

frameWidth = 640
frameHeight = 480
vid.set(3, frameWidth)
vid.set(4, frameHeight)
vid.set(10, 100)


HSVcolors = [[5, 107, 0, 19, 255, 255],
             [133, 56, 0, 159, 156, 255],
             [57, 76, 0, 100, 255, 255],
             [90, 48, 0, 118, 255, 255]]

BGRcolors = [[51, 153, 255],
             [255, 0, 255],
             [0, 255, 0],
             [255, 0, 0]]


myPoints = []     # [x, y, colorCount]


def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x, y, w, h = 0, 0, 0, 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)

    return x + w // 2, y



def findColor(img, HSVcolors, BGRcolors):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    colorCount = 0
    newPoints = []

    for color in HSVcolors:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHSV, lower, upper)
        x, y = getContours(mask)
        cv2.circle(imgResult, (x, y), 10, BGRcolors[colorCount], cv2.FILLED)
        if x != 0 and y != 0:
            newPoints.append([x, y, colorCount])
        colorCount += 1

    return newPoints



def drawOnCanvas(myPoints, BGRcolors):
    for point in myPoints:
        cv2.circle(imgResult, (point[0], point[1]), 10, BGRcolors[point[2]], cv2.FILLED)



while True:
    success, img = vid.read()
    imgResult = img.copy()
    newPoints = findColor(img, HSVcolors, BGRcolors)

    if len(newPoints) != 0:
        for newP in newPoints:
            myPoints.append(newP)

    drawOnCanvas(myPoints, BGRcolors)


    cv2.imshow("Result", imgResult)

    '''''
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(1) & 0xFF == ord('r'):
        myPoints = []
    '''''


vid.release()
cv2.destroyAllWindows()