import cv2.cv2 as cv2
import numpy as np
from Tutorials.stackedImages import stackImages

####################
frameWidth = 640
frameHeight = 480
####################


def nothing(a):
    pass


def createTrackBars(initialTrackBarVals):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 480, 240)
    cv2.createTrackbar("X Top", "Trackbars", initialTrackBarVals[0], frameWidth // 2, nothing)
    cv2.createTrackbar("Y Top", "Trackbars", initialTrackBarVals[1], frameHeight, nothing)
    cv2.createTrackbar("X Bottom", "Trackbars", initialTrackBarVals[2], frameWidth // 2, nothing)
    cv2.createTrackbar("Y Bottom", "Trackbars", initialTrackBarVals[3], frameHeight, nothing)


def getPoints():
    widthTop = cv2.getTrackbarPos("X Top", "Trackbars")
    heightTop = cv2.getTrackbarPos("Y Top", "Trackbars")
    widthBottom = cv2.getTrackbarPos("X Bottom", "Trackbars")
    heightBottom = cv2.getTrackbarPos("Y Bottom", "Trackbars")
    points = np.float32([(widthTop, heightTop), (frameWidth - widthTop, heightTop),
                        (widthBottom, heightBottom), (frameWidth - widthBottom, heightBottom)])
    return points


def drawPoints(img, points):
    for x in range(0, 4):
        cv2.circle(img, (int(points[x][0]), int(points[x][1])), 3, (0, 0, 255), 7)


########################################################################################################################


def thresholding(img):

    imgHsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    # lowerWhite = np.array([0, 0, 0])
    # upperWhite = np.array([179, 62, 255])
    lowerWhite = np.array([85, 0, 0])
    upperWhite = np.array([179, 160, 255])
    maskedWhite = cv2.inRange(imgHsv, lowerWhite, upperWhite)

    return maskedWhite


def warpImg(img, points):

    h, w = img.shape

    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    imgOutput = cv2.warpPerspective(img, matrix, (w, h))

    return imgOutput


def getHistogram(img, minVal=0.1, region=1.0, display=False):

    h, w = img.shape

    if region == 1:
        histValues = np.sum(img, axis=0)
    else:
        histValues = np.sum(img[h - int(h * region):, :], axis=0)

    # First method
    maxValue = np.max(histValues)
    minValAllowed = minVal * maxValue

    indexArray = np.where(histValues >= minValAllowed)  # ALL INDICES WITH MIN VALUE OR ABOVE
    basePoint = int(np.average(indexArray))  # AVERAGE ALL MAX INDICES VALUES

    # Second method
    midPoint = np.int(histValues.shape[0] / 2)
    leftx_base = np.argmax(histValues[:midPoint])
    rightx_base = np.argmax(histValues[midPoint:]) + midPoint
    '''''
    we're supposed to create windows (boxes) around these points and put the (x, y) values of the pixels inside these windows  
    in lists, then we continue to stack and move these windows up while moving them slightly depending on the average values from the 
    corresponding lists, then we run a polyfit on all the points from the windows (from the big list of x and y values of all the pixels).
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    '''''

    if display:
        imgHist = np.zeros((h, w, 3), np.uint8)
        for column, intensity in enumerate(histValues):
            if intensity > minValAllowed:
                color = (255, 90, 20)
            else:
                color = (0, 0, 255)
            cv2.line(imgHist, (column, h), (column, h - (intensity // 255)), color, 1)
            cv2.circle(imgHist, (basePoint, h), 20, (0, 255, 255), cv2.FILLED)

        return basePoint, imgHist
    return basePoint