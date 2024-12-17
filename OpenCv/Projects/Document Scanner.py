import cv2
import numpy as np
from Tutorials.stackedImages import stackImages



frameWidth = 640
frameHeight = 480

vid = cv2.VideoCapture(1)
vid.set(10, 150)



def preProcessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 200, 200)
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
    imgErode = cv2.erode(imgDial, kernel, iterations=1)
    return imgErode



def getContours(img):
    biggestCn = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            if area > maxArea and len(approx) == 4:
                biggestCn = approx
                maxArea = area

    cv2.drawContours(imgContour, biggestCn, -1, (255, 0, 0), 20)
    return biggestCn



def reorder (myPoints):
    myPoints = myPoints.reshape((4, 2))
    myNewPoints = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    # print("add", add)
    myNewPoints[0] = myPoints[np.argmin(add)]
    myNewPoints[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis = 1)
    myNewPoints[1] = myPoints[np.argmin(diff)]
    myNewPoints[2] = myPoints[np.argmax(diff)]
    # print("NewPoints",myNewPoints)
    return myNewPoints



def getWarp(img, biggestCn):
    biggestCn = reorder(biggestCn)
    pts1 = np.float32(biggestCn)
    pts2 = np.float32([[0, 0], [frameWidth, 0], [0, frameHeight], [frameWidth, frameHeight]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (frameWidth, frameHeight))

    imgCropped = imgOutput[20:imgOutput.shape[0]-20, 20:imgOutput.shape[1]-20]
    imgCropped = cv2.resize(imgCropped, (frameWidth, frameHeight))

    return imgCropped




while True:
    success, img = vid.read()
    img = cv2.resize(img, (frameWidth, frameHeight))
    imgContour = img.copy()

    imgResult = preProcessing(img)
    biggestCn = getContours(imgResult)

    if biggestCn.size != 0:
        imgWarped = getWarp(img, biggestCn)
        imageArray = ([img, imgResult],
                      [imgContour, imgWarped])
        # imageArray = ([imgContour, imgWarped])
        cv2.imshow("ImageWarped", imgWarped)
    else:
        imageArray = ([img, imgResult],
                      [imgContour, img])


    stackedImages = stackImages(0.5, imageArray)
    cv2.imshow("WorkFlow", stackedImages)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

