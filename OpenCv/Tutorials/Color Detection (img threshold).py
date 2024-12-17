import cv2
import numpy as np
from stackedImages import stackImages

####################
width = 320
height = 160
####################

def empty(a):
    pass

cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)
cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)


path = '../Resources/center_2023_02_12_09_40_09_717.jpg'
img = cv2.imread(path)

points = [[82, 71], [238, 71], [8, 92], [312, 92]]
pts1 = np.float32(points)
pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
imgWarp = cv2.warpPerspective(img, matrix, (width, height))

imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
imgHSV_warp = cv2.cvtColor(imgWarp, cv2.COLOR_BGR2HSV)

while True:

    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")

    # lower = np.array([h_min, s_min, v_min])
    # upper = np.array([h_max, s_max, v_max])
    lower = np.array([0, 164, 0])
    upper = np.array([179, 255, 255])
    mask = cv2.inRange(imgHSV, lower, upper)
    mask_warp = cv2.inRange(imgHSV_warp, lower, upper)
    imgResult = cv2.bitwise_and(img, img, mask=mask)
    imgResult_warp = cv2.bitwise_and(img, img, mask=mask_warp)


    imgStack = stackImages(1, ([img, imgWarp], [mask, mask_warp], [imgResult, imgResult_warp]))
    cv2.imshow("Stacked Images", imgStack)
    cv2.waitKey(1)










