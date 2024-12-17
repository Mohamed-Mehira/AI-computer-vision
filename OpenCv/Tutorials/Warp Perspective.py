import cv2
import numpy as np
from stackedImages import stackImages

####################
width = 320
height = 160
####################


def nothing(a):
    pass

initialTrackBarVals = [110, 208, 0, 480]
cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 480, 240)
cv2.createTrackbar("X Top", "Trackbars", initialTrackBarVals[0], width // 2, nothing)
cv2.createTrackbar("Y Top", "Trackbars", initialTrackBarVals[1], height, nothing)
cv2.createTrackbar("X Bottom", "Trackbars", initialTrackBarVals[2], width // 2, nothing)
cv2.createTrackbar("Y Bottom", "Trackbars", initialTrackBarVals[3], height, nothing)


# path = '../Resources/vid1.mp4'
# vid = cv2.VideoCapture(path)

path = '../Resources/center_2023_02_12_09_40_09_717.jpg'
img = cv2.imread(path)

while True:

    img = cv2.imread(path)

    # success, img = vid.read()
    #
    # if not success:
    #     vid = cv2.VideoCapture(path)
    #     continue

    widthTop = cv2.getTrackbarPos("X Top", "Trackbars")
    heightTop = cv2.getTrackbarPos("Y Top", "Trackbars")
    widthBottom = cv2.getTrackbarPos("X Bottom", "Trackbars")
    heightBottom = cv2.getTrackbarPos("Y Bottom", "Trackbars")
    points = np.float32([(widthTop, heightTop), (width - widthTop, heightTop), (widthBottom, heightBottom), (width - widthBottom, heightBottom)])
    print(points)

    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (width, height))


    for x in range(0, 4):
        cv2.circle(img, (int(points[x][0]), int(points[x][1])), 1, (0, 0, 255), 3)


    cv2.imshow("image", img)
    cv2.imshow("output", imgOutput)
    cv2.waitKey(3)
