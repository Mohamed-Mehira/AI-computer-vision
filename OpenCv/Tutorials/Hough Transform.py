import cv2.cv2 as cv2
import numpy as np

img = cv2.imread('../Resources/center_2023_02_12_09_40_09_717.jpg')

### WITHOUT MASKING
# imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
# imgCanny = cv2.Canny(imgBlur, 110, 180)

### WITH MASKING
imgBlur = cv2.GaussianBlur(img, (5, 5), 0)
imgHSV = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2HSV)
lowerYellow = np.array([23, 55, 35])
upperYellow = np.array([73, 112, 130])
maskedYellow = cv2.inRange(imgHSV, lowerYellow, upperYellow)
imgCanny = cv2.Canny(maskedYellow, 75, 150)


# lines = cv2.HoughLinesP(imgCanny, 1, np.pi/180, 100, maxLineGap=200, minLineLength=250)
lines = cv2.HoughLinesP(imgCanny, 2, np.pi/180, 50)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)


cv2.imshow("Edges", imgCanny)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()