import cv2
import numpy as np



img = cv2.imread("id")


imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(img, (7, 7), 0)
imgCanny = cv2.Canny(img, 100, 100)

kernel = np.ones((5, 5), np.uint8)
imgDilation = cv2.dilate(imgCanny, kernel, iterations=1)
imgEroded = cv2.erode(imgDilation, kernel, iterations=1)

imgResize = cv2.resize(img, (300, 200))
imgCrop = img[0:200, 200:500]


cv2.imshow("name of window", img)
cv2.waitKey(0)