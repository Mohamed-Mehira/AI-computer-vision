import cv2
import numpy as np

width = 640
height = 480

imgMask = np.zeros((height, width))

triangle = np.array([[(200, height), (320, 180), (440, height)]], np.int32)
cv2.fillPoly(imgMask, triangle, 255)


cv2.imshow('test', imgMask)
cv2.waitKey(0)
cv2.destroyAllWindows()