import cv2
import numpy as np



img = np.zeros((512, 512, 3), np.uint8)
img[:] = 255, 0, 0    # (the " : " determines the part that should be colored)
cv2.line(img, (0, 0), (300, 300), (0, 255, 0), 3)
cv2.rectangle(img, (0, 0), (250, 350), (0, 0, 255), 2)  # or cv2.FILLED
cv2.circle(img, (400, 50), 30, (255, 0, 255), 5)  # or cv2.FILLED
cv2.putText(img, " OPENCV ", (300, 100), cv2.FONT_XXXX_XXXX, 1, (0, 200, 0), 1)


cv2.imshow("name of window", img)
cv2.waitKey(0)