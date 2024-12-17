import cv2.cv2 as cv2
import numpy as np
from Tutorials.stackedImages import stackImages

######################
frameWidth = 640
frameHeight = 480
######################


def warp(img, point):

    #pts1 = np.float32([[111, 219], [287, 188], [154, 482], [352, 440]])
    pts1 = np.float32([[170, 60], [470, 60], [70, 420], [570, 420]])
    pts2 = np.float32([[0, 0], [frameWidth, 0], [0, frameHeight], [frameWidth, frameHeight]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    imgOutput = cv2.warpPerspective(img, matrix, (frameWidth, frameHeight))

    homg_point = [point[0], point[1], 1]
    transf_homg_point = matrix.dot(homg_point)
    transf_homg_point /= transf_homg_point[2]
    transf_point = transf_homg_point[:2]
    print(transf_point)

    return imgOutput, transf_point



path = '../Resources/vid1.mp4'
vid = cv2.VideoCapture(path)
vid.set(3, frameWidth)
vid.set(4, frameHeight)

frameCounter = 0

while True:

    frameCounter += 1
    if vid.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
        vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frameCounter = 0

    success, img = vid.read()
    point = np.array([360, 100])

    warp_img, warp_point = warp(img, point)

    cv2.circle(img, point, radius=2, color=(0, 0, 255), thickness=7)
    cv2.circle(warp_img, warp_point.astype(int), radius=2, color=(0, 0, 255), thickness=7)


    imgStack = stackImages(0.4, ([img, warp_img]))
    cv2.imshow("Test", imgStack)
    cv2.waitKey(1)