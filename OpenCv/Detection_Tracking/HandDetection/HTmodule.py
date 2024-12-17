import cv2
import mediapipe as mp
import numpy as np
import math
import time


mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


def findHands(img, draw=True):
    global results
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks and draw:
        for hand in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)
    return img


def findHandLmPos(img, lmsNum=False, draw=True):
    bboxes = []
    lmLists = []
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            xList = []
            yList = []
            bbox = []
            lmList = []
            for id, lm in enumerate(hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                xList.append(cx)
                yList.append(cy)
                lmList.append([id, cx, cy])
                if lmsNum:
                    cv2.putText(img, str(id), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 0.6, (0, 255, 0), 1)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            if draw:
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            bbox = xmin, ymin, xmax, ymax

            bboxes.append(bbox)
            lmLists.append(lmList)

    return lmLists, bboxes


def fingersUp(lmList):

    tipIds = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
        fingers.append(1)
    else:
        fingers.append(0)

    # 4 Fingers
    for id in range(1, 5):
        if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)
    # print(fingers)

    return fingers


def findDistance(img, lmList, p1, p2, draw=True, r=7, t=2):
    x1, y1 = lmList[p1][1], lmList[p1][2]
    x2, y2 = lmList[p2][1], lmList[p2][2]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    if draw:
        cv2.circle(img, (x1, y1), r, (200, 40, 155), cv2.FILLED)
        cv2.circle(img, (x2, y2), r, (200, 40, 155), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (200, 40, 155), t)
        cv2.circle(img, (cx, cy), r, (200, 40, 155), cv2.FILLED)

    length = math.hypot(x2 - x1, y2 - y1)
    lineInfo = [x1, y1, x2, y2, cx, cy]

    return length, lineInfo




def main():

    pTime = 0  # fps

    vid = cv2.VideoCapture(1)

    while True:
        success, img = vid.read()
        img = findHands(img)
        lmLists, bboxes = findHandLmPos(img, draw=False)

        if len(lmLists) != 0:
            print(lmLists[0][5])


        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)




if __name__ == "__main__":
    main()
