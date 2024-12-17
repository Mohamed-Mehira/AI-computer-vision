import cv2
import numpy as np
from Detection_Tracking.HandDetection import HTmodule as htm
import time
import autopy

##########################
wCam, hCam = 640, 480
frameR = 80   # Frame Reduction
smoothening = 5
#########################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

vid = cv2.VideoCapture(1)
vid.set(3, wCam)
vid.set(4, hCam)

wScreen, hScreen = autopy.screen.size()

while True:

    # 1. Find hand Landmarks
    success, img = vid.read()
    img = htm.findHands(img)
    lmLists, bbox = htm.findHandLmPos(img)

    # 2. Get the tip of the index and middle fingers
    if len(lmLists) != 0:
        x1, y1 = lmLists[0][8][1:]
        x2, y2 = lmLists[0][12][1:]

        # 3. Check which fingers are up
        fingers = htm.fingersUp(lmLists[0])
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        # 4. Only Index Finger : Moving Mode
        if fingers[1] == 1 and fingers[2] == 0:
            # 5. Convert Coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScreen))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScreen))

            # 6. Smoothen Values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
            plocX, plocY = clocX, clocY

            # 7. Move Mouse
            autopy.mouse.move(wScreen - clocX, clocY)
            cv2.circle(img, (x1, y1), 8, (255, 0, 255), cv2.FILLED)


        # 8. Both Index and middle fingers are up : Clicking Mode
        if fingers[1] == 1 and fingers[2] == 1:
            # 9. Find distance between fingers
            length, lineInfo = htm.findDistance(img, lmLists[0], 8, 12)

            # 10. Click mouse if distance short
            if length < 20:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 9, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)