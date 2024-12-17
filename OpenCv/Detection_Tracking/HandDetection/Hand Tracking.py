import cv2
import mediapipe as mp
import time


vid = cv2.VideoCapture(0)

pTime = 0    # fps
cTime = 0

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = vid.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    lmList = []
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)

        hand = results.multi_hand_landmarks[0]
        for id, lm in enumerate(hand.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            lmList.append([id, cx, cy])

            if id == 0:
                cv2.circle(img, (cx, cy), 15, (75, 140, 49), cv2.FILLED)

    if len(lmList) != 0:
        print(lmList[0])


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)


    cv2.imshow("Image", img)
    cv2.waitKey(1)
