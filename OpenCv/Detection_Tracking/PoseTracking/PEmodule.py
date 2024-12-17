import cv2
import mediapipe as mp
import math


mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

def findPose(img, draw=True, lmsNumShow=False):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    lmList = []
    if results.pose_landmarks:
        if draw:
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append([id, cx, cy])
            if lmsNumShow:
                cv2.putText(img, str(id), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 0.6, (0, 255, 0), 1)

    return img, lmList


def findAngle(img, lmList, p1, p2, p3, draw=True):

    x1, y1 = lmList[p1][1:]
    x2, y2 = lmList[p2][1:]
    x3, y3 = lmList[p3][1:]

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    if draw:
        cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
        cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)

        cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)

        cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
        cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)

        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.line(img, (x3, y3), (x2, y2), (255, 0, 0), 3)

        cv2.putText(img, str(int(angle)), (x2 - 60, y2 + 20), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    return angle




def main():

    vid = cv2.VideoCapture(0)

    while True:
        success, img = vid.read()
        img, lmList = findPose(img, False)

        if len(lmList) != 0:
            print(lmList[11])
            cv2.circle(img, (lmList[11][1], lmList[11][2]), 15, (75, 140, 49), cv2.FILLED)


        cv2.imshow("Image", img)
        cv2.waitKey(1)




if __name__ == "__main__":
    main()