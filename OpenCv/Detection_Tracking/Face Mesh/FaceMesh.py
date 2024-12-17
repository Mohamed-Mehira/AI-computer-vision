import cv2
import mediapipe as mp


vid = cv2.VideoCapture(0)

mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
mpDraw = mp.solutions.drawing_utils
drawSpecs = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

while True:
    success, img = vid.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    lmLists = []
    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, face, mpFaceMesh.FACEMESH_CONTOURS, drawSpecs, drawSpecs)

            lmList = []
            for id, lm in enumerate(face.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])

            lmLists.append(lmList)


    if len(lmLists) != 0:
        print(lmLists[0][0])


    cv2.imshow("Image", img)
    cv2.waitKey(1)
