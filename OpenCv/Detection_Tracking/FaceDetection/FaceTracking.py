import cv2
import mediapipe as mp


vid = cv2.VideoCapture(0)

mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection(0.75)
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = vid.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
            mpDraw.draw_detection(img, detection)    # drawing by default

            bboxClass = detection.location_data.relative_bounding_box
            h, w, c = img.shape
            bbox = int(bboxClass.xmin * w), int(bboxClass.ymin * h), \
                   int(bboxClass.width * w), int(bboxClass.height * h)

            cv2.rectangle(img, bbox, (255, 0, 255), 2)     # manual drawing
            cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)



    cv2.imshow("Image", img)
    cv2.waitKey(1)