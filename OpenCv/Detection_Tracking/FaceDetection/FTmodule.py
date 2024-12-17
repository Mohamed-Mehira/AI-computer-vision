import cv2
import mediapipe as mp


mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection(0.75)
mpDraw = mp.solutions.drawing_utils


def findFaces(img, drawDefault=True, drawManual=False):

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    bboxs = []
    if results.detections:
        for id, detection in enumerate(results.detections):
            if drawDefault:
                mpDraw.draw_detection(img, detection)      # drawing by default

            bboxClass = detection.location_data.relative_bounding_box
            h, w, c = img.shape
            bbox = int(bboxClass.xmin * w), int(bboxClass.ymin * h), \
                   int(bboxClass.width * w), int(bboxClass.height * h)

            if drawManual:
                cv2.rectangle(img, bbox, (255, 0, 255), 2)      # manual drawing
                cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 15),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

            bboxs.append([bbox, detection.score])
    return img, bboxs




def main():

    vid = cv2.VideoCapture(0)

    while True:
        success, img = vid.read()
        img, bbfoxs = findFaces(img)

        #if len(bboxs) != 0:
        #    cv2.rectangle(img, bboxs[0][0], (255, 0, 255), 2)  # manual drawing
        #    cv2.putText(img, f'{int(bboxs[0][1][0] * 100)}%', (bboxs[0][0][0], bboxs[0][0][1] - 15),
        #                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)


        cv2.imshow("Image", img)
        cv2.waitKey(1)



if __name__ == "__main__":
    main()