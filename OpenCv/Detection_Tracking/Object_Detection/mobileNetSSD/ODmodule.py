import cv2

classNames = []
classFile = "../../../Resources/object_detection_files/coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "../../../Resources/object_detection_files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "../../../Resources/object_detection_files/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def detObj(img, objects=[], draw=True, thres=0.5):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=0.2)

    if len(objects) == 0:
        objects = classNames

    objInfo = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):

            className = classNames[classId - 1]

            if className in objects:
                objInfo.append([className, box])

                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, className.upper(), (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 10, box[1] + 65), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    return objInfo





def main():
    vid = cv2.VideoCapture(1)
    vid.set(3, 1280)
    vid.set(4, 720)
    vid.set(10, 70)

    while True:
        success, img = vid.read()
        objectInfo = detObj(img, thres=0.6)

        cv2.imshow("Output", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()