import cv2
import numpy as np
import face_recognition as fr
from datetime import datetime
import os
import time


def getEncodings(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faceEncoding = fr.face_encodings(img)[0]
    return faceEncoding

def markAttendance(name):
    with open('Attendance.csv', 'w+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'{name},{dtString}')



path = 'faces'
knownFaces = []
names = []
imgList = os.listdir(path)
for imgName in imgList:
    # face = cv2.imread(f'{path}/{imgName}')
    face = fr.load_image_file(f'{path}/{imgName}')
    knownFaces.append(face)
    names.append(os.path.splitext(imgName)[0])

knownFaceEncodings = []
for face in knownFaces:
    encoding = getEncodings(face)
    knownFaceEncodings.append(encoding)

vid = cv2.VideoCapture(1)

while True:

    success, img = vid.read()
    img2 = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    faceLocations = fr.face_locations(img2)
    faceEncodings = fr.face_encodings(img2)
    # print("location: ", faceLocations)
    # print("encoding: ", faceEncodings)
    # print("zip: ", zip(faceEncodings, faceLocations))

    for faceEncoding, faceLocation in zip(faceEncodings, faceLocations):
        matches = fr.compare_faces(knownFaceEncodings, faceEncoding)
        confidence = fr.face_distance(knownFaceEncodings, faceEncoding)
        # print(matches, confidence)

        matchIndex = np.argmin(confidence)

        if matches[matchIndex]:
            name = names[matchIndex].upper()
            markAttendance(name)
            # print(matchIndex, name)
            y1, x2, y2, x1 = faceLocation
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        else:
            y1, x2, y2, x1 = faceLocation
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, 'Unknown', (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)


    cv2.imshow('Me', img)
    cv2.waitKey(1)