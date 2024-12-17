import cv2

path = "../../Resources/haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(path)

vid = cv2.VideoCapture(0)
vid.set(3, 640)
vid.set(4, 480)
vid.set(10, 100)

while True:
    success, img = vid.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
