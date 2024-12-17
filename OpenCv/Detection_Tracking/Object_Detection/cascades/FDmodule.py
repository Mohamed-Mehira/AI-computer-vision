import cv2

path = "../../../Resources/haarcascades/haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(path)


def getObjects(img, scaleVal=1.1, neig=4, minArea=100, display=True):

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    objects = cascade.detectMultiScale(imgGray, scaleVal, neig)

    if not display:
        return objects

    for (x, y, w, h) in objects:
        area = w * h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return objects



if __name__ == '__main__':

    vid = cv2.VideoCapture(1)

    while True:

        success, img = vid.read()

        objects = getObjects(img)

        cv2.imshow("video", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
