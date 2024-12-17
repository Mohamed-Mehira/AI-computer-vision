import cv2

frameWidth = 640
frameHeight = 480

vid = cv2.VideoCapture(1)
# address = "http://192.168.1.2:4747/video"
# vid.open(address)

vid.set(3, frameWidth)
vid.set(4, frameHeight)
vid.set(10, 100)

while True:
    success, img = vid.read()
    cv2.imshow("video", img)
    print(img.shape)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()