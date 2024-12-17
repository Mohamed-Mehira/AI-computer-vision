import cv2
import os


##############################################

path = '../Resources/data/images'

minBlur = 100
frameGapVal = 10

countFolder = 0
countSave = 0
frameCount = 0

saveData = True

##############################################

vid = cv2.VideoCapture(0)
vid.set(3, 640)
vid.set(4, 480)

##############################################



def createNewFolder():
    global countFolder
    while os.path.exists(path + str(countFolder)):
        countFolder += 1

    savePath = path + str(countFolder)
    os.makedirs(savePath)


if saveData:
    createNewFolder()


while True:
    success, img = vid.read()

    if saveData:
        blur = cv2.Laplacian(img, cv2.CV_64F).var()

        if frameCount % frameGapVal == 0 and blur > minBlur:
            cv2.imwrite(path + str(countFolder) + '/' + str(countSave) + '_' + str(int(blur)) + '.png', img)
            countSave += 1

        frameCount += 1


    cv2.imshow('image', img)
    cv2.waitKey(1)