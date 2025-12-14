import cv2
import os


haar_file = "C:/Open CV/face_identifier/haarcascade_frontalface_default.xml"


data_set = "C:/Open CV/face_identifier/datasets"

arnav_folder = "C:/Open CV/face_identifier/datasets/arnav"

img_stored = os.path.join(data_set,arnav_folder)

width = 130
height = 100

detector = cv2.CascadeClassifier(haar_file)

webcam = cv2.VideoCapture(0)


i = 1

while i < 31:
    valid, img = webcam.read()
    print(valid,i)
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = detector.detectMultiScale(imggray, 1.3, 4)
    for x,y,w,h in result:
        img = cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255), 5)
        face = imggray[y:y+h, x:x+w]
        face = cv2.resize(face, (130,100))
        cv2.imwrite("%s/%s.png" % (img_stored,i), face)
    i = i+1

    cv2.imshow("Image", img)
    key = cv2.waitKey(10)
    if key == 27:
        break

