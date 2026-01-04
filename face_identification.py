import cv2
import numpy as np
import os


haar_file = "C:/Open CV/face_identifier/haarcascade_frontalface_default.xml"

dataset = "C:/Open CV/face_identifier/datasets"

images,labels,names,id = [],[], {}, 0

for subdirs,dirs,files in os.walk(dataset):
    for i in dirs:
        names[id] = i
        newpath = os.path.join(dataset,i)
        for x in os.listdir(newpath):
            path=newpath+"/"+x
            label = id
            images.append(cv2.imread(path,0))
            labels.append(label)
        id += 1

images = np.array(images)
label = np.array(labels)

model = cv2.face.LBPHFaceRecognizer_create()
model.train(images,label)

detector = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)


