import cv2
import numpy as np


def detect(image):
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

    img = cv2.imread(image)
    #img=img[400:1200, 10:720] #cropping
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_scale, 1.2, 6)
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 4)
        roi_gray = gray_scale[y:y+h, x:x+w]  #roi=region of interest
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray,1.2,5)
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,0,255), 2)
    
    cv2.imshow("faces and eyes detected", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

detect("yapay-zeka.jpg")