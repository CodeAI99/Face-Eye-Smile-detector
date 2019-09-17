# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 18:07:36 2019

@author: anshu
"""

import cv2  
  
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')  
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
count = 0
image = cv2.imread("bean3.jpeg")
image = cv2.resize(image, None, fx=1.2, fy=1.2)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_detector.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 2)
    roi_gray = gray[y:y+h, x:x+w] 
    roi_color = image[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (x_eye,y_eye,w_eye,h_eye) in eyes:
        center = (int(x_eye + 0.5*w_eye), int(y_eye + 0.5*h_eye))
        radius = int(0.2 * (w_eye + h_eye))
        color = (0, 255, 0)
        thickness = 2
        cv2.circle(roi_color, center, radius, color, thickness)
        #cv2.circle(roi_color, (x_eye+w_eye, y_eye+h_eye), (w_eye+h_eye), (0, 255, 0), 2)
        #cv2.circle(roi_color, (x_eye, y_eye), (x_eye+w_eye, y_eye+h_eye), (0, 255, 0), 2)
    smile = smile_cascade.detectMultiScale(roi_gray, 1.4, 22)
    for (x_smile,y_smile,w_smile,h_smile) in smile:
        cv2.rectangle(roi_color, (x_smile, y_smile), (x_smile+w_smile, y_smile+h_smile), (0,0,255), 2)

    count = count +1
cv2.imshow("new",image)
cv2.imwrite("newimage.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
  