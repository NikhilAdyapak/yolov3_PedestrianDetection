import cv2
import numpy as np

image = cv2.imread('Test/Test/JPEGImages/image (1).jpg')

img = cv2.rectangle(image, (8,12), (360,510), (0,255,0), 2)
x = 8
y = 12
w = 352
h = 498
inc = 15
for i in range(5):
    img = cv2.rectangle(image,(x,y),(x+w , w+h),(0,255,0),2)
    cv2.imshow('img',img)
    cv2.waitKey(1)
    x += inc
    y += inc 
    w += inc
    h += inc

