import cv2
import numpy as np

image = cv2.imread('Test/Test/JPEGImages/image (2).jpg')
'''
img = cv2.rectangle(image, (338,12), (500,199), (255,0,0), 2)
img = cv2.rectangle(image, (22,70), (254,327), (255,0,0), 2)
img = cv2.rectangle(image, (336,11), (494,203), (0,255,0), 2)
img = cv2.rectangle(image, (27,69), (302,319), (0,255,0), 2)
'''

img = cv2.rectangle(image, (133,72), (245,284), (255,0,0), 2)
img = cv2.rectangle(image, (122,79), (250,248), (0,255,0), 2)
cv2.imshow('img',img)
cv2.waitKey(0)
'''
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
'''
'''
336.0,11.0,158,192,0.99,,,,,
26.5,69.0,275,250,0.98,,,,,

122.0,78.5,128,169
[[133, 72, 245, 284]]
'''