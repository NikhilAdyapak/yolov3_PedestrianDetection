# IOU display all images separate persons

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import cv2

df = pd.read_csv('output.csv') 
font = cv2.FONT_HERSHEY_SIMPLEX
thickness = 2
color1 = (0,255,0)
color2 = (255,0,0)
fontScale = 1
org = (00, 185)

for index, lab in df.iterrows():
    if not (pd.isna(lab['gt_x'])):
        img_path = 'Test/Test/JPEGImages/' + 'image (' + str(int(lab['Image_num'])) + ')' + '.jpg'
        img = cv2.imread(img_path)
        img = cv2.rectangle(img, (round(lab['x']), round(lab['y'])), (round(lab['xmax']), round(lab['ymax'])), color1, thickness)
        text = 'conf:' + (str(lab['conf'])[0:4]) + ' ' +'iou:' +(str(lab['iou'])[0:4])
        cv2.putText(img, text, org, font, fontScale, color1, thickness)
        cv2.rectangle(img, (round(lab['gt_x']),round(lab['gt_y'])), (round(lab['gt_xmax']),round(lab['gt_ymax'])), color2, thickness)
        cv2.imshow('img',img)
        cv2.waitKey(0)