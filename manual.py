import numpy as np
import pandas as pd 
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="dark")
#from tensorflow.keras.applications.vgg16 import VGG16
#from tensorflow.keras.models import Sequential,Model
#from tensorflow.keras.layers import Dense,Dropout,Activation
from sklearn import preprocessing
import cv2
import glob
import os
import warnings as wr

wr.filterwarnings("ignore")

train_path=r'archive/Train/Train/JPEGImages'
train_annot=r'archive/Train/Train/Annotations'

test_path=r'archive/Test/Test/JPEGImages'
test_annot=r'archive/Test/Test/Annotations'

def creatingInfoData(Annotpath):
    information={'xmin':[],'ymin':[],'xmax':[],'ymax':[],'ymax':[],'name':[]
                ,'label':[]}

    for file in sorted(glob.glob(str(Annotpath+'/*.xml*'))):
        dat=ET.parse(file)
        for element in dat.iter():    

            if 'object'==element.tag:
                for attribute in list(element):
                    if 'name' in attribute.tag:
                        name = attribute.text                 
                        information['label'] += [name]
                        information['name'] +=[file.split('/')[-1][0:-4]]

                    if 'bndbox'==attribute.tag:
                        for dim in list(attribute):
                            if 'xmin'==dim.tag:
                                xmin=int(round(float(dim.text)))
                                information['xmin']+=[xmin]
                            if 'ymin'==dim.tag:
                                ymin=int(round(float(dim.text)))
                                information['ymin']+=[ymin]
                            if 'xmax'==dim.tag:
                                xmax=int(round(float(dim.text)))
                                information['xmax']+=[xmax]
                            if 'ymax'==dim.tag:
                                ymax=int(round(float(dim.text)))
                                information['ymax']+=[ymax]
                     
    return pd.DataFrame(information)

train_info = creatingInfoData(train_annot)
test_info = creatingInfoData(test_annot)
print(test_info)
test_images = sorted(glob.glob(os.path.join(test_path,"*.jpg")))
color = (255,0,0)
thickness = 2
it = 1
for img_path in test_images:
    #if it>10:
        #break
    img_id = img_path.split(".")[0].split("/")[-1]
    labels = test_info[test_info.name == img_id]
    img = cv2.imread(img_path)
    for index, lab in labels.iterrows():
        img = cv2.rectangle(img, (lab['xmin'], lab['ymin']), (lab['xmax'], lab['ymax']), color, thickness)
        #print(lab['xmin'],lab['ymin'],lab['xmax'],lab['ymax'])
    cv2.imshow('img',img)
    #it += 1
    cv2.waitKey(0)