# Writing YOLOv3 data to JSON File

import numpy as np
import json
import cv2
import io
import pandas as pd 
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="dark")
import glob
import os
import warnings as wr

wr.filterwarnings("ignore")

try:
    to_unicode = unicode
except NameError:
    to_unicode = str

train_path=r'Train/Train/JPEGImages'
train_annot=r'Train/Train/Annotations'

test_path=r'Test/Test/JPEGImages'
test_annot=r'Test/Test/Annotations'

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


def detector(image, num):
    height, width = image.shape[:2]
    height = image.shape[0]
    width = image.shape[1]
    net.setInput(cv2.dnn.blobFromImage(image,0.00392,(416,416),(0,0,0),True,crop=False))
    person_layer_names = net.getLayerNames()
    person_output_layers = [person_layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    person_outs = net.forward(person_output_layers)
    person_class_ids, person_confidences, person_boxes =[],[],[]
    for operson in person_outs:
        for detection in operson:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x -w/2
                y = center_y - h/2
                person_class_ids.append(class_id)
                person_confidences.append(float(confidence))
                person_boxes.append([x, y, w, h])

    pindex = cv2.dnn.NMSBoxes(person_boxes, person_confidences, 0.5, 0.4)
    it = 0
    persons_in_image = []
    for i in pindex:
        i = i[0]
        box = person_boxes[i]
        lx=round(box[0]+box[2]/2)
        ly=round(box[1]+box[3])-10
        if person_class_ids[i]==0:
            label = str(coco_classes[person_class_ids[i]]) 
            x = person_boxes[it][0]
            y = person_boxes[it][1]
            w = person_boxes[it][2]
            h = person_boxes[it][3]
            persons_in_image.append({'x':x,'y':y,'xmax':w+x,'ymax':h+y,'conf':str(person_confidences[i])[0:4]})
            image = cv2.rectangle(image, (round(box[0]),round(box[1])), (round(box[0]+box[2]),round(box[1]+box[3])), (0,255,0), 2)
            text = (str(label)[0]) + ' ' + (str(person_confidences[i])[0:4])
            cv2.putText(image, text, (round(box[0])-10,round(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            it += 1
        for index, lab in labels.iterrows():
            image = cv2.rectangle(image, (lab['xmin'], lab['ymin']), (lab['xmax'], lab['ymax']), color, thickness)
    
    persons_in_image = tuple(persons_in_image)
    out_path = 'Output_Annotations/image (' + str(num) +').json'
    with io.open(out_path, 'w', encoding='utf8') as outfile:
        str_ = json.dumps(persons_in_image,
                      indent=4, sort_keys=True,
                      separators=(',', ': '), ensure_ascii=False)
        outfile.write(to_unicode(str_))
    

train_info = creatingInfoData(train_annot)
test_info = creatingInfoData(test_annot)
test_images = sorted(glob.glob(os.path.join(test_path,"*.jpg")))
color = (255,0,0)
thickness = 2
it = 1

for i in range(1,236):
    inp_path = 'Test/Test/JPEGImages/' + 'image (' + str(i) + ')' + '.jpg'
    img = cv2.imread(inp_path)
    coco_classes = None
    with open('coco.names','r') as f:
        coco_classes = [line.strip() for line in f.readlines()]

    net = cv2.dnn.readNet('yolov3.weights','yolov3.cfg')

    img_path = test_images[i-1]
    img_id = img_path.split(".")[0].split("/")[-1]
    labels = test_info[test_info.name == img_id]

    detector(img, i)