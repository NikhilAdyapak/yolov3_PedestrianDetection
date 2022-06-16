# CSV of yolov3 - gt 

import numpy as np
import cv2
import io
import pandas as pd 
import xml.etree.ElementTree as ET
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

df = pd.read_csv('output.csv') 
glob_count = 1

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def iou_mapping(box_yolo,gt_boxes):
    global df,glob_count
    overall_iou = []
    for i in gt_boxes:
        single_iou = bb_intersection_over_union(box_yolo,i)
        overall_iou.append(single_iou)
    max_iou = max(overall_iou)
    ind = overall_iou.index(max_iou)
    if max_iou > 0.5:
        gt = gt_boxes[ind]
        df.loc [glob_count-1 , 'gt_x'] = gt[0]
        df.loc [glob_count-1 , 'gt_y'] = gt[1]
        df.loc [glob_count-1 , 'gt_xmax'] = gt[2]
        df.loc [glob_count-1 , 'gt_ymax'] = gt[3]
        df.loc [glob_count-1 , 'iou'] = max_iou


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


def detector(image, num, gt_boxes):
    global glob_count,df
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
            box_yolo = [x,y,w+x,y+h]
            iou_mapping(box_yolo,gt_boxes)
            glob_count += 1
            it += 1

train_info = creatingInfoData(train_annot)
test_info = creatingInfoData(test_annot)
print(test_info)
test_images = sorted(glob.glob(os.path.join(test_path,"*.jpg")))
color = (255,0,0)
thickness = 2
it = 1
i = 1

for k in range(1,236):
    inp_path = 'Test/Test/JPEGImages/' + 'image (' + str(k) + ')' + '.jpg'
    img_id = 'image (' + str(k) + ')'
    labels = test_info[test_info.name == img_id]
    img = cv2.imread(inp_path)
    gt_boxes = []
    for index, lab in labels.iterrows():
        gt_boxes.append([lab['xmin'], lab['ymin'], lab['xmax'], lab['ymax']])
    coco_classes = None
    with open('coco.names','r') as f:
        coco_classes = [line.strip() for line in f.readlines()]
    net = cv2.dnn.readNet('yolov3.weights','yolov3.cfg')
    detector(img,i,gt_boxes)
    df.to_csv('output.csv',index = False)