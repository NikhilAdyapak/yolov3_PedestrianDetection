import numpy as np
import cv2
import json

import io

try:
    to_unicode = unicode
except NameError:
    to_unicode = str

def detector(image, num):
    height, width = image.shape[:2]
    height = image.shape[0]
    width = image.shape[1]
    net.setInput(cv2.dnn.blobFromImage(image,0.00392,(416,416),(0,0,0),True,crop=False))
    person_layer_names = net.getLayerNames()
    person_output_layers = [person_layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    person_outs = net.forward(person_output_layers)
    person_class_ids, person_confidences, person_boxes =[],[],[]
    #conf_scores = []
    for operson in person_outs:
        for detection in operson:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                #conf_scoes.append(confidence)
                #conf_scores.append(str(confidence)[0:4])
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
        #print(box)
        lx=round(box[0]+box[2]/2)
        ly=round(box[1]+box[3])-10
        if person_class_ids[i]==0:
            label = str(coco_classes[person_class_ids[i]]) 
            #print(person_boxes[it])#, '\n',len(person_boxes[it]))
            x = person_boxes[it][0]
            y = person_boxes[it][1]
            w = person_boxes[it][2]
            h = person_boxes[it][3]
            persons_in_image.append({'x':x,'y':y,'w':w,'h':h,'conf':str(person_confidences[it])[0:4]})
            cv2.rectangle(image, (round(box[0]),round(box[1])), (round(box[0]+box[2]),round(box[1]+box[3])), (0,255,0), 2)
            text = (str(label)[0]) + ' ' + (str(person_confidences[it])[0:4])
            cv2.putText(image, text, (round(box[0])-10,round(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            it += 1
    #print(persons_in_image)
    persons_in_image = tuple(persons_in_image)
    out_path = 'Output_Annotations/image (' + str(num) +').json'
    with io.open(out_path, 'w', encoding='utf8') as outfile:
        str_ = json.dumps(persons_in_image,
                      indent=4, sort_keys=True,
                      separators=(',', ': '), ensure_ascii=False)
        outfile.write(to_unicode(str_))

'''for i in range(1,11):
    inp_path = 'Test/Test/JPEGImages/' + 'image (' + str(i) + ')' + '.jpg'
    print(inp_path)
    img = cv2.imread(inp_path)
    coco_classes = None
    with open('coco.names','r') as f:
        coco_classes = [line.strip() for line in f.readlines()]

    net = cv2.dnn.readNet('yolov3.weights','yolov3.cfg')
    detector(img, i)
    #cv2.imshow('result',img)
    out_path = 'results/'+'Result' + str(i) + '.jpg'
    cv2.imwrite(out_path,img)
    #cv2.waitKey(0)
'''

for i in range(11,21):
    inp_path = 'Test/Test/JPEGImages/' + 'image (' + str(i) + ')' + '.jpg'
    #print(inp_path)
    img = cv2.imread(inp_path)
    coco_classes = None
    with open('coco.names','r') as f:
        coco_classes = [line.strip() for line in f.readlines()]

    net = cv2.dnn.readNet('yolov3.weights','yolov3.cfg')
    detector(img, i)
    cv2.imshow('result',img)
    #out_path = 'results/'+'Result' + str(i) + '.jpg'
    #cv2.imwrite(out_path,img)
    cv2.waitKey(0)