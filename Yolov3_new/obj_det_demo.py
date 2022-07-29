import os
import glob
import time
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm
import cv2
from keras.utils import np_utils
import torch
import csv

from pipeline_input import *
from constants import *
from natsort import os_sorted
import warnings as wr
wr.filterwarnings("ignore")

f = open("/tmp/output.csv", "w")
writer = csv.DictWriter(
    f, fieldnames=["Sno","Image_num","x","y","xmax","ymax","conf","gt_x","gt_y","gt_xmax","gt_ymax","iou"])
writer.writeheader()
f.close()
df_temp = pd.read_csv("/tmp/output.csv")
glob_count = 1

class obj_det_interp_1(pipeline_dataset_interpreter):
	def load(self) -> None:
		super().load()
		train_path=os.path.join(self.input_dir, 'Train/Train/JPEGImages')
		train_annot=os.path.join(self.input_dir, 'Train/Train/Annotations')
		test_path=os.path.join(self.input_dir, 'Test/Test/JPEGImages')
		test_annot=os.path.join(self.input_dir, 'Test/Test/Annotations')
		assert os.path.exists(train_path)
		assert os.path.exists(train_annot)
		assert os.path.exists(test_path)
		assert os.path.exists(test_annot)
		xtrain = self.generate_data(train_annot, train_path)
		xtest = self.generate_data(test_annot, test_path)
		self.dataset = {
			'train': {
				'x': xtrain["image"].unique(),
				'y': xtrain
			},
			'test': {
				'x': xtest["image"].unique(),
				'y': xtest
			}
		}
		
	def generate_data(self, Annotpath, Imagepath):
		information={'xmin':[],'ymin':[],'xmax':[],'ymax':[],'ymax':[],'name':[] ,'label':[], 'image':[]}
		for file in sorted(glob.glob(str(Annotpath+'/*.xml*'))):
			dat=ET.parse(file)
			for element in dat.iter():    
				if 'object'==element.tag:
					for attribute in list(element):
						if 'name' in attribute.tag:
							name = attribute.text
							file_name = file.split('/')[-1][0:-4]
							information['label'] += [name]
							information['name'] +=[file_name]
							#information['name'] +=[file]
							information['image'] += [os.path.join(Imagepath, file_name + '.jpg')]
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

class obj_det_data_visualizer(pipeline_data_visualizer):

	def visualize(self, x, y, preds, mode='') -> None:
		plot = 'plot' in mode
		plot = True
		image_names_list = y["name"].unique()
		iou_list = []
		iou_thresh = 0.5
		yolo_metrics = {
			'tp':0, 	# iou>thresh
			'fp': 0, 	# 0<iou<thresh
			'fn':0		# iou==0	
		}
		for image_name in image_names_list:
			labels = y[y["name"]==image_name]
			detections = preds[preds["name"]==image_name]
			for index1, lab in labels.iterrows():
				largest_iou = 0.0
				for index2, yolo_bb in detections.iterrows():
					iou = get_iou(lab, yolo_bb)
					if iou > largest_iou:
						largest_iou = iou
				if largest_iou==0:
					yolo_metrics['fn'] += 1
				else:
					if largest_iou>iou_thresh:
						yolo_metrics['tp'] += 1
					else:
						yolo_metrics['fp'] += 1
				iou_list.append(largest_iou)
			if plot:
				image_path = labels["image"].iloc[0]
				img = cv2.imread(image_path)
				for index1, lab in labels.iterrows():
					img = cv2.rectangle(img, (round(lab['xmin']), round(lab['ymin'])), (round(lab['xmax']), round(lab['ymax'])), (255,0,0),2)
				for index2, lab in detections.iterrows():
					img = cv2.rectangle(img, (round(lab['xmin']), round(lab['ymin'])), (round(lab['xmax']), round(lab['ymax'])), (0,255,0),2)
				print(len(labels), len(detections))
				print(labels)
				print(detections)
				cv2.imshow('img', img)
				cv2.waitKey(1)
				time.sleep(1)

class obj_det_evaluator:

	def evaluate(self, x, y, plot=False):
		preds = self.predict(x)
		image_names_list = y["name"].unique()
		iou_list = []
		iou_thresh = 0.5
		yolo_metrics = {
			'tp':0, 	# iou>thresh
			'fp': 0, 	# 0<iou<thresh
			'fn':0		# iou==0	
		}
		for image_name in image_names_list:
			labels = y[y["name"]==image_name]
			detections = preds[preds["name"]==image_name]
			for index1, lab in labels.iterrows():
				largest_iou = 0.0
				for index2, yolo_bb in detections.iterrows():
					iou = get_iou(lab, yolo_bb)
					if iou > largest_iou:
						largest_iou = iou
				if largest_iou==0:
					yolo_metrics['fn'] += 1
				else:
					if largest_iou>iou_thresh:
						yolo_metrics['tp'] += 1
					else:
						yolo_metrics['fp'] += 1
				iou_list.append(largest_iou)
			if plot:
				image_path = labels["image"].iloc[0]
				img = cv2.imread(image_path)
				for index1, lab in labels.iterrows():
					img = cv2.rectangle(img, (round(lab['xmin']), round(lab['ymin'])), (round(lab['xmax']), round(lab['ymax'])), (255,0,0),2)
				for index2, lab in detections.iterrows():
					img = cv2.rectangle(img, (round(lab['xmin']), round(lab['ymin'])), (round(lab['xmax']), round(lab['ymax'])), (0,255,0),2)
				print(len(labels), len(detections))
				print(labels)
				print(detections)
				cv2.imshow('img', img)
				cv2.waitKey(0)
		prec = yolo_metrics['tp'] / float(yolo_metrics['tp'] + yolo_metrics['fp'])
		recall = yolo_metrics['tp'] / float(yolo_metrics['tp'] + yolo_metrics['fn'])
		f1_score = 2*prec*recall/(prec+recall)
		iou_avg = sum(iou_list) / len(iou_list)
		results = {
			'prec': prec,
			'recall': recall,
			'f1_score': f1_score,
			'iou_avg': iou_avg,
			'confusion': yolo_metrics
		}
		return results, preds


class obj_det_pipeline_model(obj_det_evaluator, pipeline_model):

	def load(self):
		self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
		
	def train(self, dataset):
		# TODO: Training
		pass
		
	def predict(self, x: dict) -> np.array:
		# Runs prediction on list of values x of length n
		# Returns a list of values of length n
		predict_results = {
			'xmin': [], 'ymin':[], 'xmax':[], 'ymax':[], 'confidence': [], 'name':[], 'image':[]
		}
		for image_path in tqdm(x):
			img = cv2.imread(image_path)
			results = self.model(image_path)
			df = results.pandas().xyxyn[0]
			res = df[df["name"]=="person"]
			for index, yolo_bb in res.iterrows():
				file_name = image_path.split('/')[-1][0:-4]
				predict_results["xmin"] += [yolo_bb["xmin"]*img.shape[1]]
				predict_results["ymin"] += [yolo_bb["ymin"]*img.shape[0]]
				predict_results["xmax"] += [yolo_bb["xmax"]*img.shape[1]]
				predict_results["ymax"] += [yolo_bb["ymax"]*img.shape[0]]
				predict_results["confidence"] += [yolo_bb["confidence"]]
				predict_results["name"] += [file_name]
				predict_results["image"] += [image_path]
		predict_results = pd.DataFrame(predict_results)
		return predict_results


class obj_det_pipeline_model_yolov5n(obj_det_pipeline_model):
	def load(self):
		self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
		
class obj_det_pipeline_model_yolov5s(obj_det_pipeline_model):
	def load(self):
		self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

class obj_det_pipeline_model_yolov5m(obj_det_pipeline_model):
	def load(self):
		self.model = torch.hub.load('ultralytics/yolov5', 'yolov5m')

class obj_det_pipeline_model_yolov5l(obj_det_pipeline_model):
	def load(self):
		self.model = torch.hub.load('ultralytics/yolov5', 'yolov5l')

class obj_det_pipeline_model_yolov5x(obj_det_pipeline_model):
	def load(self):
		self.model = torch.hub.load('ultralytics/yolov5', 'yolov5x')

class obj_det_pipeline_ensembler_1(obj_det_evaluator, pipeline_ensembler):

	def predict(self, x: dict) -> np.array:
		model_names = list(x.keys())
		image_paths = x[model_names[0]]["image"].unique()
		nms_res = {'xmin':[],'ymin':[],'xmax':[],'ymax':[],'ymax':[], 'confidence':[],'name':[], 'image':[]}
		for img_path in image_paths:
			boxes = []
			scores = []
			for mod_name in model_names:
				preds = x[mod_name][x[mod_name]["image"]==img_path]
				for index, lab in preds.iterrows():
					boxes.append((
						lab['xmin'],				# x
						lab['ymin'],				# y
						lab['xmax'] - lab['xmin'],	# w
						lab['ymax'] - lab['ymin'],	# h
					))
					scores.append(lab['confidence'])
			indexes = cv2.dnn.NMSBoxes(boxes, scores,score_threshold=0.4,nms_threshold=0.8)
			for ind in indexes:
				i = ind[0]
				file_name = img_path.split('/')[-1][0:-4]
				nms_res['xmin'] += [boxes[i][0]]
				nms_res['ymin'] += [boxes[i][1]]
				nms_res['xmax'] += [boxes[i][0] + boxes[i][2]]
				nms_res['ymax'] += [boxes[i][1] + boxes[i][3]]
				nms_res['confidence'] += [scores[i]]
				nms_res['name'] += [file_name]
				nms_res['image'] += [img_path]
		nms_res = pd.DataFrame(nms_res)
		print(nms_res)
		return nms_res
		
class obj_det_pipeline_model_yolov3(obj_det_evaluator, pipeline_model):

	def load(self):
		# Load your model
		# self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
		pass

	def predict(self, x: dict) -> pd.DataFrame:
            # x: dict whose keys are model name and values are the model prediction
            # returns: pd.DataFrame with the following cols

		def detector(image, num):
			global glob_count,df_temp
			height, width = image.shape[:2]
			height = image.shape[0]
			width = image.shape[1]
			net.setInput(cv2.dnn.blobFromImage(image,0.00392,(416,416),(0,0,0),True,crop=False))
			person_layer_names = net.getLayerNames()
			uncon_lay = net.getUnconnectedOutLayers()
			if type(uncon_lay[0])==list:
				person_output_layers = [person_layer_names[i[0] - 1] for i in uncon_lay]
			else:
				person_output_layers = [person_layer_names[i - 1] for i in uncon_lay]
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
			persons_csv = []
			for i in pindex:
				if type(i)==list:
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
					dict_csv = {'Sno':glob_count,'Image_num':num,'x':x,'y':y,'xmax':w+x,'ymax':h+y,'conf':str(person_confidences[i])[0:4]}
					glob_count += 1
					persons_csv.append(dict_csv)
					cv2.rectangle(image, (round(box[0]),round(box[1])), (round(box[0]+box[2]),round(box[1]+box[3])), (0,255,0), 2)
					text = (str(label)[0]) + ' ' + (str(person_confidences[i])[0:4])
					cv2.putText(image, text, (round(box[0])-10,round(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
					it += 1
					
			for i in persons_csv:
				df_temp = df_temp.append(i,ignore_index = True)

		
		global glob_count,df_temp
		predict_results = {'xmin': [], 'ymin':[], 'xmax':[], 'ymax':[], 'confidence': [], 'name':[], 'image':[]}
		paths = x
		paths = os_sorted(paths)
		it = 1
		for i in paths:
			inp_path = i
			img = cv2.imread(inp_path)
			coco_classes = None
			coco_names = 'dependencies/coco.names'
			with open(coco_names,'r') as f:
				coco_classes = [line.strip() for line in f.readlines()]
			wts = 'dependencies/yolov3.weights'
			cfg = 'dependencies/yolov3.cfg'
			net = cv2.dnn.readNet(wts, cfg)
			detector(img, it)
			df_temp.to_csv('output.csv',index = False)
			it += 1
		predict_results = df_temp
		return predict_results

	def evaluate(self, x, y):
		# Evaluation logic, returns results, press
        # Inherit from obj_det_evaluator if you don't need anything custom
		RESULTS, PREDICTIONS = 0, self.predict(x)
		print(PREDICTIONS)
		# print(x)
		# print(y)
		return RESULTS, PREDICTIONS
		pass

	def train(self, dataset):
		# TODO: Training, not implemented yet
		pass

obj_det_input = pipeline_input("obj_det", {'karthika95-pedestrian-detection': obj_det_interp_1}, 
	{
		'obj_det_pipeline_model_yolov3':obj_det_pipeline_model_yolov3
		# 'obj_det_pipeline_model_yolov5n': obj_det_pipeline_model_yolov5n,
		# 'obj_det_pipeline_model_yolov5s': obj_det_pipeline_model_yolov5s,
		# 'obj_det_pipeline_model_yolov5m': obj_det_pipeline_model_yolov5m,
		# 'obj_det_pipeline_model_yolov5l': obj_det_pipeline_model_yolov5l,
		# 'obj_det_pipeline_model_yolov5x': obj_det_pipeline_model_yolov5x,
	}, {
		'obj_det_pipeline_ensembler_1': obj_det_pipeline_ensembler_1
	}, {
		'obj_det_data_visualizer': obj_det_data_visualizer
	})

from depth_perception_demo import depth_input

all_inputs = {}
all_inputs[obj_det_input.get_pipeline_name()] = obj_det_input

# TODO: Add yolov3
#Hello

#all_inputs[depth_input.get_pipeline_name()] = depth_input


#########################################################################

def get_iou(bb1, bb2):
	"""
	Calculate the Intersection over Union (IoU) of two bounding boxes.

	Parameters
	----------
	bb1 : dict
		Keys: {'xmin', 'xmax', 'ymin', 'ymax'}
		The (xmin, ymin) position is at the top left corner,
		the (xmax, ymax) position is at the bottom right corner
	bb2 : dict
		Keys: {'xmin', 'xmax', 'ymin', 'ymax'}
		The (x, y) position is at the top left corner,
		the (xmax, ymax) position is at the bottom right corner

	Returns
	-------
	float
		in [0, 1]
	"""
	assert bb1['xmin'] < bb1['xmax']
	assert bb1['ymin'] < bb1['ymax']
	assert bb2['xmin'] < bb2['xmax']
	assert bb2['ymin'] < bb2['ymax']

	# determine the coordinates of the intersection rectangle
	x_left = max(bb1['xmin'], bb2['xmin'])
	y_top = max(bb1['ymin'], bb2['ymin'])
	x_right = min(bb1['xmax'], bb2['xmax'])
	y_bottom = min(bb1['ymax'], bb2['ymax'])

	if x_right < x_left or y_bottom < y_top:
		return 0.0

	# The intersection of two axis-aligned bounding boxes is always an
	# axis-aligned bounding box
	intersection_area = (x_right - x_left) * (y_bottom - y_top)

	# compute the area of both AABBs
	bb1_area = (bb1['xmax'] - bb1['xmin']) * (bb1['ymax'] - bb1['ymin'])
	bb2_area = (bb2['xmax'] - bb2['xmin']) * (bb2['ymax'] - bb2['ymin'])

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
	assert iou >= 0.0
	assert iou <= 1.0
	return iou
