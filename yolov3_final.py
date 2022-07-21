class obj_det_pipeline_model_yolov3(obj_det_evaluator, pipeline_model):
	def load(self):
		self.weights = 'dependencies/yolov3.weights'
		self.cfg = 'dependencies/yolov3.cfg'
		self.coco = 'dependencies/coco.names'
		self.coco_classes = None
		with open(self.coco,'r') as f:
			self.coco_classes = [line.strip() for line in f.readlines()]
		self.net = cv2.dnn.readNet(self.weights,self.cfg)
		pass
	def train(self):
		pass
	def predict(self, x: np.array) -> np.array:
		predict_results = {
			'xmin': [], 'ymin':[], 'xmax':[], 'ymax':[], 'confidence': [], 'name':[], 'image':[]
		}
		
		for image_path in tqdm(x):
			image = cv2.imread(image_path)
			height, width = image.shape[:2]
			height = image.shape[0]
			width = image.shape[1]
			self.net.setInput(cv2.dnn.blobFromImage(image,0.00392,(416,416),(0,0,0),True,crop=False))
			person_layer_names = self.net.getLayerNames()
			uncon_lay = self.net.getUnconnectedOutLayers()
			if type(uncon_lay[0])==list:
				person_output_layers = [person_layer_names[i[0] - 1] for i in uncon_lay]
			else:
				person_output_layers = [person_layer_names[i - 1] for i in uncon_lay]
			person_outs = self.net.forward(person_output_layers)
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
				if type(i)==list:
					i = i[0]
				if person_class_ids[i]==0:
					x = person_boxes[it][0]
					y = person_boxes[it][1]
					w = person_boxes[it][2]
					h = person_boxes[it][3]
					file_name = image_path.split('/')[-1][0:-4]
					predict_results["xmin"] += [x]
					predict_results["ymin"] += [y]
					predict_results["xmax"] += [x+w]
					predict_results["ymax"] += [y+h]
					predict_results["confidence"] += [person_confidences[i]]
					predict_results["name"] += [file_name]
					predict_results["image"] += [image_path]
					it += 1
		predict_results = pd.DataFrame(predict_results)
		return predict_results