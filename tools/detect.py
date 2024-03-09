import os
import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter

class Detector:
	def __init__(self, modelpath, min_confidence=0.3, min_height=0):
		self.min_confidence = min_confidence
		self.min_height = min_height
		self.interpreter = Interpreter(model_path=modelpath)
		self.interpreter.allocate_tensors()

		self.input_details = self.interpreter.get_input_details()
		self.output_details = self.interpreter.get_output_details()
		self.height = self.input_details[0]['shape'][1]
		self.width = self.input_details[0]['shape'][2]

		self.float_input = (self.input_details[0]['dtype'] == np.float32)

		self.input_mean = 127.5
		self.input_std = 127.5

	def detect(self, image):
		# Load image and resize to expected shape [1xHxWx3]
		image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		imH, imW, _ = image.shape
		image_resized = cv2.resize(image_rgb, (self.width, self.height))
		input_data = np.expand_dims(image_resized, axis=0)

		# Normalize pixel values if using a floating model (i.e. if model is non-quantized)
		if self.float_input:
			input_data = (np.float32(input_data) - self.input_mean) / self.input_std

		# Perform the actual detection by running the model with the image as input
		self.interpreter.set_tensor(self.input_details[0]['index'],input_data)
		self.interpreter.invoke()

		# Retrieve detection results
		boxes = self.interpreter.get_tensor(self.output_details[1]['index'])[0] # Bounding box coordinates of detected objects
		classes = self.interpreter.get_tensor(self.output_details[3]['index'])[0] # Class index of detected objects
		scores = self.interpreter.get_tensor(self.output_details[0]['index'])[0] # Confidence of detected objects

		detections = []

		# Loop over all detections and draw detection box if confidence is above minimum threshold
		for i in range(len(scores)):
			if ((scores[i] > self.min_confidence) and (scores[i] <= 1.0)):

     		 # Get bounding box coordinates and draw box
     		 # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
				ymin = int(max(1,(boxes[i][0] * imH)))
				xmin = int(max(1,(boxes[i][1] * imW)))
				ymax = int(min(imH,(boxes[i][2] * imH)))
				xmax = int(min(imW,(boxes[i][3] * imW)))
				if (ymax-ymin) > self.min_height:
					detections.append([np.array([xmin, ymin, xmax-xmin, ymax-ymin]),scores[i]])

		return detections
