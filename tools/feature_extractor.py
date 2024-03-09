import tensorflow as tf
from PIL import Image
import cv2
import numpy as np

class FeatureExtractor:
	def __init__(self, model):
		self.model = model
		self.interpreter = tf.lite.Interpreter(self.model)
		_ , self.witdh, self.height, _ = self.interpreter.get_input_details()[0]['shape']

	def extract_feature(img):
		img = Image.fromarray(img)

		img = img.convert('RGB').resize(())
		input_tensor = self.interpreter.tensor(img)

		self.interpreter.invoke()

		output_details = self.interpreter.get_output_details()[0]
		output = np.squeeze(self.interpreter.get_tensor(output_details['index']))

		return output
