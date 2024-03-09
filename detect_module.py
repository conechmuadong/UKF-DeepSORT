from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision

import numpy as np

class Detect:

    def __new__(cls):
        """ Prevent object initialization. """
        raise Exception(f"{cls.__name__} object cannot be initialized")
    @classmethod
    
    def config(cls,model: str,  num_threads=5, enable_edgetpu=True, score_threshold=0.3) -> None:
        """ Initialize the object detection model. """

        base_options = core.BaseOptions(
            file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
        detection_options = processor.DetectionOptions(score_threshold=score_threshold)
        options = vision.ObjectDetectorOptions(
            base_options=base_options, detection_options=detection_options)
        cls.detector = vision.ObjectDetector.create_from_options(options)
    
    @classmethod
    def predict(cls,image):
        input_tensor = vision.TensorImage.create_from_array(image)
        detection_result = cls.detector.detect(input_tensor)
        result = []
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            category = detection.categories[0]
            category_name, score = category.category_name, category.score
            bbox = np.array([bbox.origin_x, bbox.origin_y, bbox.width, bbox.height])
            result.append([bbox, category_name, score])
        return result

