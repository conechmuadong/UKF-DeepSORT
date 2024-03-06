# vim: expandtab:ts=4:sw=4
import numpy as np


class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """

    def __init__(self, tlwh, confidence, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision

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
            min_x, min_y = bbox.origin_x, bbox.origin_y
            max_x, max_y = min_x + bbox.width, min_y + bbox.height

            category = detection.categories[0]
            category_name, score = category.category_name, category.score

            result.append([min_x, min_y, max_x, max_y, category_name, score])
        return result
    
