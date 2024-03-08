from __future__ import divison, print_function, absolute_import

import cv2
import numpy as np

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from detection_module import Detect

def feature_extractor(detection_list, extractor, img):
	extracted_list = []
	for detection in detection_list:
	    	bbox, category, confident = detection[0], detection[1], detection[2]
	        if(category != "person"):
	            continue
		ext_img = img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
		feature = extractor.extract_feature(ext_img)
		extracted_list.append([bbox, feature, confident])
	return extracted_list
def create_detections(detection_list, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    detections = []
    for detection in detection_list:
        bbox, feature, confident = detection[0], detection[1], detection[2]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list

def gather_detections(frame):
	frame = cv2.flip(image, 1)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	detection_list = Detect.predict(frame)
	return detection_list

def run(output_file, min_confidence, extractor, detector,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget):
	extractor = FeatureExtractor(extractor)
	Detect.config(detector, min_confidence)
	metric = nn_matching.NearestNeighborDistanceMetric(
		"cosine", max_cosine_distance, nn_budget)
	tracker = Tracker(metric)

	cap = cv2.VideoCapture(0)
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)

	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break
		detection_list = gather_detections(frame)
		detection_list = feature_extractor(detection_list, extractor, frame)
