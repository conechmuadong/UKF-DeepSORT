from __future__ import divison, print_function, absolute_import

import cv2
import numpy as np

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from detect_module import Detect

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
        detection_list.append(Detection(bbox, confident, feature))
    return detection_list

def gather_detections(frame):
	frame = cv2.flip(frame, 1)
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
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter('output.avi',fourcc, 5, (224,224))
	
	frame_counter = 0
	while cap.isOpened() and frame_counter < 1000:
		ret, frame = cap.read()
		if not ret:
			break
		detection_list = gather_detections(frame)
		detection_list = feature_extractor(detection_list, extractor, frame)

		
		frame_counter += 1
		out.write(frame) 
		if cv2.waitKey(0) == 27: 
			break
	cv2.destroyAllWindows() 
	out.release() 
	cap.release()

		
def draw_detections(image, detection_list) -> np.ndarray:
	_MARGIN = 10
	_ROW_SIZE = 10
	_FONT_SIZE = 1
	_FONT_THICKNESS = 1
	_TEXT_COLOR = (0, 0, 255)  # red
	for detection in detection_list:
		# Draw bounding box
		bbox, feature, confident = detection[0], detection[1], detection[2]
		start_point = bbox[1], bbox[1]+bbox[3]
		end_point = bbox[0], bbox[0]+bbox[2]
		cv2.rectangle(image, start_point, end_point, _TEXT_COLOR, 3)

		# Draw confidence
		confident = round(confident, 2)
		text_location = (_MARGIN + bbox[0],
						_MARGIN + _ROW_SIZE + bbox[1])
		cv2.putText(image, confident, text_location, cv2.FONT_HERSHEY_PLAIN,
					_FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)
	return image

def draw_tracks(image, tracks):
	_MARGIN = 10
	_ROW_SIZE = 10
	_FONT_SIZE = 1
	_FONT_THICKNESS = 1
	_TEXT_COLOR = (0, 0, 255)  # blue
	for track in tracks:
		# Draw bounding_box
		bbox, id = track.to_tlbr(), track.track_id
		start_point = bbox[1], bbox[1]+bbox[3]
		end_point = bbox[0], bbox[0]+bbox[2]
		cv2.rectangle(image, start_point, end_point, _TEXT_COLOR, 3)

		# Draw id
		text_location = (_MARGIN + bbox[0],
						_MARGIN + _ROW_SIZE + bbox[1])
		cv2.putText(image, id, text_location, cv2.FONT_HERSHEY_PLAIN,
					_FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)
	return image
