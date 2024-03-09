<<<<<<< HEAD
from __future__ import absolute_import
=======
from __future__ import print_function, absolute_import
>>>>>>> origin/main

import cv2
import numpy as np

import argparse

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from detect_module import Detect
from tools.feature_extractor import FeatureExtractor

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
        detections.append(Detection(bbox, confident, feature))
    return detections

def gather_detections(frame):
	frame = cv2.flip(frame, 1)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	detection_list = Detect.predict(frame)
	return detection_list

def draw_detections(image, detections):
	vis = visualization.Visualization()
	vis.set_image(image)
	vis.draw_detections(detections)
	image = vis.viewer.image
	return image

def draw_tracks(image, tracks):
	vis = visualization.Visualization()
	vis.set_image(image)
	vis.draw_trackers(tracks)
	image = vis.viewer.image
	return image

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
	while cap.isOpened() and frame_counter < 100:
		ret, frame = cap.read()
		if not ret:
			break
		detection_list = gather_detections(frame)
		detection_list = feature_extractor(detection_list, extractor, frame)
		detections = create_detections(detection_list, min_detection_height)
		
		boxes = np.array([d.tlwh for d in detections])
		scores = np.array([d.confidence for d in detections])
		indices = preprocessing.non_max_suppression(
			boxes, nms_max_overlap, scores)
		detections = [detections[i] for i in indices]

		tracker.predict()
		tracker.update(detections)

		frame = draw_detections(frame, detections)
		frame = draw_tracks(frame, tracker.tracks)

		frame_counter += 1
		out.write(frame) 
		if cv2.waitKey(0) == 27: 
			break
	cv2.destroyAllWindows() 
	out.release() 
	cap.release()

def parse_args():
	parser = argparse.ArgumentParser(description="Run deep sort on Raspberry Pi 4")
	parser.add_argument(
		"--output_file", help="Path to save the output video file", type=str, default="output.avi")
	parser.add_argument(
		"--min_confidence", help="Minimum confidence score for detections", type=float, default=0.3)
	parser.add_argument(
		"--extractor", help="Path to the feature extractor model", type=str, 
		default="resources/networks/mars-small128.tflite")
	parser.add_argument(
		"--detector", help="Path to the object detector model", type=str, 
		default="resources/networks/mobilenetssdv2.tflite")
	parser.add_argument(
		"--nms_max_overlap", help="Non-maxima suppression maximum overlap", type=float, default=0.5)
	parser.add_argument(
		"--min_detection_height", help="Minimum detection bounding box height", type=int, default=0)
	parser.add_argument(
		"--max_cosine_distance", help="Maximum cosine distance", type=float, default=0.2)
	parser.add_argument(
		"--nn_budget", help="Nearest neighbor budget", type=int, default=100)
	return parser.parse_args()

if __name__ == "__main__":
	args = parse_args()
	run(args.output_file, args.min_confidence, args.extractor, args.detector,
		args.nms_max_overlap, args.min_detection_height, args.max_cosine_distance,
		args.nn_budget)
