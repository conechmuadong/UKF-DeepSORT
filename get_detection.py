# from application_util.generate_detection import ObjectDetection
# import cv2
# import numpy as np
#
# detection = ObjectDetection(weights_path="dnn_model/yolov4.weights", cfg_path="dnn_model/yolov4.cfg")
# results_path = "resources/detections/custom_dataset/0000.npy"
# sequences_dir = "custom_data/testing/image_02/0000"
# detections = detection.get_detection_matrix(sequences=True, vid_path=sequences_dir, result_path=results_path)
#
# # detections = np.load("resources/detections/custom_dataset/0000.npy", allow_pickle=True)
#
#
import numpy as np

detections = np.load("resources/detections/MOT16_POI_train/MOT16-02.npy", allow_pickle=True)

for row in range(len(detections[0:])):
    print(detections[row])

