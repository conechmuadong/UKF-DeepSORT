import cv2
import numpy as np
import os

class ObjectDetection:
    def __init__(self, weights_path = "dnn_model/yolov4.weights", cfg_path = "dnn_model/yolov4.cfg"):
        yolo = cv2.dnn.readNet(weights_path, cfg_path)
        self.classes = []
        self.image_size = 608
        self.load_class_names()

        yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.model = cv2.dnn_DetectionModel(yolo)
        self.model.setInputParams(size=(self.image_size, self.image_size), scale=1/255)

    def load_class_names(self, classes_path="dnn_model/classes.txt"):
        with open(classes_path, "r") as file_object:
            for class_name in file_object.readlines():
                class_name = class_name.strip()
                self.classes.append(class_name)

        return self.classes

    def detect(self, frame):
        return self.model.detect(frame, nmsThreshold=0.4, confThreshold=0.5)

    def get_detection_matrix(self, sequences, vid_path, result_path):
        result = []
        if sequences:
            image_filenames = {
                int(os.path.splitext(f)[0]): os.path.join(vid_path, f)
                for f in os.listdir(vid_path)}
            count = 0
            for image_filename in image_filenames:
                frame = cv2.imread(image_filenames[image_filename])
                (frame, confidence, boxes) = self.detect(frame)
                box_count = 0
                for box in boxes:
                    temp = np.zeros(10)
                    temp[0] = count
                    temp[1] = -1
                    temp[2] = box[0]
                    temp[3] = box[1]
                    temp[4] = box[2]
                    temp[5] = box[3]
                    temp[6] = confidence[box_count]
                    temp[7] = -1
                    temp[8] = -1
                    temp[9] = -1
                    result.append(temp)
                    box_count += 1
                count += 1
        else:
            cap = cv2.VideoCapture(vid_path)

            count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                (frame, confidence, boxes) = self.detect(frame)
                box_count = 0
                for box in boxes:
                    temp = np.zeros(10)
                    temp[0] = count
                    temp[1] = -1
                    temp[2] = box[0]
                    temp[3] = box[1]
                    temp[4] = box[2]
                    temp[5] = box[3]
                    temp[6] = confidence[box_count]
                    temp[7] = -1
                    temp[8] = -1
                    temp[9] = -1
                    result.append(temp)
                    box_count += 1
                count += 1
        result = np.asarray(result)
        # result[:, 0:5] = result[:, 0:5].astype(np.int32)
        # result[:, 7:9] = result[:, 7:9].astype(np.int32)
        np.save(result_path, result)
        return result
