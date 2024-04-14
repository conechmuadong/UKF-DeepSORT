# UKF-DeepSORT

## Introduction
 This repository contains code for a DeepSORT-based Object Tracking using Unscented Kalman Filter with CTRV motion model. We extent the original DeepSORT algorithm for using the UKF and provide an efficient implementation on Raspberry Pi 4 Module B.

## Dependencies

The code is compatible with Python 3.10. The following dependencies are needed to run the tracker:
- NumPy
- SciPy
- OpenCV
- TensorFlow >= 2.15.0

## Installation

First, clone the repository:

```
git clone https://github.com/conechmuadong/UKF-DeepSORT
```

For evaluation on MOT16 without running on Raspberry Pi, download pre-generated detections and the CNN checkpoint file from [here]((https://drive.google.com/open?id=18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp)).

*NOTE:* 
- The candidate object locations of pre-generated detections are
taken from the following paper:

    ```
    F. Yu, W. Li, Q. Li, Y. Liu, X. Shi, J. Yan. POI: Multiple Object Tracking with High Performance Detection and Appearance Feature. In BMTT, SenseTime Group Limited, 2016.
    ```

- The pre-generated feature vectors are taken from DeepSORT:
    ```
    Wojke, N., Bewley, A. and Paulus, D. (2017) ‘Simple online and realtime tracking with a Deep Association metric’, 2017 IEEE International Conference on Image Processing (ICIP) [Preprint]. doi:10.1109/icip.2017.8296962. 
    ```
On Raspberry Pi, run the following command to install the suitable TensorFlow for Debian-ARM-x64 systems:
```
    pip3 install tensorflow-aws-cpu 
```
For evaluation on MOT16 and running on Raspberry Pi, download pre-generated detections and pre-generated feature in [here]()

## Running the tracker:
- On a normal PC: 

    The following example starts the tracker on one of the
    [MOT16 benchmark](https://motchallenge.net/data/MOT16/)
    sequences. We assume resources have been extracted to the repository root directory and the MOT16 benchmark data is in `./MOT16`:
    ```
    python deep_sort_app.py \
        --sequence_dir=./MOT16/test/MOT16-06 \
        --detection_file=./resources/detections/MOT16_POI_test/MOT16-06.npy \
        --min_confidence=0.3 \
        --nn_budget=100 \
        --display=True
    ```
    Check `python deep_sort_app.py -h` for an overview of available options. The repository also has scripts to visualize results, generate videos, and evaluate the MOT challenge benchmark.

- On Raspberry Pi:    
    The following example starts the tracker for the video captured from webcam connected to the Raspberry Pi:
    ```
    python3 raspi_deepsort.py \
        --output_video=./results/output.avi \
        --min_confidence=0.2 \
        --extractor=resources/networks/mars-small128.tflite \
        --detector=resources/networks/mobilenetssdv2.tflite \
        --nn_budget=50
    ```
    Check `python raspi_deepsort.py -h` for an overview of available options.

## High-level overview of source files

In the top-level directory are executable scripts to execute, evaluate, and
visualize the tracker. The main entry point is in `deep_sort_app.py`, runs the tracker on a MOTChallenge sequence. `raspi_deepsort.py` runs the tracker on Raspberry Pi's Webcam's captured video.  

In package `deep_sort` is the main tracking code:

* `detection.py`: Detection base class.
* `kalman_filter.py`: An Unscented Kalman filter implementation and concrete
   parametrization for image space filtering.
* `linear_assignment.py`: This module contains code for min cost matching and
   the matching cascade.
* `iou_matching.py`: This module contains the IOU matching metric.
* `nn_matching.py`: A module for a nearest neighbor matching metric.
* `track.py`: The track class contains single-target track data such as Kalman
  state, number of hits, misses, hit streak, associated feature vectors, etc.
* `tracker.py`: This is the multi-target tracker class.

## Acknowledgement
This repository reuses codes and structures from [original DeepSORT repository](https://github.com/nwojke/deep_sort).
