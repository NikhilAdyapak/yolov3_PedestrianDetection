# YOLOv3 Pedestrian Detection

Detect pedestrians in images with YOLOv3, evaluate the detections with Intersection over Union (IoU), and export the results to CSV and JSON.

## Overview

This project runs YOLOv3 through OpenCV's DNN module to find people in images. It loads the standard YOLOv3 network (trained on COCO), keeps only the `person` class, applies non-maximum suppression to drop overlapping boxes, and writes the bounding boxes out for further analysis. A set of IoU utilities scores predicted boxes against reference boxes.

## How it works

- **Detection:** `yolov3_final.py` loads the network with `cv2.dnn.readNet(weights, cfg)`, runs each image at 416x416, filters detections to the `person` class, and applies NMS (`cv2.dnn.NMSBoxes`).
- **Evaluation:** `iou.py`, `iou_sampleBox.py`, and `iou_seperate.py` compute Intersection over Union between predicted and reference boxes.
- **Export:** `yolov3_to_csv.py`, `yolov3_to_csv_full.py`, and `yolov3_to_json.py` save detections to CSV and JSON (see the sample `output.csv`, `output_cp.csv`, and JSON outputs).

## Files

```
yolov3.cfg              YOLOv3 network configuration
coco.names, coco.data   COCO class names and data config
yolov3_final.py         main detection script (person class)
iou*.py                 IoU evaluation utilities
yolov3_to_csv*.py       export detections to CSV
yolov3_to_json.py       export detections to JSON
```

## Setup

The YOLOv3 weights are not included because the file is large. Download the standard COCO weights and place `yolov3.weights` in the repo root:

```bash
wget https://pjreddie.com/media/files/yolov3.weights
pip install opencv-python numpy
```

## Usage

Set the input image path inside the script, then run:

```bash
python yolov3_final.py      # run detection on the input image
python yolov3_to_csv.py     # export detections to CSV
python yolov3_to_json.py    # export detections to JSON
```

## Author

Nikhil Adyapak - [portfolio](https://nikhiladyapak.github.io/) - [LinkedIn](https://www.linkedin.com/in/nikhil-adyapak)
