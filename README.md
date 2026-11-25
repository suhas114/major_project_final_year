# Forest Fire Detection using YOLOv8

A Python application for experimenting with fire- and smoke-related object detection using Ultralytics YOLOv8. The repository includes webcam/video detection, image prediction helpers, and scripts for preparing a custom dataset.

## Features
- Run YOLOv8 inference on a webcam or video file.
- Process a single image and return detection metadata.
- Train a custom YOLO model with a YOLO-format dataset.
- Prepare train, validation, and test folders from the source image directories.

## Requirements
- Python 3.10 or newer is recommended.
- A working webcam is needed for live camera detection.
- Install the Python dependencies from `requirements.txt`.

## Installation
1. Clone the repository and enter its directory:
```bash
git clone https://github.com/suhas114/major_project_final_year.git
cd major_project_final_year
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

3. Install dependencies:
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run live detection
The main `ForestFireDetector` implementation is in `forest_fire_prediction.py`. Start webcam detection with:
```bash
python forest_fire_prediction.py
```
Press `q` in the video window to stop the camera feed.

To use a video file or process one image, import the class:
```python
from forest_fire_prediction import ForestFireDetector

detector = ForestFireDetector()
detector.process_video("path/to/video.mp4")
detections = detector.process_image("path/to/image.jpg")
print(detections)
```
`process_image` returns a list of detections containing each bounding box, confidence score, and class ID.

## Prepare a custom dataset
`prepare_dataset.py` expects source images in these folders:
- `fire/Forest_Fire_BigData/fire_images/`
- `fire/Forest_Fire_BigData/non_fire_images/`

Run:
```bash
python prepare_dataset.py
```
The script creates `dataset/images/` and `dataset/labels/` with train, validation, and test splits, then writes `dataset/data.yaml`. Make sure the source image folders exist and contain `.jpg` images before running it.

## Train a custom model
After preparing the dataset, train using the generated configuration:
```python
from forest_fire_prediction import ForestFireDetector

detector = ForestFireDetector()
detector.train_custom_model(data_yaml_path="dataset/data.yaml", epochs=100)
```
Training requires the dataset images and labels to be present. Adjust the epoch count and training settings to suit your hardware and dataset.

## Other scripts
- `predict.py` provides the separate `FireDetector` helper for image and video inference.
- `realtime_detection.py` contains an additional real-time visualization experiment with OpenCV and sound alerts; it requires the optional runtime assets and a desktop display.

## Notes
- The default model file `yolov8n.pt` is included in the repository.
- Model predictions depend on the weights used. The included pretrained YOLOv8 model is not automatically a fire-specific model; use a fire-trained checkpoint for fire-specific detection.
- Camera access and OpenCV display windows may not work in headless server environments.
