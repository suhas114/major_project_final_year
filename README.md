# Forest Fire Detection using YOLOv8

This project implements a forest fire detection system using the YOLOv8 deep learning model. It can detect fires and smoke in both images and video streams, making it useful for early forest fire detection and monitoring.

## Features

- Real-time fire and smoke detection
- Support for both image and video processing
- Custom model training capability
- Easy-to-use Python interface
- Support for webcam and video file inputs

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd forest-fire-detection
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the default detection script using your webcam:

```bash
python forest_fire_detection.py
```

### Using the ForestFireDetector Class

```python
from forest_fire_detection import ForestFireDetector

# Initialize detector
detector = ForestFireDetector()

# Process a single image
results = detector.process_image('path/to/image.jpg')

# Process video stream (0 for webcam, or provide video file path)
detector.process_video(video_source=0)
```

### Training Custom Model

To train a custom model for fire detection:

1. Prepare your dataset in YOLO format
2. Create a data.yaml file with your dataset configuration
3. Train the model:

```python
detector = ForestFireDetector()
detector.train_custom_model(
    data_yaml_path='path/to/data.yaml',
    epochs=100
)
```

## Dataset Preparation

For custom training, prepare your dataset in the following structure:

```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Create a data.yaml file with:

```yaml
path: path/to/dataset
train: images/train
val: images/val
test: images/test

nc: 2  # number of classes
names: ['smoke', 'fire']  # class names
```

## Model Performance

The system uses YOLOv8, which offers a good balance between speed and accuracy. For optimal performance:

- Use good lighting conditions
- Position cameras strategically
- Consider environmental factors (smoke, fog, etc.)
- Adjust confidence thresholds as needed

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 