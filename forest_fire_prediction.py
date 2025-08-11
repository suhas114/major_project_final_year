from ultralytics import YOLO
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import time
from pathlib import Path
from predict import FireDetector

class ForestFireDetector:
    def __init__(self, model_path=None):
        """
        Initialize the Forest Fire Detector
        Args:
            model_path (str): Path to custom trained YOLO model. If None, uses pretrained model
        """
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
        else:
            # Load pretrained YOLOv8 model
            self.model = YOLO('yolov8n.pt')
        
        # Classes we're interested in (default COCO classes)
        self.fire_related_classes = {
            'smoke': 0,
            'fire': 1,
        }
    
    def process_image(self, image_path):
        """
        Process a single image for fire detection
        Args:
            image_path (str): Path to the image
        Returns:
            dict: Detection results with confidence scores
        """
        # Perform prediction
        results = self.model(image_path)
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                bbox = box.xyxy[0].tolist()
                
                detections.append({
                    'bbox': bbox,
                    'confidence': confidence,
                    'class_id': class_id
                })
        
        return detections

    def process_video(self, video_source=0, display=True):
        """
        Process video stream for fire detection
        Args:
            video_source: Camera index or video file path
            display (bool): Whether to display the video feed
        """
        cap = cv2.VideoCapture(video_source)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Perform detection
            results = self.model(frame)
            
            # Draw results on frame
            annotated_frame = results[0].plot()
            
            if display:
                cv2.imshow('Forest Fire Detection', annotated_frame)
                
                # Break loop with 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()

    def train_custom_model(self, data_yaml_path, epochs=100):
        """
        Train a custom YOLOv8 model for fire detection
        Args:
            data_yaml_path (str): Path to data.yaml file
            epochs (int): Number of training epochs
        """
        # Create a new YOLO model
        model = YOLO('yolov8n.yaml')
        
        # Train the model
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=640,
            batch=16,
            name='forest_fire_model'
        )
        
        return results

def main():
    # Initialize detector
    detector = ForestFireDetector()
    
    # Example: Process video from webcam
    print("Starting video detection (press 'q' to quit)...")
    detector.process_video()

    # Test on specific image (uncomment and set path to use)
    # result = detector.process_image('path/to/your/image.jpg')
    # print(result)

    # Test with video file (uncomment and set path to use)
    # detector.process_video('path/to/video.mp4')

if __name__ == "__main__":
    main()