from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

class FireDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize Fire Detector
        Args:
            model_path: Path to trained model weights
        """
        self.model = YOLO(model_path)
        
    def detect_from_image(self, image_path):
        """
        Detect fire in a single image
        Args:
            image_path: Path to the image file
        """
        results = self.model(image_path)
        
        # Get the first result (we only processed one image)
        result = results[0]
        
        # Plot the results
        plot = result.plot()
        
        # Display the image
        cv2.imshow('Fire Detection', plot)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return results
    
    def detect_from_video(self, video_source=0):
        """
        Detect fire in video stream
        Args:
            video_source: Camera index or video file path (0 for webcam)
        """
        cap = cv2.VideoCapture(video_source)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Perform detection
            results = self.model(frame)
            
            # Plot results on frame
            annotated_frame = results[0].plot()
            
            # Display the frame
            cv2.imshow('Fire Detection', annotated_frame)
            
            # Break loop with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    # Initialize detector with trained model
    detector = FireDetector()
    
    # Example: Use webcam for detection
    print("Starting video detection (press 'q' to quit)...")
    detector.detect_from_video()
    
    # Example: Detect from image
    # detector.detect_from_image('path/to/your/image.jpg')

if __name__ == "__main__":
    main() 