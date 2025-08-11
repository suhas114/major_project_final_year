from predict import FireDetector
import cv2
import os
from pathlib import Path
import random

def test_on_samples():
    # Initialize the detector with trained model
    detector = FireDetector()
    
    # Get some sample images from test set
    test_dir = Path('dataset/images/test')
    
    # Get all test images
    test_images = list(test_dir.glob('*.jpg'))
    
    # Randomly select 5 images to test (or less if fewer images available)
    num_samples = min(5, len(test_images))
    sample_images = random.sample(test_images, num_samples)
    
    print(f"\nTesting model on {num_samples} random images...")
    
    # Test each sample image
    for img_path in sample_images:
        print(f"\nProcessing image: {img_path.name}")
        
        # Perform detection
        results = detector.detect_from_image(str(img_path))
        
        # Wait for key press before proceeding to next image
        print("Press any key to continue to next image...")
        cv2.waitKey(0)
    
    print("\nTesting completed!")
    print("\nWould you like to:")
    print("1. Test on more images")
    print("2. Try video detection")
    print("3. Test on your own images")

def main():
    print("Starting model testing...")
    test_on_samples()
    
    # After testing samples, offer to test video
    response = input("\nWould you like to test video detection? (y/n): ")
    if response.lower() == 'y':
        print("Starting video detection (press 'q' to quit)...")
        detector = FireDetector()
        detector.detect_from_video()

if __name__ == "__main__":
    main() 