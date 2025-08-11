import cv2

def test_webcam():
    print("Starting webcam test...")
    
    # Try different camera indices
    for camera_index in [0, 1, 2]:
        print(f"\nTrying camera index {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Could not open camera {camera_index}")
            continue
            
        print(f"Successfully opened camera {camera_index}")
        print(f"Width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
        print(f"Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        
        # Try to read a frame
        ret, frame = cap.read()
        if ret:
            print("Successfully captured a frame")
            # Display the frame
            cv2.imshow(f'Camera Test (Index {camera_index})', frame)
            cv2.waitKey(2000)  # Show for 2 seconds
        else:
            print("Failed to capture frame")
        
        cap.release()
        cv2.destroyAllWindows()
        
    print("\nWebcam test completed. Press Enter to exit.")
    input()

if __name__ == "__main__":
    test_webcam() 