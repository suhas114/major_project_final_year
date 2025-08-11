import cv2
import numpy as np
import time

def list_available_cameras():
    print("\n=== Checking Available Cameras ===")
    available_cameras = []
    for i in range(10):  # Check first 10 indexes
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Try DirectShow
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"\nCamera {i} is working:")
                print(f"- Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
                print(f"- FPS: {cap.get(cv2.CAP_PROP_FPS)}")
                available_cameras.append(i)
                
                # Display test frame
                cv2.imshow(f'Camera {i} Test', frame)
                cv2.waitKey(1000)  # Show for 1 second
                cv2.destroyAllWindows()
            else:
                print(f"Camera {i} found but cannot read frames")
            cap.release()
        else:
            print(f"Camera {i} not available")
    
    return available_cameras

def test_camera_continuous(camera_index):
    print(f"\n=== Testing Camera {camera_index} Continuously ===")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\nCamera Properties:")
    print(f"- Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"- FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    
    print("\nShowing camera feed for 10 seconds...")
    print("Press 'q' to quit early")
    
    start_time = time.time()
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        frame_count += 1
        elapsed_time = time.time() - start_time
        
        # Calculate and display FPS
        fps = frame_count / elapsed_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Camera Test', frame)
        
        # Check for 'q' key or timeout
        if cv2.waitKey(1) & 0xFF == ord('q') or elapsed_time > 10:
            break
    
    print(f"\nTest completed:")
    print(f"- Frames captured: {frame_count}")
    print(f"- Average FPS: {frame_count/elapsed_time:.1f}")
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    print("Starting camera diagnostic...")
    
    # First, list all available cameras
    available_cameras = list_available_cameras()
    
    if not available_cameras:
        print("\nNo cameras found!")
        return
    
    print(f"\nFound {len(available_cameras)} camera(s) at indices: {available_cameras}")
    
    # Test the first available camera continuously
    if available_cameras:
        test_camera_continuous(available_cameras[0])

if __name__ == "__main__":
    main()
    print("\nPress Enter to exit...")
    input() 