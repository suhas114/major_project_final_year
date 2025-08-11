import cv2
import torch
from ultralytics import YOLO
import time
import numpy as np
from collections import deque
import datetime
import pygame  # For alert sounds

class DetectionVisualizer:
    def __init__(self, max_points=100):
        self.fps_history = deque(maxlen=max_points)
        self.detection_history = deque(maxlen=max_points)
        self.max_points = max_points
        self.start_time = time.time()
        
        # UI Control parameters
        self.conf_threshold = 0.25
        self.thermal_alpha = 0.3
        self.show_fps_graph = True
        self.show_info_panel = True
        self.show_thermal = True
        self.show_boxes = True
        self.box_thickness = 2
        self.text_scale = 1.0
        self.thermal_sensitivity = 50
        self.min_temp = 500.0
        self.max_temp = 800.0  # Increased for fire detection
        self.alert_temp = 100.0  # Temperature threshold for fire alert
        self.enable_sound = True
        self.panel_height = 360  # Fixed panel height
        self.panel_width = 300   # Fixed panel width
        
        # Initialize sound alert
        pygame.init()
        pygame.mixer.init()
        self.alert_sound = pygame.mixer.Sound("alert.wav") if self.enable_sound else None

    def create_control_window(self):
        cv2.namedWindow('Controls')
        cv2.resizeWindow('Controls', 400, 800)
        
        # Add trackbars
        cv2.createTrackbar('Confidence Threshold', 'Controls', 25, 100, 
                          lambda x: self._update_conf_threshold(x/100))
        cv2.createTrackbar('Thermal Opacity', 'Controls', 30, 100, 
                          lambda x: self._update_thermal_alpha(x/100))
        cv2.createTrackbar('Box Thickness', 'Controls', 2, 10, 
                          lambda x: self._update_box_thickness(x))
        cv2.createTrackbar('Text Scale', 'Controls', 10, 30, 
                          lambda x: self._update_text_scale(x/10))
        cv2.createTrackbar('Thermal Sensitivity', 'Controls', 50, 100,
                          lambda x: self._update_thermal_sensitivity(x))
        cv2.createTrackbar('Alert Temperature', 'Controls', 100, 1000,
                          lambda x: self._update_alert_temp(x))
        
        # Add toggle buttons
        cv2.createTrackbar('Show FPS Graph', 'Controls', 1, 1, 
                          lambda x: self._update_show_fps_graph(bool(x)))
        cv2.createTrackbar('Show Info Panel', 'Controls', 1, 1, 
                          lambda x: self._update_show_info_panel(bool(x)))
        cv2.createTrackbar('Show Thermal', 'Controls', 1, 1, 
                          lambda x: self._update_show_thermal(bool(x)))
        cv2.createTrackbar('Show Boxes', 'Controls', 1, 1, 
                          lambda x: self._update_show_boxes(bool(x)))
        cv2.createTrackbar('Enable Sound', 'Controls', 1, 1,
                          lambda x: self._update_enable_sound(bool(x)))

    # Callback methods for trackbars
    def _update_conf_threshold(self, value): self.conf_threshold = value
    def _update_thermal_alpha(self, value): self.thermal_alpha = value
    def _update_box_thickness(self, value): self.box_thickness = value
    def _update_text_scale(self, value): self.text_scale = value
    def _update_show_fps_graph(self, value): self.show_fps_graph = value
    def _update_show_info_panel(self, value): self.show_info_panel = value
    def _update_show_thermal(self, value): self.show_thermal = value
    def _update_show_boxes(self, value): self.show_boxes = value
    def _update_thermal_sensitivity(self, value): self.thermal_sensitivity = value
    def _update_alert_temp(self, value): self.alert_temp = float(value)
    def _update_enable_sound(self, value): 
        self.enable_sound = value
        if not value and self.alert_sound:
            self.alert_sound.stop()

    def create_info_panel(self, frame_count, fps, detections, temperatures, fire_detections):
        if not self.show_info_panel:
            return None
            
        panel = np.ones((self.panel_height, self.panel_width, 3), dtype=np.uint8) * 255
        
        # Add timestamp and basic stats
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(panel, timestamp, (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * self.text_scale, (0, 0, 0), 1)
        
        cv2.putText(panel, f"Frame: {frame_count}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * self.text_scale, (0, 0, 0), 1)
        cv2.putText(panel, f"FPS: {fps:.1f}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * self.text_scale, (0, 0, 0), 1)
        cv2.putText(panel, f"Detections: {len(detections)}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * self.text_scale, (0, 0, 0), 1)
        
        # Add detection information with fire status
        y_pos = 120
        for i, (detection, temp, is_fire) in enumerate(zip(detections, temperatures, fire_detections)):
            if y_pos + i*20 >= self.panel_height - 30:  # Ensure text stays within panel
                break
            class_name, conf = detection
            status_color = (0, 0, 255) if is_fire else (0, 0, 0)
            status_text = "FIRE!" if is_fire else "Normal"
            cv2.putText(panel, f"{class_name}: {temp:.1f}°C - {status_text}", 
                       (10, y_pos + i*20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5 * self.text_scale, 
                       status_color, 1)
        
        # Add thermal scale with fire threshold
        scale_width = 30
        scale_height = 200
        scale_x = 240
        scale_y = 50
        
        if scale_y + scale_height <= self.panel_height:  # Ensure scale fits within panel
            for i in range(scale_height):
                t = 1 - (i / scale_height)
                color = cv2.applyColorMap(np.array([[int(t * 255)]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
                cv2.line(panel, (scale_x, scale_y + i), (scale_x + scale_width, scale_y + i), 
                        (int(color[0]), int(color[1]), int(color[2])), 1)
            
            # Add temperature labels and fire threshold
            cv2.putText(panel, f"{self.max_temp:.0f}°C", (scale_x + scale_width + 5, scale_y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4 * self.text_scale, (0, 0, 0), 1)
            cv2.putText(panel, f"{self.min_temp:.0f}°C", (scale_x + scale_width + 5, scale_y + scale_height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4 * self.text_scale, (0, 0, 0), 1)
            
            # Add fire threshold line
            threshold_y = int(scale_y + scale_height * (1 - (self.alert_temp - self.min_temp) / (self.max_temp - self.min_temp)))
            if 0 <= threshold_y <= self.panel_height:
                cv2.line(panel, (scale_x - 5, threshold_y), (scale_x + scale_width + 5, threshold_y), (0, 0, 255), 2)
                cv2.putText(panel, f"Fire Alert: {self.alert_temp:.0f}°C", 
                            (scale_x + scale_width + 5, threshold_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4 * self.text_scale, (0, 0, 255), 1)
        
        return panel 

    def get_thermal_color(self, value):
        """Convert a value (0-1) to a thermal color using cv2's COLORMAP_JET."""
        color_value = int(value * 255)
        color_map = cv2.applyColorMap(np.array([[color_value]], dtype=np.uint8), cv2.COLORMAP_JET)
        return tuple(map(int, color_map[0, 0]))

    def create_thermal_overlay(self, frame, boxes, confidences):
        """Create thermal visualization overlay with temperature estimation."""
        if not self.show_thermal:
            return frame.copy(), [], []

        overlay = frame.copy()
        temperatures = []
        fire_detections = []
        
        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2 = map(int, box)
            
            # Calculate temperature based on confidence and sensitivity
            # Map confidence (0-1) to temperature range using thermal sensitivity
            temp_range = self.max_temp - self.min_temp
            base_temp = self.min_temp + (conf * temp_range * (self.thermal_sensitivity / 100))
            
            # Extract region for color analysis
            region = frame[y1:y2, x1:x2]
            if region.size > 0:
                # Check for fire-like colors in HSV space
                hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
                # Define fire color ranges (red-orange-yellow)
                lower_fire = np.array([0, 120, 150])
                upper_fire = np.array([35, 255, 255])
                fire_mask = cv2.inRange(hsv_region, lower_fire, upper_fire)
                fire_ratio = np.sum(fire_mask) / (255 * region.size)
                
                # Adjust temperature based on fire color presence
                if fire_ratio > 0.3:  # If significant fire-like colors detected
                    temp_boost = 100 * fire_ratio  # Boost temperature based on fire color ratio
                    base_temp += temp_boost
            
            temperatures.append(base_temp)
            is_fire = base_temp >= self.alert_temp
            fire_detections.append(is_fire)
            
            if is_fire and self.enable_sound and self.alert_sound:
                self.alert_sound.play()
            
            # Apply thermal coloring to the region
            color = self.get_thermal_color(conf)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        
        # Blend the thermal overlay with original frame
        output = cv2.addWeighted(overlay, self.thermal_alpha, frame, 1 - self.thermal_alpha, 0)
        return output, temperatures, fire_detections

    def create_fps_graph(self, current_fps):
        """Create a rolling FPS graph visualization."""
        if not self.show_fps_graph:
            return None
            
        self.fps_history.append(current_fps)
        
        # Create graph
        graph_height = 50
        graph_width = 100
        graph = np.ones((graph_height, graph_width, 3), dtype=np.uint8) * 255
        
        # Draw FPS points
        max_fps = max(self.fps_history) if self.fps_history else current_fps
        min_fps = min(self.fps_history) if self.fps_history else 0
        fps_range = max(1, max_fps - min_fps)  # Avoid division by zero
        
        points = []
        for i, fps in enumerate(self.fps_history):
            x = int((i / len(self.fps_history)) * graph_width)
            y = int(graph_height - ((fps - min_fps) / fps_range) * graph_height)
            points.append((x, y))
        
        # Draw lines connecting points
        if len(points) > 1:
            cv2.polylines(graph, [np.array(points)], False, (0, 255, 0), 1)
        
        # Add current FPS value
        cv2.putText(graph, f"FPS: {current_fps:.1f}", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        return graph

def run_realtime_detection():
    print("Initializing fire detection system...")
    
    # Load the model
    print("Loading YOLO model...")
    model = YOLO('yolov8n.pt')
    
    # Initialize webcam with DirectShow backend
    print("Initializing camera...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("Error: Could not open camera!")
        return
        
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print(f"Camera initialized:")
    print(f"- Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"- FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    
    # Initialize visualizer and create control window
    print("Setting up visualization...")
    visualizer = DetectionVisualizer()
    visualizer.create_control_window()
    
    # Initialize counters
    frame_count = 0
    fps = 0
    start_time = time.time()
    last_fps_update = time.time()
    
    print("\nStarting real-time fire detection system. Press 'q' to quit.")
    print("If you don't see the camera feed, try:")
    print("1. Checking if other applications are using the camera")
    print("2. Unplugging and plugging back in your webcam")
    print("3. Restarting your computer")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Error: Failed to grab frame!")
            break
            
        # Update FPS
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - last_fps_update
        
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            last_fps_update = current_time
        
        # Run detection with current confidence threshold
        results = model(frame, conf=visualizer.conf_threshold)
        
        # Get detection information
        detections = []
        boxes = []
        confidences = []
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = result.names[cls]
                
                detections.append((class_name, conf))
                boxes.append([x1, y1, x2, y2])
                confidences.append(conf)
        
        # Create thermal visualization
        thermal_frame, temperatures, fire_detections = visualizer.create_thermal_overlay(frame, boxes, confidences)
        
        # Draw detection boxes and labels
        if visualizer.show_boxes:
            for (box, conf, temp, is_fire, (class_name, _)) in zip(boxes, confidences, temperatures, fire_detections, detections):
                x1, y1, x2, y2 = map(int, box)
                color = (0, 0, 255) if is_fire else visualizer.get_thermal_color(conf)
                cv2.rectangle(thermal_frame, (x1, y1), (x2, y2), color, 
                            visualizer.box_thickness)
                label = f'{class_name} {conf:.2f} ({temp:.1f}°C)'
                if is_fire:
                    label += " FIRE!"
                cv2.putText(thermal_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5 * visualizer.text_scale, color, 
                           visualizer.box_thickness)
        
        # Add FPS Graph if enabled
        fps_graph = visualizer.create_fps_graph(fps)
        if fps_graph is not None:
            h, w = fps_graph.shape[:2]
            thermal_frame[10:10+h, -w-10:-10] = fps_graph
        
        # Add info panel if enabled
        info_panel = visualizer.create_info_panel(frame_count, fps, detections, temperatures, fire_detections)
        if info_panel is not None:
            h, w = info_panel.shape[:2]
            thermal_frame[120:120+h, -w-10:-10] = info_panel
        
        # Display the frame
        cv2.imshow('Enhanced Fire Detection System', thermal_frame)
        
        # Check for quit command
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    print("\nClosing application...")
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == "__main__":
    run_realtime_detection() 