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
        self.min_temp = 20.0
        self.max_temp = 800.0  # Increased for fire detection
        self.alert_temp = 100.0  # Temperature threshold for fire alert
        self.enable_sound = True
        
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

    def create_fps_graph(self, fps, width=200, height=100):
        if not self.show_fps_graph:
            return None
            
        self.fps_history.append(fps)
        graph = np.ones((height, width, 3), dtype=np.uint8) * 255
        max_fps = max(self.fps_history) if self.fps_history else fps
        max_fps = max(max_fps, 1)  # Avoid division by zero
        points = []
        
        for i, fps_value in enumerate(self.fps_history):
            x = int((i / len(self.fps_history)) * width)
            y = int((1 - (fps_value / max_fps)) * height)
            points.append((x, y))
            
        if len(points) > 1:
            cv2.polylines(graph, [np.array(points)], False, (0, 255, 0), 2)
            
        cv2.putText(graph, f'FPS: {fps:.1f}', (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * self.text_scale, (0, 0, 0), 1)
        return graph
        
    def estimate_temperature(self, frame, box):
        x1, y1, x2, y2 = map(int, box)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return self.min_temp
        
        # Convert to grayscale and calculate average intensity
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        avg_intensity = np.mean(gray_roi)
        
        # Map intensity to temperature range based on sensitivity
        temp_range = self.max_temp - self.min_temp
        sensitivity_factor = self.thermal_sensitivity / 100.0
        estimated_temp = self.min_temp + (avg_intensity / 255.0) * temp_range * sensitivity_factor
        
        return estimated_temp

    def check_fire_conditions(self, frame, box, temp):
        x1, y1, x2, y2 = map(int, box)
        roi = frame[y1:y2, x1:x2]
        
        # Check for fire-like colors
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_fire = np.array([0, 50, 50])
        upper_fire = np.array([25, 255, 255])
        fire_mask = cv2.inRange(hsv, lower_fire, upper_fire)
        fire_ratio = np.sum(fire_mask) / (255 * fire_mask.size)
        
        # Combine temperature and color information
        is_fire = temp > self.alert_temp and fire_ratio > 0.3
        return is_fire, fire_ratio

    def create_thermal_overlay(self, frame, boxes, confidences):
        if not self.show_thermal:
            return frame.copy(), [], []
            
        thermal_base = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thermal_base = cv2.GaussianBlur(thermal_base, (15, 15), 0)
        
        thermal_colored = np.zeros_like(frame)
        temperatures = []
        fire_detections = []
        
        for box in boxes:
            temp = self.estimate_temperature(frame, box)
            temperatures.append(temp)
            
            is_fire, fire_ratio = self.check_fire_conditions(frame, box, temp)
            fire_detections.append(is_fire)
            
            x1, y1, x2, y2 = map(int, box)
            roi = thermal_base[y1:y2, x1:x2]
            
            # Enhanced thermal visualization for fire
            normalized = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX)
            if is_fire:
                # Use intense red for fire regions
                thermal_roi = np.zeros_like(roi.reshape(roi.shape[0], roi.shape[1], 1))
                thermal_roi = np.concatenate([thermal_roi, thermal_roi, normalized.reshape(roi.shape[0], roi.shape[1], 1)], axis=2)
            else:
                thermal_roi = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)
            
            thermal_colored[y1:y2, x1:x2] = thermal_roi
            
            # Play alert sound if fire detected
            if is_fire and self.enable_sound and self.alert_sound:
                self.alert_sound.play()
        
        mask = np.any(thermal_colored != [0, 0, 0], axis=2).astype(np.uint8) * 255
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1) / 255.0
        
        blended = frame.copy()
        blended = frame * (1 - mask * self.thermal_alpha) + thermal_colored * (mask * self.thermal_alpha)
        
        # Add fire warning overlay
        if any(fire_detections):
            warning_color = (0, 0, 255)
            cv2.putText(blended, "FIRE DETECTED!", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, warning_color, 3)
            # Add warning border
            h, w = blended.shape[:2]
            cv2.rectangle(blended, (0, 0), (w-1, h-1), warning_color, 10)
        
        return blended.astype(np.uint8), temperatures, fire_detections

    def get_thermal_color(self, confidence):
        r = int(255 * confidence)
        b = int(255 * (1 - confidence))
        return (b, 0, r)
    
    def create_info_panel(self, frame_count, fps, detections, temperatures, fire_detections):
        if not self.show_info_panel:
            return None
            
        panel = np.ones((400, 300, 3), dtype=np.uint8) * 255
        
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
        cv2.line(panel, (scale_x - 5, threshold_y), (scale_x + scale_width + 5, threshold_y), (0, 0, 255), 2)
        cv2.putText(panel, f"Fire Alert: {self.alert_temp:.0f}°C", 
                    (scale_x + scale_width + 5, threshold_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4 * self.text_scale, (0, 0, 255), 1)
        
        return panel

def run_realtime_detection():
    model = YOLO('yolov8n.pt')
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    visualizer = DetectionVisualizer()
    visualizer.create_control_window()
    
    frame_count = 0
    fps = 0
    start_time = time.time()
    last_fps_update = time.time()
    
    print("Starting real-time fire detection system. Press 'q' to quit.")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame")
            break
            
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - last_fps_update
        
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            last_fps_update = current_time
        
        results = model(frame, conf=visualizer.conf_threshold)
        
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
        
        thermal_frame, temperatures, fire_detections = visualizer.create_thermal_overlay(frame, boxes, confidences)
        
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
        
        fps_graph = visualizer.create_fps_graph(fps)
        if fps_graph is not None:
            h, w = fps_graph.shape[:2]
            thermal_frame[10:10+h, -w-10:-10] = fps_graph
        
        info_panel = visualizer.create_info_panel(frame_count, fps, detections, temperatures, fire_detections)
        if info_panel is not None:
            h, w = info_panel.shape[:2]
            thermal_frame[120:120+h, -w-10:-10] = info_panel
        
        cv2.imshow('Enhanced Fire Detection System', thermal_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == "__main__":
    run_realtime_detection() 