from ultralytics import YOLO
import torch

def train_model():
    # Create a new YOLO model
    model = YOLO('yolov8n.yaml')
    
    # Train the model
    results = model.train(
        data='dataset/data.yaml',
        epochs=2,
        imgsz=640,
        batch=16,
        name='forest_fire_model',
        patience=20,  # Early stopping patience
        save=True,  # Save best model
        device='0' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
    )
    
    # Validate the model
    results = model.val()
    
    print("Training completed! Model saved in 'runs/detect/forest_fire_model'")

if __name__ == "__main__":
    train_model() 