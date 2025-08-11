import os
import shutil
import random
from pathlib import Path
import cv2
import numpy as np

def create_yolo_label(image_path, is_fire, output_path):
    """Create YOLO format label file for an image"""
    img = cv2.imread(str(image_path))
    height, width = img.shape[:2]
    
    # For simplicity, we'll consider the whole image as containing fire/no-fire
    # In a real application, you might want to use object detection to get precise bounding boxes
    label = "0" if is_fire else "1"  # 0 for fire, 1 for non-fire
    
    # Create label file with same name as image but .txt extension
    label_path = output_path / f"{image_path.stem}.txt"
    
    # YOLO format: <class> <x_center> <y_center> <width> <height>
    # We're using the whole image as the bounding box
    with open(label_path, 'w') as f:
        f.write(f"{label} 0.5 0.5 1.0 1.0\n")

def prepare_dataset():
    # Setup paths
    base_dir = Path(".")
    fire_dir = base_dir / "fire/Forest_Fire_BigData/fire_images"
    non_fire_dir = base_dir / "fire/Forest_Fire_BigData/non_fire_images"
    
    dataset_dir = base_dir / "dataset"
    
    # Create necessary directories
    for split in ['train', 'val', 'test']:
        (dataset_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Get all images
    fire_images = list(fire_dir.glob('*.jpg'))
    non_fire_images = list(non_fire_dir.glob('*.jpg'))
    
    # Shuffle images
    random.shuffle(fire_images)
    random.shuffle(non_fire_images)
    
    # Split ratios
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1
    
    def split_and_copy(image_list, is_fire):
        n = len(image_list)
        train_n = int(n * train_ratio)
        val_n = int(n * val_ratio)
        
        # Split the data
        train_images = image_list[:train_n]
        val_images = image_list[train_n:train_n + val_n]
        test_images = image_list[train_n + val_n:]
        
        # Process each split
        for split_name, split_images in [
            ('train', train_images),
            ('val', val_images),
            ('test', test_images)
        ]:
            for img_path in split_images:
                # Copy image
                dest_img = dataset_dir / 'images' / split_name / img_path.name
                shutil.copy2(img_path, dest_img)
                
                # Create and save label
                create_yolo_label(
                    img_path,
                    is_fire,
                    dataset_dir / 'labels' / split_name
                )
    
    # Process both fire and non-fire images
    split_and_copy(fire_images, True)
    split_and_copy(non_fire_images, False)
    
    # Create data.yaml
    data_yaml_content = f"""path: {dataset_dir.absolute()}
train: images/train
val: images/val
test: images/test

nc: 2
names: ['fire', 'non-fire']"""
    
    with open(dataset_dir / 'data.yaml', 'w') as f:
        f.write(data_yaml_content)

if __name__ == "__main__":
    prepare_dataset() 