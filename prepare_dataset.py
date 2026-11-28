import random
import shutil
from pathlib import Path

import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_RATIOS = {"train": 0.7, "val": 0.2}
RANDOM_SEED = 42


def create_yolo_label(image_path, is_fire, output_path):
    """Write a whole-image YOLO label, rejecting unreadable images."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Unable to read image: {image_path}")

    label = "0" if is_fire else "1"  # fire=0, non-fire=1
    label_path = output_path / f"{image_path.stem}.txt"
    with open(label_path, "w", encoding="utf-8") as f:
        f.write(f"{label} 0.5 0.5 1.0 1.0\n")


def split_images(image_list, rng):
    """Return reproducible 70/20/10 splits, keeping a train item when possible."""
    images = sorted(image_list)
    rng.shuffle(images)
    n = len(images)
    if n == 0:
        return {"train": [], "val": [], "test": []}

    train_n = max(1, int(n * SPLIT_RATIOS["train"]))
    val_n = int(n * SPLIT_RATIOS["val"])
    val_n = min(val_n, n - train_n)
    return {
        "train": images[:train_n],
        "val": images[train_n:train_n + val_n],
        "test": images[train_n + val_n:],
    }


def prepare_dataset():
    base_dir = Path(".")
    fire_dir = base_dir / "fire/Forest_Fire_BigData/fire_images"
    non_fire_dir = base_dir / "fire/Forest_Fire_BigData/non_fire_images"
    dataset_dir = base_dir / "dataset"

    missing = [str(path) for path in (fire_dir, non_fire_dir) if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Missing source image directory: " + ", ".join(missing))

    sources = {
        "fire": [p for p in fire_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        "non_fire": [p for p in non_fire_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
    }
    if not sources["fire"] and not sources["non_fire"]:
        raise ValueError("No supported images found in either source directory.")

    # Exclude unreadable files before splitting so they cannot interrupt a run midway.
    for category, images in sources.items():
        valid = []
        for image_path in images:
            if cv2.imread(str(image_path)) is None:
                print(f"Skipping unreadable image: {image_path}")
            else:
                valid.append(image_path)
        sources[category] = valid

    if not sources["fire"] and not sources["non_fire"]:
        raise ValueError("No readable supported images found in either source directory.")

    rng = random.Random(RANDOM_SEED)
    for split in ("train", "val", "test"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for category, images in sources.items():
        split_map = split_images(images, rng)
        is_fire = category == "fire"
        for split_name, split_images_list in split_map.items():
            for img_path in split_images_list:
                # Prefix names to prevent fire/non-fire files with the same name colliding.
                dest_name = f"{category}_{img_path.name}"
                dest_img = dataset_dir / "images" / split_name / dest_name
                dest_label = dataset_dir / "labels" / split_name / f"{Path(dest_name).stem}.txt"
                shutil.copy2(img_path, dest_img)
                create_yolo_label(dest_img, is_fire, dest_label.parent)

    dataset_path = str(dataset_dir.absolute()).replace("\\", "/")
    data_yaml_content = f"""path: "{dataset_path}"
train: images/train
val: images/val
test: images/test

nc: 2
names: ['fire', 'non-fire']
"""
    with open(dataset_dir / "data.yaml", "w", encoding="utf-8") as f:
        f.write(data_yaml_content)


if __name__ == "__main__":
    prepare_dataset()
