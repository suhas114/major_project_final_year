import os
import random
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from prepare_dataset import create_yolo_label, prepare_dataset, split_images


class DatasetPreparationTests(unittest.TestCase):
    def test_split_is_deterministic_and_keeps_train_item(self):
        files = [Path(f"image-{i}.jpg") for i in range(10)]
        first = split_images(files, random.Random(42))
        second = split_images(files, random.Random(42))
        self.assertEqual(first, second)
        self.assertEqual(len(first["train"]), 7)
        self.assertEqual(len(first["val"]), 2)
        self.assertEqual(len(first["test"]), 1)

    def test_empty_split_has_three_empty_groups(self):
        self.assertEqual(
            split_images([], random.Random(42)),
            {"train": [], "val": [], "test": []},
        )

    def test_unreadable_image_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "broken.jpg"
            label_dir = Path(temp_dir) / "labels"
            image_path.write_text("not an image", encoding="utf-8")
            label_dir.mkdir()
            with self.assertRaisesRegex(ValueError, "Unable to read image"):
                create_yolo_label(image_path, True, label_dir)

    def test_prepare_dataset_uses_class_prefixes_and_matching_yaml(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fire_dir = root / "fire/Forest_Fire_BigData/fire_images"
            non_fire_dir = root / "fire/Forest_Fire_BigData/non_fire_images"
            fire_dir.mkdir(parents=True)
            non_fire_dir.mkdir(parents=True)
            for folder in (fire_dir, non_fire_dir):
                cv2.imwrite(str(folder / "same.jpg"), np.zeros((8, 8, 3), dtype=np.uint8))
            cv2.imwrite(str(fire_dir / "second.png"), np.ones((8, 8, 3), dtype=np.uint8))

            previous = os.getcwd()
            try:
                os.chdir(root)
                prepare_dataset()
            finally:
                os.chdir(previous)

            all_images = list((root / "dataset/images").rglob("*.*"))
            self.assertEqual(len(all_images), 3)
            self.assertTrue(any(path.name.startswith("fire_") for path in all_images))
            self.assertTrue(any(path.name.startswith("non_fire_") for path in all_images))

            labels = list((root / "dataset/labels").rglob("*.txt"))
            self.assertEqual(len(labels), 3)
            self.assertCountEqual([label.read_text().split()[0] for label in labels], ["0", "0", "1"])

            yaml = (root / "dataset/data.yaml").read_text(encoding="utf-8")
            self.assertIn("names: ['fire', 'non-fire']", yaml)
            self.assertIn("train: images/train", yaml)


if __name__ == "__main__":
    unittest.main()
