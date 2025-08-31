import os
import shutil
from pathlib import Path
import yaml
import re

# === Input/Output paths ===
src_root = Path("/home/kelson/Projects/MBA/TCC/fall_detection/experiment1/dataset-xxx/fall50/datasets7-pose/pose")
dst_root = Path("/home/kelson/Projects/MBA/TCC/fall_detection/experiment1/dataset-xxx/fall50/datasets7-pose/pose-balanced")

# Create destination structure
for split in ["train", "val"]:
    for sub in ["images", "labels"]:
        for cls in ["0", "1", "2"]:
            (dst_root / sub / split / cls).mkdir(parents=True, exist_ok=True)

# Counter per class to avoid overwriting
counters = {"0": 0, "1": 0, "2": 0}

def numeric_sort(filename):
    """Extract numbers from filename for natural numeric sorting."""
    numbers = re.findall(r'\d+', filename)
    return int(numbers[-1]) if numbers else -1

def find_image(label_path, split, video_dir):
    """Find corresponding image file (case-insensitive extension)."""
    name = label_path.stem
    img_dir = src_root / "images" / split / video_dir
    for ext in [".png", ".jpg", ".jpeg"]:
        for variant in [ext, ext.upper()]:
            candidate = img_dir / f"{name}{variant}"
            if candidate.exists():
                return candidate
    return None

# Walk through original dataset
for split in ["train", "val"]:
    label_split_dir = src_root / "labels" / split
    for video_dir in sorted(label_split_dir.iterdir()):
        if not video_dir.is_dir():
            continue
        # sort label files numerically
        label_files = sorted(video_dir.glob("*.txt"), key=lambda x: numeric_sort(x.name))
        for label_file in label_files:
            with open(label_file, "r") as f:
                line = f.readline().strip()
                if not line:
                    continue
                cls = line.split()[0]  # first column = class id

            # find corresponding image
            img_file = find_image(label_file, split, video_dir.name)
            if img_file is None:
                print(f"[WARNING] Image not found for {label_file}")
                continue

            # update counters
            counters[cls] += 1
            new_img_name = f"{counters[cls]:06d}{img_file.suffix.lower()}"
            new_label_name = f"{counters[cls]:06d}.txt"

            # destination paths
            dst_img = dst_root / "images" / split / cls / new_img_name
            dst_label = dst_root / "labels" / split / cls / new_label_name

            #shutil.copy2(img_file, dst_img)
            #shutil.copy2(label_file, dst_label)

            shutil.move(str(img_file), str(dst_img))
            shutil.move(str(label_file), str(dst_label))

print("✅ Dataset redistribution completed.")
print("Files per class:", counters)

# === Write new data.yaml ===
data_yaml = {
    "path": "",
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "kpt_shape": [18, 3],
    "flip_idx": [0, 2, 1, 4, 3, 5, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16],
    "names": {0: "no_fall", 1: "fall", 2: "attention"},
    "nc": 3
}

with open(dst_root / "data.yaml", "w") as f:
    yaml.dump(data_yaml, f, sort_keys=False)

print("✅ data.yaml written to", dst_root / "data.yaml")

