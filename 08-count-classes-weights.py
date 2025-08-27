import os
from collections import Counter

pose_base = "/home/kelson/Projects/MBA/TCC/fall_detection/experiment1/dataset-xxx/fall50/datasets7-pose/pose-balanced/labels"

def compute_class_weights(pose_dir):
    class_counter = Counter()
    total_files = 0
    total_labels = 0
    
    # Walk through all files in sorted order
    for root, dirs, files in os.walk(pose_dir):
        dirs.sort()
        files.sort()
        for fname in files:
            if not fname.endswith(".txt"):
                continue
            total_files += 1
            fpath = os.path.join(root, fname)
            
            with open(fpath, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls = parts[0]
                        class_counter[cls] += 1
                        total_labels += 1

    print(f"📄 Total label files: {total_files}")
    print(f"📊 Total labels: {total_labels}")
    print("\nClass counts:")
    for cls_id in sorted(class_counter.keys()):
        print(f"Class {cls_id}: {class_counter[cls_id]}")

    # Compute suggested weights (inverse frequency)
    num_classes = len(class_counter)
    weights = {}
    for cls_id, count in class_counter.items():
        weights[cls_id] = total_labels / (num_classes * count)

    print("\nSuggested class weights for training:")
    for cls_id in sorted(weights.keys()):
        print(f"Class {cls_id}: {weights[cls_id]:.4f}")

    return class_counter, weights

def compute_train_only_weights(pose_dir):
    """Compute class weights only for images/train directory"""
    train_labels_dir = os.path.join(pose_dir, "train")
    
    if not os.path.exists(train_labels_dir):
        print(f"❌ Train directory not found: {train_labels_dir}")
        return None, None
        
    class_counter = Counter()
    total_files = 0
    total_labels = 0
    
    print("🚂 TRAIN SET ONLY Analysis:")
    print("=" * 40)
    
    # Walk through train files only
    for root, dirs, files in os.walk(train_labels_dir):
        dirs.sort()
        files.sort()
        for fname in files:
            if not fname.endswith(".txt"):
                continue
            total_files += 1
            fpath = os.path.join(root, fname)
            
            with open(fpath, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls = parts[0]
                        class_counter[cls] += 1
                        total_labels += 1

    print(f"📄 Total TRAIN label files: {total_files}")
    print(f"📊 Total TRAIN labels: {total_labels}")
    print("\nTRAIN Class counts:")
    for cls_id in sorted(class_counter.keys()):
        count = class_counter[cls_id]
        percentage = (count / total_labels) * 100
        print(f"Class {cls_id}: {count:,} ({percentage:.1f}%)")

    # Compute suggested weights (inverse frequency)
    num_classes = len(class_counter)
    weights = {}
    for cls_id, count in class_counter.items():
        weights[cls_id] = total_labels / (num_classes * count)

    print("\nSuggested class weights for TRAIN:")
    for cls_id in sorted(weights.keys()):
        print(f"Class {cls_id}: {weights[cls_id]:.4f}")

    return class_counter, weights

# Example usage - Both analyses:
print("🌟 FULL DATASET Analysis:")
print("=" * 40)
compute_class_weights(pose_base)

print("\n" + "="*60 + "\n")

compute_train_only_weights(pose_base)
