import os
from PIL import Image

base_dir = "/home/kelson/Projects/MBA/TCC/fall_detection/experiment1/dataset-xxx/fall50/datasets7-pose/pose/images"
new_width = 640

train_count = 0
val_count = 0

def resize_directory(dir_path):
    errors = []
    for file in os.listdir(dir_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(dir_path, file)
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    if width > new_width:
                        new_height = int((new_width / width) * height)
                        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
                        resized_img.save(img_path)
            except Exception:
                errors.append(file)
    return errors

# Walk through all directories recursively in sorted order
for root, dirs, files in os.walk(base_dir):
    dirs.sort()  # sort subdirectories ascending
    files.sort()  # sort files ascending

    # Process only directories containing images
    image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if image_files:
        errors = resize_directory(root)
        rel_path = os.path.relpath(root, base_dir)

        # Count train and val directories
        if "train" in rel_path:
            train_count += 1
        elif "val" in rel_path:
            val_count += 1

        if errors:
            print(f"/images/{rel_path} - [ERROR] Files: {', '.join(errors)}")
        else:
            print(f"/images/{rel_path} - [OK]")

print("\n✅ Summary:")
print(f"Total train verified: {train_count}")
print(f"Total val verified: {val_count}")

