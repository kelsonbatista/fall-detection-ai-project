import os
import shutil

dataset_root = "/home/kelson/Projects/MBA/TCC/fall_detection/experiment1/dataset-xxx/fall50/datasets7-pose/dataset"

def flatten_and_rename_videos(dataset_dir):
    counter = 1  # starting number for new video names

    # Collect all folders in sorted order using os.walk
    all_folders = []
    for root, dirs, _ in os.walk(dataset_dir):
        dirs.sort()  # sort subdirectories
        for d in dirs:
            folder_path = os.path.join(root, d)
            all_folders.append(folder_path)
    all_folders.sort()

    # Separate folders with "Fall" and "ADL"
    fall_folders = [f for f in all_folders if "Fall" in os.path.basename(f)]
    adl_folders = [f for f in all_folders if "ADL" in os.path.basename(f)]

    # Function to process a list of folders
    def process_folders(folders, counter_start):
        counter = counter_start
        for folder_path in folders:
            # List video files in sorted order
            files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))])
            for file_name in files:
                ext = os.path.splitext(file_name)[1]
                new_name = f"{counter:03d}{ext}"
                src_path = os.path.join(folder_path, file_name)
                dst_path = os.path.join(dataset_dir, new_name)

                shutil.move(src_path, dst_path)
                counter += 1
            rel_folder = os.path.relpath(folder_path, dataset_dir)
            print(f"✅ Processed folder: /dataset/{rel_folder} ({len(files)} files)")
        return counter

    # Process Fall folders first
    counter = process_folders(fall_folders, counter)

    # Then process ADL folders
    counter = process_folders(adl_folders, counter)

    print(f"\n🎉 Finished! Total videos moved: {counter - 1}")

# Usage:
flatten_and_rename_videos(dataset_root)

# After all videos have been moved, remove all subdirectories in the dataset root
for item in os.listdir(dataset_root):
    item_path = os.path.join(dataset_root, item)
    if os.path.isdir(item_path):
        shutil.rmtree(item_path)  # remove the entire folder and its contents



