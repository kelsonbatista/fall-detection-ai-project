import os

pose_base = "/home/kelson/Projects/MBA/TCC/fall_detection/experiment1/dataset-xxx/fall50/datasets7-pose/pose"
box_base = "/home/kelson/Projects/MBA/TCC/fall_detection/experiment1/dataset-xxx/fall50/datasets/box"

def count_files(folder):
    return len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])

def compare_directories(pose_dir, box_dir):
    problem_found = False

    for root, dirs, _ in os.walk(pose_dir):
        rel_path = os.path.relpath(root, pose_base)

        if rel_path == ".":
            continue  # skip the root folder

        box_subdir = os.path.join(box_dir, rel_path)

        # Check if the directory exists in the destination
        if not os.path.exists(box_subdir):
            print(f"[ERROR] Directory not found in box: /pose/{rel_path}")
            problem_found = True
            continue

        # Count files
        pose_count = count_files(root)
        box_count = count_files(box_subdir)

        if pose_count != box_count:
            print(f"[ERROR] Difference in the number of files: ")
            print(f"/pose/{rel_path} ({pose_count} files) vs /box/{rel_path} ({box_count} files)")
            problem_found = True

    if not problem_found:
        print("✅ No problems found. All directories and file counts match!")

print("🔍 Comparing structures and file counts...")
compare_directories(pose_base, box_base)

