import os

pose_base = "/home/kelson/Projects/MBA/TCC/fall_detection/experiment1/dataset-xxx/fall50/datasets7-pose/pose/labels"
box_base = "/home/kelson/Projects/MBA/TCC/fall_detection/experiment1/dataset-xxx/fall50/datasets/box/labels"

def update_pose_labels_from_box(box_dir, pose_dir):
    files_updated = 0

    # Walk through all files in the box labels directory
    for root, dirs, files in os.walk(box_dir):
        dirs.sort()   # ensure folders are processed in order
        files.sort()  # ensure files are processed in order

        for fname in files:
            if not fname.endswith(".txt"):
                continue

            box_file = os.path.join(root, fname)
            # Equivalent path in pose labels
            rel_path = os.path.relpath(box_file, box_dir)
            pose_file = os.path.join(pose_dir, rel_path)

            if not os.path.exists(pose_file):
                print(f"⚠ Pose file not found: {rel_path}")
                continue

            # Read the box label file
            with open(box_file, "r") as f:
                box_lines = f.readlines()

            # Read the pose label file
            with open(pose_file, "r") as f:
                pose_lines = f.readlines()

            # Warn if the number of lines differ
            if len(box_lines) != len(pose_lines):
                print(f"⚠ Line count mismatch in {rel_path}: box={len(box_lines)}, pose={len(pose_lines)}")

            new_pose_lines = []
            for i in range(min(len(box_lines), len(pose_lines))):
                box_parts = box_lines[i].strip().split()
                pose_parts = pose_lines[i].strip().split()

                if not box_parts or not pose_parts:
                    new_pose_lines.append(pose_lines[i])
                    continue

                # Replace only the first column (class)
                pose_parts[0] = box_parts[0]
                new_pose_lines.append(" ".join(pose_parts) + "\n")

            # Save the updated lines back to the pose file
            with open(pose_file, "w") as f:
                f.writelines(new_pose_lines)
            files_updated += 1

    print(f"✅ Updated {files_updated} pose label files from box labels.")

# Example usage:
update_pose_labels_from_box(box_base, pose_base)

