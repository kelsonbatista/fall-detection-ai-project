import os

DECIMALS = 6  # casas decimais para coordenadas

def clamp01(x: float) -> float:
    return 1.0 if x > 1.0 else (0.0 if x < 0.0 else x)

def is_visibility_index(i: int) -> bool:
    # Após bbox (i=1..4), keypoints em triplas: x,y,vis => vis quando (i - 5) % 3 == 2
    return i >= 5 and ((i - 5) % 3 == 2)

def verify_and_fix_labels(folder_labels: str):
    files_checked = 0
    issues_by_dir = {}  # {dir_rel: [frame_numbers]}
    corrections = {}    # {full_path: corrected_lines}

    print(f"🔍 Scanning: {os.path.abspath(folder_labels)}")

    for root, dirs, files in os.walk(folder_labels):
        dirs.sort()
        files.sort()

        for fname in files:
            if not fname.endswith(".txt"):
                continue

            files_checked += 1
            full_path = os.path.join(root, fname)
            rel_dir = os.path.relpath(root, folder_labels)  # ex: "train/001"
            frame_num = os.path.splitext(fname)[0].split("_")[-1]  # pega só "000205" ou "96"

            with open(full_path, "r") as f:
                lines = f.readlines()

            changed_any = False
            corrected_lines = []

            for line in lines:
                parts = line.strip().split()
                if not parts:
                    corrected_lines.append(line)
                    continue

                new_parts = []
                for i, tok in enumerate(parts):
                    if i == 0 or is_visibility_index(i):
                        # Classe ou visibilidade → inteiro
                        try:
                            num = float(tok)
                            int_val = int(round(num))
                            if f"{int_val}" != tok:
                                changed_any = True
                            new_parts.append(str(int_val))
                        except ValueError:
                            changed_any = True
                            new_parts.append("0")
                    else:
                        # Coordenadas → float clamp [0,1]
                        try:
                            num = float(tok)
                        except ValueError:
                            num = 0.0
                            changed_any = True
                        clamped = clamp01(num)
                        fmt_val = f"{clamped:.{DECIMALS}f}"
                        if fmt_val != tok:
                            changed_any = True
                        new_parts.append(fmt_val)

                corrected_lines.append(" ".join(new_parts) + "\n")

            if changed_any:
                issues_by_dir.setdefault(rel_dir, []).append(frame_num)
                corrections[full_path] = corrected_lines

    # Relatório resumido por diretório
    for d in sorted(issues_by_dir.keys()):
        frames = ", ".join(issues_by_dir[d])
        print(f"{d}/ - Found inconsistencies in {len(issues_by_dir[d])} files") #: {frames}

    print(f"\n📄 Files checked: {files_checked}")
    print(f"❗ Files needing fixes: {len(corrections)}")

    if corrections:
        ans = input("Do you want to apply fixes to all listed files? (y/n): ").strip().lower()
        if ans == "y":
            for path, new_lines in corrections.items():
                with open(path, "w") as f:
                    f.writelines(new_lines)
            print(f"✅ Fixed {len(corrections)} files.")
        else:
            print("⚠ No changes made.")
    else:
        print("✅ No problems found.")

# Usage
base_dir = "/home/kelson/Projects/MBA/TCC/fall_detection/experiment1/dataset-xxx/fall50/datasets7-pose/pose/labels/"
verify_and_fix_labels(base_dir)

