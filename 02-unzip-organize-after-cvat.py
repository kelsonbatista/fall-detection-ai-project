import os
import shutil
import zipfile

base_dir = '/home/kelson/Projects/MBA/TCC/fall_detection/experiment1/dataset-xxx/fall50/datasets7-pose/zip'
sets = ['train', 'Validation', 'val', 'test']

# 1. Descompacta os arquivos .zip numericamente nomeados
for file in os.listdir(base_dir):
    if file.endswith('.zip') and file[:-4].isdigit():
        zip_path = os.path.join(base_dir, file)
        extract_dir = os.path.join(base_dir, file[:-4])
        print(f"Descompactando {file}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

# Função para garantir que todos os diretórios necessários existam
def ensure_dirs():
    for main in ['images', 'labels']:
        os.makedirs(os.path.join(base_dir, main), exist_ok=True)

ensure_dirs()

# Função para buscar diretório com nome correspondente (case-insensitive)
def find_case_insensitive_path(parent, target):
    if not os.path.exists(parent):
        return None
    for item in os.listdir(parent):
        if item.lower() == target.lower():
            return os.path.join(parent, item)
    return None

# Itera sobre diretórios numerados (ex: 001, 002, ...)
for item in os.listdir(base_dir):
    item_path = os.path.join(base_dir, item)
    if not os.path.isdir(item_path) or not item.isdigit():
        continue  # pula se não for um diretório numérico

    for subset in sets:
        # Localiza diretórios de origem com case-insensitive
        images_src = find_case_insensitive_path(os.path.join(item_path, 'images'), subset)
        labels_src = find_case_insensitive_path(os.path.join(item_path, 'labels'), subset)

        renamed_images_dir = os.path.join(item_path, 'images', item)
        renamed_labels_dir = os.path.join(item_path, 'labels', item)

        images_dst_dir = os.path.join(base_dir, 'images', subset, item)
        labels_dst_dir = os.path.join(base_dir, 'labels', subset, item)

        # Renomear diretórios temporariamente se existirem
        if images_src and os.path.exists(images_src):
            os.rename(images_src, renamed_images_dir)
            os.makedirs(images_dst_dir, exist_ok=True)
            for filename in os.listdir(renamed_images_dir):
                shutil.move(os.path.join(renamed_images_dir, filename),
                            os.path.join(images_dst_dir, filename))

        if labels_src and os.path.exists(labels_src):
            os.rename(labels_src, renamed_labels_dir)
            os.makedirs(labels_dst_dir, exist_ok=True)
            for filename in os.listdir(renamed_labels_dir):
                shutil.move(os.path.join(renamed_labels_dir, filename),
                            os.path.join(labels_dst_dir, filename))

    shutil.rmtree(item_path, ignore_errors=True)

# 5. Remove os arquivos .zip numéricos
for file in os.listdir(base_dir):
    if file.endswith('.zip') and file[:-4].isdigit():
        os.remove(os.path.join(base_dir, file))
        print(f"Removido {file}")

print("Organização finalizada com sucesso.")
