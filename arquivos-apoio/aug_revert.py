import os
import re
from pathlib import Path

# ==== SETTINGS ====
class_id = 1  # <-- change manually (1 = fall, 2 = attention)

# Diretórios
image_dir = f"pose-balanced/images/train/{class_id}"
label_dir = f"pose-balanced/labels/train/{class_id}"

# Padrão para identificar arquivos augmentados
# Exemplo: 000002_aug1_foggy.png ou 000002_aug2_flip_light.png
augmented_pattern = re.compile(r'.*_aug\d+.(png|jpg|jpeg|txt)$', re.IGNORECASE)

def cleanup_directory(directory, file_type):
    """Remove arquivos augmentados de um diretório"""
    if not os.path.exists(directory):
        print(f"⚠️ Directory not found: {directory}")
        return 0
    
    files = os.listdir(directory)
    removed_count = 0
    
    for file in files:
        if augmented_pattern.match(file):
            file_path = os.path.join(directory, file)
            try:
                os.remove(file_path)
                removed_count += 1
                if removed_count % 100 == 0:  # Progress every 100 files
                    print(f"🗑️ Removed {removed_count} {file_type} files...")
            except Exception as e:
                print(f"❌ Failed to remove {file}: {str(e)}")
    
    return removed_count

# Confirmar antes de executar
print(f"🚨 This will delete ALL augmented files from class {class_id}")
print(f"📂 Image directory: {image_dir}")
print(f"📂 Label directory: {label_dir}")
print(f"🔍 Pattern: files containing '_aug[number]_' in the name")

confirm = input("\nAre you sure? Type 'YES' to continue: ")

if confirm.upper() == 'YES':
    print("\n🧹 Starting cleanup...")
    
    # Limpar imagens
    removed_images = cleanup_directory(image_dir, "image")
    print(f"✅ Removed {removed_images} image files from {image_dir}")
    
    # Limpar labels
    removed_labels = cleanup_directory(label_dir, "label")
    print(f"✅ Removed {removed_labels} label files from {label_dir}")
    
    print(f"\n🎉 Cleanup completed!")
    print(f"📊 Total files removed: {removed_images + removed_labels}")
    
else:
    print("❌ Cleanup cancelled.")
