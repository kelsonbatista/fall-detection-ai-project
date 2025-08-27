import os
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
import random
import shutil

# ==== SETTINGS ====
class_id = 2   # <-- change manually (1 = fall, 2 = attention)

input_dir = f"pose-balanced/images/train/{class_id}"
label_dir = f"pose-balanced/labels/train/{class_id}"
output_dir = input_dir  # save augmented images in the same folder

# Configuração por classe (baseado nos dados reais)
if class_id == 1:
    current_real_count = 3987   # Contagem real da classe 1
    augmentations_per_image = 1  # 3987 + (3987*2) = ~7,9K
    target_count = current_real_count + (current_real_count * augmentations_per_image)
elif class_id == 2:
    current_real_count = 880    # Contagem real da classe 2  
    augmentations_per_image = 9  # 880 + (880*9) = ~8.8K
    target_count = current_real_count + (current_real_count * augmentations_per_image)
else:
    raise ValueError("Este script é apenas para classes 1 e 2")

# Collect image files (mudança: PNG em vez de vídeos)
image_extensions = ('.png', '.jpg', '.jpeg')
images = [f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)]
images.sort()  # Ordenar em ordem crescente

# Current dataset stats
current_count = len(images)
to_generate = target_count - current_count
print(f"📊 Class {class_id}: {current_count} images found, need +{to_generate} augmentations")

# Definir pipelines de augmentation usando Albumentations
augmentation_pipelines = [
    {
        'name': 'flip_light',
        'transform': A.Compose([
            A.HorizontalFlip(p=1.0),
            A.RandomBrightnessContrast(0.15, 0.15, p=1.0)
        ])
    },
    {
        'name': 'flip_hue',
        'transform': A.Compose([
            A.HorizontalFlip(p=1.0),
            A.HueSaturationValue(10, 20, 10, p=1.0)
        ])
    },
    {
        'name': 'light_blur',
        'transform': A.Compose([
            A.MotionBlur(blur_limit=3, p=1.0),
            A.RandomBrightnessContrast(0.1, 0.1, p=1.0)
        ])
    },
    {
        'name': 'gaussian_noise',
        'transform': A.Compose([
            A.GaussNoise(var_limit=(10.0, 30.0), p=1.0)
        ])
    },
    {
        'name': 'lowlight',
        'transform': A.Compose([
            A.RandomGamma(gamma_limit=(60, 120), p=1.0)
        ])
    },
    {
        'name': 'foggy',
        'transform': A.Compose([
            A.RandomFog(fog_coef_range=(0.1, 0.2), alpha_coef=0.08, p=1.0)
        ])
    },
    {
        'name': 'rainy',
        'transform': A.Compose([
            A.RandomRain(slant_range=(-10, 10), drop_length=20, drop_width=2,
                         drop_color=(180, 180, 180), blur_value=3, brightness_coefficient=0.9, p=1.0)
        ])
    },
    {
        'name': 'snow_light',
        'transform': A.Compose([
            A.RandomSnow(snow_point_range=(0.1, 0.2), brightness_coeff=1.5, p=1.0)
        ])
    },
    {
        'name': 'sunflare',
        'transform': A.Compose([
            A.RandomSunFlare(flare_roi=(0.1, 0, 0.9, 0.3),
                             num_flare_circles_range=(3, 6), src_radius=100, src_color=(255, 220, 150), p=1.0)
        ])
    },
    {
        'name': 'lens_distortion',
        'transform': A.Compose([
            A.OpticalDistortion(distort_limit=0.15, shift_limit=0.05, p=1.0)
        ])
    },
    {
        'name': 'channel_shift',
        'transform': A.Compose([
            A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1.0)
        ])
    },
    {
        'name': 'contrast_boost',
        'transform': A.Compose([
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0)
        ])
    },
    {
        'name': 'affine_shift',
        'transform': A.Compose([
            A.Affine(translate_percent=(0.05, 0.1), rotate=(-5, 5), scale=(0.9, 1.1), p=1.0)
        ])
    },
    {
        'name': 'slight_rotate',
        'transform': A.Compose([
            A.Rotate(limit=5, p=1.0)
        ])
    },
    {
        'name': 'jitter_colors',
        'transform': A.Compose([
            A.ColorJitter(0.2, 0.2, 0.2, 0.05, p=1.0)
        ])
    },
    {
        'name': 'compression',
        'transform': A.Compose([
            A.ImageCompression(quality_lower=30, quality_upper=70, p=1.0)
        ])
    },
    {
        'name': 'downscale',
        'transform': A.Compose([
            A.Downscale(scale_min=0.5, scale_max=0.75, interpolation=cv2.INTER_NEAREST, p=1.0)
        ])
    },
    {
        'name': 'gray_noise',
        'transform': A.Compose([
            A.ToGray(p=1.0),
            A.GaussNoise(var_limit=(5.0, 15.0), p=1.0)
        ])
    },
    {
        'name': 'sharpen',
        'transform': A.Compose([
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.7, 1.0), p=1.0)
        ])
    },
]

# Função adaptada para imagens (em vez de vídeos)
def apply_augmentation(image_path, output_path, pipeline):
    try:
        # Carregar imagem
        image = cv2.imread(image_path)
        if image is None:
            return False
        
        # Converter para RGB (padrão Albumentations)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Aplicar augmentação
        augmented = pipeline['transform'](image=image_rgb)
        
        # Converter de volta e salvar
        image_bgr = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
        return cv2.imwrite(output_path, image_bgr)
    except:
        return False

# ==== MAIN LOOP ====
generated = 0

for img_idx, img in enumerate(images, 1):
    if generated >= to_generate:
        break

    input_path = os.path.join(input_dir, img)
    base_name = Path(img).stem

    # Choose up to N random pipelines (mesmo comportamento original)
    selected_pipelines = random.sample(augmentation_pipelines, min(augmentations_per_image, len(augmentation_pipelines)))

    for aug_idx, pipeline in enumerate(selected_pipelines, 1):
        if generated >= to_generate:
            break

        out_name = f"{base_name}_aug{aug_idx}.png"  # PNG em vez de mp4
        output_path = os.path.join(output_dir, out_name)

        if os.path.exists(output_path):
            continue

        if apply_augmentation(input_path, output_path, pipeline):
            # Copy corresponding YOLO label file
            label_in = os.path.join(label_dir, f"{base_name}.txt")
            label_out = os.path.join(label_dir, f"{Path(out_name).stem}.txt")
            if os.path.exists(label_in):
                shutil.copy(label_in, label_out)

            generated += 1
            if generated % 50 == 0:  # Progress a cada 50 imagens
                print(f"✅ Generated {generated}/{to_generate}: {out_name}")
        else:
            print(f"❌ Failed: {img} + {pipeline['name']}")

print(f"\n🎉 Finished! Generated {generated} new images for class {class_id}")
