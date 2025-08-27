import os
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
import logging
import random

# Configurar logging
logging.basicConfig(level=logging.INFO)

# Configurações de pastas
input_dir = "videos_full"
output_dir = "videos_aug"
os.makedirs(output_dir, exist_ok=True)

# Lista de vídeos suportados
video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')
videos = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(video_extensions)])

print(f"📹 Encontrados {len(videos)} vídeos para processar")

random_aug = 3

# Definir pipelines de augmentation usando Albumentations
augmentation_pipelines = [
    {
        'name': 'flip_brightness_contrast',
        'transform': A.Compose([
            A.HorizontalFlip(p=1.0),
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=1.0),
        ])
    },
    {
        'name': 'flip_hsv',
        'transform': A.Compose([
            A.HorizontalFlip(p=1.0),
            A.HueSaturationValue(hue_shift_limit=(-10, 10), sat_shift_limit=(-20, 20), val_shift_limit=(-15, 15), p=1.0),
        ])
    },
    {
        'name': 'flip_gamma',
        'transform': A.Compose([
            A.HorizontalFlip(p=1.0),
            A.RandomGamma(gamma_limit=(85, 115), p=1.0),
        ])
    },
    {
        'name': 'flip_gauss_noise',
        'transform': A.Compose([
            A.HorizontalFlip(p=1.0),
            A.GaussNoise(std_range=(0.05, 0.15), mean_range=(0.0, 0.0), per_channel=True, p=1.0),
        ])
    },
    {
        'name': 'flip_iso_noise',
        'transform': A.Compose([
            A.HorizontalFlip(p=1.0),
            A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3), p=1.0),
        ])
    },
    {
        'name': 'flip_shadow',
        'transform': A.Compose([
            A.HorizontalFlip(p=1.0),
            A.RandomShadow(shadow_roi=(0.0, 0.3, 1.0, 1.0), num_shadows_limit=(1, 2), 
                          shadow_dimension=5, shadow_intensity_range=(0.3, 0.6), p=1.0),
        ])
    },
    {
        'name': 'flip_clahe',
        'transform': A.Compose([
            A.HorizontalFlip(p=1.0),
            A.CLAHE(clip_limit=(2.0, 4.0), tile_grid_size=(8, 8), p=1.0),
        ])
    },
    {
        'name': 'flip_color_jitter',
        'transform': A.Compose([
            A.HorizontalFlip(p=1.0),
            A.ColorJitter(brightness=(0.9, 1.1), contrast=(0.9, 1.1), 
                         saturation=(0.9, 1.1), hue=(-0.05, 0.05), p=1.0),
        ])
    },
    {
        'name': 'flip_illumination',
        'transform': A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Illumination(mode='linear', intensity_range=(0.02, 0.08), 
                          effect_type='both', angle_range=(0, 360), p=1.0),
        ])
    },
    {
        'name': 'flip_planckian_jitter',
        'transform': A.Compose([
            A.HorizontalFlip(p=1.0),
            A.PlanckianJitter(mode='blackbody', temperature_limit=(4000, 8000), 
                             sampling_method='uniform', p=1.0),
        ])
    },
    {
        'name': 'flip_multiplicative_noise',
        'transform': A.Compose([
            A.HorizontalFlip(p=1.0),
            A.MultiplicativeNoise(multiplier=(0.95, 1.05), per_channel=False, elementwise=False, p=1.0),
        ])
    },
    {
        'name': 'flip_sharpen',
        'transform': A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), method='kernel', p=1.0),
        ])
    },
    {
        'name': 'flip_channel_shuffle',
        'transform': A.Compose([
            A.HorizontalFlip(p=1.0),
            A.ChannelShuffle(p=1.0),
        ])
    },
    {
        'name': 'flip_image_compression',
        'transform': A.Compose([
            A.HorizontalFlip(p=1.0),
            A.ImageCompression(compression_type='jpeg', quality_range=(80, 95), p=1.0),
        ])
    },
    {
        'name': 'flip_tone_curve',
        'transform': A.Compose([
            A.HorizontalFlip(p=1.0),
            A.RandomToneCurve(scale=0.05, per_channel=False, p=1.0),
        ])
    },
    {
        'name': 'flip_to_gray',
        'transform': A.Compose([
            A.HorizontalFlip(p=1.0),
            A.ToGray(p=1.0),
        ])
    },
    {
        'name': 'flip_rgb_shift',
        'transform': A.Compose([
            A.HorizontalFlip(p=1.0),
            A.RGBShift(r_shift_limit=(-15, 15), g_shift_limit=(-15, 15), b_shift_limit=(-15, 15), p=1.0),
        ])
    },
    {
        'name': 'flip_random_fog',
        'transform': A.Compose([
            A.HorizontalFlip(p=1.0),
            A.RandomFog(fog_coef_range=(0.1, 0.3), alpha_coef=0.05, p=1.0),
        ])
    },
    {
        'name': 'flip_spatter_water',
        'transform': A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Spatter(mode='rain', mean=(0.3, 0.5), std=(0.1, 0.2), 
                     cutout_threshold=(0.7, 0.8), intensity=(0.3, 0.5), p=1.0),
        ])
    },
    {
        'name': 'flip_downscale',
        'transform': A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Downscale(scale_range=(0.7, 0.9), 
                       interpolation_pair={'downscale': 0, 'upscale': 1}, p=1.0),
        ])
    },
    {
        'name': 'flip_posterize',
        'transform': A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Posterize(num_bits=(4, 6), p=1.0),
        ])
    },
    {
        'name': 'flip_chromatic_aberration',
        'transform': A.Compose([
            A.HorizontalFlip(p=1.0),
            A.ChromaticAberration(primary_distortion_limit=(-0.01, 0.01), 
                                 secondary_distortion_limit=(-0.02, 0.02), 
                                 mode='green_purple', p=1.0),
        ])
    },
    {
        'name': 'flip_auto_contrast',
        'transform': A.Compose([
            A.HorizontalFlip(p=1.0),
            A.AutoContrast(cutoff=5, method='cdf', p=1.0),
        ])
    },
    {
        'name': 'flip_equalize',
        'transform': A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Equalize(mode='cv', by_channels=True, p=1.0),
        ])
    },
    {
        'name': 'flip_solarize',
        'transform': A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Solarize(threshold_range=(0.7, 0.9), p=1.0),
        ])
    }
]

def load_video_frames(video_path):
    """
    Carrega TODOS os frames do vídeo em formato NumPy para Albumentations
    Retorna: (frames_array, fps, original_size)
    frames_array shape: (N, H, W, C) onde N é número de frames
    """
    cap = cv2.VideoCapture(video_path)
    
    # Obter informações do vídeo
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    #print(f"    📊 Info: {total_frames} frames, {width}x{height}, {fps:.1f} FPS")
    #print(f"    🎬 Processando TODOS os {total_frames} frames...")
    
    frames = []
    frame_count = 0
    
    # Carrega TODOS os frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Albumentations espera RGB (OpenCV usa BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)
        frames.append(frame_rgb)
        frame_count += 1
        
        # Progress indicator a cada 100 frames
        #if frame_count % 100 == 0:
            #print(f"    📼 Carregados {frame_count}/{total_frames} frames...")
    
    cap.release()
    
    if not frames:
        return None, fps, (width, height)
    
    # Converter para array NumPy com shape (N, H, W, C)
    frames_array = np.stack(frames, axis=0)
    
    #print(f"    ✅ {len(frames_array)} frames carregados em memória")
    return frames_array, fps, (width, height)

def save_video_frames(frames_array, output_path, fps):
    """
    Salva array de frames como vídeo
    frames_array shape: (N, H, W, C) em formato RGB
    """
    if frames_array is None or len(frames_array) == 0:
        return False
    
    # Obter dimensões
    num_frames, height, width, channels = frames_array.shape
    
    # Configurar codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"    ❌ Erro: Não foi possível abrir o writer para {output_path}")
        return False
    
    # Converter e salvar cada frame
    for frame_rgb in frames_array:
        # Converter RGB de volta para BGR
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    return True

def apply_albumentations_augmentation(video_path, output_path, pipeline_config):
    """
    Aplica augmentation usando Albumentations em TODOS os frames
    """
    try:
        #print(f"    📥 Carregando vídeo completo...")
        frames_array, fps, original_size = load_video_frames(video_path)
        
        if frames_array is None:
            print(f"    ❌ Erro: Nenhum frame carregado")
            return False
        
        #print(f"    🔄 Aplicando {pipeline_config['name']} em TODOS os {len(frames_array)} frames...")
        
        # Aplicar transformação usando o parâmetro 'images' (plural)
        # Albumentations aplicará as mesmas transformações a todos os frames
        augmented = pipeline_config['transform'](images=frames_array)
        augmented_frames = augmented['images']
        
        #print(f"    💾 Salvando vídeo completo augmentado ({len(augmented_frames)} frames)...")
        success = save_video_frames(augmented_frames, output_path, fps)
        
        if success:
            # Verificar se o arquivo foi salvo corretamente
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:  # Pelo menos 1KB
                return True
            else:
                print(f"    ❌ Arquivo salvo mas muito pequeno ou corrompido")
                return False
        
        return False
        
    except Exception as e:
        print(f"    ❌ Erro na augmentation: {str(e)}")
        return False

# Processamento principal
total_generated = 0
total_failed = 0

print(f"🚀 Iniciando processamento com Albumentations...")
print(f"🎬 PROCESSANDO TODOS OS FRAMES de cada vídeo (sem limitação)")

for vid_idx, vid in enumerate(videos, 1):
    input_path = os.path.join(input_dir, vid)
    print(f"\n📽️  Processando vídeo {vid_idx}/{len(videos)}: {vid}")
    
    # Verifica se o arquivo existe e não está vazio
    if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
        print(f"    ⚠️  Arquivo inválido, pulando...")
        continue

    selected_pipelines = random.sample(augmentation_pipelines, random_aug)
    
    # Aplica cada pipeline de augmentation
    for aug_idx, pipeline in enumerate(selected_pipelines, start=1):
        output_filename = f"{Path(vid).stem}_aug{aug_idx}_{pipeline['name']}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        # Pula se o arquivo já existe
        if os.path.exists(output_path):
            print(f"    ⏭️  Variação {aug_idx} ({pipeline['name']}) já existe, pulando...")
            continue
            
        #print(f"    🎯 Gerando variação {aug_idx}: {pipeline['name']}")
        
        success = apply_albumentations_augmentation(
            input_path, 
            output_path, 
            pipeline
        )
        
        if success:
            #print(f"    ✅ Variação {aug_idx} gerada com sucesso!")
            total_generated += 1
        else:
            print(f"    ❌ Falha na variação {aug_idx}")
            total_failed += 1
            # Remove arquivo vazio/corrompido se foi criado
            if os.path.exists(output_path):
                os.remove(output_path)

print(f"\n🎉 Processamento concluído!")
print(f"✅ Variações geradas com sucesso: {total_generated}")
print(f"❌ Variações que falharam: {total_failed}")
print(f"📁 Arquivos salvos em: {output_dir}")

# Lista arquivos gerados com informações detalhadas
generated_files = [f for f in os.listdir(output_dir) if f.endswith('.mp4')]
#print(f"\n📋 Arquivos gerados ({len(generated_files)}):")

#for f in sorted(generated_files):
#    file_path = os.path.join(output_dir, f)
#    if os.path.exists(file_path):
#        file_size = os.path.getsize(file_path) / (1024*1024)  # MB
#        
#        # Tenta obter informações do vídeo
#        cap = cv2.VideoCapture(file_path)
#        if cap.isOpened():
#            fps = cap.get(cv2.CAP_PROP_FPS)
#            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#            duration = frame_count / fps if fps > 0 else 0
#            cap.release()
#            
#            print(f"  • {f}")
#            print(f"    📊 {file_size:.1f} MB, {frame_count} frames, {width}x{height}, {duration:.1f}s")
#        else:
#            print(f"  • {f} ({file_size:.1f} MB) - ❌ Erro ao ler info")
