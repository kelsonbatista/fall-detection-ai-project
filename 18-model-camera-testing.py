import os
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from playsound import playsound
import threading

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parâmetros
NUM_CLASSES = 3
OUTPUT_BASE_PATH = "results/fall_detect_yolo11n_pose_balanced"
OUTPUT_DATASET_DIR = f"{OUTPUT_BASE_PATH}/windows/"
OUTPUT_LSTM_MODEL_DIR = f"{OUTPUT_BASE_PATH}/lstm_model"
OUTPUT_LSTM_MODEL_FULL = "lstm_model_full.keras"
OUTPUT_LSTM_MODEL_BEST = "lstm_model_best.keras"
OUTPUT_LSTM_MODEL_HISTORY = "lstm_model_history.json"
VIDEOS_TEST_PATH = "pose-lstm"
YOLO_MODEL_PATH = f"{OUTPUT_BASE_PATH}/yolo11n_pose_train/weights/best.pt"
LSTM_MODEL_PATH = f"{OUTPUT_LSTM_MODEL_DIR}/{OUTPUT_LSTM_MODEL_BEST}"
MODEL_TEST_PATH = f"{OUTPUT_BASE_PATH}/model_test"
os.makedirs(MODEL_TEST_PATH, exist_ok=True)
LABELS = ['no_fall', 'fall', 'attention']
WINDOW = 60

# Verificar se os arquivos de modelo existem
if not os.path.exists(YOLO_MODEL_PATH):
    raise FileNotFoundError(f"Modelo YOLO não encontrado em {YOLO_MODEL_PATH}")
if not os.path.exists(LSTM_MODEL_PATH):
    raise FileNotFoundError(f"Modelo LSTM não encontrado em {LSTM_MODEL_PATH}")
    
model_yolo = YOLO(YOLO_MODEL_PATH)
model_lstm = load_model(LSTM_MODEL_PATH)

def play_alarm():
    try:
        threading.Thread(target=playsound, args=('sounds/sound34.mp3',), daemon=True).start()
    except Exception as e:
        print("Erro ao tocar som:", e)

def extract_box_features(box):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    x_center = x1 + width / 2
    y_center = y1 + height / 2
    area = width * height
    aspect_ratio = width / height if height != 0 else 0
    return np.array([x1, y1, x2, y2, x_center, y_center, width, height, area, aspect_ratio], dtype=np.float32)

def normalize_box_features(features, frame_shape):
    frame_height, frame_width = frame_shape[:2]
    max_area = frame_width * frame_height

    # Normalize os valores com base na dimensão do frame
    normalized = np.array([
        features[0] / frame_width,     # x1
        features[1] / frame_height,    # y1
        features[2] / frame_width,     # x2
        features[3] / frame_height,    # y2
        features[4] / frame_width,     # x_center
        features[5] / frame_height,    # y_center
        features[6] / frame_width,     # width
        features[7] / frame_height,    # height
        features[8] / max_area,        # area
        features[9]                    # aspect_ratio (já é uma razão)
    ], dtype=np.float32)

    return normalized

def extract_pose_features(pose_data, frame_shape):
    keypoints = []
    frame_height, frame_width = frame_shape[:2]
    
    if pose_data is None or len(pose_data) == 0:
        return np.zeros(54, dtype=np.float32)
    
    for x, y, c in pose_data:
        #if c < 0.5: # ignora quando confiança é menor que 50% - evita ruídos
        #    continue
        keypoints.extend([x / frame_width, y / frame_height, c])
    return np.array(keypoints, dtype=np.float32)

def predict_from_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f"Erro ao abrir a câmera")
        return

    window_name = "Detecção de Queda"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    sequence = []
    print("Iniciando detecção em tempo real. Pressione 'q' para sair.")

    try:
        while True:
            valid, frame = cap.read()
            if not valid:
                print("Erro na leitura do frame")
                break
    
            # Verificar se o frame não está vazio
            if frame is None or frame.size == 0:
                print("Frame vazio encontrado")
                continue
            
            # YOLO: detecção de pessoa
            results = model_yolo.predict(frame, verbose=False)
            
            if len(results) == 0 or results[0].boxes is None or results[0].keypoints is None:
                cv2.imshow("Detecção de Queda", frame)
                if cv2.getWindowProperty("Detecção de Queda", cv2.WND_PROP_VISIBLE) < 1:
                    break
                continue
    
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            keypoints = results[0].keypoints.data.cpu().numpy()
    
            if len(boxes) == 0:
                continue
    
            for i, (box, kp) in enumerate(zip(boxes, keypoints)):
                label = classes[i]
                if label not in [0, 1, 2]:
                    continue
    
                box_feat = extract_box_features(box)
                norm_box = normalize_box_features(box_feat, frame.shape)
                pose_feat = extract_pose_features(kp, frame.shape)
                combined_feat = np.concatenate([norm_box, pose_feat])  # 64 features
    
                sequence.append(combined_feat)
    
                if len(sequence) == WINDOW:
                    input_seq = np.expand_dims(np.array(sequence), axis=0)
                    pred = model_lstm.predict(input_seq, verbose=0)
                    class_id = np.argmax(pred)
                    confidence = pred[0][class_id]
                    label_str = LABELS[class_id]
    
                    # Desenho da predição
                    x1, y1, x2, y2 = box.astype(int)
                    # Cor de acordo com a classe
                    if label_str == 'no_fall':
                        color = (255, 0, 0)  # Azul
                    elif label_str == 'attention':
                        color = (0, 255, 255)  # Amarelo
                    elif label_str == 'fall':
                        color = (0, 0, 255)  # Vermelho
                        play_alarm()
                    else:
                        color = (255, 255, 255)  # Branco (fallback)
    
                    # Desenha borda e texto
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'{label_str} ({confidence:.2f})', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
                    #print(f"Predição frame: {frame_count}: {label} - Confiança: {confidence:.2f}")
                    sequence.pop(0)
    
            cv2.imshow(window_name, frame)
            # cv2.imshow("Detecção de Queda (Pressione 'q' para sair)", frame)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print("Erro:", e)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Câmera finalizada.")

# VIDEO DA CÃMERA
predict_from_camera()

