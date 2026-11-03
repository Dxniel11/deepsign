import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'

import cv2
import time
import numpy as np
import mediapipe as mp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "raw")
MODEL_TASK_PATH = os.path.join(PROJECT_ROOT, "models", "hand_landmarker.task")

SEÑA_ACTUAL = "como estas"  
GUARDAR_PATH = os.path.join(DATASET_PATH, SEÑA_ACTUAL)
FRAMES_POR_SEÑA = 30  

if not os.path.exists(GUARDAR_PATH):
    os.makedirs(GUARDAR_PATH, exist_ok=True)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_TASK_PATH),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3
)

cap = cv2.VideoCapture(0)
contador_muestras = len([f for f in os.listdir(GUARDAR_PATH) if f.endswith('.txt')])
last_timestamp = 0

with HandLandmarker.create_from_options(options) as landmarker:
    print(f"\n[INFO] CAPTURA DE SEÑA: '{SEÑA_ACTUAL.upper()}' (126 coords por frame)")
    print("Presiona ESPACIO para grabar. Presiona 'Q' para salir.\n")
    
    grabando = False
    secuencia_frames = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success: continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        timestamp = int(time.time() * 1000)
        if timestamp <= last_timestamp: timestamp = last_timestamp + 1
        last_timestamp = timestamp
        
        resultados = landmarker.detect_for_video(mp_image, timestamp)
        
        vector_manos = []
        if resultados.hand_landmarks:
            for hand_landmarks in resultados.hand_landmarks:
                for landmark in hand_landmarks:
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
                    vector_manos.extend([landmark.x, landmark.y, landmark.z])

        faltantes = 126 - len(vector_manos)
        if faltantes > 0:
            vector_manos.extend([0.0] * faltantes)
        vector_manos = vector_manos[:126]

        if grabando:
            secuencia_frames.append(vector_manos)
            cv2.putText(frame, f"GRABANDO: {len(secuencia_frames)}/{FRAMES_POR_SEÑA}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if len(secuencia_frames) == FRAMES_POR_SEÑA:
                grabando = False
                contador_muestras += 1
                vector_plano = np.array(secuencia_frames).flatten()
                
                nombre_archivo = os.path.join(GUARDAR_PATH, f"muestra_{contador_muestras}_{int(time.time())}.txt")
                np.savetxt(nombre_archivo, vector_plano)
                print(f"[OK] Muestra {contador_muestras} guardada ({len(vector_plano)} valores).")
                secuencia_frames = []

        cv2.putText(frame, f"Sena: {SEÑA_ACTUAL.upper()} | Muestras: {contador_muestras}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if not grabando:
            cv2.putText(frame, "Presiona ESPACIO para grabar", (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        cv2.imshow('DeepSign - Captura', frame)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == 32 and not grabando:
            grabando = True
            secuencia_frames = []
        if tecla == ord('q'): break

cap.release()
cv2.destroyAllWindows()