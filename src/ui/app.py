#deepsign/src/ui/app.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['ABSL_LOGGING_MIN_LOG_LEVEL'] = '3'

import cv2
import time
import joblib
import threading
import queue
import pyttsx3
import numpy as np
from PIL import Image
from collections import deque
import customtkinter as ctk
import mediapipe as mp
import win32com.client
import pythoncom

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

MODEL_PKL_PATH = os.path.join(PROJECT_ROOT, "models", "modelo_lsc.pkl")
CLASSES_NPY_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "clases.npy")
MODEL_TASK_PATH = os.path.join(PROJECT_ROOT, "models", "hand_landmarker.task")

FRAMES_POR_SEÑA = 30
UMBRAL_CONFIANZA = 0.85
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class DeepSignApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("DeepSign - Traductor de Lengua de Señas Colombiana (LSC)")
        self.geometry("1000x640")
        self.resizable(True, True) 
        self.after(0, lambda: self.state('zoomed'))

        self._cargar_recursos()

        # Cola e Hilo dedicado para Text-to-Speech (TTS)
        self.tts_queue = queue.Queue()
        self.tts_running = True
        self.thread_tts = threading.Thread(target=self._bucle_tts, daemon=True)
        self.thread_tts.start()

        self.is_running = False
        self.camera_index = 0
        self.last_timestamp = 0
        self.buffer_secuencia = deque(maxlen=FRAMES_POR_SEÑA)
        
        self.texto_traduccion = ""
        self.ultima_sena = ""
        self.thread_camara = None

        self.confianza_actual = 0.0
        self.sena_evaluada = "Esperando..."

        self._crear_interfaz()
        self.protocol("WM_DELETE_WINDOW", self._al_cerrar)

    def _cargar_recursos(self):
        for path, nombre in [(MODEL_PKL_PATH, "modelo_lsc.pkl"),
                             (CLASSES_NPY_PATH, "clases.npy"),
                             (MODEL_TASK_PATH, "hand_landmarker.task")]:
            if not os.path.exists(path):
                print(f"[ERROR CRÍTICO] No se encontró '{nombre}' en:\n  -> {path}")
                exit(1)

        self.modelo = joblib.load(MODEL_PKL_PATH)
        self.clases = np.load(CLASSES_NPY_PATH)

    def _bucle_tts(self):
        """ Bucle en hilo secundario nativo de Windows para voz (SAPI5) sin bloqueos. """
        pythoncom.CoInitialize()
        try:
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            while self.tts_running:
                try:
                    texto = self.tts_queue.get(timeout=0.2)
                    if texto:
                        speaker.Speak(texto)
                    self.tts_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"[ERROR TTS]: {e}")
        finally:
            pythoncom.CoUninitialize()

    def _hablar(self, texto):
        if texto.strip():
            self.tts_queue.put(texto)

    def _crear_interfaz(self):
        top_frame = ctk.CTkFrame(self, fg_color="transparent")
        top_frame.pack(fill="x", padx=20, pady=10)

        title_label = ctk.CTkLabel(
            top_frame, 
            text="🖐️ DeepSign - Traductor de Lengua de Señas Colombiana (LSC)", 
            font=("Helvetica", 18, "bold")
        )
        title_label.pack(side="left")

        self.status_label = ctk.CTkLabel(
            top_frame, 
            text="🔴 [ Desconectado ]", 
            text_color="red", 
            font=("Helvetica", 14, "bold")
        )
        self.status_label.pack(side="right")

        main_container = ctk.CTkFrame(self, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=20, pady=5)

        # Se asigna un ancho y alto fijo al panel izquierdo (Cámara) y se desactiva pack_propagate
        left_card = ctk.CTkFrame(main_container, corner_radius=15, fg_color="#1E1E1E", width=680, height=520)
        left_card.pack(side="left", fill="both", expand=True, padx=(0, 10))
        left_card.pack_propagate(False)

        cam_title = ctk.CTkLabel(left_card, text="VISTA DE CÁMARA", text_color="white", font=("Helvetica", 14, "bold"))
        cam_title.pack(pady=(10, 2))

        cam_subtitle = ctk.CTkLabel(left_card, text="(Secuencia dinámica de 30 frames)", text_color="gray", font=("Helvetica", 11))
        cam_subtitle.pack(pady=(0, 5))

        self.video_label = ctk.CTkLabel(left_card, text="Cámara Apagada", text_color="white")
        self.video_label.pack(fill="both", expand=True, padx=10, pady=10)

        # Se asigna un ancho fijo al panel derecho (Traducción) y se desactiva pack_propagate
        right_card = ctk.CTkFrame(main_container, corner_radius=15, fg_color="#1E293B", width=400, height=520)
        right_card.pack(side="right", fill="both", expand=True, padx=(10, 0))
        right_card.pack_propagate(False)

        right_title = ctk.CTkLabel(right_card, text="TRADUCCIÓN EN TIEMPO REAL", text_color="#333333", font=("Helvetica", 14, "bold"))
        right_title.pack(pady=(15, 5))

        speaker_status = ctk.CTkLabel(right_card, text="🔊 [ Altavoz activo ]", text_color="#1F6AA5", font=("Helvetica", 12))
        speaker_status.pack(anchor="w", padx=20, pady=(0, 5))

        self.frame_metricas = ctk.CTkFrame(right_card, fg_color="transparent")
        self.frame_metricas.pack(fill="x", padx=20, pady=(5, 2))

        self.lbl_sena_actual = ctk.CTkLabel(
            self.frame_metricas, 
            text="Detectando...", 
            font=("Helvetica", 13, "bold"), 
            text_color="#38BDF8"
        )
        self.lbl_sena_actual.pack(side="left")

        self.lbl_porcentaje = ctk.CTkLabel(
            self.frame_metricas, 
            text="0%", 
            font=("Helvetica", 13, "bold"), 
            text_color="gray"
        )
        self.lbl_porcentaje.pack(side="right")

        self.progressbar = ctk.CTkProgressBar(
            right_card, 
            height=10, 
            corner_radius=5, 
            progress_color="#10B981"
        )
        self.progressbar.pack(fill="x", padx=20, pady=(0, 10))
        self.progressbar.set(0.0)

        self.text_box = ctk.CTkTextbox(right_card, font=("Helvetica", 18, "bold"), fg_color="#0F172A", text_color="#F8FAFC", corner_radius=10, wrap="word")
        self.text_box.pack(fill="both", expand=True, padx=20, pady=5)

        btn_listen = ctk.CTkButton(
            right_card, 
            text="🔊 Volver a escuchar", 
            command=self._repetir_voz,
            fg_color="#2563EB",
            text_color="white",
            hover_color="#1D4ED8"
        )
        btn_listen.pack(fill="x", padx=20, pady=(5, 5))

        btn_clear = ctk.CTkButton(
            right_card, 
            text="🗑️ Limpiar texto", 
            command=self._limpiar_texto,
            fg_color="#EF4444", 
            text_color="white", 
            hover_color="#DC2626"
        )
        btn_clear.pack(fill="x", padx=20, pady=(0, 15))

        # Panel Inferior
        controls_frame = ctk.CTkFrame(self, fg_color="transparent")
        controls_frame.pack(fill="x", pady=15)

        center_controls = ctk.CTkFrame(controls_frame, fg_color="transparent")
        center_controls.pack(anchor="center")

        btn_start = ctk.CTkButton(center_controls, text="[ Iniciar ]", fg_color="#10B981", hover_color="#059669", width=120, command=self.iniciar_camara)
        btn_start.pack(side="left", padx=5)

        btn_stop = ctk.CTkButton(center_controls, text="[ Detener ]", fg_color="#EF4444", hover_color="#DC2626", width=120, command=self.detener_camara)
        btn_stop.pack(side="left", padx=5)

        btn_switch = ctk.CTkButton(center_controls, text="[ Cambiar Cámara ]", fg_color="#2563EB", hover_color="#1D4ED8", width=140, command=self.cambiar_camara)
        btn_switch.pack(side="left", padx=5)

    def iniciar_camara(self):
        if not self.is_running:
            self.is_running = True
            self.status_label.configure(text="🟢 [ Conectado ]", text_color="green")
            self.thread_camara = threading.Thread(target=self._bucle_camara, daemon=True)
            self.thread_camara.start()

    def detener_camara(self):
        self.is_running = False
        self.status_label.configure(text="🔴 [ Desconectado ]", text_color="#EF4444")
        
        # Imagen con el color de fondo exacto del panel (#1E1E1E -> RGB 30, 30, 30)
        img_oscura = Image.new("RGB", (450, 320), (30, 30, 30))
        ctk_img_oscura = ctk.CTkImage(light_image=img_oscura, dark_image=img_oscura, size=(450, 320))
        
        self.video_label.configure(image=ctk_img_oscura, text="Cámara Apagada")
        self.video_label.image = ctk_img_oscura

        self.buffer_secuencia.clear()
        self.progressbar.set(0.0)
        self.lbl_porcentaje.configure(text="0%")
        self.lbl_sena_actual.configure(text="Detectando...")

    def cambiar_camara(self):
        self.detener_camara()
        self.camera_index = 1 if self.camera_index == 0 else 0
        self.after(500, self.iniciar_camara)

    def _repetir_voz(self):
        texto_actual = self.text_box.get("1.0", "end").strip()
        if texto_actual:
            self._hablar(texto_actual)

    def _limpiar_texto(self):
        self.texto_traduccion = ""
        self.ultima_sena = ""
        self.text_box.delete("1.0", "end")

    def _bucle_camara(self):
        cap = cv2.VideoCapture(self.camera_index)

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_TASK_PATH),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3
        )

        with HandLandmarker.create_from_options(options) as landmarker:
            while self.is_running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                timestamp = int(time.time() * 1000)
                if timestamp <= self.last_timestamp:
                    timestamp = self.last_timestamp + 1
                self.last_timestamp = timestamp

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = landmarker.detect_for_video(mp_image, timestamp)

                vector_frame = []
                mano_detectada = False

                if detection_result.hand_landmarks:
                    mano = detection_result.hand_landmarks[0]
                    mano_detectada = True
                    for landmark in mano:
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        cv2.circle(rgb_frame, (x, y), 4, (0, 255, 255), -1)
                        vector_frame.extend([landmark.x, landmark.y, landmark.z])

                # SOLO agregar al buffer si realmente hay una mano detectada
                if mano_detectada and len(vector_frame) == 63:
                    self.buffer_secuencia.append(vector_frame)
                else:
                    # Si retiras la mano, reseteamos el buffer e inferencia
                    self.buffer_secuencia.clear()
                    self.confianza_actual = 0.0
                    self.sena_evaluada = "Buscando mano..."
                    self.ultima_sena = ""

                nueva_sena_detectada = self._procesar_inferencia()

                img_pil = Image.fromarray(rgb_frame)
                self.after(0, self._actualizar_ui, img_pil, nueva_sena_detectada)

                time.sleep(0.01)

        cap.release()

    def _procesar_inferencia(self):
        nueva_sena = None
        if len(self.buffer_secuencia) == FRAMES_POR_SEÑA:
            vector_entrada = np.array(self.buffer_secuencia).flatten().reshape(1, -1)

            probabilidades = self.modelo.predict_proba(vector_entrada)[0]
            max_idx = np.argmax(probabilidades)
            confianza = float(probabilidades[max_idx])

            self.confianza_actual = confianza
            self.sena_evaluada = str(self.clases[max_idx]).upper()

            if confianza >= UMBRAL_CONFIANZA:
                if self.sena_evaluada != self.ultima_sena:
                    self.ultima_sena = self.sena_evaluada
                    nueva_sena = self.sena_evaluada
            else:
                self.ultima_sena = ""
        
        return nueva_sena

    def _actualizar_ui(self, img_pil, nueva_sena_detectada):
        import time # Lo importamos aquí para asegurar que el temporizador funcione
        
        if self.is_running:
            ctk_img = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(780, 580))
            self.video_label.configure(image=ctk_img, text="")
            self.video_label.image = ctk_img

            # Inicializamos la memoria y el temporizador
            if not hasattr(self, 'ultima_sena_registrada'):
                self.ultima_sena_registrada = ""
                self.tiempo_ultima_sena = 0.0

            tiempo_actual = time.time()

            if nueva_sena_detectada and nueva_sena_detectada.lower() != "neutro":
                # Escribe si la seña es diferente a la anterior, 
                # O si es la misma pero ya pasaron más de 2 segundos
                if (nueva_sena_detectada != self.ultima_sena_registrada) or (tiempo_actual - self.tiempo_ultima_sena > 2.0):
                    self.texto_traduccion += f"{nueva_sena_detectada} "
                    self.text_box.insert("end", f"{nueva_sena_detectada} ")
                    self.text_box.see("end")
                    self._hablar(nueva_sena_detectada)
                    
                    # Actualizamos la memoria y el reloj
                    self.ultima_sena_registrada = nueva_sena_detectada
                    self.tiempo_ultima_sena = tiempo_actual

            # Eliminamos el 'else' problemático para que los parpadeos de cámara no afecten

            porcentaje = int(self.confianza_actual * 100)
            self.progressbar.set(self.confianza_actual)
            self.lbl_porcentaje.configure(text=f"{porcentaje}%")
            self.lbl_sena_actual.configure(text=f"Seña: {self.sena_evaluada}")

            if self.confianza_actual >= UMBRAL_CONFIANZA:
                self.progressbar.configure(progress_color="#10B981")
            elif self.confianza_actual >= 0.40:
                self.progressbar.configure(progress_color="#F59E0B")
            else:
                self.progressbar.configure(progress_color="#EF4444")

    def _al_cerrar(self):
        self.is_running = False
        self.tts_running = False
        self.destroy()


if __name__ == "__main__":
    app = DeepSignApp()
    app.mainloop()