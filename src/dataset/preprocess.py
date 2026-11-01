#deep_sign/src/dataset/preprocess.py
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed")

os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

x_data = []
y_data = []

clases = sorted([d for d in os.listdir(RAW_DATA_PATH) if os.path.isdir(os.path.join(RAW_DATA_PATH, d))])

if not clases:
    print("[ERROR] No hay clases grabadas en 'data/raw/'. Executa primero 'capture_dataset.py'.")
    exit(1)

print(f"--> Clases encontradas: {clases}")

for idx_clase, nombre_clase in enumerate(clases):
    carpeta_clase = os.path.join(RAW_DATA_PATH, nombre_clase)
    archivos = [f for f in os.listdir(carpeta_clase) if f.endswith('.txt')]
    
    print(f"Procesando '{nombre_clase}' ({len(archivos)} muestras)...")
    
    for archivo in archivos:
        ruta_archivo = os.path.join(carpeta_clase, archivo)
        muestra = np.loadtxt(ruta_archivo)
        
        # Validar dimensión de 1890 características
        if len(muestra) == 1890:
            x_data.append(muestra)
            y_data.append(idx_clase)
        else:
            print(f"[OMITIDO] Arquivo con formato incorrecto ({len(muestra)} valores): {archivo}")

X = np.array(x_data)
y = np.array(y_data)

np.save(os.path.join(PROCESSED_DATA_PATH, "X.npy"), X)
np.save(os.path.join(PROCESSED_DATA_PATH, "y.npy"), y)
np.save(os.path.join(PROCESSED_DATA_PATH, "clases.npy"), np.array(clases))

print("\n[ÉXITO] Preprocesamiento completo:")
print(f" - Forma de X: {X.shape}")
print(f" - Forma de y: {y.shape}")
print(f" - Archivos guardados en: {PROCESSED_DATA_PATH}")