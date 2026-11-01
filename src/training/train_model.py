#deep_sign/src/training/train_model.py
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "modelo_lsc.pkl")

X_path = os.path.join(PROCESSED_DATA_PATH, "X.npy")
y_path = os.path.join(PROCESSED_DATA_PATH, "y.npy")
clases_path = os.path.join(PROCESSED_DATA_PATH, "clases.npy")

if not (os.path.exists(X_path) and os.path.exists(y_path)):
    print("[ERROR] Archivos preprocesados no encontrados. Ejecuta 'preprocess.py' primero.")
    exit(1)

X = np.load(X_path)
y = np.load(y_path)
clases = np.load(clases_path)

print(f"--> Cargando dataset: {X.shape[0]} muestras con {X.shape[1]} características.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Entrenando modelo RandomForestClassifier...")
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n==================================================")
print(f" Exactitud del Modelo (Accuracy): {accuracy * 100:.2f}%")
print("==================================================\n")
print(classification_report(y_test, y_pred, target_names=clases))

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
joblib.dump(modelo, MODEL_SAVE_PATH)
print(f"[OK] Modelo guardado correctamente en: {MODEL_SAVE_PATH}")