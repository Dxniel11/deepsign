# DeepSign: Interfaz Inteligente de Traducción de Lengua de Señas Colombiana (LSC)

DeepSign es un ecosistema de mediación lingüística basado en visión artificial diseñado para entornos escolares. Su objetivo principal es eliminar las barreras de comunicación entre estudiantes con discapacidad auditiva y personas oyentes, transformando las señas capturadas por una cámara web en texto y voz sintética en tiempo real.

---

## Estructura del Proyecto

```text
DeepSign/
│
├── env/                         # Entorno virtual aislado de Python (ignorado en Git)
├── dataset/                     # Directorio raíz del dataset
├── hand_landmarker.task         # Modelo de IA de Google MediaPipe
├── test.py                      # Verificación de cámara y detección de manos
├── crear_dataset.py             # Captura y estructuración del dataset
├── requirements.txt             # Dependencias del proyecto
└── README.md                    # Documentación principal
```

---

## Guía de Despliegue

Si descargas este proyecto en una computadora nueva o deseas volver a ejecutarlo desde cero, sigue los siguientes pasos.

### 1. Clonar el Proyecto

```bash
cd ~/Desktop
git clone https://github.com/TU_USUARIO/DeepSign.git
cd DeepSign
```

> Reemplaza `TU_USUARIO` por tu nombre de usuario de GitHub.

---

### 2. Crear y Activar el Entorno Virtual

#### Git Bash (MINGW64)

```bash
python -m venv env
source env/Scripts/activate
```

#### CMD (Símbolo del Sistema)

```cmd
python -m venv env
env\Scripts\activate
```

Cuando el entorno esté activo aparecerá `(env)` al inicio de la línea de comandos.

---

### 3. Instalar Dependencias

Con el entorno virtual activado, instala todas las librerías necesarias:

```bash
pip install -r requirements.txt
```

---

### 4. Descargar el Modelo de IA

El archivo `hand_landmarker.task` no se almacena en GitHub debido a su tamaño. Descárgalo en la raíz del proyecto ejecutando:

```bash
curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

---

## Uso del Sistema

### Paso A: Validar la Detección de Manos

Verifica que MediaPipe detecte correctamente las articulaciones de la mano en tiempo real.

```bash
python test.py
```

**Acciones:**

* Coloca tu mano frente a la cámara.
* Deberías observar la malla de puntos y conexiones sobre la mano detectada.
* Presiona `q` para cerrar la aplicación.

---

### Paso B: Crear y Alimentar el Dataset

Para registrar nuevas señas y generar muestras para el entrenamiento:

```bash
python crear_dataset.py
```

El sistema creará automáticamente las carpetas necesarias dentro de `dataset/`.

#### Captura de muestras

1. Realiza una seña estática frente a la cámara.
2. Presiona la **barra espaciadora** para guardar una muestra.
3. Captura entre **30 y 50 muestras por seña**, variando ligeramente la distancia y posición de la mano.
4. Presiona `q` para cerrar el programa de forma segura.

---

## Tecnologías Utilizadas

* Python 3.x
* OpenCV
* MediaPipe
* NumPy
* Visión Artificial
* Procesamiento de Lengua de Señas Colombiana (LSC)

---

## Objetivo Social

DeepSign busca promover la inclusión educativa mediante herramientas de inteligencia artificial capaces de facilitar la comunicación entre estudiantes sordos y oyentes, contribuyendo a la construcción de entornos escolares más accesibles y equitativos.
