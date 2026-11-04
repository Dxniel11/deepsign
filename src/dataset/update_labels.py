import os
import json

ruta_raw = "data/raw"
ruta_json = "data/processed/labels.json"

# Leer las carpetas de señas y ordenarlas alfabéticamente
senas = sorted([d for d in os.listdir(ruta_raw) if os.path.isdir(os.path.join(ruta_raw, d))])

# Crear el diccionario numerado
diccionario = {str(i): sena for i, sena in enumerate(senas)}

# Guardar automáticamente en el JSON
os.makedirs(os.path.dirname(ruta_json), exist_ok=True)
with open(ruta_json, "w") as archivo:
    json.dump(diccionario, archivo, indent=4)

print("¡labels.json actualizado con éxito!")
print(json.dumps(diccionario, indent=2))