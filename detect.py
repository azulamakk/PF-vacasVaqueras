import torch
from PIL import Image
from pathlib import Path

# Ruta del archivo de pesos del modelo
model_path = "/Users/azulmakk/Universidad/Proyecto Final/best.pt"

# Cargar el modelo YOLOv5 utilizando los pesos personalizados
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

def detect_objects(image_file):
    # Leer la imagen desde el archivo
    img = Image.open(image_file)

    # Realizar la detección de objetos
    results = model(img)

    # Obtener las predicciones de los objetos detectados
    predictions = results.xyxy[0].numpy().tolist()

    # Formatear las predicciones en un formato JSON
    objects_detected = []
    for pred in predictions:
        obj = {
            'class': int(pred[5]),
            'label': model.names[int(pred[5])],
            'confidence': float(pred[4]),
            'xmin': int(pred[0]),
            'ymin': int(pred[1]),
            'xmax': int(pred[2]),
            'ymax': int(pred[3])
        }
        objects_detected.append(obj)

    return {'objects': objects_detected}
