import torch
from PIL import Image
from pathlib import Path

model_path = "Implementacion/best.pt"

model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

def detect_objects(image_file):
    try:
        img = Image.open(image_file)

        results = model(img)

        predictions = results.xyxy[0].numpy().tolist()

        filtered_predictions = []

        for pred in predictions:
            if pred[4] >= 0.25:  # Filtrar por confianza mínima del 25%
                keep_box = True
                for existing_pred in filtered_predictions:
                    iou = calculate_iou(pred, existing_pred)
                    if iou > 0.20:  # Filtrar por umbral de superposición del 70%
                        keep_box = False
                        break
                if keep_box:
                    filtered_predictions.append(pred)

        # Formatear las predicciones filtradas en un formato JSON
        objects_detected = []
        for pred in filtered_predictions:
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

    except Exception as e:
        print(f"Error during detection: {e}")
        return {'error': str(e)}

def calculate_iou(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = box1[:4]
    xmin2, ymin2, xmax2, ymax2 = box2[:4]

    inter_xmin = max(xmin1, xmin2)
    inter_ymin = max(ymin1, ymin2)
    inter_xmax = min(xmax1, xmax2)
    inter_ymax = min(ymax1, ymax2)

    inter_area = max(0, inter_xmax - inter_xmin + 1) * max(0, inter_ymax - inter_ymin + 1)

    area1 = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1)
    area2 = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1)

    union_area = area1 + area2 - inter_area

    iou = inter_area / union_area
    return iou
