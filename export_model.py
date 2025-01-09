from ultralytics import YOLO

# Charger le modèle entraîné
model = YOLO('./results_trains/invoice_yolov8_new_train/weights/best.pt')

# Exporter le modèle au format ONNX
model.export(format='onnx', opset=11)  # ONNX Opset version (11 ou supérieur)
