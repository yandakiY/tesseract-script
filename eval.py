from ultralytics import YOLO

# Charger le modèle YOLOv8 entraîné
model = YOLO('./results_trains/invoice_yolov8_new_train/weights/best.onnx')

# Chemin vers le fichier YAML de l'ensemble de données de validation
data_yaml = 'C:/Users/HP ELITEBOOK 840/Downloads/Annotation-Factures 2.v1i.yolov8 (2)/data.yaml'

# Évaluer le modèle
results = model.val(data=data_yaml, imgsz=640, batch=16, device='cpu')

# Afficher les résultats
#print("mAP_50:", results.metrics.mAP50)
# print("mAP_50_95:", results['metrics/mAP_50_95'])
# print("Precision:", results['metrics/precision'])
# print("Recall:", results['metrics/recall'])
print("Results:", results.box.map50)
print("Precision globale:", results.box.mp)
print("Precisions:", results.box.p)
# print("Results:", results.box.recall)