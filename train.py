from ultralytics import YOLO

# Charger un modèle pré-entraîné (par ex. YOLOv8n)
model = YOLO('yolov8n.pt')  # Vous pouvez aussi utiliser yolov8s.pt, yolov8m.pt, etc.

# Entraîner le modèle
model.train(
    data='C:/Users/HP ELITEBOOK 840/Downloads/Annotation-Factures 2.v1i.yolov8/data.yaml',  # Chemin vers le fichier YAML
    epochs=50,                 # Nombre d'époques
    imgsz=640,                 # Taille des images
    batch=16,                  # Taille du batch
    name='invoice_yolov8',     # Nom du projet
    project='./results_trains',   # Dossier où sauvegarder les résultats
    device='cpu'                  # GPU (mettez "cpu" si pas de GPU disponible)
)
