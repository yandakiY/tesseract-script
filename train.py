from ultralytics import YOLO

# Charger un modèle pré-entraîné (par ex. YOLOv8n)
model = YOLO('yolov8n.pt')  # Vous pouvez aussi utiliser yolov8s.pt, yolov8m.pt, etc.

# Entraîner le modèle
model.train(
    data='C:/Users/HP ELITEBOOK 840/Downloads/Annotation-Factures 2.v4i.yolov8/data.yaml',  # Chemin vers le fichier YAML
    epochs=200,                # Nombre d'époques
    imgsz=640,                 # Taille des images
    batch=16,                  # Taille du batch
    name='entrainement_facture1_yolov8_new_field',     # Nom du projet
    project='./results_trains',   # Dossier où sauvegarder les résultats
    device='cpu'                  # GPU (mettez "cpu" si pas de GPU disponible)
)
