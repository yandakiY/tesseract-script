
from datetime import datetime
import os
import cv2

LABELS = [
    'Adresse_Facturation',
    'Adresse_Livraison',
    'Date_Facturation',
    'Devise',
    'Echeance',
    'Email_Client',
    'Fournisseur',
    'Nom_Client',
    'Numero_Facture',
    'Pourcentage_Remise',
    'Pourcentage_TVA',
    'Produits',
    'Remise',
    'TVA',
    'Tel_Client',
    'Total_Hors_TVA',
    'Total_TTC',
    'site_web'
]

annotations = {
    "TVA": {
        "x": 652,
        "y": 652,
        "width": 44,
        "height": 22,
        "confidence": 0.93
    },
    "Numero_Facture": {
        "x": 641,
        "y": 230,
        "width": 62,
        "height": 26,
        "confidence": 0.92
    },
    "Echeance": {
        "x": 625,
        "y": 308,
        "width": 77,
        "height": 24,
        "confidence": 0.92
    },
    "Produits": {
        "x": 53,
        "y": 496,
        "width": 644,
        "height": 96,
        "confidence": 0.91
    },
    "Adresse_Livraison": {
        "x": 295,
        "y": 394,
        "width": 126,
        "height": 39,
        "confidence": 0.9
    },
    "Pourcentage_TVA": {
        "x": 511,
        "y": 652,
        "width": 43,
        "height": 21,
        "confidence": 0.89
    },
    "Total_Hors_TVA": {
        "x": 645,
        "y": 622,
        "width": 51,
        "height": 23,
        "confidence": 0.88
    },
    "Adresse_Facturation": {
        "x": 51,
        "y": 396,
        "width": 90,
        "height": 37,
        "confidence": 0.84
    },
    "Total_TTC": {
        "x": 575,
        "y": 696,
        "width": 119,
        "height": 40,
        "confidence": 0.8
    },
    "Fournisseur": {
        "x": 48,
        "y": 257,
        "width": 88,
        "height": 21,
        "confidence": 0.72
    },
    "Nom_Client": {
        "x": 51,
        "y": 377,
        "width": 92,
        "height": 17,
        "confidence": 0.7
    },
    "Devise": {
        "x": 668,
        "y": 697,
        "width": 25,
        "height": 35,
        "confidence": 0.66
    },
    "Date_Facturation": {
        "x": 621,
        "y": 256,
        "width": 83,
        "height": 25,
        "confidence": 0.65
    }
}

LABELS_FOLDER = "C:/Users/HP ELITEBOOK 840/Downloads/Annotation-Factures 2.v4i.yolov8/train/labels"
IMAGE_FOLDER = "C:/Users/HP ELITEBOOK 840/Downloads/Annotation-Factures 2.v4i.yolov8/train/images"
# Fonction pour convertir les annotations en format YOLO

def process_image_and_annotations(image_path, dimensions, annotations):
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Vérifie si les dossiers existent sinon les créer
    os.makedirs(IMAGE_FOLDER, exist_ok=True)
    os.makedirs(LABELS_FOLDER, exist_ok=True)

    # Récupérer le nom de l'image sans extension
    image_name = os.path.basename(image_path)
    image_basename = timestamp+"_"+os.path.splitext(image_name)[0]
    image_extension = os.path.splitext(image_name)[1]


    # Chemin du fichier .txt de sortie
    txt_filename = f"{image_basename}.txt"
    txt_path = os.path.join(LABELS_FOLDER, txt_filename)

    yolo_lines = []
    
    
    for label, data in annotations.items():
        if label in LABELS:
            class_id = LABELS.index(label)  # Obtenir l'index de la classe

            # Conversion des coordonnées en format YOLO (normalisé)
            x_center = (data["x"] + data["width"] / 2) / dimensions["width"]
            y_center = (data["y"] + data["height"] / 2) / dimensions["height"]
            width_norm = data["width"] / dimensions["width"]
            height_norm = data["height"] / dimensions["height"]

            # Ajout de logs pour vérifier les valeurs
            print(f"Label: {label}, x: {data['x']}, y: {data['y']}, width: {data['width']}, height: {data['height']}")
            print(f"Normalized values - x_center: {x_center}, y_center: {y_center}, width_norm: {width_norm}, height_norm: {height_norm}")

            # Formater la ligne au format YOLO
            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}"
            yolo_lines.append(yolo_line)


    # Écriture du fichier .txt
    with open(txt_path, "w") as f:
        f.write("\n".join(yolo_lines))

    print(f"✅ Fichier {txt_filename} enregistré dans {LABELS_FOLDER}")

    # Redimensionner et enregistrer l'image
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (dimensions["width"], dimensions["height"]))
    resized_image_path = os.path.join(IMAGE_FOLDER, f"{image_basename}{image_extension}")
    cv2.imwrite(resized_image_path, resized_image)

    print(f"✅ Image redimensionnée enregistrée dans {resized_image_path}")

    return txt_path, resized_image_path


# dimensions = {"width": 750, "height": 1061}
# process_image_and_annotations("telechargement (6).png", dimensions ,annotations)