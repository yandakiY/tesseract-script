import json
import sys
import cv2
from ultralytics import YOLO
import pytesseract
from pytesseract import Output
import os
from PIL import Image
from paddleocr import PaddleOCR

# Chemin vers l'exécutable Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Chargement du modèle YOLOv8 entrainé
model = YOLO('./results_trains/invoice_yolov8_new_train/weights/best.onnx')  # Chemin absolu vers le modèle YOLO entraîné

labels = [
    "Adresse_Facturation",
    "Adresse_Livraison",
    "Date_Facturation",
    "Echeance",
    "Email_Client",
    "Nom_Client",
    "Numero_Facture",
    "Pourcentage_Remise",
    "Pourcentage_TVA",
    "Produits",
    "Remise",
    "TVA",
    "Tel_Client",
    "Total_Hors_TVA",
    "Total_TTC"
]

def detect_regions(image_path):
    # Charger l'image avec OpenCV
    image = cv2.imread(image_path)
    
    # Effectuer la détection
    results = model(image)
    
    # print('result of image', results[0].boxes)
    
    detections = []
    for box, conf, cls_index in zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls):
        label = labels[int(cls_index)]  # Récupérer le nom de la classe
        xmin, ymin, xmax, ymax = map(int, box.tolist())  # Coordonées absolues
        detections.append({
            "label": label,
            "confidence": conf.item(),
            "box": (xmin, ymin, xmax, ymax)
        })
        
    return detections, image


def preprocess_image(image):
    # Convertir en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Appliquer une binarisation
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Appliquer un filtre pour réduire le bruit
    filtered = cv2.medianBlur(binary, 3)
    
    return filtered

def extract_text(image_path):
    # Détecter les régions
    detections, image = detect_regions(image_path)
    
    # Stocker les résultats dans un dictionnaire
    extracted_data = {}
    overlay = image.copy()
    alpha = 0.1
    
    for detection in detections:
        # Découper chaque région détectée
        xmin, ymin, xmax, ymax = detection['box']
        cropped_region = image[ymin:ymax, xmin:xmax]
        
        # Convertir en image PIL (format attendu par Tesseract)
        cropped_pil = Image.fromarray(cv2.cvtColor(cropped_region, cv2.COLOR_BGR2RGB))
        
        # Extraire le texte avec Tesseract
        extracted_text = pytesseract.image_to_string(cropped_pil)
        
        # Préparer les données extraites
        detection_data = {
            "text": extracted_text.strip(),
            "confidence": round(detection['confidence'], 2),
            "box": {
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax
            }
        }
        
        # Ajouter les données au dictionnaire
        label = detection['label']
        if label not in extracted_data:
            extracted_data[label] = [] # Crée une liste pour chaque nouveau label
        extracted_data[label].append(detection_data)
        
        cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), (0, 255, 0), -1)  # Dessiner un rectangle rempli

        # Ajouter le libellé de la classe prédite en dessous de la zone colorée
        text_position = (xmin, ymax + 20)
        font_scale = 0.3  # Taille du texte réduite
        cv2.putText(image, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
        
        # Dessiner les rectangles sur l'image
        # cv2.rectangle(image,(xmin, ymin),(xmax, ymax),(0, 255, 0),2)
        # cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Appliquer le surlignage avec transparence
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    # Enregistrer l'image avec les rectangles dessinés
    output_image_path = os.path.splitext(image_path)[0] + "_annotated.jpg"
    cv2.imwrite(output_image_path, image)
    
    return extracted_data, output_image_path


# lancement du script par ligne de commande : `py script.py <image_path>`
if __name__ == '__main__':
    
    if len(sys.argv) != 2:
        print("Usage: python script.py <image_path>")
        sys.exit(1)
    
    # Détecter les régions d'intérêt
    json_data , image = extract_text(sys.argv[1])
    print(json.dumps(json_data , indent=4, ensure_ascii=False))
    print(f"Annotated image saved at: {image}")
