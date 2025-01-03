import json
import cv2
from ultralytics import YOLO
import pytesseract
from pytesseract import Output
import os
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Charger le modèle YOLOv8
model = YOLO('./results_trains/invoice_yolov8/weights/best.pt')  # Chemin vers le modèle YOLO entraîné

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

    # Affichage des détections
    # for detection in detections:
    #     print(f"Classe : {detection['label']}, Confiance : {detection['confidence']:.2f}, Boîte : {detection['box']}")
        
    return detections, image


def extract_text(image_path):
    # Détecter les régions
    detections, image = detect_regions(image_path)
    
    # Stocker les résultats dans un dictionnaire
    extracted_data = {}
    
    for detection in detections:
        # Découper chaque région détectée
        xmin, ymin, xmax, ymax = detection['box']
        cropped_region = image[ymin:ymax, xmin:xmax]
        
        # Convertir en image PIL (format attendu par Tesseract)
        cropped_pil = Image.fromarray(cv2.cvtColor(cropped_region, cv2.COLOR_BGR2RGB))
        
        # Extraire le texte avec Tesseract
        extracted_text = pytesseract.image_to_string(cropped_pil)  # utilisez 'fra'
        
        # Préparer les données extraites
        detection_data = {
            "text": extracted_text.strip(),
            "confidence": round(detection['confidence'], 2),
            "box": detection['box']
        }
        
        # Ajouter les données au dictionnaire
        label = detection['label']
        if label not in extracted_data:
            extracted_data[label] = []  # Crée une liste pour chaque nouveau label
        extracted_data[label].append(detection_data)
    
    # Afficher les résultats pour chaque label
    # for label, entries in extracted_data.items():
    #     print(f"Label : {label}")
    #     for entry in entries:
    #         print(f"  Confiance : {entry['confidence']:.2f}")
    #         print(f"  Texte extrait : {entry['text']}")
    #         print(f"  Boîte : {entry['box']}\n")
    
    return extracted_data


def main():
    # Chemin de l'image à traiter
    image_path = './files/facture/facture_test4.jpg'
    
    # Détecter les régions d'intérêt
    # print(extract_text(image_path))
    json_data = extract_text(image_path)
    print(json.dumps(json_data , indent=4, ensure_ascii=False))
    

main()
