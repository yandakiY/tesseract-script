import json
import sys
import cv2
from ultralytics import YOLO
import pytesseract
from PIL import Image
import easyocr



# Chemin vers l'exécutable Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Chargement du modèle YOLOv8 entrainé
model = YOLO('./results_trains/entrainement_facture1_yolov8_new_field2/weights/best.pt')  # Chemin absolu vers le modèle YOLO entraîné

labels = [
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

def detect_regions(image_path):
    # Charger l'image avec OpenCV
    image = cv2.imread(image_path)
    
    # Effectuer la détection
    results = model(image)
    
    detections = []
    for box, conf, cls_index in zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls):
        label = labels[int(cls_index)] 
        xmin, ymin, xmax, ymax = map(int, box.tolist())
        detections.append({
            "label": label,
            "confidence": conf.item(),
            "box": (xmin, ymin, xmax, ymax)
        })
        
    return detections, image

def preprocess_image(image):
    # Convertir en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Appliquer une débruitage
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # Améliorer le contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # Binarisation adaptative
    binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    
    # Appliquer une dilatation pour améliorer la lisibilité
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    processed = cv2.dilate(binary, kernel, iterations=1)
    
    return processed

def extract_text(image_path):
    # Détecter les régions
    detections, original_image = detect_regions(image_path)
    
    # Stocker les résultats dans un dictionnaire
    extracted_data = {}
    
    for detection in detections:
        # Découper chaque région détectée
        xmin, ymin, xmax, ymax = detection['box']
        cropped_region = original_image[ymin:ymax, xmin:xmax]
        
        # Prétraiter l'image
        processed_region = preprocess_image(cropped_region)
        
        # Configuration Tesseract
        custom_config = r'--oem 3 --psm 6'
        
        
        # Convertir en image PIL (format attendu par Tesseract)
        # cropped_pil = Image.fromarray(cv2.cvtColor(cropped_region, cv2.COLOR_BGR2RGB))
        label = detection['label']
        
        # Extraire le texte avec Tesseract
        extracted_text = pytesseract.image_to_string(
            processed_region, 
            config=custom_config,
            lang='fra'
        )
        # image_for_easyocr = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2RGB)
        # reader = easyocr.Reader(['fr']) # this needs to run only once to load the model into memory
        # results = reader.readtext(image_for_easyocr , detail=0 , paragraph=True)
        # print("resut of extraction", results , " for label", label)
        # extracted_text = results[0] if results else ""
        
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
        #label = detection['label']
        if label not in extracted_data:
            extracted_data[label] = [] # Crée une liste pour chaque nouveau label
        extracted_data[label].append(detection_data)
        
    
    return extracted_data


# lancement du script par ligne de commande : `py script.py <image_path>`
if __name__ == '__main__':
    
    if len(sys.argv) != 2:
        print("Usage: python script.py <image_path>")
        sys.exit(1)
    
    # Détecter les régions d'intérêt
    json_data , image = extract_text(sys.argv[1])
    print(json.dumps(json_data , indent=4, ensure_ascii=False))
    print(f"Annotated image saved at: {image}")
