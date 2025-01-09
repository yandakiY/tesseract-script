import onnxruntime as ort
import numpy as np
import cv2
import pytesseract
from PIL import Image

# Définir le chemin du binaire Tesseract (sur Windows uniquement)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Fonction pour préparer l'image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (640, 640))  # Assurez-vous que la taille correspond à l'entraînement
    image_input = image_resized.transpose(2, 0, 1).astype(np.float32) / 255.0  # HWC -> CHW, normalisation
    input_tensor = np.expand_dims(image_input, axis=0).astype(np.float32)  # Ajouter une dimension batch
    return image, input_tensor

# Fonction pour post-traiter les résultats
def postprocess_outputs(outputs, confidence_threshold=0.5):
    predictions = outputs[0]  # Prendre la sortie principale du modèle
    results = []

    for pred in predictions:
        # Chaque prédiction contient : [x_min, y_min, x_max, y_max, confidence, class_id]
        x_min, y_min, x_max, y_max, confidence, class_id = pred[:6]
        
        # Assurez-vous que confidence est une valeur scalaire
        if isinstance(confidence, np.ndarray):
            confidence = confidence.item()
        
        if confidence >= confidence_threshold:  # Comparaison directe
            result = {
                "label": int(class_id),  # Libellé du champ
                "confidence": float(confidence),  # Convertir en float
                "bbox": [float(x_min), float(y_min), float(x_max), float(y_max)]  # Convertir les coordonnées en float
            }
            results.append(result)
    
    return results

# Fonction pour exécuter l'inférence ONNX
def run_onnx_inference(model_path, image_path):
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    original_image, input_tensor = preprocess_image(image_path)
    outputs = session.run(None, {input_name: input_tensor})
    results = postprocess_outputs(outputs, confidence_threshold=0.5)
    return original_image, results

# Fonction pour extraire le texte des blocs détectés
def extract_text_from_blocks(image, blocks):
    texts = []
    for block in blocks:
        x_min, y_min, x_max, y_max = map(int, block['bbox'])
        cropped_image = image[y_min:y_max, x_min:x_max]
        pil_image = Image.fromarray(cropped_image)
        text = pytesseract.image_to_string(pil_image, lang='fra')  # 'fra' pour le français
        texts.append({
            "label": block['label'],
            "confidence": block['confidence'],
            "text": text
        })
    return texts

# Chemin vers le modèle ONNX
model_path = './results_trains/invoice_yolov8_new_train/weights/best.onnx'

# Chemin vers l'image de test
image_path = "./files/facture/facture_test10.png"

# Exécuter l'inférence et extraire le texte
original_image, detected_blocks = run_onnx_inference(model_path, image_path)
extracted_texts = extract_text_from_blocks(original_image, detected_blocks)

# Afficher les résultats
for item in extracted_texts:
    print(f"Label: {item['label']}, Confidence: {item['confidence']:.2f}")
    print(f"Text: {item['text']}")
    print("-" * 50)