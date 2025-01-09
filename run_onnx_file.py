import onnxruntime as ort
import numpy as np
import cv2
import os

# Fonction pour préparer l'image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (640, 640))  # Assurez-vous que la taille correspond à l'entraînement
    image_input = image_resized.transpose(2, 0, 1).astype(np.float32) / 255.0  # HWC -> CHW, normalisation
    input_tensor = np.expand_dims(image_input, axis=0).astype(np.float32)  # Ajouter une dimension batch
    return input_tensor

# Fonction pour post-traiter les résultats
def postprocess_outputs(outputs, confidence_threshold=0.5):
    predictions = outputs[0]  # Prendre la sortie principale du modèle
    results = []
    
    # Assurez-vous que confidence est une valeur scalaire
    
    for pred in predictions:
        print("pred",pred[:6])
        # Chaque prédiction contient : [x_min, y_min, x_max, y_max, confidence, class_id]
        
        box = pred[:,:4] # [x_min, y_min, x_max, y_max]
        confidence = pred[:,4] # confidence
        class_id = pred[:,5]# class_id
        
        print("confidence",confidence)
        print("box", box)
        print("class_id",class_id)
        
        # if isinstance(confidence, np.ndarray):
        #     confidence = confidence[0]
        # print("confidence[0]",(confidence))
        # print("x_min",x_min)
        # print("y_min",y_min)
        # print("x_max",x_max)
        # print("y_max",y_max)
        # print("class_id",class_id)
        # if isinstance(confidence, np.ndarray):
        #     confidence = confidence.item()
        
        # if float(confidence) >= confidence_threshold:  # Convertir en float pour la comparaison
        #     result = {
        #         "label": int(class_id),  # Libellé du champ
        #         "confidence": float(confidence),  # Convertir en float
        #         "bbox": box  # Convertir les coordonnées en float
        #     }
        #     results.append(result)
    
    return results

# Fonction pour exécuter l'inférence ONNX
def run_onnx_inference(model_path, image_path):
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    input_tensor = preprocess_image(image_path)
    outputs = session.run(None, {input_name: input_tensor})
    results = postprocess_outputs(outputs, confidence_threshold=0.5)
    return results

# Chemin vers le modèle ONNX
model_path = './results_trains/invoice_yolov8_new_train/weights/best.onnx'

# Chemin vers l'image de test
image_path = "./files/facture/facture_test10.png"

# Exécuter l'inférence et afficher les résultats
results = run_onnx_inference(model_path, image_path)
print(results)