import onnxruntime as ort
import numpy as np
import cv2
import os

# Vérifier si le fichier ONNX existe
onnx_model_path = './results_trains/invoice_yolov8_new_train/weights/best.onnx'
if not os.path.exists(onnx_model_path):
    raise FileNotFoundError(f"Le fichier ONNX n'existe pas à l'emplacement spécifié : {onnx_model_path}")

session = ort.InferenceSession(onnx_model_path)

# Préparer une image
image_path = "./files/facture/facture_test10.png"
image = cv2.imread(image_path)
image_resized = cv2.resize(image, (640, 640))  # Assurez-vous que la taille correspond à l'entraînement
image_input = image_resized.transpose(2, 0, 1).astype(np.float32) / 255.0  # HWC -> CHW, normalisation
image_input = np.expand_dims(image_input, axis=0)

# Exécuter une prédiction
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: image_input})

# Afficher les résultats
print("Output des resultats",outputs)
