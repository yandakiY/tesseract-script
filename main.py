import json
import logging
from typing import Dict
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from script import detect_regions, extract_text
import uvicorn as uvicorn
import os
from datetime import datetime
from models.annotation_request import AnnotationRequest
from models.coordinates import Coordinates
from models.dimensions import Dimensions
from script_annotation_yolo import process_image_and_annotations



# Configurer du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Initialiser l'application FastAPI
app = FastAPI()

origins = [
    "http://localhost:4200",
]

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger le modèle YOLOv8 (best.onnx)
model = YOLO("./results_trains/invoice_yolov8_new_train/weights/best.onnx")

# Liste des labels
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

# Créer un dossier pour les images annotées s'il n'existe pas
annotated_images_folder = "annotated_images"
os.makedirs(annotated_images_folder, exist_ok=True)

# Monter le dossier des images annotées comme fichiers statiques
app.mount("/images", StaticFiles(directory=annotated_images_folder), name="images")

@app.post("/api/v0/ocr-facture/process-image")
async def process_image(file: UploadFile = File(...)):
    try:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        # Enregistrer l'image
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        logger.info("Received file: %s", file.filename)
        
        # detection des regions et extraction du texte des boxes detectées
        extracted_data , output_image_path = extract_text(temp_file_path)
        
        # Déplacer l'image annotée dans le dossier des images annotées
        annotated_image_name = f"{timestamp}_{os.path.basename(output_image_path)}"
        annotated_image_path = os.path.join(annotated_images_folder, annotated_image_name)
        os.rename(output_image_path, annotated_image_path)
        
        logger.info("Extracted data: %s", extracted_data)
        print("Extracted data", extracted_data)

        os.remove(temp_file_path)
        return {"extracted_data": extracted_data, "image":f"/images/{annotated_image_name}"}

    except Exception as e:
        logger.error("Error processing image: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v0/ocr-facture/add_file_in_dataset")
async def add_file_in_dataset(file: UploadFile = File(...), dimensions: str = Form(...),coordonnees: str = Form(...)):
    
    try:
        dimensions_data = json.loads(dimensions)
        coordonnees_data = json.loads(coordonnees)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format"}, 422

    
    try:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # Enregistrer l'image
        temp_file_path = f"{timestamp}{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        logger.info("Received file: %s", file.filename)
        
        # Détection des régions et extraction du texte des boxes détectées
        txt_path, resized_image_path = process_image_and_annotations(temp_file_path, dimensions_data, coordonnees_data)
        
        logger.info("Extracted data: %s", coordonnees_data)
        print("Extracted data", coordonnees_data)

        os.remove(temp_file_path)
        # return {"extracted_data": coordonnees_data, "annotation_file": txt_path, "resized_image": resized_image_path}
        return {"message":"Operation fait avec succes"}

    except Exception as e:
        logger.error("Error processing image: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000 , reload=True)