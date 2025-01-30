import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from script import detect_regions, extract_text
import uvicorn as uvicorn
import os
from datetime import datetime



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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000 , reload=True)