import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract
from PIL import Image
import json
from script import detect_regions, extract_text
import uvicorn as uvicorn


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

@app.post("/api/v0/ocr-facture/process-image")
async def process_image(file: UploadFile = File(...)):
    try:
        
        logger.info("Received file: %s", file.filename)
        
        # detection des regions et extraction du texte des boxes detectées
        extracted_data = extract_text(file.file)
        
        logger.info("Extracted data: %s", extracted_data)

        return {"extracted_data": extracted_data}

    except Exception as e:
        logger.error("Error processing image: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000 , reload=True)