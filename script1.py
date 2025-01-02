from PIL import Image
import pytesseract
import cv2
import numpy as np

# Définir le chemin du binaire Tesseract (sur Windows uniquement)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

image_path = './files/facture/facture_test4.jpg'  # Remplacez par le chemin de votre image

if image_path.lower().endswith('.png'):
    raise ValueError("Les fichiers PNG ne sont pas acceptés. Veuillez fournir un fichier au format JPG ou autre.")

image = Image.open(image_path)

# Extraire le texte
texte = pytesseract.image_to_string(image , lang='fra')  # 'fra' pour le français
print("Texte extrait :")
print(texte)