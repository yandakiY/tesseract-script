from PIL import Image
import pytesseract
import cv2
import numpy as np

# Définir le chemin du binaire Tesseract (sur Windows uniquement)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

image_path = './files/facture/Facture_test2.jpg'  # Remplacez par le chemin de votre image

image = Image.open(image_path)


# Extraire le texte
texte = pytesseract.image_to_string(image , lang='fra')  # 'fra' pour le français
print("Texte extrait : ")
print(texte)