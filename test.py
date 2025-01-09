import torch

print(torch.cuda.is_available())  # Devrait retourner True si un GPU est disponible
print(torch.cuda.current_device())  # Retourne l'index du GPU actuel
print(torch.cuda.get_device_name(0))  # Retourne le nom du GPU