import torch
import cv2
import numpy as np
import matplotlib
from torchvision import transforms

print("=== Verification de l'installation ===")
print(f"PyTorch     : {torch.__version__}")
print(f"OpenCV      : {cv2.__version__}")
print(f"NumPy       : {np.__version__}")
print(f"Matplotlib  : {matplotlib.__version__}")
print(f"GPU dispo   : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU nom     : {torch.cuda.get_device_name(0)}")
else:
    print("GPU         : CPU uniquement")

print("\nTout est OK ! Pret pour le projet.")