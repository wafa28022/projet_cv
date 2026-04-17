import cv2
import os

input_folder = r"C:\projet_cv\data\frames_s4"
output_folder = r"C:\projet_cv\data\frames_resized"

os.makedirs(output_folder, exist_ok=True)

for root, dirs, files in os.walk(input_folder):
    for f in files:
        if f.endswith(".jpg"):
            path = os.path.join(root, f)
            img = cv2.imread(path)

            img_resized = cv2.resize(img, (224, 224))

            save_path = os.path.join(output_folder, f)
            cv2.imwrite(save_path, img_resized)

print("Resize terminé ✔️")