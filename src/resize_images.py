import cv2
import os

input_dir = r"C:\projet_cv\data\organized_s4"
output_dir = r"C:\projet_cv\data\resized_s4"

classes = ["safe_driving", "drinking", "reach_side"]

for class_name in classes:
    input_folder = os.path.join(input_dir, class_name)
    output_folder = os.path.join(output_dir, class_name)
    os.makedirs(output_folder, exist_ok=True)

    files = os.listdir(input_folder)
    print(f"Redimensionnement de {class_name} : {len(files)} images...")

    for img_file in files:
        if not img_file.endswith('.jpg'):
            continue
        path = os.path.join(input_folder, img_file)
        img = cv2.imread(path)
        img_resized = cv2.resize(img, (224, 224))
        cv2.imwrite(os.path.join(output_folder, img_file), img_resized)

    print(f"✅ {class_name} terminé !")

print("\n✅ Toutes les images sont en 224x224 !")