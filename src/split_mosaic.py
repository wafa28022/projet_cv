import cv2
import os
from pathlib import Path

# ============================================================
# Chemins
# ============================================================
input_dir  = Path(r"C:\projet_cv\data\organized_s4\mosaic")
output_dir = Path(r"C:\projet_cv\data\organized_s4\split")

CLASSES = ['drinking', 'reach_side', 'safe_driving']

# ============================================================
# Créer les dossiers de sortie
# ============================================================
for class_name in CLASSES:
    for part in ['hands', 'face', 'body']:
        os.makedirs(output_dir / part / class_name, exist_ok=True)

# ============================================================
# Découper chaque image
# ============================================================
for class_name in CLASSES:
    class_dir = input_dir / class_name
    files     = sorted([f for f in class_dir.iterdir() if f.suffix == '.jpg'])

    print(f"\n✂️  Découpage {class_name} : {len(files)} images...")

    for fname in files:
        img = cv2.imread(str(fname))
        h, w = img.shape[:2]

        # Découpage en 4 quadrants
        mid_h = h // 2
        mid_w = w // 2

        hands = img[0:mid_h,    0:mid_w]    # haut-gauche
        face  = img[mid_h:h,    0:mid_w]    # bas-gauche
        body  = img[mid_h:h,    mid_w:w]    # bas-droite
        # info  = img[0:mid_h, mid_w:w]     # haut-droite → ignoré

        # Redimensionner en 224x224
        hands = cv2.resize(hands, (224, 224))
        face  = cv2.resize(face,  (224, 224))
        body  = cv2.resize(body,  (224, 224))

        # Sauvegarder
        stem = fname.stem  # nom sans extension

        cv2.imwrite(str(output_dir / 'hands' / class_name / f"{stem}_hands.jpg"), hands)
        cv2.imwrite(str(output_dir / 'face'  / class_name / f"{stem}_face.jpg"),  face)
        cv2.imwrite(str(output_dir / 'body'  / class_name / f"{stem}_body.jpg"),  body)

    print(f"✅ {class_name} terminé !")

# ============================================================
# Résumé
# ============================================================
print("\n📊 Résumé :")
for part in ['hands', 'face', 'body']:
    print(f"\n📷 {part} :")
    for class_name in CLASSES:
        count = len(list((output_dir / part / class_name).iterdir()))
        print(f"   {class_name} : {count} images")

print("\n✅ Découpage terminé !")