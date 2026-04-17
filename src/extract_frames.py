import cv2
import json
from pathlib import Path

# ============================================================
# Chemins
# ============================================================
dataset_path  = Path(r"C:\Users\hp\Downloads\dmd-dataset-mini-sample-gA-3-s1")
labels_path   = Path(r"C:\Users\hp\CNN\data\processed\frame_labels.json")
output_path   = Path(r"C:\Users\hp\CNN\data\processed")

# Trouver la vidéo rgb_face
video_files = list(dataset_path.rglob("*rgb_face*"))
print(f"Vidéo trouvée : {video_files[0].name}\n")
video_path = video_files[0]

# Charger les labels
with open(labels_path, 'r') as f:
    frame_labels = json.load(f)

# ============================================================
# Créer les dossiers de sortie
# ============================================================
for classe in ['safe_drive', 'distracted']:
    dossier = output_path / classe
    dossier.mkdir(parents=True, exist_ok=True)
    print(f"Dossier créé : {dossier}")

# ============================================================
# Informations sur la vidéo
# ============================================================
cap = cv2.VideoCapture(str(video_path))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps          = cap.get(cv2.CAP_PROP_FPS)
largeur      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
hauteur      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"\n=== Informations vidéo ===")
print(f"  Total frames : {total_frames}")
print(f"  FPS          : {fps}")
print(f"  Résolution   : {largeur} x {hauteur}")
print(f"\nDémarrage extraction...\n")

# ============================================================
# Extraction frame par frame
# ============================================================
compteur = {'safe_drive': 0, 'distracted': 0}
frame_id = 0

while True:
    ret, frame = cap.read()

    # ret = False → plus de frames → on arrête
    if not ret:
        break

    # Récupérer le label de cette frame
    label = frame_labels.get(str(frame_id), 'safe_drive')

    # Redimensionner à 224x224 (taille standard pour CNN)
    frame_resized = cv2.resize(frame, (224, 224))

    # Construire le nom du fichier
    nom_fichier = f"frame_{frame_id:06d}.jpg"
    chemin_sortie = output_path / label / nom_fichier

    # Sauvegarder l'image
    cv2.imwrite(str(chemin_sortie), frame_resized)

    # Incrémenter les compteurs
    compteur[label] += 1
    frame_id += 1

    # Afficher la progression toutes les 500 frames
    if frame_id % 500 == 0:
        print(f"  Progression : {frame_id}/{total_frames} frames traitées...")

cap.release()

# ============================================================
# Résumé final
# ============================================================
print(f"\n=== Extraction terminée ===")
print(f"  safe_drive  : {compteur['safe_drive']:5d} images")
print(f"  distracted  : {compteur['distracted']:5d} images")
print(f"  TOTAL       : {sum(compteur.values()):5d} images")
print(f"\nImages sauvegardées dans : {output_path}")