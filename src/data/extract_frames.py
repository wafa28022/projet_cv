import cv2
import json
import shutil
from pathlib import Path

# ============================================================
# PARAMÈTRES
# ============================================================
SAMPLE_EVERY_N = 5
IMG_SIZE       = 224
JPEG_QUALITY   = 92

# Découpage mosaic
REGIONS = {
    'hands': (0,   360,  0,   640),
    'face' : (360, 720,  0,   640),
    'body' : (360, 720,  640, 1280),
}

# Offsets (TRÈS IMPORTANT)
FRAME_OFFSETS = {
    'face' : 0,
    'body' : 3,
    'hands': 18,
}

classes = ['safe_drive', 'reach_side', 'drinking']

# ============================================================
# CHEMINS (TES DONNÉES)
# ============================================================
dataset_path = Path(r"C:\Users\hp\Downloads\dmd-dataset-mini-sample-gA-3-s1")
labels_path  = Path(r"C:\Users\hp\CNN\src\data\processed\frame_labels.json")
output_path  = Path(r"C:\Users\hp\CNN\src\data\processed\frames_mosaic")

# ============================================================
# LOAD LABELS JSON
# ============================================================
with open(labels_path, 'r') as f:
    frame_labels = json.load(f)

# convertir en int
frame_labels = {int(k): v for k, v in frame_labels.items()}

# ============================================================
# TROUVER LA VIDÉO
# ============================================================
video_files = list(dataset_path.rglob("*mosaic*"))
video_path  = video_files[0]
print(f"Vidéo trouvée : {video_path.name}")

# ============================================================
# CRÉER DOSSIERS
# ============================================================
if output_path.exists():
    shutil.rmtree(output_path)

for stream in REGIONS:
    for cls in classes:
        (output_path / stream / cls).mkdir(parents=True, exist_ok=True)

# ============================================================
# VIDEO
# ============================================================
cap = cv2.VideoCapture(str(video_path))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"\nTotal frames : {total_frames}")
print("Début extraction...\n")

# ============================================================
# EXTRACTION
# ============================================================
frame_id = 0
saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % SAMPLE_EVERY_N == 0:

        # label global (face = référence)
        label = frame_labels.get(frame_id, 'safe_drive')

        for stream, (y1, y2, x1, x2) in REGIONS.items():

            # appliquer offset
            adjusted_frame_id = frame_id + FRAME_OFFSETS[stream]

            # éviter dépassement
            if adjusted_frame_id >= total_frames:
                continue

            # aller à la bonne frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, adjusted_frame_id)
            ret2, frame_offset = cap.read()

            if not ret2:
                continue

            # crop
            crop = frame_offset[y1:y2, x1:x2]
            crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))

            # save
            filename = f"{stream}_{frame_id:06d}.jpg"
            save_path = output_path / stream / label / filename

            cv2.imwrite(
                str(save_path),
                crop,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )

        saved += 1

        # revenir à la frame originale
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

        if frame_id % 1000 == 0:
            print(f"Progression : {frame_id}/{total_frames}")

    frame_id += 1

cap.release()

print(f"\nExtraction terminée : {saved} frames")
print(f"Images totales ≈ {saved * 3}")