"""
Extraction des frames depuis le fichier mosaic RGB.
Le mosaic contient les 3 cameras dans une seule vidéo :
    - Hands : frame[0:360,    0:640]   haut gauche
    - Face  : frame[360:720,  0:640]   bas gauche
    - Body  : frame[360:720, 640:1280] bas droite

Frame offsets (décalage entre caméras) depuis le JSON streams :
    - Face  : 0
    - Body  : 3
    - Hands : 18
"""

import cv2
import pandas as pd
from pathlib import Path


SAMPLE_EVERY_N = 5
IMG_SIZE       = 224
JPEG_QUALITY   = 92

# Régions de crop dans le mosaic [y1:y2, x1:x2]
REGIONS = {
    'hands' : (0,   360,  0,   640),
    'face'  : (360, 720,  0,   640),
    'body'  : (360, 720,  640, 1280),
}

# Décalage de frames entre caméras (depuis JSON streams)
FRAME_OFFSETS = {
    'face'  : 0,
    'body'  : 3,
    'hands' : 18,
}


def extract_mosaic_frames(video_path: str, annotations_csv: str,
                          session_name: str, output_dir: str):
    """
    Lit le mosaic et extrait les 3 régions pour chaque frame annotée.
    Sauvegarde dans :
        output_dir/
            body/  safe_drive/ drinking/ reach_side/
            face/  safe_drive/ drinking/ reach_side/
            hands/ safe_drive/ drinking/ reach_side/
    """
    output_dir = Path(output_dir)
    classes    = ['safe_drive', 'drinking', 'reach_side']

    # Crée les dossiers
    for stream in REGIONS:
        for cls in classes:
            (output_dir / stream / cls).mkdir(parents=True, exist_ok=True)

    # Charge les annotations
    df         = pd.read_csv(annotations_csv)
    df_session = df[df['session'] == session_name].copy()
    frame_to_label = dict(zip(df_session['frame_id'], df_session['label']))

    # Sampling
    sampled_frames = {
        fid: lbl
        for fid, lbl in sorted(frame_to_label.items())
        if fid % SAMPLE_EVERY_N == 0
    }

    print(f'\n=== Session {session_name} | Mosaic ===')
    print(f'Frames annotées  : {len(frame_to_label)}')
    print(f'Après sampling   : {len(sampled_frames)}')
    for cls in classes:
        count = sum(1 for l in sampled_frames.values() if l == cls)
        print(f'  {cls:20s} : {count}')

    if not sampled_frames:
        print('Aucune frame à extraire.')
        return

    # Ouvre le mosaic
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f'Impossible d ouvrir : {video_path}')

    saved     = 0
    frame_id  = 0
    max_frame = max(sampled_frames.keys())

    while frame_id <= max_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id in sampled_frames:
            label = sampled_frames[frame_id]

            # Extrait et sauvegarde chaque région
            for stream, (y1, y2, x1, x2) in REGIONS.items():
                crop          = frame[y1:y2, x1:x2]
                crop_resized  = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
                filename      = f'{session_name}_frame_{frame_id:05d}.jpg'
                save_path     = output_dir / stream / label / filename
                cv2.imwrite(str(save_path), crop_resized,
                            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            saved += 1

        frame_id += 1

    cap.release()
    print(f'Extraction terminée : {saved} frames × 3 streams = {saved*3} images')


def get_extraction_stats(output_dir: str):
    output_dir = Path(output_dir)
    print('\n=== Dataset final ===')
    for stream in ['body', 'face', 'hands']:
        print(f'\n  Stream : {stream}')
        total = 0
        for cls_dir in sorted((output_dir / stream).iterdir()):
            if cls_dir.is_dir():
                count = len(list(cls_dir.glob('*.jpg')))
                bar   = '█' * (count // 20)
                print(f'    {cls_dir.name:20s} : {count:4d}  {bar}')
                total += count
        print(f'    {"TOTAL":20s} : {total}')


if __name__ == '__main__':
    # Supprime les anciennes frames si elles existent
    import shutil
    old_dir = Path('data/processed/frames_mosaic')
    if old_dir.exists():
        shutil.rmtree(old_dir)
        print('Ancien dossier supprimé')

    extract_mosaic_frames(
        video_path='data/raw/gA_3_s1_rgb_mosaic.avi',
        annotations_csv='data/processed/distraction_annotations.csv',
        session_name='s1',
        output_dir='data/processed/frames_mosaic',
    )
    get_extraction_stats('data/processed/frames_mosaic')