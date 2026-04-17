"""
Extraction des frames avec sampling intelligent et équilibrage des classes.
Stratégie : 1 frame toutes les 15 frames + cap par classe pour équilibrage.
"""

import cv2
import pandas as pd
from pathlib import Path
from collections import defaultdict


SAMPLE_EVERY_N  = 15    # 1 frame toutes les 15 = ~1.67 fps
MAX_PER_CLASS   = 600   # cap par classe ET par session pour équilibrage
IMG_SIZE        = 224   # taille standard pour CNN (ResNet, EfficientNet...)
JPEG_QUALITY    = 92    # qualité image


def extract_frames(video_path: str, annotations_csv: str,
                   session_name: str, output_dir: str):

    output_dir = Path(output_dir)
    classes = ['safe_drive', 'drinking', 'reach_side']
    for cls in classes:
        (output_dir / cls).mkdir(parents=True, exist_ok=True)

    # Charge annotations de cette session uniquement
    df = pd.read_csv(annotations_csv)
    df_session = df[df['session'] == session_name].copy()
    frame_to_label = dict(zip(df_session['frame_id'], df_session['label']))

    # Pré-filtre : garde seulement 1 frame sur SAMPLE_EVERY_N
    sampled_frames = {
        fid: lbl
        for i, (fid, lbl) in enumerate(sorted(frame_to_label.items()))
        if fid % SAMPLE_EVERY_N == 0
    }

    # Cap par classe : max MAX_PER_CLASS frames par classe
    class_frames = defaultdict(list)
    for fid, lbl in sorted(sampled_frames.items()):
        class_frames[lbl].append(fid)

    final_frames = {}
    for cls, fids in class_frames.items():
        kept = fids[:MAX_PER_CLASS]
        for fid in kept:
            final_frames[fid] = cls

    # Stats avant extraction
    print(f'\n=== Session {session_name} ===')
    print(f'Frames annotées totales  : {len(frame_to_label)}')
    print(f'Après sampling (/{SAMPLE_EVERY_N}) : {len(sampled_frames)}')
    print(f'Après cap ({MAX_PER_CLASS}/classe) : {len(final_frames)}')
    for cls in classes:
        count = sum(1 for l in final_frames.values() if l == cls)
        print(f'  {cls:20s} : {count}')

    if not final_frames:
        print('Aucune frame à extraire.')
        return

    # Extraction
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f'Impossible d ouvrir : {video_path}')

    saved    = 0
    frame_id = 0
    max_frame = max(final_frames.keys())

    while frame_id <= max_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id in final_frames:
            label = final_frames[frame_id]
            frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            filename  = f'{session_name}_frame_{frame_id:05d}.jpg'
            save_path = output_dir / label / filename
            cv2.imwrite(str(save_path), frame_resized,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            saved += 1

        frame_id += 1

    cap.release()
    print(f'Extraction terminée : {saved} images sauvegardées')


def get_extraction_stats(output_dir: str):
    output_dir = Path(output_dir)
    print('\n=== Dataset final ===')
    total = 0
    for cls_dir in sorted(output_dir.iterdir()):
        if cls_dir.is_dir():
            count = len(list(cls_dir.glob('*.jpg')))
            bar   = '█' * (count // 20)
            print(f'  {cls_dir.name:20s} : {count:4d}  {bar}')
            total += count
    print(f'  {"TOTAL":20s} : {total}')


if __name__ == '__main__':
    extract_frames(
        video_path='data/raw/gA_3_s1_rgb_body.mp4',
        annotations_csv='data/processed/distraction_annotations.csv',
        session_name='s1',
        output_dir='data/processed/frames',
    )
    get_extraction_stats('data/processed/frames')