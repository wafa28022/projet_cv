import json
import pandas as pd
from pathlib import Path


DISTRACTION_CLASSES = ['safe_drive', 'drinking', 'reach_side']


def _build_frame_label_map(actions: dict, category_filter: str = None) -> dict:
    frame_map = {}
    for action in actions.values():
        label = action['type']
        if category_filter and not label.startswith(category_filter):
            continue
        for interval in action['frame_intervals']:
            for f in range(interval['frame_start'], interval['frame_end'] + 1):
                frame_map[f] = label
    return frame_map


def parse_distraction(json_path: str, session_name: str) -> pd.DataFrame:
    """
    Parse un fichier distraction JSON OpenLabel.
    Garde seulement les 3 classes communes : safe_drive, drinking, reach_side.

    Args:
        json_path    : chemin vers le fichier JSON
        session_name : nom court ex: 's1', 's4'

    Returns:
        DataFrame avec colonnes : frame_id, label, session
    """
    with open(json_path) as f:
        data = json.load(f)

    actions     = data['openlabel']['actions']
    total_frames = len(data['openlabel']['frames'])

    frame_map = _build_frame_label_map(actions, category_filter='driver_actions')

    rows = []
    for frame_id in range(total_frames):
        full_label = frame_map.get(frame_id, None)
        if full_label is None:
            continue

        # Garde seulement les 3 classes utiles
        short_label = full_label.replace('driver_actions/', '')
        if short_label not in DISTRACTION_CLASSES:
            continue

        rows.append({
            'frame_id' : frame_id,
            'label'    : short_label,
            'session'  : session_name,
        })

    return pd.DataFrame(rows)


def get_label_stats(df: pd.DataFrame):
    print(f'Total frames gardées : {len(df)}')
    print()
    for session in df['session'].unique():
        sub = df[df['session'] == session]
        print(f'--- Session {session} ---')
        print(sub['label'].value_counts().to_string())
        print()
    print('--- TOTAL combiné ---')
    print(df['label'].value_counts().to_string())