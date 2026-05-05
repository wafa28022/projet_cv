import json
import os
from pathlib import Path

def read_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def build_samples_from_folders(dataset_path):
    
    dataset_path = Path(dataset_path)

    # Mapping classe → index
    class_to_idx = {
        'safe_drive' : 0,
        'reach_side' : 1,
        'drinking'   : 2
    }

    samples = []
    for classe, idx in class_to_idx.items():
        dossier = dataset_path / classe
        if not dossier.exists():
            continue
        for img_path in sorted(dossier.glob("*.jpg")):
            samples.append({
                "relative_path" : f"{classe}/{img_path.name}",
                "class"         : idx
            })

    print(f"Samples construits : {len(samples)}")
    return samples