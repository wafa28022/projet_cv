import json
from pathlib import Path

# ============================================================
# Les 2 sessions avec leurs classes disponibles
# ============================================================
sessions = [
    {
        "json_path": r"C:\Users\msi\Desktop\ma-part\dmd\gE\28\s4\gE_28_s4_2019-03-21T10;19;50+01;00_rgb_ann_distraction.json",
        "video_name": "gE_28_s4",
        "classes": ["safe_driving", "drinking", "reach_side"]  # 3 classes
    },
    {
        "json_path": r"C:\Users\msi\Desktop\ma-part\dmd\gB\10\s2\gB_10_s2_2019-03-11T15;15;21+01;00_rgb_ann_distraction.json",
        "video_name": "gB_10_s2",
        "classes": ["safe_driving", "reach_side"]  # 2 classes seulement
    }
]

# Mapping des actions
action_map = {
    "driver_actions/safe_drive": "safe_driving",
    "driver_actions/drinking":   "drinking",
    "driver_actions/reach_side": "reach_side"
}

output_path = Path(r"C:\projet_cv\data")
output_path.mkdir(parents=True, exist_ok=True)

all_labels = {}

for session in sessions:
    json_path  = session["json_path"]
    video_name = session["video_name"]
    classes    = session["classes"]

    print(f"\n📂 Traitement : {video_name}")
    print(f"   Classes utilisées : {classes}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    actions = data["openlabel"]["actions"]

    # Trouver le nombre total de frames
    max_frame = 0
    for action_data in actions.values():
        for interval in action_data["frame_intervals"]:
            max_frame = max(max_frame, interval["frame_end"])

    # Initialiser toutes les frames à safe_driving par défaut
    labels = {str(i): "safe_driving" for i in range(max_frame + 1)}

    # Assigner les labels selon les annotations
    for action_data in actions.values():
        action_type = action_data["type"]

        if action_type not in action_map:
            continue

        class_name = action_map[action_type]

        # ✅ Ignorer les classes non disponibles pour cette session
        if class_name not in classes:
            continue

        for interval in action_data["frame_intervals"]:
            start = interval["frame_start"]
            end   = interval["frame_end"]
            for frame_id in range(start, end + 1):
                labels[str(frame_id)] = class_name

    # Compter par classe
    counts = {"safe_driving": 0, "drinking": 0, "reach_side": 0}
    for label in labels.values():
        if label in counts:
            counts[label] += 1

    print(f"  safe_driving : {counts['safe_driving']} frames")
    print(f"  drinking     : {counts['drinking']} frames")
    print(f"  reach_side   : {counts['reach_side']} frames")

    # Ajouter au dictionnaire global
    for frame_id, label in labels.items():
        key = f"{video_name}_{frame_id}"
        all_labels[key] = label

# Sauvegarder le JSON
labels_path = output_path / "frame_labels.json"
with open(labels_path, 'w') as f:
    json.dump(all_labels, f, indent=2)

print(f"\n✅ frame_labels.json créé : {labels_path}")
print(f"✅ Total frames labelisées : {len(all_labels)}")