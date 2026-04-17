import json
from pathlib import Path
from collections import defaultdict

dataset_path = Path(r"C:\Users\hp\Downloads\dmd-dataset-mini-sample-gA-3-s1")

json_file = list(dataset_path.rglob("*.json"))[0]
with open(json_file, 'r') as f:
    data = json.load(f)

actions = data['openlabel']['actions']
total_frames = len(data['openlabel']['frames'])

# ============================================================
# Tous les comportements dangereux → distracted
# ============================================================
distracted_types = [
    "driver_actions/radio",
    "driver_actions/reach_side",
    "driver_actions/drinking",
    "driver_actions/talking_to_passenger",
    "driver_actions/unclassified",
    "gaze_on_road/not_looking_road"   # fusionné ici
]

# Marquer chaque frame
frame_labels = {}
for action_id, action in actions.items():
    action_type = action.get('type', '')

    if action_type in distracted_types:
        label = 'distracted'
    else:
        continue

    for interval in action.get('frame_intervals', []):
        start = interval['frame_start']
        end   = interval['frame_end']
        for f in range(start, end + 1):
            frame_labels[f] = label

# Compter par classe
count = defaultdict(int)
for f in range(total_frames):
    label = frame_labels.get(f, 'safe_drive')
    count[label] += 1

# Afficher résultats
print("=== Distribution finale des classes ===\n")
for label, nombre in sorted(count.items()):
    pourcentage = (nombre / total_frames) * 100
    barre = "█" * int(pourcentage / 2)
    print(f"  {label:15s} : {nombre:5d} frames ({pourcentage:5.1f}%)  {barre}")

print(f"\n  {'TOTAL':15s} : {total_frames:5d} frames")

# ============================================================
# Sauvegarder le mapping frame → label dans un fichier
# ============================================================
output = {}
for f in range(total_frames):
    output[str(f)] = frame_labels.get(f, 'safe_drive')

output_path = Path(r"C:\Users\hp\CNN\data\processed\frame_labels.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nLabels sauvegardés dans : {output_path}")
print("On est prêt pour l'extraction des frames !")