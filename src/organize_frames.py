import json
import os
import shutil

# Chemins
json_path = r"C:\Users\msi\Desktop\ma-part\dmd\gE\28\s4\gE_28_s4_2019-03-21T10;19;50+01;00_rgb_ann_distraction.json"
frames_dir = r"C:\projet_cv\data\frames_s4\gE_28_s4_2019-03-21T10;19;50+01;00_rgb_face"
output_dir = r"C:\projet_cv\data\organized_s4"

# ✅ Seulement 3 classes
action_map = {
    "driver_actions/safe_drive": "safe_driving",
    "driver_actions/drinking": "drinking",
    "driver_actions/reach_side": "reach_side"
}

# Supprimer l'ancien dossier et recréer proprement
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

# Créer les dossiers pour les 3 classes
for class_name in action_map.values():
    os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

# Lire le JSON
with open(json_path, 'r') as f:
    data = json.load(f)

# Extraire les actions
actions = data["openlabel"]["actions"]

# Parcourir les actions et copier les frames
for action_id, action_data in actions.items():
    action_type = action_data["type"]

    if action_type not in action_map:
        continue  # Ignorer silencieusement les autres classes

    class_name = action_map[action_type]

    for interval in action_data["frame_intervals"]:
        start = interval["frame_start"]
        end = interval["frame_end"]

        print(f"Copie {class_name} : frames {start} → {end}")

        for frame_file in os.listdir(frames_dir):
            if not frame_file.endswith('.jpg'):
                continue
            try:
                frame_num = int(frame_file.split('_f')[-1].replace('.jpg', ''))
                real_frame = frame_num * 10  # car on prend 3 frames/seconde à 29.76 FPS

                if start <= real_frame <= end:
                    src = os.path.join(frames_dir, frame_file)
                    dst = os.path.join(output_dir, class_name, frame_file)
                    shutil.copy2(src, dst)
            except:
                continue

# Afficher le résumé
print("\n📊 Résumé par classe :")
for class_name in os.listdir(output_dir):
    count = len(os.listdir(os.path.join(output_dir, class_name)))
    print(f"  {class_name} : {count} frames")

print("\n✅ Organisation terminée !")