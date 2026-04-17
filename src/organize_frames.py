import json
import os
import shutil

# Chemins
json_path = r"C:\Users\msi\Desktop\ma-part\dmd\gE\28\s4\gE_28_s4_2019-03-21T10;19;50+01;00_rgb_ann_distraction.json"
frames_dir = r"C:\projet_cv\data\frames_s4\gE_28_s4_2019-03-21T10;19;50+01;00_rgb_face"
output_dir = r"C:\projet_cv\data\organized_s4"

# Lire le JSON
with open(json_path, 'r') as f:
    data = json.load(f)

# Mapping corrigé selon le vrai format du JSON
action_map = {
    "driver_actions/safe_drive": "safe_driving",
    "driver_actions/texting_right": "texting_right",
    "driver_actions/phonecall_right": "phonecall_right",
    "driver_actions/texting_left": "texting_left",
    "driver_actions/phonecall_left": "phonecall_left",
    "driver_actions/radio": "radio",
    "driver_actions/drinking": "drinking",
    "driver_actions/reach_side": "reach_side",
    "driver_actions/hair_and_makeup": "hair_makeup",
    "driver_actions/change_gear": "change_gear",
    "driver_actions/unclassified": "unclassified"
}

# Créer les dossiers par classe
for class_name in action_map.values():
    os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

# Extraire les actions
actions = data["openlabel"]["actions"]

# Parcourir les actions et copier les frames
for action_id, action_data in actions.items():
    action_type = action_data["type"]

    if action_type not in action_map:
        print(f"⚠️ Ignoré : {action_type}")
        continue

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
                sampling_rate = 3
                fps = 30
                interval = int(fps / sampling_rate)

                real_frame = frame_num * interval

                if start <= real_frame <= end:
                    src = os.path.join(frames_dir, frame_file)
                    dst = os.path.join(output_dir, class_name, frame_file)
                    shutil.copy2(src, dst)
            except:
                continue

print("✅ Organisation terminée !")

# Afficher le résumé
print("\n📊 Résumé par classe :")
for class_name in os.listdir(output_dir):
    count = len(os.listdir(os.path.join(output_dir, class_name)))
    print(f"  {class_name} : {count} frames")