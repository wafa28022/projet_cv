import json
import os
import shutil

# ✅ Les 2 sessions avec leurs JSONs
sessions = [
    {
        "json_path":  r"C:\Users\msi\Desktop\ma-part\dmd\gE\28\s4\gE_28_s4_2019-03-21T10;19;50+01;00_rgb_ann_distraction.json",
        "frames_dir": r"C:\projet_cv\data\frames_s4\gE_28_s4_2019-03-21T10;19;50+01;00_rgb_mosaic"
    },
    {
        "json_path":  r"C:\Users\msi\Desktop\ma-part\dmd\gB\10\s2\gB_10_s2_2019-03-11T15;15;21+01;00_rgb_ann_distraction.json",
        "frames_dir": r"C:\projet_cv\data\frames_s4\gB_10_s2_2019-03-11T15;15;21+01;00_rgb_mosaic"
    }
]

output_dir = r"C:\projet_cv\data\organized_s4\mosaic"

# 3 classes seulement
action_map = {
    "driver_actions/safe_drive": "safe_driving",
    "driver_actions/drinking":   "drinking",
    "driver_actions/reach_side": "reach_side"
}

# Supprimer et recréer proprement
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

for class_name in action_map.values():
    os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

# Traiter chaque session
for session in sessions:
    json_path  = session["json_path"]
    frames_dir = session["frames_dir"]

    if not os.path.exists(json_path):
        print(f"❌ JSON introuvable : {json_path}")
        continue

    if not os.path.exists(frames_dir):
        print(f"❌ Frames introuvables : {frames_dir}")
        continue

    print(f"\n📂 Traitement : {os.path.basename(json_path)}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    actions = data["openlabel"]["actions"]

    for action_id, action_data in actions.items():
        action_type = action_data["type"]

        if action_type not in action_map:
            continue

        class_name = action_map[action_type]

        for interval in action_data["frame_intervals"]:
            start = interval["frame_start"]
            end   = interval["frame_end"]

            for frame_file in os.listdir(frames_dir):
                if not frame_file.endswith('.jpg'):
                    continue
                try:
                    frame_num  = int(frame_file.split('_f')[-1].replace('.jpg', ''))
                    real_frame = frame_num * 6

                    if start <= real_frame <= end:
                        src = os.path.join(frames_dir, frame_file)
                        dst = os.path.join(output_dir, class_name, frame_file)
                        shutil.copy2(src, dst)
                except:
                    continue

# Résumé
print("\n📊 Résumé par classe :")
for class_name in action_map.values():
    count = len(os.listdir(os.path.join(output_dir, class_name)))
    print(f"  {class_name} : {count} frames")

print("\n✅ Organisation terminée !")