import cv2
import os

# ✅ Les 2 vidéos mosaic
videos = [
    {
        "path": r"C:\Users\msi\Desktop\ma-part\dmd\gE\28\s4",
        "file": "gE_28_s4_2019-03-21T10;19;50+01;00_rgb_mosaic.avi"
    },
    {
        "path": r"C:\Users\msi\Desktop\ma-part\dmd\gB\10\s2",
        "file": "gB_10_s2_2019-03-11T15;15;21+01;00_rgb_mosaic.avi"
    }
]

output_dir = r"C:\projet_cv\data\frames_s4"
os.makedirs(output_dir, exist_ok=True)

for video_info in videos:
    video_path = os.path.join(video_info["path"], video_info["file"])

    if not os.path.exists(video_path):
        print(f"❌ Fichier introuvable : {video_info['file']}")
        continue

    video_name    = os.path.splitext(video_info["file"])[0]
    frames_folder = os.path.join(output_dir, video_name)
    os.makedirs(frames_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Extraction de : {video_info['file']} ({fps:.2f} FPS)")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        interval = int(fps / 5)
        if frame_count % interval == 0:
            frame_name = f"{video_name}_f{saved_count:05d}.jpg"
            cv2.imwrite(os.path.join(frames_folder, frame_name), frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"✅ {saved_count} frames sauvegardées de {video_info['file']}")

print("\n✅ Extraction terminée !")