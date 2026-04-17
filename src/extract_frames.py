import cv2
import os

# ==============================
# PARAMÈTRES
# ==============================

video_dir = r"C:\Users\msi\Desktop\ma-part\dmd\gE\28\s4"
output_dir = r"C:\projet_cv\data\frames_s4"
sampling_rate = 3  # 🔥 nombre de frames par seconde

# ==============================
# CRÉATION DOSSIER
# ==============================

os.makedirs(output_dir, exist_ok=True)

# ==============================
# TRAITEMENT VIDÉOS
# ==============================

for video_file in os.listdir(video_dir):

    if video_file.endswith(('.mp4', '.avi', '.mov')):

        video_path = os.path.join(video_dir, video_file)
        video_name = os.path.splitext(video_file)[0]

        frames_folder = os.path.join(output_dir, video_name)
        os.makedirs(frames_folder, exist_ok=True)

        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps == 0:
            print(f"❌ Impossible de lire FPS pour {video_file}")
            continue

        interval = int(fps / sampling_rate)

        print(f"\n🎬 Vidéo : {video_file}")
        print(f"FPS : {fps:.2f} | Interval : {interval}")

        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % interval == 0:

                second = int(frame_count / fps)

                frame_name = f"{video_name}_sec{second:04d}_f{saved_count:05d}.jpg"
                frame_path = os.path.join(frames_folder, frame_name)

                cv2.imwrite(frame_path, frame)
                saved_count += 1

            frame_count += 1

        cap.release()

        print(f"✅ {saved_count} frames sauvegardées")

print("\n🎉 Extraction terminée !")