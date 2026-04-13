import cv2
import os

# Chemin vers tes vidéos S4
video_dir = r"C:\Users\msi\Desktop\ma-part\dmd\gE\28\s4"

# Chemin où les frames seront sauvegardées
output_dir = r"C:\projet_cv\data\frames_s4"

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(output_dir, exist_ok=True)

# Parcourir tous les fichiers vidéo
for video_file in os.listdir(video_dir):
    if video_file.endswith(('.mp4', '.avi', '.mov')):
        video_path = os.path.join(video_dir, video_file)
        video_name = os.path.splitext(video_file)[0]

        # Créer un dossier pour chaque vidéo
        frames_folder = os.path.join(output_dir, video_name)
        os.makedirs(frames_folder, exist_ok=True)

        # Ouvrir la vidéo
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)  # FPS de la vidéo
        
        print(f"Extraction de : {video_file} ({fps:.2f} FPS)")

        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Calculer à quelle seconde on est
            second = int(frame_count / fps)
            # Position dans la seconde actuelle
            position_in_second = frame_count - int(second * fps)
            # Prendre 5 frames par seconde uniformément
            interval = int(fps / 5)

            if frame_count % interval == 0:
                frame_name = f"{video_name}_sec{second:04d}_f{saved_count:05d}.jpg"
                frame_path = os.path.join(frames_folder, frame_name)
                cv2.imwrite(frame_path, frame)
                saved_count += 1

            frame_count += 1

        cap.release()
        print(f"✅ {saved_count} frames sauvegardées de {video_file}")

print("Extraction terminée !")