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
        frame_count = 0
        
        print(f"Extraction de : {video_file}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sauvegarder une frame sur 5 (pour ne pas avoir trop d'images)
            if frame_count % 5 == 0:
                frame_name = f"{video_name}_frame_{frame_count:05d}.jpg"
                frame_path = os.path.join(frames_folder, frame_name)
                cv2.imwrite(frame_path, frame)
            
            frame_count += 1
        
        cap.release()
        print(f"✅ {frame_count} frames extraites de {video_file}")

print("Extraction terminée !")