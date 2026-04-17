import os

output_dir = r"C:\projet_cv\data\frames_s4"

total_images = 0

for root, dirs, files in os.walk(output_dir):
    for file in files:
        if file.endswith(".jpg"):
            total_images += 1

print("Nombre total d'images extraites :", total_images)

if total_images > 0:
    print("✔️ Extraction OK")
else:
    print("❌ Problème : aucune image trouvée")