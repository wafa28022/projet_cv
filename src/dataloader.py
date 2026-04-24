import cv2
import json
import sys
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
sys.path.append(r"C:\projet_cv")
from src.utils.helpers import read_json

# ============================================================
# Chemins
# ============================================================
VIDEOS = [
    {
        "video_path": r"C:\Users\msi\Desktop\ma-part\dmd\gE\28\s4\gE_28_s4_2019-03-21T10;19;50+01;00_rgb_mosaic.avi",
        "video_name": "gE_28_s4"
    },
    {
        "video_path": r"C:\Users\msi\Desktop\ma-part\dmd\gB\10\s2\gB_10_s2_2019-03-11T15;15;21+01;00_rgb_mosaic.avi",
        "video_name": "gB_10_s2"
    }
]

CLASSES      = ['drinking', 'reach_side', 'safe_driving']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# ============================================================
# Dataset
# ============================================================
class DMDDataset(Dataset):
    def __init__(self, dataset_path, labels_path, img_size):
        self.dataset_path = Path(dataset_path)
        self.frame_labels = read_json(labels_path)
        self.samples      = self.load_dataset()
        self.transform    = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(img_size),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def load_dataset(self):
        samples = []

        for class_name in CLASSES:
            label     = CLASS_TO_IDX[class_name]
            class_dir = self.dataset_path / class_name

            if not class_dir.exists():
                print(f"⚠️ Dossier manquant : {class_dir}")
                continue

            files = sorted([f for f in class_dir.iterdir() if f.suffix == '.jpg'])
            debut = 0
            fin   = len(files) - 1

            for fname in files:
                samples.append((str(fname), label, class_name, debut, fin))

        print(f"✅ Dataset chargé : {len(samples)} images")
        print(f"✅ Classes : {CLASSES}")
        return samples

    def get_stats(self):
        """Affiche les statistiques du dataset"""
        stats = {}

        for class_name in CLASSES:
            label     = CLASS_TO_IDX[class_name]
            class_dir = self.dataset_path / class_name

            if not class_dir.exists():
                continue

            files = sorted([f for f in class_dir.iterdir() if f.suffix == '.jpg'])

            stats[class_name] = {
                "label":        label,
                "nb_images":    len(files),
                "nb_sequences": 1,
                "longueur_seq": len(files),
                "debut":        0,
                "fin":          len(files) - 1,
                "exemple_path": str(files[0]) if files else "N/A"
            }

        print("\n📊 Statistiques du Dataset :")
        print("=" * 70)
        for class_name, info in stats.items():
            print(f"\n🏷️  Classe      : {class_name} (label={info['label']})")
            print(f"   Nb images   : {info['nb_images']}")
            print(f"   Nb séquences: {info['nb_sequences']}")
            print(f"   Longueur seq: {info['longueur_seq']} frames")
            print(f"   Début       : {info['debut']}")
            print(f"   Fin         : {info['fin']}")
            print(f"   Exemple path: {info['exemple_path']}")
        print("=" * 70)
        print(f"\n✅ Total images : {sum(i['nb_images'] for i in stats.values())}")

        return stats

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, class_name, debut, fin = self.samples[idx]

        img        = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img)

        label_tensor = torch.tensor(label, dtype=torch.long)

        return img_tensor, label_tensor, debut, fin, img_path


# ============================================================
# DataLoaders
# ============================================================
def get_data_loaders(cfg):
    full_dataset = DMDDataset(
        dataset_path = cfg["dataset_path"],
        labels_path  = cfg["labels_path"],
        img_size     = cfg["img_size"]
    )

    val_size   = int(len(full_dataset) * cfg["val_split"])
    train_size = len(full_dataset) - val_size

    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg["batch_size"],
        shuffle     = True,
        num_workers = cfg["num_workers"]
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg["batch_size"],
        shuffle     = False,
        num_workers = cfg["num_workers"]
    )

    print(f"\nTrain samples: {train_size}  |  Val samples: {val_size}\n")
    return train_loader, val_loader


# ============================================================
# Test
# ============================================================
if __name__ == "__main__":
    cfg = read_json(r"C:\projet_cv\config.json")

    train_loader, val_loader = get_data_loaders(cfg)

    # Afficher les stats
    full_dataset = DMDDataset(
        dataset_path = cfg["dataset_path"],
        labels_path  = cfg["labels_path"],
        img_size     = cfg["img_size"]
    )
    full_dataset.get_stats()

    # ✅ Infos sur les batches
    print("\n📦 Informations sur les Batches :")
    print("=" * 70)
    print(f"   Batch size          : {cfg['batch_size']}")
    print(f"   Nb batches Train    : {len(train_loader)}")
    print(f"   Nb batches Val      : {len(val_loader)}")
    print(f"   Total images Train  : {len(train_loader.dataset)}")
    print(f"   Total images Val    : {len(val_loader.dataset)}")
    print("=" * 70)

    # Vérifier un batch
    images, labels, debuts, fins, paths = next(iter(train_loader))
    print(f"\n✅ Exemple d'un batch :")
    print(f"   Shape  : {images.shape}")
    print(f"   Labels : {labels}")
    print(f"   Débuts : {debuts}")
    print(f"   Fins   : {fins}")
    print(f"   Path   : {paths[0]}")