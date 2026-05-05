import os
import cv2
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms
import pandas as pd


CLASSES      = ['drinking', 'reach_side', 'safe_drive']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}

# Régions de crop dans le mosaic
REGIONS = {
    'hands' : (0,   360,  0,   640),
    'face'  : (360, 720,  0,   640),
    'body'  : (360, 720, 640, 1280),
}


class DistractionDataset(Dataset):
    def __init__(self, video_path, annotations_csv, session_name,
                 img_size, seq_len=5, sample_every=5, augment=False):
       
        self.video_path   = video_path
        self.img_size     = img_size
        self.seq_len      = seq_len
        self.samples      = []  # liste de (video_path, classe, [indices])

        if augment:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(img_size),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])

        # Charge les annotations
        df         = pd.read_csv(annotations_csv)
        df_session = df[df['session'] == session_name].copy()
        frame_to_label = dict(zip(df_session['frame_id'], df_session['label']))

        # Sampling : 1 frame sur sample_every
        sampled = {
            fid: lbl
            for fid, lbl in sorted(frame_to_label.items())
            if fid % sample_every == 0
        }

        # Groupe par classe
        class_frames = {cls: [] for cls in CLASSES}
        for fid, lbl in sorted(sampled.items()):
            if lbl in class_frames:
                class_frames[lbl].append(fid)

        # Découpe en séquences de seq_len frames
        for cls, frame_ids in class_frames.items():
            for i in range(0, len(frame_ids) - seq_len + 1, seq_len):
                sequence = frame_ids[i:i + seq_len]
                if len(sequence) == seq_len:
                    # Return : (path_video, classe, indices_séquence)
                    self.samples.append((video_path, cls, sequence))

        print(f'\n=== Dataset {session_name} | seq_len={seq_len} ===')
        print(f'Total séquences : {len(self.samples)}')
        for cls in CLASSES:
            count = sum(1 for _, c, _ in self.samples if c == cls)
            print(f'  {cls:20s} : {count} séquences')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        
        video_path, cls, frame_indices = self.samples[idx]

        # Ouvre la vidéo
        cap = cv2.VideoCapture(video_path)

        body_frames  = []
        face_frames  = []
        hands_frames = []

        for fid in frame_indices:
            # Positionne sur la bonne frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()
            if not ret:
                # Frame manquante → image noire
                blank = torch.zeros(3, *self.img_size)
                body_frames.append(blank)
                face_frames.append(blank)
                hands_frames.append(blank)
                continue

            # Crop les 3 régions depuis le mosaic
            for stream, tensors in [('body',  body_frames),
                                     ('face',  face_frames),
                                     ('hands', hands_frames)]:
                y1, y2, x1, x2 = REGIONS[stream]
                crop  = frame[y1:y2, x1:x2]
                crop  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop  = cv2.resize(crop, self.img_size)
                tensors.append(self.transform(crop))

        cap.release()

        return (
            torch.stack(body_frames),    # [seq_len, 3, 224, 224]
            torch.stack(face_frames),    # [seq_len, 3, 224, 224]
            torch.stack(hands_frames),   # [seq_len, 3, 224, 224]
            torch.tensor(CLASS_TO_IDX[cls], dtype=torch.long)
        )

    def get_class_weights(self):
        counts = [0] * len(CLASSES)
        for _, cls, _ in self.samples:
            counts[CLASS_TO_IDX[cls]] += 1
        total   = len(self.samples)
        weights = [total / (len(CLASSES) * c) if c > 0 else 0
                   for c in counts]
        return torch.FloatTensor(weights)


def get_data_loaders(cfg):
    full_dataset = DistractionDataset(
        video_path      = cfg['video_path'],
        annotations_csv = cfg['annotations_csv'],
        session_name    = cfg['session_name'],
        img_size        = cfg['img_size'],
        seq_len         = cfg['seq_len'],
        sample_every    = cfg['sample_every'],
        augment         = False
    )

    total      = len(full_dataset)
    val_size   = int(total * cfg['val_split'])
    test_size  = int(total * cfg['test_split'])
    train_size = total - val_size - test_size

    train_ds, val_ds, test_ds = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Réapplique augmentation sur train
    train_ds.dataset = DistractionDataset(
        video_path      = cfg['video_path'],
        annotations_csv = cfg['annotations_csv'],
        session_name    = cfg['session_name'],
        img_size        = cfg['img_size'],
        seq_len         = cfg['seq_len'],
        sample_every    = cfg['sample_every'],
        augment         = True
    )

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],
                              shuffle=True,  num_workers=cfg['num_workers'])
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'],
                              shuffle=False, num_workers=cfg['num_workers'])
    test_loader  = DataLoader(test_ds,  batch_size=cfg['batch_size'],
                              shuffle=False, num_workers=cfg['num_workers'])

    print(f'\nTrain : {train_size} | Val : {val_size} | Test : {test_size}')
    print(f'Class weights : {full_dataset.get_class_weights()}')

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    import sys
    sys.path.append('.')
    from src.utils.config import CFG

    train_loader, val_loader, test_loader = get_data_loaders(CFG)

    body, face, hands, labels = next(iter(train_loader))
    print(f'\nBody  shape : {body.shape}')
    print(f'Face  shape : {face.shape}')
    print(f'Hands shape : {hands.shape}')
    print(f'Labels      : {labels}')