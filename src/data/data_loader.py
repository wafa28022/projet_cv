import os
import cv2
import torch
from torch.utils.data import Dataset, random_split, DataLoader, WeightedRandomSampler
from torchvision import transforms


CLASSES      = ['drinking', 'reach_side', 'safe_drive']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
STREAMS      = ['body', 'face', 'hands']


class DistractionDataset(Dataset):
    def __init__(self, frames_dir, img_size, seq_len=5, augment=False):
        self.frames_dir = frames_dir
        self.seq_len    = seq_len
        self.samples    = []

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

        for cls in CLASSES:
            stream_files = {}
            for stream in STREAMS:
                cls_dir = os.path.join(frames_dir, stream, cls)
                if not os.path.exists(cls_dir):
                    stream_files[stream] = {}
                    continue
                files = {}
                for f in os.listdir(cls_dir):
                    if not f.endswith('.jpg'):
                        continue
                    key = f.replace(f'_{stream}.jpg', '').replace('.jpg', '')
                    files[key] = f
                stream_files[stream] = files

            common_keys = sorted(
                set(stream_files['body'].keys())  &
                set(stream_files['face'].keys())  &
                set(stream_files['hands'].keys())
            )

            for i in range(0, len(common_keys) - seq_len + 1, seq_len):
                seq_keys = common_keys[i:i + seq_len]
                if len(seq_keys) == seq_len:
                    seq_files = [
                        (k,
                         stream_files['body'][k],
                         stream_files['face'][k],
                         stream_files['hands'][k])
                        for k in seq_keys
                    ]
                    self.samples.append((cls, seq_files))

        print(f'\n=== Dataset | seq_len={seq_len} | augment={augment} ===')
        print(f'Total séquences : {len(self.samples)}')
        for cls in CLASSES:
            count = sum(1 for c, _ in self.samples if c == cls)
            print(f'  {cls:20s} : {count} séquences')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cls, seq_files = self.samples[idx]

        body_seq  = []
        face_seq  = []
        hands_seq = []

        for key, body_f, face_f, hands_f in seq_files:
            for stream, fname, seq_list in [
                ('body',  body_f,  body_seq),
                ('face',  face_f,  face_seq),
                ('hands', hands_f, hands_seq)
            ]:
                path = os.path.join(self.frames_dir, stream, cls, fname)
                img  = cv2.imread(path)
                img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                seq_list.append(self.transform(img))

        return (
            torch.stack(body_seq),
            torch.stack(face_seq),
            torch.stack(hands_seq),
            torch.tensor(CLASS_TO_IDX[cls], dtype=torch.long)
        )

    def get_class_weights(self):
        """Couche 2 — poids pour CrossEntropyLoss."""
        counts = [0] * len(CLASSES)
        for cls, _ in self.samples:
            counts[CLASS_TO_IDX[cls]] += 1
        total   = len(self.samples)
        weights = [total / (len(CLASSES) * c) if c > 0 else 0
                   for c in counts]
        print(f'\nClass weights (loss) :')
        for i, cls in enumerate(CLASSES):
            print(f'  {cls:20s} : {weights[i]:.4f}')
        return torch.FloatTensor(weights)

    def get_sampler(self):
        """Couche 1 — WeightedRandomSampler pour équilibrer les batchs."""
        class_counts = [0] * len(CLASSES)
        for cls, _ in self.samples:
            class_counts[CLASS_TO_IDX[cls]] += 1

        sample_weights = []
        for cls, _ in self.samples:
            weight = 1.0 / class_counts[CLASS_TO_IDX[cls]]
            sample_weights.append(weight)

        print(f'\nSampler weights par classe :')
        for i, cls in enumerate(CLASSES):
            print(f'  {cls:20s} : 1/{class_counts[i]} = {1/class_counts[i]:.4f}')

        return WeightedRandomSampler(
            weights     = sample_weights,
            num_samples = len(sample_weights),
            replacement = True
        )


def get_data_loaders(cfg):
    # Dataset complet pour split
    full_dataset = DistractionDataset(
        frames_dir = cfg['frames_dir'],
        img_size   = cfg['img_size'],
        seq_len    = cfg['seq_len'],
        augment    = False
    )

    total      = len(full_dataset)
    val_size   = int(total * cfg['val_split'])
    test_size  = int(total * cfg['test_split'])
    train_size = total - val_size - test_size

    _, val_ds, test_ds = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Dataset train avec augmentation (Couche 3)
    train_dataset = DistractionDataset(
        frames_dir = cfg['frames_dir'],
        img_size   = cfg['img_size'],
        seq_len    = cfg['seq_len'],
        augment    = True
    )

    # Couche 1 — WeightedRandomSampler
    sampler = train_dataset.get_sampler()

    # Les 3 DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size  = cfg['batch_size'],
        sampler     = sampler,
        num_workers = cfg['num_workers']
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg['batch_size'],
        shuffle     = False,
        num_workers = cfg['num_workers']
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = cfg['batch_size'],
        shuffle     = False,
        num_workers = cfg['num_workers']
    )

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
    print(f'\nDistribution batch :')
    for i, cls in enumerate(CLASSES):
        count = (labels == i).sum().item()
        print(f'  {cls:20s} : {count}')