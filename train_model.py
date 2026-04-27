from src.data.data_loader import get_data_loaders
from src.models.model import get_model
from src.training.train import train
import torch


run=0
cfg= {
    'frames_dir'     : 'data/processed/frames_mosaic',
    'img_size'       : (224, 224),
    'seq_len'        : 5,
    'val_split'      : 0.15,
    'test_split'     : 0.15,
    'batch_size'     : 8,
    'num_workers'    : 0,
    'epochs'         : 1,
    'lr'             : 1e-4,
    'weight_decay'   : 1e-4,
    'num_classes'    : 3,
    'pretrained'     : True,
    'checkpoint_dir' : 'results/checkpoints',
    'experiment': f'MultiStreamCNN_run_{run}',
}

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nDevice : {device}')
# Data
print('\n--- Chargement des données ---')
train_loader, val_loader, test_loader = get_data_loaders(cfg)

# Modèle
print('\n--- Modèle ---')
model = get_model(cfg).to(device)

#train
train(cfg, model, train_loader, val_loader, device)

