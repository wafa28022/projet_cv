"""
Training script pour Multi-Stream CNN Driver Distraction Detection.
Usage : python src/training/train.py
"""

import sys
import os
sys.path.append('.')

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

from src.data.data_loader import get_data_loaders, DistractionDataset
from src.models.model import get_model


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Une epoch d'entraînement."""
    model.train()
    total_loss = 0
    correct    = 0
    total      = 0

    for body, face, hands, labels in tqdm(loader, desc='Train'):
        body   = body.to(device)
        face   = face.to(device)
        hands  = hands.to(device)
        labels = labels.to(device)
       
        
        optimizer.zero_grad()
        outputs = model(body, face, hands)
        loss    = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted   = outputs.argmax(dim=1)
        correct    += (predicted == labels).sum().item()
        total      += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total * 100
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """Evaluation sur val ou test."""
    model.eval()
    total_loss = 0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for body, face, hands, labels in tqdm(loader, desc='Eval '):
            body   = body.to(device)
            face   = face.to(device)
            hands  = hands.to(device)
            labels = labels.to(device)

            outputs = model(body, face, hands)
            loss    = criterion(outputs, labels)

            total_loss += loss.item()
            predicted   = outputs.argmax(dim=1)
            correct    += (predicted == labels).sum().item()
            total      += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total * 100
    return avg_loss, accuracy


def plot_curves(history, save_dir):
    """Sauvegarde les courbes loss et accuracy."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'],   label='Val')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'],   label='Val')
    axes[1].set_title('Accuracy (%)')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_curves.png', dpi=150)
    plt.close()
    print(f'Courbes sauvegardées : {save_dir}_training_curves.png')


def train(cfg,model, train_loader, val_loader, device):
    

    # Dossiers
    Path(cfg['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path('results/plots').mkdir(parents=True, exist_ok=True)

  

    # Class weights
    full_ds = DistractionDataset(
        frames_dir = cfg['frames_dir'],
        img_size   = cfg['img_size'],
        seq_len    = cfg['seq_len'],
        augment    = False
    )
    class_weights = full_ds.get_class_weights().to(device)
    print(f'Class weights : {class_weights}')

    # Loss + Optimizer + Scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )

    # Training loop
    print(f'\n--- Training ({cfg["epochs"]} epochs) ---\n')
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc' : [], 'val_acc' : []
    }

    best_val_acc     = 0
    best_epoch       = 0
    patience         = 7
    patience_counter = 0

    for epoch in range(cfg['epochs']):
        print(f'\nEpoch [{epoch+1}/{cfg["epochs"]}]')

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}%')
        print(f'Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.1f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch + 1
            torch.save({
                'epoch'              : epoch + 1,
                'model_state_dict'   : model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc'            : val_acc,
                'val_loss'           : val_loss,
            }, f'{cfg["checkpoint_dir"]}/{cfg["experiment"]}.pth')
            print(f'  ✓ Meilleur modèle sauvegardé (val_acc={val_acc:.1f}%)')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping à epoch {epoch+1}')
                break

    print(f'\nMeilleur modèle : epoch {best_epoch} | val_acc={best_val_acc:.1f}%')

    plot_curves(history, f'results/plots/{cfg["experiment"]}')

    

    return model, history


if __name__ == '__main__':
    model, history = train(CFG)