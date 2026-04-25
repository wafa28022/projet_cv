"""
Evaluation du meilleur modèle sauvegardé.
Génère : confusion matrix, classification report, accuracy par classe.
Usage : python src/training/evaluate.py
"""

import sys
sys.path.append('.')

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
from tqdm import tqdm

from src.utils.config import CFG
from src.data.data_loader import get_data_loaders
from src.models.model import get_model


CLASSES = ['drinking', 'reach_side', 'safe_drive']


def evaluate_model(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')

    # Charge les données
    print('\n--- Chargement des données ---')
    _, val_loader, test_loader = get_data_loaders(cfg)

    # Charge le modèle
    print('\n--- Chargement du meilleur modèle ---')
    model = get_model(cfg).to(device)
    checkpoint = torch.load(
        f'{cfg["checkpoint_dir"]}/best_model.pth',
        map_location=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Modèle chargé — epoch {checkpoint["epoch"]}')
    print(f'Val accuracy sauvegardée : {checkpoint["val_acc"]:.1f}%')

    # Evaluation sur test
    print('\n--- Evaluation sur Test Set ---')
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for body, face, hands, labels in tqdm(test_loader, desc='Test'):
            body   = body.to(device)
            face   = face.to(device)
            hands  = hands.to(device)

            outputs   = model(body, face, hands)
            predicted = outputs.argmax(dim=1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Accuracy globale
    accuracy = (all_preds == all_labels).mean() * 100
    print(f'\nTest Accuracy : {accuracy:.1f}%')

    # Classification report
    print('\n--- Classification Report ---')
    print(classification_report(
        all_labels, all_preds,
        target_names=CLASSES,
        digits=3
    ))

    # Confusion Matrix
    Path('results/plots').mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(all_labels, all_preds)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Confusion matrix — valeurs absolues
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[0])
    axes[0].set_title('Confusion Matrix (counts)')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    # Confusion matrix — pourcentages
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[1])
    axes[1].set_title('Confusion Matrix (%)')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig('results/plots/confusion_matrix.png', dpi=150)
    plt.close()
    print('\nConfusion matrix sauvegardée : results/plots/confusion_matrix.png')

    # Accuracy par classe
    print('\n--- Accuracy par classe ---')
    for i, cls in enumerate(CLASSES):
        mask     = all_labels == i
        cls_acc  = (all_preds[mask] == all_labels[mask]).mean() * 100
        print(f'  {cls:20s} : {cls_acc:.1f}%')

    return accuracy, all_preds, all_labels


if __name__ == '__main__':
    # Install sklearn si pas présent
    try:
        from sklearn.metrics import confusion_matrix
    except ImportError:
        import subprocess
        subprocess.run(['pip', 'install', 'scikit-learn', 'seaborn',
                       '--break-system-packages'])

    accuracy, preds, labels = evaluate_model(CFG)