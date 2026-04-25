"""
Multi-Stream CNN for Driver Distraction Detection.

Architecture :
    3 streams (body, face, hands)
    → 3 ResNet18 encoders independants
    → Fusion par concatenation
    → Classifier fully connected
    → 3 classes (drinking, reach_side, safe_drive)

Input  : body  [B, T, 3, 224, 224]
         face  [B, T, 3, 224, 224]
         hands [B, T, 3, 224, 224]
Output : logits [B, num_classes]
"""

import torch
import torch.nn as nn
from torchvision import models


class StreamEncoder(nn.Module):
    """
    Encodeur pour un seul stream.
    ResNet18 pre-entraine → features [512]
    Traite une sequence de T frames et retourne
    un vecteur de features moyen.
    """
    def __init__(self, pretrained=True):
        super().__init__()

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )

        # Enleve la derniere couche fc
        # garde tout jusqu a AdaptiveAvgPool2d
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, x):
        """
        Input  : x [B, T, 3, 224, 224]
        Output : features [B, 512]
        """
        B, T, C, H, W = x.shape

        # Aplatis batch et temps pour traiter toutes les frames
        x = x.view(B * T, C, H, W)     # [B*T, 3, 224, 224]
        x = self.encoder(x)             # [B*T, 512, 1, 1]
        x = x.view(B, T, 512)          # [B, T, 512]

        # Moyenne temporelle sur les T frames
        x = x.mean(dim=1)              # [B, 512]
        return x


class MultiStreamCNN(nn.Module):
    """
    Architecture principale multi-stream.

    3 StreamEncoder independants
    → concatenation [B, 1536]
    → Fusion MLP
    → logits [B, num_classes]
    """
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()

        # Un encodeur independant par stream
        self.body_encoder  = StreamEncoder(pretrained=pretrained)
        self.face_encoder  = StreamEncoder(pretrained=pretrained)
        self.hands_encoder = StreamEncoder(pretrained=pretrained)

        # Couche de fusion
        # 512 * 3 streams = 1536 features concatenes
        self.fusion = nn.Sequential(
            nn.Linear(512 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, body, face, hands):
        """
        Input  : body  [B, T, 3, 224, 224]
                 face  [B, T, 3, 224, 224]
                 hands [B, T, 3, 224, 224]
        Output : logits [B, num_classes]
        """
        # Encode chaque stream independamment
        body_feat  = self.body_encoder(body)    # [B, 512]
        face_feat  = self.face_encoder(face)    # [B, 512]
        hands_feat = self.hands_encoder(hands)  # [B, 512]

        # Concatenation des 3 vecteurs de features
        fused = torch.cat(
            [body_feat, face_feat, hands_feat], dim=1
        )  # [B, 1536]

        # Classification finale
        logits = self.fusion(fused)  # [B, num_classes]
        return logits


def get_model(cfg):
    """
    Factory function — retourne le modele configure.
    """
    model = MultiStreamCNN(
        num_classes=cfg['num_classes'],
        pretrained=cfg['pretrained']
    )

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)

    print(f'Modele    : MultiStreamCNN')
    print(f'Streams   : body + face + hands')
    print(f'Classes   : {cfg["num_classes"]}')
    print(f'Pretrained: {cfg["pretrained"]}')
    print(f'Params total     : {total:,}')
    print(f'Params trainable : {trainable:,}')

    return model


if __name__ == '__main__':
    import sys
    sys.path.append('.')
    from src.utils.config import CFG

    print('=== Test Architecture ===\n')
    model = get_model(CFG)

    # Test avec un batch factice
    B, T = 2, 5
    body  = torch.randn(B, T, 3, 224, 224)
    face  = torch.randn(B, T, 3, 224, 224)
    hands = torch.randn(B, T, 3, 224, 224)

    output = model(body, face, hands)

    print(f'\nInput  body  : {body.shape}')
    print(f'Input  face  : {face.shape}')
    print(f'Input  hands : {hands.shape}')
    print(f'Output logits: {output.shape}')
    print(f'Logits       : {output}')