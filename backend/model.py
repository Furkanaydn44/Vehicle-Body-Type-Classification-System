"""
Model Mimarisi: EfficientNetV2 Small Transfer Learning
Araba Gövde Tipi Sınıflandırma (8 Sınıf)

EfficientNetV2-S'in avantajları:
- Daha iyi parametre-verimliliği
- ImageNet ön eğitimli ağırlıklarla yüksek doğruluk
- 224x224 giriş boyutunu destekler
- Transfer learning için uygun hafif sınıflandırıcı başlık
"""

import torch
import torch.nn as nn
from torchvision import models

# 8 araba gövde tipi sınıfı
CLASS_NAMES = [
    'HATCHBACK', 'MICRO', 'OPEN_WHEEL', 'PICKUP',
    'SEDAN', 'STATION_WAGON', 'SUV', 'VAN'
]

CLASS_DISPLAY_NAMES = {
    'SUV': 'SUV',
    'VAN': 'Van',
    'STATION_WAGON': 'Station Wagon',
    'MICRO': 'Micro',
    'OPEN_WHEEL': 'Açık Tekerlekli',
    'SEDAN': 'Sedan',
    'HATCHBACK': 'Hatchback',
    'PICKUP': 'Pick Up'
}

NUM_CLASSES = len(CLASS_NAMES)


def create_model(num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=True, classifier_type='standard'):
    """
    EfficientNetV2-S tabanlı sınıflandırma modeli oluşturur.

    Args:
        num_classes: Sınıf sayısı (varsayılan 8)
        pretrained: ImageNet ağırlıklarını kullan
        freeze_backbone: Backbone katmanlarını dondur
        classifier_type: 'standard' veya 'simple'
            - standard: Dropout + 256 birimli ara katman + BatchNorm + final Linear
            - simple: Dropout + final Linear

    Returns:
        model: PyTorch model
    """
    if pretrained:
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        model = models.efficientnet_v2_s(weights=weights)
    else:
        model = models.efficientnet_v2_s(weights=None)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[-1].in_features

    if classifier_type == 'simple':
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
    else:
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

    return model


def unfreeze_last_n_blocks(model, n=4):
    """
    EfficientNetV2-S'in son n bloğunu eğitime açar (fine-tuning Phase 2).

    Args:
        model: EfficientNetV2-S modeli
        n: Açılacak blok sayısı
    """
    total_blocks = len(model.features)

    for i in range(total_blocks - n, total_blocks):
        for param in model.features[i].parameters():
            param.requires_grad = True

    return model


def get_model_size_mb(model):
    """Model boyutunu MB cinsinden hesaplar."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def load_trained_model(model_path, device='cpu'):
    """
    Eğitilmiş modeli yükler.

    Args:
        model_path: .pth dosyasının yolu
        device: 'cpu' veya 'cuda'

    Returns:
        model: Yüklenmiş model (eval modunda)
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    num_classes = checkpoint.get('num_classes', NUM_CLASSES)
    architecture = checkpoint.get('model_architecture', None)

    model = create_model(
        num_classes=num_classes,
        pretrained=False,
        freeze_backbone=False,
        classifier_type='standard'
    )

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        if 'classifier.3.weight' in str(e) or 'classifier.5.weight' in str(e) or 'size mismatch for classifier.1.weight' in str(e):
            model = create_model(
                num_classes=num_classes,
                pretrained=False,
                freeze_backbone=False,
                classifier_type='simple'
            )
            model.load_state_dict(state_dict)
        else:
            raise

    model.to(device)
    model.eval()
    return model
