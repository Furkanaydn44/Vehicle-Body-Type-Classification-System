"""
Model Eğitim Scripti
Google Colab veya lokal ortamda çalışır.

Kullanım (lokal):
    python train.py

Kullanım (Google Colab):
    1. Bu dosyayı Colab'a yükleyin
    2. Runtime → GPU seçin
    3. Hücrelerde çalıştırın
"""

import os
import sys
import json
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets

from model import create_model, unfreeze_last_n_blocks, get_model_size_mb
from preprocessing import (
    get_train_transforms, get_val_transforms,
    create_data_loaders, compute_class_weights
)

# ========================
# Hiperparametreler
# ========================
BATCH_SIZE = 32
MAX_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 7
LR_PATIENCE = 3
LR_FACTOR = 0.5
PHASE1_EPOCHS = 5
UNFREEZE_BLOCKS = 4

# ========================
# Yollar
# ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.environ.get('DATASET_DIR', os.path.join(BASE_DIR, '..', 'dataset'))
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VAL_DIR = os.path.join(DATASET_DIR, 'val')
SAVE_DIR = os.path.join(BASE_DIR, 'saved_model')
HISTORY_DIR = os.path.join(BASE_DIR, 'training_history')
BEST_MODEL_FILENAME = 'best_model_efficientnet_v2s.pth'
BEST_MODEL_PATH = os.path.join(SAVE_DIR, BEST_MODEL_FILENAME)


class EarlyStopping:
    """Validation loss iyileşmezse eğitimi erken durdurur."""

    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"  -> EarlyStopping! ({self.patience} epoch iyilesme yok)")
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Tek bir epoch eğitim döngüsü."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def validate(model, dataloader, criterion, device):
    """Validation değerlendirmesi."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def train_model():
    """Ana eğitim fonksiyonu."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"{'='*60}")
    print(f"  Araba Govde Tipi Siniflandirma - Model Egitimi")
    print(f"{'='*60}")
    print(f"  Cihaz: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Veri yükleyicileri
    print(f"\n  Veri seti yukleniyor...")
    num_workers = 2 if device.type == 'cuda' else 0
    train_loader, val_loader, discovered_classes = create_data_loaders(
        TRAIN_DIR, VAL_DIR,
        batch_size=BATCH_SIZE, num_workers=num_workers
    )

    print(f"  Siniflar: {discovered_classes}")
    print(f"  Egitim: {len(train_loader.dataset)} ornek")
    print(f"  Dogrulama: {len(val_loader.dataset)} ornek")

    # Sinif agirliklari
    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=get_train_transforms())
    class_weights = compute_class_weights(train_dataset).to(device)
    print(f"  Sinif agirliklari: {class_weights.cpu().numpy().round(3)}")

    # Model
    num_classes = len(discovered_classes)
    model = create_model(num_classes=num_classes, pretrained=True, freeze_backbone=True)
    model = model.to(device)
    print(f"\n  Model: EfficientNetV2-S ({get_model_size_mb(model):.2f} MB)")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'lr': [], 'epoch_times': []
    }

    best_val_acc = 0.0
    best_model_state = None
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)

    # ========================
    # PHASE 1: Sadece Classifier
    # ========================
    print(f"\n{'='*60}")
    print(f"  PHASE 1: Classifier Egitimi ({PHASE1_EPOCHS} epoch)")
    print(f"{'='*60}")

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', patience=LR_PATIENCE,
        factor=LR_FACTOR
    )
    early_stopping = EarlyStopping(patience=PATIENCE)

    for epoch in range(MAX_EPOCHS):
        start_time = time.time()

        # Phase 2'ye geçiş
        if epoch == PHASE1_EPOCHS:
            print(f"\n{'='*60}")
            print(f"  PHASE 2: Fine-Tuning (Son {UNFREEZE_BLOCKS} blok)")
            print(f"{'='*60}")
            model = unfreeze_last_n_blocks(model, n=UNFREEZE_BLOCKS)
            optimizer = optim.Adam([
                {'params': model.features.parameters(), 'lr': LEARNING_RATE * 0.1},
                {'params': model.classifier.parameters(), 'lr': LEARNING_RATE}
            ], weight_decay=WEIGHT_DECAY)
            scheduler = ReduceLROnPlateau(
                optimizer, mode='min', patience=LR_PATIENCE,
                factor=LR_FACTOR
            )

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['train_acc'].append(float(train_acc))
        history['val_acc'].append(float(val_acc))
        history['lr'].append(float(current_lr))
        history['epoch_times'].append(float(epoch_time))

        phase = "P1" if epoch < PHASE1_EPOCHS else "P2"
        print(f"  [{phase}] Epoch {epoch+1}/{MAX_EPOCHS} "
              f"| TrLoss: {train_loss:.4f} | TrAcc: {train_acc:.4f} "
              f"| VaLoss: {val_loss:.4f} | VaAcc: {val_acc:.4f} "
              f"| LR: {current_lr:.6f} | {epoch_time:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            save_path = BEST_MODEL_PATH
            torch.save({
                'model_state_dict': best_model_state,
                'class_names': discovered_classes,
                'num_classes': num_classes,
                'best_val_acc': best_val_acc,
                'epoch': epoch + 1,
                'model_architecture': 'EfficientNetV2_S'
            }, save_path)
            print(f"  [*] En iyi model kaydedildi! (Val Acc: {best_val_acc:.4f})")

        early_stopping(val_loss)
        if early_stopping.should_stop:
            print(f"\n  Erken durma (Epoch {epoch+1})")
            break

    # Geçmişi kaydet
    history_data = {
        **{k: v for k, v in history.items()},
        'class_names': discovered_classes,
        'best_val_acc': float(best_val_acc),
        'total_epochs': len(history['train_loss']),
        'hyperparameters': {
            'batch_size': BATCH_SIZE,
            'max_epochs': MAX_EPOCHS,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'patience': PATIENCE,
            'lr_patience': LR_PATIENCE,
            'lr_factor': LR_FACTOR,
            'phase1_epochs': PHASE1_EPOCHS,
            'unfreeze_blocks': UNFREEZE_BLOCKS,
            'model_architecture': 'EfficientNetV2_S',
            'input_size': 224
        }
    }

    history_path = os.path.join(HISTORY_DIR, 'history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  Egitim Tamamlandi!")
    print(f"  En Iyi Val Accuracy: {best_val_acc:.4f}")
    print(f"  Model: {BEST_MODEL_PATH}")
    print(f"  Gecmis: {history_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    train_model()
