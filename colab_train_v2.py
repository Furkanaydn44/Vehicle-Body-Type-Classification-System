# ================================================================
# VisionAI Core - Geliştirilmiş Eğitim (MICRO/SEDAN Düzeltme)
# Google Colab - Tek Hücre
# ================================================================
# KURULUM
import subprocess
subprocess.run(['pip', 'install', '-q', 'scikit-learn'], check=True)

# DRIVE BAĞLA
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

# ========================
# AYARLAR — BURAYA BAK
# ========================
DRIVE_DATASET = '/content/drive/MyDrive/dataset'   # train/ ve val/ klasörlerinin parent'ı
DRIVE_SAVE    = '/content/drive/MyDrive/best_model_efficientnet_v2s.pth'
BATCH_SIZE    = 32
MAX_EPOCHS    = 60
LR_HEAD       = 1e-3
LR_BACKBONE   = 5e-5
WEIGHT_DECAY  = 1e-4
PATIENCE      = 10

# ========================
# IMPORTS
# ========================
import os, json, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from sklearn.metrics import f1_score, classification_report

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Cihaz: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ========================
# FOCAL LOSS
# Zor örneklere (MICRO/SEDAN sınırı) daha fazla odaklanır
# gamma=2: kolay doğruları ağırlıklandırmaz, zor olanları cezalandırır
# ========================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce   = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt   = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()

# ========================
# MIXUP AUGMENTATION
# İki görseli karıştırarak sınıf sınırlarını güçlendirir
# MICRO ile SEDAN arasındaki geçişi daha iyi öğrenir
# ========================
def mixup_data(x, y, alpha=0.3):
    lam   = np.random.beta(alpha, alpha) if alpha > 0 else 1
    idx   = torch.randperm(x.size(0)).to(device)
    mixed = lam * x + (1 - lam) * x[idx]
    return mixed, y, y[idx], lam

def mixup_loss(criterion, pred, ya, yb, lam):
    return lam * criterion(pred, ya) + (1 - lam) * criterion(pred, yb)

# ========================
# TRANSFORMS
# ========================
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.65, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.2))
])

val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# ========================
# VERİ YÜKLEYİCİ
# WeightedRandomSampler: Az örnekli sınıfları (MICRO, HATCHBACK) her batch'te dengeli örnekler
# ========================
train_ds = datasets.ImageFolder(os.path.join(DRIVE_DATASET, 'train'), transform=train_tf)
val_ds   = datasets.ImageFolder(os.path.join(DRIVE_DATASET, 'val'),   transform=val_tf)

classes    = train_ds.classes
n_classes  = len(classes)
print(f"\nSınıflar: {classes}")

# Sınıf sayıları
counts = torch.zeros(n_classes)
for _, lbl in train_ds.samples:
    counts[lbl] += 1
print("\nEğitim dağılımı:")
for cls, cnt in zip(classes, counts):
    print(f"  {cls}: {int(cnt)}")

# Kayıp ağırlıkları (seyrek sınıflara daha fazla ceza)
cls_weights = (counts.sum() / (n_classes * counts)).to(device)

# WeightedRandomSampler
sample_w = [cls_weights[lbl].item() for _, lbl in train_ds.samples]
sampler  = WeightedRandomSampler(sample_w, len(train_ds), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,   num_workers=2, pin_memory=True)

# ========================
# MODEL — EfficientNetV2-S tabanlı sınıflandırıcı
# ========================
weights_pt = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
model      = models.efficientnet_v2_s(weights=weights_pt)

# Tüm backbone'u dondur (Phase 1)
for p in model.features.parameters():
    p.requires_grad = False

in_ft = model.classifier[-1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_ft, 256),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Dropout(p=0.2),
    nn.Linear(256, n_classes)
)
model = model.to(device)

mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
print(f"\nModel: EfficientNetV2-S ({mb:.1f} MB)")

# ========================
# EARLY STOPPING — F1 bazlı (loss değil)
# F1 birincil metrik olduğu için buna göre duruyoruz
# ========================
class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.best_f1  = None
        self.counter  = 0
        self.stop     = False

    def __call__(self, f1):
        if self.best_f1 is None or f1 > self.best_f1 + 0.001:
            self.best_f1 = f1
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
                print(f"  → EarlyStopping! (F1 {self.patience} epoch iyileşmedi)")

# ========================
# EĞİTİM FONKSİYONLARI
# ========================
def train_epoch(model, loader, criterion, optimizer, use_mixup=True):
    model.train()
    total_loss = correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        if use_mixup and np.random.random() > 0.5:
            xm, ya, yb, lam = mixup_data(x, y)
            out  = model(xm)
            loss = mixup_loss(criterion, out, ya, yb, lam)
        else:
            out  = model(x)
            loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        _, pred = out.max(1)
        total   += y.size(0)
        correct += pred.eq(y).sum().item()
    return total_loss / total, correct / total

def validate(model, loader, criterion):
    model.eval()
    total_loss = correct = total = 0
    preds_all, labels_all = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out  = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            _, pred = out.max(1)
            total   += y.size(0)
            correct += pred.eq(y).sum().item()
            preds_all.extend(pred.cpu().numpy())
            labels_all.extend(y.cpu().numpy())
    f1 = f1_score(labels_all, preds_all, average='macro', zero_division=0)
    return total_loss / total, correct / total, f1

# ========================
# PHASE 1 — Sadece Classifier (5 epoch, MixUp kapalı)
# ========================
print(f"\n{'='*55}")
print("  PHASE 1: Classifier Eğitimi (5 epoch)")
print(f"{'='*55}")

criterion = FocalLoss(gamma=2.0, weight=cls_weights)
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR_HEAD, weight_decay=WEIGHT_DECAY
)

history      = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}
best_f1      = 0.0
best_state   = None
early_stop   = EarlyStopping(PATIENCE)
PHASE1       = 5

for ep in range(PHASE1):
    t0 = time.time()
    tl, ta = train_epoch(model, train_loader, criterion, optimizer, use_mixup=False)
    vl, va, vf1 = validate(model, val_loader, criterion)
    lr = optimizer.param_groups[0]['lr']
    history['train_loss'].append(tl); history['val_loss'].append(vl)
    history['train_acc'].append(ta);  history['val_acc'].append(va); history['lr'].append(lr)
    print(f"  [P1] {ep+1}/{PHASE1} | TrL:{tl:.4f} TrA:{ta:.3f} | VaL:{vl:.4f} VaA:{va:.3f} F1:{vf1:.3f} | {time.time()-t0:.0f}s")
    if vf1 > best_f1:
        best_f1    = vf1
        best_state = copy.deepcopy(model.state_dict())

# ========================
# PHASE 2 — Fine-Tuning: Son 8 blok açılır + CosineAnnealingLR + MixUp
# Son 8 blok açmak, modelin araç gövde özelliklerini daha iyi öğrenmesini sağlar
# ========================
print(f"\n{'='*55}")
print("  PHASE 2: Fine-Tuning (Son 8 blok + MixUp + Cosine LR)")
print(f"{'='*55}")

total_blocks = len(model.features)
for i in range(total_blocks - 8, total_blocks):
    for p in model.features[i].parameters():
        p.requires_grad = True

optimizer = optim.Adam([
    {'params': model.features.parameters(),   'lr': LR_BACKBONE},
    {'params': model.classifier.parameters(), 'lr': LR_HEAD}
], weight_decay=WEIGHT_DECAY)

scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS - PHASE1, eta_min=1e-6)

for ep in range(PHASE1, MAX_EPOCHS):
    t0 = time.time()
    tl, ta = train_epoch(model, train_loader, criterion, optimizer, use_mixup=True)
    vl, va, vf1 = validate(model, val_loader, criterion)
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    history['train_loss'].append(tl); history['val_loss'].append(vl)
    history['train_acc'].append(ta);  history['val_acc'].append(va); history['lr'].append(lr)
    mark = " ← En iyi!" if vf1 > best_f1 else ""
    print(f"  [P2] {ep+1}/{MAX_EPOCHS} | TrL:{tl:.4f} TrA:{ta:.3f} | VaL:{vl:.4f} VaA:{va:.3f} F1:{vf1:.3f} | LR:{lr:.1e} | {time.time()-t0:.0f}s{mark}")
    if vf1 > best_f1:
        best_f1    = vf1
        best_state = copy.deepcopy(model.state_dict())
    early_stop(vf1)
    if early_stop.stop:
        break

# ========================
# KAYDET
# ========================
print(f"\n{'='*55}")
print(f"  En İyi Val F1: {best_f1:.4f} ({best_f1*100:.2f}%)")
print(f"{'='*55}")

model.load_state_dict(best_state)
torch.save({
    'model_state_dict': best_state,
    'class_names':      classes,
    'num_classes':      n_classes,
    'best_val_f1':      float(best_f1),
    'total_epochs':     len(history['train_loss']),
    'model_architecture': 'EfficientNetV2_S'
}, DRIVE_SAVE)
print(f"  ✓ Model kaydedildi: {DRIVE_SAVE}")

# History (mevcut backend history.json formatıyla uyumlu)
history_data = {
    **history,
    'class_names':  classes,
    'best_val_acc': float(max(history['val_acc'])),
    'best_val_f1':  float(best_f1),
    'total_epochs': len(history['train_loss']),
    'hyperparameters': {
        'batch_size': BATCH_SIZE, 'max_epochs': MAX_EPOCHS,
        'learning_rate': LR_HEAD, 'weight_decay': WEIGHT_DECAY,
        'patience': PATIENCE, 'loss': 'FocalLoss(gamma=2)',
        'sampler': 'WeightedRandomSampler', 'mixup': True,
        'scheduler': 'CosineAnnealingLR', 'phase1_epochs': PHASE1,
        'unfreeze_blocks': 8, 'model_architecture': 'EfficientNetV2_S', 'input_size': 224
    }
}
history_path = DRIVE_SAVE.replace('.pth', '_history.json')
with open(history_path, 'w') as f:
    json.dump(history_data, f, indent=2)
print(f"  ✓ History kaydedildi: {history_path}")

# ========================
# FINAL RAPOR
# ========================
model.eval()
all_p, all_l = [], []
with torch.no_grad():
    for x, y in val_loader:
        out = model(x.to(device))
        _, pred = out.max(1)
        all_p.extend(pred.cpu().numpy())
        all_l.extend(y.cpu().numpy())

print(f"\n{'='*55}")
print(classification_report(all_l, all_p, target_names=classes, digits=3, zero_division=0))
print(f"{'='*55}")
print("\nBitti! Drive'dan best_model_efficientnet_v2s.pth ve best_model_history.json indirin.")
print("Dosyaları backend/saved_model/ ve backend/training_history/ klasörlerine koyun.")
