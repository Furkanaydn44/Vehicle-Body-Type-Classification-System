"""
Mevcut modeli ek verilerle fine-tune eder (CPU'da ~30-40dk).
Kullanim: python finetune.py
"""
import os, json, time, copy, shutil, random
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, '..', 'dataset')
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VAL_DIR = os.path.join(DATASET_DIR, 'val')
SAVE_DIR = os.path.join(BASE_DIR, 'saved_model')
HISTORY_DIR = os.path.join(BASE_DIR, 'training_history')
MODEL_PATH = os.path.join(SAVE_DIR, 'best_model_efficientnet_v2s.pth')

device = torch.device('cpu')
print(f"Cihaz: {device}")

# 1) Mevcut modeli yukle
print("\n[1/3] Model yukleniyor...")
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
class_names = checkpoint['class_names']
num_classes = len(class_names)

model = models.efficientnet_v2_s(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(0.3), nn.Linear(model.classifier[-1].in_features, 256), nn.ReLU(),
    nn.BatchNorm1d(256), nn.Dropout(0.2), nn.Linear(256, num_classes)
)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
print(f"  Model yuklendi (Val Acc: {checkpoint['best_val_acc']:.4f})")
print(f"  Siniflar: {class_names}")

# 2) Data loaders
print("\n[2/3] Veri seti yukleniyor...")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.2)
])
val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

train_ds = datasets.ImageFolder(TRAIN_DIR, train_tf)
val_ds = datasets.ImageFolder(VAL_DIR, val_tf)
train_loader = DataLoader(train_ds, 16, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(val_ds, 16, shuffle=False, num_workers=0)

counts = torch.zeros(num_classes)
for _, l in train_ds.samples:
    counts[l] += 1
weights = (counts.sum() / (num_classes * counts)).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

for cls in class_names:
    tr_count = len(os.listdir(os.path.join(TRAIN_DIR, cls)))
    va_count = len(os.listdir(os.path.join(VAL_DIR, cls)))
    print(f"  {cls}: {tr_count} train, {va_count} val")

# 3) Fine-tune
print("\n[3/3] Fine-tune basliyor (10 epoch)...")
for p in model.parameters():
    p.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=2e-5, weight_decay=1e-4)
best_acc = checkpoint['best_val_acc']

for epoch in range(10):
    t0 = time.time()

    model.train()
    tl, tok, tn = 0, 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        tl += loss.item() * x.size(0)
        tok += (out.argmax(1) == y).sum().item()
        tn += y.size(0)

    model.eval()
    vl, vok, vn = 0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            vl += loss.item() * x.size(0)
            vok += (out.argmax(1) == y).sum().item()
            vn += y.size(0)

    ta, va_acc = tok / tn, vok / vn
    dt = time.time() - t0
    msg = f"  E{epoch+1:02d}/10 | TrA:{ta:.4f} VaA:{va_acc:.4f} | {dt:.0f}s"

    if va_acc > best_acc:
        best_acc = va_acc
        torch.save({
            'model_state_dict': copy.deepcopy(model.state_dict()),
            'class_names': class_names,
            'num_classes': num_classes,
            'best_val_acc': best_acc,
            'epoch': epoch + 1,
            'model_architecture': 'EfficientNetV2_S'
        }, MODEL_PATH)
        print(msg + f" >>> EN IYI: {best_acc:.4f}")
    else:
        print(msg)

# 4) Metrikleri guncelle
print("\nMetrikler hesaplaniyor...")
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

best_ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(best_ckpt['model_state_dict'])
model.eval()

preds, lbls = [], []
with torch.no_grad():
    for x, y in val_loader:
        preds.extend(model(x.to(device)).argmax(1).cpu().numpy())
        lbls.extend(y.numpy())

print("\n" + classification_report(lbls, preds, target_names=class_names))

report = classification_report(lbls, preds, target_names=class_names, output_dict=True, zero_division=0)
metrics = {
    'accuracy': float(accuracy_score(lbls, preds)),
    'precision_macro': float(precision_score(lbls, preds, average='macro', zero_division=0)),
    'recall_macro': float(recall_score(lbls, preds, average='macro', zero_division=0)),
    'f1_macro': float(f1_score(lbls, preds, average='macro', zero_division=0)),
    'precision_weighted': float(precision_score(lbls, preds, average='weighted', zero_division=0)),
    'recall_weighted': float(recall_score(lbls, preds, average='weighted', zero_division=0)),
    'f1_weighted': float(f1_score(lbls, preds, average='weighted', zero_division=0)),
    'confusion_matrix': confusion_matrix(lbls, preds, normalize='true').tolist(),
    'per_class': {}
}
for cls in class_names:
    if cls in report:
        metrics['per_class'][cls] = {
            'precision': float(report[cls]['precision']),
            'recall': float(report[cls]['recall']),
            'f1': float(report[cls]['f1-score']),
            'support': int(report[cls]['support'])
        }

with open(os.path.join(HISTORY_DIR, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\nAccuracy: {metrics['accuracy']*100:.1f}%")
print(f"F1 (macro): {metrics['f1_macro']*100:.1f}%")
print(f"\nTamamlandi! Flask'i yeniden baslatin: python app.py")
