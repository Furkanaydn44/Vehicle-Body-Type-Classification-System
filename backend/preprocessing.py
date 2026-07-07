"""
Veri Ön İşleme ve Data Augmentation

Eğitim sırasında veri çoğaltma (augmentation) uygulanarak
modelin genelleme kapasitesi artırılır.
"""

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch

# ImageNet normalizasyon değerleri (EfficientNetV2-S için de uygundur)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Giriş boyutu (EfficientNetV2-S için 224x224)
INPUT_SIZE = 224


def get_train_transforms():
    """
    Eğitim veri seti için augmentation transform pipeline'ı.
    Farklı açılar, ışık koşulları ve arka planlar simüle edilir.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3,
            saturation=0.3, hue=0.1
        ),
        transforms.RandomAffine(
            degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))
    ])


def get_val_transforms():
    """Validation/test veri seti için temiz transform pipeline'ı."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_inference_transforms():
    """Tek görsel inference (tahmin) için transform pipeline'ı."""
    # Pad to square (keeps full object, e.g., bagaj/trunk), then resize
    class PadToSquare:
        def __init__(self, fill=128):
            self.fill = fill
        def __call__(self, img):
            if img.mode != 'RGB':
                img = img.convert('RGB')
            w, h = img.size
            max_wh = max(w, h)
            pad_left = (max_wh - w) // 2
            pad_top = (max_wh - h) // 2
            pad_right = max_wh - w - pad_left
            pad_bottom = max_wh - h - pad_top
            return transforms.functional.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)

    return transforms.Compose([
        PadToSquare(fill=128),         # Do not cut off trunk/vehicle parts
        transforms.Resize((384, 384)), # Use a larger inference size for detail
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def create_data_loaders(train_dir, val_dir, batch_size=32, num_workers=4):
    """
    Eğitim ve validation veri yükleyicilerini oluşturur.
    ImageFolder formatı: her sınıf ayrı klasörde.
    """
    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=get_train_transforms()
    )

    val_dataset = datasets.ImageFolder(
        root=val_dir,
        transform=get_val_transforms()
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, list(train_dataset.classes)


def compute_class_weights(dataset):
    """
    Sınıf dengesizliğini gidermek için ağırlıkları hesaplar.
    Az örnekli sınıflara daha yüksek ağırlık verilir.
    """
    class_counts = torch.zeros(len(dataset.classes))
    for _, label in dataset.samples:
        class_counts[label] += 1

    total = class_counts.sum()
    weights = total / (len(dataset.classes) * class_counts)

    return weights
