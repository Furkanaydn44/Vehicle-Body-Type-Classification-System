"""
Kaggle Veri Seti Duzenleme

Indirilen ham verileri 8 sinifa gore organize eder.
Veri setleri zaten raw_data/ altinda mevcut.

Kullanim:
    python prepare_dataset.py
"""

import os
import shutil
import random
from pathlib import Path

# Hedef 8 sinif
TARGET_CLASSES = [
    'HATCHBACK', 'MICRO', 'OPEN_WHEEL', 'PICKUP',
    'SEDAN', 'STATION_WAGON', 'SUV', 'VAN'
]

# Gercek klasor adlari -> hedef sinif eslestirmesi
# Dataset 1 (Cars_Body_Type): Convertible, Coupe, Hatchback, Pick-Up, Sedan, SUV, VAN
# Dataset 2 (stanford_cars_type): Cab, Convertible, Coupe, Hatchback, Minivan, Other, Sedan, SUV, Van, Wagon
CLASS_MAPPING = {
    # --- Dataset 1: Cars_Body_Type ---
    'SUV': 'SUV',
    'Sedan': 'SEDAN',
    'Hatchback': 'HATCHBACK',
    'Pick-Up': 'PICKUP',
    'VAN': 'VAN',
    'Convertible': 'OPEN_WHEEL',
    'Coupe': 'MICRO',
    # --- Dataset 2: stanford_cars_type ---
    'Cab': 'PICKUP',
    'Minivan': 'VAN',
    'Van': 'VAN',
    'Wagon': 'STATION_WAGON',
    'Other': 'STATION_WAGON',
    # Sedan, SUV, Hatchback, Convertible, Coupe zaten yukarida var
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, '..', 'dataset')
RAW_DIR = os.path.join(BASE_DIR, '..', 'raw_data')
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VAL_DIR = os.path.join(DATASET_DIR, 'val')

TRAIN_RATIO = 0.85


def organize_dataset():
    """Indirilen ham verileri 8 sinifa gore duzenler."""

    # Eski verileri temizle
    for cls in TARGET_CLASSES:
        train_cls = os.path.join(TRAIN_DIR, cls)
        val_cls = os.path.join(VAL_DIR, cls)
        if os.path.exists(train_cls):
            shutil.rmtree(train_cls)
        if os.path.exists(val_cls):
            shutil.rmtree(val_cls)
        os.makedirs(train_cls, exist_ok=True)
        os.makedirs(val_cls, exist_ok=True)

    class_images = {cls: [] for cls in TARGET_CLASSES}
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    if not os.path.exists(RAW_DIR):
        print("  Ham veri bulunamadi! raw_data/ klasoru yok.")
        return

    # Tum alt klasorleri tara
    for root, dirs, files in os.walk(RAW_DIR):
        folder_name = os.path.basename(root)
        target_class = CLASS_MAPPING.get(folder_name)

        if target_class:
            direct_images = [
                os.path.join(root, f) for f in files
                if Path(f).suffix.lower() in valid_extensions
            ]
            if direct_images:
                class_images[target_class].extend(direct_images)
                print(f"  {folder_name} -> {target_class}: {len(direct_images)} gorsel")

    # Train/Val bolme ve kopyalama
    print(f"\n  Veri seti bolunuyor (Train: {TRAIN_RATIO*100:.0f}%, Val: {(1-TRAIN_RATIO)*100:.0f}%)...")

    total_train = 0
    total_val = 0

    for cls, images in class_images.items():
        if not images:
            print(f"  [!] {cls} sinifinda gorsel bulunamadi!")
            continue

        random.shuffle(images)
        split_idx = int(len(images) * TRAIN_RATIO)

        train_images = images[:split_idx]
        val_images = images[split_idx:]

        for i, src in enumerate(train_images):
            ext = Path(src).suffix
            dst = os.path.join(TRAIN_DIR, cls, f"{cls}_{i:04d}{ext}")
            shutil.copy2(src, dst)

        for i, src in enumerate(val_images):
            ext = Path(src).suffix
            dst = os.path.join(VAL_DIR, cls, f"{cls}_{i:04d}{ext}")
            shutil.copy2(src, dst)

        total_train += len(train_images)
        total_val += len(val_images)
        print(f"  {cls}: {len(train_images)} train, {len(val_images)} val")

    print(f"\n  Toplam: {total_train} train, {total_val} val")


def print_dataset_stats():
    """Veri seti istatistiklerini yazdirir."""
    print(f"\n{'='*50}")
    print(f"  Veri Seti Istatistikleri")
    print(f"{'='*50}")

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    for split, split_dir in [('Train', TRAIN_DIR), ('Val', VAL_DIR)]:
        print(f"\n  {split}:")
        if not os.path.exists(split_dir):
            print(f"    Dizin bulunamadi: {split_dir}")
            continue
        for cls in sorted(os.listdir(split_dir)):
            cls_dir = os.path.join(split_dir, cls)
            if os.path.isdir(cls_dir):
                count = len([f for f in os.listdir(cls_dir)
                           if Path(f).suffix.lower() in valid_extensions])
                print(f"    {cls}: {count} gorsel")


if __name__ == '__main__':
    print(f"{'='*50}")
    print(f"  Veri Seti Hazirlama")
    print(f"{'='*50}")

    print("\n  Veri seti duzenleniyor...")
    organize_dataset()

    print("\n  Istatistikler...")
    print_dataset_stats()

    print(f"\n  [OK] Tamamlandi!")
