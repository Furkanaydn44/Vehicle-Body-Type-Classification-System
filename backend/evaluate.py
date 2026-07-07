"""
Model Değerlendirme ve Görselleştirme

Metrikler: Accuracy, Precision, Recall, F1-Score
Grafikler: Loss, Accuracy, Confusion Matrix
"""

import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
from torchvision import datasets
from torch.utils.data import DataLoader

from model import load_trained_model, CLASS_NAMES
from preprocessing import get_val_transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, 'saved_model')
HISTORY_DIR = os.path.join(BASE_DIR, 'training_history')
PLOTS_DIR = os.path.join(BASE_DIR, 'training_history', 'plots')


def evaluate_model(model, dataloader, device, class_names):
    """Model performansını değerlendirir ve tüm metrikleri döndürür."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Sınıf bazlı metrikler
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    # Genel metrikler
    metrics = {
        'accuracy': float(accuracy_score(all_labels, all_preds)),
        'precision_macro': float(precision_score(all_labels, all_preds, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(all_labels, all_preds, average='macro', zero_division=0)),
        'f1_macro': float(f1_score(all_labels, all_preds, average='macro', zero_division=0)),
        'precision_weighted': float(precision_score(all_labels, all_preds, average='weighted', zero_division=0)),
        'recall_weighted': float(recall_score(all_labels, all_preds, average='weighted', zero_division=0)),
        'f1_weighted': float(f1_score(all_labels, all_preds, average='weighted', zero_division=0)),
        'per_class': {},
        'confusion_matrix': confusion_matrix(all_labels, all_preds, normalize='true').tolist()
    }

    for cls in class_names:
        if cls in report:
            metrics['per_class'][cls] = {
                'precision': float(report[cls]['precision']),
                'recall': float(report[cls]['recall']),
                'f1': float(report[cls]['f1-score']),
                'support': int(report[cls]['support'])
            }

    return metrics


def plot_training_history(history_path=None, output_dir=None):
    """Training/Validation loss ve accuracy grafiklerini çizer."""
    if history_path is None:
        history_path = os.path.join(HISTORY_DIR, 'history.json')
    if output_dir is None:
        output_dir = PLOTS_DIR

    os.makedirs(output_dir, exist_ok=True)

    with open(history_path, 'r', encoding='utf-8') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    # Grafik stili
    plt.style.use('dark_background')
    fig_bg = '#10131a'
    card_bg = '#1d2027'
    text_color = '#e1e2ec'
    primary = '#adc6ff'
    tertiary = '#df7412'
    grid_color = '#424754'

    # --- LOSS GRAFİĞİ ---
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(fig_bg)
    ax.set_facecolor(card_bg)

    ax.plot(epochs, history['train_loss'], color=primary, linewidth=2, label='Eğitim Kaybı')
    ax.plot(epochs, history['val_loss'], color=tertiary, linewidth=2, linestyle='--', label='Doğrulama Kaybı')
    ax.set_xlabel('Epoch', color=text_color, fontsize=12)
    ax.set_ylabel('Loss', color=text_color, fontsize=12)
    ax.set_title('Eğitim ve Doğrulama Kaybı (Loss)', color=text_color, fontsize=14, fontweight='bold')
    ax.legend(facecolor=card_bg, edgecolor=grid_color, fontsize=10)
    ax.grid(True, alpha=0.3, color=grid_color)
    ax.tick_params(colors=text_color)
    for spine in ax.spines.values():
        spine.set_color(grid_color)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_graph.png'), dpi=150, facecolor=fig_bg)
    plt.close()

    # --- ACCURACY GRAFİĞİ ---
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(fig_bg)
    ax.set_facecolor(card_bg)

    train_acc_pct = [a * 100 for a in history['train_acc']]
    val_acc_pct = [a * 100 for a in history['val_acc']]

    ax.plot(epochs, train_acc_pct, color=primary, linewidth=2, label='Eğitim Doğruluğu')
    ax.plot(epochs, val_acc_pct, color=tertiary, linewidth=2, linestyle='--', label='Doğrulama Doğruluğu')
    ax.set_xlabel('Epoch', color=text_color, fontsize=12)
    ax.set_ylabel('Accuracy (%)', color=text_color, fontsize=12)
    ax.set_title('Eğitim ve Doğrulama Doğruluğu (Accuracy)', color=text_color, fontsize=14, fontweight='bold')
    ax.legend(facecolor=card_bg, edgecolor=grid_color, fontsize=10)
    ax.grid(True, alpha=0.3, color=grid_color)
    ax.tick_params(colors=text_color)
    for spine in ax.spines.values():
        spine.set_color(grid_color)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_graph.png'), dpi=150, facecolor=fig_bg)
    plt.close()

    print(f"  [OK] Grafikler kaydedildi: {output_dir}")


def plot_confusion_matrix(cm_data, class_names, output_dir=None):
    """8x8 normalize edilmiş confusion matrix heatmap çizer."""
    if output_dir is None:
        output_dir = PLOTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    cm = np.array(cm_data)

    plt.style.use('dark_background')
    fig_bg = '#10131a'
    card_bg = '#1d2027'
    text_color = '#e1e2ec'

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(fig_bg)
    ax.set_facecolor(card_bg)

    sns.heatmap(
        cm, annot=True, fmt='.2f',
        xticklabels=class_names, yticklabels=class_names,
        cmap='Blues', vmin=0, vmax=1,
        linewidths=0.5, linecolor='#424754',
        cbar_kws={'label': 'Oran'},
        ax=ax
    )

    ax.set_xlabel('Tahmin Edilen Sınıf', color=text_color, fontsize=12)
    ax.set_ylabel('Gerçek Sınıf', color=text_color, fontsize=12)
    ax.set_title('Normalize Edilmiş Karışıklık Matrisi', color=text_color, fontsize=14, fontweight='bold')
    ax.tick_params(colors=text_color)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150, facecolor=fig_bg)
    plt.close()

    print(f"  [OK] Confusion matrix kaydedildi")


def run_full_evaluation(val_dir=None):
    """Tam değerlendirme: metrikler + grafikler."""
    if val_dir is None:
        val_dir = os.path.join(BASE_DIR, '..', 'dataset', 'val')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(SAVE_DIR, 'best_model_efficientnet_v2s.pth')

    if not os.path.exists(val_dir):
        print(f"  [UYARI] Doğrulama veri seti bulunamadı: {val_dir}")
        print("  Sadece eğitim geçmişi (history) grafikleri çizilecek.")
        plot_training_history()
        return

    # Model yükle
    model = load_trained_model(model_path, device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    class_names = checkpoint.get('class_names', CLASS_NAMES)

    # Validation loader
    val_dataset = datasets.ImageFolder(root=val_dir, transform=get_val_transforms())
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Metrikler
    metrics = evaluate_model(model, val_loader, device, class_names)

    print(f"\n  Genel Metrikler:")
    print(f"    Accuracy:           {metrics['accuracy']*100:.2f}%")
    print(f"    F1-Score (macro):   {metrics['f1_macro']*100:.2f}%")
    print(f"    Precision (macro):  {metrics['precision_macro']*100:.2f}%")
    print(f"    Recall (macro):     {metrics['recall_macro']*100:.2f}%")

    print(f"\n  Sınıf Bazlı:")
    for cls, vals in metrics['per_class'].items():
        print(f"    {cls:15s} | P: {vals['precision']:.3f} | R: {vals['recall']:.3f} | F1: {vals['f1']:.3f}")

    # Metrikleri kaydet
    os.makedirs(HISTORY_DIR, exist_ok=True)
    metrics_path = os.path.join(HISTORY_DIR, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Grafikler
    plot_training_history()
    plot_confusion_matrix(metrics['confusion_matrix'], class_names)

    print(f"\n  [OK] Değerlendirme tamamlandı!")
    return metrics


if __name__ == '__main__':
    run_full_evaluation()
