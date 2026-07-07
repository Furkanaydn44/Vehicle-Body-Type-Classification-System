# 🚗 VisionAI Core — Araba Gövde Tipi Sınıflandırma Sistemi

> **Türkçe** | [English](#-visionai-core--vehicle-body-type-classification-system)

---

## 📋 Proje Hakkında

**VisionAI Core**, EfficientNetV2-S derin öğrenme modelini kullanan, araç görsellerinden gövde tipini otomatik olarak sınıflandıran bir yapay zeka sistemidir. Flask tabanlı bir REST API, modern bir web arayüzü ve Google Colab eğitim desteğinden oluşmaktadır.

### 🎯 Desteklenen Sınıflar (8 Kategori)

| ID | Sınıf | Türkçe Adı |
|----|-------|------------|
| 1 | SUV | SUV |
| 2 | VAN | Van |
| 3 | STATION_WAGON | Station Wagon |
| 4 | MICRO | Micro |
| 5 | OPEN_WHEEL | Açık Tekerlekli |
| 6 | SEDAN | Sedan |
| 7 | HATCHBACK | Hatchback |
| 8 | PICKUP | Pick-Up |

---

## 🏗️ Proje Yapısı

```
Yazlab efficientnet v2/
├── backend/
│   ├── app.py               # Flask REST API sunucusu
│   ├── model.py             # EfficientNetV2-S model mimarisi
│   ├── train.py             # Lokal eğitim scripti
│   ├── preprocessing.py     # Veri ön işleme ve augmentation
│   ├── evaluate.py          # Model değerlendirme ve metrikler
│   ├── predict.py           # Toplu tahmin (CLI)
│   ├── batch_eval.py        # Hafızadan batch değerlendirme
│   ├── prepare_dataset.py   # Kaggle veri seti düzenleme
│   ├── requirements.txt     # Python bağımlılıkları
│   ├── saved_model/         # Eğitilmiş model ağırlıkları (.pth)
│   └── training_history/    # Eğitim geçmişi ve metrikler (JSON/PNG)
├── frontend/
│   ├── index.html           # Canlı tahmin arayüzü
│   ├── performance.html     # Model performans paneli
│   ├── batch_test.html      # Toplu test arayüzü
│   ├── css/                 # Stil dosyaları
│   ├── js/                  # JavaScript dosyaları
│   └── src/                 # Diğer kaynaklar
├── colab_train_v2.py        # Google Colab eğitim scripti (gelişmiş)
├── Test/                    # Test görselleri (klasör bazlı)
├── Test.txt                 # Test değerlendirme kodu (Python)
├── true.txt                 # Gerçek etiketler (değerlendirme için)
└── PredictionScript.txt     # Tahmin scripti referansı
```

---

## ⚙️ Kurulum

### Gereksinimler

- Python 3.8+
- pip

### 1. Bağımlılıkları Yükle

```bash
cd backend
pip install -r requirements.txt
```

**requirements.txt içeriği:**
```
torch>=2.0.0
torchvision>=0.15.0
flask>=3.0.0
flask-cors>=4.0.0
Pillow>=10.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.13.0
kaggle>=1.6.0
```

---

## 🚀 Kullanım

### Web Arayüzü (Önerilen)

```bash
cd backend
python app.py
```

Tarayıcıdan açın:
- `http://localhost:5000` → Canlı Tahmin
- `http://localhost:5000/performance` → Model Performansı
- `http://localhost:5000/batch-test` → Toplu Test

### Toplu Tahmin (CLI)

```bash
cd backend
python predict.py --input /test_gorsellerim --output preds.txt
```

**Çıktı formatı:**
```
gorsel_adi.jpg | prediction: 6
gorsel_adi2.png | prediction: 1
```

### Model Değerlendirme

```bash
cd backend
python evaluate.py
```

---

## 🧠 Model Mimarisi

- **Temel Model:** EfficientNetV2-S (ImageNet ön eğitimli)
- **Giriş Boyutu:** 224×224 px (eğitim), 384×384 px (inference)
- **Sınıflandırıcı:**
  - Dropout (p=0.3)
  - Linear(1280 → 256) + ReLU + BatchNorm1d
  - Dropout (p=0.2)
  - Linear(256 → 8)

### İki Aşamalı Eğitim

| Aşama | Açıklama | Optimizer |
|-------|----------|-----------|
| **Phase 1** | Sadece sınıflandırıcı eğitimi (backbone donduruldu) | Adam, lr=1e-4 |
| **Phase 2** | Son N blok açılır (fine-tuning) | Adam, iki öğrenme hızı |

---

## 📊 Augmentation Stratejisi

Eğitim sırasında uygulanan veri çoğaltma teknikleri:
- `RandomResizedCrop` (scale: 0.7–1.0)
- `RandomHorizontalFlip`
- `RandomRotation` (±15°)
- `ColorJitter` (brightness, contrast, saturation, hue)
- `RandomAffine` (translate, scale)
- `RandomPerspective`
- `RandomErasing`

---

## 🌐 API Referansı

| Endpoint | Yöntem | Açıklama |
|----------|--------|----------|
| `/api/predict` | POST | Görsel yükle, tahmin al |
| `/api/performance` | GET | Eğitim metrikleri ve geçmiş |
| `/api/model-info` | GET | Model bilgileri ve durumu |
| `/api/health` | GET | Sunucu sağlık kontrolü |
| `/api/evaluate-batch` | POST | Toplu görsel değerlendirme |

### `/api/predict` — Örnek Yanıt

```json
{
  "prediction": "SEDAN",
  "display_name": "Sedan",
  "confidence": 94.37,
  "probabilities": {
    "SEDAN": 94.37,
    "HATCHBACK": 3.21,
    "MICRO": 1.02,
    ...
  },
  "inference_time_ms": 42.5,
  "image_size": "1920x1080",
  "model_version": "CV-BodyType-v2.4"
}
```

---

## 🔬 Google Colab ile Eğitim

Gelişmiş eğitim için `colab_train_v2.py` kullanılır. Bu script şunları içerir:
- **FocalLoss** (zor örneklere odaklanma, γ=2)
- **MixUp Augmentation** (sınıf sınırlarını güçlendirir)
- **WeightedRandomSampler** (dengesiz veri seti desteği)
- **CosineAnnealingLR** (öğrenme hızı çizelgesi)
- **F1-score tabanlı Early Stopping**

```bash
# Google Colab'da:
# 1. colab_train_v2.py dosyasını Colab'a yükleyin
# 2. Runtime → GPU seçin
# 3. Script içindeki DRIVE_DATASET ve DRIVE_SAVE yollarını güncelleyin
# 4. Tüm hücreyi çalıştırın
```

---

## 📁 Veri Seti Hazırlama

```bash
cd backend
python prepare_dataset.py
```

**Beklenen ham veri yapısı (`raw_data/`):**
```
raw_data/
├── SUV/
├── Sedan/
├── Hatchback/
├── Pick-Up/
├── VAN/
├── Convertible/
├── Coupe/
├── Wagon/
└── ...
```

Sınıf eşleştirmesi `prepare_dataset.py` içindeki `CLASS_MAPPING` sözlüğü ile yapılır.

---

## 📈 Değerlendirme Metrikleri

Sistem şu metrikleri destekler:
- **Accuracy** (genel doğruluk)
- **Precision / Recall / F1-Score** (macro & weighted)
- **Confusion Matrix** (normalize edilmiş)
- **Sınıf bazlı detay raporu**

---

## 📄 Lisans

Bu proje akademik amaçlı geliştirilmiştir.

---
---

# 🚗 VisionAI Core — Vehicle Body Type Classification System

> [Türkçe](#-visionai-core--araba-gövde-tipi-sınıflandırma-sistemi) | **English**

---

## 📋 About the Project

**VisionAI Core** is an AI-powered vehicle body type classification system built on the **EfficientNetV2-S** deep learning model. It consists of a Flask-based REST API, a modern web interface, and Google Colab training support.

### 🎯 Supported Classes (8 Categories)

| ID | Class | Description |
|----|-------|-------------|
| 1 | SUV | Sport Utility Vehicle |
| 2 | VAN | Van |
| 3 | STATION_WAGON | Station Wagon |
| 4 | MICRO | Micro Car |
| 5 | OPEN_WHEEL | Open Wheel (F1-style) |
| 6 | SEDAN | Sedan |
| 7 | HATCHBACK | Hatchback |
| 8 | PICKUP | Pick-Up Truck |

---

## 🏗️ Project Structure

```
Yazlab efficientnet v2/
├── backend/
│   ├── app.py               # Flask REST API server
│   ├── model.py             # EfficientNetV2-S architecture
│   ├── train.py             # Local training script
│   ├── preprocessing.py     # Data preprocessing & augmentation
│   ├── evaluate.py          # Model evaluation & metrics
│   ├── predict.py           # Batch prediction (CLI)
│   ├── batch_eval.py        # In-memory batch evaluation
│   ├── prepare_dataset.py   # Kaggle dataset organizer
│   ├── requirements.txt     # Python dependencies
│   ├── saved_model/         # Trained model weights (.pth)
│   └── training_history/    # Training history & metrics (JSON/PNG)
├── frontend/
│   ├── index.html           # Live prediction interface
│   ├── performance.html     # Model performance dashboard
│   ├── batch_test.html      # Batch testing interface
│   ├── css/                 # Stylesheets
│   ├── js/                  # JavaScript files
│   └── src/                 # Other assets
├── colab_train_v2.py        # Google Colab training script (advanced)
├── Test/                    # Test images (folder-based)
├── Test.txt                 # Test evaluation code (Python)
├── true.txt                 # Ground truth labels (for evaluation)
└── PredictionScript.txt     # Prediction script reference
```

---

## ⚙️ Installation

### Requirements

- Python 3.8+
- pip

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

**requirements.txt:**
```
torch>=2.0.0
torchvision>=0.15.0
flask>=3.0.0
flask-cors>=4.0.0
Pillow>=10.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.13.0
kaggle>=1.6.0
```

---

## 🚀 Usage

### Web Interface (Recommended)

```bash
cd backend
python app.py
```

Open in your browser:
- `http://localhost:5000` → Live Prediction
- `http://localhost:5000/performance` → Model Performance
- `http://localhost:5000/batch-test` → Batch Testing

### Batch Prediction (CLI)

```bash
cd backend
python predict.py --input /path/to/test_images --output preds.txt
```

**Output format:**
```
image_name.jpg | prediction: 6
image_name2.png | prediction: 1
```

### Model Evaluation

```bash
cd backend
python evaluate.py
```

---

## 🧠 Model Architecture

- **Base Model:** EfficientNetV2-S (ImageNet pre-trained)
- **Input Size:** 224×224 px (training), 384×384 px (inference)
- **Classifier Head:**
  - Dropout (p=0.3)
  - Linear(1280 → 256) + ReLU + BatchNorm1d
  - Dropout (p=0.2)
  - Linear(256 → 8)

### Two-Phase Training Strategy

| Phase | Description | Optimizer |
|-------|-------------|-----------|
| **Phase 1** | Classifier-only training (backbone frozen) | Adam, lr=1e-4 |
| **Phase 2** | Last N blocks unfrozen (fine-tuning) | Adam, dual learning rates |

---

## 📊 Data Augmentation Strategy

Augmentations applied during training:
- `RandomResizedCrop` (scale: 0.7–1.0)
- `RandomHorizontalFlip`
- `RandomRotation` (±15°)
- `ColorJitter` (brightness, contrast, saturation, hue)
- `RandomAffine` (translate, scale)
- `RandomPerspective`
- `RandomErasing`

---

## 🌐 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict` | POST | Upload image, get prediction |
| `/api/performance` | GET | Training metrics & history |
| `/api/model-info` | GET | Model info & status |
| `/api/health` | GET | Server health check |
| `/api/evaluate-batch` | POST | Batch image evaluation |

### `/api/predict` — Example Response

```json
{
  "prediction": "SEDAN",
  "display_name": "Sedan",
  "confidence": 94.37,
  "probabilities": {
    "SEDAN": 94.37,
    "HATCHBACK": 3.21,
    "MICRO": 1.02,
    ...
  },
  "inference_time_ms": 42.5,
  "image_size": "1920x1080",
  "model_version": "CV-BodyType-v2.4"
}
```

---

## 🔬 Training with Google Colab

The advanced training script `colab_train_v2.py` includes:
- **FocalLoss** (focus on hard examples, γ=2)
- **MixUp Augmentation** (strengthens class boundaries)
- **WeightedRandomSampler** (handles imbalanced datasets)
- **CosineAnnealingLR** (learning rate schedule)
- **F1-score based Early Stopping**

```bash
# In Google Colab:
# 1. Upload colab_train_v2.py to Colab
# 2. Go to Runtime → Change runtime type → GPU
# 3. Update DRIVE_DATASET and DRIVE_SAVE paths inside the script
# 4. Run the cell
```

---

## 📁 Dataset Preparation

```bash
cd backend
python prepare_dataset.py
```

**Expected raw data structure (`raw_data/`):**
```
raw_data/
├── SUV/
├── Sedan/
├── Hatchback/
├── Pick-Up/
├── VAN/
├── Convertible/
├── Coupe/
├── Wagon/
└── ...
```

Class mapping is handled via the `CLASS_MAPPING` dictionary in `prepare_dataset.py`.

---

## 📈 Evaluation Metrics

The system supports the following metrics:
- **Accuracy** (overall)
- **Precision / Recall / F1-Score** (macro & weighted)
- **Confusion Matrix** (normalized)
- **Per-class detailed report**

---


## 📄 License

This project was developed for academic purposes.
