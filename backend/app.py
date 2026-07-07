"""
Flask API Sunucusu
Eğitilmiş modeli web arayüzüne entegre eder.

Endpoints:
    POST /api/predict     - Görsel yükle, tahmin al
    GET  /api/performance - Eğitim metrikleri ve geçmişi
    GET  /api/model-info  - Model durumu
    GET  /api/health      - Sağlık kontrolü
"""

import os
import sys
import json
import time
import io
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

from model import CLASS_NAMES, CLASS_DISPLAY_NAMES, load_trained_model
from preprocessing import get_inference_transforms
from batch_eval import evaluate_predictions_from_memory

# ========================
# Yapılandırma
# ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, '..', 'frontend')
SAVE_DIR = os.path.join(BASE_DIR, 'saved_model')
HISTORY_DIR = os.path.join(BASE_DIR, 'training_history')
MODEL_PATH = os.path.join(SAVE_DIR, 'best_model_efficientnet_v2s.pth')

app = Flask(__name__, static_folder=FRONTEND_DIR)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
CORS(app)

# ========================
# Model Yükleme
# ========================
device = torch.device('cpu')  # Web sunucusu için CPU yeterli
model = None
model_class_names = CLASS_NAMES
model_loaded = False


def load_model():
    """Eğitilmiş modeli yükler."""
    global model, model_class_names, model_loaded

    if not os.path.exists(MODEL_PATH):
        print(f"  [!] Model bulunamadi: {MODEL_PATH}")
        print(f"  Demo modu aktif (model egitildikten sonra yeniden baslatin)")
        return False

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model_class_names = checkpoint.get('class_names', CLASS_NAMES)

        model = load_trained_model(MODEL_PATH, device=device)

        model_loaded = True
        print(f"  [OK] Model yuklendi: {MODEL_PATH}")
        print(f"  Siniflar: {model_class_names}")
        return True
    except Exception as e:
        print(f"  [HATA] Model yukleme hatasi: {e}")
        return False


# Başlangıçta modeli yükle
load_model()

# Inference transforms
inference_transforms = get_inference_transforms()


# ========================
# Frontend Servis
# ========================
@app.route('/')
def serve_index():
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/performance')
def serve_performance():
    return send_from_directory(FRONTEND_DIR, 'performance.html')


@app.route('/batch-test')
def serve_batch_test():
    return send_from_directory(FRONTEND_DIR, 'batch_test.html')


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(FRONTEND_DIR, path)


# ========================
# API Endpoints
# ========================
@app.route('/api/predict', methods=['POST'])
def predict():
    """Görsel yükle ve tahmin al."""
    if not model_loaded:
        return jsonify({'error': 'Model yüklenmedi. Lütfen önce modeli eğitin.'}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'Dosya bulunamadı. "file" alanında bir görsel gönderin.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi.'}), 400

    try:
        start_time = time.time()

        # Görseli oku
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        original_size = img.size

        # Ön işleme
        input_tensor = inference_transforms(img).unsqueeze(0).to(device)

        # Tahmin
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, 0)

        inference_time = (time.time() - start_time) * 1000  # ms

        predicted_class = model_class_names[predicted_idx.item()]
        display_name = CLASS_DISPLAY_NAMES.get(predicted_class, predicted_class)

        # Tüm sınıflar için olasılıklar
        prob_dict = {}
        for i, cls in enumerate(model_class_names):
            prob_dict[cls] = round(float(probabilities[i]) * 100, 2)

        # Olasılıklara göre sırala (büyükten küçüğe)
        prob_dict = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))

        return jsonify({
            'prediction': predicted_class,
            'display_name': display_name,
            'confidence': round(float(confidence) * 100, 2),
            'probabilities': prob_dict,
            'inference_time_ms': round(inference_time, 1),
            'image_size': f"{original_size[0]}x{original_size[1]}",
            'model_version': 'CV-BodyType-v2.4'
        })

    except Exception as e:
        return jsonify({'error': f'Tahmin hatası: {str(e)}'}), 500


@app.route('/api/performance', methods=['GET'])
def performance():
    """Eğitim metrikleri ve geçmişini döndürür."""
    result = {}

    # Eğitim geçmişi
    history_path = os.path.join(HISTORY_DIR, 'history.json')
    if os.path.exists(history_path):
        with open(history_path, 'r', encoding='utf-8') as f:
            result['history'] = json.load(f)

    # Değerlendirme metrikleri
    metrics_path = os.path.join(HISTORY_DIR, 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r', encoding='utf-8') as f:
            result['metrics'] = json.load(f)

    if not result:
        return jsonify({'error': 'Henüz eğitim verisi yok.'}), 404

    return jsonify(result)


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Model durumu ve bilgileri."""
    info = {
        'model_loaded': model_loaded,
        'model_architecture': 'EfficientNetV2_S',
        'model_version': 'CV-BodyType-v2.4',
        'num_classes': len(model_class_names),
        'class_names': model_class_names,
        'device': str(device),
        'model_path': MODEL_PATH,
        'model_exists': os.path.exists(MODEL_PATH)
    }
    return jsonify(info)


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': model_loaded})

@app.route('/favicon.ico')
def favicon():
    return '', 204


@app.route('/api/evaluate-batch', methods=['POST'])
def evaluate_batch():
    """Çoklu görsel ve True.txt ile modeli değerlendir."""
    if not model_loaded:
        return jsonify({'error': 'Model yüklenmedi. Lütfen önce modeli eğitin.'}), 503

    if 'true_txt' not in request.files:
        return jsonify({'error': '"true_txt" dosyası eksik.'}), 400

    images = request.files.getlist('images')
    if not images:
        return jsonify({'error': 'Görseller eksik.'}), 400

    true_file = request.files['true_txt']
    true_content = true_file.read().decode('utf-8')

    # True.txt içeriğini ayrıştır
    true_dict = {}
    for line in true_content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parts = line.split(" | ")
            filename = parts[0].strip()
            # Örn: class: 1 -> "class", "1"
            val_part = parts[1].split(":")[1].strip()
            true_dict[filename] = int(val_part)
        except Exception:
            continue

    if not true_dict:
        return jsonify({'error': 'True.txt formatı geçersiz veya dosya boş.'}), 400

    # Model sınıflarını ID'lere eşleme
    # 1. SUV, 2. VAN, 3. STATION WAGON, 4. MICRO, 5. OPEN WHEEL, 6. SEDAN, 7. HATCHBACK, 8. PICKUP
    CLASS_TO_ID = {
        'SUV': 1,
        'VAN': 2,
        'STATION_WAGON': 3,
        'MICRO': 4,
        'OPEN_WHEEL': 5,
        'SEDAN': 6,
        'HATCHBACK': 7,
        'PICKUP': 8
    }

    y_true = []
    y_pred = []

    # Her görsel için tahmin yap
    for file in images:
        filename = file.filename
        
        # Eğer bu dosya True.txt içinde yoksa atla
        # Fakat tarayıcı klasör yüklediğinde path dahil olabilir (folder/img.jpg), bazen sadece isim.
        basename = os.path.basename(filename)
        if basename not in true_dict:
            # Belki True.txt içerisinde uzantı farklı veya path var, basename üzerinden deneyelim
            pass

        # Tam eşleşme arayalım
        lookup_name = filename if filename in true_dict else basename
        
        if lookup_name not in true_dict:
            continue

        try:
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            input_tensor = inference_transforms(img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)[0]
                confidence, predicted_idx = torch.max(probabilities, 0)
            
            predicted_class = model_class_names[predicted_idx.item()]
            predicted_id = CLASS_TO_ID.get(predicted_class, -1)
            
            if predicted_id != -1:
                y_true.append(true_dict[lookup_name])
                y_pred.append(predicted_id)
        except Exception as e:
            print(f"Görsel işleme hatası ({filename}): {e}")
            continue

    if not y_true:
        return jsonify({'error': 'Görseller ile True.txt arasında ortak dosya bulunamadı.'}), 400

    # Değerlendirme
    class_labels = [1, 2, 3, 4, 5, 6, 7, 8]
    result = evaluate_predictions_from_memory(y_true, y_pred, class_labels)
    
    # İstatistik ekle
    result['total_evaluated'] = len(y_true)
    
    return jsonify(result)

# ========================
# Ana Giriş
# ========================
if __name__ == '__main__':
    print(f"\n{'='*60}")
    print(f"  VisionAI Core - Araba Govde Tipi Siniflandirma API")
    print(f"{'='*60}")
    print(f"  Frontend: http://localhost:5000")
    print(f"  Model:    {'Yuklendi [OK]' if model_loaded else 'Yuklenmedi [!]'}")
    print(f"{'='*60}\n")

    app.run(host='0.0.0.0', port=5000, debug=True)
