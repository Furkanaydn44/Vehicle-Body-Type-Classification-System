"""
Toplu tahmin scripti (PredictionScript kurallarina uygun)
Kullanim: python predict.py --input /klasor/yolu --output preds.txt
"""
import os
import sys
import argparse
import torch
from PIL import Image
from model import CLASS_NAMES, load_trained_model
from preprocessing import get_inference_transforms

# Istenen ID formatina uygun eslestirme:
# 1. SUV
# 2. VAN
# 3. STATION WAGON
# 4. MICRO
# 5. AÇIK TEKERLEKLİ (OPEN_WHEEL)
# 6. SEDAN
# 7. HATCHBACK
# 8. PICK UP
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

def predict_folder(input_dir, output_file, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Cihaz: {device}")
    
    if not os.path.exists(model_path):
        print(f"[!] Model bulunamadi: {model_path}")
        sys.exit(1)
        
    print(f"[1/3] Model yukleniyor...")
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model_class_names = checkpoint.get('class_names', CLASS_NAMES)

        model = load_trained_model(model_path, device=device)

        print(f"  [OK] Model yuklendi (Siniflar: {model_class_names})")
    except Exception as e:
        print(f"  [HATA] Model yukleme basarisiz: {e}")
        sys.exit(1)
        
    print(f"\n[2/3] Gorseller hazirlaniyor...")
    if not os.path.exists(input_dir):
        print(f"[!] Girdi klasoru bulunamadi: {input_dir}")
        sys.exit(1)
        
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = []
    
    for root, _, files in os.walk(input_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in valid_extensions:
                image_files.append(os.path.join(root, f))
                
    print(f"  Bulunan gorsel sayisi: {len(image_files)}")
    if len(image_files) == 0:
        print("  Tahmin yapilacak gorsel bulunamadi.")
        sys.exit(0)
        
    transform = get_inference_transforms()
    
    print(f"\n[3/3] Tahminler yapiliyor ve kaydediliyor...")
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        with torch.no_grad():
            for idx, img_path in enumerate(image_files):
                try:
                    # Goruntu acma ve donusturme
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = transform(img).unsqueeze(0).to(device)
                    
                    # Tahmin
                    outputs = model(img_tensor)
                    _, predicted = torch.max(outputs, 1)
                    
                    # Tahmin edilen sinif adi
                    pred_idx = predicted.item()
                    pred_class_name = model_class_names[pred_idx]
                    
                    # Istenen ID formatina cevir
                    # Eger model klas adi ile eslesmeyen bir sey varsa (olmamasi lazim), id = 1 yap varsayilan
                    pred_id = CLASS_TO_ID.get(pred_class_name, 1)
                    
                    # Sadece dosya adini al (istenen formata gore)
                    filename = os.path.basename(img_path)
                    
                    # Dosyaya yaz: "filename | prediction: value"
                    line = f"{filename} | prediction: {pred_id}\n"
                    f_out.write(line)
                    
                    if (idx + 1) % 50 == 0:
                        print(f"  {idx + 1}/{len(image_files)} islendi...")
                        
                except Exception as e:
                    print(f"  [HATA] {img_path} islenemedi: {e}")
                    
    print(f"\n[TAMAMLANDI] Sonuclar '{output_file}' dosyasina kaydedildi.")
    print("Test kodunuzu kullanarak 'preds.txt' ve 'True.txt' dosyalarini karsilastirabilirsiniz.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Araba govde tipi toplu tahmin scripti")
    parser.add_argument("--input", type=str, default="test_images", help="Test gorsellerinin bulundugu klasor")
    parser.add_argument("--output", type=str, default="preds.txt", help="Cikti dosyasi adi (ornek: preds.txt)")
    parser.add_argument("--model", type=str, default="saved_model/best_model_efficientnet_v2s.pth", help="Egitilmis modelin yolu")
    
    args = parser.parse_args()
    
    # Eger input_dir yoksa bos bir klasor olusturalim ki hata vermesin
    if not os.path.exists(args.input) and args.input == "test_images":
        os.makedirs(args.input)
        print(f"(!) '{args.input}' klasoru olusturuldu. Lutfen test gorsellerini bu klasore koyup scripti tekrar calistirin.")
    else:
        predict_folder(args.input, args.output, args.model)
