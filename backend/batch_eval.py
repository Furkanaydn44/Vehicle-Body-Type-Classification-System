from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import traceback

def evaluate_predictions_from_memory(y_true_list, y_pred_list, class_labels):
    """
    Hafızadaki (RAM) listeler üzerinden değerlendirme yapar.
    
    y_true_list: Gerçek etiketlerin ID listesi (Örn: [1, 2, 1, 8])
    y_pred_list: Modelin tahmin ettiği etiketlerin ID listesi
    class_labels: Tüm sınıf ID'lerinin listesi (Örn: [1, 2, 3, 4, 5, 6, 7, 8])
    """
    try:
        if not y_true_list or not y_pred_list:
            return {"error": "Değerlendirme için yeterli veri yok."}

        # Ham Confusion Matrix
        cm = confusion_matrix(y_true_list, y_pred_list, labels=class_labels)
        cm_list = cm.tolist()

        # Normalize Edilmiş Confusion Matrix (satır toplamına böl)
        cm_normalized = []
        for row in cm:
            row_sum = row.sum()
            if row_sum > 0:
                cm_normalized.append([round(v / row_sum, 4) for v in row])
            else:
                cm_normalized.append([0.0] * len(row))

        # Classification Report
        report = classification_report(
            y_true_list,
            y_pred_list,
            labels=class_labels,
            target_names=[str(c) for c in class_labels],
            zero_division=0,
            output_dict=True
        )

        # Genel accuracy
        accuracy = accuracy_score(y_true_list, y_pred_list)

        return {
            "confusion_matrix": cm_list,
            "confusion_matrix_normalized": cm_normalized,
            "classification_report": report,
            "class_labels": class_labels,
            "accuracy": round(accuracy * 100, 2)
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": f"Değerlendirme hatası: {str(e)}"}

