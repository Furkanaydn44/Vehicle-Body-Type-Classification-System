/**
 * Batch Test (Toplu Test) Logic
 */

const API_BASE = ''; // Same-origin

const trueTxtInput = document.getElementById('true-txt-input');
const imagesInput = document.getElementById('images-input');
const startTestBtn = document.getElementById('start-test-btn');
const loadingOverlay = document.getElementById('loading-overlay');
const resultsSection = document.getElementById('results-section');
const errorMessage = document.getElementById('error-message');
const totalEvaluated = document.getElementById('total-evaluated');

const cmHead = document.getElementById('cm-head');
const cmBody = document.getElementById('cm-body');
const crBody = document.getElementById('cr-body');

startTestBtn.addEventListener('click', async () => {
    // Validation
    const trueFile = trueTxtInput.files[0];
    const imageFiles = imagesInput.files;

    if (!trueFile) {
        showError('Lütfen True.txt dosyasını yükleyin.');
        return;
    }
    if (imageFiles.length === 0) {
        showError('Lütfen test edilecek görsellerin bulunduğu klasörü seçin.');
        return;
    }

    // Hide previous errors/results
    errorMessage.classList.add('hidden');
    resultsSection.classList.add('hidden');
    loadingOverlay.classList.remove('hidden');

    const formData = new FormData();
    formData.append('true_txt', trueFile);
    
    // Klasör içindeki resimleri tek tek ekle
    for (let i = 0; i < imageFiles.length; i++) {
        // İsteğe bağlı olarak sadece resimleri filtreleyebiliriz:
        const file = imageFiles[i];
        if (file.type.startsWith('image/')) {
            formData.append('images', file);
        }
    }

    try {
        const res = await fetch(`${API_BASE}/api/evaluate-batch`, {
            method: 'POST',
            body: formData
        });

        const data = await res.json();

        if (!res.ok) {
            throw new Error(data.error || 'Test sırasında sunucu hatası.');
        }

        if (data.error) {
            throw new Error(data.error);
        }

        displayResults(data);

    } catch (error) {
        showError(error.message);
    } finally {
        loadingOverlay.classList.add('hidden');
    }
});

function showError(msg) {
    errorMessage.textContent = msg;
    errorMessage.classList.remove('hidden');
}

function displayResults(data) {
    const { confusion_matrix, classification_report, class_labels, total_evaluated } = data;

    totalEvaluated.textContent = `${total_evaluated} Görsel Değerlendirildi`;

    // 1. Confusion Matrix
    cmHead.innerHTML = '<th class="p-3 bg-surface-container-high border-r border-outline-variant">Gerçek \\ Tahmin</th>';
    class_labels.forEach(lbl => {
        cmHead.innerHTML += `<th class="p-3 text-center">${lbl}</th>`;
    });

    cmBody.innerHTML = '';
    confusion_matrix.forEach((row, rowIndex) => {
        const trueLabel = class_labels[rowIndex];
        let rowHtml = `<tr><td class="p-3 font-bold bg-surface-container border-r border-outline-variant">${trueLabel}</td>`;
        
        row.forEach((val, colIndex) => {
            const isMatch = (rowIndex === colIndex);
            // Sadece doğru bilinenleri ve yanlışları renklendir
            const bgClass = val > 0 ? (isMatch ? 'bg-primary/20 text-primary' : 'bg-error/20 text-error') : '';
            rowHtml += `<td class="p-3 text-center ${bgClass}">${val}</td>`;
        });
        rowHtml += `</tr>`;
        cmBody.innerHTML += rowHtml;
    });

    // 2. Classification Report
    crBody.innerHTML = '';
    const keys = Object.keys(classification_report);
    
    keys.forEach(key => {
        if (key === 'accuracy') {
            // Accuracy tek değerli, farklı bir satır olarak ekle
            crBody.innerHTML += `
                <tr class="bg-surface-container-highest font-bold">
                    <td class="p-3">accuracy</td>
                    <td class="p-3" colspan="3"></td>
                    <td class="p-3 text-primary">${(classification_report[key] * 100).toFixed(2)}%</td>
                </tr>
            `;
            return;
        }

        const metrics = classification_report[key];
        const isAverage = key.includes('avg');
        const rowClass = isAverage ? 'bg-surface-container-high font-bold' : '';

        crBody.innerHTML += `
            <tr class="${rowClass}">
                <td class="p-3">${key}</td>
                <td class="p-3">${(metrics.precision * 100).toFixed(2)}%</td>
                <td class="p-3">${(metrics.recall * 100).toFixed(2)}%</td>
                <td class="p-3">${metrics['f1-score'] ? (metrics['f1-score'] * 100).toFixed(2) : '0.00'}%</td>
                <td class="p-3">${metrics.support || 0}</td>
            </tr>
        `;
    });

    resultsSection.classList.remove('hidden');
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
