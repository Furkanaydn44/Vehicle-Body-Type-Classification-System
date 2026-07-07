/**
 * VisionAI Core — Live Prediction Logic
 * Görsel yükleme, tahmin ve sonuç gösterimi
 */

const API_BASE = '';  // Same-origin (Flask serves both)

// DOM Elemanları
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const selectBtn = document.getElementById('select-btn');
const uploadSection = document.getElementById('upload-section');
const resultsSection = document.getElementById('results-section');
const loadingOverlay = document.getElementById('loading-overlay');
const previewImage = document.getElementById('preview-image');
const predictionLabel = document.getElementById('prediction-label');
const confidenceLabel = document.getElementById('confidence-label');
const inferenceTime = document.getElementById('inference-time');
const imageSize = document.getElementById('image-size');
const modelVersion = document.getElementById('model-version');
const resetBtn = document.getElementById('reset-btn');
const modelStatus = document.getElementById('model-status');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const predictBtnContainer = document.getElementById('predict-btn-container');
const predictBtn = document.getElementById('predict-btn');
const selectedFilename = document.getElementById('selected-filename');
const probPlaceholder = document.getElementById('prob-placeholder');
const probChartWrapper = document.getElementById('prob-chart-wrapper');

// Seçili dosya ve chart instance'ı
let selectedFile = null;
let probChart = null;

// Sınıf görünen adları (Türkçe)
const CLASS_DISPLAY = {
    'SUV': 'SUV',
    'VAN': 'Van',
    'STATION_WAGON': 'Station Wagon',
    'MICRO': 'Micro',
    'OPEN_WHEEL': 'Açık Tekerlekli',
    'SEDAN': 'Sedan',
    'HATCHBACK': 'Hatchback',
    'PICKUP': 'Pick Up'
};

// Grafik renkleri (tema uyumlu)
const CHART_COLORS = {
    primary: '#adc6ff',
    primaryDim: 'rgba(173, 198, 255, 0.25)',
    inactive: 'rgba(66, 71, 84, 0.6)',
    inactiveBorder: '#424754',
    onSurface: '#e1e2ec',
    onSurfaceVariant: '#c2c6d6',
    surface: '#1d2027',
    outline: '#42475430'
};

// ========================
// Model Durumu Kontrolü
// ========================
async function checkModelStatus() {
    try {
        const res = await fetch(`${API_BASE}/api/health`);
        const data = await res.json();
        if (data.model_loaded) {
            statusDot.classList.add('animate-pulse');
            statusDot.classList.remove('bg-error');
            statusDot.classList.add('bg-primary');
            statusText.textContent = 'MODEL STATUS: ONLINE';
            modelStatus.classList.remove('status-offline');
        } else {
            statusDot.classList.remove('animate-pulse');
            statusDot.classList.remove('bg-primary');
            statusDot.classList.add('bg-error');
            statusText.textContent = 'MODEL STATUS: OFFLINE';
            modelStatus.classList.add('status-offline');
        }
    } catch (e) {
        statusDot.classList.remove('animate-pulse');
        statusText.textContent = 'MODEL STATUS: BAĞLANTI YOK';
        modelStatus.classList.add('status-offline');
    }
}

// ========================
// Drag & Drop
// ========================
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

// Tıklama ile dosya seçme
dropZone.addEventListener('click', () => fileInput.click());
selectBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

// ========================
// Dosya İşleme (Otomatik TAHMİN YAPMA — buton gerekli)
// ========================
function handleFile(file) {
    const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
    if (!validTypes.includes(file.type)) {
        showError('Geçersiz dosya formatı. JPG, PNG, WebP veya BMP kullanın.');
        return;
    }

    if (file.size > 5 * 1024 * 1024) {
        showError('Dosya boyutu 5MB\'dan büyük olamaz.');
        return;
    }

    selectedFile = file;

    // Önizleme göster (sonuçlar bölümü için)
    const reader = new FileReader();
    reader.onload = (e) => {
        // Sonuçlar bölümündeki preview'ı doldur
        previewImage.src = e.target.result;

        // Drop zone'u küçük thumbnail gösterecek şekilde güncelle
        const inner = uploadSection.querySelector('.drop-zone-inner');
        if (inner) {
            inner.innerHTML = `
                <img src="${e.target.result}" alt="Önizleme" class="h-28 rounded-lg object-cover shadow-lg mb-2 border border-outline-variant" />
                <p class="font-body-base text-body-base text-on-surface font-medium">${file.name}</p>
                <p class="font-body-sm text-body-sm text-on-surface-variant mt-1">${(file.size / 1024).toFixed(1)} KB · Tahmin yapmak için butona tıklayın</p>
            `;
        }
    };
    reader.readAsDataURL(file);

    // Dosya adını göster
    selectedFilename.textContent = file.name;

    // Tahmin Yap butonunu ve sıfırla butonunu göster
    predictBtnContainer.classList.remove('hidden');
    resetBtn.classList.remove('hidden');

    // Önceki sonuçları gizle
    resultsSection.classList.add('hidden');

    // Grafik ve placeholder sıfırla
    if (probChart) {
        probChart.destroy();
        probChart = null;
    }
    probPlaceholder.classList.remove('hidden');
    probChartWrapper.classList.add('hidden');
}

// ========================
// "Tahmin Yap" Butonu
// ========================
predictBtn.addEventListener('click', () => {
    if (selectedFile) {
        predictImage(selectedFile);
    }
});

// ========================
// Tahmin API Çağrısı
// ========================
async function predictImage(file) {
    // Loading göster
    resultsSection.classList.remove('hidden');
    loadingOverlay.classList.remove('hidden');

    // Butonu devre dışı bırak (çift tıklamayı önle)
    predictBtn.disabled = true;
    predictBtn.classList.add('opacity-50', 'cursor-not-allowed');

    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch(`${API_BASE}/api/predict`, {
            method: 'POST',
            body: formData
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || 'Sunucu hatası');
        }

        const data = await res.json();
        displayResults(data);

    } catch (error) {
        showError(error.message || 'Tahmin sırasında bir hata oluştu.');
    } finally {
        loadingOverlay.classList.add('hidden');
        predictBtn.disabled = false;
        predictBtn.classList.remove('opacity-50', 'cursor-not-allowed');
    }
}

// ========================
// Sonuçları Göster
// ========================
function displayResults(data) {
    resultsSection.classList.add('animate-slide-up');

    // Ana tahmin
    const displayName = CLASS_DISPLAY[data.prediction] || data.prediction;
    predictionLabel.textContent = displayName.toUpperCase();
    confidenceLabel.textContent = `%${data.confidence} Güven`;

    // Meta bilgiler
    inferenceTime.textContent = `${data.inference_time_ms}ms`;
    imageSize.textContent = data.image_size;
    modelVersion.textContent = data.model_version;

    // ========================
    // Chart.js Yatay Bar Chart (Olasılık Dağılımı)
    // ========================
    const probs = data.probabilities;
    const labels = Object.keys(probs).map(k => CLASS_DISPLAY[k] || k);
    const values = Object.values(probs);
    const bgColors = Object.keys(probs).map(k =>
        k === data.prediction ? CHART_COLORS.primary : CHART_COLORS.inactive
    );
    const borderColors = Object.keys(probs).map(k =>
        k === data.prediction ? CHART_COLORS.primary : CHART_COLORS.inactiveBorder
    );

    // Önceki grafiği temizle
    if (probChart) {
        probChart.destroy();
        probChart = null;
    }

    // Placeholder gizle, chart göster
    probPlaceholder.classList.add('hidden');
    probChartWrapper.classList.remove('hidden');

    const canvas = document.getElementById('prob-chart');
    probChart = new Chart(canvas, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Olasılık (%)',
                data: values,
                backgroundColor: bgColors,
                borderColor: borderColors,
                borderWidth: 1,
                borderRadius: 4,
                borderSkipped: false
            }]
        },
        options: {
            indexAxis: 'y',  // Yatay bar chart
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 700, easing: 'easeOutQuart' },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#1d2027',
                    titleColor: CHART_COLORS.onSurface,
                    bodyColor: CHART_COLORS.onSurfaceVariant,
                    borderColor: '#424754',
                    borderWidth: 1,
                    padding: 10,
                    callbacks: {
                        label: (ctx) => ` ${ctx.raw.toFixed(2)}%`
                    }
                }
            },
            scales: {
                x: {
                    min: 0,
                    max: 100,
                    ticks: {
                        color: CHART_COLORS.onSurfaceVariant,
                        font: { size: 10, family: 'JetBrains Mono' },
                        callback: (v) => `${v}%`
                    },
                    grid: { color: CHART_COLORS.outline }
                },
                y: {
                    ticks: {
                        color: CHART_COLORS.onSurface,
                        font: { size: 10, family: 'JetBrains Mono' }
                    },
                    grid: { display: false }
                }
            }
        }
    });
}

// ========================
// Hata Göster
// ========================
function showError(message) {
    loadingOverlay.classList.add('hidden');
    resultsSection.classList.remove('hidden');

    const resultContent = resultsSection.querySelector('.result-content');
    if (resultContent) {
        resultContent.innerHTML = `
            <div class="flex flex-col items-center justify-center p-xl text-center col-span-2">
                <span class="material-symbols-outlined text-[48px] text-error mb-sm">error</span>
                <p class="font-h2 text-h2 text-error mb-2">Hata</p>
                <p class="font-body-base text-body-base text-on-surface-variant">${message}</p>
            </div>
        `;
    }
}

// ========================
// Sıfırla
// ========================
if (resetBtn) {
    resetBtn.addEventListener('click', () => {
        selectedFile = null;
        resultsSection.classList.add('hidden');
        resultsSection.classList.remove('animate-slide-up');
        resetBtn.classList.add('hidden');
        predictBtnContainer.classList.add('hidden');
        fileInput.value = '';
        previewImage.src = '';
        selectedFilename.textContent = '';

        // Grafiği sıfırla
        if (probChart) {
            probChart.destroy();
            probChart = null;
        }
        probPlaceholder.classList.remove('hidden');
        probChartWrapper.classList.add('hidden');

        // Drop zone'u geri yükle
        restoreDropZone();
    });
}

function restoreDropZone() {
    const inner = uploadSection.querySelector('.drop-zone-inner');
    if (inner) {
        inner.innerHTML = `
            <span class="material-symbols-outlined text-[48px] text-primary mb-sm drop-icon transition-transform">cloud_upload</span>
            <p class="font-body-base text-body-base text-on-surface font-medium">Sınıflandırmak için bir araç görseli sürükleyip bırakın veya tıklayın</p>
            <p class="font-body-sm text-body-sm text-on-surface-variant mt-2">Desteklenen formatlar: JPG, PNG, WebP (Max 5MB)</p>
            <button id="select-btn-inner" class="mt-md px-4 py-2 bg-inverse-primary text-on-primary rounded-DEFAULT font-body-base text-body-base font-medium flex items-center gap-2 btn-primary" onclick="event.stopPropagation(); document.getElementById('file-input').click();">
                <span class="material-symbols-outlined text-[16px]">upload_file</span>
                Dosya Seç
            </button>
        `;
    }
}

// Başlangıç
checkModelStatus();
setInterval(checkModelStatus, 30000);
