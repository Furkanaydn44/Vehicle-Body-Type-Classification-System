/**
 * VisionAI Core — Model Performance Visualization
 * Chart.js ile Loss, Accuracy grafikleri ve Confusion Matrix
 */

const API_BASE = '';

// Sınıf görünen adları
const CLASS_DISPLAY = {
    'SUV': 'SUV', 'VAN': 'Van', 'STATION_WAGON': 'St. Wagon',
    'MICRO': 'Micro', 'OPEN_WHEEL': 'Açık Tek.', 'SEDAN': 'Sedan',
    'HATCHBACK': 'Hatchback', 'PICKUP': 'Pick Up'
};

// Tema renkleri
const COLORS = {
    primary: '#adc6ff',
    tertiary: '#df7412',
    surface: '#10131a',
    surfaceContainer: '#1d2027',
    onSurface: '#e1e2ec',
    outline: '#424754',
    onSurfaceVariant: '#c2c6d6',
    error: '#ffb4ab',
    secondary: '#b7c8e1'
};

// ========================
// Veri Yükleme
// ========================
async function loadPerformanceData() {
    try {
        const res = await fetch(`${API_BASE}/api/performance`);
        if (!res.ok) {
            showNoData();
            return;
        }
        const data = await res.json();

        if (data.history) {
            renderMetricCards(data);
            renderLossChart(data.history);
            renderAccuracyChart(data.history);
        }

        if (data.metrics && data.metrics.confusion_matrix) {
            const classNames = data.history?.class_names || data.metrics?.class_names || [];
            renderConfusionMatrix(data.metrics.confusion_matrix, classNames);
        }

        // Per-class metrikleri göster (macro ve weighted average dahil)
        if (data.metrics && data.metrics.per_class) {
            renderPerClassTable(data.metrics.per_class, data.metrics);
        }

    } catch (e) {
        console.error('Performans verisi yüklenemedi:', e);
        showNoData();
    }
}

// ========================
// Metrik Kartları
// ========================
function renderMetricCards(data) {
    const m = data.metrics || {};
    const h = data.history || {};

    setMetric('accuracy-value', m.accuracy ? (m.accuracy * 100).toFixed(1) : (h.best_val_acc ? (h.best_val_acc * 100).toFixed(1) : '--'));
    setMetric('precision-value', m.precision_macro ? (m.precision_macro * 100).toFixed(1) : '--');
    setMetric('recall-value', m.recall_macro ? (m.recall_macro * 100).toFixed(1) : '--');
    setMetric('f1-value', m.f1_macro ? (m.f1_macro * 100).toFixed(1) : '--');

    // Progress barları
    setProgressBar('accuracy-bar', m.accuracy ? m.accuracy * 100 : 0);
    setProgressBar('precision-bar', m.precision_macro ? m.precision_macro * 100 : 0);
    setProgressBar('recall-bar', m.recall_macro ? m.recall_macro * 100 : 0);
    setProgressBar('f1-bar', m.f1_macro ? m.f1_macro * 100 : 0);

    // Epoch sayısı
    const epochCount = h.total_epochs || (h.train_loss ? h.train_loss.length : '--');
    document.querySelectorAll('.epoch-count').forEach(el => el.textContent = `EPOCHS: ${epochCount}`);
}

function setMetric(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}

function setProgressBar(id, value) {
    const el = document.getElementById(id);
    if (el) el.style.width = `${value}%`;
}

// ========================
// Loss Grafiği (Chart.js)
// ========================
function renderLossChart(history) {
    const ctx = document.getElementById('loss-chart');
    if (!ctx) return;

    const epochs = history.train_loss.map((_, i) => i + 1);

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: epochs,
            datasets: [
                {
                    label: 'Eğitim Kaybı',
                    data: history.train_loss,
                    borderColor: COLORS.primary,
                    backgroundColor: COLORS.primary + '20',
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    fill: false,
                    tension: 0.3
                },
                {
                    label: 'Doğrulama Kaybı',
                    data: history.val_loss,
                    borderColor: COLORS.tertiary,
                    backgroundColor: COLORS.tertiary + '20',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    fill: false,
                    tension: 0.3
                }
            ]
        },
        options: getChartOptions('Loss')
    });
}

// ========================
// Accuracy Grafiği
// ========================
function renderAccuracyChart(history) {
    const ctx = document.getElementById('accuracy-chart');
    if (!ctx) return;

    const epochs = history.train_acc.map((_, i) => i + 1);
    const trainPct = history.train_acc.map(v => (v * 100));
    const valPct = history.val_acc.map(v => (v * 100));

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: epochs,
            datasets: [
                {
                    label: 'Eğitim Doğruluğu',
                    data: trainPct,
                    borderColor: COLORS.primary,
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    fill: false,
                    tension: 0.3
                },
                {
                    label: 'Doğrulama Doğruluğu',
                    data: valPct,
                    borderColor: COLORS.tertiary,
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    fill: false,
                    tension: 0.3
                }
            ]
        },
        options: getChartOptions('Accuracy (%)')
    });
}

function getChartOptions(yLabel) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'top',
                align: 'end',
                labels: {
                    color: COLORS.onSurface,
                    font: { family: 'JetBrains Mono', size: 11 },
                    boxWidth: 12,
                    padding: 12,
                    usePointStyle: true,
                    pointStyle: 'line'
                }
            },
            tooltip: {
                backgroundColor: COLORS.surfaceContainer,
                titleColor: COLORS.onSurface,
                bodyColor: COLORS.onSurfaceVariant,
                borderColor: COLORS.outline,
                borderWidth: 1,
                titleFont: { family: 'JetBrains Mono', size: 11 },
                bodyFont: { family: 'JetBrains Mono', size: 11 },
                padding: 10
            }
        },
        scales: {
            x: {
                title: {
                    display: true,
                    text: 'Epoch',
                    color: COLORS.onSurfaceVariant,
                    font: { family: 'JetBrains Mono', size: 11 }
                },
                ticks: { color: COLORS.onSurfaceVariant, font: { size: 10 } },
                grid: { color: COLORS.outline + '30' }
            },
            y: {
                title: {
                    display: true,
                    text: yLabel,
                    color: COLORS.onSurfaceVariant,
                    font: { family: 'JetBrains Mono', size: 11 }
                },
                ticks: { color: COLORS.onSurfaceVariant, font: { size: 10 } },
                grid: { color: COLORS.outline + '30' }
            }
        },
        interaction: { mode: 'index', intersect: false },
        animation: { duration: 1000, easing: 'easeOutQuart' }
    };
}

// ========================
// Confusion Matrix
// ========================
function renderConfusionMatrix(matrix, classNames) {
    const container = document.getElementById('confusion-matrix');
    if (!container) return;

    const displayNames = classNames.map(c => CLASS_DISPLAY[c] || c);
    const n = matrix.length;

    // Satır etiketleri (Y ekseni)
    const yLabels = document.getElementById('cm-y-labels');
    if (yLabels) {
        yLabels.innerHTML = displayNames.map(name =>
            `<div class="h-12 flex items-center justify-end pr-3 font-mono-label text-mono-label text-on-surface-variant whitespace-nowrap">${name}</div>`
        ).join('');
    }

    // Grid
    container.innerHTML = '';
    container.style.gridTemplateColumns = `repeat(${n}, minmax(0, 1fr))`;

    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            const val = matrix[i][j];
            const cell = document.createElement('div');
            const isDiagonal = i === j;

            // Renk yoğunluğu
            const alpha = Math.min(val * 0.95 + 0.05, 1);
            const bgOpacity = isDiagonal ? alpha : alpha * 0.6;

            cell.className = 'cm-cell h-12 flex items-center justify-center font-mono-label text-mono-label rounded-sm';

            if (val > 0.5) {
                cell.style.backgroundColor = `rgba(173, 198, 255, ${bgOpacity})`;
                cell.style.color = '#002e6a';
                cell.style.fontWeight = 'bold';
            } else if (val > 0.1) {
                cell.style.backgroundColor = `rgba(173, 198, 255, ${bgOpacity * 0.5})`;
                cell.style.color = COLORS.onSurface;
            } else {
                cell.style.backgroundColor = val > 0 ? `rgba(173, 198, 255, ${bgOpacity * 0.3})` : 'transparent';
                cell.style.color = COLORS.onSurfaceVariant + '80';
            }

            cell.textContent = val.toFixed(2);
            cell.title = `Gerçek: ${displayNames[i]} → Tahmin: ${displayNames[j]}: ${(val * 100).toFixed(1)}%`;

            container.appendChild(cell);
        }
    }

    // X ekseni etiketleri
    const xLabels = document.getElementById('cm-x-labels');
    if (xLabels) {
        xLabels.innerHTML = '';
        xLabels.style.gridTemplateColumns = `repeat(${n}, minmax(0, 1fr))`;
        displayNames.forEach(name => {
            const div = document.createElement('div');
            div.className = 'flex justify-center items-flex-start pt-1';
            div.style.cssText = 'writing-mode: vertical-rl; transform: rotate(180deg); text-align: left; height: 80px; font-family: "JetBrains Mono", monospace; font-size: 10px; letter-spacing: 0.05em; color: #c2c6d6; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;';
            div.textContent = name;
            xLabels.appendChild(div);
        });
    }
}

// ========================
// Per-Class Tablo
// ========================
function renderPerClassTable(perClass, metrics = {}) {
    const container = document.getElementById('per-class-table');
    if (!container) return;

    let rows = '';
    for (const [cls, vals] of Object.entries(perClass)) {
        const displayName = CLASS_DISPLAY[cls] || cls;
        rows += `
            <tr class="border-b border-outline-variant/30 hover:bg-surface-container-high/30 transition-colors">
                <td class="py-2 px-3 font-mono-label text-mono-label text-on-surface">${displayName}</td>
                <td class="py-2 px-3 font-mono-label text-mono-label text-on-surface-variant text-center">${(vals.precision * 100).toFixed(1)}%</td>
                <td class="py-2 px-3 font-mono-label text-mono-label text-on-surface-variant text-center">${(vals.recall * 100).toFixed(1)}%</td>
                <td class="py-2 px-3 font-mono-label text-mono-label text-primary text-center font-bold">${(vals.f1 * 100).toFixed(1)}%</td>
                <td class="py-2 px-3 font-mono-label text-mono-label text-on-surface-variant text-center">${vals.support}</td>
            </tr>
        `;
    }

    // Macro Average satırı
    if (metrics.precision_macro !== undefined) {
        rows += `
            <tr class="border-t-2 border-t-outline-variant border-b border-outline-variant/30 bg-surface-container-high">
                <td class="py-2 px-3 font-mono-label text-mono-label text-secondary font-bold">Macro Ortalama</td>
                <td class="py-2 px-3 font-mono-label text-mono-label text-secondary text-center">${(metrics.precision_macro * 100).toFixed(1)}%</td>
                <td class="py-2 px-3 font-mono-label text-mono-label text-secondary text-center">${(metrics.recall_macro * 100).toFixed(1)}%</td>
                <td class="py-2 px-3 font-mono-label text-mono-label text-primary text-center font-bold">${(metrics.f1_macro * 100).toFixed(1)}%</td>
                <td class="py-2 px-3 font-mono-label text-mono-label text-on-surface-variant text-center">—</td>
            </tr>
        `;
    }

    // Weighted Average satırı
    if (metrics.precision_weighted !== undefined) {
        rows += `
            <tr class="border-b border-outline-variant/30 bg-surface-container-high">
                <td class="py-2 px-3 font-mono-label text-mono-label text-secondary font-bold">Weighted Ortalama</td>
                <td class="py-2 px-3 font-mono-label text-mono-label text-secondary text-center">${(metrics.precision_weighted * 100).toFixed(1)}%</td>
                <td class="py-2 px-3 font-mono-label text-mono-label text-secondary text-center">${(metrics.recall_weighted * 100).toFixed(1)}%</td>
                <td class="py-2 px-3 font-mono-label text-mono-label text-primary text-center font-bold">${(metrics.f1_weighted * 100).toFixed(1)}%</td>
                <td class="py-2 px-3 font-mono-label text-mono-label text-on-surface-variant text-center">—</td>
            </tr>
        `;
    }

    container.innerHTML = rows;
}

// ========================
// Veri Yok Durumu
// ========================
function showNoData() {
    const sections = ['loss-chart-section', 'accuracy-chart-section', 'confusion-matrix-section'];
    sections.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            const inner = el.querySelector('.chart-container, .cm-wrapper');
            if (inner) {
                inner.innerHTML = `
                    <div class="flex flex-col items-center justify-center h-full text-center p-xl">
                        <span class="material-symbols-outlined text-[48px] text-on-surface-variant mb-sm">hourglass_empty</span>
                        <p class="font-body-base text-body-base text-on-surface-variant">Henüz eğitim verisi mevcut değil.</p>
                        <p class="font-body-sm text-body-sm text-on-surface-variant mt-1">Modeli eğittikten sonra grafikler burada görünecektir.</p>
                    </div>
                `;
            }
        }
    });
}

// Başlangıç
document.addEventListener('DOMContentLoaded', loadPerformanceData);
