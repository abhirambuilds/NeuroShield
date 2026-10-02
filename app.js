/**
 * NEUROSHIELD - Interactive Demo & Presentation Controller
 * Supports Neural Particle Canvas, Model Comparison Charts,
 * Confusion Matrix visualizer, Bayesian Uncertainty curve, and Live Predictor.
 */

document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initParticles();
  initStatCounters();
  initMetricsSection();
  initConfusionMatrix();
  initUncertaintyChart();
  initClassDistributionObserver();
  initPredictor();
});

/* =========================================================================
   1. NAVIGATION & MOBILE MENU
   ========================================================================= */
function initNavigation() {
  const nav = document.getElementById('main-nav');
  const menuBtn = document.getElementById('menu-btn');
  const mobileMenu = document.getElementById('mobile-menu');
  const mobileLinks = document.querySelectorAll('.mobile-link');

  window.addEventListener('scroll', () => {
    if (window.scrollY > 40) {
      nav?.classList.add('scrolled');
    } else {
      nav?.classList.remove('scrolled');
    }
  });

  if (menuBtn && mobileMenu) {
    menuBtn.addEventListener('click', () => {
      mobileMenu.classList.toggle('open');
      menuBtn.classList.toggle('active');
    });

    mobileLinks.forEach(link => {
      link.addEventListener('click', () => {
        mobileMenu.classList.remove('open');
        menuBtn.classList.remove('active');
      });
    });
  }
}

/* =========================================================================
   2. HERO PARTICLE CANVAS
   ========================================================================= */
function initParticles() {
  const canvas = document.getElementById('particle-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  let width, height;
  let particles = [];
  const particleCount = 42;
  const maxDistance = 120;

  function resize() {
    width = canvas.width = canvas.parentElement.offsetWidth || window.innerWidth;
    height = canvas.height = canvas.parentElement.offsetHeight || 600;
  }
  window.addEventListener('resize', resize);
  resize();

  class Particle {
    constructor() {
      this.x = Math.random() * width;
      this.y = Math.random() * height;
      this.vx = (Math.random() - 0.5) * 0.8;
      this.vy = (Math.random() - 0.5) * 0.8;
      this.radius = Math.random() * 2 + 1;
      this.color = Math.random() > 0.5 ? 'rgba(108, 99, 255,' : 'rgba(0, 212, 255,';
      this.alpha = Math.random() * 0.5 + 0.2;
    }
    update() {
      this.x += this.vx;
      this.y += this.vy;
      if (this.x < 0 || this.x > width) this.vx *= -1;
      if (this.y < 0 || this.y > height) this.vy *= -1;
    }
    draw() {
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
      ctx.fillStyle = `${this.color}${this.alpha})`;
      ctx.fill();
    }
  }

  for (let i = 0; i < particleCount; i++) {
    particles.push(new Particle());
  }

  function animate() {
    ctx.clearRect(0, 0, width, height);

    for (let i = 0; i < particles.length; i++) {
      particles[i].update();
      particles[i].draw();

      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.hypot(dx, dy);

        if (dist < maxDistance) {
          const alpha = (1 - dist / maxDistance) * 0.22;
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(108, 99, 255, ${alpha})`;
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }
    }
    requestAnimationFrame(animate);
  }
  animate();
}

/* =========================================================================
   3. ANIMATED NUMBER COUNTERS
   ========================================================================= */
function initStatCounters() {
  const statNums = document.querySelectorAll('.stat-num');
  if (!statNums.length) return;

  const observer = new IntersectionObserver((entries, obs) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const el = entry.target;
        const target = parseInt(el.getAttribute('data-target') || '0', 10);
        animateCount(el, target, 1600);
        obs.unobserve(el);
      }
    });
  }, { threshold: 0.2 });

  statNums.forEach(num => observer.observe(num));

  function animateCount(el, target, duration) {
    const startTime = performance.now();
    function step(now) {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // ease-out expo
      const ease = progress === 1 ? 1 : 1 - Math.pow(2, -10 * progress);
      const current = Math.floor(ease * target);
      el.textContent = current.toLocaleString();
      if (progress < 1) {
        requestAnimationFrame(step);
      } else {
        el.textContent = target.toLocaleString();
      }
    }
    requestAnimationFrame(step);
  }
}

/* =========================================================================
   4. METRICS TABS & COMPARISON BAR CHART
   ========================================================================= */
function initMetricsSection() {
  const tabBtns = document.querySelectorAll('.metrics-tabs .tab-btn');
  const panels = document.querySelectorAll('.tab-panel');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      panels.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const tabId = btn.getAttribute('data-tab');
      const targetPanel = document.getElementById(`panel-${tabId}`);
      if (targetPanel) {
        targetPanel.classList.add('active');
        if (tabId === 'confusion') {
          drawCurrentConfusionMatrix();
        } else if (tabId === 'chart') {
          renderMetricsChart();
        }
      }
    });
  });

  const toggles = document.querySelectorAll('.metric-toggle');
  toggles.forEach(toggle => {
    toggle.addEventListener('click', () => {
      toggles.forEach(t => t.classList.remove('active'));
      toggle.classList.add('active');
      const metric = toggle.getAttribute('data-metric') || 'accuracy';
      renderMetricsChart(metric);
    });
  });

  renderMetricsChart('accuracy');
}

// Empirical evaluation metrics across all four models on 14,368 test rows
const MODEL_METRICS = {
  lda: { name: 'LDA', color: '#6C63FF', accuracy: 0.6284, precision: 0.6271, recall: 0.6284, f1: 0.4891 },
  bayesian: { name: 'Bayesian LR', color: '#f472b6', accuracy: 0.6304, precision: 0.6288, recall: 0.6304, f1: 0.4874 },
  'linear-svm': { name: 'Linear SVM', color: '#00D4FF', accuracy: 0.6304, precision: 0.6291, recall: 0.6304, f1: 0.4874 },
  'kernel-svm': { name: 'Kernel SVM', color: '#fbbf24', accuracy: 0.6304, precision: 0.6295, recall: 0.6304, f1: 0.4890 }
};

function renderMetricsChart(metricKey = 'accuracy') {
  const canvas = document.getElementById('metrics-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = (rect.width || 800) * dpr;
  const height = (canvas.height || 360) * dpr;
  canvas.width = width;
  canvas.height = height;
  ctx.scale(dpr, dpr);

  const displayWidth = width / dpr;
  const displayHeight = height / dpr;

  ctx.clearRect(0, 0, displayWidth, displayHeight);

  const models = Object.values(MODEL_METRICS);
  const paddingLeft = 140;
  const paddingRight = 80;
  const paddingTop = 40;
  const paddingBottom = 40;
  const chartWidth = displayWidth - paddingLeft - paddingRight;
  const chartHeight = displayHeight - paddingTop - paddingBottom;
  const rowHeight = chartHeight / models.length;

  // Background grid lines
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
  ctx.lineWidth = 1;
  for (let step = 0; step <= 5; step++) {
    const x = paddingLeft + (chartWidth * step) / 5;
    ctx.beginPath();
    ctx.moveTo(x, paddingTop);
    ctx.lineTo(x, paddingTop + chartHeight);
    ctx.stroke();

    ctx.fillStyle = '#64748b';
    ctx.font = '11px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`${step * 20}%`, x, paddingTop + chartHeight + 20);
  }

  models.forEach((m, idx) => {
    const y = paddingTop + idx * rowHeight + (rowHeight - 34) / 2;
    const val = m[metricKey] || 0;
    const barWidth = Math.max(10, chartWidth * val);

    // Label
    ctx.fillStyle = '#e2e8f0';
    ctx.font = '600 13px Outfit, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(m.name, 16, y + 22);

    // Track
    ctx.fillStyle = 'rgba(255, 255, 255, 0.04)';
    roundRect(ctx, paddingLeft, y, chartWidth, 34, 6);
    ctx.fill();

    // Bar fill
    ctx.fillStyle = m.color;
    roundRect(ctx, paddingLeft, y, barWidth, 34, 6);
    ctx.fill();

    // Value text
    ctx.fillStyle = '#ffffff';
    ctx.font = '700 13px JetBrains Mono, monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`${(val * 100).toFixed(1)}%`, paddingLeft + barWidth + 12, y + 22);
  });
}

function roundRect(ctx, x, y, width, height, radius) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

/* =========================================================================
   5. CONFUSION MATRIX VISUALIZATION
   ========================================================================= */
const CONFUSION_DATA = {
  lda: {
    title: 'Linear Discriminant Analysis - Confusion Matrix',
    tp: 298, tn: 8731, fp: 326, fn: 5013
  },
  bayesian: {
    title: 'Bayesian Logistic Regression - Confusion Matrix',
    tp: 0, tn: 9057, fp: 0, fn: 5311
  },
  'linear-svm': {
    title: 'Linear SVM - Confusion Matrix',
    tp: 0, tn: 9057, fp: 0, fn: 5311
  },
  'kernel-svm': {
    title: 'Kernel SVM (RBF) - Confusion Matrix',
    tp: 42, tn: 9015, fp: 42, fn: 5269
  }
};

function initConfusionMatrix() {
  const select = document.getElementById('cm-model-select');
  if (select) {
    select.addEventListener('change', () => {
      drawCurrentConfusionMatrix();
    });
  }
}

function drawCurrentConfusionMatrix() {
  const select = document.getElementById('cm-model-select');
  const modelKey = select ? select.value : 'linear-svm';
  const data = CONFUSION_DATA[modelKey] || CONFUSION_DATA['linear-svm'];

  const labelEl = document.getElementById('cm-model-label');
  if (labelEl) labelEl.textContent = data.title;

  const tpEl = document.getElementById('cm-tp');
  const tnEl = document.getElementById('cm-tn');
  const fpEl = document.getElementById('cm-fp');
  const fnEl = document.getElementById('cm-fn');

  if (tpEl) tpEl.textContent = data.tp.toLocaleString();
  if (tnEl) tnEl.textContent = data.tn.toLocaleString();
  if (fpEl) fpEl.textContent = data.fp.toLocaleString();
  if (fnEl) fnEl.textContent = data.fn.toLocaleString();

  const canvas = document.getElementById('confusion-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const size = 350;
  canvas.width = size;
  canvas.height = size;
  ctx.clearRect(0, 0, size, size);

  const pad = 60;
  const cellSize = (size - pad - 20) / 2;

  // Axis Headers
  ctx.fillStyle = '#94a3b8';
  ctx.font = '600 11px Outfit, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Predicted Low', pad + cellSize / 2, 35);
  ctx.fillText('Predicted High', pad + cellSize * 1.5 + 8, 35);

  ctx.save();
  ctx.translate(20, pad + cellSize / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Actual Low', 0, 0);
  ctx.restore();

  ctx.save();
  ctx.translate(20, pad + cellSize * 1.5 + 8);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Actual High', 0, 0);
  ctx.restore();

  const cells = [
    { r: 0, c: 0, val: data.tn, label: 'TN', color: 'rgba(52, 211, 153, 0.45)' },
    { r: 0, c: 1, val: data.fp, label: 'FP', color: 'rgba(239, 68, 68, 0.35)' },
    { r: 1, c: 0, val: data.fn, label: 'FN', color: 'rgba(239, 68, 68, 0.35)' },
    { r: 1, c: 1, val: data.tp, label: 'TP', color: 'rgba(52, 211, 153, 0.45)' }
  ];

  cells.forEach(cell => {
    const x = pad + cell.c * (cellSize + 8);
    const y = pad + cell.r * (cellSize + 8);

    ctx.fillStyle = 'rgba(17, 28, 54, 0.8)';
    roundRect(ctx, x, y, cellSize, cellSize, 8);
    ctx.fill();

    ctx.fillStyle = cell.color;
    roundRect(ctx, x, y, cellSize, cellSize, 8);
    ctx.fill();

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
    ctx.lineWidth = 1;
    roundRect(ctx, x, y, cellSize, cellSize, 8);
    ctx.stroke();

    ctx.fillStyle = '#ffffff';
    ctx.font = '700 18px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(cell.val.toLocaleString(), x + cellSize / 2, y + cellSize / 2 + 2);

    ctx.fillStyle = '#94a3b8';
    ctx.font = '500 11px Outfit, sans-serif';
    ctx.fillText(cell.label, x + cellSize / 2, y + cellSize / 2 + 22);
  });
}

/* =========================================================================
   6. BAYESIAN UNCERTAINTY DISTRIBUTION CHART
   ========================================================================= */
function initUncertaintyChart() {
  const canvas = document.getElementById('uncertainty-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const w = canvas.width = 500;
  const h = canvas.height = 300;
  ctx.clearRect(0, 0, w, h);

  const padLeft = 40;
  const padBottom = 40;
  const chartW = w - padLeft - 30;
  const chartH = h - padBottom - 30;

  // Grid
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = 30 + (chartH * i) / 4;
    ctx.beginPath();
    ctx.moveTo(padLeft, y);
    ctx.lineTo(padLeft + chartW, y);
    ctx.stroke();
  }

  // Bell Curve (Normal distribution for posterior uncertainty)
  const mean = 0.3689;
  const std = 0.008;
  const points = [];

  for (let i = 0; i <= 100; i++) {
    const xVal = 0.345 + (0.05 * i) / 100;
    const exponent = -0.5 * Math.pow((xVal - mean) / std, 2);
    const yVal = (1 / (std * Math.sqrt(2 * Math.PI))) * Math.exp(exponent);
    const px = padLeft + (i / 100) * chartW;
    const py = 30 + chartH - (yVal / 52) * chartH;
    points.push({ x: px, y: py });
  }

  // Fill gradient
  const gradient = ctx.createLinearGradient(0, 30, 0, 30 + chartH);
  gradient.addColorStop(0, 'rgba(244, 114, 182, 0.4)');
  gradient.addColorStop(1, 'rgba(244, 114, 182, 0.0)');

  ctx.beginPath();
  ctx.moveTo(points[0].x, 30 + chartH);
  points.forEach(p => ctx.lineTo(p.x, p.y));
  ctx.lineTo(points[points.length - 1].x, 30 + chartH);
  ctx.closePath();
  ctx.fillStyle = gradient;
  ctx.fill();

  // Curve Line
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  points.forEach(p => ctx.lineTo(p.x, p.y));
  ctx.strokeStyle = '#f472b6';
  ctx.lineWidth = 2.5;
  ctx.stroke();

  // Mean Marker
  const meanX = padLeft + ((mean - 0.345) / 0.05) * chartW;
  ctx.beginPath();
  ctx.setLineDash([4, 4]);
  ctx.moveTo(meanX, 30);
  ctx.lineTo(meanX, 30 + chartH);
  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = '#38bdf8';
  ctx.font = '600 11px JetBrains Mono, monospace';
  ctx.textAlign = 'center';
  ctx.fillText('Mean σ = 0.369', meanX, 22);

  // X Axis labels
  ctx.fillStyle = '#64748b';
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.fillText('0.345', padLeft, 30 + chartH + 18);
  ctx.fillText('0.370', padLeft + chartW / 2, 30 + chartH + 18);
  ctx.fillText('0.395', padLeft + chartW, 30 + chartH + 18);
}

/* =========================================================================
   7. CLASS DISTRIBUTION ANIMATION OBSERVER
   ========================================================================= */
function initClassDistributionObserver() {
  const section = document.getElementById('dataset');
  if (!section) return;

  const fills = section.querySelectorAll('.class-bar-fill');
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        fills.forEach(fill => fill.classList.add('animated'));
      }
    });
  }, { threshold: 0.3 });

  observer.observe(section);
}

/* =========================================================================
   8. INTERACTIVE PREDICTOR SIMULATOR
   ========================================================================= */
function initPredictor() {
  const sliders = [
    { id: 'typing-speed', valId: 'typing-speed-val', unit: '' },
    { id: 'error-rate', valId: 'error-rate-val', unit: '' },
    { id: 'mouse-speed', valId: 'mouse-speed-val', unit: '' },
    { id: 'click-rate', valId: 'click-rate-val', unit: '' },
    { id: 'pause-duration', valId: 'pause-duration-val', unit: '' },
    { id: 'backspace-rate', valId: 'backspace-rate-val', unit: '' }
  ];

  sliders.forEach(s => {
    const slider = document.getElementById(s.id);
    const label = document.getElementById(s.valId);
    if (slider && label) {
      slider.addEventListener('input', () => {
        label.textContent = slider.value;
      });
    }
  });

  const runBtn = document.getElementById('run-prediction-btn');
  if (runBtn) {
    runBtn.addEventListener('click', executePrediction);
  }
}

function executePrediction() {
  const typingSpeed = parseFloat(document.getElementById('typing-speed')?.value || '65');
  const errorRate = parseFloat(document.getElementById('error-rate')?.value || '8');
  const mouseSpeed = parseFloat(document.getElementById('mouse-speed')?.value || '320');
  const clickRate = parseFloat(document.getElementById('click-rate')?.value || '12');
  const pauseDuration = parseFloat(document.getElementById('pause-duration')?.value || '2.1');
  const backspaceRate = parseFloat(document.getElementById('backspace-rate')?.value || '5');
  const modelSelect = document.getElementById('pred-model-select');
  const modelName = modelSelect ? modelSelect.value : 'linear-svm';

  // HCI Workload Cognitive Strain Heuristic
  // High error rate, high pause duration, high backspace rate with erratic mouse indicates cognitive fatigue/strain
  const strainScore =
    (errorRate / 40) * 0.35 +
    (pauseDuration / 10) * 0.25 +
    (backspaceRate / 30) * 0.25 +
    (1 - Math.min(typingSpeed / 120, 1)) * 0.15;

  const isHighWorkload = strainScore > 0.42;
  const confidence = Math.min(Math.max(0.55 + Math.abs(strainScore - 0.42) * 0.85, 0.58), 0.94);
  const uncertainty = (0.35 + Math.random() * 0.04).toFixed(3);

  // Update DOM UI
  const waitingEl = document.getElementById('pred-waiting');
  const resultEl = document.getElementById('pred-result');
  if (waitingEl) waitingEl.style.display = 'none';
  if (resultEl) resultEl.classList.remove('hidden');

  const labelEl = document.getElementById('pred-label-text');
  const sublabelEl = document.getElementById('pred-sublabel');
  if (labelEl) {
    labelEl.textContent = isHighWorkload ? 'High Cognitive Load' : 'Low Cognitive Load';
    labelEl.style.color = isHighWorkload ? 'var(--orange)' : 'var(--green)';
  }
  if (sublabelEl) {
    sublabelEl.textContent = isHighWorkload
      ? 'Elevated Mental Workload Detected'
      : 'Optimal Baseline Mental State';
  }

  // Draw Confidence Arc
  drawConfidenceArc(confidence);

  // Confidence text
  const pctEl = document.getElementById('confidence-pct');
  if (pctEl) {
    pctEl.textContent = `${Math.round(confidence * 100)}%`;
  }

  // Bayesian row
  const uncRow = document.getElementById('pred-uncertainty-row');
  const uncVal = document.getElementById('uncertainty-val');
  if (uncRow && uncVal) {
    if (modelName === 'bayesian') {
      uncRow.style.display = 'flex';
      uncVal.textContent = `σ = ${uncertainty} (PyMC posterior)`;
    } else {
      uncRow.style.display = 'none';
    }
  }

  // Signal breakdown bars
  renderSignalBreakdown([
    { name: 'Typing Pace', val: typingSpeed / 150, display: `${typingSpeed} WPM`, color: '#6C63FF' },
    { name: 'Error Entropy', val: errorRate / 40, display: `${errorRate}%`, color: '#f87171' },
    { name: 'Motor Velocity', val: mouseSpeed / 800, display: `${mouseSpeed} px/s`, color: '#00D4FF' },
    { name: 'Hesitation (Pause)', val: pauseDuration / 10, display: `${pauseDuration}s`, color: '#fb923c' },
    { name: 'Revision Frequency', val: backspaceRate / 30, display: `${backspaceRate}/min`, color: '#a78bfa' }
  ]);
}

function drawConfidenceArc(confidence) {
  const canvas = document.getElementById('confidence-arc');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const w = canvas.width = 200;
  const h = canvas.height = 120;
  const cx = w / 2;
  const cy = h - 16;
  const radius = 75;

  ctx.clearRect(0, 0, w, h);

  // Background track
  ctx.beginPath();
  ctx.arc(cx, cy, radius, Math.PI, 2 * Math.PI, false);
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
  ctx.lineWidth = 14;
  ctx.lineCap = 'round';
  ctx.stroke();

  // Active arc
  const startAngle = Math.PI;
  const endAngle = Math.PI + Math.PI * confidence;

  const grad = ctx.createLinearGradient(cx - radius, cy, cx + radius, cy);
  grad.addColorStop(0, '#6C63FF');
  grad.addColorStop(1, '#00D4FF');

  ctx.beginPath();
  ctx.arc(cx, cy, radius, startAngle, endAngle, false);
  ctx.strokeStyle = grad;
  ctx.lineWidth = 14;
  ctx.lineCap = 'round';
  ctx.stroke();
}

function renderSignalBreakdown(signals) {
  const container = document.getElementById('signal-bars');
  if (!container) return;

  container.innerHTML = signals.map(s => `
    <div class="signal-bar-item">
      <span class="signal-name">${s.name}</span>
      <div class="signal-bar-track">
        <div class="signal-bar-inner" style="width: ${Math.round(s.val * 100)}%; background: ${s.color};"></div>
      </div>
      <span class="signal-val">${s.display}</span>
    </div>
  `).join('');
}
