const API = "http://127.0.0.1:8000";

let currentThreshold = 50;
let currentOpacity = 70;
let lastResult = null;
let selectedFile = null;

window.addEventListener("DOMContentLoaded", () => {
  checkAPI();
  loadHistory();
  setupDragDrop();
  checkShareParam();
  setInterval(checkAPI, 30_000);
});

function toggleTheme() {
  const html = document.documentElement;
  const next = html.dataset.theme === "dark" ? "light" : "dark";
  html.dataset.theme = next;
  localStorage.setItem("chestai_theme", next);
}

(function () {
  const saved = localStorage.getItem("chestai_theme");
  if (saved) document.documentElement.dataset.theme = saved;
})();

function showPanel(name, el) {
  document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));
  document.getElementById(`panel-${name}`).classList.add("active");
  el.classList.add("active");

  const titles = {
    upload: "Chest X-Ray Analysis",
    history: "Analysis History",
    report: "Structured Report",
    about: "About ChestAI"
  };
  document.getElementById("panelTitle").textContent = titles[name] || "";
  if (window.innerWidth <= 800) {
    document.getElementById("sidebar").classList.remove("open");
  }
}

function toggleSidebar() {
  document.getElementById("sidebar").classList.toggle("open");
}

async function checkAPI() {
  const dot = document.getElementById("statusDot");
  const text = document.getElementById("statusText");
  try {
    const r = await fetch(`${API}/health`, { signal: AbortSignal.timeout(4000) });
    if (r.ok) {
      dot.className = "status-dot online";
      text.textContent = "Online";
    } else throw new Error();
  } catch {
    dot.className = "status-dot offline";
    text.textContent = "Offline";
  }
}

function setupDragDrop() {
  const zone = document.getElementById("dropZone");

  zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("drag-over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
  zone.addEventListener("drop", e => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  });

  document.getElementById("fileInput").addEventListener("change", e => {
    if (e.target.files[0]) handleFile(e.target.files[0]);
  });
}

function handleFile(file) {
  hideError();

  if (!["image/jpeg", "image/png"].includes(file.type)) {
    showError("Only JPEG and PNG files are supported.");
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showError("File exceeds 10 MB limit.");
    return;
  }

  selectedFile = file;

  const reader = new FileReader();
  reader.onload = e => {
    document.getElementById("previewImg").src = e.target.result;
    document.getElementById("previewName").textContent = file.name;
    document.getElementById("previewSize").textContent = formatBytes(file.size);

    document.getElementById("uploadInner").classList.add("hidden");
    document.getElementById("uploadPreview").classList.remove("hidden");
    document.getElementById("predictBtn").disabled = false;
  };
  reader.readAsDataURL(file);
}

function clearUpload() {
  selectedFile = null;
  document.getElementById("fileInput").value = "";
  document.getElementById("uploadInner").classList.remove("hidden");
  document.getElementById("uploadPreview").classList.add("hidden");
  document.getElementById("predictBtn").disabled = true;
  hideError();
}

function showError(msg) {
  const el = document.getElementById("uploadError");
  el.textContent = msg;
  el.classList.remove("hidden");
}

function hideError() {
  document.getElementById("uploadError").classList.add("hidden");
}

function updateThresh(v) {
  currentThreshold = parseInt(v);
  document.getElementById("threshLabel").textContent = v + "%";
  applyThresholdFilter();
}

function updateOpacity(v) {
  currentOpacity = parseInt(v);
  document.getElementById("opacLabel").textContent = v + "%";
  document.querySelectorAll(".card-img-wrap img").forEach(img => {
    img.style.opacity = currentOpacity / 100;
  });
}

function applyThresholdFilter() {
  document.querySelectorAll(".gradcam-card").forEach(card => {
    const prob = parseFloat(card.dataset.prob) * 100;
    card.classList.toggle("filtered", prob < currentThreshold);
  });
}

async function runPredict() {
  if (!selectedFile) return;

  const btn = document.getElementById("predictBtn");
  btn.disabled = true;
  btn.classList.add("loading");
  btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="animation:spin 0.8s linear infinite"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg> Analyzing…`;

  renderSkeletons(5);
  document.getElementById("resultsHeader").classList.remove("hidden");
  document.getElementById("summaryBanner").classList.add("hidden");

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch(`${API}/predict`, { method: "POST", body: formData });

    if (!response.ok) throw new Error(`Server error ${response.status}`);

    const data = await response.json();

    if (data.error) {
      showError(data.error);
      clearGrid();
      return;
    }

    lastResult = data;
    renderResults(data);
    renderSummary(data);
    renderReport(data);
    saveToHistory(data);

  } catch (err) {
    showError(`Failed to connect to API. Is the server running? (${err.message})`);
    clearGrid();
    document.getElementById("resultsHeader").classList.add("hidden");
  } finally {
    btn.disabled = false;
    btn.classList.remove("loading");
    btn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg> Analyze X-Ray`;
  }
}

function renderSkeletons(n) {
  const grid = document.getElementById("cardsGrid");
  grid.innerHTML = Array.from({ length: n }, () => `
    <div class="skeleton-card">
      <div class="skel-img"></div>
      <div class="skel-body">
        <div class="skel-line" style="width:70%"></div>
        <div class="skel-line short"></div>
        <div class="skel-line" style="width:90%"></div>
      </div>
    </div>`).join("");
}

function clearGrid() {
  document.getElementById("cardsGrid").innerHTML = "";
}

function renderResults(data) {
  const grid = document.getElementById("cardsGrid");
  grid.innerHTML = "";

  data.results.forEach((item, idx) => {
    const isTop = idx === 0;
    const sevCls = sevClass(item.severity);
    const barCls = barClass(item.severity);
    const pct = (item.prob * 100).toFixed(1);
    const delayed = idx * 60;

    const card = document.createElement("div");
    card.className = `gradcam-card${isTop ? " top-pred" : ""}`;
    card.dataset.prob = item.prob;
    card.style.animationDelay = `${delayed}ms`;
    card.title = "Click to zoom";

    card.innerHTML = `
      <div class="card-img-wrap">
        <img src="data:image/jpeg;base64,${item.image}" alt="${item.class} GradCAM" style="opacity:${currentOpacity / 100}">
        ${isTop ? `<span class="card-badge badge-top">Top Finding</span>` : `<span class="card-badge badge-${item.severity.toLowerCase()}">${item.severity}</span>`}
      </div>
      <div class="card-body">
        <div class="card-disease">${item.class}</div>
        <div class="card-meta">
          <span class="card-prob">${pct}%</span>
          <span class="card-sev ${sevCls}">${item.severity}</span>
        </div>
        <div class="conf-bar-bg">
          <div class="conf-bar-fill ${barCls}" style="width:${pct}%"></div>
        </div>
        <div class="card-pred-row">
          <span class="pred-dot ${item.pred === 1 ? 'positive' : 'negative'}"></span>
          <span>${item.pred === 1 ? "Detected (above threshold)" : "Not detected"}</span>
        </div>
      </div>`;

    card.addEventListener("click", () => openModal(item));
    grid.appendChild(card);
  });

  applyThresholdFilter();
}

function renderSummary(data) {
  document.getElementById("summaryBanner").classList.remove("hidden");
  document.getElementById("summaryOriginal").src = `data:image/jpeg;base64,${data.original}`;

  const pid = document.getElementById("patientId").value.trim() || "—";
  document.getElementById("statPatient").textContent = pid;
  document.getElementById("statTop").textContent = data.top_class;
  document.getElementById("statConf").textContent = (data.top_prob * 100).toFixed(1) + "%";
  document.getElementById("statSev").textContent = data.top_severity;
  document.getElementById("statTime").textContent = formatTime(data.timestamp);
  document.getElementById("statDevice").textContent = data.model_info?.device || "cpu";
}

function renderReport(data) {
  const pid = document.getElementById("patientId").value.trim() || "Unknown";
  const ts = formatTime(data.timestamp);

  const findings = data.results.map(r => `
    <div class="report-finding-item">
      <div>
        <div class="rf-name">${r.class}</div>
        <div style="font-size:11px;color:var(--text3);font-family:var(--font-mono);margin-top:2px">${r.severity} · ${r.pred === 1 ? "Detected" : "Not detected"}</div>
      </div>
      <div class="rf-prob">${(r.prob * 100).toFixed(1)}%</div>
    </div>`).join("");

  document.getElementById("reportContent").innerHTML = `
    <div class="report-header-section">
      <div>
        <div class="report-logo">ChestAI</div>
        <div style="font-size:13px;color:var(--text2);margin-top:4px">Radiology Intelligence Platform</div>
      </div>
      <div class="report-meta">
        Patient ID: <strong>${pid}</strong><br>
        Date: ${ts}<br>
        Model: ${data.model_info?.name || "DenseNet-ECA"}<br>
        Threshold: ${data.model_info?.threshold ?? 0.5}
      </div>
    </div>
    <div class="report-section-title">Primary Finding</div>
    <div style="background:var(--cyan-dim);border:1px solid var(--cyan);border-radius:10px;padding:14px 18px;">
      <div style="font-family:var(--font-head);font-size:18px;font-weight:600">${data.top_class}</div>
      <div style="font-family:var(--font-mono);font-size:14px;color:var(--cyan);margin-top:4px">${(data.top_prob * 100).toFixed(1)}% confidence · ${data.top_severity} severity</div>
    </div>
    <div class="report-section-title">All Findings</div>
    <div class="report-findings">${findings}</div>
    <div class="report-section-title">Disclaimer</div>
    <div class="disclaimer-box" style="margin-top:0">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
      <p>This report is generated by an AI system and is for research purposes only. It does not constitute a clinical diagnosis. Results must be reviewed and validated by a qualified radiologist before any clinical decision is made.</p>
    </div>`;
}

function saveToHistory(data) {
  let history = getHistory();
  const pid = document.getElementById("patientId").value.trim() || "Unknown";

  history.unshift({
    pid,
    top_class: data.top_class,
    top_prob: data.top_prob,
    top_severity: data.top_severity,
    timestamp: data.timestamp,
    original: data.original,
    full: data
  });

  history = history.slice(0, 8);
  localStorage.setItem("chestai_history", JSON.stringify(history));
  loadHistory();
}

function getHistory() {
  try { return JSON.parse(localStorage.getItem("chestai_history") || "[]"); }
  catch { return []; }
}

function loadHistory() {
  const history = getHistory();
  document.getElementById("histBadge").textContent = history.length;

  const list = document.getElementById("historyList");

  if (history.length === 0) {
    list.innerHTML = `<div class="empty-state">
      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
      <p>No analyses yet. Upload an X-ray to get started.</p>
    </div>`;
    return;
  }

  list.innerHTML = history.map((h, i) => `
    <div class="history-item" onclick="restoreHistory(${i})">
      <img class="history-thumb" src="data:image/jpeg;base64,${h.original}" alt="thumb">
      <div class="history-info">
        <div class="history-title">${h.pid} — ${h.top_class}</div>
        <div class="history-sub">${formatTime(h.timestamp)}</div>
      </div>
      <div class="history-conf">${(h.top_prob * 100).toFixed(1)}%</div>
    </div>`).join("");
}

function restoreHistory(idx) {
  const history = getHistory();
  const h = history[idx];
  if (!h) return;

  lastResult = h.full;
  renderResults(h.full);
  renderSummary(h.full);
  renderReport(h.full);

  showPanel("upload", document.querySelector('[data-panel="upload"]'));
  document.getElementById("resultsHeader").classList.remove("hidden");
  showToast("History loaded — " + h.top_class);
}

function clearHistory() {
  localStorage.removeItem("chestai_history");
  loadHistory();
  showToast("History cleared");
}

function openModal(item) {
  const pct = (item.prob * 100).toFixed(1);
  document.getElementById("modalImg").src = `data:image/jpeg;base64,${item.image}`;
  document.getElementById("modalCaption").textContent = `${item.class}  ·  ${pct}% confidence  ·  ${item.severity} severity`;
  document.getElementById("modalBackdrop").classList.remove("hidden");
  document.body.style.overflow = "hidden";
}

function closeModal() {
  document.getElementById("modalBackdrop").classList.add("hidden");
  document.body.style.overflow = "";
}

document.addEventListener("keydown", e => {
  if (e.key === "Escape") closeModal();
});

async function exportPDF() {
  if (!lastResult) { showToast("Run an analysis first."); return; }

  showToast("Generating PDF…");

  const { jsPDF } = window.jspdf;
  const doc = new jsPDF({ unit: "mm", format: "a4" });
  const pid = document.getElementById("patientId").value.trim() || "Unknown";
  const ts = formatTime(lastResult.timestamp);

  doc.setFont("helvetica", "bold");
  doc.setFontSize(20);
  doc.setTextColor(2, 100, 180);
  doc.text("ChestAI — Radiology Report", 20, 22);

  doc.setDrawColor(2, 100, 180);
  doc.setLineWidth(0.5);
  doc.line(20, 26, 190, 26);

  doc.setFont("helvetica", "normal");
  doc.setFontSize(10);
  doc.setTextColor(80, 80, 80);
  doc.text(`Patient ID: ${pid}`, 20, 34);
  doc.text(`Date: ${ts}`, 20, 40);
  doc.text(`Model: ${lastResult.model_info?.name || "DenseNet-ECA"}  |  Threshold: ${lastResult.model_info?.threshold}  |  Device: ${lastResult.model_info?.device}`, 20, 46);

  doc.setFontSize(12);
  doc.setFont("helvetica", "bold");
  doc.setTextColor(30, 30, 30);
  doc.text("Primary Finding", 20, 58);
  doc.setFont("helvetica", "normal");
  doc.setFontSize(11);
  doc.text(`${lastResult.top_class}  —  ${(lastResult.top_prob * 100).toFixed(1)}%  (${lastResult.top_severity})`, 20, 65);

  doc.setFontSize(12);
  doc.setFont("helvetica", "bold");
  doc.text("All Findings", 20, 78);

  let y = 86;
  doc.setFontSize(10);
  lastResult.results.forEach(r => {
    doc.setFont("helvetica", "bold");
    doc.setTextColor(30, 30, 30);
    doc.text(r.class, 20, y);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(100, 100, 100);
    doc.text(`${(r.prob * 100).toFixed(1)}%  |  ${r.severity}  |  ${r.pred ? "Detected" : "Not detected"}`, 90, y);
    y += 8;
  });

  if (lastResult.original) {
    y += 6;
    doc.setFont("helvetica", "bold");
    doc.setFontSize(12);
    doc.setTextColor(30, 30, 30);
    doc.text("Original X-Ray", 20, y);
    y += 5;
    doc.addImage(`data:image/jpeg;base64,${lastResult.original}`, "JPEG", 20, y, 50, 50);
  }

  const dY = 270;
  doc.setFontSize(8);
  doc.setTextColor(160, 100, 0);
  doc.setFont("helvetica", "italic");
  doc.text("⚕ For research and educational use only. Not a substitute for clinical diagnosis. Consult a qualified radiologist.", 20, dY);

  doc.save(`ChestAI_Report_${pid}_${Date.now()}.pdf`);
  showToast("PDF exported successfully");
}

function copyShareLink() {
  if (!lastResult) { showToast("Run an analysis first."); return; }

  try {
    const payload = {
      top: lastResult.top_class,
      prob: lastResult.top_prob,
      severity: lastResult.top_severity,
      timestamp: lastResult.timestamp
    };
    const b64 = btoa(JSON.stringify(payload));
    const url = `${location.href.split("?")[0]}?r=${b64}`;
    navigator.clipboard.writeText(url).then(() => showToast("Share link copied to clipboard"));
  } catch {
    showToast("Could not copy link");
  }
}

function checkShareParam() {
  const params = new URLSearchParams(location.search);
  const r = params.get("r");
  if (!r) return;
  try {
    const data = JSON.parse(atob(r));
    showToast(`Shared result: ${data.top} (${(data.prob * 100).toFixed(1)}%)`, 5000);
  } catch { }
}

let toastTimer;
function showToast(msg, duration = 2800) {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add("hidden"), duration);
}

function sevClass(s) {
  const m = { High: "sev-high", Moderate: "sev-mod", Low: "sev-low", Normal: "sev-normal" };
  return m[s] || "sev-normal";
}

function barClass(s) {
  const m = { High: "bar-high", Moderate: "bar-mod", Low: "bar-low", Normal: "bar-normal" };
  return m[s] || "bar-normal";
}

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

function formatTime(iso) {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString(undefined, {
      year: "numeric", month: "short", day: "2-digit",
      hour: "2-digit", minute: "2-digit"
    });
  } catch { return iso; }
}

const spinStyle = document.createElement("style");
spinStyle.textContent = `@keyframes spin { to { transform: rotate(360deg); } }`;
document.head.appendChild(spinStyle);
