/* DERICH YOLO Auto-Training Studio — WebSocket Client & Interactive Canvas UI */

const CLASS_COLORS = [
  "#3b82f6", "#10b981", "#f59e0b", "#f43f5e", "#8b5cf6",
  "#06b6d4", "#ec4899", "#84cc16", "#f97316", "#14b8a6"
];

const state = {
  ws: null,
  wsReconnectDelay: 1200,
  bootstrap: null,
  telemetry: null,
  activeProjectId: null,
  images: [],
  health: null,
  selectedImageIdx: -1,
  selectedImageFilename: null,
  selectedThumbFilenames: new Set(),
  projectLoadSeq: 0,
  editorBoxes: [],
  selectedBoxIdx: -1,
  editorImgObj: null,
  editorImgUrl: null,
  canvasZoom: 1.0,
  drawing: false,
  drawStart: null,
  drawCurr: null,
  ui: {
    sidebarCollapsed: false,
    galleryCollapsed: false,
    galleryWide: false,
    setupToolsCollapsed: false,
    focusLabelMode: false,
    preFocusSnapshot: null,
  },
  activePresetId: "small_shapes_default",
  activeModelFamily: "yolo26",
  activeParamGroup: "all",
  activePlatformOs: "all",
  params: {},
  autoExportFormats: ["pt", "onnx"],
  selectedExportFormats: new Set(["pt", "tf32_bundle", "onnx"]),
  training: null,
  runs: [],
  artifacts: [],
  webcamStream: null,
  burstTimer: null,
  webcamCaptured: 0,
};

const $ = (id) => document.getElementById(id);

function classColor(clsId) {
  return CLASS_COLORS[Math.abs(clsId || 0) % CLASS_COLORS.length];
}

function toast(msg, kind = "info", duration = 4200) {
  const stack = $("toastStack");
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = msg;
  stack.appendChild(el);
  setTimeout(() => {
    el.style.opacity = "0";
    setTimeout(() => el.remove(), 250);
  }, duration);
}

async function api(url, opts = {}) {
  const res = await fetch(url, opts);
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = { detail: text }; }
  if (!res.ok) {
    throw new Error((data && (data.detail || data.message)) || `HTTP ${res.status}`);
  }
  return data;
}

function formatSeconds(sec) {
  const s = Math.max(0, Math.round(sec || 0));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const rem = s % 60;
  if (m < 60) return `${m}p ${rem}s`;
  return `${Math.floor(m / 60)}g ${m % 60}p`;
}

// ======================================================================
// WebSocket Real-Time Link
// ======================================================================

function connectWebSocket() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const url = `${proto}://${location.host}/ws`;
  setWsBadge("connecting", "Đang nối WS...");

  const ws = new WebSocket(url);
  state.ws = ws;

  ws.onopen = () => {
    state.wsReconnectDelay = 1200;
    setWsBadge("connected", "WebSocket Live");
    appendConsole({ time: nowTime(), source: "ws", level: "success", message: `Đã kết nối WebSocket (${url})` });
  };

  ws.onmessage = (evt) => {
    try {
      const msg = JSON.parse(evt.data);
      handleWsEvent(msg);
    } catch (err) {
      console.error("WS parse error:", err);
    }
  };

  ws.onclose = () => {
    setWsBadge("disconnected", "Mất nối · Thử lại...");
    setTimeout(() => {
      state.wsReconnectDelay = Math.min(6000, state.wsReconnectDelay * 1.4);
      connectWebSocket();
    }, state.wsReconnectDelay);
  };

  ws.onerror = () => {
    ws.close();
  };
}

function wsSend(action, payload = {}) {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ action, payload }));
    return true;
  }
  return false;
}

function setWsBadge(cls, text) {
  const b = $("wsBadge");
  b.className = `ws-badge ws-${cls}`;
  $("wsBadgeText").textContent = text;
}

function nowTime() {
  return new Date().toTimeString().slice(0, 8);
}

function handleWsEvent(msg) {
  const { type, data } = msg;
  if (type === "connected") {
    if (data && Array.isArray(data.recent_logs)) {
      data.recent_logs.forEach((l) => appendConsole(l.data || l));
    }
  } else if (type === "telemetry") {
    renderTelemetry(data);
  } else if (type === "log") {
    appendConsole(data);
  } else if (type === "task_progress") {
    showGlobalProgress(data.message, data.progress);
    if (data.progress >= 100) {
      setTimeout(() => $("globalProgressBar").classList.add("hidden"), 1800);
    }
  } else if (type === "dataset_updated") {
    if (!data || !data.project_id || data.project_id === state.activeProjectId) {
      loadActiveProjectData();
    }
  } else if (type === "train_status") {
    state.training = data;
    renderTrainingState();
  } else if (type === "train_metrics") {
    state.training = data.state;
    renderTrainingState();
  } else if (type === "train_done") {
    state.training = data;
    renderTrainingState();
    refreshRunsAndExports();
    if (data.status === "completed") {
      toast(`Huấn luyện hoàn tất! mAP50 cao nhất: ${(data.best_map50 * 100).toFixed(1)}%`, "success", 6000);
    } else if (data.status === "error") {
      toast(`Lỗi huấn luyện: ${data.error}`, "error", 7000);
    }
  } else if (type === "export_status") {
    showGlobalProgress(data.message, data.progress);
    $("exportStatusText").textContent = data.message;
  } else if (type === "export_done") {
    showGlobalProgress("Hoàn tất xuất mô hình!", 100);
    setTimeout(() => $("globalProgressBar").classList.add("hidden"), 2000);
    $("exportStatusText").textContent = "Hoàn tất xuất!";
    if (data.artifacts) {
      state.artifacts = data.artifacts;
      renderArtifactsTable();
    }
    const okCount = (data.results || []).filter((r) => r.ok).length;
    const failCount = (data.results || []).length - okCount;
    toast(`Xuất mô hình xong: ${okCount} thành công${failCount ? `, ${failCount} thất bại (xem Terminal)` : ""}.`, failCount ? "warn" : "success", 5500);
  } else if (type === "error") {
    toast(data.message || "Có lỗi xảy ra", "error", 6000);
  }
}

function showGlobalProgress(text, pct) {
  $("globalProgressBar").classList.remove("hidden");
  $("globalProgressText").textContent = text || "Đang xử lý...";
  const clamped = Math.max(0, Math.min(100, Math.round(pct || 0)));
  $("globalProgressPct").textContent = `${clamped}%`;
  $("globalProgressFill").style.width = `${clamped}%`;
}

function appendConsole(entry) {
  if (!entry || !entry.message) return;
  const body = $("consoleBody");
  const line = document.createElement("div");
  line.className = `log-line log-${entry.level || "info"}`;
  line.innerHTML = `<span class="log-time">[${entry.time || nowTime()}]</span><span class="log-src">[${entry.source || "sys"}]</span>${escapeHtml(entry.message)}`;
  body.appendChild(line);
  while (body.childNodes.length > 350) body.removeChild(body.firstChild);
  body.scrollTop = body.scrollHeight;
  $("consoleLastLine").textContent = `[${entry.source || "sys"}] ${entry.message}`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function renderTelemetry(t) {
  if (!t) return;
  state.telemetry = t;
  const osType = (t.os_type || "").toLowerCase();
  const osBadge = $("teleOs");
  if (osBadge) {
    osBadge.textContent = osType === "linux" ? "🐧 Linux" : osType === "macos" ? "🍎 macOS" : "🪟 Win";
    osBadge.title = t.os || osType;
  }
  const folderInput = $("inputLocalFolder");
  if (folderInput && !folderInput.value) {
    folderInput.placeholder = osType === "linux" || osType === "macos"
      ? "VD: /home/user/datasets/shapes_dataset"
      : "VD: C:\\Users\\ADMIN\\Pictures\\shapes_dataset";
  }
  $("teleCpu").textContent = `CPU ${Math.round(t.cpu_percent || 0)}%`;
  $("teleRam").textContent = `RAM ${t.ram_used_gb}/${t.ram_total_gb}GB`;
  if (t.gpu) {
    const cudaTag = t.torch_cuda_version ? ` · CUDA ${t.torch_cuda_version}` : "";
    $("teleGpu").textContent = `${t.gpu.name.replace("NVIDIA GeForce ", "")}${cudaTag} · ${Math.round(t.gpu.vram_used_mb)}/${Math.round(t.gpu.vram_total_mb)}MB · ${Math.round(t.gpu.temp_c)}°C`;
  } else {
    $("teleGpu").textContent = "GPU: CPU Mode";
  }
  const pill = $("cudaAdvisoryPill");
  if (t.cuda_advisory) {
    pill.textContent = "⚠️ GPU RTX (PyTorch CPU)";
    pill.classList.remove("hidden");
  } else if (t.cuda_version_mismatch) {
    pill.textContent = `ℹ️ PyTorch CUDA ${t.torch_cuda_version} (Đích ${t.target_cuda || "11.8"})`;
    pill.classList.remove("hidden");
  } else {
    pill.classList.add("hidden");
  }
}

// ======================================================================
// Bootstrap & Navigation
// ======================================================================

async function initApp() {
  loadLayoutPrefs();
  bindStaticEvents();
  applyLayoutUi(false);
  connectWebSocket();

  try {
    const data = await api("/api/bootstrap");
    state.bootstrap = data;
    state.runs = data.runs || [];
    state.artifacts = data.artifacts || [];
    state.training = data.training_state;

    renderTelemetry(data.telemetry);
    populateProjectSelect(data.projects || []);
    applyPreset("small_shapes_default", false);
    renderPlatformCards();
    renderFamilyCards();
    renderPresetCards();
    renderModelPicker();
    renderHyperparams();
    renderAutoExportChips();
    renderExportFormatCards();
    renderTrainingState();
    renderRunsTable();
    renderWeightSelectors();
    renderArtifactsTable();

    if (state.activeProjectId) {
      await loadActiveProjectData();
    }
  } catch (err) {
    toast(`Không thể tải dữ liệu khởi tạo: ${err.message}`, "error", 7000);
  }
}

function switchTab(tabId) {
  document.querySelectorAll(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.tab === tabId));
  document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${tabId}`));
  const content = document.querySelector(".content");
  if (content) {
    content.classList.toggle("focus-mode", tabId === "dataset" && state.ui.focusLabelMode);
  }
  if (tabId === "train") {
    drawAllCharts();
  } else if (tabId === "dataset") {
    scheduleCanvasResize();
  }
}

// ======================================================================
// Layout Toggles (Collapsible Sidebar, Gallery Panel, Focus Label Mode)
// ======================================================================

const LAYOUT_PREF_KEY = "derich_studio_layout_v1";

function loadLayoutPrefs() {
  try {
    const raw = localStorage.getItem(LAYOUT_PREF_KEY);
    if (!raw) return;
    const saved = JSON.parse(raw);
    if (typeof saved.sidebarCollapsed === "boolean") state.ui.sidebarCollapsed = saved.sidebarCollapsed;
    if (typeof saved.galleryCollapsed === "boolean") state.ui.galleryCollapsed = saved.galleryCollapsed;
    if (typeof saved.galleryWide === "boolean") state.ui.galleryWide = saved.galleryWide;
    if (typeof saved.setupToolsCollapsed === "boolean") state.ui.setupToolsCollapsed = saved.setupToolsCollapsed;
  } catch {
    // Ignore storage parse issues
  }
}

function saveLayoutPrefs() {
  try {
    localStorage.setItem(LAYOUT_PREF_KEY, JSON.stringify({
      sidebarCollapsed: state.ui.sidebarCollapsed,
      galleryCollapsed: state.ui.galleryCollapsed,
      galleryWide: state.ui.galleryWide,
      setupToolsCollapsed: state.ui.setupToolsCollapsed,
    }));
  } catch {
    // Ignore storage quota issues
  }
}

let canvasResizeRaf = 0;
function scheduleCanvasResize(withTransitionDelay = false) {
  if (canvasResizeRaf) cancelAnimationFrame(canvasResizeRaf);
  canvasResizeRaf = requestAnimationFrame(() => {
    canvasResizeRaf = 0;
    resizeEditorCanvas();
    if ($("tab-train")?.classList.contains("active")) drawAllCharts();
  });
  if (withTransitionDelay) {
    setTimeout(() => {
      resizeEditorCanvas();
      if ($("tab-train")?.classList.contains("active")) drawAllCharts();
    }, 230);
  }
}

function applyLayoutUi(persist = true) {
  const { sidebarCollapsed, galleryCollapsed, galleryWide, setupToolsCollapsed, focusLabelMode } = state.ui;

  const wsEl = $("workspace");
  if (wsEl) wsEl.classList.toggle("sidebar-collapsed", sidebarCollapsed);

  const topToggle = $("btnToggleSidebar");
  if (topToggle) {
    topToggle.classList.toggle("collapsed", sidebarCollapsed);
    topToggle.setAttribute("aria-expanded", String(!sidebarCollapsed));
    topToggle.title = sidebarCollapsed
      ? "Hiện thanh menu bên trái (Ctrl+B)"
      : "Thu gọn thanh menu bên trái để tăng diện tích (Ctrl+B)";
  }

  const edgeChevron = $("sidebarEdgeChevron");
  const edgeBtn = $("btnSidebarEdgeToggle");
  if (edgeChevron) edgeChevron.textContent = sidebarCollapsed ? "▶" : "◀";
  if (edgeBtn) {
    edgeBtn.title = sidebarCollapsed
      ? "Mở thanh menu bên trái (Ctrl+B)"
      : "Thu gọn thanh menu bên trái để tăng diện tích (Ctrl+B)";
  }

  const labelerWs = $("labelerWorkspace");
  if (labelerWs) {
    labelerWs.classList.toggle("gallery-collapsed", galleryCollapsed);
    labelerWs.classList.toggle("gallery-wide", !galleryCollapsed && galleryWide);
  }

  const btnGal = $("btnToggleGallery");
  if (btnGal) {
    btnGal.textContent = galleryCollapsed ? "◧ Hiện khung ảnh" : "◧ Ẩn khung ảnh";
    btnGal.classList.toggle("btn-primary", galleryCollapsed);
    btnGal.classList.toggle("btn-secondary", !galleryCollapsed);
  }
  const railGal = $("btnRailToggleGallery");
  const railGalIcon = $("railGalleryIcon");
  if (railGal) {
    railGal.classList.toggle("active", !galleryCollapsed);
    railGal.title = galleryCollapsed ? "Hiện khung ảnh nhỏ bên trái (G)" : "Ẩn khung ảnh nhỏ bên trái (G)";
  }
  if (railGalIcon) railGalIcon.textContent = galleryCollapsed ? "▶" : "◀";

  const btnWide = $("btnToggleGalleryWide");
  const railWide = $("btnRailWideGallery");
  [btnWide, railWide].forEach((b) => {
    if (!b) return;
    b.disabled = galleryCollapsed;
    b.classList.toggle("active", !galleryCollapsed && galleryWide);
    b.title = galleryWide ? "Thu gọn cột khung ảnh về 2 cột" : "Mở rộng cột khung ảnh (3 cột)";
  });

  const toolsRegion = $("datasetToolsRegion");
  const tabDataset = $("tab-dataset");
  const effectiveToolsHidden = setupToolsCollapsed || focusLabelMode;
  if (toolsRegion) toolsRegion.classList.toggle("collapsed", effectiveToolsHidden);
  if (tabDataset) {
    tabDataset.classList.toggle("tools-collapsed", effectiveToolsHidden);
    tabDataset.classList.toggle("labeler-focus", focusLabelMode);
  }
  const content = document.querySelector(".content");
  if (content) {
    content.classList.toggle("focus-mode", focusLabelMode && tabDataset?.classList.contains("active"));
  }

  const btnSetup = $("btnToggleSetupTools");
  if (btnSetup) {
    btnSetup.textContent = effectiveToolsHidden ? "🧰 Hiện công cụ trên" : "🧰 Ẩn công cụ trên";
  }

  const btnFocus = $("btnToggleFocusLabel");
  const btnHeaderFocus = $("btnHeaderFocusLabel");
  if (btnFocus) {
    btnFocus.textContent = focusLabelMode ? "⤡ Thoát mở rộng" : "⛶ Mở rộng Label";
    btnFocus.classList.toggle("btn-accent", focusLabelMode);
    btnFocus.classList.toggle("btn-primary", !focusLabelMode);
  }
  if (btnHeaderFocus) {
    btnHeaderFocus.textContent = focusLabelMode ? "⤡ Thoát mở rộng Label" : "⛶ Mở rộng vùng Label";
  }

  const railSidebar = $("btnRailToggleSidebar");
  if (railSidebar) {
    railSidebar.classList.toggle("active", !sidebarCollapsed);
    railSidebar.title = sidebarCollapsed ? "Hiện thanh menu bên trái (Ctrl+B)" : "Ẩn thanh menu bên trái (Ctrl+B)";
  }
  const railSetup = $("btnRailToggleSetup");
  if (railSetup) {
    railSetup.classList.toggle("active", !effectiveToolsHidden);
    railSetup.title = effectiveToolsHidden ? "Hiện khu vực nhập & tiền xử lý phía trên" : "Ẩn khu vực nhập & tiền xử lý phía trên";
  }
  const railFocus = $("btnRailToggleFocus");
  if (railFocus) {
    railFocus.classList.toggle("active", focusLabelMode);
    railFocus.title = focusLabelMode ? "Thoát chế độ mở rộng tối đa (F / Esc)" : "Mở rộng tối đa diện tích Label (F)";
  }

  if (persist) saveLayoutPrefs();
  scheduleCanvasResize(true);
}

function toggleSidebar(forceState) {
  state.ui.sidebarCollapsed = typeof forceState === "boolean" ? forceState : !state.ui.sidebarCollapsed;
  applyLayoutUi(true);
}

function toggleGallery(forceState) {
  state.ui.galleryCollapsed = typeof forceState === "boolean" ? forceState : !state.ui.galleryCollapsed;
  applyLayoutUi(true);
}

function toggleGalleryWide() {
  if (state.ui.galleryCollapsed) state.ui.galleryCollapsed = false;
  state.ui.galleryWide = !state.ui.galleryWide;
  applyLayoutUi(true);
}

function toggleSetupTools() {
  if (state.ui.focusLabelMode) {
    state.ui.focusLabelMode = false;
    state.ui.setupToolsCollapsed = false;
  } else {
    state.ui.setupToolsCollapsed = !state.ui.setupToolsCollapsed;
  }
  applyLayoutUi(true);
}

function toggleFocusLabelMode(forceState) {
  const next = typeof forceState === "boolean" ? forceState : !state.ui.focusLabelMode;
  if (next === state.ui.focusLabelMode) return;
  if (next) {
    state.ui.preFocusSnapshot = {
      sidebarCollapsed: state.ui.sidebarCollapsed,
      setupToolsCollapsed: state.ui.setupToolsCollapsed,
    };
    state.ui.focusLabelMode = true;
    state.ui.sidebarCollapsed = true;
    state.ui.setupToolsCollapsed = true;
  } else {
    state.ui.focusLabelMode = false;
    if (state.ui.preFocusSnapshot) {
      state.ui.sidebarCollapsed = state.ui.preFocusSnapshot.sidebarCollapsed;
      state.ui.setupToolsCollapsed = state.ui.preFocusSnapshot.setupToolsCollapsed;
      state.ui.preFocusSnapshot = null;
    } else {
      state.ui.setupToolsCollapsed = false;
    }
  }
  applyLayoutUi(false);
}

function updateSidebarSummary() {
  $("sumModel").textContent = state.params.model || "yolo26n.pt";
  const presetObj = state.bootstrap?.training_presets?.[state.activePresetId];
  $("sumPreset").textContent = presetObj ? presetObj.badge : "Tùy chỉnh";
  $("sumImgsz").textContent = `${state.params.imgsz || 640} px`;
  $("sumFreeze").textContent = `${state.params.freeze ?? 10} tầng`;
  if (state.health) {
    $("sumLabeled").textContent = `${state.health.labeled_images} / ${state.health.total_images}`;
  }
}

// ======================================================================
// Projects, Dataset Health & Multi-Source Ingestion
// ======================================================================

function populateProjectSelect(projects) {
  const sel = $("projectSelect");
  sel.innerHTML = "";
  projects.forEach((p) => {
    const opt = document.createElement("option");
    opt.value = p.id;
    opt.textContent = `${p.name} (${p.labeled_count}/${p.image_count} ảnh)`;
    sel.appendChild(opt);
  });
  if (!state.activeProjectId || !projects.some((p) => p.id === state.activeProjectId)) {
    state.activeProjectId = projects[0]?.id || null;
  }
  if (state.activeProjectId) sel.value = state.activeProjectId;
}

async function loadActiveProjectData(preferredFilename = null) {
  if (!state.activeProjectId) return;
  const seq = ++state.projectLoadSeq;
  const targetFilename = preferredFilename !== null ? preferredFilename : state.selectedImageFilename;
  const prevIdx = state.selectedImageIdx;

  try {
    const [projects, detail] = await Promise.all([
      api("/api/projects"),
      api(`/api/projects/${state.activeProjectId}/images`),
    ]);
    if (seq !== state.projectLoadSeq) return;

    populateProjectSelect(projects);
    state.images = detail.images || [];
    state.health = detail.health;

    // Prune checked thumbnails that no longer exist
    const existingNames = new Set(state.images.map((im) => im.filename));
    Array.from(state.selectedThumbFilenames).forEach((fname) => {
      if (!existingNames.has(fname)) state.selectedThumbFilenames.delete(fname);
    });

    // Resolve next active image index while keeping user's current image if still present
    let nextIdx = -1;
    if (targetFilename) {
      nextIdx = state.images.findIndex((im) => im.filename === targetFilename);
    }
    if (nextIdx < 0 && state.images.length > 0) {
      if (prevIdx >= 0) {
        nextIdx = Math.min(prevIdx, state.images.length - 1);
      } else {
        nextIdx = 0;
      }
    }
    state.selectedImageIdx = nextIdx;
    state.selectedImageFilename = nextIdx >= 0 ? state.images[nextIdx].filename : null;

    renderClassManager(detail.project.classes || []);
    renderHealthCard(detail.health);
    renderImageGallery();
    updateSidebarSummary();

    if (nextIdx >= 0) {
      selectImageInEditor(nextIdx, { syncGalleryOnly: true });
    } else {
      resetEditorCanvas();
    }
  } catch (err) {
    if (seq === state.projectLoadSeq) {
      toast(`Lỗi tải dữ liệu dự án: ${err.message}`, "error");
    }
  }
}

function renderClassManager(classes) {
  const list = $("classChipList");
  list.innerHTML = "";
  const alSel = $("alTargetClass");
  const edSel = $("editorActiveClass");
  const prevEd = edSel.value;
  alSel.innerHTML = "";
  edSel.innerHTML = "";

  classes.forEach((cname, idx) => {
    const col = classColor(idx);
    const chip = document.createElement("span");
    chip.className = "class-chip";
    chip.innerHTML = `<span class="class-swatch" style="background:${col}"></span>${idx}: ${escapeHtml(cname)} <button data-idx="${idx}" title="Xóa lớp">×</button>`;
    chip.querySelector("button").onclick = async () => {
      if (classes.length <= 1) return toast("Phải giữ ít nhất 1 lớp vật thể.", "warn");
      const next = classes.filter((_, i) => i !== idx);
      await api(`/api/projects/${state.activeProjectId}/classes`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ classes: next }),
      });
      await loadActiveProjectData();
    };
    list.appendChild(chip);

    [alSel, edSel].forEach((sel) => {
      const o = document.createElement("option");
      o.value = String(idx);
      o.textContent = `${idx}: ${cname}`;
      sel.appendChild(o);
    });
  });
  if (prevEd && Number(prevEd) < classes.length) edSel.value = prevEd;
}

function renderHealthCard(h) {
  if (!h) return;
  $("statTotalImgs").textContent = h.total_images;
  $("statLabeledImgs").textContent = h.labeled_images;
  $("statUnlabeledImgs").textContent = h.unlabeled_images;
  $("statTotalBoxes").textContent = h.total_boxes;
  $("healthRegimeBadge").textContent = h.regime_label;

  const bars = $("classDistBars");
  bars.innerHTML = "";
  const maxCnt = Math.max(1, ...(h.per_class || []).map((c) => c.count));
  (h.per_class || []).forEach((c) => {
    const pct = Math.round((c.count / maxCnt) * 100);
    const row = document.createElement("div");
    row.className = "dist-row";
    row.innerHTML = `
      <span class="dist-name">${escapeHtml(c.name)}</span>
      <div class="dist-track"><div class="dist-fill" style="width:${pct}%;background:${classColor(c.id)}"></div></div>
      <span class="dist-cnt">${c.count}</span>
    `;
    bars.appendChild(row);
  });

  const alerts = $("healthAlerts");
  alerts.innerHTML = "";
  (h.warnings || []).forEach((w) => {
    const d = document.createElement("div");
    d.className = "alert-item alert-warn";
    d.textContent = `⚠️ ${w}`;
    alerts.appendChild(d);
  });
  (h.recommendations || []).forEach((r) => {
    const d = document.createElement("div");
    d.className = "alert-item alert-info";
    d.textContent = `✅ ${r}`;
    alerts.appendChild(d);
  });
}

// ======================================================================
// Image Gallery & Interactive Canvas Bounding Box Labeler
// ======================================================================

function getFilteredImages() {
  const filter = $("galleryFilter")?.value || "all";
  return state.images
    .map((im, idx) => ({ im, idx }))
    .filter(({ im }) => filter === "all" || (filter === "labeled" ? im.labeled : !im.labeled));
}

function updateGallerySelectionUi() {
  const count = state.selectedThumbFilenames.size;
  const delBtn = $("btnDeleteSelectedImgs");
  const countEl = $("selectedImgsCount");
  if (countEl) countEl.textContent = String(count);
  if (delBtn) delBtn.classList.toggle("hidden", count === 0);

  const chkAll = $("chkSelectAllGallery");
  if (chkAll) {
    const filtered = getFilteredImages();
    const selectedVisible = filtered.filter(({ im }) => state.selectedThumbFilenames.has(im.filename)).length;
    chkAll.disabled = filtered.length === 0;
    chkAll.checked = filtered.length > 0 && selectedVisible === filtered.length;
    chkAll.indeterminate = selectedVisible > 0 && selectedVisible < filtered.length;
  }
}

async function deleteSingleImage(filename, cardEl = null) {
  if (!state.activeProjectId || !filename) return;
  if (cardEl) cardEl.classList.add("deleting");

  // Choose neighbor image if we're deleting the currently active one
  let preferredNext = state.selectedImageFilename;
  if (filename === state.selectedImageFilename) {
    const curIdx = state.images.findIndex((im) => im.filename === filename);
    if (curIdx >= 0) {
      const nextItem = state.images[curIdx + 1] || state.images[curIdx - 1] || null;
      preferredNext = nextItem ? nextItem.filename : "";
    }
  }

  try {
    await api(`/api/projects/${state.activeProjectId}/images/${encodeURIComponent(filename)}`, { method: "DELETE" });
    state.selectedThumbFilenames.delete(filename);
    toast(`Đã xóa ảnh ${filename}`, "info", 2200);
    await loadActiveProjectData(preferredNext);
  } catch (err) {
    if (cardEl) cardEl.classList.remove("deleting");
    toast(`Không thể xóa ảnh: ${err.message}`, "error");
  }
}

async function deleteSelectedImages() {
  const names = Array.from(state.selectedThumbFilenames);
  if (!names.length || !state.activeProjectId) return;
  if (!confirm(`Xóa ${names.length} ảnh đã chọn (kèm nhãn) khỏi dự án hiện tại?`)) return;

  const btn = $("btnDeleteSelectedImgs");
  if (btn) btn.disabled = true;
  try {
    const res = await api(`/api/projects/${state.activeProjectId}/images/batch-delete`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filenames: names }),
    });
    state.selectedThumbFilenames.clear();
    toast(`Đã xóa ${res.deleted ?? names.length} ảnh đã chọn.`, "info", 2600);
    await loadActiveProjectData();
  } catch (err) {
    toast(`Lỗi xóa nhiều ảnh: ${err.message}`, "error");
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function deleteCurrentEditorImage() {
  if (state.selectedImageIdx < 0 || !state.images[state.selectedImageIdx]) return;
  const item = state.images[state.selectedImageIdx];
  const btn = $("btnDeleteCurrentImg");
  if (btn) btn.disabled = true;
  try {
    await deleteSingleImage(item.filename);
  } finally {
    if (btn) btn.disabled = state.selectedImageIdx < 0;
  }
}

function renderImageGallery() {
  const grid = $("imageGallery");
  const scroller = $("galleryScroll") || grid;
  const prevScroll = scroller.scrollTop;
  grid.innerHTML = "";
  const filtered = getFilteredImages();

  $("galleryCount").textContent = `${filtered.length} / ${state.images.length} ảnh`;

  if (filtered.length === 0) {
    grid.innerHTML = `<div class="muted-text" style="grid-column:1/-1;padding:18px;text-align:center">Chưa có ảnh phù hợp bộ lọc.</div>`;
    updateGallerySelectionUi();
    return;
  }

  filtered.forEach(({ im, idx }) => {
    const isActive = idx === state.selectedImageIdx;
    const isChecked = state.selectedThumbFilenames.has(im.filename);
    const card = document.createElement("div");
    card.className = `thumb-card${isActive ? " active" : ""}${isChecked ? " checked" : ""}`;
    card.dataset.idx = String(idx);
    card.dataset.filename = im.filename;
    card.innerHTML = `
      <div class="thumb-img-wrap">
        <img src="${im.url}?t=${im.box_count}" decoding="async" alt="${escapeHtml(im.filename)}" />
        <label class="thumb-check-wrap" title="Chọn ảnh để xóa hàng loạt">
          <input type="checkbox" class="thumb-select" ${isChecked ? "checked" : ""} aria-label="Chọn ${escapeHtml(im.filename)}" />
        </label>
        <button type="button" class="thumb-del" title="Xóa ảnh này" aria-label="Xóa ảnh ${escapeHtml(im.filename)}">
          <svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M2.5 4.2h11M6 4.2V2.8a.8.8 0 0 1 .8-.8h2.4a.8.8 0 0 1 .8.8v1.4M12.2 4.2l-.6 8.5a1.2 1.2 0 0 1-1.2 1.1H5.6a1.2 1.2 0 0 1-1.2-1.1l-.6-8.5M6.6 7v4M9.4 7v4"/>
          </svg>
        </button>
        <span class="thumb-badge ${im.labeled ? "ok" : "no"}">${im.labeled ? `${im.box_count} box` : "Chưa nhãn"}</span>
      </div>
      <div class="thumb-meta">
        <span class="thumb-name" title="${escapeHtml(im.filename)}">${escapeHtml(im.filename)}</span>
        <span class="thumb-dot ${im.labeled ? "ok" : "no"}" title="${im.labeled ? im.box_count + " boxes" : "Chưa nhãn"}"></span>
      </div>
    `;

    const checkWrap = card.querySelector(".thumb-check-wrap");
    const checkInput = card.querySelector(".thumb-select");
    checkWrap.onclick = (e) => e.stopPropagation();
    checkInput.onchange = (e) => {
      e.stopPropagation();
      if (checkInput.checked) state.selectedThumbFilenames.add(im.filename);
      else state.selectedThumbFilenames.delete(im.filename);
      card.classList.toggle("checked", checkInput.checked);
      updateGallerySelectionUi();
    };

    card.querySelector(".thumb-del").onclick = async (e) => {
      e.stopPropagation();
      e.preventDefault();
      await deleteSingleImage(im.filename, card);
    };

    card.onclick = (e) => {
      if (e.target.closest(".thumb-del") || e.target.closest(".thumb-check-wrap")) return;
      selectImageInEditor(idx);
    };

    grid.appendChild(card);
  });

  scroller.scrollTop = prevScroll;
  updateGallerySelectionUi();
}

function resetEditorCanvas() {
  state.selectedImageIdx = -1;
  state.selectedImageFilename = null;
  state.editorBoxes = [];
  state.selectedBoxIdx = -1;
  state.editorImgObj = null;
  state.editorImgUrl = null;
  $("labelCanvas").style.display = "none";
  $("canvasEmptyHint").style.display = "block";
  $("editorFilename").textContent = "Chọn một ảnh bên trái để xem/chỉnh nhãn";
  $("editorBoxList").innerHTML = "";
  const delBtn = $("btnDeleteCurrentImg");
  if (delBtn) delBtn.disabled = true;
}

function setCanvasZoom(nextZoom) {
  state.canvasZoom = Math.max(0.5, Math.min(3.0, Math.round(nextZoom * 100) / 100));
  const resetBtn = $("btnZoomReset");
  if (resetBtn) {
    resetBtn.textContent = Math.abs(state.canvasZoom - 1.0) < 0.01 ? "Fit" : `${Math.round(state.canvasZoom * 100)}%`;
  }
  resizeEditorCanvas();
}

function resizeEditorCanvas() {
  const img = state.editorImgObj;
  const canvas = $("labelCanvas");
  const stage = $("canvasStage");
  if (!img || !canvas || !stage) return;
  const availW = stage.clientWidth - 20;
  const availH = stage.clientHeight - 20;
  if (availW < 40 || availH < 40) return;

  const baseScale = Math.min(availW / img.width, availH / img.height, 2.5);
  const scale = Math.max(0.1, baseScale * (state.canvasZoom || 1.0));
  const targetW = Math.max(1, Math.round(img.width * scale));
  const targetH = Math.max(1, Math.round(img.height * scale));
  if (canvas.width !== targetW || canvas.height !== targetH) {
    canvas.width = targetW;
    canvas.height = targetH;
  }
  redrawEditorCanvas();
}

function stepEditorImage(delta) {
  if (!state.images.length) return;
  const filtered = getFilteredImages();
  if (filtered.length > 0) {
    const pos = filtered.findIndex((x) => x.idx === state.selectedImageIdx);
    if (pos >= 0) {
      const nextPos = pos + delta;
      if (nextPos >= 0 && nextPos < filtered.length) {
        selectImageInEditor(filtered[nextPos].idx);
      }
      return;
    }
    selectImageInEditor(filtered[delta > 0 ? 0 : filtered.length - 1].idx);
    return;
  }
  const nextIdx = state.selectedImageIdx + delta;
  if (nextIdx >= 0 && nextIdx < state.images.length) {
    selectImageInEditor(nextIdx);
  }
}

function selectImageInEditor(idx, opts = {}) {
  if (idx < 0 || idx >= state.images.length) return resetEditorCanvas();
  const prevFilename = state.selectedImageFilename;
  state.selectedImageIdx = idx;
  const item = state.images[idx];
  const isSameImage = prevFilename === item.filename;
  state.selectedImageFilename = item.filename;
  state.editorBoxes = JSON.parse(JSON.stringify(item.boxes || []));
  if (!isSameImage) {
    state.selectedBoxIdx = state.editorBoxes.length ? 0 : -1;
  } else if (state.selectedBoxIdx >= state.editorBoxes.length) {
    state.selectedBoxIdx = state.editorBoxes.length - 1;
  } else if (state.selectedBoxIdx < 0 && state.editorBoxes.length > 0) {
    state.selectedBoxIdx = 0;
  }

  $("editorFilename").textContent = `${item.filename} (${idx + 1}/${state.images.length}) · ${item.width}×${item.height}`;
  const delBtn = $("btnDeleteCurrentImg");
  if (delBtn) delBtn.disabled = false;

  // Sync active highlight in gallery without wiping DOM or jumping scroll
  document.querySelectorAll("#imageGallery .thumb-card").forEach((el) => {
    const isAct = Number(el.dataset.idx) === idx;
    el.classList.toggle("active", isAct);
    if (isAct && !opts.syncGalleryOnly) {
      el.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  });

  if (state.editorImgObj && state.editorImgUrl === item.url) {
    $("labelCanvas").style.display = "block";
    $("canvasEmptyHint").style.display = "none";
    resizeEditorCanvas();
    renderEditorBoxList();
    return;
  }

  const targetFilename = item.filename;
  const img = new Image();
  img.onload = () => {
    if (state.selectedImageFilename !== targetFilename) return;
    state.editorImgObj = img;
    state.editorImgUrl = item.url;
    $("labelCanvas").style.display = "block";
    $("canvasEmptyHint").style.display = "none";
    resizeEditorCanvas();
    renderEditorBoxList();
  };
  img.src = item.url;
}

function getCurrentClasses() {
  const sel = $("editorActiveClass");
  return Array.from(sel.options).map((o) => o.textContent.replace(/^\d+:\s*/, ""));
}

function redrawEditorCanvas() {
  const canvas = $("labelCanvas");
  const ctx = canvas.getContext("2d");
  const img = state.editorImgObj;
  if (!img) return;
  const cw = canvas.width;
  const ch = canvas.height;
  ctx.clearRect(0, 0, cw, ch);
  ctx.drawImage(img, 0, 0, cw, ch);

  const classes = getCurrentClasses();
  state.editorBoxes.forEach((b, i) => {
    const col = classColor(b.class_id);
    const x = (b.cx - b.w / 2) * cw;
    const y = (b.cy - b.h / 2) * ch;
    const w = b.w * cw;
    const h = b.h * ch;
    const isSel = i === state.selectedBoxIdx;

    ctx.lineWidth = isSel ? 3 : 2;
    ctx.strokeStyle = col;
    ctx.fillStyle = isSel ? `${col}33` : `${col}18`;
    ctx.fillRect(x, y, w, h);
    ctx.strokeRect(x, y, w, h);

    const label = `${classes[b.class_id] || b.class_id}`;
    ctx.font = "600 12px Segoe UI, sans-serif";
    const tw = ctx.measureText(label).width + 10;
    const ly = Math.max(18, y);
    ctx.fillStyle = col;
    ctx.fillRect(x, ly - 18, tw, 18);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, x + 5, ly - 5);
  });

  if (state.drawing && state.drawStart && state.drawCurr) {
    const x = Math.min(state.drawStart.x, state.drawCurr.x);
    const y = Math.min(state.drawStart.y, state.drawCurr.y);
    const w = Math.abs(state.drawCurr.x - state.drawStart.x);
    const h = Math.abs(state.drawCurr.y - state.drawStart.y);
    const cid = Number($("editorActiveClass").value || 0);
    const col = classColor(cid);
    ctx.setLineDash([5, 4]);
    ctx.lineWidth = 2;
    ctx.strokeStyle = col;
    ctx.fillStyle = `${col}28`;
    ctx.fillRect(x, y, w, h);
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);
  }
}

function renderEditorBoxList() {
  const wrap = $("editorBoxList");
  wrap.innerHTML = "";
  const classes = getCurrentClasses();
  if (!state.editorBoxes.length) {
    wrap.innerHTML = `<span class="muted-text">Chưa có bounding box nào trên ảnh này.</span>`;
    return;
  }
  state.editorBoxes.forEach((b, idx) => {
    const chip = document.createElement("span");
    chip.className = `box-chip ${idx === state.selectedBoxIdx ? "selected" : ""}`;
    chip.innerHTML = `<span class="class-swatch" style="background:${classColor(b.class_id)}"></span>#${idx + 1} ${escapeHtml(classes[b.class_id] || String(b.class_id))}`;
    chip.onclick = () => {
      state.selectedBoxIdx = idx;
      $("editorActiveClass").value = String(b.class_id);
      redrawEditorCanvas();
      renderEditorBoxList();
    };
    wrap.appendChild(chip);
  });
}

async function saveCurrentLabels(silent = false) {
  if (state.selectedImageIdx < 0) return;
  const item = state.images[state.selectedImageIdx];
  try {
    await api(`/api/projects/${state.activeProjectId}/labels/${encodeURIComponent(item.filename)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ boxes: state.editorBoxes }),
    });
    item.boxes = JSON.parse(JSON.stringify(state.editorBoxes));
    item.box_count = item.boxes.length;
    item.labeled = item.boxes.length > 0;
    if (!silent) toast(`Đã lưu ${item.box_count} nhãn cho ${item.filename}`, "success", 2200);
  } catch (err) {
    toast(`Lỗi lưu nhãn: ${err.message}`, "error");
  }
}

function bindCanvasEvents() {
  const canvas = $("labelCanvas");
  const getPos = (e) => {
    const rect = canvas.getBoundingClientRect();
    return {
      x: ((e.clientX - rect.left) / rect.width) * canvas.width,
      y: ((e.clientY - rect.top) / rect.height) * canvas.height,
    };
  };

  canvas.addEventListener("mousedown", (e) => {
    if (!state.editorImgObj) return;
    const pos = getPos(e);
    // Check if user clicked inside an existing box (with Shift held -> select, else draw new)
    if (e.button === 0) {
      state.drawing = true;
      state.drawStart = pos;
      state.drawCurr = pos;
    }
  });

  canvas.addEventListener("mousemove", (e) => {
    if (!state.drawing) return;
    state.drawCurr = getPos(e);
    redrawEditorCanvas();
  });

  window.addEventListener("mouseup", () => {
    if (!state.drawing) return;
    state.drawing = false;
    const cw = canvas.width;
    const ch = canvas.height;
    const x1 = Math.min(state.drawStart.x, state.drawCurr.x);
    const y1 = Math.min(state.drawStart.y, state.drawCurr.y);
    const x2 = Math.max(state.drawStart.x, state.drawCurr.x);
    const y2 = Math.max(state.drawStart.y, state.drawCurr.y);
    const bw = x2 - x1;
    const bh = y2 - y1;

    if (bw >= 8 && bh >= 8) {
      const cid = Number($("editorActiveClass").value || 0);
      state.editorBoxes.push({
        class_id: cid,
        cx: ((x1 + x2) / 2) / cw,
        cy: ((y1 + y2) / 2) / ch,
        w: bw / cw,
        h: bh / ch,
      });
      state.selectedBoxIdx = state.editorBoxes.length - 1;
      renderEditorBoxList();
      redrawEditorCanvas();
      saveCurrentLabels(true);
    } else {
      // Short click: select box under pointer
      const px = state.drawStart.x / cw;
      const py = state.drawStart.y / ch;
      const hit = state.editorBoxes.findIndex(
        (b) => px >= b.cx - b.w / 2 && px <= b.cx + b.w / 2 && py >= b.cy - b.h / 2 && py <= b.cy + b.h / 2
      );
      if (hit >= 0) {
        state.selectedBoxIdx = hit;
        $("editorActiveClass").value = String(state.editorBoxes[hit].class_id);
        renderEditorBoxList();
        redrawEditorCanvas();
      }
    }
  });

  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "b") {
      e.preventDefault();
      toggleSidebar();
      return;
    }
    if (e.key === "Escape") {
      if (!$("modalOverlay").classList.contains("hidden")) {
        closeModal();
        return;
      }
      if (state.ui.focusLabelMode) {
        toggleFocusLabelMode(false);
        return;
      }
    }
    if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) return;
    if (!$("tab-dataset").classList.contains("active")) return;
    if (e.key === "Delete" || e.key === "Backspace") {
      if (state.selectedBoxIdx >= 0) {
        state.editorBoxes.splice(state.selectedBoxIdx, 1);
        state.selectedBoxIdx = Math.min(state.selectedBoxIdx, state.editorBoxes.length - 1);
        redrawEditorCanvas();
        renderEditorBoxList();
        saveCurrentLabels(true);
      }
    } else if (e.key === "ArrowRight") {
      stepEditorImage(1);
    } else if (e.key === "ArrowLeft") {
      stepEditorImage(-1);
    } else if (!e.ctrlKey && !e.metaKey && !e.altKey && e.key.toLowerCase() === "g") {
      toggleGallery();
    } else if (!e.ctrlKey && !e.metaKey && !e.altKey && e.key.toLowerCase() === "f") {
      toggleFocusLabelMode();
    }
  });

  if (typeof ResizeObserver !== "undefined" && $("canvasStage")) {
    const ro = new ResizeObserver(() => scheduleCanvasResize(false));
    ro.observe($("canvasStage"));
  }
}

// ======================================================================
// Platform Advisor & Model Family Cards
// ======================================================================

function renderPlatformCards() {
  const grid = $("platformCardsGrid");
  grid.innerHTML = "";
  const profiles = (state.bootstrap?.platform_profiles || []).filter(
    (p) => state.activePlatformOs === "all" || p.os === state.activePlatformOs
  );

  profiles.forEach((p) => {
    const card = document.createElement("div");
    card.className = "platform-card";
    card.innerHTML = `
      <div>
        <div class="platform-top">
          <div>
            <div class="platform-os">${escapeHtml(p.os)}</div>
            <h4 class="platform-title">${escapeHtml(p.title)}</h4>
          </div>
          <span class="badge badge-accent">${escapeHtml(p.primary_model)}</span>
        </div>
        <p class="muted-text mt-8">${escapeHtml(p.why)}</p>
        <div class="platform-specs mt-8">
          <div><span>Model thay thế</span><strong>${escapeHtml(p.alternative_models.join(", "))}</strong></div>
          <div><span>Image Size khuyên dùng</span><strong>${p.recommended_imgsz} px</strong></div>
          <div><span>Định dạng xuất tối ưu</span><strong>${escapeHtml(p.recommended_formats.map((x) => x.toUpperCase()).join(" · "))}</strong></div>
          <div><span>Tốc độ ước tính</span><strong>${escapeHtml(p.expected_fps)}</strong></div>
        </div>
        <p class="muted-text mt-8"><strong>Định dạng:</strong> ${escapeHtml(p.format_guide)}</p>
        <ul class="platform-tips mt-4">${p.tips.map((t) => `<li>${escapeHtml(t)}</li>`).join("")}</ul>
      </div>
      <button class="btn btn-primary btn-block">⚡ Áp dụng cấu hình tối ưu cho ${escapeHtml(p.os)}</button>
    `;
    card.querySelector("button").onclick = () => {
      state.params.model = p.primary_model;
      state.params.imgsz = p.recommended_imgsz;
      const fam = p.primary_model.startsWith("yolo26")
        ? "yolo26"
        : p.primary_model.startsWith("yolo11")
          ? "yolo11"
          : p.primary_model.startsWith("yolov8")
            ? "yolov8"
            : "yolo12";
      state.activeModelFamily = fam;
      state.autoExportFormats = [...p.recommended_formats];
      state.selectedExportFormats = new Set(p.recommended_formats);
      $("autoExportPrecision").value = p.recommended_precision || "fp32";
      $("exportPrecision").value = p.recommended_precision || "fp32";
      $("exportImgsz").value = String(p.recommended_imgsz);

      renderModelPicker();
      renderHyperparams();
      renderAutoExportChips();
      renderExportFormatCards();
      updateSidebarSummary();
      toast(`Đã áp dụng cấu hình cho ${p.title}: ${p.primary_model} (${p.recommended_imgsz}px) + xuất ${p.recommended_formats.join(", ").toUpperCase()}!`, "success", 5000);
      switchTab("config");
    };
    grid.appendChild(card);
  });
}

function renderFamilyCards() {
  const grid = $("familyCardsGrid");
  grid.innerHTML = "";
  const fams = state.bootstrap?.model_families || {};
  Object.entries(fams).forEach(([key, info]) => {
    const card = document.createElement("div");
    card.className = "family-card";
    card.innerHTML = `
      <h4><span>${escapeHtml(info.name)}</span><span class="badge badge-accent">${escapeHtml(info.badge)}</span></h4>
      <ul>${info.highlights.map((h) => `<li>${escapeHtml(h)}</li>`).join("")}</ul>
      <div class="muted-text">${escapeHtml(info.note)}</div>
    `;
    grid.appendChild(card);
  });
}

// ======================================================================
// Presets, Model Picker & Hyperparameters with Impact Notes
// ======================================================================

function applyPreset(presetId, notify = true) {
  const preset = state.bootstrap?.training_presets?.[presetId];
  if (!preset) return;
  state.activePresetId = presetId;
  state.params = JSON.parse(JSON.stringify(preset.params));
  if (state.params.model?.startsWith("yolo26")) state.activeModelFamily = "yolo26";
  renderPresetCards();
  renderModelPicker();
  renderHyperparams();
  updateSidebarSummary();
  if (notify) toast(`Đã áp dụng preset: ${preset.name}`, "success");
}

function renderPresetCards() {
  const wrap = $("presetCards");
  wrap.innerHTML = "";
  const presets = state.bootstrap?.training_presets || {};
  Object.entries(presets).forEach(([id, p]) => {
    const el = document.createElement("div");
    el.className = `preset-card ${id === state.activePresetId ? "active" : ""}`;
    el.innerHTML = `
      <div class="preset-head">
        <strong>${escapeHtml(p.name)}</strong>
        <span class="badge ${id === "small_shapes_default" ? "badge-success" : "badge-accent"}">${escapeHtml(p.badge)}</span>
      </div>
      <p class="preset-desc">${escapeHtml(p.description)}</p>
    `;
    el.onclick = () => applyPreset(id, true);
    wrap.appendChild(el);
  });
}

function renderModelPicker() {
  document.querySelectorAll("#modelFamilyTabs .subtab").forEach((b) =>
    b.classList.toggle("active", b.dataset.family === state.activeModelFamily)
  );
  const grid = $("modelPickerGrid");
  grid.innerHTML = "";

  if (state.activeModelFamily === "custom") {
    const customs = state.bootstrap?.custom_weights || [];
    if (!customs.length) {
      grid.innerHTML = `<div class="muted-text" style="grid-column:1/-1;padding:12px">Chưa có trọng số tùy chỉnh nào. Hãy tải file <code>.pt</code> ở Tab 2 (Model theo Nền tảng).</div>`;
      return;
    }
    customs.forEach((cw) => {
      const card = document.createElement("div");
      card.className = `model-card ${state.params.model === cw.id ? "active" : ""}`;
      card.innerHTML = `
        <div class="model-card-head"><strong>${escapeHtml(cw.name)}</strong><span class="badge badge-accent">${cw.size_mb} MB</span></div>
        <div class="model-stats">Trọng số tùy chỉnh cục bộ (.pt)</div>
      `;
      card.onclick = () => {
        state.params.model = cw.id;
        renderModelPicker();
        updateSidebarSummary();
      };
      grid.appendChild(card);
    });
    return;
  }

  const models = (state.bootstrap?.model_catalog || []).filter((m) => m.family === state.activeModelFamily);
  models.forEach((m) => {
    const card = document.createElement("div");
    card.className = `model-card ${state.params.model === m.id ? "active" : ""}`;
    const riskCls = m.size === "n" ? "badge-success" : m.size === "s" ? "badge-accent" : "badge-warn";
    card.innerHTML = `
      <div class="model-card-head">
        <strong>${escapeHtml(m.label)}</strong>
        <span class="badge ${riskCls}">${escapeHtml(m.speed)}</span>
      </div>
      <div class="model-stats">${m.params_m}M params · ~${m.pt_mb} MB · Overfit: ${escapeHtml(m.overfit_risk)}</div>
      <div class="model-tags">${m.best_for.map((t) => `<span class="model-tag">${escapeHtml(t)}</span>`).join("")}</div>
    `;
    card.onclick = () => {
      state.params.model = m.id;
      renderModelPicker();
      updateSidebarSummary();
    };
    grid.appendChild(card);
  });
}

function renderHyperparams() {
  const wrap = $("hyperparamList");
  wrap.innerHTML = "";
  const docs = (state.bootstrap?.hyperparam_docs || []).filter(
    (d) => state.activeParamGroup === "all" || d.group === state.activeParamGroup
  );

  docs.forEach((doc) => {
    const val = state.params[doc.key] !== undefined ? state.params[doc.key] : doc.default;
    const card = document.createElement("div");
    card.className = "param-card";

    let ctrlHtml = "";
    if (doc.type === "bool") {
      ctrlHtml = `
        <label class="check-label">
          <input type="checkbox" data-key="${doc.key}" ${val ? "checked" : ""} />
          <span>${val ? "Đang BẬT (True)" : "Đang TẮT (False)"}</span>
        </label>
      `;
    } else if (doc.type === "select") {
      const opts = (doc.options || []).map((o) => {
        const lbl = doc.option_labels?.[String(o)] || String(o);
        return `<option value="${escapeHtml(String(o))}" ${String(val) === String(o) ? "selected" : ""}>${escapeHtml(lbl)}</option>`;
      }).join("");
      ctrlHtml = `<select class="input flex-1" data-key="${doc.key}">${opts}</select>`;
    } else {
      ctrlHtml = `
        <input type="range" data-range="${doc.key}" min="${doc.min}" max="${doc.max}" step="${doc.step}" value="${val}" />
        <input type="number" class="input input-sm" data-key="${doc.key}" min="${doc.min}" max="${doc.max}" step="${doc.step}" value="${val}" />
      `;
    }

    card.innerHTML = `
      <div class="param-top">
        <div>
          <div class="param-label">${escapeHtml(doc.label)}</div>
          <span class="param-key">${escapeHtml(doc.key)}</span>
        </div>
        <span class="badge badge-muted">Mặc định: ${escapeHtml(String(doc.default))}</span>
      </div>
      <div class="param-control-row">${ctrlHtml}</div>
      <div class="param-short">${escapeHtml(doc.short_note)}</div>
      <div class="param-notes">
        <div class="note-up"><strong>📈 ${escapeHtml(doc.impact_increase)}</strong></div>
        <div class="note-down">📉 ${escapeHtml(doc.impact_decrease)}</div>
        <div class="note-tip">💡 <strong>&lt;100 ảnh:</strong> ${escapeHtml(doc.small_data_tip)}</div>
      </div>
    `;

    const setVal = (raw) => {
      if (doc.type === "int") state.params[doc.key] = parseInt(raw, 10);
      else if (doc.type === "float") state.params[doc.key] = parseFloat(raw);
      else if (doc.type === "bool") state.params[doc.key] = Boolean(raw);
      else if (doc.type === "select") {
        const num = Number(raw);
        state.params[doc.key] = Number.isNaN(num) || typeof doc.options[0] === "string" ? raw : num;
      }
      updateSidebarSummary();
    };

    const numInput = card.querySelector(`[data-key="${doc.key}"]`);
    const rangeInput = card.querySelector(`[data-range="${doc.key}"]`);
    if (doc.type === "bool" && numInput) {
      numInput.onchange = () => {
        setVal(numInput.checked);
        numInput.nextElementSibling.textContent = numInput.checked ? "Đang BẬT (True)" : "Đang TẮT (False)";
      };
    } else if (doc.type === "select" && numInput) {
      numInput.onchange = () => setVal(numInput.value);
    } else if (numInput && rangeInput) {
      rangeInput.oninput = () => {
        numInput.value = rangeInput.value;
        setVal(rangeInput.value);
      };
      numInput.onchange = () => {
        rangeInput.value = numInput.value;
        setVal(numInput.value);
      };
    }

    wrap.appendChild(card);
  });
}

function renderAutoExportChips() {
  const wrap = $("autoExportFormatChips");
  wrap.innerHTML = "";
  (state.bootstrap?.export_formats || []).forEach((fmt) => {
    const checked = state.autoExportFormats.includes(fmt.id);
    const lbl = document.createElement("label");
    lbl.className = "check-label";
    lbl.style.padding = "6px 8px";
    lbl.style.background = "var(--bg)";
    lbl.style.border = "1px solid var(--border)";
    lbl.style.borderRadius = "6px";
    lbl.innerHTML = `<input type="checkbox" ${checked ? "checked" : ""} /> <span>${escapeHtml(fmt.name)}</span>`;
    lbl.querySelector("input").onchange = (e) => {
      if (e.target.checked) {
        if (!state.autoExportFormats.includes(fmt.id)) state.autoExportFormats.push(fmt.id);
      } else {
        state.autoExportFormats = state.autoExportFormats.filter((x) => x !== fmt.id);
      }
    };
    wrap.appendChild(lbl);
  });
}

// ======================================================================
// Live Training Dashboard & Canvas Charts
// ======================================================================

function renderTrainingState() {
  const st = state.training;
  if (!st) return;

  const running = ["preparing", "training", "exporting"].includes(st.status);
  $("btnTrainStart").disabled = running;
  $("btnStartTrainFromConfig").disabled = running;
  $("btnQuickAutoTrain").disabled = running;
  $("btnTrainStop").disabled = !running;

  const badge = $("trainStatusBadge");
  const statusMap = {
    idle: ["Sẵn sàng (Idle)", "badge-muted"],
    preparing: ["Đang chuẩn bị & Nhân bản tập Train...", "badge-accent"],
    training: ["Đang Huấn luyện YOLO (Live)", "badge-accent"],
    exporting: ["Đang tự động xuất file...", "badge-warn"],
    completed: ["Hoàn tất thành công", "badge-success"],
    stopped: ["Đã dừng an toàn", "badge-warn"],
    error: ["Gặp lỗi", "badge-danger"],
  };
  const [text, cls] = statusMap[st.status] || [st.status, "badge-muted"];
  badge.textContent = text;
  badge.className = `badge ${cls}`;

  $("trainRunTitle").textContent = st.run_id ? `${st.run_id} (${st.model})` : "Chưa bắt đầu phiên huấn luyện";
  $("trainEpochLabel").textContent = `Epoch: ${st.epoch || 0} / ${st.total_epochs || 0}`;
  $("trainElapsedLabel").textContent = `Đã chạy: ${formatSeconds(st.elapsed_s)}`;
  $("trainEtaLabel").textContent = running ? `Còn lại: ~${formatSeconds(st.eta_s)}` : "Còn lại: --";
  $("trainProgressFill").style.width = `${Math.min(100, st.progress_pct || 0)}%`;

  if (st.dataset_summary) {
    const ds = st.dataset_summary;
    $("trainDatasetStrip").textContent =
      `Train gốc: ${ds.train_original} ảnh → Nhân bản an toàn x${ds.offline_multiplier}: ${ds.train_effective} ảnh | Val giữ nguyên (không rò rỉ): ${ds.val_count} ảnh | Classes: ${(ds.classes || []).join(", ")}`;
  } else {
    $("trainDatasetStrip").textContent = st.error ? `Lỗi: ${st.error}` : "";
  }

  const hist = st.metrics_history || [];
  const last = hist[hist.length - 1] || {};
  $("kpiMap50").textContent = `${((last.map50 || 0) * 100).toFixed(1)}%`;
  $("kpiBestMap50").textContent = `Cao nhất: ${((st.best_map50 || 0) * 100).toFixed(1)}%`;
  $("kpiMap5095").textContent = `${((last.map50_95 || 0) * 100).toFixed(1)}%`;
  $("kpiBestMap5095").textContent = `Cao nhất: ${((st.best_map50_95 || 0) * 100).toFixed(1)}%`;
  $("kpiPrecision").textContent = `${((last.precision || 0) * 100).toFixed(1)}%`;
  $("kpiRecall").textContent = `${((last.recall || 0) * 100).toFixed(1)}%`;

  drawAllCharts();
}

function drawAllCharts() {
  const hist = state.training?.metrics_history || [];
  drawLineChart("lossChartCanvas", hist, [
    { key: "box_loss", label: "Box Loss", color: "#3b82f6" },
    { key: "cls_loss", label: "Cls Loss", color: "#f59e0b" },
    { key: "dfl_loss", label: "DFL/L1", color: "#8b5cf6" },
  ], false);

  drawLineChart("metricsChartCanvas", hist, [
    { key: "map50", label: "mAP@50", color: "#10b981" },
    { key: "map50_95", label: "mAP@50-95", color: "#06b6d4" },
    { key: "precision", label: "Precision", color: "#f59e0b" },
    { key: "recall", label: "Recall", color: "#ec4899" },
  ], true);
}

function drawLineChart(canvasId, history, series, isPercent) {
  const canvas = $(canvasId);
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(300, Math.floor(rect.width || 500));
  const h = 250;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  ctx.fillStyle = "#0c1017";
  ctx.fillRect(0, 0, w, h);

  const pad = { l: 46, r: 16, t: 28, b: 28 };
  const plotW = w - pad.l - pad.r;
  const plotH = h - pad.t - pad.b;

  // Legend
  let lx = pad.l;
  ctx.font = "600 11.5px Segoe UI, sans-serif";
  series.forEach((s) => {
    ctx.fillStyle = s.color;
    ctx.fillRect(lx, 10, 10, 10);
    ctx.fillStyle = "#cbd5e1";
    ctx.fillText(s.label, lx + 14, 19);
    lx += ctx.measureText(s.label).width + 28;
  });

  if (!history.length) {
    ctx.fillStyle = "#64748b";
    ctx.font = "13px Segoe UI, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Biểu đồ sẽ cập nhật trực tiếp theo từng Epoch qua WebSocket...", w / 2, h / 2);
    ctx.textAlign = "left";
    return;
  }

  let maxVal = isPercent ? 1.0 : 0.1;
  if (!isPercent) {
    history.forEach((pt) => {
      series.forEach((s) => {
        if ((pt[s.key] || 0) > maxVal) maxVal = pt[s.key];
      });
    });
    maxVal *= 1.12;
  }

  // Grid lines
  ctx.strokeStyle = "#1e293b";
  ctx.lineWidth = 1;
  ctx.font = "11px Consolas, monospace";
  ctx.fillStyle = "#64748b";
  for (let i = 0; i <= 4; i++) {
    const gy = pad.t + (plotH * i) / 4;
    ctx.beginPath();
    ctx.moveTo(pad.l, gy);
    ctx.lineTo(w - pad.r, gy);
    ctx.stroke();
    const v = maxVal * (1 - i / 4);
    ctx.fillText(isPercent ? `${Math.round(v * 100)}%` : v.toFixed(2), 4, gy + 4);
  }

  // Curves
  const n = history.length;
  series.forEach((s) => {
    ctx.beginPath();
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2.2;
    history.forEach((pt, idx) => {
      const x = pad.l + (n === 1 ? plotW / 2 : (idx / (n - 1)) * plotW);
      const val = Math.max(0, Math.min(maxVal, pt[s.key] || 0));
      const y = pad.t + plotH * (1 - val / maxVal);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });

  // X axis epoch labels
  ctx.fillStyle = "#64748b";
  ctx.fillText(`Ep 1`, pad.l, h - 8);
  ctx.fillText(`Ep ${history[n - 1].epoch}`, w - pad.r - 36, h - 8);
}

async function triggerStartTraining() {
  if (!state.activeProjectId) return toast("Vui lòng chọn dự án trước.", "warn");
  if (!state.health || !state.health.ready_to_train) {
    return toast("Cần tối thiểu 4 ảnh đã gán nhãn! Hãy vào Tab 1 tạo dữ liệu mẫu hoặc chạy Tự động đánh dấu.", "warn", 5500);
  }
  const payload = {
    project_id: state.activeProjectId,
    params: state.params,
    auto_export_formats: state.autoExportFormats,
    export_precision: $("autoExportPrecision").value,
  };
  switchTab("train");
  $("consoleDrawer").classList.add("open");
  try {
    const res = await api("/api/train/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.training = res;
    renderTrainingState();
    toast("Đã khởi chạy huấn luyện tự động!", "success");
  } catch (err) {
    toast(`Không thể bắt đầu train: ${err.message}`, "error", 6000);
  }
}

async function refreshRunsAndExports() {
  const [runs, artifacts] = await Promise.all([api("/api/runs"), api("/api/exports")]);
  state.runs = runs || [];
  state.artifacts = artifacts || [];
  renderRunsTable();
  renderWeightSelectors();
  renderArtifactsTable();
}

function renderRunsTable() {
  const wrap = $("runsTableWrap");
  if (!state.runs.length) {
    wrap.innerHTML = `<div class="muted-text" style="padding:10px">Chưa có phiên huấn luyện nào được lưu.</div>`;
    $("runPlotsGallery").innerHTML = "";
    return;
  }
  wrap.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Run ID</th>
          <th>Model</th>
          <th>Trạng thái</th>
          <th>Epoch</th>
          <th>Best mAP50</th>
          <th>mAP50-95</th>
          <th>Hành động</th>
        </tr>
      </thead>
      <tbody>
        ${state.runs.map((r, i) => `
          <tr>
            <td style="font-family:var(--mono);font-size:12px">${escapeHtml(r.run_id)}</td>
            <td><span class="badge">${escapeHtml(r.model || "")}</span></td>
            <td><span class="badge ${r.status === "completed" ? "badge-success" : r.status === "error" ? "badge-danger" : "badge-warn"}">${escapeHtml(r.status)}</span></td>
            <td>${r.epoch}/${r.total_epochs}</td>
            <td><strong>${((r.best_map50 || 0) * 100).toFixed(1)}%</strong></td>
            <td>${((r.best_map50_95 || 0) * 100).toFixed(1)}%</td>
            <td>
              <div class="inline-row">
                <button class="btn btn-secondary btn-sm" data-view-plots="${i}">📊 Biểu đồ</button>
                ${r.best_path ? `<button class="btn btn-primary btn-sm" data-use-export="${i}">📤 Xuất / Test</button>` : ""}
              </div>
            </td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;

  wrap.querySelectorAll("[data-view-plots]").forEach((btn) => {
    btn.onclick = () => showRunPlots(state.runs[Number(btn.dataset.viewPlots)]);
  });
  wrap.querySelectorAll("[data-use-export]").forEach((btn) => {
    btn.onclick = () => {
      const run = state.runs[Number(btn.dataset.useExport)];
      if (run?.best_path) {
        $("exportWeightSelect").value = run.best_path;
        $("testWeightSelect").value = run.best_path;
        switchTab("export");
      }
    };
  });

  if (state.runs[0]?.plots?.length) {
    showRunPlots(state.runs[0]);
  }
}

function showRunPlots(run) {
  const gal = $("runPlotsGallery");
  gal.innerHTML = "";
  if (!run?.plots?.length) {
    gal.innerHTML = `<div class="muted-text">Phiên này chưa có ảnh biểu đồ Ultralytics.</div>`;
    return;
  }
  run.plots.forEach((pl) => {
    const c = document.createElement("div");
    c.className = "plot-card";
    c.innerHTML = `<img src="${pl.url}" alt="${escapeHtml(pl.name)}" /><div>${escapeHtml(pl.name)}</div>`;
    c.querySelector("img").onclick = () => window.open(pl.url, "_blank");
    gal.appendChild(c);
  });
}

// ======================================================================
// Multi-Format Export & Live Inference Playground
// ======================================================================

function renderExportFormatCards() {
  const wrap = $("exportFormatCards");
  wrap.innerHTML = "";
  (state.bootstrap?.export_formats || []).forEach((fmt) => {
    const sel = state.selectedExportFormats.has(fmt.id);
    const card = document.createElement("div");
    card.className = `format-card ${sel ? "selected" : ""}`;
    card.innerHTML = `
      <input type="checkbox" ${sel ? "checked" : ""} />
      <div>
        <strong>${escapeHtml(fmt.name)}</strong>
        <small>${escapeHtml(fmt.platforms.join(" · "))}</small>
        <small>${escapeHtml(fmt.description)}</small>
      </div>
    `;
    card.onclick = (e) => {
      if (e.target.tagName !== "INPUT") {
        const cb = card.querySelector("input");
        cb.checked = !cb.checked;
      }
      const checked = card.querySelector("input").checked;
      if (checked) state.selectedExportFormats.add(fmt.id);
      else state.selectedExportFormats.delete(fmt.id);
      card.classList.toggle("selected", checked);
    };
    wrap.appendChild(card);
  });
}

function renderWeightSelectors() {
  const opts = [];
  state.runs.forEach((r) => {
    if (r.best_path) {
      opts.push({ val: r.best_path, label: `[Đã train] ${r.run_id} (mAP50: ${((r.best_map50 || 0) * 100).toFixed(1)}%)` });
    }
  });
  (state.bootstrap?.custom_weights || []).forEach((cw) => {
    opts.push({ val: cw.id, label: `[Custom] ${cw.name}` });
  });
  ["yolo26n.pt", "yolo11n.pt", "yolov8n.pt"].forEach((pre) => {
    opts.push({ val: pre, label: `[Pretrained] ${pre}` });
  });

  ["exportWeightSelect", "testWeightSelect"].forEach((id) => {
    const sel = $(id);
    const prev = sel.value;
    sel.innerHTML = opts.map((o) => `<option value="${escapeHtml(o.val)}">${escapeHtml(o.label)}</option>`).join("");
    if (prev && opts.some((o) => o.val === prev)) sel.value = prev;
  });
}

function renderArtifactsTable() {
  const wrap = $("artifactsTableWrap");
  if (!state.artifacts.length) {
    wrap.innerHTML = `<div class="muted-text" style="padding:10px">Chưa có file model nào được xuất. Hãy chọn định dạng ở trên rồi bấm Xuất!</div>`;
    return;
  }
  wrap.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Tên tệp</th>
          <th>Định dạng</th>
          <th>Precision</th>
          <th>ImgSz</th>
          <th>Nền tảng phù hợp</th>
          <th>Dung lượng</th>
          <th>Hành động</th>
        </tr>
      </thead>
      <tbody>
        ${state.artifacts.map((a, idx) => `
          <tr>
            <td style="font-family:var(--mono);font-size:12px">${escapeHtml(a.filename)}</td>
            <td><span class="badge badge-accent">${escapeHtml(a.format_name || a.format)}</span></td>
            <td><span class="badge badge-success">${escapeHtml(a.precision)}</span></td>
            <td>${a.imgsz}px</td>
            <td><small class="muted-text">${escapeHtml((a.platforms || []).join(", "))}</small></td>
            <td style="font-family:var(--mono)">${a.size_mb} MB</td>
            <td>
              <div class="inline-row">
                <a class="btn btn-primary btn-sm" href="${a.download_url}" download>⬇ Tải xuống</a>
                <button class="btn btn-secondary btn-sm" data-snippet="${idx}">📋 Code mẫu</button>
                <button class="btn btn-danger-outline btn-sm" data-del-art="${escapeHtml(a.filename)}">✕</button>
              </div>
            </td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;

  wrap.querySelectorAll("[data-snippet]").forEach((btn) => {
    btn.onclick = () => {
      const art = state.artifacts[Number(btn.dataset.snippet)];
      openModal(
        `Mã mẫu tích hợp — ${art.format_name}`,
        `<p class="muted-text mb-12">Tệp: <code>${escapeHtml(art.filename)}</code> · Nền tảng: ${escapeHtml((art.platforms || []).join(", "))}</p>
         <pre class="code-block">${escapeHtml(art.snippet || "")}</pre>
         <button class="btn btn-primary mt-12" id="btnCopySnippet">📋 Sao chép mã nguồn</button>`
      );
      $("btnCopySnippet").onclick = () => {
        navigator.clipboard?.writeText(art.snippet || "");
        toast("Đã sao chép mã tích hợp!", "success");
      };
    };
  });

  wrap.querySelectorAll("[data-del-art]").forEach((btn) => {
    btn.onclick = async () => {
      await api(`/api/exports/${encodeURIComponent(btn.dataset.delArt)}`, { method: "DELETE" });
      await refreshRunsAndExports();
    };
  });
}

// ======================================================================
// Modals & Webcam Helpers
// ======================================================================

function openModal(title, htmlContent) {
  $("modalTitle").textContent = title;
  $("modalBody").innerHTML = htmlContent;
  $("modalOverlay").classList.remove("hidden");
}

function closeModal() {
  $("modalOverlay").classList.add("hidden");
}

async function startWebcam() {
  try {
    state.webcamStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    $("webcamVideo").srcObject = state.webcamStream;
    $("btnSnapCam").disabled = false;
    $("btnBurstCam").disabled = false;
    $("btnStopCam").disabled = false;
    $("btnStartCam").disabled = true;
    toast("Camera đã sẵn sàng! Đưa vật thể vào khung hình và bấm Chụp.", "success");
  } catch (err) {
    toast(`Không thể truy cập Camera: ${err.message}`, "error");
  }
}

function stopWebcam() {
  if (state.burstTimer) {
    clearInterval(state.burstTimer);
    state.burstTimer = null;
    $("btnBurstCam").textContent = "⚡ Chụp liên tiếp (0.6s)";
  }
  if (state.webcamStream) {
    state.webcamStream.getTracks().forEach((t) => t.stop());
    state.webcamStream = null;
  }
  $("webcamVideo").srcObject = null;
  $("btnSnapCam").disabled = true;
  $("btnBurstCam").disabled = true;
  $("btnStopCam").disabled = true;
  $("btnStartCam").disabled = false;
}

async function captureWebcamOnce() {
  const video = $("webcamVideo");
  const canvas = $("webcamCanvas");
  if (!video.videoWidth) return;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  const dataUrl = canvas.toDataURL("image/jpeg", 0.92);
  await api(`/api/projects/${state.activeProjectId}/ingest/webcam`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ data_url: dataUrl, label_hint: "webcam" }),
  });
  state.webcamCaptured += 1;
  $("camCountBadge").textContent = `Đã chụp: ${state.webcamCaptured}`;
}

// ======================================================================
// Bind All UI Events
// ======================================================================

function bindStaticEvents() {
  bindCanvasEvents();

  // Main sidebar navigation
  document.querySelectorAll("#mainNav .nav-item").forEach((btn) => {
    btn.onclick = () => switchTab(btn.dataset.tab);
  });

  // Ingest subtabs
  document.querySelectorAll("#ingestTabs .subtab").forEach((btn) => {
    btn.onclick = () => {
      document.querySelectorAll("#ingestTabs .subtab").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".ingest-pane").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      $(`ingest-${btn.dataset.ingest}`).classList.add("active");
    };
  });

  // Platform OS filter
  document.querySelectorAll("#platformFilterTabs .subtab").forEach((btn) => {
    btn.onclick = () => {
      document.querySelectorAll("#platformFilterTabs .subtab").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.activePlatformOs = btn.dataset.os;
      renderPlatformCards();
    };
  });

  // Model family tabs
  document.querySelectorAll("#modelFamilyTabs .subtab").forEach((btn) => {
    btn.onclick = () => {
      state.activeModelFamily = btn.dataset.family;
      renderModelPicker();
    };
  });

  // Hyperparam group tabs
  document.querySelectorAll("#paramGroupTabs .subtab").forEach((btn) => {
    btn.onclick = () => {
      document.querySelectorAll("#paramGroupTabs .subtab").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.activeParamGroup = btn.dataset.group;
      renderHyperparams();
    };
  });

  // Project select & new project modal
  $("projectSelect").onchange = async (e) => {
    state.activeProjectId = e.target.value;
    state.selectedImageIdx = -1;
    state.selectedImageFilename = null;
    state.selectedThumbFilenames.clear();
    state.editorImgObj = null;
    state.editorImgUrl = null;
    await loadActiveProjectData();
  };

  $("btnNewProject").onclick = () => {
    openModal(
      "Tạo Dự án Dataset mới",
      `<div class="field">
         <label>Tên dự án</label>
         <input type="text" id="newProjName" class="input" placeholder="VD: PhanLoai_HinhKhoi_3D" />
       </div>
       <div class="field mt-12">
         <label>Danh sách lớp vật thể (cách nhau bằng dấu phẩy)</label>
         <input type="text" id="newProjClasses" class="input" value="cube, cylinder, sphere" />
       </div>
       <div class="inline-row mt-16">
         <button id="btnConfirmNewProj" class="btn btn-primary">Tạo dự án</button>
       </div>`
    );
    $("btnConfirmNewProj").onclick = async () => {
      const name = $("newProjName").value.trim() || "Dataset_Moi";
      const classes = $("newProjClasses").value.split(",").map((s) => s.trim()).filter(Boolean);
      const proj = await api("/api/projects", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, classes }),
      });
      state.activeProjectId = proj.id;
      state.selectedImageIdx = -1;
      state.selectedImageFilename = null;
      state.selectedThumbFilenames.clear();
      state.editorImgObj = null;
      state.editorImgUrl = null;
      closeModal();
      await loadActiveProjectData();
      toast(`Đã tạo dự án "${proj.name}"!`, "success");
    };
  };

  // CUDA 11.8 advisory modal
  $("cudaAdvisoryPill").onclick = () => {
    const t = state.telemetry || {};
    const isLinux = (t.os_type || "").toLowerCase() === "linux";
    const setupCmd = t.setup_cmd || (isLinux ? "python3 run.py --setup" : "python run.py --setup");
    const cmd =
      t.cuda_install_cmd ||
      `${isLinux ? "python3" : "python"} -m pip install --upgrade --force-reinstall torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118`;
    const helperScript = isLinux ? "./start_app.sh --setup" : "setup_env_cu118.bat";
    const statusLine = t.cuda_advisory
      ? `Hệ thống (${escapeHtml(t.os || "OS")}) phát hiện GPU <strong>NVIDIA</strong> nhưng PyTorch hiện tại (<code>${escapeHtml(t.torch_version || "unknown")}</code>) trong Python <code>${escapeHtml(t.python || "3.11.6")}</code> chưa bật CUDA.`
      : `Môi trường (${escapeHtml(t.os || "OS")}) hiện đang dùng Python <code>${escapeHtml(t.python || "3.11.6")}</code>, PyTorch <code>${escapeHtml(t.torch_version || "unknown")}</code> (CUDA <code>${escapeHtml(t.torch_cuda_version || "N/A")}</code>), trong khi cấu hình đích của dự án là <strong>Python 3.11.6 + CUDA 11.8</strong>.`;
    openModal(
      "⚡ Cấu hình GPU NVIDIA (Python 3.11.6 + CUDA 11.8 / TF32)",
      `<p>${statusLine}</p>
       <p class="mt-8">Cách nhanh nhất (tự nhận diện Windows / Linux): chạy lệnh sau hoặc <code>${escapeHtml(helperScript)}</code>:</p>
       <pre class="code-block mt-8">${escapeHtml(setupCmd)}</pre>
       <p class="mt-8">Hoặc cài trực tiếp PyTorch 2.7.1 + CUDA 11.8 bằng pip:</p>
       <pre class="code-block mt-8">${escapeHtml(cmd)}</pre>
       <p class="muted-text mt-8">Sau khi cài xong, khởi động lại app — thanh trạng thái sẽ tự nhận GPU CUDA 11.8 &amp; bật TF32!</p>`
    );
  };

  $("modalClose").onclick = closeModal;
  $("modalOverlay").onclick = (e) => { if (e.target === $("modalOverlay")) closeModal(); };

  // Image dropzone & file input
  const dz = $("imageDropZone");
  const fi = $("inputImageFiles");
  dz.onclick = () => fi.click();
  dz.ondragover = (e) => { e.preventDefault(); dz.classList.add("dragover"); };
  dz.ondragleave = () => dz.classList.remove("dragover");
  dz.ondrop = async (e) => {
    e.preventDefault();
    dz.classList.remove("dragover");
    if (e.dataTransfer.files.length) await uploadImageFileList(e.dataTransfer.files);
  };
  fi.onchange = async () => {
    if (fi.files.length) await uploadImageFileList(fi.files);
    fi.value = "";
  };

  async function uploadImageFileList(files) {
    const fd = new FormData();
    Array.from(files).forEach((f) => fd.append("files", f));
    showGlobalProgress(`Đang tải lên ${files.length} ảnh...`, 40);
    try {
      const res = await api(`/api/projects/${state.activeProjectId}/upload/images`, { method: "POST", body: fd });
      toast(`Đã tải lên ${res.added} ảnh thành công!`, "success");
      await loadActiveProjectData();
    } catch (err) {
      toast(`Lỗi tải ảnh: ${err.message}`, "error");
    } finally {
      $("globalProgressBar").classList.add("hidden");
    }
  }

  // ZIP upload
  $("btnUploadZip").onclick = async () => {
    const f = $("inputZipFile").files[0];
    if (!f) return toast("Chọn file .zip trước.", "warn");
    const fd = new FormData();
    fd.append("file", f);
    showGlobalProgress("Đang giải nén & chuyển đổi định dạng nhãn ZIP...", 50);
    try {
      const res = await api(`/api/projects/${state.activeProjectId}/upload/zip`, { method: "POST", body: fd });
      toast(`Đã nhập ${res.added_images} ảnh và ${res.added_labels} nhãn từ ZIP!`, "success");
      await loadActiveProjectData();
    } catch (err) {
      toast(err.message, "error");
    } finally {
      $("globalProgressBar").classList.add("hidden");
    }
  };

  // Video upload
  $("btnUploadVideo").onclick = async () => {
    const f = $("inputVideoFile").files[0];
    if (!f) return toast("Chọn file video trước.", "warn");
    const fd = new FormData();
    fd.append("file", f);
    fd.append("fps_extract", $("videoFps").value);
    fd.append("max_frames", $("videoMaxFrames").value);
    fd.append("skip_blurry", $("videoSkipBlur").checked);
    fd.append("dedup", $("videoDedup").checked);
    showGlobalProgress("Đang trích xuất khung hình từ video...", 20);
    try {
      const res = await api(`/api/projects/${state.activeProjectId}/upload/video`, { method: "POST", body: fd });
      toast(`Đã trích xuất ${res.added} ảnh rõ nét (lọc mờ ${res.skipped_blur}, lọc trùng ${res.skipped_dup}).`, "success");
      await loadActiveProjectData();
    } catch (err) {
      toast(err.message, "error");
    }
  };

  // Webcam controls
  $("btnStartCam").onclick = startWebcam;
  $("btnStopCam").onclick = stopWebcam;
  $("btnSnapCam").onclick = captureWebcamOnce;
  $("btnBurstCam").onclick = () => {
    if (state.burstTimer) {
      clearInterval(state.burstTimer);
      state.burstTimer = null;
      $("btnBurstCam").textContent = "⚡ Chụp liên tiếp (0.6s)";
    } else {
      $("btnBurstCam").textContent = "⏹ Dừng chụp liên tiếp";
      captureWebcamOnce();
      state.burstTimer = setInterval(captureWebcamOnce, 600);
    }
  };

  // Local folder ingest
  $("btnIngestFolder").onclick = async () => {
    const folder_path = $("inputLocalFolder").value.trim();
    if (!folder_path) return toast("Nhập đường dẫn thư mục máy tính.", "warn");
    try {
      const res = await api(`/api/projects/${state.activeProjectId}/ingest/folder`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ folder_path }),
      });
      toast(`Đã nhập ${res.added_images} ảnh và ${res.added_labels} nhãn từ thư mục!`, "success");
      await loadActiveProjectData();
    } catch (err) {
      toast(err.message, "error");
    }
  };

  // Demo 3D shapes generator
  $("btnGenerateDemo").onclick = async () => {
    const count = Number($("demoCount").value || 45);
    const include_labels = $("demoIncludeLabels").value === "true";
    showGlobalProgress(`Đang sinh ${count} ảnh hình khối 3D...`, 50);
    try {
      const res = await api(`/api/projects/${state.activeProjectId}/generate-demo`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ count, include_labels }),
      });
      toast(`Đã tạo ${res.created} ảnh hình khối 3D (cube, cylinder, sphere)!`, "success");
      await loadActiveProjectData();
    } catch (err) {
      toast(err.message, "error");
    } finally {
      $("globalProgressBar").classList.add("hidden");
    }
  };

  // Add class
  $("btnAddClass").onclick = async () => {
    const val = $("inputNewClass").value.trim();
    if (!val) return;
    const curr = getCurrentClasses();
    if (curr.includes(val)) return toast("Lớp này đã tồn tại.", "warn");
    await api(`/api/projects/${state.activeProjectId}/classes`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ classes: [...curr, val] }),
    });
    $("inputNewClass").value = "";
    await loadActiveProjectData();
  };

  // Refresh & Clear dataset
  $("btnRefreshDataset").onclick = loadActiveProjectData;
  $("btnClearImages").onclick = async () => {
    if (!confirm("Xóa toàn bộ ảnh và nhãn trong dự án hiện tại?")) return;
    await api(`/api/projects/${state.activeProjectId}/images`, { method: "DELETE" });
    state.selectedImageIdx = -1;
    state.selectedImageFilename = null;
    state.selectedThumbFilenames.clear();
    await loadActiveProjectData();
    toast("Đã xóa toàn bộ ảnh trong dự án.", "info");
  };

  // Auto-preprocess
  $("btnRunPreprocess").onclick = async () => {
    const body = {
      remove_duplicates: $("ppDedup").checked,
      remove_blurry: $("ppBlur").checked,
      max_side: Number($("ppMaxSide").value || 1280),
      apply_clahe: $("ppClahe").checked,
      clean_labels: $("ppCleanLabels").checked,
    };
    try {
      const res = await api(`/api/projects/${state.activeProjectId}/preprocess`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      toast(`Tiền xử lý xong! Quét ${res.total_scanned} · Xóa trùng ${res.removed_duplicates} · Xóa mờ ${res.removed_blurry} · Resize ${res.resized_count}.`, "success", 5000);
      await loadActiveProjectData();
    } catch (err) {
      toast(err.message, "error");
    }
  };

  // Auto-label engine UI toggles
  $("alEngine").onchange = () => {
    const eng = $("alEngine").value;
    $("alClassModeWrap").style.display = eng === "cv_smart" ? "flex" : "none";
    $("alTargetClassWrap").style.display = eng === "open_vocab" ? "none" : "flex";
    $("alModelWrap").style.display = eng === "cv_smart" ? "none" : "flex";
    if (eng === "open_vocab") $("alModelName").value = "yolov8s-worldv2.pt";
    if (eng === "yolo_model") $("alModelName").value = state.runs[0]?.best_path || "yolo26n.pt";
  };
  $("alSensitivity").oninput = () => { $("alSensVal").textContent = Number($("alSensitivity").value).toFixed(2); };

  $("btnRunAutoLabel").onclick = async () => {
    const eng = $("alEngine").value;
    const payload = {
      engine: eng,
      overwrite_existing: $("alOverwrite").checked,
      conf: Number($("alSensitivity").value),
      sensitivity: Number($("alSensitivity").value),
      class_mode: $("alClassMode").value,
      target_class_id: Number($("alTargetClass").value || 0),
      model_name: $("alModelName").value.trim() || "yolo26n.pt",
    };
    showGlobalProgress("Đang chạy Tự động đánh dấu...", 15);
    try {
      const res = await api(`/api/projects/${state.activeProjectId}/autolabel`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      toast(res.message || "Tự động đánh dấu hoàn tất!", "success", 5000);
      await loadActiveProjectData();
    } catch (err) {
      toast(`Auto-Label lỗi: ${err.message}`, "error", 6500);
      $("globalProgressBar").classList.add("hidden");
    }
  };

  // Sidebar & Labeler Layout Toggles
  $("btnToggleSidebar").onclick = () => toggleSidebar();
  $("btnSidebarEdgeToggle").onclick = () => toggleSidebar();
  $("btnToggleGallery").onclick = () => toggleGallery();
  $("btnRailToggleGallery").onclick = () => toggleGallery();
  $("btnToggleGalleryWide").onclick = () => toggleGalleryWide();
  $("btnRailWideGallery").onclick = () => toggleGalleryWide();
  $("btnToggleSetupTools").onclick = () => toggleSetupTools();
  $("btnRailToggleSetup").onclick = () => toggleSetupTools();
  $("btnRailToggleSidebar").onclick = () => toggleSidebar();
  $("btnToggleFocusLabel").onclick = () => toggleFocusLabelMode();
  $("btnHeaderFocusLabel").onclick = () => toggleFocusLabelMode();
  $("btnRailToggleFocus").onclick = () => toggleFocusLabelMode();

  // Gallery selection & batch delete
  $("chkSelectAllGallery").onchange = (e) => {
    const filtered = getFilteredImages();
    if (e.target.checked) {
      filtered.forEach(({ im }) => state.selectedThumbFilenames.add(im.filename));
    } else {
      filtered.forEach(({ im }) => state.selectedThumbFilenames.delete(im.filename));
    }
    renderImageGallery();
  };
  $("btnDeleteSelectedImgs").onclick = deleteSelectedImages;

  // Gallery & editor actions
  $("galleryFilter").onchange = renderImageGallery;
  $("btnPrevImg").onclick = () => stepEditorImage(-1);
  $("btnNextImg").onclick = () => stepEditorImage(1);
  $("btnZoomOut").onclick = () => setCanvasZoom((state.canvasZoom || 1.0) - 0.25);
  $("btnZoomReset").onclick = () => setCanvasZoom(1.0);
  $("btnZoomIn").onclick = () => setCanvasZoom((state.canvasZoom || 1.0) + 0.25);
  $("btnDeleteCurrentImg").onclick = deleteCurrentEditorImage;
  $("editorActiveClass").onchange = () => {
    if (state.selectedBoxIdx >= 0 && state.editorBoxes[state.selectedBoxIdx]) {
      state.editorBoxes[state.selectedBoxIdx].class_id = Number($("editorActiveClass").value);
      redrawEditorCanvas();
      renderEditorBoxList();
      saveCurrentLabels(true);
    }
  };
  $("btnDeleteSelectedBox").onclick = () => {
    if (state.selectedBoxIdx < 0) return;
    state.editorBoxes.splice(state.selectedBoxIdx, 1);
    state.selectedBoxIdx = Math.min(state.selectedBoxIdx, state.editorBoxes.length - 1);
    redrawEditorCanvas();
    renderEditorBoxList();
    saveCurrentLabels(true);
  };
  $("btnClearBoxesImg").onclick = () => {
    state.editorBoxes = [];
    state.selectedBoxIdx = -1;
    redrawEditorCanvas();
    renderEditorBoxList();
    saveCurrentLabels(true);
  };
  $("btnSaveLabels").onclick = () => saveCurrentLabels(false);

  // Upload custom .pt weight
  $("btnUploadCustomWeight").onclick = async () => {
    const f = $("inputCustomWeight").files[0];
    if (!f) return toast("Chọn file trọng số .pt trước.", "warn");
    const fd = new FormData();
    fd.append("file", f);
    try {
      const res = await api("/api/weights/upload", { method: "POST", body: fd });
      state.bootstrap.custom_weights.unshift(res);
      state.activeModelFamily = "custom";
      state.params.model = res.id;
      renderModelPicker();
      renderWeightSelectors();
      updateSidebarSummary();
      toast(`Đã tải lên trọng số tùy chỉnh ${res.name}!`, "success");
    } catch (err) {
      toast(err.message, "error");
    }
  };

  // Config & Training buttons
  $("btnResetPreset").onclick = () => applyPreset("small_shapes_default", true);
  $("btnQuickAutoTrain").onclick = triggerStartTraining;
  $("btnStartTrainFromConfig").onclick = triggerStartTraining;
  $("btnTrainStart").onclick = triggerStartTraining;
  $("btnTrainStop").onclick = async () => {
    const res = await api("/api/train/stop", { method: "POST" });
    toast(res.message, "warn");
  };
  $("btnRefreshRuns").onclick = refreshRunsAndExports;
  $("btnRefreshExports").onclick = refreshRunsAndExports;

  // Export action
  $("btnRunExport").onclick = async () => {
    const formats = Array.from(state.selectedExportFormats);
    if (!formats.length) return toast("Hãy tích chọn ít nhất 1 định dạng muốn xuất!", "warn");
    const weights_path = $("exportWeightSelect").value;
    if (!weights_path) return toast("Chọn trọng số nguồn trước.", "warn");

    const matchedRun = state.runs.find((r) => r.best_path === weights_path);
    const payload = {
      weights_path,
      formats,
      imgsz: Number($("exportImgsz").value || 640),
      precision: $("exportPrecision").value,
      simplify: $("exportSimplify").checked,
      dynamic: $("exportDynamic").checked,
      nms: $("exportNms").checked ? true : null,
      data_yaml: matchedRun?.dataset_summary?.yaml_path || null,
      run_id: matchedRun?.run_id || null,
    };
    $("btnRunExport").disabled = true;
    $("exportStatusText").textContent = "Đang chuyển đổi mô hình...";
    $("consoleDrawer").classList.add("open");
    try {
      await api("/api/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      await refreshRunsAndExports();
    } catch (err) {
      toast(`Xuất thất bại: ${err.message}`, "error", 6500);
    } finally {
      $("btnRunExport").disabled = false;
    }
  };

  // Live Inference Test Playground
  $("testConf").oninput = () => { $("testConfVal").textContent = Number($("testConf").value).toFixed(2); };
  $("btnRunTest").onclick = async () => {
    const f = $("inputTestImage").files[0];
    if (!f) return toast("Chọn 1 ảnh để kiểm thử nhận diện.", "warn");
    const weights_path = $("testWeightSelect").value;
    const fd = new FormData();
    fd.append("file", f);
    fd.append("weights_path", weights_path);
    fd.append("conf", $("testConf").value);
    fd.append("imgsz", $("exportImgsz").value || "640");
    $("testResultWrap").innerHTML = `<div class="muted-text">Đang chạy suy luận mô hình...</div>`;
    try {
      const res = await api("/api/predict", { method: "POST", body: fd });
      const rows = (res.detections || []).map((d) =>
        `<span class="badge badge-accent">${escapeHtml(d.class_name)} · ${(d.confidence * 100).toFixed(1)}%</span>`
      ).join(" ");
      $("testResultWrap").innerHTML = `
        <img src="${res.annotated_data_url}" alt="Prediction result" />
        <div class="inline-row">
          <span class="badge badge-success">⚡ ${res.inference_ms} ms</span>
          <span class="badge">Phát hiện: ${res.count} vật thể</span>
          ${rows || '<span class="muted-text">Không phát hiện vật thể ở ngưỡng conf hiện tại.</span>'}
        </div>
      `;
    } catch (err) {
      $("testResultWrap").innerHTML = `<div class="alert-item alert-warn">❌ ${escapeHtml(err.message)}</div>`;
    }
  };

  // Console Drawer
  $("consoleToggle").onclick = (e) => {
    if (e.target.id === "btnClearConsole") return;
    const d = $("consoleDrawer");
    d.classList.toggle("open");
    $("consoleChevron").textContent = d.classList.contains("open") ? "▼ Thu gọn" : "▲ Mở rộng";
  };
  $("btnClearConsole").onclick = (e) => {
    e.stopPropagation();
    $("consoleBody").innerHTML = "";
    $("consoleLastLine").textContent = "Đã xóa log.";
  };

  window.addEventListener("resize", () => {
    if ($("tab-train").classList.contains("active")) drawAllCharts();
    if ($("tab-dataset").classList.contains("active")) scheduleCanvasResize(false);
  });
}

document.addEventListener("DOMContentLoaded", initApp);
