const STORAGE_RESULTS_KEY = "clinical_eval_results_v2";
const STORAGE_INDEX_KEY = "clinical_eval_current_index_v2";
const STORAGE_CASES_CACHE_KEY = "clinical_eval_cases_cache_v1";
const STORAGE_THEME_KEY = "clinical_eval_theme_v1";

const state = {
  cases: [],
  currentIndex: 0,
  resultsByImageId: new Map(),
  draftCommentsByImageId: new Map(),
  galleryFilters: {
    status: null, // null | "match" | "no-match" | "pending"
    withComment: false,
  },
};

const dom = {
  evaluationView: document.getElementById("evaluationView"),
  completionView: document.getElementById("completionView"),
  actionPanel: document.querySelector(".action-panel"),
  localLoader: document.getElementById("localLoader"),
  dataManagerModal: document.getElementById("dataManagerModal"),
  dataManagerBackdrop: document.getElementById("dataManagerBackdrop"),
  galleryModal: document.getElementById("galleryModal"),
  galleryBackdrop: document.getElementById("galleryBackdrop"),
  galleryGrid: document.getElementById("galleryGrid"),
  casesFileInput: document.getElementById("casesFileInput"),
  casesFileInputManager: document.getElementById("casesFileInputManager"),
  progressText: document.getElementById("progressText"),
  progressBar: document.getElementById("progressBar"),
  caseImage: document.getElementById("caseImage"),
  trueClass: document.getElementById("trueClass"),
  currentVoteBadge: document.getElementById("currentVoteBadge"),
  aiExplanation: document.getElementById("aiExplanation"),
  commentBox: document.getElementById("commentBox"),
  btnMatch: document.getElementById("btnMatch"),
  btnNoMatch: document.getElementById("btnNoMatch"),
  btnPrev: document.getElementById("btnPrev"),
  btnNext: document.getElementById("btnNext"),
  btnOpenGallery: document.getElementById("btnOpenGallery"),
  btnOpenDataManager: document.getElementById("btnOpenDataManager"),
  btnCloseGallery: document.getElementById("btnCloseGallery"),
  btnCloseDataManager: document.getElementById("btnCloseDataManager"),
  filterMatch: document.getElementById("filterMatch"),
  filterNoMatch: document.getElementById("filterNoMatch"),
  filterPending: document.getElementById("filterPending"),
  filterComment: document.getElementById("filterComment"),
  btnTheme: document.getElementById("btnTheme"),
  btnExportQuick: document.getElementById("btnExportQuick"),
  btnExportFinal: document.getElementById("btnExportFinal"),
  btnResetCurrentData: document.getElementById("btnResetCurrentData"),
  summaryMatch: document.getElementById("summaryMatch"),
  summaryNoMatch: document.getElementById("summaryNoMatch"),
  summaryPending: document.getElementById("summaryPending"),
  summaryComments: document.getElementById("summaryComments"),
  statusLine: document.getElementById("statusLine"),
};

window.onload = async () => {
  applyStoredTheme();
  wireEvents();
  await loadCases();
};

function wireEvents() {
  dom.btnMatch.addEventListener("click", () => recordVote("Match"));
  dom.btnNoMatch.addEventListener("click", () => recordVote("No Match"));
  dom.btnPrev.addEventListener("click", goToPreviousCase);
  dom.btnNext.addEventListener("click", goToNextCase);
  dom.btnOpenGallery.addEventListener("click", openGallery);
  dom.btnOpenDataManager.addEventListener("click", openDataManager);
  dom.btnCloseGallery.addEventListener("click", closeGallery);
  dom.btnCloseDataManager.addEventListener("click", closeDataManager);
  dom.galleryBackdrop.addEventListener("click", closeGallery);
  dom.dataManagerBackdrop.addEventListener("click", closeDataManager);
  if (dom.filterMatch) {
    dom.filterMatch.addEventListener("click", () => toggleGalleryStatusFilter("match"));
  }
  if (dom.filterNoMatch) {
    dom.filterNoMatch.addEventListener("click", () => toggleGalleryStatusFilter("no-match"));
  }
  if (dom.filterPending) {
    dom.filterPending.addEventListener("click", () => toggleGalleryStatusFilter("pending"));
  }
  if (dom.filterComment) {
    dom.filterComment.addEventListener("click", toggleGalleryCommentFilter);
  }
  dom.btnTheme.addEventListener("click", toggleTheme);
  dom.btnExportQuick.addEventListener("click", exportCsv);
  dom.btnExportFinal.addEventListener("click", exportCsv);
  if (dom.btnResetCurrentData) {
    dom.btnResetCurrentData.addEventListener("click", resetCurrentDataFromZero);
  }
  dom.casesFileInput.addEventListener("change", onCasesFileSelected);
  dom.casesFileInputManager.addEventListener("change", onCasesFileSelected);
  document.addEventListener("keydown", onGlobalKeydown);
  updateActionAvailability();
}

function applyStoredTheme() {
  const savedTheme = localStorage.getItem(STORAGE_THEME_KEY);
  const theme = savedTheme === "dark" ? "dark" : "light";
  document.body.setAttribute("data-theme", theme);
  dom.btnTheme.textContent = theme === "dark" ? "☀️" : "🌙";
  dom.btnTheme.title = theme === "dark" ? "Cambiar a modo claro" : "Cambiar a modo oscuro";
}

function toggleTheme() {
  const current = document.body.getAttribute("data-theme") === "dark" ? "dark" : "light";
  const next = current === "dark" ? "light" : "dark";
  document.body.setAttribute("data-theme", next);
  localStorage.setItem(STORAGE_THEME_KEY, next);
  dom.btnTheme.textContent = next === "dark" ? "☀️" : "🌙";
  dom.btnTheme.title = next === "dark" ? "Cambiar a modo claro" : "Cambiar a modo oscuro";
}

function onGlobalKeydown(event) {
  if (event.key === "Escape" && !dom.galleryModal.classList.contains("hidden")) {
    closeGallery();
    return;
  }

  if (event.key === "Escape" && !dom.dataManagerModal.classList.contains("hidden")) {
    closeDataManager();
  }
}

async function loadCases() {
  const cachedCases = readCachedCases();
  if (cachedCases.length) {
    state.cases = cachedCases;
    restoreProgress();
    hideLocalLoader();
    render();
    setStatus("Casos cargados desde cache local.");
    return;
  }

  moveToInitialLoaderState();
  setStatus("Carga manual requerida. Selecciona un cases.json para continuar.");
}

function restoreProgress() {
  state.resultsByImageId = readStoredResults();

  const savedIndexRaw = localStorage.getItem(STORAGE_INDEX_KEY);
  const savedIndex = Number.parseInt(savedIndexRaw ?? "0", 10);
  const maxIndex = state.cases.length;

  if (Number.isFinite(savedIndex) && savedIndex >= 0 && savedIndex <= maxIndex) {
    state.currentIndex = savedIndex;
  }

  if (state.currentIndex > 0) {
    setStatus(`Progreso recuperado: continuas en el caso ${state.currentIndex + 1}.`);
  }
}

function readStoredResults() {
  const raw = localStorage.getItem(STORAGE_RESULTS_KEY);
  if (!raw) {
    return new Map();
  }

  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return new Map();
    }

    const map = new Map();
    for (const item of parsed) {
      if (!item || typeof item.id_imagen === "undefined") {
        continue;
      }

      map.set(String(item.id_imagen), {
        id_imagen: String(item.id_imagen),
        id_modelo: String(item.id_modelo ?? ""),
        veredicto: String(item.veredicto ?? ""),
        comentario_medico: String(item.comentario_medico ?? ""),
      });
    }

    return map;
  } catch (error) {
    console.warn("No se pudieron leer resultados guardados. Se inicializara un estado limpio.", error);
    return new Map();
  }
}

function persistProgress() {
  const orderedResults = buildOrderedResults();
  localStorage.setItem(STORAGE_RESULTS_KEY, JSON.stringify(orderedResults));
  localStorage.setItem(STORAGE_INDEX_KEY, String(state.currentIndex));
}

function recordVote(vote) {
  const currentCase = state.cases[state.currentIndex];
  if (!currentCase) {
    return;
  }

  const result = {
    id_imagen: currentCase.id_imagen,
    id_modelo: currentCase.id_modelo,
    veredicto: vote,
    comentario_medico: dom.commentBox.value.trim(),
  };

  state.resultsByImageId.set(currentCase.id_imagen, result);
  state.draftCommentsByImageId.delete(currentCase.id_imagen);

  state.currentIndex += 1;
  dom.commentBox.value = "";
  persistProgress();
  render();
}

function goToPreviousCase() {
  if (!state.cases.length || state.currentIndex <= 0) {
    return;
  }

  syncCurrentCommentDraft();
  state.currentIndex -= 1;
  render();
}

function goToNextCase() {
  if (!state.cases.length || state.currentIndex >= state.cases.length) {
    return;
  }

  syncCurrentCommentDraft();
  if (state.currentIndex === state.cases.length - 1) {
    state.currentIndex = state.cases.length;
  } else {
    state.currentIndex += 1;
  }
  render();
}

function syncCurrentCommentDraft() {
  const currentCase = state.cases[state.currentIndex];
  if (!currentCase) {
    return;
  }

  const draft = dom.commentBox.value.trim();
  const saved = state.resultsByImageId.get(currentCase.id_imagen);
  if (saved) {
    saved.comentario_medico = draft;
    state.resultsByImageId.set(currentCase.id_imagen, saved);
    persistProgress();
    return;
  }

  if (draft) {
    state.draftCommentsByImageId.set(currentCase.id_imagen, draft);
  } else {
    state.draftCommentsByImageId.delete(currentCase.id_imagen);
  }
}

function render() {
  if (!state.cases.length) {
    updateProgress(0, 0);
    updateActionAvailability();
    return;
  }

  if (state.currentIndex >= state.cases.length) {
    showCompletion();
    return;
  }

  showCurrentCase();
}

function showCurrentCase() {
  const currentCase = state.cases[state.currentIndex];
  if (!currentCase) {
    moveToInitialLoaderState();
    setStatus("No hay casos disponibles. Carga data nuevamente.", true);
    return;
  }

  const answeredCount = state.resultsByImageId.size;

  dom.evaluationView.classList.remove("hidden");
  dom.actionPanel.classList.remove("hidden");
  dom.completionView.classList.add("hidden");
  hideLocalLoader();

  updateProgress(state.currentIndex + 1, state.cases.length);

  dom.trueClass.textContent = String(currentCase.ground_truth_class ?? "No especificado");
  dom.aiExplanation.textContent = String(currentCase.clinical_justification ?? "Sin justificacion.");

  const existing = state.resultsByImageId.get(currentCase.id_imagen);
  const draftComment = state.draftCommentsByImageId.get(currentCase.id_imagen) ?? "";
  dom.commentBox.value = existing ? existing.comentario_medico : draftComment;

  if (existing?.veredicto === "Match") {
    dom.currentVoteBadge.textContent = "👍 Match";
    dom.currentVoteBadge.classList.remove("hidden", "no-match");
    dom.currentVoteBadge.classList.add("match");
  } else if (existing?.veredicto === "No Match") {
    dom.currentVoteBadge.textContent = "👎 No Match";
    dom.currentVoteBadge.classList.remove("hidden", "match");
    dom.currentVoteBadge.classList.add("no-match");
  } else {
    dom.currentVoteBadge.textContent = "";
    dom.currentVoteBadge.classList.add("hidden");
    dom.currentVoteBadge.classList.remove("match", "no-match");
  }

  dom.caseImage.classList.remove("is-ready");
  dom.caseImage.src = String(currentCase.image_path ?? "");
  dom.caseImage.alt = `Imagen clinica del caso ${currentCase.id_imagen}`;
  dom.caseImage.onload = () => {
    dom.caseImage.classList.add("is-ready");
  };
  dom.caseImage.onerror = () => {
    handleMissingDatasetAssets(currentCase.image_path);
  };

  updateActionAvailability();
  renderGallery();
  setStatus(`Respuestas guardadas: ${answeredCount} de ${state.cases.length}.`);
}

function showCompletion() {
  updateProgress(state.cases.length, state.cases.length);
  dom.evaluationView.classList.add("hidden");
  dom.actionPanel.classList.add("hidden");
  hideLocalLoader();
  dom.completionView.classList.remove("hidden");
  renderCompletionSummary();
  updateActionAvailability();
  renderGallery();
  setStatus("Evaluacion finalizada. Exporta el CSV para cerrar el proceso.");
}

function handleMissingDatasetAssets(imagePath) {
  clearAllRuntimeData();
  showLocalLoader();
  setStatus(
    `La data actual parece incompleta o borrada (imagen no encontrada: ${String(imagePath ?? "")}). Carga un cases.json nuevo.`,
    true
  );
}

function renderCompletionSummary() {
  if (!dom.summaryMatch) {
    return;
  }

  const total = state.cases.length;
  let match = 0;
  let noMatch = 0;
  let comments = 0;

  for (const row of state.resultsByImageId.values()) {
    if (row.veredicto === "Match") {
      match += 1;
    } else if (row.veredicto === "No Match") {
      noMatch += 1;
    }

    if (String(row.comentario_medico ?? "").trim()) {
      comments += 1;
    }
  }

  const pending = Math.max(0, total - (match + noMatch));
  dom.summaryMatch.textContent = String(match);
  dom.summaryNoMatch.textContent = String(noMatch);
  dom.summaryPending.textContent = String(pending);
  dom.summaryComments.textContent = String(comments);
}

function openGallery() {
  if (!state.cases.length) {
    return;
  }
  syncCurrentCommentDraft();
  closeDataManager();
  renderGallery();
  dom.galleryModal.classList.remove("hidden");
}

function closeGallery() {
  dom.galleryModal.classList.add("hidden");
}

function openDataManager() {
  closeGallery();
  dom.dataManagerModal.classList.remove("hidden");
}

function closeDataManager() {
  dom.dataManagerModal.classList.add("hidden");
}

function renderGallery() {
  if (!dom.galleryGrid) {
    return;
  }

  updateGalleryLegendUI();
  dom.galleryGrid.innerHTML = "";

  for (let index = 0; index < state.cases.length; index += 1) {
    const item = state.cases[index];
    const caseState = getCaseGalleryState(item);

    if (state.galleryFilters.status && caseState.statusClass !== state.galleryFilters.status) {
      continue;
    }

    if (state.galleryFilters.withComment && !caseState.hasComment) {
      continue;
    }

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `gallery-item${index === state.currentIndex ? " current" : ""}`;
    btn.setAttribute("aria-label", `Ir al caso ${index + 1} (${caseState.statusLabel})`);
    btn.addEventListener("click", () => jumpToCase(index));

    btn.innerHTML = `
      <img class="gallery-thumb" src="${escapeHtml(item.image_path)}" alt="Miniatura caso ${index + 1}" loading="lazy" />
      <div class="gallery-meta">
        <div class="gallery-title">Caso ${index + 1}: ${escapeHtml(item.id_imagen)}</div>
        <div class="gallery-flags">
          <span class="gallery-status ${caseState.statusClass}" title="${caseState.statusLabel}">${caseState.statusIcon}<span class="status-text">${caseState.statusLabel}</span></span>
          ${caseState.hasComment ? '<span class="gallery-comment-flag" title="Con comentario" aria-label="Con comentario">💬</span>' : ''}
        </div>
      </div>
    `;

    dom.galleryGrid.appendChild(btn);
  }
}

function getCaseGalleryState(item) {
  const saved = state.resultsByImageId.get(item.id_imagen);
  const draftComment = state.draftCommentsByImageId.get(item.id_imagen) ?? "";
  const hasComment = Boolean((saved?.comentario_medico || draftComment || "").trim());

  if (saved?.veredicto === "Match") {
    return {
      statusClass: "match",
      statusLabel: "Match",
      statusIcon: "👍",
      hasComment,
    };
  }

  if (saved?.veredicto === "No Match") {
    return {
      statusClass: "no-match",
      statusLabel: "No Match",
      statusIcon: "👎",
      hasComment,
    };
  }

  return {
    statusClass: "pending",
    statusLabel: "Pendiente",
    statusIcon: "•",
    hasComment,
  };
}

function toggleGalleryStatusFilter(targetStatus) {
  if (state.galleryFilters.status === targetStatus) {
    state.galleryFilters.status = null;
  } else {
    state.galleryFilters.status = targetStatus;
  }
  renderGallery();
}

function toggleGalleryCommentFilter() {
  state.galleryFilters.withComment = !state.galleryFilters.withComment;
  renderGallery();
}

function updateGalleryLegendUI() {
  if (!dom.filterMatch || !dom.filterNoMatch || !dom.filterPending || !dom.filterComment) {
    return;
  }

  const activeStatus = state.galleryFilters.status;
  setFilterChipState(dom.filterMatch, activeStatus === "match");
  setFilterChipState(dom.filterNoMatch, activeStatus === "no-match");
  setFilterChipState(dom.filterPending, activeStatus === "pending");
  setFilterChipState(dom.filterComment, state.galleryFilters.withComment);
}

function setFilterChipState(element, isActive) {
  element.classList.toggle("active", Boolean(isActive));
  element.setAttribute("aria-pressed", isActive ? "true" : "false");
}

function jumpToCase(index) {
  if (!Number.isInteger(index) || index < 0 || index >= state.cases.length) {
    return;
  }

  syncCurrentCommentDraft();
  state.currentIndex = index;
  closeGallery();
  render();
}

function updateProgress(current, total) {
  dom.progressText.textContent = `Caso ${current} de ${total}`;
  const percentage = total > 0 ? Math.round((current / total) * 100) : 0;
  dom.progressBar.style.width = `${percentage}%`;
  dom.progressBar.parentElement.setAttribute("aria-valuenow", String(percentage));
}

function buildOrderedResults() {
  const list = [];

  for (const item of state.cases) {
    const saved = state.resultsByImageId.get(item.id_imagen);
    if (!saved) {
      continue;
    }

    list.push({
      id_imagen: saved.id_imagen,
      tipo_polipo: item.ground_truth_class,
      id_modelo: saved.id_modelo,
      veredicto: saved.veredicto,
      comentario_medico: saved.comentario_medico,
    });
  }

  return list;
}

function exportCsv() {
  const rows = buildOrderedResults();
  if (!rows.length) {
    setStatus("No hay resultados para exportar.", true);
    return;
  }

  const header = ["id_imagen", "tipo_polipo", "id_modelo", "veredicto", "comentario_medico"];
  const lines = [header.join(",")];

  for (const row of rows) {
    lines.push(
      [
        toCsvCell(row.id_imagen),
        toCsvCell(row.tipo_polipo),
        toCsvCell(row.id_modelo),
        toCsvCell(row.veredicto),
        toCsvCell(row.comentario_medico),
      ].join(",")
    );
  }

  const csvContent = lines.join("\n");
  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);

  const link = document.createElement("a");
  link.href = url;
  link.download = "evaluacion_clinica_resultados.csv";
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);

  setStatus("CSV descargado correctamente.");
  updateActionAvailability();
}

function toCsvCell(value) {
  const text = String(value ?? "");
  return `"${text.replaceAll('"', '""')}"`;
}

function setStatus(message, isError = false) {
  dom.statusLine.textContent = message;
  dom.statusLine.style.color = isError ? "#a93333" : "#36566a";
}

function showLocalLoader() {
  dom.evaluationView.classList.add("hidden");
  dom.actionPanel.classList.add("hidden");
  dom.completionView.classList.add("hidden");
  dom.localLoader.classList.remove("hidden");
  updateProgress(0, 0);
  closeGallery();
  closeDataManager();
  updateActionAvailability();
}

function hideLocalLoader() {
  dom.localLoader.classList.add("hidden");
}

function readCachedCases() {
  const raw = localStorage.getItem(STORAGE_CASES_CACHE_KEY);
  if (!raw) {
    return [];
  }

  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed.map(normalizeCase).filter(Boolean);
  } catch {
    return [];
  }
}

function saveCasesCache(casesList) {
  localStorage.setItem(STORAGE_CASES_CACHE_KEY, JSON.stringify(casesList));
}

function clearAllRuntimeData() {
  state.cases = [];
  state.currentIndex = 0;
  state.resultsByImageId = new Map();
  state.draftCommentsByImageId = new Map();

  localStorage.removeItem(STORAGE_CASES_CACHE_KEY);
  localStorage.removeItem(STORAGE_RESULTS_KEY);
  localStorage.removeItem(STORAGE_INDEX_KEY);
}

function moveToInitialLoaderState() {
  clearAllRuntimeData();
  closeGallery();
  closeDataManager();
  showLocalLoader();
  updateProgress(0, 0);
  updateActionAvailability();
}

function updateActionAvailability() {
  const hasActiveCase = state.cases.length > 0 && state.currentIndex < state.cases.length;
  dom.btnMatch.disabled = !hasActiveCase;
  dom.btnNoMatch.disabled = !hasActiveCase;
  dom.btnPrev.disabled = !state.cases.length || state.currentIndex <= 0;
  dom.btnNext.disabled = !state.cases.length || state.currentIndex >= state.cases.length;
  dom.btnOpenGallery.disabled = !state.cases.length;

  const hasRows = buildOrderedResults().length > 0;
  dom.btnExportQuick.disabled = !hasRows;
  dom.btnExportFinal.disabled = !hasRows;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

async function onCasesFileSelected(event) {
  const file = event.target.files && event.target.files[0];
  if (!file) {
    return;
  }

  try {
    const text = await file.text();
    const payload = JSON.parse(text);
    if (!Array.isArray(payload)) {
      throw new Error("El archivo seleccionado no contiene un array JSON valido.");
    }

    const normalized = payload.map(normalizeCase).filter(Boolean);
    if (!normalized.length) {
      throw new Error("No hay casos validos en el archivo seleccionado.");
    }

    applyImportedCases(normalized, file.name);
  } catch (error) {
    setStatus("No se pudo leer el archivo seleccionado. Verifica el formato JSON.", true);
    console.error(error);
  } finally {
    if (event.target) {
      event.target.value = "";
    }
  }
}

function clearEvaluationProgress() {
  state.resultsByImageId = new Map();
  state.draftCommentsByImageId = new Map();
  state.currentIndex = 0;
  localStorage.removeItem(STORAGE_RESULTS_KEY);
  localStorage.removeItem(STORAGE_INDEX_KEY);
}

function applyImportedCases(casesList, fileName = "") {
  state.cases = casesList;
  saveCasesCache(state.cases);
  clearEvaluationProgress();
  hideLocalLoader();
  closeDataManager();
  render();
  const nameHint = fileName ? ` (${fileName})` : "";
  setStatus(`Data cargada correctamente${nameHint}. Progreso reiniciado para este dataset.`);
}

function resetCurrentDataFromZero() {
  moveToInitialLoaderState();
  setStatus("Estado local borrado desde cero. Carga un nuevo cases.json para continuar.");
}

function normalizeCase(rawCase) {
  if (!rawCase || typeof rawCase !== "object") {
    return null;
  }

  const idImagen = String(rawCase.id_imagen ?? rawCase.id ?? "").trim();
  const imagePath = String(rawCase.image_path ?? rawCase.image_file ?? "").trim();
  const gtClass = String(rawCase.ground_truth_class ?? rawCase.true_class ?? "").trim();
  const justification = String(rawCase.clinical_justification ?? rawCase.ai_explanation ?? "").trim();
  const idModelo = String(rawCase.id_modelo ?? "").trim();

  if (!idImagen || !imagePath) {
    return null;
  }

  return {
    id_imagen: idImagen,
    image_path: imagePath,
    ground_truth_class: gtClass,
    id_modelo: idModelo,
    clinical_justification: justification,
  };
}
