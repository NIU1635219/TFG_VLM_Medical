const STORAGE_RESULTS_KEY = "clinical_eval_results_v3";
const STORAGE_INDEX_KEY = "clinical_eval_current_index_v3";
const STORAGE_CASES_CACHE_KEY = "clinical_eval_cases_cache_v3";
const STORAGE_CASES_CACHE_LEGACY_KEY = "clinical_eval_cases_cache_v2";
const STORAGE_THEME_KEY = "clinical_eval_theme_v1";

const state = {
  cases: [],
  currentIndex: 0,
  resultsByImageId: new Map(),
  draftCommentsByImageId: new Map(),
  datasetLibrary: [],
  activeDatasetId: "",
  activeDatasetLabel: "",
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
  aiPredictedClass: document.getElementById("aiPredictedClass"),
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
  datasetSelector: document.getElementById("datasetSelector"),
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
    dom.btnResetCurrentData.addEventListener("click", deleteActiveDataset);
  }
  dom.casesFileInput.addEventListener("change", onCasesFileSelected);
  dom.casesFileInputManager.addEventListener("change", onCasesFileSelected);
  if (dom.datasetSelector) {
    dom.datasetSelector.addEventListener("change", onDatasetSelected);
  }
  document.addEventListener("keydown", onGlobalKeydown);
  updateActionAvailability();
}

function applyStoredTheme() {
  const savedTheme = localStorage.getItem(STORAGE_THEME_KEY);
  const theme = savedTheme === "dark" ? "dark" : "light";
  document.body.setAttribute("data-theme", theme);
  document.documentElement.style.colorScheme = theme;
  dom.btnTheme.textContent = theme === "dark" ? "☀️" : "🌙";
  dom.btnTheme.title = theme === "dark" ? "Cambiar a modo claro" : "Cambiar a modo oscuro";
}

function toggleTheme() {
  const current = document.body.getAttribute("data-theme") === "dark" ? "dark" : "light";
  const next = current === "dark" ? "light" : "dark";
  document.body.setAttribute("data-theme", next);
  document.documentElement.style.colorScheme = next;
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

function createEmptyLibrary() {
  return {
    activeDatasetId: "",
    datasets: [],
  };
}

function buildDatasetId(sourceName = "dataset") {
  const slug = String(sourceName)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 24) || "dataset";
  return `${slug}-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
}

function formatDatasetTimestamp(timestamp = new Date()) {
  const date = timestamp instanceof Date ? timestamp : new Date(timestamp);
  if (Number.isNaN(date.getTime())) {
    return "sin fecha";
  }

  const parts = new Intl.DateTimeFormat("es-ES", {
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).formatToParts(date);

  const lookup = Object.fromEntries(parts.map((part) => [part.type, part.value]));
  return `${lookup.day}/${lookup.month} ${lookup.hour}:${lookup.minute}`;
}

function buildDatasetLabel(sourceName = "cases.json", caseCount = 0, importedAt = new Date()) {
  const safeName = String(sourceName).trim() || "cases.json";
  return `${safeName} · ${caseCount} casos · ${formatDatasetTimestamp(importedAt)}`;
}

function formatDatasetSummary(dataset) {
  if (!dataset) {
    return "dataset desconocido";
  }

  return `${dataset.label || dataset.sourceName || "dataset"}`;
}

function formatDatasetOptionLabel(dataset, index = 0) {
  return `${index + 1}. ${dataset.label || dataset.sourceName || "dataset"}`;
}

function normalizeDatasetRecord(rawDataset, fallbackIndex = 0) {
  if (!rawDataset || typeof rawDataset !== "object") {
    return null;
  }

  const cases = Array.isArray(rawDataset.cases) ? rawDataset.cases.map(normalizeCase).filter(Boolean) : [];
  const labelBase = String(rawDataset.label ?? rawDataset.sourceName ?? `Case ${fallbackIndex + 1}`).trim();
  const sourceName = String(rawDataset.sourceName ?? rawDataset.fileName ?? labelBase ?? "cases.json").trim() || "cases.json";
  const importedAt = String(rawDataset.importedAt ?? new Date().toISOString());
  const id = String(rawDataset.id ?? buildDatasetId(sourceName)).trim() || buildDatasetId(sourceName);
  const currentIndexRaw = Number.parseInt(String(rawDataset.currentIndex ?? 0), 10);
  const currentIndex = Number.isFinite(currentIndexRaw) ? Math.min(Math.max(currentIndexRaw, 0), cases.length) : 0;
  const results = Array.isArray(rawDataset.results) ? rawDataset.results : [];
  const draftComments = Array.isArray(rawDataset.draftComments) ? rawDataset.draftComments : [];

  return {
    id,
    label: labelBase || buildDatasetLabel(sourceName, cases.length, importedAt),
    sourceName,
    importedAt,
    cases,
    currentIndex,
    results,
    draftComments,
  };
}

function normalizeLegacyLibrary(casesArray) {
  const normalizedCases = Array.isArray(casesArray) ? casesArray.map(normalizeCase).filter(Boolean) : [];
  const legacyResults = readLegacyResultsFromStorage();
  const legacyIndex = readLegacyIndexFromStorage();
  const dataset = {
    id: buildDatasetId("legacy-cases"),
    label: buildDatasetLabel("cases.json", normalizedCases.length, new Date()),
    sourceName: "cases.json",
    importedAt: new Date().toISOString(),
    cases: normalizedCases,
    currentIndex: legacyIndex,
    results: legacyResults,
    draftComments: [],
  };

  return {
    activeDatasetId: dataset.id,
    datasets: [dataset],
  };
}

function normalizeStoredLibrary(rawLibrary) {
  if (Array.isArray(rawLibrary)) {
    return normalizeLegacyLibrary(rawLibrary);
  }

  if (!rawLibrary || typeof rawLibrary !== "object") {
    return createEmptyLibrary();
  }

  const datasets = Array.isArray(rawLibrary.datasets)
    ? rawLibrary.datasets.map((dataset, index) => normalizeDatasetRecord(dataset, index)).filter(Boolean)
    : [];

  const activeDatasetIdRaw = String(rawLibrary.activeDatasetId ?? "").trim();
  const activeDatasetId = datasets.some((dataset) => dataset.id === activeDatasetIdRaw)
    ? activeDatasetIdRaw
    : datasets[0]?.id ?? "";

  return {
    activeDatasetId,
    datasets,
  };
}

function readLegacyIndexFromStorage() {
  const savedIndexRaw = localStorage.getItem(STORAGE_INDEX_KEY);
  const savedIndex = Number.parseInt(savedIndexRaw ?? "0", 10);
  return Number.isFinite(savedIndex) && savedIndex >= 0 ? savedIndex : 0;
}

function readLegacyResultsFromStorage() {
  const raw = localStorage.getItem(STORAGE_RESULTS_KEY);
  if (!raw) {
    return [];
  }

  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function readCasesLibrary() {
  const raw = localStorage.getItem(STORAGE_CASES_CACHE_KEY);
  if (raw) {
    try {
      return normalizeStoredLibrary(JSON.parse(raw));
    } catch (error) {
      console.warn("No se pudo leer la biblioteca de cases guardada. Se intentara migrar o reiniciar.", error);
    }
  }

  const legacyRaw = localStorage.getItem(STORAGE_CASES_CACHE_LEGACY_KEY);
  if (legacyRaw) {
    try {
      return normalizeStoredLibrary(JSON.parse(legacyRaw));
    } catch (error) {
      console.warn("No se pudo leer la cache legacy de cases. Se ignorara.", error);
    }
  }

  return createEmptyLibrary();
}

function saveCasesLibrary(library) {
  localStorage.setItem(STORAGE_CASES_CACHE_KEY, JSON.stringify(normalizeStoredLibrary(library)));
}

function getActiveDatasetFromLibrary(library = readCasesLibrary()) {
  if (!library.datasets.length) {
    return null;
  }

  return library.datasets.find((dataset) => dataset.id === library.activeDatasetId) ?? library.datasets[0] ?? null;
}

function loadDatasetIntoState(dataset) {
  const normalizedDataset = normalizeDatasetRecord(dataset);
  if (!normalizedDataset) {
    return false;
  }

  state.cases = normalizedDataset.cases;
  state.currentIndex = normalizedDataset.currentIndex;
  state.resultsByImageId = readStoredResults(normalizedDataset.results);
  state.draftCommentsByImageId = readStoredDrafts(normalizedDataset.draftComments);
  state.activeDatasetId = normalizedDataset.id;
  state.activeDatasetLabel = normalizedDataset.label;
  state.datasetLibrary = state.datasetLibrary.map((item) => (item.id === normalizedDataset.id ? normalizedDataset : item));
  renderDatasetSelector();
  return true;
}

function renderDatasetSelector() {
  if (!dom.datasetSelector) {
    return;
  }

  const datasets = Array.isArray(state.datasetLibrary) ? state.datasetLibrary : [];
  dom.datasetSelector.innerHTML = "";

  if (!datasets.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "Sin cases guardados";
    dom.datasetSelector.appendChild(option);
    dom.datasetSelector.disabled = true;
    return;
  }

  dom.datasetSelector.disabled = false;
  datasets.forEach((dataset, index) => {
    const option = document.createElement("option");
    option.value = dataset.id;
    option.textContent = formatDatasetOptionLabel(dataset, index);
    if (dataset.id === state.activeDatasetId) {
      option.selected = true;
    }
    dom.datasetSelector.appendChild(option);
  });

  if (dom.datasetSelector.value !== state.activeDatasetId && state.activeDatasetId) {
    dom.datasetSelector.value = state.activeDatasetId;
  }
}

function onDatasetSelected(event) {
  const selectedDatasetId = String(event.target.value ?? "").trim();
  if (!selectedDatasetId || selectedDatasetId === state.activeDatasetId) {
    return;
  }

  switchActiveDataset(selectedDatasetId);
}

function switchActiveDataset(datasetId) {
  const library = readCasesLibrary();
  const currentDataset = library.datasets.find((dataset) => dataset.id === state.activeDatasetId);
  if (currentDataset) {
    currentDataset.cases = state.cases.map((item) => ({ ...item }));
    currentDataset.currentIndex = state.currentIndex;
    currentDataset.results = buildOrderedResults();
    currentDataset.draftComments = Array.from(state.draftCommentsByImageId.entries()).map(([caseKey, comment]) => ({ case_key: caseKey, comment }));
    currentDataset.label = state.activeDatasetLabel || currentDataset.label;
  }

  const targetDataset = library.datasets.find((dataset) => dataset.id === datasetId);
  if (!targetDataset) {
    renderDatasetSelector();
    return;
  }

  library.activeDatasetId = targetDataset.id;
  saveCasesLibrary(library);
  state.datasetLibrary = library.datasets;
  loadDatasetIntoState(targetDataset);
  hideLocalLoader();
  render();
  setStatus(`Caso activo cambiado a ${formatDatasetSummary(targetDataset)}.`);
}

function deleteActiveDataset() {
  const library = readCasesLibrary();
  if (!library.datasets.length || !state.activeDatasetId) {
    setStatus("No hay case activo para borrar.", true);
    return;
  }

  const removedIndex = library.datasets.findIndex((dataset) => dataset.id === state.activeDatasetId);
  if (removedIndex === -1) {
    setStatus("No se encontró el case activo en la lista guardada.", true);
    return;
  }

  const [removedDataset] = library.datasets.splice(removedIndex, 1);
  if (library.datasets.length) {
    const nextDataset = library.datasets[Math.min(removedIndex, library.datasets.length - 1)];
    library.activeDatasetId = nextDataset.id;
    saveCasesLibrary(library);
    state.datasetLibrary = library.datasets;
    loadDatasetIntoState(nextDataset);
    hideLocalLoader();
    closeDataManager();
    render();
    setStatus(`Case eliminado: ${removedDataset.label}. Ahora queda activo ${formatDatasetSummary(nextDataset)}.`);
    return;
  }

  clearAllRuntimeData();
  state.datasetLibrary = [];
  showLocalLoader();
  renderDatasetSelector();
  updateProgress(0, 0);
  closeDataManager();
  setStatus(`Case eliminado: ${removedDataset.label}. No quedan cases guardados.`);
}

function createDatasetRecord(casesList, sourceName = "cases.json") {
  const normalizedCases = casesList.map(normalizeCase).filter(Boolean);
  const importedAt = new Date().toISOString();
  const displayName = buildDatasetLabel(sourceName, normalizedCases.length, importedAt);

  return {
    id: buildDatasetId(sourceName),
    label: displayName,
    sourceName: String(sourceName || "cases.json").trim() || "cases.json",
    importedAt,
    cases: normalizedCases,
    currentIndex: 0,
    results: [],
    draftComments: [],
  };
}

async function loadCases() {
  const library = readCasesLibrary();
  state.datasetLibrary = library.datasets;

  if (library.datasets.length) {
    const activeDataset = getActiveDatasetFromLibrary(library) ?? library.datasets[0];
    if (activeDataset) {
      library.activeDatasetId = activeDataset.id;
      saveCasesLibrary(library);
      state.datasetLibrary = library.datasets;
      loadDatasetIntoState(activeDataset);
      renderDatasetSelector();
      restoreProgress();
      hideLocalLoader();
      render();
      setStatus(`Dataset activo cargado: ${formatDatasetSummary(activeDataset)}.`);
      return;
    }
  }

  resetRuntimeState();
  renderDatasetSelector();
  showLocalLoader();
  updateProgress(0, 0);
  setStatus("Carga manual requerida. Selecciona un cases.json para continuar.");
}

function restoreProgress() {
  if (!state.activeDatasetId) {
    return;
  }

  const currentDataset = state.datasetLibrary.find((dataset) => dataset.id === state.activeDatasetId);
  if (!currentDataset) {
    return;
  }

  state.resultsByImageId = readStoredResults(currentDataset.results);
  state.draftCommentsByImageId = readStoredDrafts(currentDataset.draftComments);

  const savedIndex = Number.parseInt(String(currentDataset.currentIndex ?? "0"), 10);
  const maxIndex = state.cases.length;
  if (Number.isFinite(savedIndex) && savedIndex >= 0 && savedIndex <= maxIndex) {
    state.currentIndex = savedIndex;
  }

  if (state.currentIndex > 0) {
    setStatus(`Progreso recuperado: continuas en el caso ${state.currentIndex + 1}.`);
  }
}

function readStoredResults(resultsSource = []) {
  if (!Array.isArray(resultsSource) || !resultsSource.length) {
    return new Map();
  }

  const map = new Map();
  for (const item of resultsSource) {
    if (!item || typeof item.id_imagen === "undefined") {
      continue;
    }

    const caseKey = String(item.case_key ?? buildCaseKey(item) ?? "");
    if (!caseKey) {
      continue;
    }

    map.set(caseKey, {
      id_imagen: String(item.id_imagen),
      id_modelo_oculto: String(item.id_modelo_oculto ?? item.id_modelo ?? ""),
      veredicto: String(item.veredicto ?? ""),
      comentario_medico: String(item.comentario_medico ?? ""),
      case_key: caseKey,
    });
  }

  return map;
}

function readStoredDrafts(draftSource = []) {
  if (!Array.isArray(draftSource) || !draftSource.length) {
    return new Map();
  }

  const map = new Map();
  for (const entry of draftSource) {
    if (!entry) {
      continue;
    }

    let caseKey = "";
    let comment = "";
    if (Array.isArray(entry)) {
      if (entry.length < 2) {
        continue;
      }
      caseKey = String(entry[0] ?? "").trim();
      comment = String(entry[1] ?? "").trim();
    } else if (typeof entry === "object") {
      caseKey = String(entry.case_key ?? entry.caseKey ?? "").trim();
      comment = String(entry.comment ?? entry.comentario_medico ?? "").trim();
    }

    if (!caseKey || !comment) {
      continue;
    }

    map.set(caseKey, comment);
  }

  return map;
}

function resetRuntimeState() {
  state.cases = [];
  state.currentIndex = 0;
  state.resultsByImageId = new Map();
  state.draftCommentsByImageId = new Map();
  state.activeDatasetId = "";
  state.activeDatasetLabel = "";
}

function persistProgress() {
  persistActiveDatasetState();
}

function persistActiveDatasetState() {
  if (!state.activeDatasetId) {
    return;
  }

  const library = readCasesLibrary();
  const dataset = library.datasets.find((item) => item.id === state.activeDatasetId);
  if (!dataset) {
    return;
  }

  dataset.cases = state.cases.map((item) => ({ ...item }));
  dataset.currentIndex = state.currentIndex;
  dataset.results = buildOrderedResults();
  dataset.draftComments = Array.from(state.draftCommentsByImageId.entries()).map(([caseKey, comment]) => ({ case_key: caseKey, comment }));
  dataset.label = state.activeDatasetLabel || dataset.label;

  library.activeDatasetId = state.activeDatasetId;
  library.datasets = library.datasets.map((item) => (item.id === dataset.id ? dataset : item));
  state.datasetLibrary = library.datasets;
  saveCasesLibrary(library);
  renderDatasetSelector();
}

function recordVote(vote) {
  const currentCase = state.cases[state.currentIndex];
  if (!currentCase) {
    return;
  }

  const caseKey = buildCaseKey(currentCase);
  const result = {
    id_imagen: currentCase.id_imagen,
    id_modelo_oculto: String(currentCase.id_modelo_oculto ?? ""),
    veredicto: vote,
    comentario_medico: dom.commentBox.value.trim(),
    case_key: caseKey,
  };

  state.resultsByImageId.set(caseKey, result);
  state.draftCommentsByImageId.delete(caseKey);

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
  persistProgress();
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
  persistProgress();
  render();
}

function syncCurrentCommentDraft() {
  const currentCase = state.cases[state.currentIndex];
  if (!currentCase) {
    return;
  }

  const caseKey = buildCaseKey(currentCase);

  const draft = dom.commentBox.value.trim();
  const saved = state.resultsByImageId.get(caseKey);
  if (saved) {
    saved.comentario_medico = draft;
    state.resultsByImageId.set(caseKey, saved);
    persistProgress();
    return;
  }

  if (draft) {
    state.draftCommentsByImageId.set(caseKey, draft);
  } else {
    state.draftCommentsByImageId.delete(caseKey);
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
  if (dom.aiPredictedClass) {
    dom.aiPredictedClass.textContent = String(currentCase.ai_predicted_class ?? "No especificado");
  }
  dom.aiExplanation.textContent = String(currentCase.clinical_justification ?? "Sin justificacion.");

  const caseKey = buildCaseKey(currentCase);
  const existing = state.resultsByImageId.get(caseKey);
  const draftComment = state.draftCommentsByImageId.get(caseKey) ?? "";
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
  dom.caseImage.classList.remove("is-ready");
  dom.caseImage.removeAttribute("src");
  setStatus(
    `La imagen del case activo no se pudo cargar (${String(imagePath ?? "")}). Usa el selector inferior para cambiar a otro case o importa el archivo correcto.`,
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
  const caseKey = buildCaseKey(item);
  const saved = state.resultsByImageId.get(caseKey);
  const draftComment = state.draftCommentsByImageId.get(caseKey) ?? "";
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
  persistProgress();
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
    const caseKey = buildCaseKey(item);
    const saved = state.resultsByImageId.get(caseKey);
    if (!saved) {
      continue;
    }

    list.push({
      id_imagen: saved.id_imagen,
      id_modelo_oculto: saved.id_modelo_oculto,
      veredicto: saved.veredicto,
      comentario_medico: saved.comentario_medico,
      case_key: caseKey,
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

  const header = ["id_imagen", "id_modelo_oculto", "veredicto", "comentario_medico"];
  const lines = [header.join(",")];

  for (const row of rows) {
    lines.push(
      [
        toCsvCell(row.id_imagen),
        toCsvCell(row.id_modelo_oculto),
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
  resetRuntimeState();
  closeGallery();
  closeDataManager();
  showLocalLoader();
  renderDatasetSelector();
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
  if (dom.datasetSelector) {
    dom.datasetSelector.disabled = !state.datasetLibrary.length;
  }

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

function applyImportedCases(casesList, fileName = "") {
  const library = readCasesLibrary();
  const dataset = createDatasetRecord(casesList, fileName || "cases.json");
  library.datasets.push(dataset);
  library.activeDatasetId = dataset.id;
  saveCasesLibrary(library);

  state.datasetLibrary = library.datasets;
  loadDatasetIntoState(dataset);
  renderDatasetSelector();
  hideLocalLoader();
  closeDataManager();
  render();

  const nameHint = fileName ? ` (${fileName})` : "";
  setStatus(`Data cargada correctamente${nameHint}. Se añadió a la lista de cases guardados.`);
}

function resetCurrentDataFromZero() {
  deleteActiveDataset();
}

function readCachedCases() {
  return readCasesLibrary().datasets.flatMap((dataset) => dataset.cases);
}

function saveCasesCache(casesList) {
  const library = readCasesLibrary();
  const dataset = createDatasetRecord(casesList, "cases.json");
  library.datasets = [dataset];
  library.activeDatasetId = dataset.id;
  saveCasesLibrary(library);
}

function clearAllRuntimeData() {
  state.cases = [];
  state.currentIndex = 0;
  state.resultsByImageId = new Map();
  state.draftCommentsByImageId = new Map();
  state.activeDatasetId = "";
  state.activeDatasetLabel = "";

  localStorage.removeItem(STORAGE_CASES_CACHE_KEY);
  localStorage.removeItem(STORAGE_CASES_CACHE_LEGACY_KEY);
  localStorage.removeItem(STORAGE_RESULTS_KEY);
  localStorage.removeItem(STORAGE_INDEX_KEY);
}

function normalizeCase(rawCase) {
  if (!rawCase || typeof rawCase !== "object") {
    return null;
  }

  const idImagen = String(rawCase.id_imagen ?? rawCase.id ?? "").trim();
  const imagePath = normalizeCaseImagePath(String(rawCase.image_path ?? rawCase.image_file ?? "").trim());
  const gtClass = String(rawCase.ground_truth_class ?? rawCase.true_class ?? "").trim();
  const aiPredictedClass = String(rawCase.ai_predicted_class ?? rawCase.predicted_class ?? rawCase.final_diagnosis_class ?? "").trim();
  const justification = String(rawCase.clinical_justification ?? rawCase.ai_explanation ?? rawCase.diagnostic_rationale ?? "").trim();
  const idModeloOculto = String(rawCase.id_modelo_oculto ?? rawCase.id_modelo ?? "").trim();

  if (!idImagen || !imagePath) {
    return null;
  }

  return {
    id_imagen: idImagen,
    image_path: imagePath,
    ground_truth_class: gtClass,
    ai_predicted_class: aiPredictedClass,
    id_modelo_oculto: idModeloOculto,
    clinical_justification: justification,
  };
}

function normalizeCaseImagePath(imagePath) {
  const cleaned = String(imagePath ?? "").trim().replaceAll("\\", "/");
  if (!cleaned) {
    return "";
  }

  if (cleaned.startsWith("assets/images/")) {
    return cleaned.replace("assets/images/", "images/");
  }

  if (cleaned.startsWith("data/images/")) {
    return cleaned.replace("data/images/", "images/");
  }

  return cleaned;
}

function buildCaseKey(item) {
  if (!item || typeof item !== "object") {
    return "";
  }

  const idImagen = String(item.id_imagen ?? "").trim();
  const modelId = String(item.id_modelo_oculto ?? item.id_modelo ?? "").trim();
  if (!idImagen || !modelId) {
    return "";
  }

  return `${idImagen}::${modelId}`;
}
