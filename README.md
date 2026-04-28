# TFG VLM Medical

Proyecto de TFG centrado en inferencia local con modelos VLM (Vision-Language Models) sobre imágenes médicas, usando el SDK oficial de LM Studio (`lmstudio-python`) y una herramienta de gestión interactiva en terminal.

## Objetivo

- Ejecutar inferencia VLM en local (privacidad y control total del entorno).
- Obtener respuestas estructuradas y validadas con Pydantic.
- Mantener un flujo reproducible para instalación, diagnóstico, gestión de modelos y testing.

## Estado actual (resumen)

- Backend de inferencia migrado a LM Studio SDK oficial (`lmstudio`).
- Flujo multimodal estable con `Chat + add_user_message(..., images=[...])`.
- Compatibilidad para distintas firmas de `respond()` (`config`, `temperature`, etc.).
- Respuesta estructurada con esquemas Pydantic dinámicos y variantes `WithReasoning`.
- Telemetría opcional por inferencia en `VLMLoader`: basada en `response.stats` del SDK y, cuando aporta datos, en `GET /v1/models` para recursos del modelo.
- Smoke test con 5 casos (`sample_01`...`sample_05`) y descarga automática de imágenes si faltan.
- Manager TUI modularizado y con tests de regresión.
- `UIKit` incluye API de tablas tipadas y reutilizables (`TableColumn`, `TableRow`, `TableCell`, `build_table_items`, `table_menu`) con anchos adaptativos al terminal.
- Motor de tablas TUI ampliado con celdas avanzadas (`rowspan`/`colspan`) y control de truncado por líneas (`max_cell_lines`) para mejorar legibilidad en terminales estrechos.
- `UIKit` añade `render_and_wait_responsive(...)` como helper común de espera reactiva, con repintado automático al redimensionar el terminal.
- Dashboard de escenarios de grounding (A/B/C/D/E/F/S) en vivo desde Setup Tests UI:
    - barra de progreso con clamp defensivo (`current` no excede `total`),
    - resumen de clase (match/mismatch),
    - línea de estado con bloque `Métricas` (Acc/IoU/Proximity) en tiempo real,
    - heatmap GT→Pred en tabla TUI con color por celda y gradiente continuo.
- Evaluación espacial dual en grounding: IoU como métrica base + `Proximity` como métrica complementaria.
- `Proximity` se calcula por imagen a partir de dos componentes:
    - `proximity_center_score`: cercanía entre centros de bbox (función gaussiana),
    - `proximity_size_score`: similitud relativa de tamaño (área).
- Score combinado `proximity_score` en rango [0,1] con pesos 0.6 (centros) y 0.4 (tamaño),
  reforzado por compuerta geométrica:
    - sin contacto entre cajas => score cercano a 0,
    - con contacto => penalización/bonificación según contención (intersección respecto a la caja menor).
- Registros JSONL de escenarios A/B/C/E/F ampliados con `proximity_score`, `proximity_center_score` y `proximity_size_score`.
- Resumen acumulado de escenarios ampliado con `avg_proximity`.
- Reporte Markdown de escenarios A/B/C/E/F ampliado con sección `Análisis Proximity` y 5 gráficos en `report_assets`.
- Reporte Markdown de Scenario S con KPIs de sycophancy (contradicción/obediencia/detección), desglose por clase GT y gráficos en `report_assets`.
- Schema Tester integrado en el manager: selección interactiva de modelo + esquema + inferencia por lotes.
- Batch Runner CLI con exportado incremental en JSONL compartido por manifiesto+schema.
- Batch Runner: enriquecimiento espacial opcional en JSONL para detecciones con `iou_score` (por bbox),
  `order_index` y `bbox_ordering` (izquierda->derecha, arriba->abajo).
- Experimento A/B de prompting integrado: comparación zero-shot vs prompt asistido con Ground Truth AD desde CLI y desde Setup Tests UI.
- Telemetry Probe integrado en el manager con progreso en vivo, cobertura de métricas y resumen final legible.
- Response Inspector integrado en el manager para inspeccionar campos reales del SDK con modo crudo o estructurado.
- UI/IO reforzada en `setup_ui_io.py`:
    - `ask_text(...)` como helper general de entrada reactiva en terminal.
    - `ask_choice(...)` como selector horizontal reutilizable para prompts interactivos.
    - `ask_user(...)` mantenido por compatibilidad y delegado sobre `ask_choice(...)`.
- Flujo de `BATCH RUNNER · MANIFEST SAMPLE SIZE` mejorado con preview estratificado en vivo:
    - reparto por clase mientras se escribe,
    - distribución total del dataset,
    - resumen `Total estimado / solicitado`,
    - repintado completo opcional para evitar duplicación visual/ghosting.
- Manifests de experimentos con formato compacto:
    - configuración global en cabecera (`run_schema_name`, `run_model_variants`, `run_iterations_per_image`, `run_seed`, `run_derived_from`),
    - filas de imágenes más limpias y sin repetir configuración por registro,
    - extracción estricta de configuración desde metadata de cabecera para evitar ambigüedades.
- Refactor modular de menús de modelos y tests:
    - extracción de pantallas a `src/utils/models_ui/` y `src/utils/tests_ui/`,
    - menor acoplamiento y mayor reutilización,
    - mantenimiento de compatibilidad funcional con pruebas de regresión.
- `batch_results` con metadatos canónicos `__batch_meta__` (v2):
    - formato sin claves legacy,
    - `model_ids`, `schema_names`, `input_sources`, `manifest_paths`.
- Resumen agregado `__batch_summary__` al final del JSONL:
    - min/max/media de métricas de latencia por modelo y global,
    - aciertos y accuracy por modelo y global (`correct/evaluated`).
- Reejecución de batch mejorada:
    - una única decisión al inicio (sí/no/selección múltiple de variantes),
    - selección múltiple real con `SPACE` por variante (`sin razonamiento` / `con razonamiento`),
    - barra de cola oculta cuando solo se ejecuta una variante.
- Gestión de manifiestos en Batch Runner:
    - opción para eliminar manifiestos existentes desde el selector,
    - confirmación extra que muestra explícitamente los JSONL compartidos a borrar,
    - limpieza conjunta del manifest y sus `batch_*.jsonl` vinculados.
- Notebook de análisis `02_zeroshot_analysis.ipynb` actualizado:
    - lectura robusta de `batch_results` en JSONL con soporte de `__batch_meta__` y `__batch_summary__`,
    - export de métricas/tablas en JSONL (sin CSV),
    - bundle ZIP con estructura controlada: `images/` plana + dos JSONL en raíz (batch + manifest).
- Catálogo de esquemas simplificado:
    - retirada de `PolypClassificationDetailed` y su variante `WithReasoning`.

## Novedades recientes (marzo-abril 2026)

- Estandarización de docstrings en estilo Google en módulos refactorizados.
- Corrección de imports tras mover helpers de LM Studio a `src/utils/models_ui/`.
- Consolidación del input reactivo en `src/utils/setup_ui_io.py` y reutilización en:
    - pantalla de selección de tamaño de subgrupo (Batch Runner),
    - pantalla de descarga manual por ID/URL (Model Manager).
- Validación continua con `pytest` en bloques focalizados y suite completa durante la refactorización.
- Batch Runner: metadata de salida unificado en `__batch_meta__` v2 y eliminación de campos legacy.
- Batch Runner: generación automática de `__batch_summary__` con métricas globales y por modelo (incluye accuracy).
- Batch Runner UI: estrategia de reejecución por dataset con selector múltiple de variantes por modelo.
- Manifests: metadata compacta con `run_model_variants` y `run_seed`, y soporte de derivación (`run_derived_from`).
- Batch Runner: registros por imagen incluyen `include_reasoning` y agregación de summary por variante.
- Batch Runner: borrado de manifiesto desde TUI con limpieza de outputs JSONL vinculados.
- Convención UI: pantallas finales estructuradas migradas al helper común `render_and_wait_responsive(...)` para espera reactiva con repintado por redimensionado (manifest/schema/smoke/telemetry/response inspector/batch summary).
- Setup Tests UI: nueva opción `Run A/B Prompting Experiment (AD)` con preview previo, confirmación y pantalla final reactiva.
- Setup Tests UI: selector `Select Grounding Scenario (A/B/C/D/E/F/S)` integrado con ejecución real para escenarios A, B, C, D, E, F y S.
- Grounding: nuevo Scenario F para reporte clínico asistido (clase GT inyectada + bbox + explicación clínica estructurada).
- Grounding: nuevo Scenario S para stress test de sycophancy con niveles de autoridad (level 1/2/3).
- Grounding scenarios: robustez de completitud/resume con skeleton JSONL (`pending`), upsert por `image_id` y detección de registros sin rellenar.
- Grounding scenarios: registros con `status=error/failed/fail` se consideran incompletos para `resume`.
- Dashboard live de grounding: heatmap clínico GT→Pred con porcentaje global y porcentaje por fila GT en cada celda.
- Dashboard live de grounding: leyenda compacta en una línea y panel reordenado (tabla heatmap arriba, ruta de salida JSONL debajo).
- Dashboard live de grounding: en Scenario S se renderiza heatmap de contradicción (`TRUE`/`FALSE`) por clase GT.
- Nuevo script `src/scripts/experiment_ab_text.py` para detección automática de variantes desde JSONL (`__batch_meta__`) y comparación cualitativa A/B sobre muestras AD estratificadas.
- Motor visual/TUI: tablas estáticas y dashboards migrados al render unificado de `table_menu(..., interactive=False, return_lines=True)` con mejor reparto de columnas y truncado controlado.
- Cobertura de pruebas ampliada para flujo A/B y renderizado reactivo de tablas en pantallas finales.
- Nuevo script `src/preprocessing/extract_gt_bboxes.py` para extraer bounding boxes Ground Truth desde máscaras binarias (`.tif/.tiff`) y normalizarlas a escala 0-1000 (`[xmin, ymin, xmax, ymax]`).
- Nuevo notebook `notebooks/03_ground_truth_bboxes.ipynb` para orquestar la ejecución del extractor y validar visualmente los BBoxes.
- Grounding: integración de evaluación espacial combinada `IoU + Proximity` para escenarios A/B/C/E/F.
- Grounding: persistencia de `proximity_score`, `proximity_center_score` y `proximity_size_score` por registro en JSONL.
- Grounding reportes: nueva sección `Análisis Proximity` con distribución, resumen por clase, comparación por acierto/fallo, boxplot y curva por umbrales.
- Grounding aggregation: resumen acumulado con `avg_proximity` además de `avg_iou`.
- Tests nuevos para proximidad geométrica: `tests/test_proximity_metrics.py` y `tests/test_grounding_report_proximity_metrics.py`.
- Proximity endurecido para reducir falsos positivos espaciales:
    - cajas sin contacto pasan a puntuaciones casi nulas,
    - contención completa/alta se premia frente a solapamientos parciales.
- Rehidratación de runs históricos soportada: se pueden recalcular campos `proximity_*` en `results.jsonl` y regenerar `results.md`/`report_assets` con la nueva lógica.

## Stack técnico

- Python >= 3.12
- Gestor de dependencias: `uv`
- Inferencia: LM Studio + `lmstudio-python`
- Telemetría REST: `requests`
- Librerías principales: `lmstudio`, `pydantic`, `pillow`, `opencv-python`, `pandas`, `matplotlib`, `pytest`, `tqdm`, `jupyter`, `numpy`, `rarfile`

Dependencias declaradas en `pyproject.toml`.

## Estructura del repositorio

```text
.
├── README.md                           # Documentación principal del proyecto y guía de uso.
├── pyproject.toml                      # Metadatos del proyecto y dependencias Python.
├── uv.lock                             # Lockfile reproducible del entorno gestionado con uv.
├── setup_env.py                        # Punto de entrada del manager TUI de instalación/tests/modelos.
├── setup.bat                           # Arranque rápido del manager en Windows.
├── setup.sh                            # Arranque rápido del manager en Linux/macOS.
├── data/
│   ├── raw/                            # Dataset original y particiones sin preprocesar.
│   │   ├── m_train/
│   │   ├── m_valid/
│   │   └── m_test/
│   ├── processed/                      # Imágenes recortadas/preprocesadas y CSV replicados.
│   │   ├── m_train/
│   │   ├── m_valid/
│   │   └── m_test/
│   └── smoke_test/                     # Imágenes pequeñas de validación end-to-end.
├── notebooks/
│   ├── 01_eda_polypsegm.ipynb          # EDA, extracción y comparativas visuales del dataset.
│   ├── 02_zeroshot_analysis.ipynb      # Análisis de batch_results zero-shot, métricas y export bundle JSONL/ZIP.
│   └── 03_ground_truth_bboxes.ipynb    # Orquestación del extractor de BBoxes GT y sanity check visual sobre máscaras.
├── src/
│   ├── inference/
│   │   ├── schemas.py                  # Esquemas Pydantic base y variantes WithReasoning.
│   │   └── vlm_runner.py               # Carga VLM, inferencia multimodal, compatibilidad SDK y telemetría.
│   ├── preprocessing/
│   │   ├── preprocess.py               # CLI de recorte de bordes negros y copia de CSV.
│   │   └── extract_gt_bboxes.py        # CLI para extraer BBoxes GT desde máscaras y exportar CSV normalizado (escala 1000).
│   ├── scripts/
│   │   ├── batch_runner.py             # Orquestador masivo con exportado incremental en JSONL compartido.
│   │   ├── experiment_ab_text.py       # Experimento A/B zero-shot vs asistido con Ground Truth AD y reporte Markdown.
│   │   ├── grounding_experiments/      # Escenarios de grounding A/B/C/D/E/F/S y módulos de reporte/agregación.
│   │   │   ├── run_scenario_A.py       # Scenario A: zero-shot (bbox + clase) con reporte Markdown.
│   │   │   ├── run_scenario_B.py       # Scenario B: asistido por clase GT (lookup split) con reporte Markdown.
│   │   │   ├── run_scenario_C.py       # Scenario C: bbox GT superpuesto sobre imagen completa para guiar la localización.
│   │   │   ├── run_scenario_D.py       # Scenario D: clasificación focalizada con recorte ROI y schema sin bbox/IoU.
│   │   │   ├── run_scenario_E.py       # Scenario E: combinación de B+C (clase asistida + bbox GT dibujada sobre imagen completa).
│   │   │   ├── run_scenario_F.py       # Scenario F: reporte clínico asistido (bbox + explicabilidad clínica con diagnóstico GT inyectado).
│   │   │   ├── run_scenario_S.py       # Scenario S: stress test de sycophancy con prompts de autoridad por nivel (1/2/3).
│   │   │   ├── runner_core.py          # Fachada pública compartida usada por escenarios y UI (IoU + Proximity safe).
│   │   │   ├── report_common.py        # Helpers comunes (normalización de clase, extracción de clase predicha, bbox [xmin,ymin,xmax,ymax]).
│   │   │   ├── report_aggregation.py   # Persistencia y agregación JSONL (meta/summary/resume, avg_iou + avg_proximity).
│   │   │   ├── report_serialization.py # Serialización compacta de registros por imagen (incluye campos proximity_*).
│   │   │   ├── report_markdown.py      # Construcción de records y generación de informe Markdown (secciones IoU y Proximity).
│   │   │   ├── report_visuals.py       # Generación de imágenes anotadas, heatmap y gráficos IoU/Proximity.
│   │   │   ├── report_metrics.py       # Accuracy, macro-F1, confusión + agregaciones IoU/Proximity.
│   │   │   └── report_narrative.py     # Explicaciones narrativas por clase y top de errores.
│   │   ├── poc_bbox.py                 # PoC de visual grounding con soporte multi-bbox y reporter para UI reactiva.
│   │   ├── test_inference.py           # Smoke test CLI con descarga automática de imágenes de muestra.
│   │   ├── test_response_inspector.py  # Inspector CLI de respuestas reales del SDK LM Studio.
│   │   ├── test_schema.py              # Lógica batch reutilizable para el Schema Tester interactivo.
│   │   └── test_telemetry.py           # Medición CLI de TTFT/TPS y resumen de latencias.
│   └── utils/
│       ├── app_config.py               # Configuración estática, librerías requeridas y registro de modelos.
│       ├── menu_kit.py                 # UIKit + AppContext: abstracción de tablas, menús, render y helper común de espera reactiva.
│       ├── setup_diagnostics.py        # Diagnóstico del entorno, dependencias y smart-fix.
│       ├── setup_install_flow.py       # Flujo principal de instalación y navegación del manager.
│       ├── setup_menu_engine.py        # Motor de menús interactivos y estado del cursor.
│       ├── setup_reinstall_ui.py       # UI de reinstalación selectiva de librerías/herramientas.
│       ├── setup_models_ui.py          # Orquestador de UI para gestión de modelos LM Studio.
│       ├── setup_tests_ui.py           # Orquestador de UI para tests y utilidades de evaluación.
│       ├── setup_ui_io.py              # Núcleo UI/IO: ask_text, ask_choice, ask_user, ANSI, render incremental.
│       ├── models_ui/                  # Pantallas y helpers de Model Manager desacoplados.
│       │   ├── custom_screen.py        # Pantalla de descarga manual por ID/URL y selección de cuantización.
│       │   ├── registry_screen.py      # Pantalla de catálogo/registro de modelos disponibles para descarga.
│       │   ├── shared.py               # Helpers comunes del manager de modelos (selección y progreso).
│       │   ├── lms_download_manager.py # Estado y ciclo de vida de descargas en segundo plano.
│       │   ├── lms_menu_helpers.py     # Utilidades de formateo/capacidad para menús de modelos.
│       │   └── lms_models.py           # Wrappers de LM Studio para listar, descargar y resolver opciones.
│       └── tests_ui/                   # Pantallas y helpers extraídos de Setup Tests UI.
│           ├── ab_experiment.py        # Flujo UI del experimento A/B con preview de variantes y resumen final.
│           ├── batch.py                # Flujo UI del Batch Runner y selección de parámetros de ejecución.
│           ├── cli_reporters.py        # Reportes CLI y helpers de render para dashboards y tests
│           ├── grounding_scenarios.py  # Selector y orquestación de escenarios A/B/C/D/E/F/S desde la TUI.
│           ├── grounding_scenarios_helpers.py # Helpers de validación y preparación de inputs para grounding.
│           ├── grounding_scenarios_live_runner.py # Ejecución live desacoplada para dashboards de grounding.
│           ├── manifest.py             # Gestión completa de manifests (crear/derivar/usar/eliminar) y limpieza de outputs.
│           ├── manifest_generation.py  # Generación estratificada de manifests para muestreo reproducible.
│           ├── poc_bbox.py             # Flujo UI reactivo de la PoC BBox (dashboard vivo + resumen responsive).
│           ├── response_inspector.py   # Pantalla UI para inspección de respuestas reales del SDK.
│           ├── run_pytest.py           # Ejecución de pytest desde TUI con resumen y navegación.
│           ├── schema.py               # Pantallas del Schema Tester (modelo, esquema y variante reasoning).
│           ├── shared.py               # Helpers comunes de tests_ui y cabeceras reutilizables.
│           ├── smoke.py                # Flujo interactivo del smoke test sobre imágenes de validación.
│           ├── telemetry.py            # Pantallas del Telemetry Probe con métricas y resumen visual.
│           └── test_dashboards_ui.py   # Dashboards/visualización en terminal para resultados de tests.
└── tests/
    ├── test_ab_experiment_ui.py        # Validación del preview/limiter del experimento A/B en la TUI.
    ├── test_app_config.py              # Contratos de configuración estática y registro de modelos.
    ├── test_batch_runner.py            # Exportado incremental y persistencia por imagen.
    ├── test_experiment_ab_text.py      # Detección de variantes y selección estratificada de muestras AD.
    ├── test_inference_script.py        # Smoke test, keywords esperadas y validación por etiquetas.
    ├── test_inspect_lmstudio_response.py # Inspector de respuesta, autodetección y resumen visual.
    ├── test_lms_download_manager.py    # Descargas de modelos y manejo de estados.
    ├── test_lms_menu_helpers.py        # Helpers de menús LM Studio.
    ├── test_lms_models.py              # Wrappers del SDK/CLI y utilidades de modelos.
    ├── test_menu_kit.py                # Tablas, AppContext y primitives de UI.
    ├── test_poc_bbox.py                # Tests unitarios de descarga, reporter y flujo principal de la PoC BBox.
    ├── test_preprocess.py              # Preprocesado de imágenes endoscópicas.
    ├── test_schemas.py                 # Esquemas Pydantic, reasoning y registro público.
    ├── test_setup_diagnostics.py       # Diagnóstico y smart-fix del entorno.
    ├── test_setup_env_menu.py          # Integración general de setup_env.
    ├── test_setup_extracted_menus.py   # Menús extraídos de tests/modelos/reinstalación.
    ├── test_setup_install_flow.py      # Flujo de instalación y navegación principal.
    ├── test_setup_menu_engine.py       # Motor de menús interactivos.
    ├── test_setup_models_ui.py         # UI del gestor de modelos.
    ├── test_setup_ui_io_render.py      # Render de terminal, wrap y repintado incremental.
    ├── test_telemetry.py               # Extracción de TTFT/TPS y resumen de telemetría.
    └── test_vlm_runner.py              # Backend de inferencia, compatibilidad y resize multimodal.
```

## Notebooks

- `notebooks/01_eda_polypsegm.ipynb`: libreta principal de EDA y preparación de datos.
- `notebooks/02_zeroshot_analysis.ipynb`: análisis de resultados zero-shot desde `batch_results` JSONL.
- `notebooks/03_ground_truth_bboxes.ipynb`: ejecución del extractor de Ground Truth espacial y sanity check visual de BBoxes.

Incluye:

- Extracción automática del dataset y comprimidos anidados (ZIP/RAR).
- Verificación de formato, resoluciones y revisión básica de metadatos EXIF.
- Ejecución del preprocesado completo y comparativa visual antes/después.
- Cálculo de métricas y reportes por `model_id` y `variant`.
- Exportación en JSONL de métricas, tablas y reportes.
- Empaquetado ZIP con imágenes planas (`images/`) y JSONL en raíz.

Para abrirla:

```bash
uv run jupyter notebook
```

> Si usas VS Code, puedes abrir la libreta directamente desde el explorador del proyecto.

## Preprocesado (`src/preprocessing/preprocess.py`)

Script CLI para recortar bordes negros en imágenes endoscópicas usando OpenCV.

### Qué hace

- Detecta el contorno principal del campo visible.
- Recorta la imagen al bounding box del contorno.
- Guarda el resultado en `data/processed` respetando la estructura relativa.
- Duplica automáticamente los CSV de partición (`train.csv`, `valid.csv`, `gt_test.csv`) desde `data/raw` a `data/processed`.

### Ejecución

```bash
uv run python src/preprocessing/preprocess.py --input-dir data/raw --output-dir data/processed
```

Opciones útiles:

- `--max-images N`: limita el número de imágenes (smoke rápido).
- `--min-area-ratio 0.1`: umbral mínimo de área del contorno.
- `--dry-run`: calcula recortes sin escribir archivos.

## Extracción de Ground Truth BBoxes (`src/preprocessing/extract_gt_bboxes.py`)

Script CLI para extraer coordenadas espaciales desde máscaras binarias (`.tif/.tiff`) y exportarlas a CSV en escala normalizada 0-1000.

### Qué hace

- Busca máscaras recursivamente dentro de carpetas `masks`.
- Calcula el contorno principal y su `boundingRect` con OpenCV.
- Normaliza coordenadas al formato `[xmin, ymin, xmax, ymax]` en escala 0-1000.
- Genera CSV por split en `m_train`/`m_valid` (modo recomendado), por ejemplo:
    - `data/processed/m_train/ground_truth_bboxes.csv`
    - `data/processed/m_valid/ground_truth_bboxes.csv`
- Incluye protección contra sobreescritura (requiere `--force` si el CSV ya existe).

### Ejecución

```bash
uv run python src/preprocessing/extract_gt_bboxes.py \
    --input-dir data/raw \
    --output-root data/raw \
    --split-aware \
    --splits m_train m_valid \
    --force
```

Opciones útiles:

- `--input-dir`: raíz del dataset (por defecto `data/raw`).
- `--output-root`: raíz de salida en modo `--split-aware`.
- `--split-aware`: exporta un CSV por split.
- `--splits`: lista de splits a procesar (por defecto `m_train m_valid`).
- `--output-csv`: ruta de salida en modo simple (por defecto `data/processed/m_train/ground_truth_bboxes.csv`).
- `--force`: sobrescribe el CSV si ya existe.

## Prerrequisitos

1. Python 3.12 disponible en PATH.
2. LM Studio instalado. Puedes descargarlo desde aqui: https://lmstudio.ai/download
3. CLI `lms` disponible (si no, instalarla desde LM Studio).
4. Al menos un modelo VLM descargado/cargado (por ejemplo `qwen3-vl-8b-instruct@q8_0` en tu entorno actual).

## Instalación y arranque

Guía corta para tutor en Linux/HPC (clonado + setup + batch runner):
- Ver `docs/INSTRUCCIONES_TUTOR.md`.

### Opción A: flujo recomendado (manager interactivo)

Windows:

```bash
./setup.bat
```

Linux/macOS:

```bash
bash ./setup.sh
```

Esto lanza `setup_env.py`, que guía instalación/diagnóstico/tests/modelos desde TUI.

### Opción B: setup manual con `uv`

```bash
uv sync
```

Para ejecutar código/tests sin activar manualmente venv:

```bash
uv run python -m pytest -q
```

## Manager Tool (`setup_env.py`)

El manager ofrece:

- Diagnóstico de entorno (`uv`, dependencias, `lms`, etc.).
- Gestión de modelos LM Studio (listado, descarga, carga/descarga, pull).
- Ejecución de tests y smoke test.
- Ejecución de Schema Tester, Telemetry Probe, Response Inspector, Batch Runner y experimento A/B de prompting.
- Reinstalación selectiva de librerías.
- Factory reset con confirmación segura.

Las pruebas automatizadas no deben dejar artefactos en disco. El guardado del
Response Inspector se centraliza en un helper mockeable para que los tests de
menú y de integración validen el flujo sin crear archivos reales fuera de los
directorios temporales del propio test.

La UI de terminal está centralizada en `UIKit` (`src/utils/menu_kit.py`), que encapsula estilo ANSI, helpers de renderizado (banners, tablas, menús, divisores…) y módulos del SO. Todos los submódulos reciben una instancia de `UIKit` + `AppContext` en lugar de dicts `ctx` con referencias sueltas.

`UIKit` incluye un generador de tablas tipadas con columnas, filas y celdas avanzadas:

```python
kit.build_table_items(
    columns=[kit.TableColumn("MODEL", min_width=24), kit.TableColumn("QUANT", fixed_width=12)],
    rows=[
        kit.TableRow(cells=["my-model", "Q4_K_M"], action=fn,
                     cell_colors=[None, "OKCYAN"]),
    ],
)
```

O como menú completo con `kit.table_menu(columns, rows)` que devuelve el `TableRow` seleccionado.

Además, para tablas no interactivas (resúmenes finales, dashboards o bloques informativos) se puede reutilizar el mismo motor con `interactive=False` y recuperar líneas renderizadas con `return_lines=True`.

La API legacy `kit.table(...)` fue retirada; la convención actual unifica la salida tabular en `table_menu`/`build_table_items`.

Convención UI: las pantallas finales estructuradas usan `kit.render_and_wait_responsive(...)` para mantener la espera reactiva con repintado automático al redimensionar el terminal.

Controles de teclado:

- `↑/↓`: navegar
- `ENTER`: seleccionar/ejecutar
- `SPACE`: selección múltiple/submenú (según contexto)
- `ESC`: volver/salir

## Backend de inferencia (`src/inference/vlm_runner.py`)

`VLMLoader` implementa:

- Carga/descarga explícita de modelo (`load_model`, `unload_model`).
- Precalentamiento (`preload_model`).
- Soporte de host/token opcionales:
    - `server_api_host`
    - `api_token`
- Preparación robusta de imagen:
    - normalización RGB
    - resize preventivo preservando relación de aspecto hasta 1.8 millones de píxeles
    - subida como `prepare_image(...)`
    - fallback a `data URL` base64 si aplica
- Extracción opcional de telemetría por inferencia:
    - `total_duration_seconds`
    - `ttft_seconds`
    - `generation_duration_seconds`
    - `prompt_tokens_per_second`
    - `tokens_per_second`
    - `reasoning_tokens` a partir del campo `reasoning` del schema validado
    - `stop_reason`
    - `model_id`, `vram_usage_mb`, `total_params`, `architecture`, `quantization`, `gpu_layers`
    - `prompt_tokens`, `completion_tokens`, `total_tokens`
- Compatibilidad mejorada de errores cuando LM Studio no está accesible.

El Schema Tester ofrece ahora dos variantes por esquema: base y con razonamiento.
La variante con razonamiento añade el campo `reasoning` como primer elemento del
JSON para forzar el análisis previo al veredicto, sin depender del thinking nativo
del modelo, que no es compatible con la salida estructurada.

Esta variante no se implementa duplicando manualmente cada clase Pydantic. En
`src/inference/schemas.py` cada esquema base se define una sola vez y la versión
con razonamiento se genera dinámicamente con `pydantic.create_model(...)`. El
helper interno `_create_reasoning_schema(...)` reutiliza los campos del esquema
base, inserta `reasoning` como primer campo y conserva el resto de restricciones,
tipos y descripciones. De este modo:

- el contrato base sigue siendo simple y reutilizable;
- el modo con razonamiento mantiene `reasoning` en primera posición del JSON;
- no hay duplicación de campos entre `Schema` y `SchemaWithReasoning`;
- el selector del Schema Tester solo decide qué variante pedir al modelo.

La resolución de la variante activa se centraliza en `get_schema_variant(...)`,
que devuelve el esquema base o su versión `WithReasoning` según la opción elegida
en el menú. Esto permite que la inferencia estructurada siga usando exactamente el
mismo flujo en `VLMLoader`, cambiando únicamente el `response_format` solicitado.

Entre los esquemas disponibles se incluyen:

- `GenericObjectDetection`
- `PolypDetection`
- `PolypClassification`
- `PolypDiagnosisClassificationOnly`
- `PolypDiagnosisAndGrounding`
- `SycophancyTest`
- `ImageQualityAssessment`

### Ejemplo rápido de uso

```python
from src.inference.vlm_runner import VLMLoader

loader = VLMLoader(
        model_path="qwen3-vl-8b-instruct@q8_0",
        verbose=True,
        # Opcionales:
        # server_api_host="localhost:1234",
        # api_token="tu-token",
)

loader.preload_model()
result = loader.inference("data/smoke_test/sample_01.jpg", "Describe la imagen")
print(result.model_dump())
loader.unload_model()
```

Con telemetría:

```python
from src.inference.schemas import GenericObjectDetection
from src.inference.vlm_runner import VLMLoader

loader = VLMLoader(model_path="qwen3-vl-8b-instruct@q8_0")
result = loader.inference(
    "data/smoke_test/sample_01.jpg",
    "Describe la imagen",
    schema=GenericObjectDetection,
    include_telemetry=True,
)

print(result.data.model_dump())
print(result.telemetry.tokens_per_second)
```

## Smoke test (`src/scripts/test_inference.py`)

### Qué hace

- Detecta modelo (SDK primero, fallback `lms ls`).
- Asegura imágenes de prueba; si faltan, las descarga automáticamente desde URLs fallback.
- Ejecuta inferencia sobre 5 casos:
    - `sample_01` (cat)
    - `sample_02` (cat)
    - `sample_03` (dog)
    - `sample_04` (dog)
    - `sample_05` (none, ausencia de gato/perro)
- Valida salida JSON y contenido esperado por etiqueta.
- Acepta variantes léxicas y taxonómicas razonables en `object_detected` y/o `justification`
    (por ejemplo `dog`, `perro`, `canino`, `Canis`).
- Expone un parser dedicado (`build_parser()`) para facilitar reutilización y testeo del CLI.

### Ejecutar smoke test

No interactivo:

```bash
uv run python src/scripts/test_inference.py --model_path qwen3-vl-8b-instruct@q8_0
```

Interactivo (selector de modelo):

```bash
uv run python src/scripts/test_inference.py --interactive
```

## Telemetry Probe (`src/scripts/test_telemetry.py`)

Mide viabilidad práctica del modelo en hardware de consumo a partir de una muestra
de imágenes del proyecto.

La implementación actual se basa en dos fuentes explícitas:

- `response.stats` del SDK `lmstudio-python`
- `GET http://localhost:1234/v1/models` como fuente complementaria de recursos y metadatos cuando LM Studio los expone

### Qué extrae

- Latencia total por inferencia.
- TTFT desde `response.stats.time_to_first_token_sec` o alias compatibles.
- TPS de generación desde `response.stats.predicted_tokens_per_sec`.
- Velocidad de ingesta del prompt desde `response.stats.prompt_tokens_per_sec`.
- Tiempo total desde `response.stats.total_duration_ms`.
- Tiempo de generación derivado desde `predicted_tokens_count / predicted_tokens_per_sec`.
- Tokens de prompt, salida y total.
- `reasoning_tokens` estimados a partir del campo `reasoning` del JSON validado por el schema.
- `stop_reason` desde el objeto `stats` del SDK.
- Recursos del modelo desde REST: VRAM usada, arquitectura, quantización, capas en GPU y total de parámetros.

Si LM Studio no devuelve una métrica, el probe muestra `N/D`, mantiene el resto del informe y resume la cobertura real de cada bloque de telemetría sin lanzar avisos ruidosos sobre REST.

## Response Inspector (`src/scripts/test_response_inspector.py`)

Utilidad CLI y opción integrada en `setup_env.py` para inspeccionar qué campos
devuelve realmente el SDK de LM Studio en una petición multimodal.

### Qué hace

- Autodetecta modelo cargado/instalado si no se pasa `--model`.
- Elige automáticamente la primera imagen disponible del proyecto si no se pasa `--image`.
- Permite modo crudo o estructurado (`GenericObjectDetection` / `WithReasoning`).
- Resume en consola solo las secciones con información real:
    - atributos visibles del `PredictionResult`
    - `model_info`
    - `stats`
    - `parsed`
- Guarda el volcado JSON completo en `data/processed/debug/response_inspector_*.json`.

### Ejecución

Modo estructurado con razonamiento y defaults automáticos:

```bash
uv run python src/scripts/test_response_inspector.py --structured --with-reasoning
```

Modo crudo con imagen/modelo explícitos:

```bash
uv run python src/scripts/test_response_inspector.py --model qwen3_5-9b@q8_0 --image data/processed/m_test/images/48.tif
```

### Referencias útiles de LM Studio v1

- `GET http://localhost:1234/api/v1/models`: lista modelos con estado, cuantización y contexto.
- `GET http://localhost:1234/api/v1/models/{model_id}`: detalle estático del modelo seleccionado.
- `GET http://localhost:1234/v1/models`: endpoint estilo OpenAI usado en el proyecto para recursos/metadatos.
- `POST http://localhost:1234/api/v1/chat`: endpoint nativo enriquecido. Si se usa con `stream: true`,
    emite SSE con eventos como `chat.start`, `model_load.*`, `prompt_processing.*`, `reasoning.*`,
    `message.*`, `error` y `chat.end`.
- `POST http://localhost:1234/api/v0/chat/completions`: endpoint enriquecido usado por versiones anteriores,
    con `stats` en el cuerpo JSON.
- `POST http://localhost:1234/v1/chat/completions`: endpoint estilo OpenAI. Puede incluir `stats`
    en versiones recientes de LM Studio, pero no siempre está garantizado.

Para logs internos en tiempo real sigue siendo más práctico usar el CLI:

```bash
lms chat --stats
lms log stream --stats
```

Y, si necesitas logs estructurados para depuración:

```bash
lms log stream --json
```

### Ejecutar

```bash
uv run python src/scripts/test_telemetry.py --model opengvlab_internvl3_5-14b --schema GenericObjectDetection
```

Con reasoning:

```bash
uv run python src/scripts/test_telemetry.py --model opengvlab_internvl3_5-14b --schema PolypDetection --with-reasoning
```

## Batch Runner (`src/scripts/batch_runner.py`)

Ejecuta inferencia masiva sobre un directorio o árbol de directorios y guarda el
resultado en caliente para no perder trabajo si se interrumpe el proceso.

### Qué hace

- Recorre imágenes compatibles de forma recursiva.
- Permite limitar muestra o barajar antes de procesar.
- Carga el modelo una vez y lo descarga en bloque `finally`.
- Exporta incrementalmente en `JSONL`.
- Incluye telemetría por imagen junto al payload estructurado.
- Marca explícitamente `include_reasoning` por registro para diferenciar variantes.
- Si el payload contiene `detections`, cada bbox incluye `iou_score` (si hay GT disponible) y `order_index`.
- El registro añade `bbox_ordering = left-to-right, top-to-bottom (xmin, ymin)` para documentar el criterio.
- Permite generar reporte Markdown al finalizar con `--report` (`report.md` junto al JSONL).
- Opción `--report-details` para incluir tabla detallada por imagen dentro del Markdown.
- Añade cabecera `__batch_meta__` canónica para trazabilidad del archivo compartido.
- Añade línea final `__batch_summary__` con:
    - min/max/media de métricas por variante y global,
    - aciertos y accuracy por variante y global cuando existe `ground_truth_cls`.

### Ejecución basada en manifest (recomendado)

El flujo principal del manager usa manifests autosuficientes (`data/experiments/*.jsonl`) con metadata de ejecución en cabecera.

Campos de cabecera soportados:

- `run_schema_name`
- `run_model_variants` (lista de `{model_id, include_reasoning}`)
- `run_iterations_per_image`
- `run_seed`
- `run_derived_from` (opcional)

La TUI permite:

- usar manifiestos existentes,
- generar nuevos manifests,
- derivar manifests precargando valores del origen,
- eliminar manifests y borrar también sus `batch_*.jsonl` asociados (con confirmación explícita).

### Ejecutar

```bash
uv run python src/scripts/batch_runner.py --model opengvlab_internvl3_5-14b --image-dir data/processed/m_test/images --schema PolypDetection
```

Ejemplo con reasoning y JSONL:

```bash
uv run python src/scripts/batch_runner.py --model opengvlab_internvl3_5-14b --image-dir data --schema PolypClassification --with-reasoning --shuffle --max-images 25
```

Ejemplo con reporte Markdown:

```bash
uv run python src/scripts/batch_runner.py --model opengvlab_internvl3_5-14b --image-dir data/processed/m_test/images --schema PolypDetection --report --report-details
```

Nota: en el flujo TUI no se mezclan resultados de variantes; cada variante se agrega y resume de forma independiente dentro del JSONL compartido.

Nota de reporte PoC BBox (`src/scripts/poc_bbox.py`):

- La cabecera del `results.md` incluye `Global IoU (avg)`.
- Cada bloque `Resumen` incluye `IoU medio` por imagen.
- En `Detecciones` se muestra `IoU` por bbox (sin listar `norm`/`px`, priorizando la visualización en imágenes).

## Grounding Scenarios (`src/scripts/grounding_experiments/`)

Módulo dedicado a escenarios de visual grounding clínico y robustez de razonamiento (Scenario S) con ejecución incremental en JSONL, modo `resume`, generación de artefactos visuales y reporte Markdown enriquecido.

Arquitectura actual:

- `runner_core.py` actúa como fachada única para escenarios y UI.
- Todos los escenarios A/B/C/D/E/F/S importan utilidades compartidas exclusivamente desde `runner_core.py`.
- Persistencia/agregación JSONL en `report_aggregation.py`.
- Helpers reutilizables transversales en `report_common.py` (sin duplicación de lógica de clase/bbox).
- Serialización de registros por imagen en `report_serialization.py`.
- Métricas, narrativa, visuales y markdown desacoplados en módulos dedicados.

### Evaluación espacial combinada: IoU + Proximity

Para escenarios con localización (A/B/C/E/F), la evaluación espacial se interpreta en doble plano:

- `IoU` como medida base de solapamiento geométrico estricto.
- `Proximity` como señal complementaria de cercanía clínica entre bbox predicha y bbox GT.

Definición operativa de `Proximity`:

- `proximity_center_score`: score [0,1] basado en distancia de centros normalizada por diagonal conjunta y penalización gaussiana.
- `proximity_size_score`: score [0,1] basado en diferencia relativa de área entre bbox predicha y GT.
- `proximity_score`: combinación ponderada `0.6 * center + 0.4 * size`.

Campos persistidos por registro JSONL:

- `proximity_score`
- `proximity_center_score`
- `proximity_size_score`

Resumen acumulado por ejecución:

- `avg_iou`
- `avg_proximity`

Interpretación recomendada:

- `IoU` bajo + `Proximity` alta: posible atención correcta con regresión geométrica imperfecta.
- `IoU` alto + `Proximity` baja: posible solapamiento aceptable con desajuste centro/tamaño.

Los reportes Markdown muestran ambas señales y deben leerse junto con la inspección visual de artefactos de grounding.

### Scenario A (`run_scenario_A.py`)

Zero-shot sobre imágenes clínicas:

- Prompt sin pista explícita de clase.
- Clasificación + localización bbox en una sola inferencia.
- Clase GT resuelta por lookup de splits (`m_train`/`m_valid`/`m_test`) para evitar dependencia de columnas de clase en el CSV base.
- Reporte final con accuracy, macro-F1, recall por clase, heatmap de confusión, top de errores e imágenes anotadas.

Ejemplo:

```bash
uv run python -m src.scripts.grounding_experiments.run_scenario_A \
    --model qwen3_5-9b@q8_0 \
    --img-dir data/processed/m_train/images \
    --limit 50 \
    --seed 42
```

### Scenario B (`run_scenario_B.py`)

Escenario asistido por clase GT:

- Prompt condicionado por la clase real para guiar la localización.
- Clase GT resuelta por diseño desde lookup de splits (igual criterio robusto que Scenario A).
- Misma estrategia de salida que Scenario A: JSONL incremental + `__scenario_meta__` + `__scenario_summary__` + reporte Markdown.
- Compatible con `--resume` y acumulación histórica del summary sobre todo el JSONL.

Ejemplo:

```bash
uv run python -m src.scripts.grounding_experiments.run_scenario_B \
    --model qwen3_5-9b@q8_0 \
    --img-dir data/processed/m_train/images \
    --limit 50 \
    --seed 42
```

### Scenario C (`run_scenario_C.py`)

Escenario con guía visual explícita por bbox GT:

- Dibuja la bbox GT sobre la imagen completa y usa esa imagen anotada como entrada de inferencia.
- Mantiene clasificación + localización en una única respuesta estructurada.
- Persistencia incremental y reporte final alineados con el pipeline común.

### Scenario D (`run_scenario_D.py`)

Escenario de clasificación focalizada con recorte ROI:

- Envía una sola imagen: el recorte ROI derivado de la bbox GT.
- Usa un schema sin localización, así que no calcula IoU ni devuelve bbox.
- Permite medir cuánto ayuda el foco visual cuando el modelo solo debe clasificar el pólipo.

### Scenario E (`run_scenario_E.py`)

Escenario que combina B + C:

- Combina la clase GT asistida del escenario B con la imagen completa anotada con bbox GT del escenario C.
- Mantiene una sola imagen de entrada para evitar multiimagen en esta fase.
- Conserva el contrato común de salida: JSONL incremental + summary + Markdown.

### Scenario F (`run_scenario_F.py`)

Escenario de reporte clínico asistido:

- Inyecta en prompt la clase histológica GT (AD/HP/ASS) y desactiva el diagnóstico libre.
- Mantiene localización por bbox (`xmin`, `ymin`, `xmax`, `ymax`) para anclaje visual.
- Exige explicabilidad clínica estructurada de alto detalle (morfología, patrón vascular y razonamiento diagnóstico).
- Conserva contrato común: JSONL incremental + `__scenario_meta__` + `__scenario_summary__` + reporte Markdown.

### Scenario S (`run_scenario_S.py`)

Escenario de stress test de sycophancy con presión de autoridad:

- Ejecuta prompts engañosos con tres niveles de intensidad (`--level 1|2|3`).
- Evalúa si el modelo mantiene criterio visual frente a la instrucción autoritaria.
- KPIs específicos:
    - `prompt_contradiction_rate`
    - `prompt_obedience_rate`
    - `polyp_detected_rate`
    - desglose `TRUE/FALSE` de contradicción por clase GT.
- Conserva contrato común: JSONL incremental + `__scenario_meta__` + `__scenario_summary__` + reporte Markdown.

### Artefactos visuales de reporte (A/B/C/E/F/S)

Además del heatmap de confusión (`class_confusion_heatmap.png`), el reporte genera:

- IoU:
    - `iou_distribution_histogram.png`
    - `iou_class_summary_grouped_bars.png`
    - `iou_correctness_comparison.png`
    - `iou_boxplot_class_correctness.png`
    - `iou_threshold_cumulative_curve.png`
- Proximity:
    - `proximity_distribution_histogram.png`
    - `proximity_class_summary_grouped_bars.png`
    - `proximity_correctness_comparison.png`
    - `proximity_boxplot_class_correctness.png`
    - `proximity_threshold_cumulative_curve.png`
- Scenario S:
    - `scenario_s_kpi_rates.png`
    - `scenario_s_contradiction_by_gt_class.png`

### Convenciones de datos para escenarios

- CSV de ground truth por defecto: `data/processed/m_train/ground_truth_bboxes.csv`.
- Exportación recomendada de BBoxes por split operativo: `m_train` y `m_valid`.
- `m_test` puede existir como partición de evaluación, pero no se cuenta en el flujo operativo por defecto cuando no dispone de máscaras.

### UI live y completitud en escenarios A/B/C/D/E/F/S

- Desde Setup Tests UI se ejecutan A/B/C/D/E/F/S con dashboard en vivo y cierre en pantalla final reactiva.
- El archivo de resultados usa skeleton inicial (`pending`) y actualización incremental por `image_id`.
- Un run se considera incompleto si queda cualquier registro `pending` o `error/failed/fail`, incluso con `__scenario_summary__` presente.
- En modo `resume`, el runner solo salta registros realmente completados (`ok` o `skip`) y vuelve a intentar pendientes/errores.
- El dashboard live incluye heatmap GT→Pred (AD/HP/ASS) con celdas en formato `conteo / %global / %filaGT`.
- Escala de color del heatmap: blanco → amarillo → naranja → escarlata, con contraste automático de texto por luminancia.

Ejemplo de reanudación explícita:

```bash
uv run python -m src.scripts.grounding_experiments.run_scenario_A \
    --model qwen3_5-9b@q8_0 \
    --img-dir data/processed/m_train/images \
    --limit 50 \
    --seed 42 \
    --output data/processed/scenario_results/scenario_A/run_20260331_120000/results.jsonl \
    --resume
```

## A/B Prompting Experiment (`src/scripts/experiment_ab_text.py`)

Comparativa controlada entre dos estrategias de prompting sobre casos AD previamente evaluados:

- Prompt A (zero-shot): descripción de textura/color/morfología sin contexto adicional.
- Prompt B (asistido): misma descripción, condicionada al contexto de Ground Truth AD.

### Qué hace

- Autodetecta un JSONL previo en `data/processed/batch_results` (o usa `--results-file` si se indica).
- Detecta variantes por `model_id` + `include_reasoning` + `schema_name`.
- Selecciona muestras AD estratificadas (aciertos/fallos previos) de forma reproducible con semilla.
- Ejecuta ambos prompts por muestra con salida estructurada (`texture`, `color`, `morphology`, `conclusion`).
- Genera un reporte Markdown comparativo en `data/processed/ab_results/results_ab_experiment.md`.

### Ejecución CLI

Automática (valores por defecto y autodetección de JSONL):

```bash
uv run python src/scripts/experiment_ab_text.py
```

Con parámetros explícitos:

```bash
uv run python src/scripts/experiment_ab_text.py --results-file data/processed/batch_results/batch_zeroshot_manifest_20260320_184504_PolypDetection.jsonl --n-correct 5 --n-incorrect 5 --seed 42 --temperature 0.2
```

Parámetros más usados:

- `--model`: filtra una variante concreta.
- `--results-file`: JSONL de entrada explícito.
- `--results-dir`: directorio para autodetección de JSONL.
- `--output-md`: ruta de salida del reporte Markdown.
- `--n-correct` / `--n-incorrect`: tamaño estratificado por aciertos/fallos AD.

### Integración en TUI

Disponible en Setup Tests como `Run A/B Prompting Experiment (AD)`, con:

- preview de variantes detectadas y muestras candidatas,
- confirmación previa a ejecución,
- pantalla final reactiva adaptada al ancho de terminal.

## Testing

Suite completa:

```bash
uv run python -m pytest -q
```

Tests más relevantes:

- `tests/test_vlm_runner.py`: backend de inferencia, compatibilidades SDK, parseo estructurado.
- `tests/test_inference_script.py`: smoke test y validación por etiquetas.
- `tests/test_telemetry.py`: extracción de TTFT/TPS y resumen de telemetría.
- `tests/test_batch_runner.py`: exportado incremental y persistencia de resultados por imagen.
- `tests/test_experiment_ab_text.py`: detección de variantes y balanceo estratificado de muestras A/B.
- `tests/test_grounding_runner_core.py`: completitud/resume, agregación acumulada y reportes para escenarios A/B/C/D/E/F.
- `tests/test_grounding_report_iou_metrics.py`: agregaciones IoU usadas por el reporte (histograma, resumen por clase, cohorts y umbrales).
- `tests/test_grounding_report_proximity_metrics.py`: agregaciones Proximity equivalentes a IoU para el reporte.
- `tests/test_proximity_metrics.py`: validación de funciones base de proximidad (centro, tamaño y score combinado).
- `tests/test_grounding_live_heatmap.py`: render del heatmap live (conteo + %global/%fila, gradiente y leyenda).
- `tests/test_progress_bar.py`: clamp de barra de progreso y casos límite sin muestra.
- `tests/test_schemas.py`: validación de esquemas Pydantic base y variantes `WithReasoning`.
- `tests/test_manifest_iterations.py`: manifests compactos, snapshots por variante y limpieza de JSONL vinculados.
- `tests/test_menu_kit.py`: `TableColumn`/`TableRow`/`TableCell`, `build_table_items`, `table_menu`, `AppContext`.
- `tests/test_setup_menu_engine.py`: `MenuItem`, `MenuSeparator`, `interactive_menu`.
- `tests/test_setup_diagnostics.py`: diagnóstico de entorno y smart-fix.
- `tests/test_setup_install_flow.py`: flujos de instalación y check de dependencias.
- `tests/test_setup_ui_io_render.py`: render incremental, cálculo de columnas, wrap.
- `tests/test_setup_models_ui.py`: gestor de modelos y descarga.
- `tests/test_ab_experiment_ui.py`: validación del preview de ejecución A/B y límite de muestras mostrado en UI.
- `tests/test_app_config.py`: configuración estática de la aplicación.
- `tests/test_lms_download_manager.py`: máquina de estado de descargas.
- `tests/test_lms_models.py`, `tests/test_lms_menu_helpers.py`: wrappers lmstudio SDK.
- `tests/test_preprocess.py`: validación del preprocesado de imágenes.

## Configuración avanzada de LM Studio

- Si usas host/puerto no estándar, pásalo en `VLMLoader(server_api_host=...)`.
- Si activas autenticación por token, usa `api_token` o variable de entorno de LM Studio.
- El código prioriza API scoped (`lms.Client`) y mantiene degradación compatible.

## Troubleshooting

### Error: `Fallo de inferencia multimodal ... failed to process image`

El backend ya aplica normalización de imagen antes de subirla. Si persiste:

1. Verifica que el modelo cargado es VLM (no LLM puro).
2. Comprueba que LM Studio está en versión compatible.
3. Reintenta con otra imagen de prueba o vuelve a descargar `data/smoke_test/*`.

### Error: `LLM.respond() got an unexpected keyword argument ...`

La capa de compatibilidad en `VLMLoader` ya hace fallback automático entre variantes de firma.

### El smoke test falla aunque la clase parece correcta

Algunos modelos devuelven nombres taxonómicos o más específicos (`Canis`, `Felis`, `canino doméstico`) en lugar de etiquetas coloquiales exactas. El validador ya contempla estas variantes razonables usando `object_detected` y `justification` de forma conjunta.

Si sigue fallando:

1. Revisa el JSON devuelto completo, no solo `justification`.
2. Añade aliases semánticos en `KEYWORDS_BY_LABEL` si estás evaluando un modelo con terminología sistemáticamente distinta.
3. Verifica que el modelo realmente sea VLM y no una variante sin capacidad visual.

### `lms` no encontrado

Instala/activa la CLI de LM Studio y reabre terminal.

## Reproducibilidad

- Usa `uv sync` para reconstruir entorno.
- Versiona siempre `pyproject.toml` y `uv.lock` al cambiar dependencias.

## Nota académica

Este repositorio está orientado a evaluación experimental (TFG). No sustituye validación clínica ni certificación médica para uso asistencial real.
