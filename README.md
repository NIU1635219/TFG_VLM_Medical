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
- `UIKit` incluye API de tablas interactivas (`TableColumn`, `TableRow`, `build_table_items`, `table_menu`) con anchos de columna adaptativos al terminal.
- Schema Tester integrado en el manager: selección interactiva de modelo + esquema + inferencia por lotes.
- Batch Runner CLI con exportado incremental en JSONL compartido por manifiesto+schema.
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
    - configuración global (`run_models`, `run_schema_name`, `run_include_reasoning`) en metadatos de cabecera,
    - filas de imágenes más limpias y sin repetir configuración por registro,
    - compatibilidad mantenida con manifests legacy que aún embeben esos campos en cada línea.
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
    - una única decisión al inicio (sí/no/selección múltiple de modelos),
    - selección múltiple real con `SPACE`,
    - barra de cola oculta cuando solo se ejecuta un modelo.

## Novedades recientes (marzo 2026)

- Estandarización de docstrings en estilo Google en módulos refactorizados.
- Corrección de imports tras mover helpers de LM Studio a `src/utils/models_ui/`.
- Consolidación del input reactivo en `src/utils/setup_ui_io.py` y reutilización en:
    - pantalla de selección de tamaño de subgrupo (Batch Runner),
    - pantalla de descarga manual por ID/URL (Model Manager).
- Validación continua con `pytest` en bloques focalizados y suite completa durante la refactorización.
- Batch Runner: metadata de salida unificado en `__batch_meta__` v2 y eliminación de campos legacy.
- Batch Runner: generación automática de `__batch_summary__` con métricas globales y por modelo (incluye accuracy).
- Batch Runner UI: estrategia de reejecución por dataset con selector múltiple de modelos.

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
│   └── 01_eda_polypsegm.ipynb          # EDA, extracción y comparativas visuales del dataset.
├── src/
│   ├── inference/
│   │   ├── schemas.py                  # Esquemas Pydantic base y variantes WithReasoning.
│   │   └── vlm_runner.py               # Carga VLM, inferencia multimodal, compatibilidad SDK y telemetría.
│   ├── preprocessing/
│   │   └── preprocess.py               # CLI de recorte de bordes negros y copia de CSV.
│   ├── scripts/
│   │   ├── batch_runner.py             # Orquestador masivo con exportado incremental CSV/JSONL.
│   │   ├── test_inference.py           # Smoke test CLI con descarga automática de imágenes de muestra.
│   │   ├── test_response_inspector.py  # Inspector CLI de respuestas reales del SDK LM Studio.
│   │   ├── test_schema.py              # Lógica batch reutilizable para el Schema Tester interactivo.
│   │   └── test_telemetry.py           # Medición CLI de TTFT/TPS y resumen de latencias.
│   └── utils/
│       ├── app_config.py               # Configuración estática, librerías requeridas y registro de modelos.
│       ├── menu_kit.py                 # UIKit + AppContext: abstracción de tablas, menús y render.
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
│           ├── batch.py                # Flujo UI del Batch Runner y selección de parámetros de ejecución.
│           ├── manifest.py             # Gestión de manifests y selección reactiva de tamaño de subgrupo.
│           ├── manifest_generation.py  # Generación estratificada de manifests para muestreo reproducible.
│           ├── response_inspector.py   # Pantalla UI para inspección de respuestas reales del SDK.
│           ├── run_pytest.py           # Ejecución de pytest desde TUI con resumen y navegación.
│           ├── schema.py               # Pantallas del Schema Tester (modelo, esquema y variante reasoning).
│           ├── shared.py               # Helpers comunes de tests_ui y cabeceras reutilizables.
│           ├── smoke.py                # Flujo interactivo del smoke test sobre imágenes de validación.
│           ├── telemetry.py            # Pantallas del Telemetry Probe con métricas y resumen visual.
│           └── test_dashboards_ui.py   # Dashboards/visualización en terminal para resultados de tests.
└── tests/
    ├── test_app_config.py              # Contratos de configuración estática y registro de modelos.
    ├── test_batch_runner.py            # Exportado incremental y persistencia por imagen.
    ├── test_inference_script.py        # Smoke test, keywords esperadas y validación por etiquetas.
    ├── test_inspect_lmstudio_response.py # Inspector de respuesta, autodetección y resumen visual.
    ├── test_lms_download_manager.py    # Descargas de modelos y manejo de estados.
    ├── test_lms_menu_helpers.py        # Helpers de menús LM Studio.
    ├── test_lms_models.py              # Wrappers del SDK/CLI y utilidades de modelos.
    ├── test_menu_kit.py                # Tablas, AppContext y primitives de UI.
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

Incluye:

- Extracción automática del dataset y comprimidos anidados (ZIP/RAR).
- Verificación de formato, resoluciones y revisión básica de metadatos EXIF.
- Ejecución del preprocesado completo y comparativa visual antes/después.

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

## Prerrequisitos

1. Python 3.12 disponible en PATH.
2. LM Studio instalado. Puedes descargarlo desde aqui: https://lmstudio.ai/download
3. CLI `lms` disponible (si no, instalarla desde LM Studio).
4. Al menos un modelo VLM descargado/cargado (por ejemplo `qwen3-vl-8b-instruct@q8_0` en tu entorno actual).

## Instalación y arranque

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
- Ejecución de Schema Tester, Telemetry Probe, Response Inspector y Batch Runner.
- Reinstalación selectiva de librerías.
- Factory reset con confirmación segura.

Las pruebas automatizadas no deben dejar artefactos en disco. El guardado del
Response Inspector se centraliza en un helper mockeable para que los tests de
menú y de integración validen el flujo sin crear archivos reales fuera de los
directorios temporales del propio test.

La UI de terminal está centralizada en `UIKit` (`src/utils/menu_kit.py`), que encapsula estilo ANSI, helpers de renderizado (banners, tablas, menús, divisores…) y módulos del SO. Todos los submódulos reciben una instancia de `UIKit` + `AppContext` en lugar de dicts `ctx` con referencias sueltas.

`UIKit` incluye un generador de tablas interactivas con columnas y filas tipadas:

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
- Añade cabecera `__batch_meta__` canónica para trazabilidad del archivo compartido.
- Añade línea final `__batch_summary__` con:
    - min/max/media de métricas por modelo y global,
    - aciertos y accuracy por modelo y global cuando existe `ground_truth_cls`.

### Ejecutar

```bash
uv run python src/scripts/batch_runner.py --model opengvlab_internvl3_5-14b --image-dir data/processed/m_test/images --schema PolypDetection
```

Ejemplo con reasoning y JSONL:

```bash
uv run python src/scripts/batch_runner.py --model opengvlab_internvl3_5-14b --image-dir data --schema PolypClassification --with-reasoning --shuffle --max-images 25
```

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
- `tests/test_schemas.py`: validación de esquemas Pydantic (`VLMStructuredResponse`).
- `tests/test_menu_kit.py`: `UIKit.table()`, `TableColumn`/`TableRow`, `build_table_items`, `AppContext`.
- `tests/test_setup_menu_engine.py`: `MenuItem`, `MenuSeparator`, `interactive_menu`.
- `tests/test_setup_diagnostics.py`: diagnóstico de entorno y smart-fix.
- `tests/test_setup_install_flow.py`: flujos de instalación y check de dependencias.
- `tests/test_setup_ui_io_render.py`: render incremental, cálculo de columnas, wrap.
- `tests/test_setup_models_ui.py`: gestor de modelos y descarga.
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
