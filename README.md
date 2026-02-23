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
- Respuesta estructurada con esquema `VLMStructuredResponse`.
- Smoke test con 5 casos (`sample_01`...`sample_05`) y descarga automática de imágenes si faltan.
- Manager TUI modularizado y con tests de regresión.

## Stack técnico

- Python >= 3.12
- Gestor de dependencias: `uv`
- Inferencia: LM Studio + `lmstudio-python`
- Librerías principales: `lmstudio`, `pydantic`, `pillow`, `opencv-python`, `pandas`, `matplotlib`, `pytest`, `tqdm`, `jupyter`

Dependencias declaradas en `pyproject.toml`.

## Estructura del repositorio

```text
.
├── .venv/
├── data/
│   ├── raw/
│   ├── processed/
│   └── smoke_test/
├── notebooks/
│   └── 01_eda_polypsegm.ipynb
├── src/
│   ├── inference/
│   │   └── vlm_runner.py
│   ├── preprocessing/
│   │   └── preprocess.py
│   ├── scripts/
│   │   └── test_inference.py
│   └── utils/
│       ├── lms_menu_helpers.py
│       ├── lms_models.py
│       ├── setup_ui_io.py
│       ├── setup_menu_engine.py
│       ├── setup_models_ui.py
│       ├── setup_tests_ui.py
│       ├── setup_reinstall_ui.py
│       ├── setup_install_flow.py
│       └── setup_diagnostics.py
├── tests/
│   ├── test_vlm_runner.py
│   ├── test_inference_script.py
│   ├── test_setup_env_menu.py
│   └── test_setup_extracted_menus.py
├── setup_env.py
├── setup.bat
├── setup.sh
├── pyproject.toml
├── uv.lock
└── README.md
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
- Reinstalación selectiva de librerías.
- Factory reset con confirmación segura.

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
    - resize preventivo
    - subida como `prepare_image(...)`
    - fallback a `data URL` base64 si aplica
- Respuesta estructurada con `VLMStructuredResponse`:
    - `polyp_detected: bool`
    - `confidence_score: int`
    - `justification: str`

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

### Ejecutar smoke test

No interactivo:

```bash
uv run python src/scripts/test_inference.py --model_path qwen3-vl-8b-instruct@q8_0
```

Interactivo (selector de modelo):

```bash
uv run python src/scripts/test_inference.py --interactive
```

## Testing

Suite completa:

```bash
uv run python -m pytest -q
```

Tests más relevantes:

- `tests/test_vlm_runner.py`: backend de inferencia, compatibilidades SDK, parseo estructurado.
- `tests/test_inference_script.py`: smoke test y validación por etiquetas.
- `tests/test_setup_env_menu.py`: menú principal y comportamiento del manager.
- `tests/test_setup_extracted_menus.py`: menús extraídos/modularizados.

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

### `lms` no encontrado

Instala/activa la CLI de LM Studio y reabre terminal.

## Reproducibilidad

- Usa `uv sync` para reconstruir entorno.
- Versiona siempre `pyproject.toml` y `uv.lock` al cambiar dependencias.

## Nota académica

Este repositorio está orientado a evaluación experimental (TFG). No sustituye validación clínica ni certificación médica para uso asistencial real.
