# Instrucciones de Ejecucion para Tutor (Linux/HPC)

Objetivo: clonar el repositorio y ejecutar inferencia masiva en Linux con un entorno reproducible usando `uv`.

## 1) Clonar repositorio

```bash
git clone https://github.com/NIU1635219/TFG_VLM_Medical
cd TFG_VLM_Medical
```

## 2) Verificar prerequisitos

```bash
python3 --version
uv --version
```

Requisitos minimos:
- Python 3.12+
- `curl` o `wget` disponibles para auto-instalar `uv` durante `setup.sh`
- LM Studio y CLI `lms` disponibles si se va a ejecutar inferencia real

Nota: si `uv` no está instalado, `setup.sh` intentará instalarlo automáticamente
con el instalador oficial (sin usar `pip install uv`).

## 3) Inicializar entorno

```bash
chmod +x setup.sh
./setup.sh
```

Esto ejecuta `setup_env.py` y prepara el entorno del proyecto.

## 4) Ejecutar Batch Runner (ejemplo directo)

```bash
uv run python src/scripts/batch_runner.py \
  --model opengvlab_internvl3_5-14b \
  --image-dir data/processed/m_test/images \
  --schema PolypDetection \
  --report --report-details
```

Salida esperada:
- JSONL incremental en `data/processed/batch_results/`
- Reporte Markdown junto al JSONL cuando se usa `--report`

## 5) Comando rapido de validacion

```bash
uv run python -m pytest -q
```

## Notas de compatibilidad

- El flujo de setup y terminal ya es compatible con Windows y Linux.
- No hay dependencias declaradas en `pyproject.toml` exclusivas de Windows.
- En `uv.lock` pueden aparecer wheels de multiples plataformas (incluyendo Windows) de forma normal; `uv` selecciona automaticamente los artefactos compatibles con Linux al resolver/instalar.

## 6) Ejecutar escenarios A–E (Grounding experiments)

Los escenarios A..E están implementados en `src/scripts/grounding_experiments/` como scripts independientes.

Ejemplos rápidos (reemplaza `--model` y `--img-dir` según tu entorno):

```bash
# Scenario A (zero-shot grounding + diagnosis)
uv run python src/scripts/grounding_experiments/run_scenario_A.py \
  --model opengvlab_internvl3_5-14b \
  --img-dir data/processed/m_test/images

# Scenario B (clase asistida / lookup split)
uv run python src/scripts/grounding_experiments/run_scenario_B.py \
  --model opengvlab_internvl3_5-14b \
  --img-dir data/processed/m_test/images

# Scenario C (GT bbox overlay guide)
uv run python src/scripts/grounding_experiments/run_scenario_C.py \
  --model opengvlab_internvl3_5-14b \
  --img-dir data/processed/m_test/images

# Scenario D (ROI + classification)
uv run python src/scripts/grounding_experiments/run_scenario_D.py \
  --model opengvlab_internvl3_5-14b \
  --img-dir data/processed/m_test/images

# Scenario E (combinado B+C)
uv run python src/scripts/grounding_experiments/run_scenario_E.py \
  --model opengvlab_internvl3_5-14b \
  --img-dir data/processed/m_test/images
```

Flags útiles comunes:

- `--limit N`: procesar solo N imágenes (útil para pruebas rápidas).
- `--seed <int>`: semilla reproducible para muestreo (use la misma seed entre escenarios para comparar el mismo subconjunto).
- `--resume`: reanuda ejecución reutilizando JSONL de salida si existe (salta imágenes ya procesadas).

Los scripts crean resultados incrementales en `data/processed/scenario_results/` y utilizan las mismas utilidades de reporte y visualización que el batch runner.
