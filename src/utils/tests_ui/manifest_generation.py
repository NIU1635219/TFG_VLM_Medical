"""Manifest generation utilities used by Tests UI flows.

This module contains the subset of experiment-manifest generation logic needed
by the interactive tests manager workflow.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .shared import normalize_model_variants

_IMG_EXTENSIONS: tuple[str, ...] = (".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp", ".webp")
_MANIFEST_META_KEY = "__manifest_meta__"


def _validate_generate_manifest_inputs(
    *,
    input_csv: Path,
    images_dir: Path,
    run_iterations_per_image: int,
) -> int:
    """Valida entradas base para generación de manifiesto.

    Args:
        input_csv: CSV de entrada.
        images_dir: Carpeta de imágenes.
        run_iterations_per_image: Iteraciones solicitadas.

    Returns:
        Número de iteraciones normalizado a entero positivo.
    """
    if not input_csv.exists() or not input_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if not images_dir.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    normalized_iterations = int(run_iterations_per_image)
    if normalized_iterations <= 0:
        raise ValueError("run_iterations_per_image must be greater than zero")
    return normalized_iterations


def _format_output_image_path(*, resolved: Path, relative_paths: bool) -> str:
    """
    Construye el valor final de image_path en el manifiesto.
    
    Args:
        resolved (Path): Path resuelto de la imagen.
        relative_paths (bool): Si es True, se construye el path relativo.
        
    Returns:
        str: Valor final de image_path en el manifiesto.
    """
    if not relative_paths:
        return str(resolved)

    try:
        return str(resolved.relative_to(Path.cwd()))
    except ValueError:
        return str(resolved)


def _build_manifest_record(
    *,
    image_path_value: str,
    ground_truth: Any,
    image_id: str,
    iteration_index: int,
    run_iterations_per_image: int,
) -> dict[str, Any]:
    """
    Compone una fila JSONL del manifiesto.
    
    Args:
        image_path_value (str): Valor de image_path.
        ground_truth (Any): Ground truth de la imagen.
        image_id (str): ID de la imagen.
        iteration_index (int): Índice de iteración.
        run_iterations_per_image (int): Número total de iteraciones.
        
    Returns:
        dict[str, Any]: Fila JSONL del manifiesto.
    """
    record: dict[str, Any] = {
        "image_path": image_path_value,
        "ground_truth_cls": str(ground_truth),
        "image_id": image_id,
        "run_iteration_index": iteration_index,
        "run_iteration_total": run_iterations_per_image,
    }
    return record


def _normalize_image_id(value: Any) -> str | None:
    """Normaliza los identificadores de imagen a nombres de archivo estables.

    Args:
        value: Valor de identificador en crudo del CSV.

    Returns:
        Identificador de cadena normalizado o `None` cuando el valor está vacío.
    """
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return str(int(value))
        return str(value).strip()

    text = str(value).strip()
    if not text:
        return None
    if text.endswith(".0"):
        integer_candidate = text[:-2]
        if integer_candidate.isdigit():
            return integer_candidate
    return text


def _compute_target_counts(class_counts: dict[str, int], sample_size: int) -> dict[str, int]:
    """
    Calcula la asignación de muestras por clase preservando las proporciones de las clases.

    Args:
        class_counts: Mapeo de clase -> filas disponibles.
        sample_size: Tamaño total de muestra objetivo.

    Returns:
        Mapeo de clase -> filas de muestra.
    """
    if sample_size <= 0:
        raise ValueError("sample_size must be greater than zero")

    total_rows = sum(class_counts.values())
    if sample_size >= total_rows:
        return dict(class_counts)

    classes = list(class_counts.keys())
    target_counts = {class_name: 0 for class_name in classes}

    if sample_size >= len(classes):
        for class_name, count in class_counts.items():
            if count > 0:
                target_counts[class_name] = 1

    reserved = sum(target_counts.values())
    remaining = sample_size - reserved
    if remaining <= 0:
        return target_counts

    fractional_allocations: list[tuple[str, float]] = []
    for class_name, count in class_counts.items():
        available = count - target_counts[class_name]
        if available <= 0:
            continue
        exact = remaining * (count / total_rows)
        floor_value = min(int(math.floor(exact)), available)
        target_counts[class_name] += floor_value
        fractional_allocations.append((class_name, exact - floor_value))

    distributed = sum(target_counts.values())
    still_needed = sample_size - distributed
    if still_needed <= 0:
        return target_counts

    ranked = sorted(
        fractional_allocations,
        key=lambda item: (item[1], class_counts[item[0]]),
        reverse=True,
    )
    rank_index = 0
    while still_needed > 0 and ranked:
        class_name = ranked[rank_index % len(ranked)][0]
        if target_counts[class_name] < class_counts[class_name]:
            target_counts[class_name] += 1
            still_needed -= 1
        rank_index += 1

    return target_counts


def stratified_sample(df: pd.DataFrame, *, stratify_col: str, sample_size: int, seed: int) -> pd.DataFrame:
    """Genera una muestra estratificada reproducible.

    Args:
        df: Dataframe de entrada completo.
        stratify_col: Columna utilizada para la estratificación.
        sample_size: Tamaño de muestra objetivo.
        seed: Semilla aleatoria.

    Returns:
        Dataframe muestreado.
    """
    if stratify_col not in df.columns:
        raise ValueError(f"Missing stratification column: {stratify_col}")

    if df.empty:
        raise ValueError("Input CSV has no rows")

    grouped = df.groupby(stratify_col, sort=True)
    class_counts = {str(class_name): len(group_df) for class_name, group_df in grouped}
    target_counts = _compute_target_counts(class_counts, sample_size)

    sampled_frames: list[pd.DataFrame] = []
    for class_name, group_df in grouped:
        class_key = str(class_name)
        take = target_counts.get(class_key, 0)
        if take <= 0:
            continue
        sampled_frames.append(group_df.sample(n=take, random_state=seed))

    if not sampled_frames:
        raise RuntimeError("Unable to sample rows with the provided configuration")

    sampled = pd.concat(sampled_frames, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return sampled


def _build_image_index(images_dir: Path) -> dict[str, list[Path]]:
    """Indexa las imágenes por nombre base para una resolución rápida de id->ruta.

    Args:
        images_dir: Directorio que contiene los archivos de imagen.

    Returns:
        Mapeo de nombre base de imagen -> rutas de archivo coincidentes.
    """
    index: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(images_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _IMG_EXTENSIONS:
            continue
        index[path.stem].append(path.resolve())
    return index


def _resolve_image_path(image_id: str, image_index: dict[str, list[Path]], images_dir: Path) -> Path | None:
    """Resuelve la ruta física de la imagen a partir del id de imagen normalizado.

    Args:
        image_id: Identificador de imagen normalizado.
        image_index: Índice de imágenes precalculado.
        images_dir: Directorio base de imágenes.

    Returns:
        Ruta de imagen resuelta o `None` si no se encuentra.
    """
    if not image_id:
        return None

    direct_path = Path(image_id)
    if direct_path.suffix:
        candidate = direct_path if direct_path.is_absolute() else (images_dir / direct_path)
        candidate = candidate.resolve()
        if candidate.exists() and candidate.is_file() and candidate.suffix.lower() in _IMG_EXTENSIONS:
            return candidate

    candidates = image_index.get(image_id, [])
    if not candidates:
        return None

    ordered = sorted(candidates, key=lambda path: _IMG_EXTENSIONS.index(path.suffix.lower()))
    return ordered[0]


def generate_manifest(
    *,
    input_csv: Path,
    images_dir: Path,
    output_path: Path,
    sample_size: int,
    seed: int,
    stratify_col: str,
    id_col: str,
    label_col: str,
    relative_paths: bool,
    run_schema_name: str,
    run_model_variants: list[dict[str, Any]],
    run_iterations_per_image: int = 1,
    run_derived_from: str | None = None,
) -> dict[str, Any]:
    """Crea un manifiesto JSONL de experimento y devuelve un resumen.

    Args:
        input_csv: Ruta del CSV que contiene metadatos de origen.
        images_dir: Directorio con imágenes procesadas.
        output_path: Ruta de salida JSONL.
        sample_size: Tamaño de muestra objetivo.
        seed: Semilla aleatoria.
        stratify_col: Nombre de columna de estratificación.
        id_col: Nombre de columna de id de imagen.
        label_col: Nombre de columna de etiqueta.
        relative_paths: Si se deben almacenar las rutas de imagen relativas a la raíz del proyecto.
        run_schema_name: Nombre base del esquema almacenado en los metadatos del manifiesto.
        run_model_variants: Variantes de razonamiento por modelo almacenadas en metadatos.
        run_iterations_per_image: Número de ejecuciones repetidas por imagen/modelo.
        run_derived_from: Ruta opcional del manifiesto de origen al derivar.

    Returns:
        Diccionario de resumen para registro e interfaz de usuario.
    """
    run_iterations_per_image = _validate_generate_manifest_inputs(
        input_csv=input_csv,
        images_dir=images_dir,
        run_iterations_per_image=run_iterations_per_image,
    )
    iterations_range = range(1, run_iterations_per_image + 1)
    run_schema_name_str = str(run_schema_name or "")
    run_seed_value = int(seed)
    run_derived_from_value = str(run_derived_from or "").strip()
    run_model_variants_list = normalize_model_variants(run_model_variants)
    if not run_schema_name_str.strip():
        raise ValueError("run_schema_name is required")
    if not run_model_variants_list:
        raise ValueError("run_model_variants must include at least one variant")

    df = pd.read_csv(input_csv)
    missing_columns = [column for column in (id_col, label_col, stratify_col) if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in CSV: {', '.join(missing_columns)}")

    sampled = stratified_sample(df, stratify_col=stratify_col, sample_size=sample_size, seed=seed)
    image_index = _build_image_index(images_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    missing_images = 0
    missing_by_class: dict[str, int] = defaultdict(int)
    written_by_class: dict[str, int] = defaultdict(int)

    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        meta_record = {
            _MANIFEST_META_KEY: {
                "version": 1,
                "run_schema_name": run_schema_name_str,
                "run_model_variants": run_model_variants_list,
                "run_iterations_per_image": run_iterations_per_image,
                "run_seed": run_seed_value,
                "run_derived_from": run_derived_from_value,
            }
        }
        handle.write(json.dumps(meta_record, ensure_ascii=False) + "\n")

        for row_dict in sampled.to_dict(orient="records"):
            image_id = _normalize_image_id(row_dict.get(id_col))
            ground_truth = row_dict.get(label_col)
            if image_id is None or ground_truth is None or pd.isna(ground_truth):
                missing_images += 1
                continue

            resolved = _resolve_image_path(image_id, image_index, images_dir)
            if resolved is None:
                missing_images += 1
                missing_by_class[str(ground_truth)] += 1
                continue

            image_path_value = _format_output_image_path(
                resolved=resolved,
                relative_paths=relative_paths,
            )

            for iteration_index in iterations_range:
                record = _build_manifest_record(
                    image_path_value=image_path_value,
                    ground_truth=ground_truth,
                    image_id=image_id,
                    iteration_index=iteration_index,
                    run_iterations_per_image=run_iterations_per_image,
                )
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                written_by_class[str(ground_truth)] += 1

    return {
        "input_rows": len(df),
        "sampled_rows": len(sampled),
        "written_rows": written,
        "missing_images": missing_images,
        "output_path": str(output_path),
        "written_by_class": dict(sorted(written_by_class.items())),
        "missing_by_class": dict(sorted(missing_by_class.items())),
        "run_schema_name": run_schema_name_str,
        "run_model_variants": run_model_variants_list,
        "run_iterations_per_image": run_iterations_per_image,
        "run_seed": run_seed_value,
        "run_derived_from": run_derived_from_value,
    }
