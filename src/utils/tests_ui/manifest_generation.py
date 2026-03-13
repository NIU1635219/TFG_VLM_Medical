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

_IMG_EXTENSIONS: tuple[str, ...] = (".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp", ".webp")
_MANIFEST_META_KEY = "__manifest_meta__"


def _normalize_image_id(value: Any) -> str | None:
    """Normalize image identifiers to stable filename stems.

    Args:
        value: Raw identifier value from CSV.

    Returns:
        Normalized string identifier or `None` when value is empty.
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
    """Compute per-class sample allocation preserving class proportions.

    Args:
        class_counts: Mapping class -> available rows.
        sample_size: Target total sample size.

    Returns:
        Mapping class -> sampled rows.
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
    """Generate a reproducible stratified sample.

    Args:
        df: Full input dataframe.
        stratify_col: Column used for stratification.
        sample_size: Target sample size.
        seed: Random seed.

    Returns:
        Sampled dataframe.
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
    """Index images by stem for fast id->path resolution.

    Args:
        images_dir: Directory containing image files.

    Returns:
        Mapping image stem -> matching file paths.
    """
    index: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(images_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _IMG_EXTENSIONS:
            continue
        index[path.stem].append(path.resolve())
    return index


def _resolve_image_path(image_id: str, image_index: dict[str, list[Path]], images_dir: Path) -> Path | None:
    """Resolve physical image path from normalized image id.

    Args:
        image_id: Normalized image identifier.
        image_index: Precomputed image index.
        images_dir: Base images directory.

    Returns:
        Resolved image path or `None` if not found.
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
    run_models: list[str] | None = None,
    run_schema_name: str | None = None,
    run_include_reasoning: bool = False,
    compact_run_config: bool = True,
) -> dict[str, Any]:
    """Create experiment JSONL manifest and return summary.

    Args:
        input_csv: CSV path containing source metadata.
        images_dir: Directory with processed images.
        output_path: JSONL output path.
        sample_size: Target sample size.
        seed: Random seed.
        stratify_col: Stratification column name.
        id_col: Image id column name.
        label_col: Label column name.
        relative_paths: Whether to store image paths relative to project root.
        run_models: Optional execution model queue embedded into each row.
        run_schema_name: Optional schema name embedded into each row.
        run_include_reasoning: Optional reasoning flag embedded into each row.
        compact_run_config: If `True`, writes run config once as manifest metadata.

    Returns:
        Summary dictionary for logging and UI.
    """
    if not input_csv.exists() or not input_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if not images_dir.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

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
        has_run_config = bool(run_models) and bool(str(run_schema_name or "").strip())
        if has_run_config and compact_run_config:
            meta_record = {
                _MANIFEST_META_KEY: {
                    "version": 1,
                    "run_models": list(run_models or []),
                    "run_schema_name": str(run_schema_name or ""),
                    "run_include_reasoning": bool(run_include_reasoning),
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

            if relative_paths:
                try:
                    image_path_value = str(resolved.relative_to(Path.cwd()))
                except ValueError:
                    image_path_value = str(resolved)
            else:
                image_path_value = str(resolved)

            record: dict[str, Any] = {
                "image_path": image_path_value,
                "ground_truth_cls": str(ground_truth),
                "image_id": image_id,
            }
            if run_models:
                if not compact_run_config:
                    record["run_models"] = list(run_models)
            if run_schema_name:
                if not compact_run_config:
                    record["run_schema_name"] = str(run_schema_name)
                    record["run_include_reasoning"] = bool(run_include_reasoning)
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
        "run_models": list(run_models or []),
        "run_schema_name": str(run_schema_name or ""),
        "run_include_reasoning": bool(run_include_reasoning),
        "compact_run_config": bool(compact_run_config),
    }
