"""Shared utilities for visual grounding experiment scenarios.

This module centralizes data loading, image encoding, and robust JSONL
persistence so all scenarios can reuse the same execution primitives.
"""

from __future__ import annotations

import dataclasses
import io
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

import cv2
import pandas as pd

from src.inference.schemas import (
    AssistedClinicalReport,
    PolypDiagnosisAndGrounding,
    PolypDiagnosisClassificationOnly,
)
from src.utils.tests_ui.metrics import calculate_iou, calculate_proximity_score

from .report_aggregation import (
    collect_processed_image_ids_from_jsonl as _collect_processed_image_ids_from_jsonl,
    has_unfilled_scenario_records as _has_unfilled_scenario_records,
    initialize_scenario_result_skeleton as _initialize_scenario_result_skeleton,
    load_jsonl_records as _load_jsonl_records,
    save_result as _save_result,
    summarize_scenario_records_from_jsonl as _summarize_scenario_records_from_jsonl,
    upsert_scenario_result_record as _upsert_scenario_result_record,
    upsert_scenario_execution_summary as _upsert_scenario_execution_summary,
    upsert_scenario_meta_header as _upsert_scenario_meta_header,
)
from .report_common import (
    build_bbox_xyxy,
    build_bbox_xyxy_from_mapping,
    extract_predicted_class,
    normalize_polyp_class,
)
from .report_metrics import (
    build_class_confusion_matrix,
    compute_classification_accuracy_from_records,
    compute_macro_f1_and_recall_by_class,
)
from .report_markdown import (
    build_markdown_records_from_scenario_jsonl as _build_markdown_records_from_scenario_jsonl,
    generate_scenario_s_markdown_report as _generate_scenario_s_markdown_report,
    generate_single_detection_markdown_report as _generate_single_detection_markdown_report,
)
from .report_narrative import (
    build_classwise_heatmap_explanations,
    build_top_confusion_errors,
)
from .report_serialization import (
    build_scenario_record as _build_scenario_record,
    select_telemetry_fields_for_record as _select_telemetry_fields_for_record,
)
from .report_visuals import (
    build_annotated_comparison_filename as _build_annotated_comparison_filename,
    write_class_confusion_heatmap as _write_class_confusion_heatmap,
    write_comparison_image as _write_comparison_image,
)

DEFAULT_GROUND_TRUTH_CSV = Path("data/processed/m_train/ground_truth_bboxes.csv")
DEFAULT_SCENARIO_RESULTS_DIR = Path("data/processed/scenario_results")
DEFAULT_IMAGE_EXTENSIONS: tuple[str, ...] = ("", ".tif", ".tiff", ".jpg", ".jpeg", ".png")
SCENARIO_META_KEY = "__scenario_meta__"
SCENARIO_SUMMARY_KEY = "__scenario_summary__"
SCENARIO_TELEMETRY_RECORD_FIELDS: dict[str, str] = {
    "total_duration_seconds": "total_duration_seconds",
    "ttft_seconds": "ttft_seconds",
    "generation_duration_seconds": "generation_duration_seconds",
    "tokens_per_second": "tokens_per_second",
    "reasoning_tokens": "reasoning_tokens",
    "architecture": "architecture",
    "prompt_tokens": "prompt_tokens",
    "completion_tokens": "completion_tokens",
    "total_tokens": "total_tokens",
}
CLASS_COLUMN_CANDIDATES: tuple[str, ...] = (
    "cls",
    "ground_truth_cls",
    "gt_class",
    "class",
    "label",
    "diagnosis",
    "histology",
    "Histologia",
)
ID_COLUMN_CANDIDATES: tuple[str, ...] = ("image_id", "id", "image", "image_name")
Reporter = Callable[[str, dict[str, Any]], None]
SchemaT = TypeVar("SchemaT")


def load_ground_truth(csv_path: str) -> pd.DataFrame:
    """Load ground-truth bounding boxes from a CSV file.

    Args:
        csv_path: Path to the ground-truth CSV file.

    Returns:
        A pandas DataFrame with ground-truth rows.

    Raises:
        FileNotFoundError: If the CSV path does not exist.
        RuntimeError: If the CSV cannot be read.
    """
    path = Path(csv_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Ground-truth CSV not found: {path}")

    try:
        dataframe = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(f"Failed to read ground-truth CSV at {path}: {exc}") from exc

    return dataframe


def select_ground_truth_rows(
    dataframe: pd.DataFrame,
    *,
    limit: int | None,
    seed: int | None,
) -> pd.DataFrame:
    """Select a deterministic subset of rows for scenario execution.

    When `limit` is provided and smaller than the available rows, selection is
    randomized with `seed` and then restored to original CSV order.
    """
    if limit is None:
        return dataframe

    limit_value = int(limit)
    if limit_value <= 0:
        raise ValueError("limit must be greater than 0")

    if len(dataframe) <= limit_value:
        return dataframe.head(limit_value)

    if seed is None:
        return dataframe.head(limit_value)

    sampled = dataframe.sample(n=limit_value, random_state=int(seed), replace=False)
    return sampled.sort_index()


def normalize_image_stem(value: Any) -> str:
    """Normalize image identifier, fixing float-like integer ids (e.g., 12.0 -> 12)."""
    if value is None:
        raise ValueError("image_id no puede ser None")

    text = str(value).strip()
    if not text:
        raise ValueError("image_id no puede estar vacío")

    try:
        numeric = float(text)
        if numeric.is_integer():
            return str(int(numeric))
    except ValueError:
        pass

    return text


def build_annotated_comparison_filename(*, image_id: Any, image_path: Path) -> str:
    """Backward-compatible wrapper for visual artifact naming helper."""
    return _build_annotated_comparison_filename(image_id=image_id, image_path=image_path)


def build_class_lookup_from_m_split_csvs(*, img_dir: Path) -> dict[str, str]:
    """Build image_id->class lookup from m_* split CSV files around the active image dir."""
    split_dir = img_dir.parent
    split_name = split_dir.name
    root = Path.cwd()

    candidate_csvs = [
        split_dir / "train.csv",
        split_dir / "valid.csv",
        split_dir / "test.csv",
        root / "data" / "processed" / split_name / "train.csv",
        root / "data" / "processed" / split_name / "valid.csv",
        root / "data" / "processed" / split_name / "test.csv",
        root / "data" / "raw" / split_name / "train.csv",
        root / "data" / "raw" / split_name / "valid.csv",
        root / "data" / "raw" / split_name / "test.csv",
    ]

    ordered_unique_csvs: list[Path] = []
    seen: set[Path] = set()
    for csv_path in candidate_csvs:
        resolved = csv_path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered_unique_csvs.append(resolved)

    lookup: dict[str, str] = {}
    for csv_path in ordered_unique_csvs:
        if not csv_path.exists() or not csv_path.is_file():
            continue

        try:
            dataframe = pd.read_csv(csv_path)
        except Exception:
            continue

        id_col = next((col for col in ID_COLUMN_CANDIDATES if col in dataframe.columns), None)
        label_col = next((col for col in CLASS_COLUMN_CANDIDATES if col in dataframe.columns), None)
        if id_col is None or label_col is None:
            continue

        for row in dataframe.to_dict(orient="records"):
            image_id_raw = row.get(id_col)
            label_raw = row.get(label_col)
            if image_id_raw is None or label_raw is None or pd.isna(label_raw):
                continue
            try:
                normalized_id = normalize_image_stem(image_id_raw)
            except ValueError:
                continue

            label_text = normalize_polyp_class(label_raw)
            if not label_text:
                continue
            lookup[normalized_id] = label_text

    return lookup


def resolve_ground_truth_class_from_lookup(
    *,
    row: dict[str, Any],
    class_lookup: dict[str, str],
) -> str:
    """Resolve ground-truth class from precomputed lookup using row image_id."""
    try:
        row_image_id = normalize_image_stem(row.get("image_id"))
    except ValueError:
        raise ValueError(
            "No se pudo normalizar image_id para resolver ground_truth_cls desde CSV de split."
        ) from None

    from_lookup = class_lookup.get(row_image_id)
    if from_lookup:
        return normalize_polyp_class(from_lookup)

    raise ValueError(
        "No se encontró ground_truth_cls en CSV de split para "
        f"image_id='{row_image_id}'."
    )


def draw_bbox_and_save_temp_image(
    image_path: Path,
    bbox_norm: list[int],
) -> io.BytesIO:
    """Dibuja un bbox GT y devuelve la imagen en memoria (sin escribir en disco)."""
    image = _draw_bbox_overlay(image_path=image_path, bbox_norm=bbox_norm)

    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), 95],
    )
    if not ok:
        raise RuntimeError(f"No se pudo codificar imagen forzada en memoria: {image_path}")

    buffer = io.BytesIO(encoded.tobytes())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    buffer.name = f"{image_path.stem}_forced_bbox_{timestamp}.jpg"
    buffer.seek(0)
    return buffer


def draw_gt_bbox_and_save_artifact(
    image_path: Path,
    bbox_norm: list[int],
    *,
    output_path: Path,
    label: str | None = None,
) -> Path:
    """Dibuja solo el bbox GT sobre la imagen original y guarda el artefacto."""
    image = _draw_bbox_overlay(image_path=image_path, bbox_norm=bbox_norm)

    if label:
        cv2.putText(
            image,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), image)
    if not ok:
        raise RuntimeError(f"No se pudo escribir imagen anotada en: {output_path}")
    return output_path


def crop_roi_and_save_temp_image(
    image_path: Path,
    bbox_norm: list[int],
    padding_pct: float = 0.1,
) -> io.BytesIO:
    """Recorta la ROI del bbox GT con padding y devuelve JPEG en memoria."""
    if len(bbox_norm) != 4:
        raise ValueError("bbox_norm debe contener 4 elementos: [xmin, ymin, xmax, ymax]")

    if padding_pct < 0:
        raise ValueError("padding_pct debe ser mayor o igual que 0")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"No se pudo leer la imagen con OpenCV: {image_path}")

    height, width = image.shape[:2]

    try:
        xmin = int(float(bbox_norm[0]) * float(width) / 1000.0)
        ymin = int(float(bbox_norm[1]) * float(height) / 1000.0)
        xmax = int(float(bbox_norm[2]) * float(width) / 1000.0)
        ymax = int(float(bbox_norm[3]) * float(height) / 1000.0)
    except Exception as exc:
        raise ValueError("bbox_norm contiene valores no numéricos") from exc

    y1 = max(0, min(height - 1, min(ymin, ymax)))
    y2 = max(0, min(height, max(ymin, ymax)))
    x1 = max(0, min(width - 1, min(xmin, xmax)))
    x2 = max(0, min(width, max(xmin, xmax)))

    bbox_h = max(1, y2 - y1)
    bbox_w = max(1, x2 - x1)

    pad_y = int(round(float(bbox_h) * float(padding_pct)))
    pad_x = int(round(float(bbox_w) * float(padding_pct)))

    crop_y1 = max(0, y1 - pad_y)
    crop_y2 = min(height, y2 + pad_y)
    crop_x1 = max(0, x1 - pad_x)
    crop_x2 = min(width, x2 + pad_x)

    if crop_y2 <= crop_y1:
        crop_y2 = min(height, crop_y1 + 1)
    if crop_x2 <= crop_x1:
        crop_x2 = min(width, crop_x1 + 1)

    cropped_img = image[crop_y1:crop_y2, crop_x1:crop_x2]
    if cropped_img.size == 0:
        raise RuntimeError(f"El recorte ROI quedó vacío para la imagen: {image_path}")

    ok, encoded = cv2.imencode(
        ".jpg",
        cropped_img,
        [int(cv2.IMWRITE_JPEG_QUALITY), 95],
    )
    if not ok:
        raise RuntimeError(f"No se pudo codificar ROI en memoria: {image_path}")

    buffer = io.BytesIO(encoded.tobytes())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    buffer.name = f"{image_path.stem}_roi_crop_{timestamp}.jpg"
    buffer.seek(0)
    return buffer


def _draw_bbox_overlay(*, image_path: Path, bbox_norm: list[int]) -> Any:
    """Construye imagen OpenCV con el bbox dibujado en rojo."""
    if len(bbox_norm) != 4:
        raise ValueError("bbox_norm debe contener 4 elementos: [xmin, ymin, xmax, ymax]")

    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"No se pudo leer la imagen con OpenCV: {image_path}")

    height, width = image.shape[:2]

    try:
        xmin = int(float(bbox_norm[0]) * float(width) / 1000.0)
        ymin = int(float(bbox_norm[1]) * float(height) / 1000.0)
        xmax = int(float(bbox_norm[2]) * float(width) / 1000.0)
        ymax = int(float(bbox_norm[3]) * float(height) / 1000.0)
    except Exception as exc:
        raise ValueError("bbox_norm contiene valores no numéricos") from exc

    # Recorta coordenadas para evitar errores de dibujo fuera de rango.
    ymin = max(0, min(height - 1, ymin))
    ymax = max(0, min(height - 1, ymax))
    xmin = max(0, min(width - 1, xmin))
    xmax = max(0, min(width - 1, xmax))

    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=3)
    return image


def save_result(output_file: str, result_dict: dict[str, Any]) -> None:
    """Backward-compatible wrapper for JSONL append helper."""
    _save_result(output_file=output_file, result_dict=result_dict)


def initialize_scenario_result_skeleton(*, output_path: Path, skeleton_records: list[dict[str, Any]]) -> None:
    """Initialize a full pending skeleton for scenario records in JSONL."""
    _initialize_scenario_result_skeleton(
        output_path=output_path,
        skeleton_records=skeleton_records,
        scenario_meta_key=SCENARIO_META_KEY,
        scenario_summary_key=SCENARIO_SUMMARY_KEY,
    )


def upsert_scenario_result_record(*, output_path: Path, result_dict: dict[str, Any]) -> None:
    """Replace skeleton/pending line for image_id with final image result."""
    _upsert_scenario_result_record(
        output_path=output_path,
        result_dict=result_dict,
        normalize_image_stem=normalize_image_stem,
        scenario_meta_key=SCENARIO_META_KEY,
        scenario_summary_key=SCENARIO_SUMMARY_KEY,
    )


def has_unfilled_scenario_records(*, output_path: Path) -> bool:
    """Return True when scenario JSONL contains pending/skeleton lines."""
    return _has_unfilled_scenario_records(
        output_path=output_path,
        scenario_meta_key=SCENARIO_META_KEY,
        scenario_summary_key=SCENARIO_SUMMARY_KEY,
    )


def load_jsonl_records(path: Path, *, include_system_records: bool = False) -> list[dict[str, Any]]:
    """Backward-compatible wrapper for JSONL record loader."""
    return _load_jsonl_records(
        path=path,
        include_system_records=include_system_records,
        scenario_meta_key=SCENARIO_META_KEY,
        scenario_summary_key=SCENARIO_SUMMARY_KEY,
    )


def collect_processed_image_ids_from_jsonl(*, output_jsonl_path: Path) -> set[str]:
    """Backward-compatible wrapper for processed-id extraction from JSONL."""
    return _collect_processed_image_ids_from_jsonl(
        output_jsonl_path=output_jsonl_path,
        normalize_image_stem=normalize_image_stem,
        load_jsonl_records=load_jsonl_records,
    )


def upsert_scenario_meta_header(
    *,
    output_path: Path,
    scenario_name: str,
    model_id: str,
    input_csv: str,
    img_dir: str,
    seed: int | None,
    requested_limit: int | None,
    resume_mode: bool,
) -> None:
    """Backward-compatible wrapper for scenario metadata upsert."""
    _upsert_scenario_meta_header(
        output_path=output_path,
        scenario_name=scenario_name,
        model_id=model_id,
        input_csv=input_csv,
        img_dir=img_dir,
        seed=seed,
        requested_limit=requested_limit,
        resume_mode=resume_mode,
        scenario_meta_key=SCENARIO_META_KEY,
    )


def upsert_scenario_execution_summary(*, output_path: Path, summary_payload: dict[str, Any]) -> None:
    """Backward-compatible wrapper for scenario summary upsert."""
    _upsert_scenario_execution_summary(
        output_path=output_path,
        summary_payload=summary_payload,
        scenario_summary_key=SCENARIO_SUMMARY_KEY,
    )


def emit_report_event(reporter: Reporter | None, event: str, **payload: Any) -> None:
    """Emit optional progress event for CLI/TUI adapters."""
    if reporter is not None:
        reporter(event, payload)


def default_scenario_output_path(*, scenario_name: str) -> Path:
    """Build the default JSONL path for a scenario-specific run directory."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_SCENARIO_RESULTS_DIR / scenario_name / f"run_{run_id}" / "results.jsonl"


def resolve_output_jsonl_path(*, raw_output: str | None, default_output: Path) -> Path:
    """Resolve output path and enforce `.jsonl` extension."""
    if raw_output is None or not str(raw_output).strip():
        return default_output

    output = Path(str(raw_output).strip())
    if output.suffix.lower() != ".jsonl":
        output = output.with_suffix(".jsonl")
    return output


def prepare_run_artifact_dirs(*, output_jsonl_path: Path, images_subdir_name: str = "annotated") -> tuple[Path, Path]:
    """Create run directory and annotated images directory for scenario artifacts."""
    run_dir = output_jsonl_path.parent
    images_dir = run_dir / images_subdir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, images_dir


def resolve_image_path_from_id(
    *,
    image_id: Any,
    img_dir: Path,
    image_extensions: tuple[str, ...] = DEFAULT_IMAGE_EXTENSIONS,
) -> Path:
    """Resolve image path by id, trying common extensions when extension is absent."""
    stem_or_name = normalize_image_stem(image_id)
    base_candidate = img_dir / stem_or_name

    if base_candidate.suffix:
        if base_candidate.is_file():
            return base_candidate
        raise FileNotFoundError(f"Imagen no encontrada: {base_candidate}")

    for ext in image_extensions:
        candidate = img_dir / f"{stem_or_name}{ext}"
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"No se encontró imagen para image_id='{stem_or_name}' en {img_dir} "
        f"con extensiones {image_extensions}."
    )


def serialize_telemetry_payload(telemetry: Any) -> dict[str, Any]:
    """Serialize telemetry from dataclass or plain object into dictionary."""
    if dataclasses.is_dataclass(telemetry) and not isinstance(telemetry, type):
        return dataclasses.asdict(telemetry)
    return dict(getattr(telemetry, "__dict__", {}))


def select_telemetry_fields_for_record(telemetry_payload: dict[str, Any] | None) -> dict[str, Any]:
    """Backward-compatible wrapper for telemetry field selection."""
    return _select_telemetry_fields_for_record(
        telemetry_payload,
        telemetry_fields_map=SCENARIO_TELEMETRY_RECORD_FIELDS,
    )


def build_scenario_record(
    *,
    scenario_name: str,
    model_id: str,
    schema_name: str,
    image_id: Any,
    image_path: Path,
    status: str,
    duration_seconds: float,
    ground_truth_bbox: list[Any] | None,
    ground_truth_cls: str | None,
    payload: dict[str, Any] | None,
    telemetry_payload: dict[str, Any] | None,
    class_match: bool | None,
    iou_score: float | None,
    proximity_score: float | None = None,
    proximity_center_score: float | None = None,
    proximity_size_score: float | None = None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    """Backward-compatible wrapper for scenario-record serialization."""
    return _build_scenario_record(
        scenario_name=scenario_name,
        model_id=model_id,
        schema_name=schema_name,
        image_id=image_id,
        image_path=image_path,
        status=status,
        duration_seconds=duration_seconds,
        ground_truth_bbox=ground_truth_bbox,
        ground_truth_cls=ground_truth_cls,
        payload=payload,
        telemetry_payload=telemetry_payload,
        class_match=class_match,
        iou_score=iou_score,
        proximity_score=proximity_score,
        proximity_center_score=proximity_center_score,
        proximity_size_score=proximity_size_score,
        telemetry_fields_map=SCENARIO_TELEMETRY_RECORD_FIELDS,
        error=error,
    )


def safe_inference_with_optional_telemetry(
    *,
    loader: Any,
    image_path: Any,
    prompt: str,
    schema: type[SchemaT],
) -> tuple[SchemaT, dict[str, Any] | None]:
    """Run inference and return parsed schema plus optional telemetry payload."""
    result = loader.inference(
        image_path=image_path,
        prompt=prompt,
        schema=schema,
        include_telemetry=True,
    )

    if hasattr(result, "data") and hasattr(result, "telemetry"):
        parsed = getattr(result, "data")
        telemetry = serialize_telemetry_payload(getattr(result, "telemetry"))
        return cast(SchemaT, parsed), telemetry

    if isinstance(result, schema):
        return result, None

    raise TypeError(
        "El loader devolvió un tipo de respuesta no esperado para el schema solicitado."
    )


def normalize_bbox_for_metrics(values: list[Any]) -> list[int] | None:
    """Normalize bbox values to integer [xmin, ymin, xmax, ymax] format."""
    if len(values) != 4:
        return None
    normalized: list[int] = []
    for value in values:
        try:
            normalized.append(int(round(float(value))))
        except Exception:
            return None
    return normalized


def compute_iou_safe(*, gt_bbox: list[Any], pred_bbox: list[Any]) -> float | None:
    """Compute IoU safely from raw bbox values; returns None when invalid."""
    gt_norm = normalize_bbox_for_metrics(gt_bbox)
    pred_norm = normalize_bbox_for_metrics(pred_bbox)
    if gt_norm is None or pred_norm is None:
        return None

    try:
        return float(calculate_iou(gt_norm, pred_norm))
    except Exception:
        return None


def compute_proximity_safe(*, gt_bbox: list[Any], pred_bbox: list[Any]) -> dict[str, float] | None:
    """Compute proximity score/components safely from raw bbox values."""
    gt_norm = normalize_bbox_for_metrics(gt_bbox)
    pred_norm = normalize_bbox_for_metrics(pred_bbox)
    if gt_norm is None or pred_norm is None:
        return None

    try:
        return calculate_proximity_score(gt_norm, pred_norm)
    except Exception:
        return None


def write_comparison_image(
    *,
    image_path: Path,
    output_path: Path,
    gt_bbox: list[Any],
    pred_bbox: list[Any],
    model_name: str,
    gt_label: str,
    pred_label: str,
    iou_score: float | None,
) -> Path | None:
    """Backward-compatible wrapper for comparison image writer."""
    return _write_comparison_image(
        image_path=image_path,
        output_path=output_path,
        gt_bbox=gt_bbox,
        pred_bbox=pred_bbox,
        model_name=model_name,
        gt_label=gt_label,
        pred_label=pred_label,
        iou_score=iou_score,
    )


def build_markdown_records_from_scenario_jsonl(*, output_path: Path, run_dir: Path) -> list[dict[str, Any]]:
    """Backward-compatible wrapper for markdown-record builder module."""
    return _build_markdown_records_from_scenario_jsonl(output_path=output_path, run_dir=run_dir)


def summarize_scenario_records_from_jsonl(*, output_path: Path) -> dict[str, Any]:
    """Backward-compatible wrapper for cumulative JSONL summary aggregation."""
    return _summarize_scenario_records_from_jsonl(
        output_path=output_path,
        load_jsonl_records=load_jsonl_records,
        normalize_polyp_class=normalize_polyp_class,
        compute_iou_safe=compute_iou_safe,
        compute_proximity_safe=compute_proximity_safe,
    )


def write_class_confusion_heatmap(
    *,
    run_dir: Path,
    records: list[dict[str, Any]],
    scenario_name: str,
) -> Path | None:
    """Backward-compatible wrapper for heatmap writer module."""
    return _write_class_confusion_heatmap(
        run_dir=run_dir,
        records=records,
        scenario_name=scenario_name,
    )


def generate_single_detection_markdown_report(
    *,
    run_dir: Path,
    report_title: str,
    scenario_name: str,
    model_id: str,
    jsonl_path: Path,
    records: list[dict[str, Any]],
) -> Path:
    """Backward-compatible wrapper for markdown report generator module."""
    return _generate_single_detection_markdown_report(
        run_dir=run_dir,
        report_title=report_title,
        scenario_name=scenario_name,
        model_id=model_id,
        jsonl_path=jsonl_path,
        records=records,
    )


def generate_scenario_s_markdown_report(
    *,
    run_dir: Path,
    output_path: Path,
    model_id: str,
    level: int,
    seed: int | None,
    summary: dict[str, Any],
    kpis: dict[str, Any],
) -> Path:
    """Backward-compatible wrapper for Scenario S markdown report generator module."""
    return _generate_scenario_s_markdown_report(
        run_dir=run_dir,
        output_path=output_path,
        model_id=model_id,
        level=level,
        seed=seed,
        summary=summary,
        kpis=kpis,
    )


__all__ = [
    "AssistedClinicalReport",
    "DEFAULT_GROUND_TRUTH_CSV",
    "DEFAULT_SCENARIO_RESULTS_DIR",
    "CLASS_COLUMN_CANDIDATES",
    "DEFAULT_IMAGE_EXTENSIONS",
    "ID_COLUMN_CANDIDATES",
    "PolypDiagnosisAndGrounding",
    "SCENARIO_META_KEY",
    "SCENARIO_SUMMARY_KEY",
    "SCENARIO_TELEMETRY_RECORD_FIELDS",
    "build_class_lookup_from_m_split_csvs",
    "build_bbox_xyxy",
    "build_bbox_xyxy_from_mapping",
    "build_markdown_records_from_scenario_jsonl",
    "build_scenario_record",
    "build_class_confusion_matrix",
    "build_classwise_heatmap_explanations",
    "build_top_confusion_errors",
    "collect_processed_image_ids_from_jsonl",
    "compute_macro_f1_and_recall_by_class",
    "compute_iou_safe",
    "compute_proximity_safe",
    "compute_classification_accuracy_from_records",
    "default_scenario_output_path",
    "emit_report_event",
    "extract_predicted_class",
    "draw_bbox_and_save_temp_image",
    "crop_roi_and_save_temp_image",
    "generate_scenario_s_markdown_report",
    "generate_single_detection_markdown_report",
    "has_unfilled_scenario_records",
    "load_ground_truth",
    "load_jsonl_records",
    "initialize_scenario_result_skeleton",
    "normalize_bbox_for_metrics",
    "normalize_image_stem",
    "normalize_polyp_class",
    "build_annotated_comparison_filename",
    "prepare_run_artifact_dirs",
    "resolve_image_path_from_id",
    "resolve_ground_truth_class_from_lookup",
    "resolve_output_jsonl_path",
    "safe_inference_with_optional_telemetry",
    "select_telemetry_fields_for_record",
    "select_ground_truth_rows",
    "save_result",
    "upsert_scenario_result_record",
    "serialize_telemetry_payload",
    "summarize_scenario_records_from_jsonl",
    "write_class_confusion_heatmap",
    "upsert_scenario_execution_summary",
    "upsert_scenario_meta_header",
    "write_comparison_image",
]
