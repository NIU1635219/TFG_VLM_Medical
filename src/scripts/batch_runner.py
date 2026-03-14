"""Orquestador CLI para inferencia masiva sobre directorios de imágenes."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pydantic import BaseModel

from src.inference.schemas import SCHEMA_REGISTRY, get_schema_variant
from src.inference.vlm_runner import VLMLoader
from src.scripts.test_schema import build_prompt_for_schema

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable: Any, **_: Any) -> Any:
        return iterable


_IMG_EXT: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
_TELEMETRY_RECORD_FIELDS: dict[str, str] = {
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
_MANIFEST_BASE_KEYS: set[str] = {
    "timestamp_utc",
    "model_id",
    "schema_name",
    "image_path",
    "image_name",
    "status",
    "duration_seconds",
    "error_type",
    "error_message",
    "payload",
}
_MANIFEST_BASE_KEYS.update(_TELEMETRY_RECORD_FIELDS.keys())
_BATCH_META_KEY = "__batch_meta__"
_BATCH_SUMMARY_KEY = "__batch_summary__"
_MANIFEST_META_KEY = "__manifest_meta__"


@dataclasses.dataclass(frozen=True)
class BatchInputItem:
    """Representa una entrada de inferencia con metadatos opcionales."""

    image_path: Path
    metadata: dict[str, Any] | None = None


def _filter_items_by_pending_entries(
    *,
    discovered_items: list[BatchInputItem],
    pending_entries: list[dict[str, Any]],
    manifest_path: Path,
) -> list[BatchInputItem]:
    """Filtra entradas descubiertas usando clave (ruta, iteración) pendiente."""
    pending_keys: set[tuple[str, int]] = set()
    resolved_manifest_path = manifest_path.expanduser().resolve()
    for pending_entry in pending_entries:
        if not isinstance(pending_entry, dict):
            continue
        resolved_pending_path = _resolve_manifest_image_path(
            pending_entry.get("image_path"),
            resolved_manifest_path,
        )
        if resolved_pending_path is None:
            continue
        pending_keys.add(
            (
                _as_posix_key(resolved_pending_path),
                _safe_positive_int(pending_entry.get("run_iteration_index")),
            )
        )

    if not pending_keys:
        return []

    return [
        item
        for item in discovered_items
        if _manifest_entry_key(image_path=item.image_path, metadata=item.metadata) in pending_keys
    ]


def _filter_items_by_pending_image_paths(
    *,
    discovered_items: list[BatchInputItem],
    pending_image_paths: list[str],
) -> list[BatchInputItem]:
    """Filtra entradas descubiertas por path de imagen pendiente."""
    pending_set = {
        _as_posix_key(path)
        for path in pending_image_paths
        if str(path).strip()
    }
    if not pending_set:
        return []

    return [
        item
        for item in discovered_items
        if _as_posix_key(item.image_path) in pending_set
    ]


def _update_metric_totals(
    *,
    totals: dict[str, float],
    counts: dict[str, int],
    telemetry_payload: dict[str, Any],
) -> None:
    """Acumula métricas numéricas de telemetría para promedios de resumen."""
    metric_key_mapping = {
        "ttft_total": "ttft_seconds",
        "tps_total": "tokens_per_second",
        "total_duration_total": "total_duration_seconds",
    }
    count_key_mapping = {
        "ttft_total": "ttft_count",
        "tps_total": "tps_count",
        "total_duration_total": "total_duration_count",
    }

    for total_key, telemetry_key in metric_key_mapping.items():
        metric_value = telemetry_payload.get(telemetry_key)
        if not isinstance(metric_value, (int, float)):
            continue
        totals[total_key] += float(metric_value)
        counts[count_key_mapping[total_key]] += 1


T = TypeVar("T")


def build_parser() -> argparse.ArgumentParser:
    """Construye el parser CLI del batch runner."""
    parser = argparse.ArgumentParser(
        description=(
            "Procesa un directorio de imágenes con un VLM en LM Studio y guarda "
            "resultados incrementales en JSONL."
        )
    )
    parser.add_argument("--model", required=True, help="Identificador del modelo en LM Studio.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--image-dir", help="Directorio raíz de imágenes a procesar.")
    source_group.add_argument("--manifest", help="Ruta a un JSONL con al menos el campo image_path por línea.")
    parser.add_argument(
        "--schema",
        required=True,
        help="Nombre del esquema base registrado (por ejemplo, PolypDetection).",
    )
    parser.add_argument(
        "--with-reasoning",
        action="store_true",
        help="Usa la variante del esquema que exige reasoning explícito.",
    )
    parser.add_argument(
        "--output",
        help="Ruta del archivo de salida. Si se omite, se genera en data/processed/batch_results/.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Número máximo de imágenes a procesar. Si se omite, procesa todas.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Baraja las imágenes antes de seleccionar el subconjunto a procesar.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semilla aleatoria usada cuando --shuffle está activo.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperatura de inferencia. Por defecto 0.0 para mayor determinismo.",
    )
    parser.add_argument("--prompt", help="Prompt manual. Si se omite, se construye desde el esquema.")
    parser.add_argument("--server-api-host", help="Host del servidor LM Studio si no es el local por defecto.")
    parser.add_argument("--api-token", help="Token de API para LM Studio si aplica.")
    parser.add_argument("--verbose", action="store_true", help="Activa logs verbosos del loader.")
    return parser


def iter_image_paths(image_dir: Path) -> list[Path]:
    """Descubre imágenes compatibles bajo un directorio o archivo concreto."""
    normalized_path = image_dir.expanduser().resolve()
    if not normalized_path.exists():
        raise FileNotFoundError(f"No existe la ruta indicada: {normalized_path}")

    if normalized_path.is_file():
        if normalized_path.suffix.lower() not in _IMG_EXT:
            raise ValueError(f"El archivo no tiene una extensión de imagen soportada: {normalized_path}")
        return [normalized_path]

    image_paths = [
        path.resolve()
        for path in sorted(normalized_path.rglob("*"))
        if path.is_file() and path.suffix.lower() in _IMG_EXT
    ]
    if not image_paths:
        raise RuntimeError(f"No se encontraron imágenes compatibles dentro de: {normalized_path}")
    return image_paths


def _resolve_manifest_image_path(raw_value: Any, manifest_path: Path) -> Path | None:
    """Resuelve una ruta de imagen desde el manifiesto admitiendo rutas relativas."""
    if raw_value is None:
        return None
    candidate_text = str(raw_value).strip()
    if not candidate_text:
        return None

    candidate = Path(candidate_text).expanduser()
    candidate_pool: list[Path] = []
    if candidate.is_absolute():
        candidate_pool.append(candidate)
    else:
        candidate_pool.append(manifest_path.parent / candidate)
        candidate_pool.append(Path(_PROJECT_ROOT) / candidate)

    for path in candidate_pool:
        resolved = path.resolve()
        if resolved.exists() and resolved.is_file():
            return resolved

    return None


def _safe_positive_int(value: Any, default: int = 1) -> int:
    """Convierte valores a entero positivo con fallback seguro."""
    try:
        return max(1, int(value or default))
    except (TypeError, ValueError):
        return default


def _as_posix_key(path_like: str | Path) -> str:
    """Normaliza rutas a string POSIX estable para comparaciones."""
    return str(Path(path_like).as_posix()).strip()


def _manifest_entry_key(*, image_path: Path, metadata: dict[str, Any] | None) -> tuple[str, int]:
    """Construye clave de entrada para diferenciar iteraciones de una misma imagen."""
    iteration_index = _safe_positive_int((metadata or {}).get("run_iteration_index"))
    return _as_posix_key(image_path), iteration_index


def iter_manifest_items(manifest_path: Path) -> tuple[list[BatchInputItem], int]:
    """Lee un manifiesto JSONL y devuelve entradas válidas junto al número descartado."""
    normalized_manifest = manifest_path.expanduser().resolve()
    if not normalized_manifest.exists() or not normalized_manifest.is_file():
        raise FileNotFoundError(f"No existe el manifiesto indicado: {normalized_manifest}")

    discarded_rows = 0
    items: list[BatchInputItem] = []
    with normalized_manifest.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                discarded_rows += 1
                continue
            if not isinstance(payload, dict):
                discarded_rows += 1
                continue

            # Metadatos globales del manifiesto (config de ejecución). No son una entrada de imagen.
            if isinstance(payload.get(_MANIFEST_META_KEY), dict):
                continue

            image_path = _resolve_manifest_image_path(payload.get("image_path"), normalized_manifest)
            if image_path is None:
                discarded_rows += 1
                continue

            metadata = {key: value for key, value in payload.items() if key != "image_path"}
            items.append(BatchInputItem(image_path=image_path, metadata=metadata or None))

    if not items:
        raise RuntimeError(f"No se encontraron entradas válidas en el manifiesto: {normalized_manifest}")

    return items, discarded_rows


def resolve_schema(schema_name: str, include_reasoning: bool) -> tuple[str, type[BaseModel]]:
    """Resuelve el esquema final a partir del nombre solicitado por CLI."""
    normalized_name = schema_name.strip()
    if normalized_name.endswith("WithReasoning"):
        normalized_name = normalized_name.removesuffix("WithReasoning")
        include_reasoning = True

    if normalized_name not in SCHEMA_REGISTRY:
        available = ", ".join(sorted(SCHEMA_REGISTRY.keys()))
        raise ValueError(f"Esquema desconocido '{schema_name}'. Disponibles: {available}")

    return get_schema_variant(normalized_name, include_reasoning)


def select_images(
    image_paths: list[T],
    *,
    max_images: int | None = None,
    shuffle: bool = False,
    seed: int | None = None,
) -> list[T]:
    """Selecciona el subconjunto final de imágenes a procesar."""
    selected = list(image_paths)
    if shuffle:
        random.Random(seed).shuffle(selected)

    if max_images is not None:
        if max_images <= 0:
            raise ValueError("max_images debe ser mayor que cero")
        selected = selected[:max_images]

    return selected


def build_default_output_path(schema_name: str) -> Path:
    """Genera una ruta JSONL por defecto bajo data/processed/batch_results."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path(_PROJECT_ROOT) / "data" / "processed" / "batch_results" / f"batch_{schema_name}_{timestamp}.jsonl"


def _sync_file(file_handle: Any) -> None:
    """Fuerza el volcado a disco tras cada append para minimizar pérdida de datos."""
    file_handle.flush()
    os.fsync(file_handle.fileno())


def append_jsonl_record(output_path: Path, record: dict[str, Any]) -> None:
    """Añade un registro JSONL y lo fuerza a disco inmediatamente."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8", newline="\n") as file_handle:
        file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        _sync_file(file_handle)


def _parse_jsonl_dict(line: str) -> dict[str, Any] | None:
    """Intenta parsear una línea JSONL como diccionario."""
    try:
        payload = json.loads(line)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _as_float(value: Any) -> float | None:
    """Convierte un valor escalar a float cuando es posible."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _empty_metric_agg() -> dict[str, float | int | None]:
    """Inicializa un acumulador para min/max/media."""
    return {
        "count": 0,
        "sum": 0.0,
        "min": None,
        "max": None,
    }


def _update_metric_agg(agg: dict[str, float | int | None], value: float | None) -> None:
    """Actualiza acumulador de métrica con un nuevo valor válido."""
    if value is None:
        return
    count = int(agg.get("count", 0) or 0) + 1
    total_sum = float(agg.get("sum", 0.0) or 0.0) + float(value)
    current_min = agg.get("min")
    current_max = agg.get("max")
    agg["count"] = count
    agg["sum"] = total_sum
    agg["min"] = float(value) if current_min is None else min(float(current_min), float(value))
    agg["max"] = float(value) if current_max is None else max(float(current_max), float(value))


def _metric_agg_summary(agg: dict[str, float | int | None]) -> dict[str, float | int | None]:
    """Materializa estadísticas min/max/media desde un acumulador."""
    count = int(agg.get("count", 0) or 0)
    total_sum = float(agg.get("sum", 0.0) or 0.0)
    avg: float | None = None
    if count > 0:
        avg = total_sum / count
    return {
        "count": count,
        "min": agg.get("min"),
        "max": agg.get("max"),
        "avg": avg,
    }


def _read_jsonl_lines(path: Path) -> list[str]:
    """Lee líneas de un JSONL sin saltos de línea finales."""
    if not path.exists() or not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def _write_jsonl_lines(path: Path, lines: list[str]) -> None:
    """Reescribe un JSONL completo de forma atómica para cabeceras/resúmenes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        if lines:
            handle.write("\n".join(lines) + "\n")
        _sync_file(handle)

def _ensure_batch_meta_header(
    *,
    output_path: Path,
    model_id: str,
    schema_name: str,
    input_source: str,
    manifest_path: Path | None,
) -> None:
    """Upsert de metadatos globales para JSONL compartido entre modelos."""
    lines = _read_jsonl_lines(output_path)
    meta_index: int | None = None
    existing_meta: dict[str, Any] = {}

    for index, line in enumerate(lines):
        payload = _parse_jsonl_dict(line.strip())
        if not isinstance(payload, dict):
            continue
        meta = payload.get(_BATCH_META_KEY)
        if isinstance(meta, dict):
            meta_index = index
            existing_meta = dict(meta)
            break

    now_utc = datetime.now(timezone.utc).isoformat()

    model_ids: set[str] = set()
    raw_model_ids = existing_meta.get("model_ids")
    if isinstance(raw_model_ids, list):
        model_ids.update(str(item).strip() for item in raw_model_ids if str(item).strip())
    current_model = str(model_id or "").strip()
    if current_model:
        model_ids.add(current_model)

    schema_names: set[str] = set()
    raw_schema_names = existing_meta.get("schema_names")
    if isinstance(raw_schema_names, list):
        schema_names.update(str(item).strip() for item in raw_schema_names if str(item).strip())
    current_schema = str(schema_name or "").strip()
    if current_schema:
        schema_names.add(current_schema)

    input_sources: set[str] = set()
    raw_input_sources = existing_meta.get("input_sources")
    if isinstance(raw_input_sources, list):
        input_sources.update(str(item).strip() for item in raw_input_sources if str(item).strip())
    current_input = str(input_source or "").strip()
    if current_input:
        input_sources.add(current_input)

    manifest_paths: set[str] = set()
    raw_manifest_paths = existing_meta.get("manifest_paths")
    if isinstance(raw_manifest_paths, list):
        manifest_paths.update(str(item).strip() for item in raw_manifest_paths if str(item).strip())
    if manifest_path is not None:
        manifest_paths.add(str(manifest_path.resolve()))

    ordered_models = sorted(model_ids)
    ordered_schemas = sorted(schema_names)
    ordered_inputs = sorted(input_sources)
    ordered_manifests = sorted(manifest_paths)

    meta_payload: dict[str, Any] = {
        "version": 2,
        "created_at_utc": str(existing_meta.get("created_at_utc") or now_utc),
        "updated_at_utc": now_utc,
        "output_mode": "shared_jsonl",
        "model_ids": ordered_models,
        "schema_names": ordered_schemas,
        "input_sources": ordered_inputs,
    }
    if ordered_manifests:
        meta_payload["manifest_paths"] = ordered_manifests

    header_line = json.dumps({_BATCH_META_KEY: meta_payload}, ensure_ascii=False)
    if meta_index is None:
        lines.insert(0, header_line)
    else:
        lines[meta_index] = header_line

    _write_jsonl_lines(output_path, lines)


def _normalize_label_for_accuracy(value: Any) -> str:
    """Normaliza etiquetas de clase para comparar aciertos sin ruido de formato."""
    if value is None:
        return ""
    return str(value).strip().lower()


def _aggregate_models_from_jsonl_lines(
    *,
    lines: list[str],
    selected_models: set[str],
    schema_name: str,
    metric_keys: list[str],
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, dict[str, float | int | None]]]:
    """Agrega métricas y exactitud por modelo desde líneas JSONL."""
    kept_lines: list[str] = []
    model_acc: dict[str, dict[str, Any]] = {}
    global_metric_acc: dict[str, dict[str, float | int | None]] = {
        key: _empty_metric_agg() for key in metric_keys
    }

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        payload = _parse_jsonl_dict(line)
        if payload is None:
            kept_lines.append(raw_line)
            continue
        if isinstance(payload.get(_BATCH_SUMMARY_KEY), dict):
            # Reemplazamos cualquier resumen previo por uno nuevo actualizado.
            continue

        kept_lines.append(raw_line)

        if isinstance(payload.get(_BATCH_META_KEY), dict):
            continue

        model_id_value = str(payload.get("model_id") or "").strip()
        if not model_id_value:
            continue
        if selected_models and model_id_value not in selected_models:
            continue
        schema_value = str(payload.get("schema_name") or "").strip()
        if schema_name and schema_value and schema_value != schema_name:
            continue

        status_value = str(payload.get("status") or "").strip().lower()
        model_entry = model_acc.setdefault(
            model_id_value,
            {
                "processed": 0,
                "ok": 0,
                "invalid": 0,
                "error": 0,
                "eval_total": 0,
                "eval_correct": 0,
                "metrics": {key: _empty_metric_agg() for key in metric_keys},
            },
        )
        model_entry["processed"] = int(model_entry["processed"]) + 1
        if status_value == "ok":
            model_entry["ok"] = int(model_entry["ok"]) + 1
        elif status_value == "invalid":
            model_entry["invalid"] = int(model_entry["invalid"]) + 1
        else:
            model_entry["error"] = int(model_entry["error"]) + 1

        metric_bucket = model_entry["metrics"]
        for metric_key in metric_keys:
            value = _as_float(payload.get(metric_key))
            _update_metric_agg(metric_bucket[metric_key], value)
            _update_metric_agg(global_metric_acc[metric_key], value)

        payload_obj = payload.get("payload")
        predicted_label = ""
        if isinstance(payload_obj, dict):
            predicted_label = _normalize_label_for_accuracy(payload_obj.get("predicted_class"))
        ground_truth_label = _normalize_label_for_accuracy(payload.get("ground_truth_cls"))
        if predicted_label and ground_truth_label:
            model_entry["eval_total"] = int(model_entry["eval_total"]) + 1
            if predicted_label == ground_truth_label:
                model_entry["eval_correct"] = int(model_entry["eval_correct"]) + 1

    return kept_lines, model_acc, global_metric_acc


def _repair_summary_meta_line(
    *,
    kept_lines: list[str],
    model_acc: dict[str, dict[str, Any]],
    schema_name: str,
) -> None:
    """Repara/actualiza cabecera __batch_meta__ en formato canónico."""
    for index, raw_line in enumerate(kept_lines):
        payload = _parse_jsonl_dict(raw_line.strip())
        if payload is None:
            continue
        meta = payload.get(_BATCH_META_KEY)
        if not isinstance(meta, dict):
            continue

        existing_model_ids: set[str] = set()
        raw_model_ids = meta.get("model_ids")
        if isinstance(raw_model_ids, list):
            existing_model_ids.update(str(item).strip() for item in raw_model_ids if str(item).strip())
        existing_model_ids.update(model_acc.keys())

        schema_names: set[str] = set()
        raw_schema_names = meta.get("schema_names")
        if isinstance(raw_schema_names, list):
            schema_names.update(str(item).strip() for item in raw_schema_names if str(item).strip())
        if schema_name:
            schema_names.add(schema_name)

        input_sources: set[str] = set()
        raw_input_sources = meta.get("input_sources")
        if isinstance(raw_input_sources, list):
            input_sources.update(str(item).strip() for item in raw_input_sources if str(item).strip())

        manifest_paths: set[str] = set()
        raw_manifest_paths = meta.get("manifest_paths")
        if isinstance(raw_manifest_paths, list):
            manifest_paths.update(str(item).strip() for item in raw_manifest_paths if str(item).strip())

        ordered_models = sorted(existing_model_ids)
        ordered_schemas = sorted(schema_names)
        ordered_inputs = sorted(input_sources)
        ordered_manifests = sorted(manifest_paths)

        repaired_meta: dict[str, Any] = {
            "version": max(int(meta.get("version", 1) or 1), 2),
            "created_at_utc": str(meta.get("created_at_utc") or datetime.now(timezone.utc).isoformat()),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_mode": "shared_jsonl",
            "model_ids": ordered_models,
            "schema_names": ordered_schemas,
            "input_sources": ordered_inputs,
        }
        if ordered_manifests:
            repaired_meta["manifest_paths"] = ordered_manifests

        kept_lines[index] = json.dumps({_BATCH_META_KEY: repaired_meta}, ensure_ascii=False)
        break


def _materialize_summary_payload(
    *,
    model_acc: dict[str, dict[str, Any]],
    global_metric_acc: dict[str, dict[str, float | int | None]],
    metric_keys: list[str],
    schema_name: str,
) -> dict[str, Any]:
    """Construye el payload final de __batch_summary__."""
    models_summary: dict[str, Any] = {}
    total_processed = 0
    total_ok = 0
    total_invalid = 0
    total_error = 0
    total_eval = 0
    total_correct = 0
    for model_name in sorted(model_acc.keys()):
        item = model_acc[model_name]
        total_processed += int(item["processed"])
        total_ok += int(item["ok"])
        total_invalid += int(item["invalid"])
        total_error += int(item["error"])
        eval_total = int(item["eval_total"])
        eval_correct = int(item["eval_correct"])
        total_eval += eval_total
        total_correct += eval_correct
        accuracy_value: float | None = None
        if eval_total > 0:
            accuracy_value = eval_correct / eval_total
        models_summary[model_name] = {
            "processed": int(item["processed"]),
            "ok": int(item["ok"]),
            "invalid": int(item["invalid"]),
            "error": int(item["error"]),
            "accuracy": {
                "evaluated": eval_total,
                "correct": eval_correct,
                "value": accuracy_value,
            },
            "metrics": {
                metric_key: _metric_agg_summary(item["metrics"][metric_key])
                for metric_key in metric_keys
            },
        }

    global_accuracy: float | None = None
    if total_eval > 0:
        global_accuracy = total_correct / total_eval

    return {
        "version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema_name": schema_name,
        "model_count": len(models_summary),
        "global": {
            "processed": total_processed,
            "ok": total_ok,
            "invalid": total_invalid,
            "error": total_error,
            "accuracy": {
                "evaluated": total_eval,
                "correct": total_correct,
                "value": global_accuracy,
            },
            "metrics": {
                metric_key: _metric_agg_summary(global_metric_acc[metric_key])
                for metric_key in metric_keys
            },
        },
        "models": models_summary,
    }


def upsert_batch_execution_summary(
    *,
    output_path: str | Path,
    schema_name: str,
    model_ids: list[str] | None = None,
) -> dict[str, Any] | None:
    """Escribe/actualiza una línea de resumen agregado del JSONL batch.

    Incluye métricas min/max/media por modelo y global para facilitar auditoría.
    """
    out_path = Path(output_path)
    if not out_path.exists() or not out_path.is_file():
        return None

    selected_models = {str(item).strip() for item in (model_ids or []) if str(item).strip()}
    metric_keys = ["duration_seconds", "ttft_seconds", "tokens_per_second", "total_duration_seconds"]

    lines = _read_jsonl_lines(out_path)
    kept_lines, model_acc, global_metric_acc = _aggregate_models_from_jsonl_lines(
        lines=lines,
        selected_models=selected_models,
        schema_name=schema_name,
        metric_keys=metric_keys,
    )

    if not model_acc:
        return None

    _repair_summary_meta_line(
        kept_lines=kept_lines,
        model_acc=model_acc,
        schema_name=schema_name,
    )
    summary_payload = _materialize_summary_payload(
        model_acc=model_acc,
        global_metric_acc=global_metric_acc,
        metric_keys=metric_keys,
        schema_name=schema_name,
    )

    kept_lines.append(json.dumps({_BATCH_SUMMARY_KEY: summary_payload}, ensure_ascii=False))
    _write_jsonl_lines(out_path, kept_lines)
    return summary_payload


def classify_error(error: BaseException) -> str:
    """Clasifica un error para la telemetría del lote."""
    message = str(error).lower()
    if "no cumple el esquema" in message or "json válido" in message:
        return "invalid"
    return "error"


def _prune_missing_fields(value: Any) -> Any:
    """Elimina claves opcionales ausentes en salidas JSON-like sin tocar 0 o False."""
    if isinstance(value, dict):
        pruned: dict[str, Any] = {}
        for key, item in value.items():
            cleaned = _prune_missing_fields(item)
            if cleaned is None:
                continue
            if isinstance(cleaned, dict) and not cleaned:
                continue
            if isinstance(cleaned, list) and not cleaned:
                continue
            pruned[key] = cleaned
        return pruned
    if isinstance(value, list):
        items = [_prune_missing_fields(item) for item in value]
        return [item for item in items if item is not None]
    return value


def build_record(
    *,
    model_id: str,
    schema_name: str,
    image_path: Path,
    duration_seconds: float,
    status: str,
    payload: dict[str, Any] | None = None,
    telemetry: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    """Construye un registro homogéneo para JSONL."""
    record: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "schema_name": schema_name,
        "image_path": str(image_path),
        "image_name": image_path.name,
        "status": status,
        "duration_seconds": round(duration_seconds, 6),
        "error_type": type(error).__name__ if error is not None else None,
        "error_message": str(error) if error is not None else None,
    }
    if telemetry:
        record.update(
            {
                record_key: telemetry.get(telemetry_key)
                for record_key, telemetry_key in _TELEMETRY_RECORD_FIELDS.items()
            }
        )
    if payload:
        record["payload"] = payload

    if metadata:
        for key, value in metadata.items():
            if key in _MANIFEST_BASE_KEYS:
                continue
            record[key] = value

    return _prune_missing_fields(record)


def _empty_metric_summary() -> dict[str, float | None]:
    """Construye una estructura homogénea para métricas opcionales."""
    return {"avg": None}


def _compute_metric_summary(total: float, count: int) -> dict[str, float | None]:
    """Calcula una media incremental o marca la métrica como no disponible."""
    if count <= 0:
        return _empty_metric_summary()
    return {"avg": total / count}


def _build_batch_summary(
    *,
    model_id: str,
    schema_name: str,
    prompt: str,
    output_path: Path,
    processed: int,
    ok: int,
    invalid: int,
    fail: int,
    total_available: int,
    sample_size: int,
    ttft_total: float,
    ttft_count: int,
    tps_total: float,
    tps_count: int,
    total_duration_total: float,
    total_duration_count: int,
    input_source: str,
    discarded_manifest_rows: int = 0,
) -> dict[str, Any]:
    """Construye el resumen parcial o final del batch runner."""
    return {
        "model_id": model_id,
        "schema_name": schema_name,
        "prompt": prompt,
        "output_path": str(output_path),
        "processed": processed,
        "ok": ok,
        "invalid": invalid,
        "fail": fail,
        "total_available": total_available,
        "sample_size": sample_size,
        "input_source": input_source,
        "discarded_manifest_rows": discarded_manifest_rows,
        "ttft": _compute_metric_summary(ttft_total, ttft_count),
        "tps": _compute_metric_summary(tps_total, tps_count),
        "total_duration": _compute_metric_summary(total_duration_total, total_duration_count),
    }


def _resolve_batch_input_items(
    *,
    image_dir: str | Path | None,
    manifest: str | Path | None,
    pending_image_paths: list[str] | None,
    pending_entries: list[dict[str, Any]] | None,
) -> tuple[str, list[BatchInputItem], int, Path | None]:
    """Resuelve origen de entrada y aplica filtros de pendientes cuando corresponde."""
    discarded_manifest_rows = 0
    manifest_path_obj: Path | None = None

    if manifest is not None:
        input_source = "manifest"
        manifest_path_obj = Path(manifest)
        discovered_items, discarded_manifest_rows = iter_manifest_items(manifest_path_obj)
        if pending_entries:
            discovered_items = _filter_items_by_pending_entries(
                discovered_items=discovered_items,
                pending_entries=pending_entries,
                manifest_path=manifest_path_obj,
            )
        elif pending_image_paths:
            discovered_items = _filter_items_by_pending_image_paths(
                discovered_items=discovered_items,
                pending_image_paths=pending_image_paths,
            )
        return input_source, discovered_items, discarded_manifest_rows, manifest_path_obj

    if image_dir is None:
        raise ValueError("image_dir no puede ser None cuando no se usa manifest")
    input_source = "image_dir"
    discovered_items = [BatchInputItem(image_path=path) for path in iter_image_paths(Path(image_dir))]
    return input_source, discovered_items, discarded_manifest_rows, manifest_path_obj


def run_batch_job(
    *,
    model_id: str,
    image_dir: str | Path | None = None,
    manifest: str | Path | None = None,
    schema_name: str,
    include_reasoning: bool = False,
    output_path: str | Path | None = None,
    max_images: int | None = None,
    shuffle: bool = False,
    seed: int | None = None,
    temperature: float = 0.0,
    prompt: str | None = None,
    server_api_host: str | None = None,
    api_token: str | None = None,
    verbose: bool = False,
    pending_image_paths: list[str] | None = None,
    pending_entries: list[dict[str, Any]] | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Ejecuta el lote completo y devuelve un resumen final."""
    if (image_dir is None) == (manifest is None):
        raise ValueError("Debes indicar exactamente uno de estos argumentos: image_dir o manifest")

    public_schema_name, schema_cls = resolve_schema(schema_name, include_reasoning)
    selected_prompt = prompt.strip() if prompt and prompt.strip() else build_prompt_for_schema(public_schema_name, schema_cls)
    input_source, discovered_items, discarded_manifest_rows, manifest_path_obj = _resolve_batch_input_items(
        image_dir=image_dir,
        manifest=manifest,
        pending_image_paths=pending_image_paths,
        pending_entries=pending_entries,
    )

    images = select_images(discovered_items, max_images=max_images, shuffle=shuffle, seed=seed)

    resolved_output_path = Path(output_path) if output_path is not None else build_default_output_path(public_schema_name)
    if resolved_output_path.suffix.lower() != ".jsonl":
        resolved_output_path = resolved_output_path.with_suffix(".jsonl")

    _ensure_batch_meta_header(
        output_path=resolved_output_path,
        model_id=model_id,
        schema_name=public_schema_name,
        input_source=input_source,
        manifest_path=manifest_path_obj,
    )

    loader = VLMLoader(
        model_path=model_id,
        verbose=verbose,
        server_api_host=server_api_host,
        api_token=api_token,
    )

    ok = 0
    invalid = 0
    fail = 0
    processed = 0
    metric_totals = {
        "ttft_total": 0.0,
        "tps_total": 0.0,
        "total_duration_total": 0.0,
    }
    metric_counts = {
        "ttft_count": 0,
        "tps_count": 0,
        "total_duration_count": 0,
    }

    def build_summary() -> dict[str, Any]:
        return _build_batch_summary(
            model_id=model_id,
            schema_name=public_schema_name,
            prompt=selected_prompt,
            output_path=resolved_output_path,
            processed=processed,
            ok=ok,
            invalid=invalid,
            fail=fail,
            total_available=len(discovered_items),
            sample_size=len(images),
            ttft_total=metric_totals["ttft_total"],
            ttft_count=metric_counts["ttft_count"],
            tps_total=metric_totals["tps_total"],
            tps_count=metric_counts["tps_count"],
            total_duration_total=metric_totals["total_duration_total"],
            total_duration_count=metric_counts["total_duration_count"],
            input_source=input_source,
            discarded_manifest_rows=discarded_manifest_rows,
        )

    def emit_progress(
        event: str,
        *,
        index: int = 0,
        image_path: str | None = None,
        status: str | None = None,
        record: dict[str, Any] | None = None,
        run_iteration_index: int | None = None,
        run_iteration_total: int | None = None,
    ) -> None:
        if on_progress is None:
            return
        on_progress(
            {
                "event": event,
                "index": index,
                "total": len(images),
                "image_path": image_path,
                "status": status,
                "record": dict(record) if record is not None else None,
                "run_iteration_index": run_iteration_index,
                "run_iteration_total": run_iteration_total,
                "summary": build_summary(),
            }
        )

    emit_progress("start")

    try:
        loader.load_model()
        progress_iter = tqdm(images, desc="Procesando imágenes", unit="img") if on_progress is None else images
        for index, item in enumerate(progress_iter, start=1):
            image_path = item.image_path
            metadata_dict = item.metadata if isinstance(item.metadata, dict) else {}
            emit_progress(
                "image_start",
                index=index,
                image_path=str(image_path),
                status="running",
                run_iteration_index=_safe_positive_int(metadata_dict.get("run_iteration_index")),
                run_iteration_total=_safe_positive_int(metadata_dict.get("run_iteration_total")),
            )
            started_at = time.perf_counter()
            try:
                result = loader.inference(
                    image_path=image_path,
                    prompt=selected_prompt,
                    schema=schema_cls,
                    temperature=temperature,
                    include_telemetry=True,
                )
                duration_seconds = time.perf_counter() - started_at
                payload = result.data.model_dump()
                telemetry_payload = (
                    dataclasses.asdict(result.telemetry)
                    if dataclasses.is_dataclass(result.telemetry)
                    else dict(getattr(result.telemetry, "__dict__", {}))
                )
                record = build_record(
                    model_id=model_id,
                    schema_name=public_schema_name,
                    image_path=image_path,
                    duration_seconds=duration_seconds,
                    status="ok",
                    payload=payload,
                    telemetry=telemetry_payload,
                    metadata=item.metadata,
                )
                ok += 1
                _update_metric_totals(
                    totals=metric_totals,
                    counts=metric_counts,
                    telemetry_payload=telemetry_payload,
                )
            except Exception as error:
                duration_seconds = time.perf_counter() - started_at
                status = classify_error(error)
                record = build_record(
                    model_id=model_id,
                    schema_name=public_schema_name,
                    image_path=image_path,
                    duration_seconds=duration_seconds,
                    status=status,
                    metadata=item.metadata,
                    error=error,
                )
                if status == "invalid":
                    invalid += 1
                else:
                    fail += 1

            append_jsonl_record(resolved_output_path, record)
            processed += 1
            emit_progress(
                "image_done",
                index=index,
                image_path=str(image_path),
                status=str(record.get("status") or "error"),
                record=record,
                run_iteration_index=_safe_positive_int(record.get("run_iteration_index")),
                run_iteration_total=_safe_positive_int(record.get("run_iteration_total")),
            )
    finally:
        loader.unload_model()

    summary = build_summary()
    emit_progress("complete", index=processed, status="complete")
    return summary


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada CLI del batch runner."""
    parser = build_parser()
    args = parser.parse_args(argv)

    summary = run_batch_job(
        model_id=args.model,
        image_dir=args.image_dir,
        manifest=args.manifest,
        schema_name=args.schema,
        include_reasoning=args.with_reasoning,
        output_path=args.output,
        max_images=args.max_images,
        shuffle=args.shuffle,
        seed=args.seed,
        temperature=args.temperature,
        prompt=args.prompt,
        server_api_host=args.server_api_host,
        api_token=args.api_token,
        verbose=args.verbose,
    )

    print("\n=== BATCH RUNNER ===")
    print(f"Modelo        : {summary['model_id']}")
    print(f"Esquema       : {summary['schema_name']}")
    print(f"Procesadas    : {summary['processed']} de {summary['total_available']}")
    print(f"OK            : {summary['ok']}")
    print(f"Inválidas     : {summary['invalid']}")
    print(f"Errores       : {summary['fail']}")
    print(f"Origen entrada: {summary['input_source']}")
    if summary.get("input_source") == "manifest":
        print(f"Descartadas   : {summary['discarded_manifest_rows']}")
    print(f"Salida        : {summary['output_path']}")

    return 0 if summary["fail"] == 0 and summary["invalid"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())