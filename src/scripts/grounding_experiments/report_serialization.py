"""Serialization helpers for scenario JSONL records."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any


_BASE64_CHUNK_RE = re.compile(r"(?:[A-Za-z0-9+/]{120,}={0,2})")


def _looks_like_inline_base64(value: str) -> bool:
    """Heurística conservadora para detectar payloads base64 inline."""
    text = str(value or "").strip()
    if not text:
        return False
    if text.startswith("data:image/") and "base64," in text:
        return True
    compact = "".join(text.split())
    if len(compact) < 120:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", compact))


def _sanitize_text_for_jsonl(value: Any) -> str:
    """Elimina blobs base64 extensos para que no se persistan en JSONL."""
    text = str(value or "")
    if not text:
        return text
    if _looks_like_inline_base64(text):
        return "[inline_base64_omitted]"

    sanitized = _BASE64_CHUNK_RE.sub("[base64_omitted]", text)
    if "data:image/" in sanitized and "base64," in sanitized:
        return "[inline_base64_omitted]"
    return sanitized


def _sanitize_record_value(value: Any) -> Any:
    """Recorre estructuras anidadas para sanear texto antes de persistir."""
    if isinstance(value, str):
        return _sanitize_text_for_jsonl(value)
    if isinstance(value, dict):
        return {str(k): _sanitize_record_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_record_value(item) for item in value]
    return value


def select_telemetry_fields_for_record(
    telemetry_payload: dict[str, Any] | None,
    *,
    telemetry_fields_map: dict[str, str],
) -> dict[str, Any]:
    """Select only compact telemetry fields used by batch-style JSONL records."""
    source = telemetry_payload if isinstance(telemetry_payload, dict) else {}
    return {
        record_key: source.get(telemetry_key)
        for record_key, telemetry_key in telemetry_fields_map.items()
    }


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
    telemetry_fields_map: dict[str, str],
    error: BaseException | None = None,
) -> dict[str, Any]:
    """Build a compact scenario JSONL record, aligned with batch_runner conventions."""
    image_path_text = _sanitize_text_for_jsonl(str(image_path))
    image_name = Path(image_path_text).name if image_path_text and image_path_text != "[inline_base64_omitted]" else "inline_image"

    record: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scenario_name": str(scenario_name),
        "model_id": str(model_id),
        "schema_name": str(schema_name),
        "image_id": image_id,
        "image_path": image_path_text,
        "image_name": image_name,
        "status": str(status),
        "duration_seconds": round(float(duration_seconds), 6),
        "ground_truth_bbox": ground_truth_bbox,
        "ground_truth_cls": ground_truth_cls,
        "class_match": class_match,
        "iou_score": iou_score,
        "proximity_score": proximity_score,
        "proximity_center_score": proximity_center_score,
        "proximity_size_score": proximity_size_score,
        "error_type": type(error).__name__ if error is not None else None,
        "error_message": _sanitize_text_for_jsonl(str(error)) if error is not None else None,
    }
    if payload:
        record["payload"] = _sanitize_record_value(payload)
    if telemetry_payload:
        record.update(
            _sanitize_record_value(select_telemetry_fields_for_record(
                telemetry_payload,
                telemetry_fields_map=telemetry_fields_map,
            ))
        )

    # Remove null/empty values to keep JSONL compact and stable.
    pruned: dict[str, Any] = {}
    for key, value in record.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        pruned[key] = value
    return pruned
