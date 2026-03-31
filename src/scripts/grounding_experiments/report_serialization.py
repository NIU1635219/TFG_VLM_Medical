"""Serialization helpers for scenario JSONL records."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
    telemetry_fields_map: dict[str, str],
    error: BaseException | None = None,
) -> dict[str, Any]:
    """Build a compact scenario JSONL record, aligned with batch_runner conventions."""
    record: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scenario_name": str(scenario_name),
        "model_id": str(model_id),
        "schema_name": str(schema_name),
        "image_id": image_id,
        "image_path": str(image_path),
        "image_name": image_path.name,
        "status": str(status),
        "duration_seconds": round(float(duration_seconds), 6),
        "ground_truth_bbox": ground_truth_bbox,
        "ground_truth_cls": ground_truth_cls,
        "class_match": class_match,
        "iou_score": iou_score,
        "error_type": type(error).__name__ if error is not None else None,
        "error_message": str(error) if error is not None else None,
    }
    if payload:
        record["payload"] = payload
    if telemetry_payload:
        record.update(
            select_telemetry_fields_for_record(
                telemetry_payload,
                telemetry_fields_map=telemetry_fields_map,
            )
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
