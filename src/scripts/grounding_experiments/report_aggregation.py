"""Aggregation helpers for scenario JSONL outputs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.utils.tests_ui.metrics import mean_or_none


LoadJsonlRecordsFn = Callable[[Path], list[dict[str, Any]]]
NormalizePolypClassFn = Callable[[Any], str]
NormalizeImageStemFn = Callable[[Any], str]
ComputeIouSafeFn = Callable[..., float | None]
ComputeProximitySafeFn = Callable[..., dict[str, float] | None]


def _extract_predicted_class(payload: dict[str, Any], fallback: Any = "") -> str:
    """Extract predicted class from canonical payload field without class normalization."""
    value = payload.get("final_diagnosis_class")
    if isinstance(value, str):
        return value.strip()
    if isinstance(fallback, str):
        return fallback.strip()
    return ""


def save_result(output_file: str, result_dict: dict[str, Any]) -> None:
    """Append one result entry to a JSONL file safely."""
    output_path = Path(output_file)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(result_dict, ensure_ascii=False)
        with output_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(payload + "\n")
            handle.flush()
    except Exception as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(f"Failed to append JSONL record to {output_path}: {exc}") from exc


def _read_jsonl_lines(path: Path) -> list[str]:
    """Read JSONL lines without trailing newline characters."""
    if not path.exists() or not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def _write_jsonl_lines(path: Path, lines: list[str]) -> None:
    """Rewrite complete JSONL file atomically for system header/summary upserts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        if lines:
            handle.write("\n".join(lines) + "\n")


def load_jsonl_records(
    path: Path,
    *,
    include_system_records: bool = False,
    scenario_meta_key: str,
    scenario_summary_key: str,
) -> list[dict[str, Any]]:
    """Load JSONL dictionary records defensively, skipping malformed/system lines by default."""
    records: list[dict[str, Any]] = []
    if not path.exists() or not path.is_file():
        return records

    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload_line = line.strip()
                if not payload_line:
                    continue
                try:
                    payload = json.loads(payload_line)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    if not include_system_records:
                        if isinstance(payload.get(scenario_meta_key), dict):
                            continue
                        if isinstance(payload.get(scenario_summary_key), dict):
                            continue
                    records.append(payload)
    except Exception:
        return records

    return records


def collect_processed_image_ids_from_jsonl(
    *,
    output_jsonl_path: Path,
    normalize_image_stem: NormalizeImageStemFn,
    load_jsonl_records: LoadJsonlRecordsFn,
) -> set[str]:
    """Collect normalized image ids already present in an output JSONL file."""
    processed_ids: set[str] = set()
    for record in load_jsonl_records(output_jsonl_path):
        status = str(record.get("status") or "").strip().lower()
        if status in {
            "pending",
            "queued",
            "in_progress",
            "running",
            "error",
            "failed",
            "fail",
        }:
            continue
        try:
            normalized = normalize_image_stem(record.get("image_id"))
        except Exception:
            continue
        processed_ids.add(normalized)
    return processed_ids


def initialize_scenario_result_skeleton(
    *,
    output_path: Path,
    skeleton_records: list[dict[str, Any]],
    scenario_meta_key: str,
    scenario_summary_key: str,
) -> None:
    """Initialize JSONL with one pending line per sample while preserving metadata header."""
    lines = _read_jsonl_lines(output_path)
    system_lines: list[str] = []

    for line in lines:
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if isinstance(payload.get(scenario_meta_key), dict):
            system_lines.append(line)
            continue
        if isinstance(payload.get(scenario_summary_key), dict):
            continue

    skeleton_lines = [json.dumps(record, ensure_ascii=False) for record in skeleton_records]
    _write_jsonl_lines(output_path, [*system_lines, *skeleton_lines])


def upsert_scenario_result_record(
    *,
    output_path: Path,
    result_dict: dict[str, Any],
    normalize_image_stem: NormalizeImageStemFn,
    scenario_meta_key: str,
    scenario_summary_key: str,
) -> None:
    """Replace pending line for image_id with latest result; append if no matching slot exists."""
    target_image_id = result_dict.get("image_id")
    try:
        target_normalized = normalize_image_stem(target_image_id)
    except Exception:
        save_result(str(output_path), result_dict)
        return

    lines = _read_jsonl_lines(output_path)
    replacement_line = json.dumps(result_dict, ensure_ascii=False)
    replaced = False
    first_matching_index: int | None = None
    pending_matching_index: int | None = None
    for index, line in enumerate(lines):
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if isinstance(payload.get(scenario_meta_key), dict):
            continue
        if isinstance(payload.get(scenario_summary_key), dict):
            continue
        try:
            current_normalized = normalize_image_stem(payload.get("image_id"))
        except Exception:
            continue
        if current_normalized != target_normalized:
            continue

        if first_matching_index is None:
            first_matching_index = index

        current_status = str(payload.get("status") or "").strip().lower()
        if current_status in {"pending", "queued", "in_progress", "running"}:
            pending_matching_index = index
            break

    if pending_matching_index is not None:
        lines[pending_matching_index] = replacement_line
        replaced = True
    elif first_matching_index is not None:
        lines[first_matching_index] = replacement_line
        replaced = True

    if not replaced:
        lines.append(replacement_line)
    _write_jsonl_lines(output_path, lines)


def has_unfilled_scenario_records(
    *,
    output_path: Path,
    scenario_meta_key: str,
    scenario_summary_key: str,
) -> bool:
    """Return True when JSONL still contains entries that require refill/resume."""
    for line in _read_jsonl_lines(output_path):
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if isinstance(payload.get(scenario_meta_key), dict):
            continue
        if isinstance(payload.get(scenario_summary_key), dict):
            continue

        status = str(payload.get("status") or "").strip().lower()
        if status in {
            "pending",
            "queued",
            "in_progress",
            "running",
            "error",
            "failed",
            "fail",
        }:
            return True
    return False


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
    scenario_meta_key: str,
) -> None:
    """Upsert first JSONL line with scenario metadata, mirroring batch_runner style."""
    lines = _read_jsonl_lines(output_path)
    meta_index: int | None = None
    existing_meta: dict[str, Any] = {}
    for index, line in enumerate(lines):
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        current_meta = payload.get(scenario_meta_key)
        if isinstance(current_meta, dict):
            meta_index = index
            existing_meta = dict(current_meta)
            break

    now_utc = datetime.now(timezone.utc).isoformat()
    meta_payload: dict[str, Any] = {
        "version": 1,
        "created_at_utc": str(existing_meta.get("created_at_utc") or now_utc),
        "updated_at_utc": now_utc,
        "scenario_name": str(scenario_name),
        "model_id": str(model_id),
        "input_csv": str(input_csv),
        "img_dir": str(img_dir),
        "seed": int(seed) if seed is not None else None,
        "requested_limit": int(requested_limit) if requested_limit is not None else None,
        "resume_mode": bool(resume_mode),
    }

    header_line = json.dumps({scenario_meta_key: meta_payload}, ensure_ascii=False)
    if meta_index is None:
        lines.insert(0, header_line)
    else:
        lines[meta_index] = header_line
    _write_jsonl_lines(output_path, lines)


def upsert_scenario_execution_summary(
    *,
    output_path: Path,
    summary_payload: dict[str, Any],
    scenario_summary_key: str,
) -> None:
    """Upsert trailing scenario summary line in JSONL, mirroring batch_runner style."""
    lines = _read_jsonl_lines(output_path)
    kept_lines: list[str] = []
    for line in lines:
        try:
            payload = json.loads(line)
        except Exception:
            kept_lines.append(line)
            continue
        if isinstance(payload, dict) and isinstance(payload.get(scenario_summary_key), dict):
            continue
        kept_lines.append(line)

    kept_lines.append(json.dumps({scenario_summary_key: summary_payload}, ensure_ascii=False))
    _write_jsonl_lines(output_path, kept_lines)


def summarize_scenario_records_from_jsonl(
    *,
    output_path: Path,
    load_jsonl_records: LoadJsonlRecordsFn,
    normalize_polyp_class: NormalizePolypClassFn,
    compute_iou_safe: ComputeIouSafeFn,
    compute_proximity_safe: ComputeProximitySafeFn,
) -> dict[str, Any]:
    """Build cumulative execution summary from all non-system records in a scenario JSONL."""
    total = 0
    ok = 0
    fail = 0
    skip = 0
    matched_class = 0
    mismatched_class = 0
    iou_values: list[float] = []
    proximity_values: list[float] = []
    ttft_values: list[float] = []
    tps_values: list[float] = []
    total_duration_values: list[float] = []

    for entry in load_jsonl_records(output_path):
        if not isinstance(entry, dict):
            continue
        status = str(entry.get("status") or "ok").strip().lower()
        if status in {"pending", "queued", "in_progress", "running"}:
            continue
        total += 1

        if status in {"ok", "success"}:
            ok += 1
        elif status in {"invalid", "skip", "skipped"}:
            skip += 1
        else:
            fail += 1

        raw_payload = entry.get("payload")
        if isinstance(raw_payload, dict):
            payload = raw_payload
        else:
            raw_result = entry.get("result")
            payload = raw_result if isinstance(raw_result, dict) else {}

        predicted_cls = _extract_predicted_class(payload, entry.get("predicted_cls") or "")
        gt_cls = normalize_polyp_class(entry.get("ground_truth_cls") or "")
        class_match = entry.get("class_match")
        if isinstance(class_match, bool):
            if class_match:
                matched_class += 1
            else:
                mismatched_class += 1
        elif predicted_cls and gt_cls:
            if predicted_cls == gt_cls:
                matched_class += 1
            else:
                mismatched_class += 1

        iou_value = entry.get("iou_score")
        if not isinstance(iou_value, (int, float)):
            pred_bbox = [
                payload.get("ymin"),
                payload.get("xmin"),
                payload.get("ymax"),
                payload.get("xmax"),
            ]
            gt_bbox = entry.get("ground_truth_bbox")
            if isinstance(gt_bbox, list):
                iou_value = compute_iou_safe(gt_bbox=gt_bbox, pred_bbox=pred_bbox)
            else:
                iou_value = None
        if isinstance(iou_value, (int, float)):
            iou_values.append(float(iou_value))

        proximity_value = entry.get("proximity_score")
        if not isinstance(proximity_value, (int, float)):
            pred_bbox = [
                payload.get("ymin"),
                payload.get("xmin"),
                payload.get("ymax"),
                payload.get("xmax"),
            ]
            gt_bbox = entry.get("ground_truth_bbox")
            if isinstance(gt_bbox, list):
                proximity_payload = compute_proximity_safe(gt_bbox=gt_bbox, pred_bbox=pred_bbox)
                if isinstance(proximity_payload, dict):
                    proximity_value = proximity_payload.get("proximity_score")
                else:
                    proximity_value = None
            else:
                proximity_value = None
        if isinstance(proximity_value, (int, float)):
            proximity_values.append(float(proximity_value))

        ttft_value = entry.get("ttft_seconds")
        if isinstance(ttft_value, (int, float)):
            ttft_values.append(float(ttft_value))
        tps_value = entry.get("tokens_per_second")
        if isinstance(tps_value, (int, float)):
            tps_values.append(float(tps_value))
        total_duration_value = entry.get("total_duration_seconds")
        if isinstance(total_duration_value, (int, float)):
            total_duration_values.append(float(total_duration_value))

    return {
        "total": total,
        "ok": ok,
        "fail": fail,
        "skip": skip,
        "matched_class": matched_class,
        "mismatched_class": mismatched_class,
        "avg_iou": mean_or_none(iou_values),
        "avg_proximity": mean_or_none(proximity_values),
        "avg_ttft_seconds": mean_or_none(ttft_values),
        "avg_tokens_per_second": mean_or_none(tps_values),
        "avg_total_duration_seconds": mean_or_none(total_duration_values),
    }
