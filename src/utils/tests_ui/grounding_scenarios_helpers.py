"""Helpers reutilizables para UI de escenarios de grounding.

Centraliza utilidades de heatmap, resolución de rutas y resumen de registros
para mantener grounding_scenarios.py más pequeño y legible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .metrics import calculate_iou, calculate_proximity_score, summarize_numeric
from .test_dashboards_ui import _render_table_lines

HEATMAP_CLASS_ORDER: tuple[str, ...] = ("AD", "HP", "ASS")


def normalize_heatmap_class(value: Any) -> str | None:
    """Normaliza etiquetas de clase para el mapa de calor en vivo."""
    label = str(value or "").strip().upper()
    if label in HEATMAP_CLASS_ORDER:
        return label
    return None


def empty_live_confusion_counts() -> dict[str, dict[str, int]]:
    """Inicializa matriz de confusión live (GT->Pred) para AD/HP/ASS."""
    return {
        gt_label: {pred_label: 0 for pred_label in HEATMAP_CLASS_ORDER}
        for gt_label in HEATMAP_CLASS_ORDER
    }


def _lerp_color_rgb(start: tuple[int, int, int], end: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    """Interpolación lineal RGB para gradiente de calor."""
    clamped = max(0.0, min(1.0, float(t)))
    return (
        int(round(start[0] + (end[0] - start[0]) * clamped)),
        int(round(start[1] + (end[1] - start[1]) * clamped)),
        int(round(start[2] + (end[2] - start[2]) * clamped)),
    )


def _heatmap_rgb_from_percentage(percentage: float) -> tuple[int, int, int]:
    """Gradiente continuo blanco->amarillo->naranja->escarlata/rojo oscuro."""
    pct = max(0.0, min(100.0, float(percentage)))
    if pct <= 33.0:
        return _lerp_color_rgb((255, 255, 255), (255, 232, 140), pct / 33.0)
    if pct <= 66.0:
        return _lerp_color_rgb((255, 232, 140), (255, 165, 0), (pct - 33.0) / 33.0)
    return _lerp_color_rgb((255, 165, 0), (140, 0, 0), (pct - 66.0) / 34.0)


def _heatmap_cell_style(*, percentage: float) -> str:
    """Devuelve escape ANSI fg/bg con contraste automático para una celda."""
    red, green, blue = _heatmap_rgb_from_percentage(percentage)
    luminance = (0.2126 * red) + (0.7152 * green) + (0.0722 * blue)
    fg = "30" if luminance >= 145.0 else "97"
    return f"\033[{fg};48;2;{red};{green};{blue}m"


def heatmap_rgb_from_percentage(percentage: float) -> tuple[int, int, int]:
    """API pública para tests y módulos externos."""
    return _heatmap_rgb_from_percentage(percentage)


def heatmap_cell_style(*, percentage: float) -> str:
    """API pública para tests y módulos externos."""
    return _heatmap_cell_style(percentage=percentage)


def build_live_confusion_heatmap_lines(
    kit: Any,
    confusion: dict[str, dict[str, int]],
    *,
    ui_width: int | None = None,
) -> list[str]:
    """Construye mapa de calor live en formato tabla del entorno TUI."""
    total_all = 0
    for gt_label in HEATMAP_CLASS_ORDER:
        row = confusion.get(gt_label) or {}
        for pred_label in HEATMAP_CLASS_ORDER:
            total_all += int(row.get(pred_label) or 0)

    rows: list[list[str]] = []
    colored_rows: list[Any] = []
    pred_totals = {pred_label: 0 for pred_label in HEATMAP_CLASS_ORDER}
    for gt_label in HEATMAP_CLASS_ORDER:
        row = confusion.get(gt_label) or {}
        row_total = int(sum(int(row.get(pred_label) or 0) for pred_label in HEATMAP_CLASS_ORDER))
        cell_values: list[str] = []
        cell_colors: list[str | None] = []
        for pred_label in HEATMAP_CLASS_ORDER:
            value = int(row.get(pred_label) or 0)
            pred_totals[pred_label] = int(pred_totals.get(pred_label) or 0) + value
            pct_global = (float(value) / float(total_all) * 100.0) if total_all > 0 else 0.0
            pct_row = (float(value) / float(row_total) * 100.0) if row_total > 0 else 0.0
            cell_values.append(f"{value:>3} {pct_global:>4.1f}%/{pct_row:>4.1f}%")
            cell_colors.append(_heatmap_cell_style(percentage=pct_global))
        row_total_pct = (float(row_total) / float(total_all) * 100.0) if total_all > 0 else 0.0
        row_total_text = f"{row_total:>3} {row_total_pct:>4.1f}%"
        rows.append([f"GT {gt_label}", cell_values[0], cell_values[1], cell_values[2], row_total_text])
        colored_rows.append(
            {
                "cells": [f"GT {gt_label}", cell_values[0], cell_values[1], cell_values[2], row_total_text],
                "colors": [
                    "DIM",
                    cell_colors[0],
                    cell_colors[1],
                    cell_colors[2],
                    _heatmap_cell_style(percentage=row_total_pct),
                ],
            }
        )

    total_all = int(sum(pred_totals.values()))
    total_safe = max(1, total_all)
    total_row = [
        "Total",
        f"{int(pred_totals.get('AD') or 0):>3} {(float(pred_totals.get('AD') or 0) / float(total_safe) * 100.0):>4.1f}%",
        f"{int(pred_totals.get('HP') or 0):>3} {(float(pred_totals.get('HP') or 0) / float(total_safe) * 100.0):>4.1f}%",
        f"{int(pred_totals.get('ASS') or 0):>3} {(float(pred_totals.get('ASS') or 0) / float(total_safe) * 100.0):>4.1f}%",
        f"{total_all:>3} 100.0%",
    ]
    rows.append(total_row)
    total_colors = [
        _heatmap_cell_style(percentage=100.0),
        _heatmap_cell_style(percentage=(float(pred_totals.get("AD") or 0) / float(total_safe) * 100.0)),
        _heatmap_cell_style(percentage=(float(pred_totals.get("HP") or 0) / float(total_safe) * 100.0)),
        _heatmap_cell_style(percentage=(float(pred_totals.get("ASS") or 0) / float(total_safe) * 100.0)),
        _heatmap_cell_style(percentage=100.0),
    ]
    colored_rows.append({"cells": total_row, "colors": total_colors})

    table_menu_fn = getattr(kit, "table_menu", None)
    table_column_cls = getattr(kit, "TableColumn", None)
    table_row_cls = getattr(kit, "TableRow", None)
    if callable(table_menu_fn) and table_column_cls is not None and table_row_cls is not None:
        width = kit.width() if ui_width is None else int(ui_width)
        columns = [
            table_column_cls(label="GT \\ Pred", width_ratio=0.2, min_width=12),
            table_column_cls(label="AD", width_ratio=0.2, min_width=12),
            table_column_cls(label="HP", width_ratio=0.2, min_width=12),
            table_column_cls(label="ASS", width_ratio=0.2, min_width=12),
            table_column_cls(label="Total", width_ratio=0.2, min_width=10),
        ]
        table_rows = [
            table_row_cls(cells=list(item["cells"]), cell_colors=list(item["colors"]))
            for item in colored_rows
        ]
        rendered = table_menu_fn(
            columns,
            table_rows,
            interactive=False,
            return_lines=True,
            width=width,
            max_cell_lines=False,
        )
        if isinstance(rendered, list):
            table_lines = [str(line) for line in rendered]
        else:
            table_lines = _render_table_lines(
                kit,
                ["GT \\ Pred", "AD", "HP", "ASS", "Total"],
                rows,
                ui_width=ui_width,
            )
    else:
        table_lines = _render_table_lines(
            kit,
            ["GT \\ Pred", "AD", "HP", "ASS", "Total"],
            rows,
            ui_width=ui_width,
        )

    return [
        "Heatmap GT→Pred (live):",
        *table_lines,
        "Leyenda: celda = conteo / %global / %filaGT · color = %global (blanco→amarillo→naranja→escarlata)",
    ]


def project_root() -> Path:
    """Obtiene la raíz del repositorio independientemente del cwd actual."""
    return Path(__file__).resolve().parents[3]


def resolve_existing_file(path_like: str | Path) -> Path | None:
    """Resuelve una ruta de archivo existente probando absoluta, cwd y raíz de proyecto."""
    raw = Path(path_like)
    candidates: list[Path] = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(Path.cwd() / raw)
        candidates.append(project_root() / raw)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        try:
            if candidate.exists() and candidate.is_file():
                return candidate
        except Exception:
            continue
    return None


def resolve_existing_dir(path_like: str | Path) -> Path | None:
    """Resuelve una ruta de directorio existente probando absoluta, cwd y raíz de proyecto."""
    raw = Path(path_like)
    candidates: list[Path] = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(Path.cwd() / raw)
        candidates.append(project_root() / raw)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        try:
            if candidate.exists() and candidate.is_dir():
                return candidate
        except Exception:
            continue
    return None


def coerce_bbox4(value: Any) -> list[int] | None:
    """Convierte bbox [ymin, xmin, ymax, xmax] a lista de enteros o None."""
    if not isinstance(value, list) or len(value) != 4:
        return None
    converted: list[int] = []
    for item in value:
        if item is None:
            return None
        try:
            converted.append(int(float(item)))
        except Exception:
            return None
    return converted


def extract_final_diagnosis_class(entry: dict[str, Any]) -> str:
    """Obtiene clase final normalizada desde un registro JSONL de escenarios grounding."""
    raw_payload = entry.get("payload")
    if isinstance(raw_payload, dict):
        result_payload: dict[str, Any] = raw_payload
    else:
        raw_result = entry.get("result")
        result_payload = raw_result if isinstance(raw_result, dict) else {}
    return str(result_payload.get("final_diagnosis_class") or "").strip().upper()


def extract_iou(entry: dict[str, Any]) -> float | None:
    """Calcula IoU para un registro JSONL cuando ambas cajas existen."""
    direct_iou = entry.get("iou_score")
    if isinstance(direct_iou, (int, float)):
        return float(direct_iou)

    gt_bbox = coerce_bbox4(entry.get("ground_truth_bbox"))
    raw_payload = entry.get("payload")
    if isinstance(raw_payload, dict):
        result_payload: dict[str, Any] = raw_payload
    else:
        raw_result = entry.get("result")
        result_payload = raw_result if isinstance(raw_result, dict) else {}
    pred_bbox = coerce_bbox4(
        [
            result_payload.get("ymin"),
            result_payload.get("xmin"),
            result_payload.get("ymax"),
            result_payload.get("xmax"),
        ]
    )

    if gt_bbox is None or pred_bbox is None:
        return None

    try:
        return float(calculate_iou(gt_bbox, pred_bbox))
    except Exception:
        return None


def extract_proximity(entry: dict[str, Any]) -> float | None:
    """Calcula Proximity para un registro JSONL cuando hay cajas validas."""
    direct_proximity = entry.get("proximity_score")
    if isinstance(direct_proximity, (int, float)):
        return float(direct_proximity)

    gt_bbox = coerce_bbox4(entry.get("ground_truth_bbox"))
    raw_payload = entry.get("payload")
    if isinstance(raw_payload, dict):
        result_payload: dict[str, Any] = raw_payload
    else:
        raw_result = entry.get("result")
        result_payload = raw_result if isinstance(raw_result, dict) else {}
    pred_bbox = coerce_bbox4(
        [
            result_payload.get("ymin"),
            result_payload.get("xmin"),
            result_payload.get("ymax"),
            result_payload.get("xmax"),
        ]
    )

    if gt_bbox is None or pred_bbox is None:
        return None

    try:
        payload = calculate_proximity_score(gt_bbox, pred_bbox)
        value = payload.get("proximity_score") if isinstance(payload, dict) else None
        return float(value) if isinstance(value, (int, float)) else None
    except Exception:
        return None


def scenario_recent_record(entry: dict[str, Any]) -> dict[str, object]:
    """Convierte una línea JSONL de grounding en registro de actividad reciente."""
    image_name = str(entry.get("image_id") or "imagen")
    raw_payload = entry.get("payload")
    if isinstance(raw_payload, dict):
        result_payload: dict[str, Any] = raw_payload
    else:
        raw_result = entry.get("result")
        result_payload = raw_result if isinstance(raw_result, dict) else {}
    payload: dict[str, object] = {
        "gt_cls": str(entry.get("ground_truth_cls") or "N/D"),
        "pred_cls": str(result_payload.get("final_diagnosis_class") or "N/D"),
    }
    gt_bbox = coerce_bbox4(entry.get("ground_truth_bbox"))
    pred_bbox = coerce_bbox4(
        [
            result_payload.get("ymin"),
            result_payload.get("xmin"),
            result_payload.get("ymax"),
            result_payload.get("xmax"),
        ]
    )
    if gt_bbox is not None and pred_bbox is not None:
        try:
            iou_value = calculate_iou(gt_bbox, pred_bbox)
            payload["iou"] = f"{iou_value:.3f}"
        except Exception:
            pass
    return {
        "image_name": image_name,
        "status": "ok",
        "payload": payload,
    }


def build_comparison_summary_rows(records: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    """Construye resumen de comparación GT vs predicción e IoU para pantalla final."""
    compared = 0
    matched = 0
    mismatched = 0
    iou_values: list[float] = []

    for entry in records:
        gt_cls = str(entry.get("ground_truth_cls") or "").strip().upper()
        pred_cls = extract_final_diagnosis_class(entry)
        if gt_cls and pred_cls:
            compared += 1
            if gt_cls == pred_cls:
                matched += 1
            else:
                mismatched += 1

        iou_value = extract_iou(entry)
        if isinstance(iou_value, float):
            iou_values.append(iou_value)

    accuracy = (matched / compared * 100.0) if compared > 0 else None
    iou_stats = summarize_numeric(iou_values)
    avg_iou = iou_stats.get("avg")
    min_iou = iou_stats.get("min")
    max_iou = iou_stats.get("max")

    return [
        ("Comparaciones clase", str(compared), "OK" if compared > 0 else "WARN"),
        ("Matches clase", str(matched), "OK"),
        ("Mismatch clase", str(mismatched), "OK" if mismatched == 0 else "WARN"),
        (
            "Accuracy clase",
            f"{accuracy:.1f}%" if accuracy is not None else "N/D",
            "OK" if accuracy is not None else "WARN",
        ),
        (
            "IoU medio",
            f"{avg_iou:.4f}" if isinstance(avg_iou, float) else "N/D",
            "OK" if isinstance(avg_iou, float) else "WARN",
        ),
        (
            "IoU min/max",
            (
                f"{min_iou:.4f} / {max_iou:.4f}"
                if isinstance(min_iou, float) and isinstance(max_iou, float)
                else "N/D"
            ),
            "OK" if isinstance(avg_iou, float) else "WARN",
        ),
    ]


def summarize_existing_scenario_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Resume métricas acumuladas para inicializar dashboard al reanudar."""
    ok = 0
    fail = 0
    skip = 0
    matched = 0
    mismatched = 0
    ttft_sum = 0.0
    ttft_count = 0
    tps_sum = 0.0
    tps_count = 0
    duration_sum = 0.0
    duration_count = 0
    iou_sum = 0.0
    iou_count = 0
    proximity_sum = 0.0
    proximity_count = 0
    gt_class_counts: dict[str, int] = {"AD": 0, "HP": 0, "ASS": 0, "OTHER": 0}
    pred_class_counts: dict[str, int] = {"AD": 0, "HP": 0, "ASS": 0, "OTHER": 0}
    confusion_counts = empty_live_confusion_counts()

    def _bucket_class(target: dict[str, int], value: Any) -> None:
        label = str(value or "").strip().upper()
        if label in {"AD", "HP", "ASS"}:
            target[label] = int(target.get(label) or 0) + 1
        else:
            target["OTHER"] = int(target.get("OTHER") or 0) + 1

    for entry in records:
        status = str(entry.get("status") or "ok").strip().lower()
        if status in {"error", "failed", "fail"}:
            continue
        if status in {"pending", "queued", "in_progress", "running"}:
            continue
        if status in {"invalid", "skip", "skipped"}:
            skip += 1
        elif status in {"ok", "success", ""}:
            ok += 1
        else:
            continue

        class_match = entry.get("class_match")
        if isinstance(class_match, bool):
            if class_match:
                matched += 1
            else:
                mismatched += 1
        else:
            gt_cls = str(entry.get("ground_truth_cls") or "").strip().upper()
            pred_cls = extract_final_diagnosis_class(entry)
            if gt_cls and pred_cls:
                if gt_cls == pred_cls:
                    matched += 1
                else:
                    mismatched += 1

        _bucket_class(gt_class_counts, entry.get("ground_truth_cls"))
        pred_cls_value = extract_final_diagnosis_class(entry)
        _bucket_class(pred_class_counts, pred_cls_value)

        gt_conf = normalize_heatmap_class(entry.get("ground_truth_cls"))
        pred_conf = normalize_heatmap_class(pred_cls_value)
        if gt_conf is not None and pred_conf is not None:
            confusion_counts[gt_conf][pred_conf] = int(confusion_counts[gt_conf].get(pred_conf) or 0) + 1

        ttft_value = entry.get("ttft_seconds")
        if isinstance(ttft_value, (int, float)):
            ttft_sum += float(ttft_value)
            ttft_count += 1

        tps_value = entry.get("tokens_per_second")
        if isinstance(tps_value, (int, float)):
            tps_sum += float(tps_value)
            tps_count += 1

        duration_value = entry.get("total_duration_seconds")
        if isinstance(duration_value, (int, float)):
            duration_sum += float(duration_value)
            duration_count += 1

        iou_value = extract_iou(entry)
        if isinstance(iou_value, float):
            iou_sum += iou_value
            iou_count += 1

        proximity_value = extract_proximity(entry)
        if isinstance(proximity_value, float):
            proximity_sum += proximity_value
            proximity_count += 1

    current = ok + skip
    avg_ttft = (ttft_sum / ttft_count) if ttft_count > 0 else None
    avg_tps = (tps_sum / tps_count) if tps_count > 0 else None
    avg_duration = (duration_sum / duration_count) if duration_count > 0 else None
    avg_iou = (iou_sum / iou_count) if iou_count > 0 else None
    avg_proximity = (proximity_sum / proximity_count) if proximity_count > 0 else None

    return {
        "current": current,
        "ok": ok,
        "fail": fail,
        "skip": skip,
        "matched": matched,
        "mismatched": mismatched,
        "avg_ttft": avg_ttft,
        "avg_tps": avg_tps,
        "avg_duration": avg_duration,
        "avg_iou": avg_iou,
        "avg_proximity": avg_proximity,
        "ttft_sum": ttft_sum,
        "ttft_count": ttft_count,
        "tps_sum": tps_sum,
        "tps_count": tps_count,
        "duration_sum": duration_sum,
        "duration_count": duration_count,
        "iou_sum": iou_sum,
        "iou_count": iou_count,
        "proximity_sum": proximity_sum,
        "proximity_count": proximity_count,
        "gt_class_counts": gt_class_counts,
        "pred_class_counts": pred_class_counts,
        "confusion_counts": confusion_counts,
    }
