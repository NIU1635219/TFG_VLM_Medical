"""Selector de escenarios de grounding (stub) separado para mantener el menú principal limpio.

Este módulo contiene la UI de selección de escenarios A/B/C/D y está pensado
para recibir un `make_header_fn` desde el menú principal para conservar
consistencia en el renderizado de cabeceras.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

from ..setup_ui_io import ask_text
from .metrics import summarize_numeric
from .shared import ReactiveTerminalRenderer
from .test_dashboards_ui import (
    _build_recent_record_lines,
    _render_table_lines,
    _build_summary_lines,
    _make_live_panel,
    _make_recent_records,
    _render_final_sections_screen,
    _render_live_dashboard,
    _standard_final_intro,
)

if TYPE_CHECKING:
    from ..menu_kit import AppContext, UIKit


_GROUNDING_SEED_CURSOR_KEY = "grounding_scenarios_seed"
_GROUNDING_LAST_OUTPUTS_CURSOR_KEY = "grounding_scenarios_last_outputs"
_HEATMAP_CLASS_ORDER: tuple[str, ...] = ("AD", "HP", "ASS")


def _normalize_heatmap_class(value: Any) -> str | None:
    """Normaliza etiquetas de clase para el mapa de calor en vivo."""
    label = str(value or "").strip().upper()
    if label in _HEATMAP_CLASS_ORDER:
        return label
    return None


def _empty_live_confusion_counts() -> dict[str, dict[str, int]]:
    """Inicializa matriz de confusión live (GT->Pred) para AD/HP/ASS."""
    return {
        gt_label: {pred_label: 0 for pred_label in _HEATMAP_CLASS_ORDER}
        for gt_label in _HEATMAP_CLASS_ORDER
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
        # Blanco -> Amarillo
        return _lerp_color_rgb((255, 255, 255), (255, 232, 140), pct / 33.0)
    if pct <= 66.0:
        # Amarillo -> Naranja
        return _lerp_color_rgb((255, 232, 140), (255, 165, 0), (pct - 33.0) / 33.0)
    # Naranja -> Escarlata/rojo oscuro
    return _lerp_color_rgb((255, 165, 0), (140, 0, 0), (pct - 66.0) / 34.0)


def _heatmap_cell_style(*, percentage: float) -> str:
    """Devuelve escape ANSI fg/bg con contraste automático para una celda."""
    red, green, blue = _heatmap_rgb_from_percentage(percentage)
    # Luminancia perceptual para elegir texto oscuro/claro.
    luminance = (0.2126 * red) + (0.7152 * green) + (0.0722 * blue)
    fg = "30" if luminance >= 145.0 else "97"
    return f"\033[{fg};48;2;{red};{green};{blue}m"


def _build_live_confusion_heatmap_lines(
    kit: "UIKit",
    confusion: dict[str, dict[str, int]],
    *,
    ui_width: int | None = None,
) -> list[str]:
    """Construye mapa de calor live en formato tabla del entorno TUI."""
    total_all = 0
    max_cell = 0
    for gt_label in _HEATMAP_CLASS_ORDER:
        row = confusion.get(gt_label) or {}
        for pred_label in _HEATMAP_CLASS_ORDER:
            value = int(row.get(pred_label) or 0)
            total_all += value
            if value > max_cell:
                max_cell = value

    rows: list[list[str]] = []
    colored_rows: list[Any] = []
    pred_totals = {pred_label: 0 for pred_label in _HEATMAP_CLASS_ORDER}
    for gt_label in _HEATMAP_CLASS_ORDER:
        row = confusion.get(gt_label) or {}
        row_total = int(sum(int(row.get(pred_label) or 0) for pred_label in _HEATMAP_CLASS_ORDER))
        cell_values: list[str] = []
        cell_colors: list[str | None] = []
        for pred_label in _HEATMAP_CLASS_ORDER:
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
                "colors": ["DIM", cell_colors[0], cell_colors[1], cell_colors[2], _heatmap_cell_style(percentage=row_total_pct)],
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
    colored_rows.append(
        {
            "cells": total_row,
            "colors": total_colors,
        }
    )

    table_lines: list[str]
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
    lines: list[str] = [
        "Heatmap GT→Pred (live):",
        *table_lines,
        "Leyenda: celda = conteo / %global / %filaGT · color = %global (blanco→amarillo→naranja→escarlata)",
    ]
    return lines


def _project_root() -> Path:
    """Obtiene la raíz del repositorio independientemente del cwd actual."""
    return Path(__file__).resolve().parents[3]


def _resolve_existing_file(path_like: str | Path) -> Path | None:
    """Resuelve una ruta de archivo existente probando absoluta, cwd y raíz de proyecto."""
    raw = Path(path_like)
    candidates: list[Path] = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(Path.cwd() / raw)
        candidates.append(_project_root() / raw)

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


def _resolve_existing_dir(path_like: str | Path) -> Path | None:
    """Resuelve una ruta de directorio existente probando absoluta, cwd y raíz de proyecto."""
    raw = Path(path_like)
    candidates: list[Path] = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(Path.cwd() / raw)
        candidates.append(_project_root() / raw)

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


def _discover_last_scenario_output_from_meta(*, scenario_code: str) -> Path | None:
    """Descubre el último JSONL válido de un escenario usando cabecera __scenario_meta__."""
    scenario_dir = _resolve_existing_dir(
        Path("data/processed/scenario_results") / f"scenario_{scenario_code}"
    )
    if scenario_dir is None:
        return None

    candidates = sorted(
        scenario_dir.glob("run_*/results.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None

    try:
        from src.scripts.grounding_experiments.runner_core import (
            SCENARIO_META_KEY,
            load_jsonl_records,
        )
    except Exception:
        return candidates[0]

    for candidate in candidates:
        try:
            records = load_jsonl_records(candidate, include_system_records=True)
            if any(isinstance(item.get(SCENARIO_META_KEY), dict) for item in records):
                return candidate
        except Exception:
            continue
    return candidates[0]


def _is_scenario_run_incomplete(output_path: Path) -> bool:
    """Detecta si un run está incompleto por falta de summary o líneas sin rellenar."""
    try:
        from src.scripts.grounding_experiments.runner_core import (
            SCENARIO_META_KEY,
            SCENARIO_SUMMARY_KEY,
            has_unfilled_scenario_records,
            load_jsonl_records,
        )

        records = load_jsonl_records(output_path, include_system_records=True)
        has_meta = any(isinstance(item.get(SCENARIO_META_KEY), dict) for item in records)
        has_summary = any(isinstance(item.get(SCENARIO_SUMMARY_KEY), dict) for item in records)
        has_unfilled = has_unfilled_scenario_records(output_path=output_path)
        return bool(has_meta and ((not has_summary) or has_unfilled))
    except Exception:
        return False


def _load_scenario_meta(output_path: Path) -> dict[str, Any]:
    """Carga la cabecera __scenario_meta__ si existe en el JSONL de un run."""
    try:
        from src.scripts.grounding_experiments.runner_core import (
            SCENARIO_META_KEY,
            load_jsonl_records,
        )

        records = load_jsonl_records(output_path, include_system_records=True)
        for entry in records:
            raw_meta = entry.get(SCENARIO_META_KEY)
            if isinstance(raw_meta, dict):
                return {str(k): v for k, v in raw_meta.items()}
    except Exception:
        return {}
    return {}


def _select_grounding_sample_size(kit: "UIKit") -> int | None:
    """Reutiliza el selector numérico de batch para definir número de inferencias."""
    from .manifest import _select_manifest_sample_size_with_default

    selected = _select_manifest_sample_size_with_default(kit, initial_value=None)
    if selected == "BACK":
        return None
    if isinstance(selected, int) and selected > 0:
        return selected
    return None


def _select_scenario_a_sample_size(kit: "UIKit") -> int | None:
    """Alias de compatibilidad para tests existentes."""
    return _select_grounding_sample_size(kit)


def _select_grounding_seed(kit: "UIKit", *, initial_value: int = 42) -> int | None:
    """Solicita semilla común para muestreo reproducible en escenarios A/B/C/D."""

    def _validate_seed(raw_value: str) -> str | None:
        if not raw_value:
            return "Debes indicar una semilla entera (>= 0)."
        if not raw_value.isdigit():
            return "Entrada inválida. Usa solo dígitos."
        return None

    value = ask_text(
        kit=kit,
        title="GROUNDING SCENARIOS · RANDOM SEED",
        intro_lines=[
            (
                f"{kit.style.DIM}Usa la misma seed para que los escenarios A/B/C/D"
                f" evalúen el mismo subconjunto de imágenes.{kit.style.ENDC}"
            ),
        ],
        prompt_label="Seed:",
        help_line="[ENTER] confirmar · [Backspace] borrar · [ESC] cancelar",
        initial_value=str(max(0, int(initial_value))),
        allow_char_fn=lambda ch: ch.isdigit(),
        normalize_on_submit_fn=lambda text: text.strip(),
        validate_on_submit_fn=_validate_seed,
        force_full_on_update=True,
    )
    if value is None:
        return None
    return int(value)


def _select_grounding_run_mode(
    kit: "UIKit",
    *,
    scenario_code: str,
    last_output_path: Path | None,
) -> tuple[bool, Path | None] | None:
    """Permite elegir explícitamente entre reanudar el último run o iniciar uno nuevo."""
    if last_output_path is None or not last_output_path.exists() or not last_output_path.is_file():
        return False, None

    is_incomplete = _is_scenario_run_incomplete(last_output_path)
    if not is_incomplete:
        # Si el último run está completo, no se ofrece reanudar.
        return False, None

    resume_label = " Reanudar último run (recomendado)" if is_incomplete else " Reanudar último run"

    options = [
        kit.MenuItem(
            resume_label,
            lambda: "RESUME",
            description=f"Continuar desde {last_output_path}.",
        ),
        kit.MenuItem(
            " Nuevo run",
            lambda: "NEW",
            description="Crear una nueva carpeta de ejecución.",
        ),
        kit.MenuItem(
            " Back",
            lambda: "BACK",
            description="Cancelar ejecución del escenario.",
        ),
    ]

    selected = kit.menu(
        options,
        header_func=None,
        multi_select=False,
        menu_id=f"grounding_scenario_{str(scenario_code).lower()}_run_mode_selector",
        nav_hint_text="↑/↓ navegar · ENTER seleccionar · ESC volver",
    )

    if not selected or selected == "BACK":
        return None
    if isinstance(selected, list):
        selected = selected[0] if len(selected) > 0 else None
    if not selected:
        return None
    if hasattr(selected, "action") and callable(selected.action):
        choice = selected.action()
        if choice == "RESUME":
            return True, last_output_path
        if choice == "NEW":
            return False, None
    return None


def _safe_read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Lee registros JSONL válidos de forma tolerante a escrituras concurrentes."""
    try:
        from src.scripts.grounding_experiments.runner_core import load_jsonl_records

        return load_jsonl_records(path)
    except Exception:
        return []


def _scenario_recent_record(entry: dict[str, Any]) -> dict[str, object]:
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
    gt_bbox = _coerce_bbox4(entry.get("ground_truth_bbox"))
    pred_bbox = _coerce_bbox4(
        [
            result_payload.get("ymin"),
            result_payload.get("xmin"),
            result_payload.get("ymax"),
            result_payload.get("xmax"),
        ]
    )
    if gt_bbox is not None and pred_bbox is not None:
        try:
            from .metrics import calculate_iou

            iou_value = calculate_iou(
                gt_bbox,
                pred_bbox,
            )
            payload["iou"] = f"{iou_value:.3f}"
        except Exception:
            pass
    return {
        "image_name": image_name,
        "status": "ok",
        "payload": payload,
    }


def _extract_final_diagnosis_class(entry: dict[str, Any]) -> str:
    """Obtiene clase final normalizada desde un registro JSONL de Scenario A."""
    raw_payload = entry.get("payload")
    if isinstance(raw_payload, dict):
        result_payload: dict[str, Any] = raw_payload
    else:
        raw_result = entry.get("result")
        result_payload = raw_result if isinstance(raw_result, dict) else {}
    return str(
        result_payload.get("final_diagnosis_class")
        or ""
    ).strip().upper()


def _extract_iou(entry: dict[str, Any]) -> float | None:
    """Calcula IoU para un registro JSONL de Scenario A cuando ambas cajas existen."""
    direct_iou = entry.get("iou_score")
    if isinstance(direct_iou, (int, float)):
        return float(direct_iou)

    gt_bbox = _coerce_bbox4(entry.get("ground_truth_bbox"))
    raw_payload = entry.get("payload")
    if isinstance(raw_payload, dict):
        result_payload: dict[str, Any] = raw_payload
    else:
        raw_result = entry.get("result")
        result_payload = raw_result if isinstance(raw_result, dict) else {}
    pred_bbox = _coerce_bbox4(
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
        from .metrics import calculate_iou

        return float(calculate_iou(gt_bbox, pred_bbox))
    except Exception:
        return None


def _coerce_bbox4(value: Any) -> list[int] | None:
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


def _build_comparison_summary_rows(records: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    """Construye resumen de comparación GT vs predicción e IoU para pantalla final."""
    compared = 0
    matched = 0
    mismatched = 0
    iou_values: list[float] = []

    for entry in records:
        gt_cls = str(entry.get("ground_truth_cls") or "").strip().upper()
        pred_cls = _extract_final_diagnosis_class(entry)
        if gt_cls and pred_cls:
            compared += 1
            if gt_cls == pred_cls:
                matched += 1
            else:
                mismatched += 1

        iou_value = _extract_iou(entry)
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


def _summarize_existing_scenario_records(records: list[dict[str, Any]]) -> dict[str, Any]:
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
    gt_class_counts: dict[str, int] = {"AD": 0, "HP": 0, "ASS": 0, "OTHER": 0}
    pred_class_counts: dict[str, int] = {"AD": 0, "HP": 0, "ASS": 0, "OTHER": 0}
    confusion_counts = _empty_live_confusion_counts()

    def _bucket_class(target: dict[str, int], value: Any) -> None:
        label = str(value or "").strip().upper()
        if label in {"AD", "HP", "ASS"}:
            target[label] = int(target.get(label) or 0) + 1
        else:
            target["OTHER"] = int(target.get("OTHER") or 0) + 1

    for entry in records:
        status = str(entry.get("status") or "ok").strip().lower()
        if status in {"error", "failed", "fail"}:
            # En resume, los errores previos no cuentan como progreso ya completado.
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
            pred_cls = _extract_final_diagnosis_class(entry)
            if gt_cls and pred_cls:
                if gt_cls == pred_cls:
                    matched += 1
                else:
                    mismatched += 1

        _bucket_class(gt_class_counts, entry.get("ground_truth_cls"))
        pred_cls_value = _extract_final_diagnosis_class(entry)
        _bucket_class(pred_class_counts, pred_cls_value)

        gt_conf = _normalize_heatmap_class(entry.get("ground_truth_cls"))
        pred_conf = _normalize_heatmap_class(pred_cls_value)
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

        iou_value = _extract_iou(entry)
        if isinstance(iou_value, float):
            iou_sum += iou_value
            iou_count += 1

    current = ok + skip
    avg_ttft = (ttft_sum / ttft_count) if ttft_count > 0 else None
    avg_tps = (tps_sum / tps_count) if tps_count > 0 else None
    avg_duration = (duration_sum / duration_count) if duration_count > 0 else None
    avg_iou = (iou_sum / iou_count) if iou_count > 0 else None

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
        "ttft_sum": ttft_sum,
        "ttft_count": ttft_count,
        "tps_sum": tps_sum,
        "tps_count": tps_count,
        "duration_sum": duration_sum,
        "duration_count": duration_count,
        "iou_sum": iou_sum,
        "iou_count": iou_count,
        "gt_class_counts": gt_class_counts,
        "pred_class_counts": pred_class_counts,
        "confusion_counts": confusion_counts,
    }


def _render_scenario_final_screen(
    *,
    kit: "UIKit",
    app: "AppContext",
    scenario_code: str,
    model_tag: str,
    sample_size: int,
    seed: int,
    output_path: Path,
    markdown_path: Path | None,
    records: list[dict[str, Any]],
) -> None:
    """Muestra pantalla final consistente con otros tests UI."""
    ok_count = 0
    error_count = 0
    skip_count = 0
    pending_count = 0
    for entry in records:
        status = str(entry.get("status") or "").strip().lower()
        if status in {"ok", "success"}:
            ok_count += 1
        elif status in {"invalid", "skip", "skipped"}:
            skip_count += 1
        elif status in {"error", "failed", "fail"}:
            error_count += 1
            pending_count += 1
        elif status in {"pending", "queued", "in_progress", "running"}:
            pending_count += 1
        else:
            error_count += 1
            pending_count += 1

    completed_count = ok_count + skip_count
    missing_count = max(0, sample_size - len(records))
    recent_records = [_scenario_recent_record(item) for item in records[-5:]]
    comparison_rows = _build_comparison_summary_rows(records)

    def _redraw(ui_width: int) -> None:
        _render_final_sections_screen(
            kit,
            app,
            subtitle=f"GROUNDING SCENARIO {scenario_code}",
            intro=_standard_final_intro(),
            ui_width=ui_width,
            sections=[
                (
                    "Resumen",
                    _build_summary_lines(
                        kit,
                        [
                            ("Modelo", model_tag, "OK"),
                            ("Muestra solicitada", str(sample_size), "OK"),
                            ("Seed", str(seed), "OK"),
                            ("Registros OK", str(ok_count), "OK" if ok_count > 0 else "WARN"),
                            (
                                "Completados (ok/skip)",
                                str(completed_count),
                                "OK" if pending_count == 0 and missing_count == 0 else "WARN",
                            ),
                            ("Con error", str(error_count), "WARN" if error_count > 0 else "OK"),
                            ("Pendientes por rellenar", str(pending_count), "OK" if pending_count == 0 else "WARN"),
                            ("Sin registro", str(missing_count), "OK" if missing_count == 0 else "WARN"),
                            (
                                "Acción recomendada",
                                (
                                    "Reejecutar en modo Resume para completar pendientes"
                                    if pending_count > 0 or missing_count > 0
                                    else "Run completo"
                                ),
                                "WARN" if pending_count > 0 or missing_count > 0 else "OK",
                            ),
                            ("Salida", str(output_path), "OK"),
                            ("Reporte MD", str(markdown_path) if markdown_path is not None else "N/D", "OK"),
                        ],
                        ui_width=ui_width,
                    ),
                ),
                (
                    "Actividad reciente",
                    _build_recent_record_lines(
                        kit,
                        recent_records,
                        empty_message="  Sin resultados exportados.",
                        truncate=True,
                        ui_width=ui_width,
                    ),
                ),
                (
                    "Comparación GT vs Predicción",
                    _build_summary_lines(
                        kit,
                        comparison_rows,
                        ui_width=ui_width,
                    ),
                ),
            ],
        )

    _redraw(kit.width())
    kit.render_and_wait_responsive(
        render_fn=_redraw,
        message="Press any key to return to tests menu...",
        initial_render=False,
    )


def _run_scenario_with_dashboards(
    *,
    kit: "UIKit",
    app: "AppContext",
    scenario_code: str,
    run_main: Callable[..., int],
    model_tag: str,
    sample_size: int,
    seed: int,
    resume_mode: bool = False,
    resume_output_path: Path | None = None,
) -> int:
    """Ejecuta un escenario de grounding con dashboard en vivo y pantalla final."""

    scenario_name = f"scenario_{scenario_code}"
    default_output_path = Path("data/processed/scenario_results") / scenario_name / f"run_{time.strftime('%Y%m%d_%H%M%S')}" / "results.jsonl"
    output_path = (
        resume_output_path
        if resume_mode and resume_output_path is not None
        else default_output_path
    )
    raw_outputs = kit.cursor_memory.get(_GROUNDING_LAST_OUTPUTS_CURSOR_KEY)
    known_outputs = cast(dict[str, str], raw_outputs) if isinstance(raw_outputs, dict) else {}
    markdown_path: Path | None = None
    existing_records: list[dict[str, Any]] = []
    existing_snapshot: dict[str, Any] = {}
    if resume_mode and output_path.exists() and output_path.is_file():
        existing_records = _safe_read_jsonl_records(output_path)
        existing_snapshot = _summarize_existing_scenario_records(existing_records)

    panel = _make_live_panel(
        kit,
        app,
        subtitle=f"GROUNDING SCENARIO {scenario_code}",
        intro=(
            f"Ejecutando Scenario {scenario_code} con modelo {model_tag} "
            f"sobre {sample_size} imágenes · seed={seed}..."
        ),
    )
    recent_records = _make_recent_records(limit=4)
    run_done = False
    run_exit_code = 1
    run_error: BaseException | None = None
    cached_records: list[dict[str, Any]] = []
    state_lock = threading.Lock()
    state_dirty = True
    existing_current = int(existing_snapshot.get("current") or 0)
    live_state: dict[str, Any] = {
        "current": existing_current,
        "total": sample_size,
        "ok": int(existing_snapshot.get("ok") or 0),
        "fail": int(existing_snapshot.get("fail") or 0),
        "skip": int(existing_snapshot.get("skip") or 0),
        "matched": int(existing_snapshot.get("matched") or 0),
        "mismatched": int(existing_snapshot.get("mismatched") or 0),
        "avg_ttft": existing_snapshot.get("avg_ttft"),
        "avg_tps": existing_snapshot.get("avg_tps"),
        "avg_duration": existing_snapshot.get("avg_duration"),
        "avg_iou": existing_snapshot.get("avg_iou"),
        "ttft_sum": float(existing_snapshot.get("ttft_sum") or 0.0),
        "ttft_count": int(existing_snapshot.get("ttft_count") or 0),
        "tps_sum": float(existing_snapshot.get("tps_sum") or 0.0),
        "tps_count": int(existing_snapshot.get("tps_count") or 0),
        "duration_sum": float(existing_snapshot.get("duration_sum") or 0.0),
        "duration_count": int(existing_snapshot.get("duration_count") or 0),
        "iou_sum": float(existing_snapshot.get("iou_sum") or 0.0),
        "iou_count": int(existing_snapshot.get("iou_count") or 0),
        "resumed_existing": existing_current,
        "gt_class_counts": cast(
            dict[str, int],
            existing_snapshot.get("gt_class_counts") or {"AD": 0, "HP": 0, "ASS": 0, "OTHER": 0},
        ),
        "pred_class_counts": cast(
            dict[str, int],
            existing_snapshot.get("pred_class_counts") or {"AD": 0, "HP": 0, "ASS": 0, "OTHER": 0},
        ),
        "confusion_counts": cast(
            dict[str, dict[str, int]],
            existing_snapshot.get("confusion_counts") or _empty_live_confusion_counts(),
        ),
        "status_line": (
            f"Estado: preparando ejecución ({existing_current}/{sample_size})"
            if not resume_mode
            else f"Estado: reanudando ejecución ({existing_current}/{sample_size})"
        ),
    }

    argv = [
        "--model",
        model_tag,
        "--no-progress",
        "--limit",
        str(sample_size),
        "--seed",
        str(seed),
        "--output",
        str(output_path),
    ]
    if resume_mode:
        argv.append("--resume")

    def _on_event(event: str, payload: dict[str, Any]) -> None:
        nonlocal cached_records, state_dirty, output_path, markdown_path

        def _inc_class_count(target: dict[str, int], value: Any) -> None:
            label = str(value or "").strip().upper()
            if label in {"AD", "HP", "ASS"}:
                target[label] = int(target.get(label) or 0) + 1
            else:
                target["OTHER"] = int(target.get("OTHER") or 0) + 1

        with state_lock:
            if event == "run_start":
                remaining_total = int(payload.get("total") or sample_size)
                resumed_existing = int(payload.get("resumed_existing") or 0)
                if resumed_existing > 0:
                    live_state["total"] = resumed_existing + max(0, remaining_total)
                    live_state["current"] = resumed_existing
                else:
                    live_state["total"] = max(remaining_total, int(live_state.get("current") or 0), sample_size)
                live_state["resumed_existing"] = resumed_existing
                live_state["current"] = min(
                    int(live_state.get("current") or 0),
                    int(live_state.get("total") or sample_size),
                )
                live_state["status_line"] = (
                    "Estado: inicializando modelo..."
                    if resumed_existing <= 0
                    else (
                        "Estado: reanudando "
                        f"({int(live_state.get('current') or resumed_existing)}/{int(live_state.get('total') or sample_size)})"
                    )
                )
                payload_output = payload.get("output_path")
                if payload_output:
                    try:
                        output_path = Path(str(payload_output))
                    except Exception:
                        pass
                state_dirty = True
                return

            if event == "image_start":
                index = int(payload.get("index") or live_state.get("current") or 0)
                total = int(payload.get("total") or live_state.get("total") or sample_size)
                image_id = str(payload.get("image_id") or "imagen")
                resumed_existing = int(live_state.get("resumed_existing") or 0)
                if resumed_existing > 0:
                    display_index = resumed_existing + index
                    display_total = resumed_existing + total
                else:
                    display_index = index
                    display_total = total
                live_state["status_line"] = (
                    f"Estado  |  Imagen {display_index}/{display_total}  |  ID {image_id}"
                )
                state_dirty = True
                return

            if event == "image_ok":
                live_state["current"] = int(live_state.get("current") or 0) + 1
                live_state["current"] = min(int(live_state.get("current") or 0), int(live_state.get("total") or sample_size))
                live_state["ok"] = int(live_state.get("ok") or 0) + 1
                if bool(payload.get("class_match")):
                    live_state["matched"] = int(live_state.get("matched") or 0) + 1
                else:
                    live_state["mismatched"] = int(live_state.get("mismatched") or 0) + 1

                telemetry = payload.get("telemetry") if isinstance(payload.get("telemetry"), dict) else {}
                ttft_text = telemetry.get("ttft_seconds") if isinstance(telemetry, dict) else None
                tps_text = telemetry.get("tokens_per_second") if isinstance(telemetry, dict) else None
                if isinstance(telemetry, dict):
                    ttft_value = telemetry.get("ttft_seconds")
                    if isinstance(ttft_value, (int, float)):
                        live_state["ttft_sum"] = float(live_state.get("ttft_sum") or 0.0) + float(ttft_value)
                        live_state["ttft_count"] = int(live_state.get("ttft_count") or 0) + 1
                        ttft_count = int(live_state.get("ttft_count") or 0)
                        if ttft_count > 0:
                            live_state["avg_ttft"] = float(live_state.get("ttft_sum") or 0.0) / ttft_count

                    tps_value = telemetry.get("tokens_per_second")
                    if isinstance(tps_value, (int, float)):
                        live_state["tps_sum"] = float(live_state.get("tps_sum") or 0.0) + float(tps_value)
                        live_state["tps_count"] = int(live_state.get("tps_count") or 0) + 1
                        tps_count = int(live_state.get("tps_count") or 0)
                        if tps_count > 0:
                            live_state["avg_tps"] = float(live_state.get("tps_sum") or 0.0) / tps_count

                    duration_value = telemetry.get("total_duration_seconds")
                    if isinstance(duration_value, (int, float)):
                        live_state["duration_sum"] = float(live_state.get("duration_sum") or 0.0) + float(duration_value)
                        live_state["duration_count"] = int(live_state.get("duration_count") or 0) + 1
                        duration_count = int(live_state.get("duration_count") or 0)
                        if duration_count > 0:
                            live_state["avg_duration"] = float(live_state.get("duration_sum") or 0.0) / duration_count

                iou_value = payload.get("iou_score")
                if isinstance(iou_value, (int, float)):
                    live_state["iou_sum"] = float(live_state.get("iou_sum") or 0.0) + float(iou_value)
                    live_state["iou_count"] = int(live_state.get("iou_count") or 0) + 1
                    iou_count = int(live_state.get("iou_count") or 0)
                    if iou_count > 0:
                        live_state["avg_iou"] = float(live_state.get("iou_sum") or 0.0) / iou_count
                _inc_class_count(cast(dict[str, int], live_state["gt_class_counts"]), payload.get("ground_truth_cls"))
                _inc_class_count(cast(dict[str, int], live_state["pred_class_counts"]), payload.get("predicted_cls"))

                gt_conf = _normalize_heatmap_class(payload.get("ground_truth_cls"))
                pred_conf = _normalize_heatmap_class(payload.get("predicted_cls"))
                if gt_conf is not None and pred_conf is not None:
                    confusion_counts = cast(dict[str, dict[str, int]], live_state.get("confusion_counts") or _empty_live_confusion_counts())
                    row = confusion_counts.setdefault(gt_conf, {})
                    row[pred_conf] = int(row.get(pred_conf) or 0) + 1
                    live_state["confusion_counts"] = confusion_counts

                recent_records.append(
                    {
                        "image_name": str(payload.get("image_id") or "imagen"),
                        "status": "ok",
                        "payload": {
                            "gt_cls": str(payload.get("ground_truth_cls") or "N/D"),
                            "pred_cls": str(payload.get("predicted_cls") or "N/D"),
                            "ttft": str(ttft_text) if ttft_text is not None else "N/D",
                            "tps": str(tps_text) if tps_text is not None else "N/D",
                        },
                    }
                )
                state_dirty = True
                return

            if event == "image_error":
                live_state["current"] = int(live_state.get("current") or 0) + 1
                live_state["current"] = min(int(live_state.get("current") or 0), int(live_state.get("total") or sample_size))
                live_state["fail"] = int(live_state.get("fail") or 0) + 1
                recent_records.append(
                    {
                        "image_name": str(payload.get("image_id") or "imagen"),
                        "status": "error",
                        "error": str(payload.get("error") or "inference error"),
                    }
                )
                state_dirty = True
                return

            if event == "image_skip":
                live_state["current"] = int(live_state.get("current") or 0) + 1
                live_state["current"] = min(int(live_state.get("current") or 0), int(live_state.get("total") or sample_size))
                live_state["skip"] = int(live_state.get("skip") or 0) + 1
                recent_records.append(
                    {
                        "image_name": str(payload.get("image_id") or "imagen"),
                        "status": "invalid",
                        "error": str(payload.get("error") or "skip"),
                    }
                )
                state_dirty = True
                return

            if event == "run_complete":
                resumed_existing = int(live_state.get("resumed_existing") or 0)
                if resumed_existing <= 0:
                    live_state["ok"] = int(payload.get("ok") or live_state.get("ok") or 0)
                    live_state["fail"] = int(payload.get("fail") or live_state.get("fail") or 0)
                    live_state["skip"] = int(payload.get("skip") or live_state.get("skip") or 0)
                    live_state["matched"] = int(payload.get("matched_class") or live_state.get("matched") or 0)
                    live_state["mismatched"] = int(payload.get("mismatched_class") or live_state.get("mismatched") or 0)
                    live_state["avg_ttft"] = payload.get("avg_ttft_seconds")
                    live_state["avg_tps"] = payload.get("avg_tokens_per_second")
                    live_state["avg_duration"] = payload.get("avg_total_duration_seconds")
                    live_state["avg_iou"] = payload.get("avg_iou")
                payload_md = payload.get("markdown_path")
                if payload_md:
                    try:
                        markdown_path = Path(str(payload_md))
                    except Exception:
                        markdown_path = None
                live_state["status_line"] = (
                    f"Estado  |  Finalizado  |  OK {live_state['ok']} · Error {live_state['fail']} · Skip {live_state['skip']}"
                )
                state_dirty = True

    def _runner() -> None:
        nonlocal run_done, run_exit_code, run_error
        try:
            run_exit_code = int(run_main(argv, reporter=_on_event))
        except Exception as error:
            run_error = error
        finally:
            run_done = True

    def _render_live(*, force_full: bool | None = None) -> None:
        with state_lock:
            current = int(live_state.get("current") or 0)
            total = int(live_state.get("total") or sample_size)
            ok_count = int(live_state.get("ok") or 0)
            fail_count = int(live_state.get("fail") or 0)
            skip_count = int(live_state.get("skip") or 0)
            matched = int(live_state.get("matched") or 0)
            mismatched = int(live_state.get("mismatched") or 0)
            avg_ttft = live_state.get("avg_ttft")
            avg_tps = live_state.get("avg_tps")
            avg_duration = live_state.get("avg_duration")
            avg_iou = live_state.get("avg_iou")
            status_line = str(live_state.get("status_line") or "Estado: ejecutando")
            confusion_counts = cast(
                dict[str, dict[str, int]],
                live_state.get("confusion_counts") or _empty_live_confusion_counts(),
            )

        safe_total = max(0, total)
        safe_current = max(0, min(current, safe_total if safe_total > 0 else current))

        cls_total = matched + mismatched
        accuracy_text = f"{(matched / cls_total * 100.0):.1f}%" if cls_total > 0 else "N/D"
        ttft_text = f"{float(avg_ttft):.3f}s" if isinstance(avg_ttft, (int, float)) else "N/D"
        tps_text = f"{float(avg_tps):.2f}" if isinstance(avg_tps, (int, float)) else "N/D"
        duration_text = f"{float(avg_duration):.3f}s" if isinstance(avg_duration, (int, float)) else "N/D"
        metrics_line = f"Rendimiento  |  TTFT {ttft_text}  ·  TPS {tps_text}  ·  Lat {duration_text}"
        iou_text = f"{avg_iou:.4f}" if isinstance(avg_iou, (int, float)) else "N/D"
        heatmap_lines = _build_live_confusion_heatmap_lines(
            kit,
            confusion_counts,
            ui_width=kit.width(),
        )
        output_text = str(output_path).replace("\\", "/")

        _render_live_dashboard(
            kit,
            panel,
            current=safe_current,
            total=safe_total,
            stats_line=(
                f"Resultados  |  OK {ok_count}  ·  Error {fail_count}  ·  Skip {skip_count}"
                f"  ||  Clase  Match {matched}  ·  Mismatch {mismatched}  ·  Acc {accuracy_text}  ·  IoU {iou_text}"
            ),
            status_line=status_line,
            metrics_line=metrics_line,
            coverage_line=None,
            extra_lines=[*heatmap_lines, f"Salida JSONL  |  {output_text}"],
            recent_title="Últimos registros exportados:",
            recent_records=list(recent_records),
            force_full=force_full,
        )

    live_renderer = ReactiveTerminalRenderer(kit=kit, render_fn=_render_live)
    worker = threading.Thread(target=_runner, daemon=True)

    try:
        worker.start()
        live_renderer.start()
        while not run_done:
            should_render = False
            with state_lock:
                if state_dirty:
                    state_dirty = False
                    should_render = True

            if should_render:
                live_renderer.render()
            time.sleep(0.08)

        # Fuerza un último frame con el estado final recibido del runner.
        live_renderer.render()
        cached_records = _safe_read_jsonl_records(output_path)
        known_outputs[scenario_code] = str(output_path)
        kit.cursor_memory[_GROUNDING_LAST_OUTPUTS_CURSOR_KEY] = known_outputs
    finally:
        live_renderer.stop()
        worker.join(timeout=1.0)

    if run_error is not None:
        raise RuntimeError(f"Scenario {scenario_code} failed: {run_error}") from run_error

    _render_scenario_final_screen(
        kit=kit,
        app=app,
        scenario_code=scenario_code,
        model_tag=model_tag,
        sample_size=sample_size,
        seed=seed,
        output_path=output_path,
        markdown_path=markdown_path,
        records=cached_records,
    )

    return run_exit_code


def _run_scenario_a_with_dashboards(
    *,
    kit: "UIKit",
    app: "AppContext",
    model_tag: str,
    sample_size: int,
    seed: int,
    resume_mode: bool = False,
    resume_output_path: Path | None = None,
) -> int:
    """Ejecuta Scenario A con dashboard en vivo y pantalla final."""
    from src.scripts.grounding_experiments.run_scenario_A import main as run_scenario_a_main

    return _run_scenario_with_dashboards(
        kit=kit,
        app=app,
        scenario_code="A",
        run_main=run_scenario_a_main,
        model_tag=model_tag,
        sample_size=sample_size,
        seed=seed,
        resume_mode=resume_mode,
        resume_output_path=resume_output_path,
    )


def _run_scenario_b_with_dashboards(
    *,
    kit: "UIKit",
    app: "AppContext",
    model_tag: str,
    sample_size: int,
    seed: int,
    resume_mode: bool = False,
    resume_output_path: Path | None = None,
) -> int:
    """Ejecuta Scenario B con dashboard en vivo y pantalla final."""
    from src.scripts.grounding_experiments.run_scenario_B import main as run_scenario_b_main

    return _run_scenario_with_dashboards(
        kit=kit,
        app=app,
        scenario_code="B",
        run_main=run_scenario_b_main,
        model_tag=model_tag,
        sample_size=sample_size,
        seed=seed,
        resume_mode=resume_mode,
        resume_output_path=resume_output_path,
    )


def run_grounding_scenarios_selector_wrapper(
    kit: "UIKit",
    app: "AppContext",
    *,
    make_header_fn: Callable[[str], Any],
    select_model: Callable[[str, str], str | None],
) -> None:
    """Muestra selector stub de escenarios de grounding A/B/C/D.

    Este stub mantiene la UX y el texto tal como estaba en el menú original.
    Más adelante se podrá reemplazar la acción de cada opción por el runner
    correspondiente (A/B/C/D).
    """

    scenario_options = [
        kit.MenuItem(
            " Scenario A (Zero-Shot BBox)",
            lambda: "A",
            description="Escenario A seleccionado (stub).",
        ),
        kit.MenuItem(
            " Scenario B (BBox Asistido)",
            lambda: "B",
            description="Escenario B seleccionado (stub).",
        ),
        kit.MenuItem(
            " Scenario C (BBox Forzado visualmente)",
            lambda: "C",
            description="Escenario C seleccionado (stub).",
        ),
        kit.MenuItem(
            " Scenario D (Techo de Calidad: BBox + Clase)",
            lambda: "D",
            description="Escenario D seleccionado (stub).",
        ),
        kit.MenuItem(
            " Back",
            lambda: "BACK",
            description="Volver al menú de tests.",
        ),
    ]

    selected = kit.menu(
        scenario_options,
        header_func=make_header_fn("GROUNDING SCENARIOS SELECTOR"),
        multi_select=False,
        menu_id="grounding_scenarios_selector",
        nav_hint_text="↑/↓ navegar escenarios · ENTER seleccionar · ESC volver",
    )
    if not selected or selected == "BACK":
        return

    if isinstance(selected, list):
        selected = selected[0] if len(selected) > 0 else None
    if not selected:
        return

    if hasattr(selected, "action") and callable(selected.action):
        scenario_code = str(selected.action())
        if scenario_code in {"A", "B"}:
            raw_outputs = kit.cursor_memory.get(_GROUNDING_LAST_OUTPUTS_CURSOR_KEY)
            known_outputs = cast(dict[str, str], raw_outputs) if isinstance(raw_outputs, dict) else {}
            last_output_path = None
            last_output_raw = known_outputs.get(scenario_code)
            if last_output_raw:
                last_output_path = _resolve_existing_file(str(last_output_raw))
            if last_output_path is None:
                last_output_path = _discover_last_scenario_output_from_meta(scenario_code=scenario_code)

            run_mode = _select_grounding_run_mode(
                kit,
                scenario_code=scenario_code,
                last_output_path=last_output_path,
            )
            if run_mode is None:
                return
            resume_mode, resume_output_path = run_mode

            previous_seed = int(kit.cursor_memory.get(_GROUNDING_SEED_CURSOR_KEY) or 42)
            selected_seed = previous_seed
            selected_sample_size: int | None = None
            model_tag: str | None = None

            if resume_mode and resume_output_path is not None:
                meta = _load_scenario_meta(resume_output_path)
                raw_model = str(meta.get("model_id") or "").strip()
                if raw_model:
                    model_tag = raw_model

                raw_seed = meta.get("seed")
                if isinstance(raw_seed, int) and raw_seed >= 0:
                    selected_seed = raw_seed

                raw_limit = meta.get("requested_limit")
                if isinstance(raw_limit, int) and raw_limit > 0:
                    selected_sample_size = raw_limit

            if model_tag is None:
                model_tag = select_model(
                    f"grounding_scenario_{str(scenario_code).lower()}_model_selector",
                    f"GROUNDING SCENARIO {scenario_code} · SELECT MODEL",
                )
                if model_tag is None:
                    return

            if selected_sample_size is None:
                selected_sample_size = _select_grounding_sample_size(kit)
                if selected_sample_size is None:
                    return

            if not (resume_mode and resume_output_path is not None and selected_seed >= 0):
                selected_seed_input = _select_grounding_seed(kit, initial_value=previous_seed)
                if selected_seed_input is None:
                    return
                selected_seed = selected_seed_input

            kit.cursor_memory[_GROUNDING_SEED_CURSOR_KEY] = selected_seed

            try:
                if scenario_code == "A":
                    exit_code = _run_scenario_a_with_dashboards(
                        kit=kit,
                        app=app,
                        model_tag=model_tag,
                        sample_size=selected_sample_size,
                        seed=selected_seed,
                        resume_mode=resume_mode,
                        resume_output_path=resume_output_path,
                    )
                else:
                    exit_code = _run_scenario_b_with_dashboards(
                        kit=kit,
                        app=app,
                        model_tag=model_tag,
                        sample_size=selected_sample_size,
                        seed=selected_seed,
                        resume_mode=resume_mode,
                        resume_output_path=resume_output_path,
                    )
                if exit_code == 0:
                    kit.log(f"Scenario {scenario_code} completed successfully.", "success")
                else:
                    kit.log(f"Scenario {scenario_code} finished with warnings/errors.", "warning")
            except Exception as error:
                kit.log(f"Scenario {scenario_code} failed: {error}", "error")
            return

        if scenario_code in {"C", "D"}:
            kit.wait(
                f"Scenario {scenario_code} seleccionado. Placeholder: pendiente de integración con runner."
            )
