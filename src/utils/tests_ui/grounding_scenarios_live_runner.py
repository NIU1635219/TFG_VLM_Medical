"""Runner live para escenarios de grounding con estado encapsulado.

Separa la lógica de ejecución reactiva (hilo + dashboard + agregación live)
del selector de escenarios para mantener el archivo principal compacto.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, cast

from .grounding_scenarios_helpers import (
    HEATMAP_CLASS_ORDER,
    build_comparison_summary_rows,
    build_live_confusion_heatmap_lines,
    empty_live_confusion_counts,
    normalize_heatmap_class,
    scenario_recent_record,
    summarize_existing_scenario_records,
)
from .shared import ReactiveTerminalRenderer
from .test_dashboards_ui import (
    _build_recent_record_lines,
    _build_summary_lines,
    _make_live_panel,
    _make_recent_records,
    _render_final_sections_screen,
    _render_live_dashboard,
    _standard_final_intro,
)


@dataclass
class LiveRunState:
    """Estado mutable del dashboard live durante la ejecución."""

    current: int
    total: int
    ok: int
    fail: int
    skip: int
    matched: int
    mismatched: int
    avg_ttft: float | None
    avg_tps: float | None
    avg_duration: float | None
    avg_iou: float | None
    ttft_sum: float
    ttft_count: int
    tps_sum: float
    tps_count: int
    duration_sum: float
    duration_count: int
    iou_sum: float
    iou_count: int
    resumed_existing: int
    gt_class_counts: dict[str, int]
    pred_class_counts: dict[str, int]
    confusion_counts: dict[str, dict[str, int]]
    status_line: str


def _safe_read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Lee registros JSONL válidos de forma tolerante a escrituras concurrentes."""
    try:
        from src.scripts.grounding_experiments.runner_core import load_jsonl_records

        return load_jsonl_records(path)
    except Exception:
        return []


def _render_scenario_final_screen(
    *,
    kit: Any,
    app: Any,
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
    recent_records = [scenario_recent_record(item) for item in records[-5:]]
    comparison_rows = build_comparison_summary_rows(records)

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


def run_scenario_with_dashboards(
    *,
    kit: Any,
    app: Any,
    scenario_code: str,
    run_main: Callable[..., int],
    model_tag: str,
    sample_size: int,
    seed: int,
    outputs_cursor_key: str,
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
    raw_outputs = kit.cursor_memory.get(outputs_cursor_key)
    known_outputs = cast(dict[str, str], raw_outputs) if isinstance(raw_outputs, dict) else {}
    markdown_path: Path | None = None
    existing_snapshot: dict[str, Any] = {}
    if resume_mode and output_path.exists() and output_path.is_file():
        existing_records = _safe_read_jsonl_records(output_path)
        existing_snapshot = summarize_existing_scenario_records(existing_records)

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

    state = LiveRunState(
        current=existing_current,
        total=sample_size,
        ok=int(existing_snapshot.get("ok") or 0),
        fail=int(existing_snapshot.get("fail") or 0),
        skip=int(existing_snapshot.get("skip") or 0),
        matched=int(existing_snapshot.get("matched") or 0),
        mismatched=int(existing_snapshot.get("mismatched") or 0),
        avg_ttft=existing_snapshot.get("avg_ttft"),
        avg_tps=existing_snapshot.get("avg_tps"),
        avg_duration=existing_snapshot.get("avg_duration"),
        avg_iou=existing_snapshot.get("avg_iou"),
        ttft_sum=float(existing_snapshot.get("ttft_sum") or 0.0),
        ttft_count=int(existing_snapshot.get("ttft_count") or 0),
        tps_sum=float(existing_snapshot.get("tps_sum") or 0.0),
        tps_count=int(existing_snapshot.get("tps_count") or 0),
        duration_sum=float(existing_snapshot.get("duration_sum") or 0.0),
        duration_count=int(existing_snapshot.get("duration_count") or 0),
        iou_sum=float(existing_snapshot.get("iou_sum") or 0.0),
        iou_count=int(existing_snapshot.get("iou_count") or 0),
        resumed_existing=existing_current,
        gt_class_counts=cast(dict[str, int], existing_snapshot.get("gt_class_counts") or {"AD": 0, "HP": 0, "ASS": 0, "OTHER": 0}),
        pred_class_counts=cast(dict[str, int], existing_snapshot.get("pred_class_counts") or {"AD": 0, "HP": 0, "ASS": 0, "OTHER": 0}),
        confusion_counts=cast(dict[str, dict[str, int]], existing_snapshot.get("confusion_counts") or empty_live_confusion_counts()),
        status_line=(
            f"Estado: preparando ejecución ({existing_current}/{sample_size})"
            if not resume_mode
            else f"Estado: reanudando ejecución ({existing_current}/{sample_size})"
        ),
    )

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
            if label in set(HEATMAP_CLASS_ORDER):
                target[label] = int(target.get(label) or 0) + 1
            else:
                target["OTHER"] = int(target.get("OTHER") or 0) + 1

        with state_lock:
            if event == "run_start":
                remaining_total = int(payload.get("total") or sample_size)
                resumed_existing = int(payload.get("resumed_existing") or 0)
                if resumed_existing > 0:
                    state.total = resumed_existing + max(0, remaining_total)
                    state.current = resumed_existing
                else:
                    state.total = max(remaining_total, int(state.current or 0), sample_size)
                state.resumed_existing = resumed_existing
                state.current = min(int(state.current or 0), int(state.total or sample_size))
                state.status_line = (
                    "Estado: inicializando modelo..."
                    if resumed_existing <= 0
                    else f"Estado: reanudando ({int(state.current or resumed_existing)}/{int(state.total or sample_size)})"
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
                index = int(payload.get("index") or state.current or 0)
                total = int(payload.get("total") or state.total or sample_size)
                image_id = str(payload.get("image_id") or "imagen")
                resumed_existing = int(state.resumed_existing or 0)
                if resumed_existing > 0:
                    display_index = resumed_existing + index
                    display_total = resumed_existing + total
                else:
                    display_index = index
                    display_total = total
                state.status_line = f"Estado  |  Imagen {display_index}/{display_total}  |  ID {image_id}"
                state_dirty = True
                return

            if event == "image_ok":
                state.current = int(state.current or 0) + 1
                state.current = min(int(state.current or 0), int(state.total or sample_size))
                state.ok = int(state.ok or 0) + 1
                if bool(payload.get("class_match")):
                    state.matched = int(state.matched or 0) + 1
                else:
                    state.mismatched = int(state.mismatched or 0) + 1

                telemetry = payload.get("telemetry") if isinstance(payload.get("telemetry"), dict) else {}
                ttft_text = telemetry.get("ttft_seconds") if isinstance(telemetry, dict) else None
                tps_text = telemetry.get("tokens_per_second") if isinstance(telemetry, dict) else None
                if isinstance(telemetry, dict):
                    ttft_value = telemetry.get("ttft_seconds")
                    if isinstance(ttft_value, (int, float)):
                        state.ttft_sum = float(state.ttft_sum or 0.0) + float(ttft_value)
                        state.ttft_count = int(state.ttft_count or 0) + 1
                        if state.ttft_count > 0:
                            state.avg_ttft = float(state.ttft_sum or 0.0) / state.ttft_count

                    tps_value = telemetry.get("tokens_per_second")
                    if isinstance(tps_value, (int, float)):
                        state.tps_sum = float(state.tps_sum or 0.0) + float(tps_value)
                        state.tps_count = int(state.tps_count or 0) + 1
                        if state.tps_count > 0:
                            state.avg_tps = float(state.tps_sum or 0.0) / state.tps_count

                    duration_value = telemetry.get("total_duration_seconds")
                    if isinstance(duration_value, (int, float)):
                        state.duration_sum = float(state.duration_sum or 0.0) + float(duration_value)
                        state.duration_count = int(state.duration_count or 0) + 1
                        if state.duration_count > 0:
                            state.avg_duration = float(state.duration_sum or 0.0) / state.duration_count

                iou_value = payload.get("iou_score")
                if isinstance(iou_value, (int, float)):
                    state.iou_sum = float(state.iou_sum or 0.0) + float(iou_value)
                    state.iou_count = int(state.iou_count or 0) + 1
                    if state.iou_count > 0:
                        state.avg_iou = float(state.iou_sum or 0.0) / state.iou_count

                _inc_class_count(cast(dict[str, int], state.gt_class_counts), payload.get("ground_truth_cls"))
                _inc_class_count(cast(dict[str, int], state.pred_class_counts), payload.get("predicted_cls"))

                gt_conf = normalize_heatmap_class(payload.get("ground_truth_cls"))
                pred_conf = normalize_heatmap_class(payload.get("predicted_cls"))
                if gt_conf is not None and pred_conf is not None:
                    row = state.confusion_counts.setdefault(gt_conf, {})
                    row[pred_conf] = int(row.get(pred_conf) or 0) + 1

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
                state.current = int(state.current or 0) + 1
                state.current = min(int(state.current or 0), int(state.total or sample_size))
                state.fail = int(state.fail or 0) + 1
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
                state.current = int(state.current or 0) + 1
                state.current = min(int(state.current or 0), int(state.total or sample_size))
                state.skip = int(state.skip or 0) + 1
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
                resumed_existing = int(state.resumed_existing or 0)
                if resumed_existing <= 0:
                    state.ok = int(payload.get("ok") or state.ok or 0)
                    state.fail = int(payload.get("fail") or state.fail or 0)
                    state.skip = int(payload.get("skip") or state.skip or 0)
                    state.matched = int(payload.get("matched_class") or state.matched or 0)
                    state.mismatched = int(payload.get("mismatched_class") or state.mismatched or 0)
                    state.avg_ttft = payload.get("avg_ttft_seconds")
                    state.avg_tps = payload.get("avg_tokens_per_second")
                    state.avg_duration = payload.get("avg_total_duration_seconds")
                    state.avg_iou = payload.get("avg_iou")
                payload_md = payload.get("markdown_path")
                if payload_md:
                    try:
                        markdown_path = Path(str(payload_md))
                    except Exception:
                        markdown_path = None
                state.status_line = f"Estado  |  Finalizado  |  OK {state.ok} · Error {state.fail} · Skip {state.skip}"
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
            current = int(state.current or 0)
            total = int(state.total or sample_size)
            ok_count = int(state.ok or 0)
            fail_count = int(state.fail or 0)
            skip_count = int(state.skip or 0)
            matched = int(state.matched or 0)
            mismatched = int(state.mismatched or 0)
            avg_ttft = state.avg_ttft
            avg_tps = state.avg_tps
            avg_duration = state.avg_duration
            avg_iou = state.avg_iou
            status_line = str(state.status_line or "Estado: ejecutando")
            confusion_counts = cast(dict[str, dict[str, int]], state.confusion_counts or empty_live_confusion_counts())

        safe_total = max(0, total)
        safe_current = max(0, min(current, safe_total if safe_total > 0 else current))

        cls_total = matched + mismatched
        accuracy_text = f"{(matched / cls_total * 100.0):.1f}%" if cls_total > 0 else "N/D"
        ttft_text = f"{float(avg_ttft):.3f}s" if isinstance(avg_ttft, (int, float)) else "N/D"
        tps_text = f"{float(avg_tps):.2f}" if isinstance(avg_tps, (int, float)) else "N/D"
        duration_text = f"{float(avg_duration):.3f}s" if isinstance(avg_duration, (int, float)) else "N/D"
        metrics_line = f"Rendimiento  |  TTFT {ttft_text}  ·  TPS {tps_text}  ·  Lat {duration_text}"
        iou_text = f"{avg_iou:.4f}" if isinstance(avg_iou, (int, float)) else "N/D"
        heatmap_lines = build_live_confusion_heatmap_lines(
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

        live_renderer.render()
        cached_records = _safe_read_jsonl_records(output_path)
        known_outputs[scenario_code] = str(output_path)
        kit.cursor_memory[outputs_cursor_key] = known_outputs
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
