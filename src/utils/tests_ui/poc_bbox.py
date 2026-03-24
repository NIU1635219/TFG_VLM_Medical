from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from .shared import ReactiveTerminalRenderer
from .test_dashboards_ui import (
    _append_recent_record,
    _build_recent_record_lines,
    _build_summary_lines,
    _coerce_int,
    _make_live_panel,
    _make_recent_records,
    _render_final_sections_screen,
    _render_live_dashboard,
    _standard_final_intro,
)

if TYPE_CHECKING:
    from ..menu_kit import AppContext, UIKit


def _format_bbox_payload(
    *,
    count: int,
    reasoning: object,
    detections: list[dict[str, Any]],
) -> dict[str, object]:
    """Genera un payload compacto para la tabla de actividad reciente.

    El truncado/wrap final lo aplica `_build_recent_record_lines` para mantener
    consistencia visual con el resto de menús.
    """
    payload: dict[str, object] = {
        "detected_subjects_count": count,
        "object_count_reasoning": str(reasoning or ""),
    }

    max_rows = 4
    for idx, detection in enumerate(detections[:max_rows], start=1):
        subject = str(detection.get("detected_subject") or "")
        bbox = (
            f"[{detection.get('ymin', 0)}, {detection.get('xmin', 0)}, "
            f"{detection.get('ymax', 0)}, {detection.get('xmax', 0)}]"
        )
        payload[f"bbox_{idx}"] = f"{subject} | {bbox}"

    remaining = max(0, len(detections) - max_rows)
    if remaining:
        payload["bbox_more"] = f"{remaining} detecciones adicionales…"

    return payload


def run_poc_bbox_wrapper(
    kit: "UIKit",
    app: "AppContext",
    *,
    select_model: Callable[[str, str], str | None],
) -> None:
    """Ejecuta la PoC de visual grounding (bounding boxes) desde la TUI de tests."""
    from src.scripts.poc_bbox import main as run_poc_bbox_main

    while True:
        final_screen_renderer: Callable[[int], None] | None = None
        model_tag = select_model(
            "run_poc_bbox_model_selector",
            "BBOX POC · SELECT MODEL",
        )
        if model_tag is None:
            return

        panel = _make_live_panel(
            kit,
            app,
            subtitle="BBOX POC · VISUAL GROUNDING",
            intro=f"Ejecutando PoC de bounding boxes con modelo {model_tag}...",
        )
        recent_records = _make_recent_records(limit=4)
        all_records: list[dict[str, object]] = []

        summary: dict[str, object] = {
            "model_tag": model_tag,
            "downloads_started": 0,
            "downloads_ok": 0,
            "downloads_fail": 0,
            "images_total": 0,
            "images_started": 0,
            "images_processed": 0,
            "images_ok": 0,
            "images_fail": 0,
            "detected_subjects_total": 0,
            "output_dir": None,
        }
        live_status_line = "Estado: preparando ejecución..."

        def _render_live_bbox_dashboard(*, force_full: bool | None = None) -> None:
            processed = _coerce_int(summary.get("images_processed"))
            started = _coerce_int(summary.get("images_started"))
            total = _coerce_int(summary.get("images_total"))
            _render_live_dashboard(
                kit,
                panel,
                current=processed,
                total=total if total > 0 else max(started, processed),
                stats_line=(
                    f"Estadísticas: OK={summary.get('images_ok', 0)} | "
                    f"Errores={summary.get('images_fail', 0)} | "
                    f"Subjects totales={summary.get('detected_subjects_total', 0)}"
                ),
                status_line=live_status_line,
                metrics_line=(
                    f"Descargas: {summary.get('downloads_ok', 0)} OK / "
                    f"{summary.get('downloads_fail', 0)} FAIL"
                ),
                recent_title="Actividad reciente:",
                recent_records=list(recent_records),
                force_full=force_full,
            )

        live_renderer = ReactiveTerminalRenderer(
            kit=kit,
            render_fn=_render_live_bbox_dashboard,
        )

        def _report_to_ui(event: str, payload: dict[str, Any]) -> None:
            nonlocal live_status_line
            if event == "download_start":
                summary["downloads_started"] = _coerce_int(summary.get("downloads_started")) + 1
                live_status_line = f"Descargando: {payload.get('name', 'sample')}..."
            elif event == "download_retry":
                wait_seconds = payload.get("wait_seconds", "?")
                live_status_line = f"Servidor saturado. Reintentando en {wait_seconds}s..."
            elif event == "download_ok":
                summary["downloads_ok"] = _coerce_int(summary.get("downloads_ok")) + 1
                live_status_line = "Descarga completada."
            elif event == "download_error":
                name = payload.get("name", "sample")
                error = payload.get("error", "unknown error")
                summary["downloads_fail"] = _coerce_int(summary.get("downloads_fail")) + 1
                live_status_line = f"Error descarga {name}: {error}"
            elif event == "blank_generated":
                live_status_line = "Imagen de control generada."
            elif event == "run_start":
                summary["images_total"] = _coerce_int(payload.get("total_images", 0))
                live_status_line = f"Iniciando PoC: {payload.get('model_tag', model_tag)}"
            elif event == "image_start":
                summary["images_started"] = _coerce_int(summary.get("images_started")) + 1
                live_status_line = f"Analizando {payload.get('image_name', 'imagen')}..."
            elif event == "image_result":
                count = payload.get("count", 0)
                detections = payload.get("detections") or []
                reasoning = payload.get("object_count_reasoning")
                summary["images_processed"] = _coerce_int(summary.get("images_processed")) + 1
                summary["images_ok"] = _coerce_int(summary.get("images_ok")) + 1
                summary["detected_subjects_total"] = _coerce_int(summary.get("detected_subjects_total")) + _coerce_int(count)
                live_status_line = f"Resultado OK: {count} subjects detectados"

                record: dict[str, object] = {
                    "image_name": str(payload.get("image_name") or "imagen"),
                    "status": "ok",
                    "payload": _format_bbox_payload(
                        count=_coerce_int(count),
                        reasoning=reasoning,
                        detections=list(detections),
                    ),
                }
                _append_recent_record(recent_records, record)
                all_records.append(record)
            elif event == "image_error":
                summary["images_processed"] = _coerce_int(summary.get("images_processed")) + 1
                summary["images_fail"] = _coerce_int(summary.get("images_fail")) + 1
                live_status_line = f"Error: {payload.get('error', 'unknown error')}"

                record: dict[str, object] = {
                    "image_name": str(payload.get("image_name") or "imagen"),
                    "status": "error",
                    "error": str(payload.get("error") or "unknown error"),
                }
                _append_recent_record(recent_records, record)
                all_records.append(record)
            elif event == "run_saved":
                summary["output_dir"] = str(payload.get("output_dir") or "N/A")
                live_status_line = "Resultados guardados."

            live_renderer.render()

        result_code = 1
        try:
            try:
                live_renderer.start()
                result_code = int(run_poc_bbox_main(["--model", model_tag], reporter=_report_to_ui))
            finally:
                live_renderer.stop()

            def _redraw_final_bbox_screen(ui_width: int) -> None:
                _render_final_sections_screen(
                    kit,
                    app,
                    subtitle="BBOX POC · VISUAL GROUNDING",
                    intro=_standard_final_intro(),
                    ui_width=ui_width,
                    sections=[
                        (
                            "Resumen",
                            _build_summary_lines(
                                kit,
                                [
                                    ("Modelo", str(summary.get("model_tag") or model_tag), "OK"),
                                    (
                                        "Imágenes procesadas",
                                        f"{summary.get('images_processed', 0)} de {summary.get('images_total', summary.get('images_started', 0))}",
                                        "OK",
                                    ),
                                    ("Resultados OK", str(summary.get("images_ok", 0)), "OK"),
                                    (
                                        "Errores",
                                        str(summary.get("images_fail", 0)),
                                        "OK" if _coerce_int(summary.get("images_fail")) == 0 else "WARN",
                                    ),
                                    ("Subjects detectados", str(summary.get("detected_subjects_total", 0)), "OK"),
                                    (
                                        "Descargas",
                                        (
                                            f"OK={summary.get('downloads_ok', 0)} | "
                                            f"FAIL={summary.get('downloads_fail', 0)}"
                                        ),
                                        "OK" if _coerce_int(summary.get("downloads_fail")) == 0 else "WARN",
                                    ),
                                    ("Salida", str(summary.get("output_dir") or "N/D"), "OK"),
                                ],
                                ui_width=ui_width,
                            ),
                        ),
                        (
                            "Actividad reciente",
                            _build_recent_record_lines(
                                kit,
                                all_records,
                                empty_message="  Sin registros.",
                                truncate=True,
                                ui_width=ui_width,
                            ),
                        ),
                    ],
                )

            _redraw_final_bbox_screen(kit.width())
            final_screen_renderer = _redraw_final_bbox_screen
        except Exception as error:
            kit.log(f"BBox PoC crashed: {error}", "error")

        if result_code == 0:
            kit.log("BBox PoC completed successfully.", "success")
        else:
            kit.log("BBox PoC finished with errors. Check output above.", "warning")
        if final_screen_renderer is None:
            kit.wait("Press any key to return to model selector...")
        else:
            kit.render_and_wait_responsive(
                render_fn=final_screen_renderer,
                message="Press any key to return to model selector...",
                initial_render=False,
            )
