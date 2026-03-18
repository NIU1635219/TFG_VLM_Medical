from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Any, cast

from .test_dashboards_ui import (
    _append_recent_record,
    _build_recent_record_lines,
    _build_summary_lines,
    _coerce_int,
    _format_metric_value,
    _make_live_panel,
    _make_recent_records,
    _render_final_sections_screen,
    _render_live_dashboard,
    _standard_final_intro,
)
from .shared import ReactiveTerminalRenderer, build_live_status_line

if TYPE_CHECKING:
    from ..menu_kit import AppContext, UIKit


def run_smoke_test_wrapper(
    kit: "UIKit",
    app: "AppContext",
    *,
    select_model: Callable[[str, str], str | None],
) -> None:
    """
    Ejecuta pruebas de inferencia interactivas (smoke tests) para un modelo seleccionado.

    Args:
        kit: Toolkit de interfaz de usuario de terminal.
        app: Contexto de la aplicación utilizado para renderizar la UI compartida.
        select_model: Callback que resuelve la etiqueta del modelo a ejecutar.
    """
    while True:
        final_screen_renderer: Callable[[int], None] | None = None
        model_tag = select_model(
            "run_smoke_model_selector",
            "SELECT INFERENCE MODEL",
        )
        if model_tag is None:
            return

        try:
            from src.scripts.test_inference import ensure_test_images, run_smoke_test, main as smoke_test_main
        except Exception as error:
            kit.log(f"Could not import smoke test script: {error}", "error")
            kit.log("Smoke test failed (Exit Code 1). Check output above.", "error")
            kit.wait("Press any key to return to model selector...")
            continue

        recent_records = _make_recent_records(limit=2)
        all_smoke_records: list[dict[str, object]] = []
        panel = _make_live_panel(
            kit,
            app,
            subtitle="SMOKE TEST · INFERENCE DEMO",
            intro=f"Ejecutando smoke test con {model_tag}...",
        )
        smoke_summary: dict[str, object] = {
            "model_tag": model_tag,
            "total_cases": 0,
            "processed": 0,
            "passed": 0,
            "failed": 0,
            "preload_seconds": None,
        }
        live_status_line = "Estado: preparando ejecución · muestra 0"

        def _render_live_smoke_dashboard(*, force_full: bool | None = None) -> None:
            """Renderiza el dashboard vivo usando el ultimo estado conocido."""
            _render_live_dashboard(
                kit,
                panel,
                current=_coerce_int(smoke_summary.get("processed")),
                total=_coerce_int(smoke_summary.get("total_cases")),
                stats_line=(
                    f"Estadísticas: OK={smoke_summary.get('passed', 0)} | "
                    f"Inválidos={smoke_summary.get('failed', 0)}"
                ),
                status_line=live_status_line,
                metrics_line=(
                    f"Modelo precargado: {_format_metric_value(smoke_summary.get('preload_seconds'), suffix=' s')}"
                    if smoke_summary.get("preload_seconds") is not None
                    else None
                ),
                recent_title="Últimos casos:",
                recent_records=list(recent_records),
                force_full=force_full,
            )

        live_renderer = ReactiveTerminalRenderer(
            kit=kit,
            render_fn=_render_live_smoke_dashboard,
        )

        def on_smoke_progress(payload: dict[str, object]) -> None:
            """
            Actualiza el dashboard en vivo durante el smoke test.

            Args:
                payload: Evento de progreso emitido por `run_smoke_test`.
            """
            nonlocal live_status_line
            summary = cast(dict[str, object], payload.get("summary") or {})
            smoke_summary.update(summary)
            total = _coerce_int(payload.get("total"))
            event = str(payload.get("event") or "")
            current_index = _coerce_int(payload.get("index"))
            case = cast(dict[str, object], payload.get("case") or {})
            record_payload = payload.get("record")
            record = record_payload if isinstance(record_payload, dict) else None
            processed = _coerce_int(summary.get("processed"))

            if event == "model_ready":
                live_status_line = (
                    "Estado: modelo precargado en "
                    f"{_format_metric_value(summary.get('preload_seconds'), suffix=' s')}"
                )
            else:
                live_status_line = build_live_status_line(
                    event=event,
                    current_index=current_index,
                    total=total,
                    item_label=str(case.get("id", "caso")),
                    status=str(payload.get("status") or ""),
                    completed=processed,
                    on_start="Estado: procesando {current_index}/{total} · {item}",
                    on_done_default="Estado: completado {current_index}/{total} · {item}",
                    on_complete="Estado: finalizado · {completed}/{total} casos revisados",
                    on_prepare="Estado: preparando ejecución · muestra {total}",
                    start_event="case_start",
                    done_event="case_done",
                )

            if event == "case_done" and record is not None:
                record_dict = cast(dict[str, object], record)
                payload_data = record_dict.get("payload")
                if payload_data is None and record_dict.get("response_preview") is not None:
                    payload_data = {"preview": record_dict.get("response_preview")}
                recent_record = {
                    "image_name": str(record_dict.get("case_id") or record_dict.get("label") or "caso"),
                    "status": record_dict.get("status") or "unknown",
                    "payload": payload_data,
                    "validation_error": record_dict.get("message"),
                    "error": record_dict.get("error"),
                }
                _append_recent_record(recent_records, recent_record)
                all_smoke_records.append(dict(recent_record))

            live_renderer.render()

        kit.clear()
        result_code = 1
        try:
            try:
                live_renderer.start()
                result_code = run_smoke_test(
                    model_tag,
                    ensure_test_images(),
                    on_progress=on_smoke_progress,
                )
            except TypeError as error:
                if "on_progress" in str(error):
                    raise
                result_code = int(smoke_test_main(model_path=model_tag))
            finally:
                live_renderer.stop()

            def _redraw_final_smoke_screen(_ui_width: int) -> None:
                """Re-renderiza la pantalla final de smoke test tras cambio de ancho."""
                _render_final_sections_screen(
                    kit,
                    app,
                    subtitle="SMOKE TEST · INFERENCE DEMO",
                    intro=_standard_final_intro(),
                    ui_width=_ui_width,
                    sections=[
                        (
                            "Resumen",
                            _build_summary_lines(
                                kit,
                                [
                                    ("Modelo", str(smoke_summary.get("model_tag") or model_tag), "OK"),
                                    (
                                        "Casos procesados",
                                        f"{smoke_summary.get('processed', 0)} de {smoke_summary.get('total_cases', 0)}",
                                        "OK",
                                    ),
                                    ("Válidos", str(smoke_summary.get("passed", 0)), "OK"),
                                    (
                                        "Incidencias",
                                        str(smoke_summary.get("failed", 0)),
                                        "OK" if _coerce_int(smoke_summary.get("failed")) == 0 else "WARN",
                                    ),
                                    (
                                        "Precarga (s)",
                                        _format_metric_value(smoke_summary.get("preload_seconds"), suffix=" s"),
                                        "OK",
                                    ),
                                ],
                                ui_width=_ui_width,
                            ),
                        ),
                        (
                            "Actividad reciente",
                            _build_recent_record_lines(
                                kit,
                                all_smoke_records,
                                empty_message="  Sin registros.",
                                truncate=False,
                                ui_width=_ui_width,
                            ),
                        ),
                    ],
                )

            _redraw_final_smoke_screen(kit.width())
            final_screen_renderer = _redraw_final_smoke_screen
        except Exception as error:
            kit.log(f"Smoke test crashed: {error}", "error")
        if result_code != 0:
            kit.log("Smoke test failed (Exit Code 1). Check output above.", "error")
        if final_screen_renderer is None:
            kit.wait("Press any key to return to model selector...")
        else:
            kit.render_and_wait_responsive(
                render_fn=final_screen_renderer,
                message="Press any key to return to model selector...",
                initial_render=False,
            )
