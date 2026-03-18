from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, cast

from .test_dashboards_ui import (
    _append_recent_record,
    _build_partial_metrics_line,
    _build_recent_record_lines,
    _build_summary_lines,
    _coverage_fragments,
    _coerce_int,
    _format_metric_value,
    _make_live_panel,
    _make_recent_records,
    _rel_probe_path,
    _render_final_sections_screen,
    _render_live_dashboard,
    _standard_final_intro,
    _telemetry_available,
)
from .shared import ReactiveTerminalRenderer, build_live_status_line

if TYPE_CHECKING:
    from ..menu_kit import AppContext, UIKit


def run_telemetry_probe_wrapper(
    kit: "UIKit",
    app: "AppContext",
    *,
    select_model: Callable[[str, str], str | None],
    select_schema_variant: Callable[[str, str], tuple[str, Any] | None],
) -> None:
    """
    Ejecuta el flujo de prueba de telemetría con selección interactiva de modelo y esquema.

    Args:
        kit: Toolkit de interfaz de usuario de terminal.
        app: Contexto de la aplicación utilizado para renderizar la UI compartida.
        select_model: Callback que resuelve la etiqueta del modelo.
        select_schema_variant: Callback que resuelve la clase y variante del esquema.
    """
    from src.scripts.test_schema import find_images
    from src.scripts.test_telemetry import run_telemetry_batch

    while True:
        final_screen_renderer: Callable[[int], None] | None = None
        model_tag = select_model(
            "telemetry_probe_model_selector",
            "TELEMETRY PROBE · SELECT MODEL",
        )
        if model_tag is None:
            return

        schema_variant = select_schema_variant("telemetry_probe", "TELEMETRY PROBE")
        if schema_variant is None:
            continue
        schema_name, schema_cls = schema_variant

        images = find_images()
        if not images:
            kit.log("No se encontraron imágenes en los directorios del proyecto.", "warning")
            kit.wait("Press any key to return to model selector...")
            continue

        kit.clear()
        app.print_banner()
        kit.subtitle(f"TELEMETRY PROBE · {schema_name}")
        kit.log(
            f"Midiendo telemetría sobre hasta 5 imágenes con el modelo {model_tag}...",
            "step",
        )
        try:
            recent_records = _make_recent_records(limit=2)
            panel = _make_live_panel(
                kit,
                app,
                subtitle=f"TELEMETRY PROBE · {schema_name}",
                intro=f"Midiendo telemetría sobre hasta 5 imágenes con el modelo {model_tag}...",
            )
            live_summary: dict[str, object] = {
                "ok": 0,
                "fail": 0,
                "telemetry_availability": {},
            }
            live_total = min(5, len(images))
            live_status_line = f"Estado: preparando ejecución · muestra {live_total}"

            def _render_live_telemetry_dashboard(*, force_full: bool | None = None) -> None:
                """Renderiza el dashboard de telemetría con estado incremental."""
                completed_live = _coerce_int(live_summary.get("ok")) + _coerce_int(live_summary.get("fail"))
                availability_live = cast(dict[str, object], live_summary.get("telemetry_availability") or {})
                coverage_live = " · ".join(_coverage_fragments(availability_live))
                _render_live_dashboard(
                    kit,
                    panel,
                    current=completed_live,
                    total=live_total,
                    stats_line=(
                        f"Estadísticas: OK={live_summary.get('ok', 0)} | "
                        f"Errores={live_summary.get('fail', 0)}"
                    ),
                    status_line=live_status_line,
                    metrics_line=_build_partial_metrics_line(live_summary),
                    coverage_line=f"Cobertura parcial: {coverage_live}" if coverage_live else None,
                    recent_title="Últimas inferencias:",
                    recent_records=list(recent_records),
                    force_full=force_full,
                )

            live_renderer = ReactiveTerminalRenderer(
                kit=kit,
                render_fn=_render_live_telemetry_dashboard,
            )

            def on_probe_progress(payload: dict[str, object]) -> None:
                """
                Refresca el panel para cada evento de inferencia.

                Args:
                    payload: Evento de progreso emitido por el probe de telemetría.
                """
                nonlocal live_total, live_status_line
                summary = cast(dict[str, object], payload.get("summary") or {})
                live_summary.clear()
                live_summary.update(summary)
                completed = _coerce_int(summary.get("ok")) + _coerce_int(summary.get("fail"))
                total = _coerce_int(payload.get("total"))
                live_total = total
                event = str(payload.get("event") or "")
                image_path = _rel_probe_path(cast(str | None, payload.get("image_path")))
                status = str(payload.get("status") or "")
                current_index = _coerce_int(payload.get("index"))
                record_payload = payload.get("record")
                last_record = record_payload if isinstance(record_payload, dict) else None

                live_status_line = build_live_status_line(
                    event=event,
                    current_index=current_index,
                    total=total,
                    item_label=image_path,
                    status=status,
                    completed=completed,
                    on_start="Estado: procesando {current_index}/{total} · {item}",
                    on_done_ok="Estado: completada {current_index}/{total} · {item}",
                    on_done_default="Estado: error {current_index}/{total} · {item}",
                    on_complete="Estado: finalizado · {completed}/{total} imágenes procesadas",
                    on_prepare="Estado: preparando ejecución · muestra {total}",
                )

                if event == "image_done" and last_record is not None:
                    _append_recent_record(recent_records, last_record)

                live_renderer.render()

            live_renderer.start()
            try:
                summary = run_telemetry_batch(
                    model_id=model_tag,
                    schema_name=schema_name,
                    schema_cls=schema_cls,
                    images=images,
                    max_images=5,
                    on_progress=on_probe_progress,
                )
            finally:
                live_renderer.stop()
            availability = cast(dict[str, object], summary.get("telemetry_availability") or {})
            rows = [
                ("Modelo", summary["model_id"], "OK"),
                ("Esquema", summary["schema_name"], "OK"),
                ("Muestra", str(summary["sample_size"]), "OK"),
                ("Inferencias OK", str(summary["ok"]), "OK" if summary["ok"] > 0 else "WARN"),
                ("Errores", str(summary["fail"]), "OK" if summary["fail"] == 0 else "WARN"),
                (
                    "Modelo resuelto",
                    str(summary["static_model_info"]["resolved_model_id"]),
                    "OK" if summary["static_model_info"]["resolved_model_id"] else "WARN",
                ),
                (
                    "Arquitectura",
                    str(summary["static_model_info"]["architecture"]),
                    "OK" if summary["static_model_info"]["architecture"] else "WARN",
                ),
                (
                    "Stop reason",
                    str(summary["static_model_info"]["stop_reason"]),
                    "OK" if summary["static_model_info"]["stop_reason"] else "WARN",
                ),
                (
                    "TTFT medio (s)",
                    _format_metric_value(summary["ttft"]["avg"]),
                    "OK" if summary["ttft"]["avg"] is not None else "WARN",
                ),
                (
                    "TPS medio",
                    _format_metric_value(summary["tps"]["avg"]),
                    "OK" if summary["tps"]["avg"] is not None else "WARN",
                ),
                (
                    "Generacion media (s)",
                    _format_metric_value(summary["generation_duration"]["avg"]),
                    "OK" if summary["generation_duration"]["avg"] is not None else "WARN",
                ),
                (
                    "Prompt tokens medios",
                    _format_metric_value(summary["prompt_tokens"]["avg"]),
                    "OK" if summary["prompt_tokens"]["avg"] is not None else "WARN",
                ),
                (
                    "Output tokens medios",
                    _format_metric_value(summary["completion_tokens"]["avg"]),
                    "OK" if summary["completion_tokens"]["avg"] is not None else "WARN",
                ),
                (
                    "Total tokens medios",
                    _format_metric_value(summary["total_tokens"]["avg"]),
                    "OK" if summary["total_tokens"]["avg"] is not None else "WARN",
                ),
                (
                    "Latencia media (s)",
                    _format_metric_value(summary["total_duration"]["avg"]),
                    "OK" if summary["total_duration"]["avg"] is not None else "WARN",
                ),
            ]
            if _telemetry_available(availability, "reasoning_records"):
                rows.append(
                    (
                        "Reasoning schema medio",
                        _format_metric_value(summary["reasoning_tokens"]["avg"]),
                        "OK" if summary["reasoning_tokens"]["avg"] is not None else "WARN",
                    )
                )
            if _telemetry_available(availability, "gpu_layer_records"):
                rows.append(
                    (
                        "Capas GPU medias",
                        _format_metric_value(summary["gpu_layers"]["avg"]),
                        "OK" if summary["gpu_layers"]["avg"] is not None else "WARN",
                    )
                )
            ttft_note = summary.get("notes", {}).get("ttft")
            tps_note = summary.get("notes", {}).get("tps")
            coverage_fragments = _coverage_fragments(availability)

            detail_lines = _build_recent_record_lines(
                kit,
                cast(list[dict[str, object]], summary["records"]),
                truncate=False,
                ui_width=kit.width(),
            )

            note_lines = []
            if ttft_note:
                kit.log(str(ttft_note), "warning")
                note_lines.append(f"  TTFT: {ttft_note}")
            if tps_note:
                kit.log(str(tps_note), "warning")
                note_lines.append(f"  TPS: {tps_note}")
            if coverage_fragments:
                kit.log(f"Cobertura: {' · '.join(coverage_fragments)}", "step")
                note_lines.append(f"  Cobertura: {' · '.join(coverage_fragments)}")
            def _redraw_final_telemetry_screen(_ui_width: int) -> None:
                """Re-renderiza la pantalla final de telemetría tras cambio de ancho."""
                sections = [
                    (
                        "Resumen",
                        _build_summary_lines(
                            kit,
                            [(str(a), str(b), str(c)) for a, b, c in rows],
                            ui_width=_ui_width,
                        ),
                    ),
                    ("Observaciones", note_lines or ["  Sin observaciones adicionales."]),
                    ("Actividad reciente", _build_recent_record_lines(kit, cast(list[dict[str, object]], summary["records"]), truncate=False, ui_width=_ui_width)),
                ]
                _render_final_sections_screen(
                    kit,
                    app,
                    subtitle=f"TELEMETRY PROBE · {schema_name}",
                    intro=_standard_final_intro(),
                    ui_width=_ui_width,
                    sections=sections,
                )

            _redraw_final_telemetry_screen(kit.width())
            final_screen_renderer = _redraw_final_telemetry_screen
            if summary["fail"] == 0:
                kit.log(
                    f"Telemetry Probe completado: {summary['ok']} inferencias OK.",
                    "success",
                )
            else:
                kit.log(
                    f"Telemetry Probe completado con incidencias: {summary['ok']} OK, {summary['fail']} errores.",
                    "warning",
                )
        except Exception as error:
            kit.log(f"Telemetry Probe terminó con error: {error}", "error")
        if final_screen_renderer is None:
            kit.wait("Press any key to return to model selector...")
        else:
            kit.render_and_wait_responsive(
                render_fn=final_screen_renderer,
                message="Press any key to return to model selector...",
                initial_render=False,
            )
