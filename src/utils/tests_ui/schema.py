from __future__ import annotations

import shutil
from typing import TYPE_CHECKING, Any, Callable, cast

from .test_dashboards_ui import (
    _append_recent_record,
    _build_recent_record_lines,
    _build_summary_lines,
    _coerce_int,
    _format_schema_summary_rows,
    _make_live_panel,
    _make_recent_records,
    _rel_probe_path,
    _render_final_sections_screen,
    _render_live_dashboard,
    _standard_final_intro,
)
from .shared import ReactiveTerminalRenderer, build_live_status_line

if TYPE_CHECKING:
    from ..menu_kit import AppContext, UIKit


def run_schema_tester_wrapper(
    kit: "UIKit",
    app: "AppContext",
    *,
    select_model: Callable[[str, str], str | None],
    select_schema_variant: Callable[[str, str], tuple[str, Any] | None],
) -> None:
    """
    Ejecuta el flujo de validación de esquemas sobre imágenes muestreadas.

    Args:
        kit: Toolkit de interfaz de usuario de terminal.
        app: Contexto de la aplicación utilizado para renderizar el cromo de UI compartido.
        select_model: Callback que resuelve la etiqueta del modelo.
        select_schema_variant: Callback que resuelve la clase y variante del esquema.
    """
    from src.scripts.test_schema import find_images, format_schema_info, run_batch

    while True:
        final_screen_renderer: Callable[[int], None] | None = None
        model_tag = select_model(
            "schema_tester_model_selector",
            "SCHEMA TESTER · SELECT MODEL",
        )
        if model_tag is None:
            return

        schema_variant = select_schema_variant("schema_tester", "SCHEMA TESTER")
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
        kit.subtitle(f"SCHEMA TESTER · {schema_name}")
        print()
        tw = max(60, shutil.get_terminal_size(fallback=(120, 30)).columns - 4)
        schema_info_lines = format_schema_info(schema_name, schema_cls, text_width=tw).splitlines()
        for line in schema_info_lines:
            print(f"  {line}")
        print()
        kit.log(f"Modelo: {model_tag} · max 5 imágenes de {len(images)} disponibles...", "step")
        try:
            recent_records = _make_recent_records(limit=2)
            all_records: list[dict[str, object]] = []
            panel = _make_live_panel(
                kit,
                app,
                subtitle=f"SCHEMA TESTER · {schema_name}",
                intro=f"Modelo: {model_tag} · validando hasta 5 imágenes de {len(images)} disponibles...",
                static_lines=[f"  {line}" for line in schema_info_lines],
            )
            live_summary: dict[str, object] = {
                "ok": 0,
                "invalid": 0,
                "fail": 0,
            }
            live_total = min(5, len(images))
            live_status_line = f"Estado: preparando ejecución · muestra {live_total}"

            def _render_live_schema_dashboard(*, force_full: bool | None = None) -> None:
                """Renderiza el dashboard de schema con el estado actual."""
                completed_live = (
                    _coerce_int(live_summary.get("ok"))
                    + _coerce_int(live_summary.get("invalid"))
                    + _coerce_int(live_summary.get("fail"))
                )
                _render_live_dashboard(
                    kit,
                    panel,
                    current=completed_live,
                    total=live_total,
                    stats_line=(
                        f"Estadísticas: OK={live_summary.get('ok', 0)} | "
                        f"Inválidas={live_summary.get('invalid', 0)} | Errores={live_summary.get('fail', 0)}"
                    ),
                    status_line=live_status_line,
                    metrics_line=None,
                    recent_title="Últimas validaciones:",
                    recent_records=list(recent_records),
                    force_full=force_full,
                )

            live_renderer = ReactiveTerminalRenderer(
                kit=kit,
                render_fn=_render_live_schema_dashboard,
            )

            def on_schema_progress(payload: dict[str, object]) -> None:
                """
                Actualiza el panel en vivo para eventos del schema tester.

                Args:
                    payload: Evento de progreso emitido por el runner de schema.
                """
                nonlocal live_total, live_status_line
                summary = cast(dict[str, object], payload.get("summary") or {})
                live_summary.clear()
                live_summary.update(summary)
                total = _coerce_int(payload.get("total"))
                live_total = total
                event = str(payload.get("event") or "")
                current_index = _coerce_int(payload.get("index"))
                image_path = _rel_probe_path(cast(str | None, payload.get("image_path")))
                record_payload = payload.get("record")
                last_record = record_payload if isinstance(record_payload, dict) else None
                completed = (
                    _coerce_int(summary.get("ok"))
                    + _coerce_int(summary.get("invalid"))
                    + _coerce_int(summary.get("fail"))
                )

                live_status_line = build_live_status_line(
                    event=event,
                    current_index=current_index,
                    total=total,
                    item_label=image_path,
                    status=str(payload.get("status") or ""),
                    completed=completed,
                    on_start="Estado: procesando {current_index}/{total} · {item}",
                    on_done_ok="Estado: validada {current_index}/{total} · {item}",
                    on_done_invalid="Estado: JSON inválido {current_index}/{total} · {item}",
                    on_complete="Estado: finalizado · {completed}/{total} imágenes revisadas",
                    on_prepare="Estado: preparando ejecución · muestra {total}",
                )

                if event == "image_done" and last_record is not None:
                    _append_recent_record(recent_records, last_record)
                    all_records.append(dict(last_record))

                live_renderer.render()

            try:
                live_renderer.start()
                ok, fail, invalid = run_batch(
                    model_tag,
                    schema_name,
                    schema_cls,
                    images,
                    on_progress=on_schema_progress,
                )
            except TypeError as error:
                if "on_progress" not in str(error):
                    raise
                ok, fail, invalid = run_batch(model_tag, schema_name, schema_cls, images)
            finally:
                live_renderer.stop()

            summary_rows = _format_schema_summary_rows(
                {
                    "model_id": model_tag,
                    "schema_name": schema_name,
                    "sample_size": min(5, len(images)),
                    "total_available": len(images),
                    "ok": ok,
                    "invalid": invalid,
                    "fail": fail,
                }
            )
            def _redraw_final_schema_screen(_ui_width: int) -> None:
                """Re-renderiza la pantalla final del schema tester tras cambio de ancho."""
                _render_final_sections_screen(
                    kit,
                    app,
                    subtitle=f"SCHEMA TESTER · {schema_name}",
                    intro=_standard_final_intro(),
                    ui_width=_ui_width,
                    sections=[
                        ("Schema utilizado", [f"  {line}" for line in schema_info_lines]),
                        ("Resumen", _build_summary_lines(kit, summary_rows, ui_width=_ui_width)),
                        (
                            "Actividad reciente",
                            _build_recent_record_lines(
                                kit,
                                all_records,
                                empty_message="  Sin validaciones recientes.",
                                truncate=False,
                                ui_width=_ui_width,
                            ),
                        ),
                    ],
                )

            _redraw_final_schema_screen(kit.width())
            final_screen_renderer = _redraw_final_schema_screen
            if fail > 0 or invalid > 0:
                kit.log(
                    f"Schema Tester completado: {ok} válidas, {invalid} inválidas, {fail} errores.",
                    "warning",
                )
            else:
                kit.log(f"Schema Tester completado: {ok}/{ok} inferencias válidas.", "success")
        except Exception as error:
            kit.log(f"Schema Tester terminó con error: {error}", "error")
        if final_screen_renderer is None:
            kit.wait("Press any key to return to model selector...")
        else:
            kit.render_and_wait_responsive(
                render_fn=final_screen_renderer,
                message="Press any key to return to model selector...",
                initial_render=False,
            )
