"""UI de tests y selección de modelos para setup_env."""

from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING, Any, cast

from .test_dashboards_ui import (
    _append_recent_record,
    _build_recent_record_lines,
    _build_named_value_lines,
    _build_probe_detail_parts,
    _build_summary_lines,
    _coerce_int,
    _coverage_fragments,
    _format_batch_summary_rows,
    _format_metric_value,
    _format_schema_summary_rows,
    _make_live_panel,
    _make_recent_lines,
    _rel_probe_path,
    _render_final_sections_screen,
    _render_live_dashboard,
    _standard_final_intro,
    _telemetry_available,
)

if TYPE_CHECKING:
    from .menu_kit import AppContext, UIKit


# ---------------------------------------------------------------------------
# Utilidades de sistema puras
# ---------------------------------------------------------------------------


def list_test_files() -> list[str]:
    """Escanea la carpeta ``tests/`` y devuelve los archivos de prueba.

    Returns:
        list[str]: Lista ordenada de nombres de archivo que empiezan por
        ``test_`` y terminan en ``.py``.  Lista vacía si la carpeta no existe.
    """
    test_dir = "tests"
    if not os.path.exists(test_dir):
        return []
    files = [
        f for f in os.listdir(test_dir)
        if f.startswith("test_") and f.endswith(".py")
    ]
    return sorted(files)


# ---------------------------------------------------------------------------
# Menú de tests
# ---------------------------------------------------------------------------

def run_tests_menu(kit: "UIKit", app: "AppContext") -> None:
    """
    Despliega el submenú para ejecución de tests y gestión de pruebas.

    Ofrece opciones para:
    - Gestionar modelos (descarga/borrado).
    - Ejecutar todos los tests unitarios.
    - Ejecutar tests unitarios específicos.
    - Ejecutar 'Smoke Test' (prueba de inferencia end-to-end con modelo).
    - Ejecutar 'Schema Tester' (inferencia + validación por esquema Pydantic).

    Args:
        kit (UIKit): Interfaz de UI de terminal.
        app (AppContext): Contexto de dominio de la aplicación.
    """
    from . import setup_models_ui

    # ------------------------------------------------------------------
    # Helpers reutilizables
    # ------------------------------------------------------------------

    def _make_header(subtitle: str):
        """Devuelve una función de encabezado reutilizable."""
        def _hdr() -> None:
            app.print_banner()
            kit.subtitle(subtitle)
        return _hdr

    def _select_schema_reasoning_mode(menu_id: str, subtitle: str) -> bool | str:
        """Muestra selector del modo del schema: con o sin razonamiento."""
        items = [
            kit.MenuItem(
                "Con razonamiento",
                description="Añade el campo reasoning al JSON y obliga a justificar antes del veredicto.",
            ),
            kit.MenuItem(
                "Sin razonamiento",
                description="Usa el schema base sin campo reasoning para una salida estructurada más compacta.",
            ),
            kit.MenuItem(
                "← Volver al selector de schema",
                lambda: None,
                description="Cancela y vuelve a elegir el schema base.",
            ),
        ]
        sel = kit.menu(
            items,
            header_func=_make_header(subtitle),
            menu_id=menu_id,
            nav_hint_text="↑/↓ elegir modo · ENTER confirmar · ESC volver al selector de schema",
        )
        if not sel or "Volver" in sel.label:
            return "BACK"
        return "Con razonamiento" in sel.label

    def _select_schema_variant(menu_prefix: str, subtitle_prefix: str):
        """Resuelve modelo de schema base y variante reasoning mediante menús reutilizables."""
        from src.inference.schemas import SCHEMA_REGISTRY, get_schema_variant
        from src.scripts.test_schema import format_schema_menu_description

        while True:
            schema_items = [
                kit.MenuItem(name, description=format_schema_menu_description(name, cls))
                for name, cls in SCHEMA_REGISTRY.items()
            ]
            schema_items.append(
                kit.MenuItem(
                    "Volver al selector de modelos",
                    lambda: None,
                    description="Vuelve a la selección de modelo.",
                )
            )

            schema_sel = kit.menu(
                schema_items,
                header_func=_make_header(f"{subtitle_prefix} · SELECT SCHEMA"),
                menu_id=f"{menu_prefix}_schema_selector",
                nav_hint_text="↑/↓ elegir esquema · ENTER confirmar · ESC volver",
                description_slot_rows=15,
            )
            if not schema_sel or schema_sel.label.strip() == "Volver al selector de modelos":
                return None

            base_schema_name = schema_sel.label
            reasoning_mode = _select_schema_reasoning_mode(
                menu_id=f"{menu_prefix}_reasoning_selector",
                subtitle=f"{subtitle_prefix} · SELECT SCHEMA MODE",
            )
            if reasoning_mode == "BACK":
                continue

            return get_schema_variant(base_schema_name, bool(reasoning_mode))

    def _select_batch_size() -> int | None | str:
        """Selector del tamaño de muestra para ejecución batch desde la TUI."""
        items = [
            kit.MenuItem("5 imágenes", description="Smoke rápido para validar el pipeline y el archivo de salida."),
            kit.MenuItem("25 imágenes", description="Muestra intermedia para una primera recogida de resultados."),
            kit.MenuItem("100 imágenes", description="Ejecución amplia para experimentos más serios."),
            kit.MenuItem("Todas las imágenes", description="Procesa todo lo encontrado bajo data/ usando exportado incremental."),
            kit.MenuItem("Cancel", lambda: None, description="Vuelve al menú anterior."),
        ]
        sel = kit.menu(
            items,
            header_func=_make_header("BATCH RUNNER · SELECT SAMPLE SIZE"),
            menu_id="batch_runner_size_selector",
            nav_hint_text="↑/↓ elegir tamaño · ENTER confirmar · ESC volver",
        )
        if not sel or sel.label.strip() == "Cancel":
            return "BACK"
        if sel.label.startswith("5"):
            return 5
        if sel.label.startswith("25"):
            return 25
        if sel.label.startswith("100"):
            return 100
        return None

    def _select_response_inspector_mode() -> tuple[bool, bool] | None:
        """Permite elegir si inspeccionar una respuesta cruda o estructurada."""
        items = [
            kit.MenuItem(
                "Cruda (sin schema)",
                description="Envía una petición multimodal sin response_format para ver la respuesta nativa del SDK.",
            ),
            kit.MenuItem(
                "Estructurada con reasoning",
                description="Usa GenericObjectDetectionWithReasoning para inspeccionar parsed, stats y schema activo.",
            ),
            kit.MenuItem("Cancel", lambda: None, description="Vuelve al selector de modelo."),
        ]
        sel = kit.menu(
            items,
            header_func=_make_header("RESPONSE INSPECTOR · SELECT MODE"),
            menu_id="response_inspector_mode_selector",
            nav_hint_text="↑/↓ elegir modo · ENTER confirmar · ESC volver",
        )
        if not sel or sel.label.strip() == "Cancel":
            return None
        if sel.label.startswith("Cruda"):
            return False, False
        return True, True

    def _select_model(menu_id: str, subtitle: str) -> str | None:
        """
        Muestra un selector de modelo LM Studio.

        Returns:
            Etiqueta del modelo seleccionado, o ``None`` si se cancela.
        """
        installed = app.get_installed_lms_models()
        if not installed:
            kit.log(
                "No hay modelos disponibles en LM Studio. "
                "Carga uno desde 'Manage/Pull LM Studio Models...'.",
                "warning",
            )
            kit.wait("Press any key to return to tests menu...")
            return None

        opts = [
            kit.MenuItem(tag, description="Usa este modelo para la inferencia.")
            for tag in installed
        ]
        opts.append(kit.MenuItem("Cancel", lambda: None, description="Vuelve al menú anterior."))

        sel = kit.menu(
            opts,
            header_func=_make_header(subtitle),
            menu_id=menu_id,
            nav_hint_text="↑/↓ elegir modelo · ENTER confirmar · ESC volver",
        )
        if not sel or sel.label.strip() == "Cancel":
            return None
        tag = sel.label.strip()
        if not tag:
            kit.log("No model selected.", "warning")
            kit.wait("Press any key to return to tests menu...")
            return None
        return tag

    # ------------------------------------------------------------------
    # Acciones de menú
    # ------------------------------------------------------------------

    def run_smoke_test_in_process(model_tag: str) -> int:
        """Ejecuta smoke test sin subprocess."""
        try:
            from src.scripts.test_inference import ensure_test_images, main as smoke_test_main, run_smoke_test
        except Exception as error:
            kit.log(f"Could not import smoke test script: {error}", "error")
            return 1
        try:
            try:
                return int(run_smoke_test(model_tag, ensure_test_images()))
            except TypeError as error:
                if "on_progress" in str(error):
                    raise
                return int(smoke_test_main(model_path=model_tag))
        except Exception as error:
            kit.log(f"Smoke test crashed: {error}", "error")
            return 1

    def run_all_unit_tests() -> None:
        """Ejecuta todos los tests unitarios."""
        kit.log("Running All Unit Tests...", "step")
        kit.run_cmd("uv run python -m pytest tests/")
        kit.wait()

    def run_specific_test() -> None:
        """Ejecuta un test específico."""
        while True:
            files = app.list_test_files()
            if not files:
                kit.log("No tests found in tests/ folder.", "warning")
                app.time_module.sleep(1)
                return

            test_opts = [
                kit.MenuItem(f, description="Ejecuta solo este archivo de tests con pytest.")
                for f in files
            ]
            test_opts.append(
                kit.MenuItem("Cancel", lambda: None, description="Vuelve al menú anterior sin ejecutar pruebas.")
            )

            selection = kit.menu(
                test_opts,
                header_func=_make_header("SELECT TEST FILE"),
                menu_id="run_specific_test_selector",
                nav_hint_text="↑/↓ navegar archivos · ENTER ejecutar test · ESC volver",
            )

            if not selection or selection.label.strip() == "Cancel":
                return

            fname = selection.label
            kit.clear()
            kit.log(f"Running {fname}...", "step")
            kit.run_cmd(f"uv run python -m pytest tests/{fname}")
            kit.wait("Finished. Press any key to return to test selector...")

    def run_schema_tester_wrapper() -> None:
        """Schema Tester: modelo → schema base → modo reasoning → inferencia."""
        from src.scripts.test_schema import (
            find_images,
            format_schema_info,
            run_batch,
        )
        while True:
            # ─ PASO 1: Seleccionar modelo ─────────────────────
            model_tag = _select_model(
                menu_id="schema_tester_model_selector",
                subtitle="SCHEMA TESTER · SELECT MODEL",
            )
            if model_tag is None:
                return

            schema_variant = _select_schema_variant("schema_tester", "SCHEMA TESTER")
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
            _tw = max(60, shutil.get_terminal_size(fallback=(120, 30)).columns - 4)
            schema_info_lines = format_schema_info(schema_name, schema_cls, text_width=_tw).splitlines()
            for line in schema_info_lines:
                print(f"  {line}")
            print()
            kit.log(
                f"Modelo: {model_tag} · max 5 imágenes de {len(images)} disponibles...",
                "step",
            )
            try:
                recent_lines = _make_recent_lines()
                recent_records: list[dict[str, object]] = []
                panel = _make_live_panel(
                    kit,
                    app,
                    subtitle=f"SCHEMA TESTER · {schema_name}",
                    intro=f"Modelo: {model_tag} · validando hasta 5 imágenes de {len(images)} disponibles...",
                    static_lines=[f"  {line}" for line in schema_info_lines],
                )

                def on_schema_progress(payload: dict[str, object]) -> None:
                    summary = cast(dict[str, object], payload.get("summary") or {})
                    total = _coerce_int(payload.get("total"))
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

                    if event == "image_start":
                        status_line = f"Estado: procesando {current_index}/{total} · {image_path}"
                    elif event == "image_done" and str(payload.get("status") or "") == "ok":
                        status_line = f"Estado: validada {current_index}/{total} · {image_path}"
                    elif event == "image_done" and str(payload.get("status") or "") == "invalid":
                        status_line = f"Estado: JSON inválido {current_index}/{total} · {image_path}"
                    elif event == "complete":
                        status_line = f"Estado: finalizado · {completed}/{total} imágenes revisadas"
                    else:
                        status_line = f"Estado: preparando ejecución · muestra {total}"

                    if event == "image_done" and last_record is not None:
                        _append_recent_record(kit, recent_lines, last_record)
                        recent_records.append(dict(last_record))
                        if len(recent_records) > 5:
                            recent_records.pop(0)

                    _render_live_dashboard(
                        kit,
                        panel,
                        current=completed,
                        total=total,
                        stats_line=(
                            f"Estadísticas: OK={summary.get('ok', 0)} | "
                            f"Inválidas={summary.get('invalid', 0)} | Errores={summary.get('fail', 0)}"
                        ),
                        status_line=status_line,
                        metrics_line=None,
                        recent_title="Últimas validaciones:",
                        recent_lines=list(recent_lines),
                    )

                try:
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
                _render_final_sections_screen(
                    kit,
                    app,
                    subtitle=f"SCHEMA TESTER · {schema_name}",
                    intro=_standard_final_intro(),
                    sections=[
                        ("Schema utilizado", [f"  {line}" for line in schema_info_lines]),
                        ("Resumen", _build_summary_lines(kit, summary_rows)),
                        (
                            "Actividad reciente",
                            _build_recent_record_lines(
                                kit,
                                recent_records,
                                empty_message="  Sin validaciones recientes.",
                            ),
                        ),
                    ],
                )
                if fail > 0 or invalid > 0:
                    kit.log(
                        f"Schema Tester completado: {ok} válidas, "
                        f"{invalid} inválidas, {fail} errores.",
                        "warning",
                    )
                else:
                    kit.log(
                        f"Schema Tester completado: {ok}/{ok} inferencias válidas.",
                        "success",
                    )
            except Exception as error:
                kit.log(f"Schema Tester terminó con error: {error}", "error")
            kit.wait("Press any key to return to model selector...")

    def run_telemetry_probe_wrapper() -> None:
        """Ejecuta una prueba de telemetría TTFT/TPS con selección interactiva."""
        from src.scripts.test_schema import find_images
        from src.scripts.test_telemetry import run_telemetry_batch

        while True:
            model_tag = _select_model(
                menu_id="telemetry_probe_model_selector",
                subtitle="TELEMETRY PROBE · SELECT MODEL",
            )
            if model_tag is None:
                return

            schema_variant = _select_schema_variant("telemetry_probe", "TELEMETRY PROBE")
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
                recent_lines = _make_recent_lines()
                panel = _make_live_panel(
                    kit,
                    app,
                    subtitle=f"TELEMETRY PROBE · {schema_name}",
                    intro=f"Midiendo telemetría sobre hasta 5 imágenes con el modelo {model_tag}...",
                )

                def on_probe_progress(payload: dict[str, object]) -> None:
                    summary = cast(dict[str, object], payload.get("summary") or {})
                    completed = _coerce_int(summary.get("ok")) + _coerce_int(summary.get("fail"))
                    total = _coerce_int(payload.get("total"))
                    event = str(payload.get("event") or "")
                    image_path = _rel_probe_path(cast(str | None, payload.get("image_path")))
                    status = str(payload.get("status") or "")
                    current_index = _coerce_int(payload.get("index"))
                    record_payload = payload.get("record")
                    last_record = record_payload if isinstance(record_payload, dict) else None
                    availability = cast(dict[str, object], summary.get("telemetry_availability") or {})
                    coverage_line = " · ".join(_coverage_fragments(availability))

                    if event == "image_start":
                        status_line = f"Estado: procesando {current_index}/{total} · {image_path}"
                    elif event == "image_done" and status == "ok":
                        status_line = f"Estado: completada {current_index}/{total} · {image_path}"
                    elif event == "image_done":
                        status_line = f"Estado: error {current_index}/{total} · {image_path}"
                    elif event == "complete":
                        status_line = f"Estado: finalizado · {completed}/{total} imágenes procesadas"
                    else:
                        status_line = f"Estado: preparando ejecución · muestra {total}"

                    if event == "image_done" and last_record is not None:
                        _append_recent_record(kit, recent_lines, last_record)

                    _render_live_dashboard(
                        kit,
                        panel,
                        current=completed,
                        total=total,
                        stats_line=f"Estadísticas: OK={summary.get('ok', 0)} | Errores={summary.get('fail', 0)}",
                        status_line=status_line,
                        metrics_line=(
                            "Promedios parciales: "
                            f"TTFT={_format_metric_value(cast(dict[str, object], summary.get('ttft') or {}).get('avg'), suffix=' s')} | "
                            f"TPS={_format_metric_value(cast(dict[str, object], summary.get('tps') or {}).get('avg'))} | "
                            f"latencia={_format_metric_value(cast(dict[str, object], summary.get('total_duration') or {}).get('avg'), suffix=' s')}"
                        ),
                        coverage_line=f"Cobertura parcial: {coverage_line}" if coverage_line else None,
                        recent_title="Últimas inferencias:",
                        recent_lines=list(recent_lines),
                    )

                summary = run_telemetry_batch(
                    model_id=model_tag,
                    schema_name=schema_name,
                    schema_cls=schema_cls,
                    images=images,
                    max_images=5,
                    on_progress=on_probe_progress,
                )
                availability = cast(dict[str, object], summary.get("telemetry_availability") or {})
                rows = [
                    ("Modelo", summary["model_id"], "OK"),
                    ("Esquema", summary["schema_name"], "OK"),
                    ("Muestra", str(summary["sample_size"]), "OK"),
                    ("Inferencias OK", str(summary["ok"]), "OK" if summary["ok"] > 0 else "WARN"),
                    ("Errores", str(summary["fail"]), "OK" if summary["fail"] == 0 else "WARN"),
                    ("Modelo resuelto", str(summary["static_model_info"]["resolved_model_id"]), "OK" if summary["static_model_info"]["resolved_model_id"] else "WARN"),
                    ("Arquitectura", str(summary["static_model_info"]["architecture"]), "OK" if summary["static_model_info"]["architecture"] else "WARN"),
                    ("Stop reason", str(summary["static_model_info"]["stop_reason"]), "OK" if summary["static_model_info"]["stop_reason"] else "WARN"),
                    ("TTFT medio (s)", _format_metric_value(summary["ttft"]["avg"]), "OK" if summary["ttft"]["avg"] is not None else "WARN"),
                    ("TPS medio", _format_metric_value(summary["tps"]["avg"]), "OK" if summary["tps"]["avg"] is not None else "WARN"),
                    ("Generacion media (s)", _format_metric_value(summary["generation_duration"]["avg"]), "OK" if summary["generation_duration"]["avg"] is not None else "WARN"),
                    ("Prompt tokens medios", _format_metric_value(summary["prompt_tokens"]["avg"]), "OK" if summary["prompt_tokens"]["avg"] is not None else "WARN"),
                    ("Output tokens medios", _format_metric_value(summary["completion_tokens"]["avg"]), "OK" if summary["completion_tokens"]["avg"] is not None else "WARN"),
                    ("Total tokens medios", _format_metric_value(summary["total_tokens"]["avg"]), "OK" if summary["total_tokens"]["avg"] is not None else "WARN"),
                    ("Latencia media (s)", _format_metric_value(summary["total_duration"]["avg"]), "OK" if summary["total_duration"]["avg"] is not None else "WARN"),
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
                )
                note_lines = []
                if ttft_note:
                    kit.log(ttft_note, "warning")
                    note_lines.append(f"  TTFT: {ttft_note}")
                if tps_note:
                    kit.log(tps_note, "warning")
                    note_lines.append(f"  TPS: {tps_note}")
                if coverage_fragments:
                    kit.log(f"Cobertura: {' · '.join(coverage_fragments)}", "step")
                    note_lines.append(f"  Cobertura: {' · '.join(coverage_fragments)}")
                _render_final_sections_screen(
                    kit,
                    app,
                    subtitle=f"TELEMETRY PROBE · {schema_name}",
                    intro=_standard_final_intro(),
                    sections=[
                        ("Resumen", _build_summary_lines(kit, [(str(a), str(b), str(c)) for a, b, c in rows])),
                        ("Observaciones", note_lines or ["  Sin observaciones adicionales."]),
                        ("Actividad reciente", detail_lines),
                    ],
                )
                if summary["fail"] == 0:
                    kit.log(f"Telemetry Probe completado: {summary['ok']} inferencias OK.", "success")
                else:
                    kit.log(
                        f"Telemetry Probe completado con incidencias: {summary['ok']} OK, {summary['fail']} errores.",
                        "warning",
                    )
            except Exception as error:
                kit.log(f"Telemetry Probe terminó con error: {error}", "error")
            kit.wait("Press any key to return to model selector...")

    def run_batch_runner_wrapper() -> None:
        """Ejecuta el batch runner con exportado incremental desde la TUI."""
        from src.scripts.batch_runner import run_batch_job

        while True:
            model_tag = _select_model(
                menu_id="batch_runner_model_selector",
                subtitle="BATCH RUNNER · SELECT MODEL",
            )
            if model_tag is None:
                return

            schema_variant = _select_schema_variant("batch_runner", "BATCH RUNNER")
            if schema_variant is None:
                continue
            schema_name, _schema_cls = schema_variant

            max_images = _select_batch_size()
            if max_images == "BACK":
                continue
            selected_max_images = cast(int | None, max_images)

            kit.clear()
            app.print_banner()
            kit.subtitle(f"BATCH RUNNER · {schema_name}")
            kit.log(
                "Exportando resultados incrementales en JSONL desde data/...",
                "step",
            )
            try:
                recent_lines = _make_recent_lines()
                recent_records: list[dict[str, object]] = []
                panel = _make_live_panel(
                    kit,
                    app,
                    subtitle=f"BATCH RUNNER · {schema_name}",
                    intro="Exportando resultados incrementales en JSONL desde data/...",
                )

                def on_batch_progress(payload: dict[str, object]) -> None:
                    summary = cast(dict[str, object], payload.get("summary") or {})
                    total = _coerce_int(payload.get("total"))
                    event = str(payload.get("event") or "")
                    current_index = _coerce_int(payload.get("index"))
                    image_path = _rel_probe_path(cast(str | None, payload.get("image_path")))
                    record_payload = payload.get("record")
                    last_record = record_payload if isinstance(record_payload, dict) else None
                    completed = _coerce_int(summary.get("processed"))
                    invalid_count = _coerce_int(summary.get("invalid"))
                    fail_count = _coerce_int(summary.get("fail"))

                    if event == "image_start":
                        status_line = f"Estado: procesando {current_index}/{total} · {image_path}"
                    elif event == "image_done" and str(payload.get("status") or "") == "ok":
                        status_line = f"Estado: exportada {current_index}/{total} · {image_path}"
                    elif event == "image_done" and str(payload.get("status") or "") == "invalid":
                        status_line = f"Estado: respuesta inválida {current_index}/{total} · {image_path}"
                    elif event == "complete":
                        status_line = f"Estado: finalizado · {completed}/{total} imágenes exportadas"
                    else:
                        status_line = f"Estado: preparando ejecución · muestra {total}"

                    if event == "image_done" and last_record is not None:
                        _append_recent_record(kit, recent_lines, last_record)
                        recent_records.append(dict(last_record))
                        if len(recent_records) > 5:
                            recent_records.pop(0)

                    _render_live_dashboard(
                        kit,
                        panel,
                        current=completed,
                        total=total,
                        stats_line=(
                            f"Estadísticas: OK={summary.get('ok', 0)} | "
                            f"Inválidas={invalid_count} | Errores={fail_count}"
                        ),
                        status_line=status_line,
                        metrics_line=(
                            "Promedios parciales: "
                            f"TTFT={_format_metric_value(cast(dict[str, object], summary.get('ttft') or {}).get('avg'), suffix=' s')} | "
                            f"TPS={_format_metric_value(cast(dict[str, object], summary.get('tps') or {}).get('avg'))} | "
                            f"latencia={_format_metric_value(cast(dict[str, object], summary.get('total_duration') or {}).get('avg'), suffix=' s')}"
                        ),
                        recent_title="Últimos registros exportados:",
                        recent_lines=list(recent_lines),
                    )

                try:
                    summary = run_batch_job(
                        model_id=model_tag,
                        image_dir="data",
                        schema_name=schema_name,
                        max_images=selected_max_images,
                        shuffle=True,
                        on_progress=on_batch_progress,
                    )
                except TypeError as error:
                    if "on_progress" not in str(error):
                        raise
                    summary = run_batch_job(
                        model_id=model_tag,
                        image_dir="data",
                        schema_name=schema_name,
                        max_images=selected_max_images,
                        shuffle=True,
                    )
                _render_final_sections_screen(
                    kit,
                    app,
                    subtitle=f"BATCH RUNNER · {schema_name}",
                    intro=_standard_final_intro(),
                    sections=[
                        ("Resumen", _build_summary_lines(kit, _format_batch_summary_rows(summary))),
                        (
                            "Actividad reciente",
                            _build_recent_record_lines(
                                kit,
                                recent_records,
                                empty_message="  Sin registros recientes.",
                            ),
                        ),
                    ],
                )
                level = "success" if summary["fail"] == 0 and summary["invalid"] == 0 else "warning"
                kit.log(
                    f"Batch Runner completado: {summary['ok']} OK, {summary['invalid']} inválidas, "
                    f"{summary['fail']} errores. Salida: {summary['output_path']}",
                    level,
                )
            except Exception as error:
                kit.log(f"Batch Runner terminó con error: {error}", "error")
            kit.wait("Press any key to return to model selector...")

    def run_response_inspector_wrapper() -> None:
        """Ejecuta el inspector de respuesta del SDK con defaults automáticos."""
        from argparse import Namespace
        from src.scripts.test_response import build_summary_sections, run_inspection, save_inspection_payload

        while True:
            model_tag = _select_model(
                menu_id="response_inspector_model_selector",
                subtitle="RESPONSE INSPECTOR · SELECT MODEL",
            )
            if model_tag is None:
                return

            mode = _select_response_inspector_mode()
            if mode is None:
                continue
            structured, with_reasoning = mode

            kit.clear()
            app.print_banner()
            kit.subtitle("RESPONSE INSPECTOR")
            kit.log(
                f"Inspeccionando respuesta real del SDK con el modelo {model_tag} y defaults automáticos...",
                "step",
            )
            try:
                payload = run_inspection(
                    Namespace(
                        model=model_tag,
                        image=None,
                        prompt=None,
                        schema=None,
                        structured=structured,
                        with_reasoning=with_reasoning,
                        temperature=0.0,
                        server_api_host=None,
                        api_token=None,
                        output=None,
                        print_json=False,
                    )
                )
                output_path = save_inspection_payload(payload)
                sections = []
                for title, rows in build_summary_sections(payload):
                    sections.append((title, _build_named_value_lines(kit, rows)))
                sections.append(("Salida", [f"  Archivo guardado en: {output_path}"]))
                _render_final_sections_screen(
                    kit,
                    app,
                    subtitle="RESPONSE INSPECTOR",
                    intro=_standard_final_intro(),
                    sections=sections,
                )
                kit.log(f"Response Inspector completado. Salida: {output_path}", "success")
            except Exception as error:
                kit.log(f"Response Inspector terminó con error: {error}", "error")
            kit.wait("Press any key to return to model selector...")

    def run_smoke_test_wrapper() -> None:
        """Lanza smoke test con selector de modelo."""
        while True:
            model_tag = _select_model(
                menu_id="run_smoke_model_selector",
                subtitle="SELECT INFERENCE MODEL",
            )
            if model_tag is None:
                return

            try:
                from src.scripts.test_inference import ensure_test_images, run_smoke_test
            except Exception as error:
                kit.log(f"Could not import smoke test script: {error}", "error")
                kit.log("Smoke test failed (Exit Code 1). Check output above.", "error")
                kit.wait("Press any key to return to model selector...")
                continue

            recent_lines = _make_recent_lines()
            recent_records: list[dict[str, object]] = []
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

            def on_smoke_progress(payload: dict[str, object]) -> None:
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
                    status_line = (
                        "Estado: modelo precargado en "
                        f"{_format_metric_value(summary.get('preload_seconds'), suffix=' s')}"
                    )
                elif event == "case_start":
                    status_line = f"Estado: procesando {current_index}/{total} · {case.get('id', 'caso')}"
                elif event == "case_done":
                    status_line = f"Estado: completado {current_index}/{total} · {case.get('id', 'caso')}"
                elif event == "complete":
                    status_line = f"Estado: finalizado · {processed}/{total} casos revisados"
                else:
                    status_line = f"Estado: preparando ejecución · muestra {total}"

                if event == "case_done" and record is not None:
                    record_dict = cast(dict[str, object], record)
                    payload = record_dict.get("payload")
                    if payload is None and record_dict.get("response_preview") is not None:
                        payload = {"preview": record_dict.get("response_preview")}
                    recent_record = {
                        "image_name": str(record_dict.get("case_id") or record_dict.get("label") or "caso"),
                        "status": record_dict.get("status") or "unknown",
                        "payload": payload,
                        "validation_error": record_dict.get("message"),
                        "error": record_dict.get("error"),
                    }
                    _append_recent_record(kit, recent_lines, recent_record)
                    recent_records.append(recent_record)
                    if len(recent_records) > 5:
                        recent_records.pop(0)

                _render_live_dashboard(
                    kit,
                    panel,
                    current=processed,
                    total=total,
                    stats_line=(
                        f"Estadísticas: OK={summary.get('passed', 0)} | "
                        f"Inválidos={summary.get('failed', 0)}"
                    ),
                    status_line=status_line,
                    metrics_line=(
                        f"Modelo precargado: {_format_metric_value(summary.get('preload_seconds'), suffix=' s')}"
                        if summary.get("preload_seconds") is not None
                        else None
                    ),
                    recent_title="Últimos casos:",
                    recent_lines=list(recent_lines),
                )

            kit.clear()
            result_code = 1
            try:
                result_code = run_smoke_test(
                    model_tag,
                    ensure_test_images(),
                    on_progress=on_smoke_progress,
                )
                _render_final_sections_screen(
                    kit,
                    app,
                    subtitle="SMOKE TEST · INFERENCE DEMO",
                    intro=_standard_final_intro(),
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
                            ),
                        ),
                        (
                            "Actividad reciente",
                            _build_recent_record_lines(
                                kit,
                                recent_records,
                                empty_message="  Sin registros.",
                            ),
                        ),
                    ],
                )
            except Exception as error:
                kit.log(f"Smoke test crashed: {error}", "error")
            if result_code != 0:
                kit.log("Smoke test failed (Exit Code 1). Check output above.", "error")
            kit.wait("Press any key to return to model selector...")

    # ------------------------------------------------------------------
    # Menú principal de tests
    # ------------------------------------------------------------------

    options = [
        kit.MenuItem(
            " Manage/Pull LM Studio Models...",
            lambda: setup_models_ui.manage_models_menu_ui(kit, app),
            description="Descarga, carga o descarga de memoria modelos mediante LM Studio CLI.",
        ),
        kit.MenuItem(
            " Run All Unit Tests (pytest)",
            run_all_unit_tests,
            description="Ejecuta todos los tests dentro de la carpeta tests/.",
        ),
        kit.MenuItem(
            " Run Specific Test File...",
            run_specific_test,
            description="Abre un selector para ejecutar un único archivo de test.",
        ),
        kit.MenuItem(
            " Run Smoke Test (Inference Demo)",
            run_smoke_test_wrapper,
            description="Lanza una inferencia rápida para validar flujo modelo+imagen.",
        ),
        kit.MenuItem(
            " Run Schema Tester (VLM Interactive)",
            run_schema_tester_wrapper,
            description=(
                "Selecciona modelo y esquema Pydantic, ejecuta inferencia sobre 5 imágenes "
                "aleatorias y valida automáticamente que las respuestas cumplan el esquema."
            ),
        ),
        kit.MenuItem(
            " Run Telemetry Probe (TTFT/TPS)",
            run_telemetry_probe_wrapper,
            description=(
                "Mide métricas del SDK response.stats y recursos de /v1/models "
                "sobre una muestra de imágenes del proyecto."
            ),
        ),
        kit.MenuItem(
            " Run Batch Runner",
            run_batch_runner_wrapper,
            description=(
                "Ejecuta inferencia masiva con guardado incremental para análisis posterior "
                "sin romper la TUI."
            ),
        ),
        kit.MenuItem(
            " Run Response Inspector (SDK Fields)",
            run_response_inspector_wrapper,
            description=(
                "Lanza una petición real y resume solo los campos útiles que devuelve el SDK, "
                "con autodetección de imagen y prompt."
            ),
        ),
        kit.MenuItem(
            " Return to Main Menu",
            lambda: "BACK",
            description="Vuelve al menú principal.",
        ),
    ]

    while True:
        choice = kit.menu(
            options,
            header_func=_make_header("TEST & MODEL MANAGER"),
            multi_select=False,
            menu_id="tests_manager_menu",
            nav_hint_text="↑/↓ navegar opciones · ENTER abrir/ejecutar · ESC volver al menú principal",
        )
        if choice == "BACK" or not choice:
            break

        if isinstance(choice, list):
            choice = choice[0] if len(choice) > 0 else None

        if choice and hasattr(choice, "action") and callable(choice.action):
            kit.clear()
            res = choice.action()
            if res == "BACK":
                break
