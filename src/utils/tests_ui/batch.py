from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Callable, cast

from ..setup_ui_io import ask_choice
from .test_dashboards_ui import (
    _append_recent_record,
    _build_progress_bar,
    _coerce_int,
    _format_metric_value,
    _make_live_panel,
    _make_recent_records,
    _rel_probe_path,
    _render_final_sections_screen,
    _render_live_dashboard,
    _standard_final_intro,
)

if TYPE_CHECKING:
    from ..menu_kit import AppContext, UIKit


def run_batch_runner_wrapper(
    kit: "UIKit",
    app: "AppContext",
    *,
    select_manifest_for_batch: Callable[[], dict[str, Any] | None],
    execution_schema_name: Callable[[str, bool], str],
    linked_batch_output_path: Callable[..., Any],
    manifest_execution_snapshot: Callable[..., dict[str, Any]],
    prune_output_records_for_model: Callable[..., bool],
    status_label: Callable[[dict[str, Any]], str],
) -> None:
    """Run queued batch inference jobs defined by autosufficient manifests.

    Args:
        kit: Terminal UI toolkit.
        app: Application context used to render shared UI chrome.
        select_manifest_for_batch: Callback that selects or creates the manifest bundle.
        execution_schema_name: Callback that derives execution schema display name.
        linked_batch_output_path: Callback that computes output JSONL path.
        manifest_execution_snapshot: Callback that computes current execution status.
        prune_output_records_for_model: Callback that removes rows for one model in shared JSONL.
        status_label: Callback that renders colored status labels.
    """
    from src.scripts.batch_runner import run_batch_job, upsert_batch_execution_summary

    def _as_yes(value: object) -> bool:
        """Interpret interactive answers as explicit yes/no values.

        Args:
            value: Raw value returned by UI confirmation prompt.

        Returns:
            `True` only for explicit affirmative values.
        """
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        return text in {"y", "yes", "s", "si", "true", "1"}
    
    def _compact_model_label(model_tag: str) -> str:
        """Compacta el model tag para pantallas estrechas preservando inicio/fin."""
        text = str(model_tag or "").strip()
        if not text:
            return "N/D"
        max_len = max(14, min(30, int(kit.width() * 0.33)))
        if len(text) <= max_len:
            return text
        head = max(6, (max_len - 1) // 2)
        tail = max(5, max_len - head - 1)
        return f"{text[:head]}…{text[-tail:]}"

    def _is_model_snapshot_complete(snapshot: dict[str, Any]) -> bool:
        """Valida de forma estricta si un modelo está realmente completado."""
        status = str(snapshot.get("status") or "").strip().lower()
        total = int(snapshot.get("total", 0) or 0)
        ok = int(snapshot.get("ok", 0) or 0)
        errors = int(snapshot.get("errors", 0) or 0)
        pending = int(snapshot.get("pending", 0) or 0)
        return status == "green" and total > 0 and ok == total and errors == 0 and pending == 0

    def _rel_path(path_value: str) -> str:
        """Convierte rutas absolutas a relativas para mejorar legibilidad en terminal."""
        path_text = str(path_value or "").strip()
        if not path_text:
            return "N/D"
        try:
            return os.path.relpath(path_text, ".")
        except ValueError:
            return path_text

    def _truncate(text: str, limit: int) -> str:
        """Recorta texto largo preservando inicio/fin para columnas estrechas."""
        value = str(text or "")
        if len(value) <= limit:
            return value
        if limit <= 6:
            return value[:limit]
        head = max(3, (limit - 1) // 2)
        tail = max(2, limit - head - 1)
        return f"{value[:head]}…{value[-tail:]}"

    def _status_text(snapshot: dict[str, Any]) -> str:
        """Etiqueta de estado sin ANSI para la tabla resumen final."""
        state = str(snapshot.get("status") or "").strip().lower()
        if state == "green":
            return "COMPLETO"
        if state == "yellow":
            return "PARCIAL"
        return "ERROR"

    def _status_color_code(snapshot: dict[str, Any]) -> str:
        """Obtiene el color ANSI asociado al estado del snapshot."""
        state = str(snapshot.get("status") or "").strip().lower()
        if state == "green":
            return str(getattr(kit.style, "OKGREEN", ""))
        if state == "yellow":
            return str(getattr(kit.style, "WARNING", ""))
        return str(getattr(kit.style, "FAIL", ""))

    def _colorize_model(model_tag: str, snapshot: dict[str, Any]) -> str:
        """Pinta el nombre del modelo según estado (verde/amarillo/rojo)."""
        color = _status_color_code(snapshot)
        endc = str(getattr(kit.style, "ENDC", "")) if color else ""
        return f"{color}{model_tag}{endc}" if color else model_tag

    def _snapshot_summary_line(model_tag: str, snapshot: dict[str, Any]) -> str:
        """Resumen compacto del estado actual para prompts interactivos."""
        ok = int(snapshot.get("ok", 0) or 0)
        total = int(snapshot.get("total", 0) or 0)
        pending = int(snapshot.get("pending", 0) or 0)
        errors = int(snapshot.get("errors", 0) or 0)
        status_plain = _status_text(snapshot)
        return (
            f"{_colorize_model(model_tag, snapshot)} · {status_plain} · "
            f"OK/TOT={ok}/{total} · Pend={pending} · Err={errors}"
        )

    while True:
        selected_manifest_bundle = select_manifest_for_batch()
        if selected_manifest_bundle is None:
            return

        selected_manifest = str(selected_manifest_bundle.get("manifest_path") or "")
        manifest_overview = cast(dict[str, Any], selected_manifest_bundle.get("overview") or {})
        manifest_config = cast(dict[str, Any], selected_manifest_bundle.get("config") or {})
        if not selected_manifest:
            kit.log("Batch Runner cancelado: manifiesto inválido.", "error")
            kit.wait("Press any key to return to tests manager...")
            continue

        if not manifest_config:
            kit.log(
                "El manifiesto seleccionado no incluye configuración de ejecución (modelos/schema).",
                "error",
            )
            kit.log("Genera uno nuevo desde 'Generar manifiesto nuevo'.", "info")
            kit.wait("Press any key to return to tests manager...")
            continue

        schema_base_name = cast(str, manifest_config.get("schema_name") or "").strip()
        include_reasoning = bool(manifest_config.get("include_reasoning"))
        schema_exec_name = execution_schema_name(schema_base_name, include_reasoning)
        queued_models = cast(list[str], manifest_config.get("models") or [])
        if not schema_base_name or not queued_models:
            kit.log("Configuración incompleta en manifiesto (faltan schema/modelos).", "error")
            kit.wait("Press any key to return to tests manager...")
            continue

        kit.clear()
        app.print_banner()
        kit.subtitle(f"BATCH RUNNER · {schema_exec_name}")
        kit.log(f"Manifiesto seleccionado: {selected_manifest}", "step")
        kit.log(cast(str, manifest_overview.get("description") or "Sin descripción disponible."), "info")
        kit.log(f"Modelos en cola: {', '.join(queued_models)}", "info")

        queue_results: list[tuple[str, dict[str, Any], dict[str, Any], str]] = []
        initial_snapshots: dict[str, dict[str, Any]] = {
            model: manifest_execution_snapshot(
                manifest_path=selected_manifest,
                model_tag=model,
                schema_name=schema_exec_name,
            )
            for model in queued_models
        }
        colored_queue_models = [
            _colorize_model(model, cast(dict[str, Any], initial_snapshots.get(model) or {}))
            for model in queued_models
        ]
        kit.log("Modelos por estado: " + " | ".join(colored_queue_models), "info")
        completed_models = [
            model for model in queued_models if _is_model_snapshot_complete(initial_snapshots.get(model, {}))
        ]
        pending_or_partial_models = [model for model in queued_models if model not in completed_models]
        models_to_execute: set[str] = set(pending_or_partial_models)

        if completed_models:
            def _rerun_info_text() -> str:
                lines: list[str] = []
                lines.append("Resumen actual por modelo:")
                lines.extend(
                    [
                        "  - " + _snapshot_summary_line(
                            model,
                            cast(dict[str, Any], initial_snapshots.get(model) or {}),
                        )
                        for model in queued_models
                    ]
                )
                lines.append("")
                lines.append(
                    "Regla por defecto: se ejecutan los pendientes/parciales; "
                    "los completos solo si los incluyes."
                )
                return "\n".join(lines)

            rerun_choice = ask_choice(
                question=(
                    "Hay modelos ya completos en este manifiesto. "
                    "¿Cómo quieres proceder con la reejecución?"
                ),
                options=["Si (dataset completo)", "No", "Seleccionar modelos..."],
                default_index=1,
                style=kit.style,
                read_key_fn=kit.read_key,
                clear_screen_fn=kit.clear,
                info_text=_rerun_info_text,
            )

            if rerun_choice == 0:
                models_to_execute = set(queued_models)
            elif rerun_choice == 2:
                def _render_rerun_header() -> None:
                    kit.clear()
                    app.print_banner()
                    kit.subtitle(f"BATCH RUNNER · {schema_exec_name} · REEJECUTAR MODELOS")

                model_items = [
                    kit.MenuItem(
                        model_tag,
                        description=(
                            "Estado actual: COMPLETO (se limpiará y reejecutará)."
                            if model_tag in completed_models
                            else "Estado actual: PENDIENTE/PARCIAL (se ejecutará)."
                        ),
                    )
                    for model_tag in queued_models
                ]
                for model_item in model_items:
                    model_label = str(model_item.label).strip()
                    model_snapshot = cast(dict[str, Any], initial_snapshots.get(model_label) or {})

                    def _dynamic_model_label(
                        _is_selected_row: bool,
                        _label: str = model_label,
                        _snapshot: dict[str, Any] = model_snapshot,
                    ) -> str:
                        return _snapshot_summary_line(_label, _snapshot)

                    model_item.dynamic_label = _dynamic_model_label

                selected_models = kit.menu(
                    model_items,
                    header_func=_render_rerun_header,
                    menu_id="batch_rerun_completed_models",
                    multi_select=True,
                    nav_hint_text="↑/↓ navegar · SPACE seleccionar · ENTER confirmar · ESC cancelar",
                )
                if selected_models:
                    if not isinstance(selected_models, list):
                        selected_models = [selected_models]
                    models_to_execute = {
                        str(item.label).strip()
                        for item in selected_models
                        if str(item.label).strip()
                    }
                else:
                    models_to_execute = set()

            if models_to_execute:
                kit.log(
                    "Modelos seleccionados para ejecutar: "
                    + ", ".join(sorted(models_to_execute)),
                    "info",
                )
            else:
                kit.log("No se ejecutará ningún modelo en esta corrida.", "warning")

        runnable_models = [
            model
            for model in queued_models
            if model in models_to_execute
        ]
        runnable_index_by_model = {model: index for index, model in enumerate(runnable_models)}

        queue_targets: dict[str, int] = {
            model: int(
                cast(dict[str, Any], initial_snapshots.get(model) or {}).get("total", 0)
            )
            for model in runnable_models
        }
        total_global_target = sum(queue_targets.values())
        completed_global_images = 0

        for queued_model in queued_models:
            output_path = str(
                linked_batch_output_path(
                    manifest_path=selected_manifest,
                    model_tag=queued_model,
                    schema_name=schema_exec_name,
                )
            )
            manifest_snapshot = manifest_execution_snapshot(
                manifest_path=selected_manifest,
                model_tag=queued_model,
                schema_name=schema_exec_name,
            )

            if queued_model not in models_to_execute:
                kit.log(f"[{queued_model}] Omitido por selección de ejecución.", "info")
                queue_results.append((queued_model, manifest_snapshot, manifest_snapshot, output_path))
                continue

            if _is_model_snapshot_complete(manifest_snapshot):
                ok_prune = prune_output_records_for_model(output_path=output_path, model_tag=queued_model)
                if not ok_prune:
                    kit.log(f"[{queued_model}] No se pudo limpiar el JSONL compartido para este modelo.", "error")
                    queue_results.append((queued_model, manifest_snapshot, manifest_snapshot, output_path))
                    continue

            rerun_only_pending = str(manifest_snapshot.get("status") or "") == "yellow"

            kit.clear()
            app.print_banner()
            kit.subtitle(f"BATCH RUNNER · {schema_exec_name} · {queued_model}")
            kit.log(
                f"Exportando resultados incrementales en JSONL desde manifiesto: {selected_manifest}",
                "step",
            )

            summary: dict[str, Any] = {
                "ok": 0,
                "invalid": 0,
                "fail": 0,
                "output_path": output_path,
            }
            current_model_target = int(manifest_snapshot.get("total", queue_targets.get(queued_model, 0)) or 0)
            models_done_before = int(runnable_index_by_model.get(queued_model, 0))
            models_target = len(runnable_models)
            show_model_queue_bar = models_target > 1
            compact_model = _compact_model_label(queued_model)

            try:
                recent_records = _make_recent_records(limit=2)
                panel = _make_live_panel(
                    kit,
                    app,
                    subtitle=f"BATCH RUNNER · {schema_exec_name} · {queued_model}",
                    intro=f"Exportando resultados incrementales en JSONL desde manifiesto: {selected_manifest}",
                )

                def on_batch_progress(payload: dict[str, object]) -> None:
                    """Renderiza progreso en vivo durante el batch actual.

                    Args:
                        payload: Evento de progreso emitido por `run_batch_job`.
                    """
                    summary_progress = cast(dict[str, object], payload.get("summary") or {})
                    total = _coerce_int(payload.get("total"))
                    event = str(payload.get("event") or "")
                    current_index = _coerce_int(payload.get("index"))
                    image_path = _rel_probe_path(cast(str | None, payload.get("image_path")))
                    record_payload = payload.get("record")
                    last_record = record_payload if isinstance(record_payload, dict) else None
                    completed = _coerce_int(summary_progress.get("processed"))
                    invalid_count = _coerce_int(summary_progress.get("invalid"))
                    fail_count = _coerce_int(summary_progress.get("fail"))
                    global_completed = completed_global_images + completed
                    global_target = total_global_target if total_global_target > 0 else current_model_target

                    model_queue_current = models_done_before
                    if event == "complete" and models_target > 0:
                        model_queue_current = min(models_target, models_done_before + 1)

                    model_progress_line = (
                        f"Cola de modelos: {_build_progress_bar(kit, model_queue_current, models_target)}"
                    )
                    current_images_line = (
                        f"Modelo actual ({compact_model}): {_build_progress_bar(kit, completed, max(total, current_model_target))}"
                    )

                    if event == "image_start":
                        status_line = f"Estado actual: procesando imagen {current_index}/{total} · {image_path}"
                    elif event == "image_done" and str(payload.get("status") or "") == "ok":
                        status_line = f"Estado actual: exportada imagen {current_index}/{total} · {image_path}"
                    elif event == "image_done" and str(payload.get("status") or "") == "invalid":
                        status_line = f"Estado actual: respuesta inválida en {current_index}/{total} · {image_path}"
                    elif event == "complete":
                        status_line = f"Estado actual: finalizado · {completed}/{total} imágenes exportadas"
                    else:
                        status_line = f"Estado actual: preparando ejecución · muestra {total}"

                    if event == "image_done" and last_record is not None:
                        _append_recent_record(recent_records, last_record)

                    _render_live_dashboard(
                        kit,
                        panel,
                        current=global_completed,
                        total=global_target,
                        stats_line=(
                            f"Resumen del modelo actual: OK={summary_progress.get('ok', 0)} | "
                            f"Inválidas={invalid_count} | Errores={fail_count}"
                        ),
                        status_line=status_line,
                        metrics_line=(
                            "Promedios parciales: "
                            f"TTFT={_format_metric_value(cast(dict[str, object], summary_progress.get('ttft') or {}).get('avg'), suffix=' s')} | "
                            f"TPS={_format_metric_value(cast(dict[str, object], summary_progress.get('tps') or {}).get('avg'))} | "
                            f"latencia={_format_metric_value(cast(dict[str, object], summary_progress.get('total_duration') or {}).get('avg'), suffix=' s')}"
                        ),
                        extra_lines=[model_progress_line, current_images_line] if show_model_queue_bar else [current_images_line],
                        recent_title="Últimos registros exportados:",
                        recent_records=list(recent_records),
                    )

                def execute_batch_once(
                    *,
                    manifest_to_run: str,
                    batch_output: str,
                    pending_image_paths: list[str] | None = None,
                ) -> dict[str, Any]:
                    """Ejecuta una iteración de batch (completa o reanudada)."""
                    summary_local: dict[str, Any]
                    try:
                        summary_local = run_batch_job(
                            model_id=queued_model,
                            manifest=manifest_to_run,
                            schema_name=schema_base_name,
                            include_reasoning=include_reasoning,
                            output_path=batch_output,
                            max_images=None,
                            shuffle=not rerun_only_pending,
                            pending_image_paths=pending_image_paths,
                            on_progress=on_batch_progress,
                        )
                    except TypeError as error:
                        if "on_progress" not in str(error):
                            raise
                        summary_local = run_batch_job(
                            model_id=queued_model,
                            manifest=manifest_to_run,
                            schema_name=schema_base_name,
                            include_reasoning=include_reasoning,
                            output_path=batch_output,
                            max_images=None,
                            shuffle=not rerun_only_pending,
                            pending_image_paths=pending_image_paths,
                        )
                    return summary_local

                pending_image_paths: list[str] | None = None
                if rerun_only_pending:
                    pending_entries = cast(list[dict[str, Any]], manifest_snapshot.get("pending_entries") or [])
                    pending_image_paths = [
                        str(item.get("image_path") or "").strip()
                        for item in pending_entries
                        if str(item.get("image_path") or "").strip()
                    ]

                summary = execute_batch_once(
                    manifest_to_run=selected_manifest,
                    batch_output=output_path,
                    pending_image_paths=pending_image_paths,
                )

                while True:
                    refreshed = manifest_execution_snapshot(
                        manifest_path=selected_manifest,
                        model_tag=queued_model,
                        schema_name=schema_exec_name,
                    )
                    pending_before_retry = int(refreshed.get("pending", 0))
                    if pending_before_retry <= 0:
                        break
                    retry_now = _as_yes(
                        kit.ask(
                            (
                                f"[{queued_model}] Quedan imágenes pendientes o con error. "
                                "¿Quieres reejecutarlas ahora sin salir del menú?"
                            ),
                            "y",
                        )
                    )
                    if not retry_now:
                        break
                    retry_pending_paths = [
                        str(item.get("image_path") or "").strip()
                        for item in cast(list[dict[str, Any]], refreshed.get("pending_entries") or [])
                        if str(item.get("image_path") or "").strip()
                    ]
                    if not retry_pending_paths:
                        break
                    summary = execute_batch_once(
                        manifest_to_run=selected_manifest,
                        batch_output=output_path,
                        pending_image_paths=retry_pending_paths,
                    )

                    refreshed_after_retry = manifest_execution_snapshot(
                        manifest_path=selected_manifest,
                        model_tag=queued_model,
                        schema_name=schema_exec_name,
                    )
                    pending_after_retry = int(refreshed_after_retry.get("pending", 0))
                    if pending_after_retry >= pending_before_retry:
                        kit.log(
                            (
                                f"[{queued_model}] Reintento sin progreso "
                                f"({pending_after_retry} pendientes). Se detiene para evitar bucle."
                            ),
                            "warning",
                        )
                        break

                final_snapshot = manifest_execution_snapshot(
                    manifest_path=selected_manifest,
                    model_tag=queued_model,
                    schema_name=schema_exec_name,
                )
                completed_global_images += int(final_snapshot.get("ok", 0))
                queue_results.append((queued_model, manifest_snapshot, final_snapshot, output_path))
                level = "success" if int(final_snapshot.get("pending", 0)) == 0 else "warning"
                kit.log(
                    f"[{queued_model}] Batch completado: {summary['ok']} OK, {summary['invalid']} inválidas, "
                    f"{summary['fail']} errores. Salida: {summary['output_path']}",
                    level,
                )
            except Exception as error:
                final_snapshot = manifest_execution_snapshot(
                    manifest_path=selected_manifest,
                    model_tag=queued_model,
                    schema_name=schema_exec_name,
                )
                completed_global_images += int(final_snapshot.get("ok", 0))
                queue_results.append((queued_model, manifest_snapshot, final_snapshot, output_path))
                kit.log(f"[{queued_model}] Batch Runner terminó con error: {error}", "error")

        total_ok = 0
        total_items = 0
        total_pending = 0
        complete_models = 0
        partial_models = 0
        failed_models = 0
        output_paths: list[str] = []
        for _model_name, _initial_snapshot, final_snapshot, output_path in queue_results:
            total_ok += int(final_snapshot.get("ok", 0) or 0)
            total_items += int(final_snapshot.get("total", 0) or 0)
            pending_value = int(final_snapshot.get("pending", 0) or 0)
            total_pending += pending_value
            state = str(final_snapshot.get("status") or "").strip().lower()
            if state == "green":
                complete_models += 1
            elif state == "yellow":
                partial_models += 1
            else:
                failed_models += 1
            output_paths.append(str(final_snapshot.get("output_path") or output_path or ""))

        unique_outputs = sorted({path for path in output_paths if path})
        aggregate_summaries: dict[str, dict[str, Any]] = {}
        for output_item in unique_outputs:
            aggregate_summary = upsert_batch_execution_summary(
                output_path=output_item,
                schema_name=schema_exec_name,
                model_ids=queued_models,
            )
            if aggregate_summary:
                aggregate_summaries[output_item] = aggregate_summary
        primary_output = unique_outputs[0] if unique_outputs else ""
        aggregate_models_accuracy: dict[str, str] = {}

        def _format_accuracy_text(accuracy_payload: dict[str, Any] | None) -> str:
            """Formatea aciertos como 'ok/evaluados (xx.x%)' o N/D."""
            if not isinstance(accuracy_payload, dict):
                return "N/D"
            evaluated = int(accuracy_payload.get("evaluated", 0) or 0)
            correct = int(accuracy_payload.get("correct", 0) or 0)
            value = accuracy_payload.get("value")
            if evaluated <= 0 or not isinstance(value, (int, float)):
                return "N/D"
            return f"{correct}/{evaluated} ({float(value) * 100.0:.1f}%)"

        execution_rows: list[tuple[str, str, str]] = [
            ("Manifiesto", _rel_path(selected_manifest), "OK"),
            ("Schema", schema_exec_name, "OK"),
            ("Modelos", f"{len(queued_models)} (completos={complete_models}, parciales={partial_models}, error={failed_models})", "OK" if failed_models == 0 else "WARN"),
            ("Cobertura global", f"{total_ok}/{total_items} imágenes exportadas", "OK" if total_pending == 0 else "WARN"),
            ("Pendientes globales", str(total_pending), "OK" if total_pending == 0 else "WARN"),
        ]
        if primary_output:
            execution_rows.append(("Salida JSONL", _rel_path(primary_output), "OK"))
            summary_payload = cast(dict[str, Any], aggregate_summaries.get(primary_output) or {})
            models_payload = cast(dict[str, Any], summary_payload.get("models") or {})
            for model_name, model_block in models_payload.items():
                if not isinstance(model_block, dict):
                    continue
                aggregate_models_accuracy[str(model_name)] = _format_accuracy_text(
                    cast(dict[str, Any], model_block.get("accuracy") or {})
                )

            global_summary = cast(dict[str, Any], summary_payload.get("global") or {})
            if isinstance(global_summary, dict):
                global_metrics = cast(dict[str, Any], global_summary.get("metrics") or {})
                global_accuracy = _format_accuracy_text(cast(dict[str, Any], global_summary.get("accuracy") or {}))
                duration_stats = cast(dict[str, Any], global_metrics.get("duration_seconds") or {})
                ttft_stats = cast(dict[str, Any], global_metrics.get("ttft_seconds") or {})
                tps_stats = cast(dict[str, Any], global_metrics.get("tokens_per_second") or {})
                execution_rows.append(("Acierto global", global_accuracy, "OK"))
                execution_rows.append(
                    (
                        "Duración global",
                        (
                            f"min={_format_metric_value(duration_stats.get('min'), suffix=' s')} | "
                            f"media={_format_metric_value(duration_stats.get('avg'), suffix=' s')} | "
                            f"max={_format_metric_value(duration_stats.get('max'), suffix=' s')}"
                        ),
                        "OK",
                    )
                )
                execution_rows.append(
                    (
                        "TTFT global",
                        (
                            f"min={_format_metric_value(ttft_stats.get('min'), suffix=' s')} | "
                            f"media={_format_metric_value(ttft_stats.get('avg'), suffix=' s')} | "
                            f"max={_format_metric_value(ttft_stats.get('max'), suffix=' s')}"
                        ),
                        "OK",
                    )
                )
                execution_rows.append(
                    (
                        "TPS global",
                        (
                            f"min={_format_metric_value(tps_stats.get('min'))} | "
                            f"media={_format_metric_value(tps_stats.get('avg'))} | "
                            f"max={_format_metric_value(tps_stats.get('max'))}"
                        ),
                        "OK",
                    )
                )

        execution_lines = [f"  {name:<18} : {value}" for name, value, _status in execution_rows]

        ui_width = max(80, kit.width())
        model_w = max(16, min(30, int(ui_width * 0.30)))
        state_w = 10
        progress_w = 8
        acc_w = 15
        pending_w = 6
        jsonl_w = max(16, ui_width - (model_w + state_w + progress_w + acc_w + pending_w + 25))

        header = (
            f"  {'Modelo':<{model_w}} │ "
            f"{'Estado':<{state_w}} │ "
            f"{'OK/TOT':>{progress_w}} │ "
            f"{'ACC':>{acc_w}} │ "
            f"{'Pend':>{pending_w}} │ "
            f"{'JSONL':<{jsonl_w}}"
        )
        divider = (
            f"  {'─' * model_w}─┼─"
            f"{'─' * state_w}─┼─"
            f"{'─' * progress_w}─┼─"
            f"{'─' * acc_w}─┼─"
            f"{'─' * pending_w}─┼─"
            f"{'─' * jsonl_w}"
        )

        model_table_lines: list[str] = [header, divider]
        colored_state_lines: list[str] = []
        for model_name, _initial_snapshot, final_snapshot, output_path in queue_results:
            status_plain = _status_text(final_snapshot)
            ok_value = int(final_snapshot.get("ok", 0) or 0)
            total_value = int(final_snapshot.get("total", 0) or 0)
            pending_value = int(final_snapshot.get("pending", 0) or 0)
            accuracy_text = aggregate_models_accuracy.get(model_name, "N/D")
            jsonl_text = _rel_path(str(final_snapshot.get("output_path") or output_path or ""))
            colored_model_name = _colorize_model(model_name, final_snapshot)
            colored_state_lines.append(f"  - {colored_model_name} · {status_plain}")
            model_table_lines.append(
                f"  {_truncate(model_name, model_w):<{model_w}} │ "
                f"{status_plain:<{state_w}} │ "
                f"{f'{ok_value}/{total_value}':>{progress_w}} │ "
                f"{_truncate(accuracy_text, acc_w):>{acc_w}} │ "
                f"{str(pending_value):>{pending_w}} │ "
                f"{_truncate(jsonl_text, jsonl_w):<{jsonl_w}}"
            )

        if len(unique_outputs) > 1:
            model_table_lines.append("")
            model_table_lines.append("  Rutas de salida detectadas:")
            for path_item in unique_outputs:
                model_table_lines.append(f"  - {_rel_path(path_item)}")

        _render_final_sections_screen(
            kit,
            app,
            subtitle=f"BATCH RUNNER · {schema_exec_name}",
            intro=_standard_final_intro(),
            sections=[
                ("Resumen de ejecución", execution_lines),
                ("Estado por modelo (colores)", colored_state_lines),
                ("Resultados por modelo", model_table_lines),
            ],
        )
        kit.wait("Press any key to return to tests manager...")
        return
