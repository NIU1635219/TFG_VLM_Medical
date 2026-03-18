from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, cast

from ..setup_ui_io import ask_choice
from .shared import (
    ReactiveTerminalRenderer,
    as_yes,
    build_live_status_line,
    colorize_model,
    compact_model_label,
    is_model_snapshot_complete,
    normalize_model_variants,
    normalize_state_value,
    rel_path,
    snapshot_status_text,
    snapshot_summary_line,
    truncate_middle,
    variant_label,
)
from .test_dashboards_ui import (
    _append_recent_record,
    _build_partial_metrics_line,
    _build_progress_bar,
    _coerce_int,
    _format_metric_value,
    _make_live_panel,
    _make_recent_records,
    _rel_probe_path,
    _render_final_sections_screen,
    _render_live_dashboard,
    _should_show_progress_bar,
    _standard_final_intro,
)

if TYPE_CHECKING:
    from ..menu_kit import AppContext, UIKit


@dataclass(frozen=True)
class _BatchQueueResultContext:
    """
    Resultado de ejecución por variante para resumen final.
    
    Args:
        model_label (str): Etiqueta del modelo.
        model_id (str): ID del modelo.
        include_reasoning (bool): Incluir razonamiento en la etiqueta.
        initial_snapshot (dict[str, Any]): Snapshot inicial del modelo.
        final_snapshot (dict[str, Any]): Snapshot final del modelo.
        output_path (str): Ruta del archivo de salida.
    """

    model_label: str
    model_id: str
    include_reasoning: bool
    initial_snapshot: dict[str, Any]
    final_snapshot: dict[str, Any]
    output_path: str


@dataclass(frozen=True)
class _BatchFinalRenderContext:
    """
    Contexto compacto para render de pantalla final de batch.
    
    Args:
        selected_manifest (str): Manifest seleccionado.
        summary_schema_base_name (str): Nombre base del esquema.
        schema_exec_name (str): Nombre de ejecución del esquema.
        selected_model_ids (list[str]): Lista de IDs de modelos seleccionados.
        queue_results (list[_BatchQueueResultContext]): Resultados de ejecución por variante.
    """

    selected_manifest: str
    summary_schema_base_name: str
    schema_exec_name: str
    selected_model_ids: list[str]
    queue_results: list[_BatchQueueResultContext]


@dataclass(frozen=True)
class _BatchModelSelectionContext:
    """
    Contexto para resolver qué variantes ejecutar.
    
    Args:
        schema_exec_name (str): Nombre de ejecución del esquema.
        queued_models (list[str]): Lista de modelos en cola.
        completed_models (list[str]): Lista de modelos completados.
        snapshot_for (Callable[[str], dict[str, Any]]): Función para obtener el snapshot del modelo.
        default_models (set[str]): Conjunto de modelos por defecto.
        display_label_for (Callable[[str], str]): Función para obtener la etiqueta del modelo.
    """

    schema_exec_name: str
    queued_models: list[str]
    completed_models: list[str]
    snapshot_for: Callable[[str], dict[str, Any]]
    default_models: set[str]
    display_label_for: Callable[[str], str]


@dataclass(frozen=True)
class _BatchProgressBarsContext:
    """
    Contexto para construir líneas de barras de progreso extra.
    
    Args:
        show_model_queue_bar (bool): Mostrar barra de cola de modelos.
        model_queue_current (int): Contador actual de la cola de modelos.
        models_target (int): Objetivo de la cola de modelos.
        compact_model (str): Modelo compacto.
        model_completed_total (int): Total de modelos completados.
        current_model_target (int): Objetivo actual del modelo.
        iteration_progress (int): Progreso de la iteración.
        iterations_value (int): Valor de las iteraciones.
    """

    show_model_queue_bar: bool
    model_queue_current: int
    models_target: int
    compact_model: str
    model_completed_total: int
    current_model_target: int
    iteration_progress: int
    iterations_value: int


@dataclass(frozen=True)
class _BatchExecutionCatalog:
    """
    Índices por execution_id para evitar pasar múltiples dicts sueltos.
    
    Args:
        queued_models (list[str]): Lista de modelos en cola.
        label_by_id (dict[str, str]): Diccionario de etiquetas por ID.
        schema_by_id (dict[str, str]): Diccionario de esquemas por ID.
        model_id_by_id (dict[str, str]): Diccionario de IDs por ID.
        include_reasoning_by_id (dict[str, bool]): Diccionario de razonamiento por ID.
    """

    queued_models: list[str]
    label_by_id: dict[str, str]
    schema_by_id: dict[str, str]
    model_id_by_id: dict[str, str]
    include_reasoning_by_id: dict[str, bool]


def _select_models_to_execute(
    *,
    kit: "UIKit",
    app: "AppContext",
    context: _BatchModelSelectionContext,
) -> set[str]:
    """
    Resuelve la lista final de modelos a ejecutar incluyendo reejecuciones.
    
    Args:
        kit ("UIKit"): Kit de UI.
        app ("AppContext"): Contexto de la aplicación.
        context (_BatchModelSelectionContext): Contexto de selección de modelos.
        
    Returns:
        set[str]: Conjunto de modelos a ejecutar.
    """
    schema_exec_name = context.schema_exec_name
    queued_models = context.queued_models
    completed_models = context.completed_models
    snapshot_for = context.snapshot_for
    display_label_for = context.display_label_for

    models_to_execute = set(context.default_models)
    if not completed_models:
        return models_to_execute

    # Si hay trabajo pendiente/parcial, priorizamos retomar automáticamente sin menú.
    if models_to_execute:
        kit.log(
            "Manifiesto incompleto detectado: se retoman automáticamente pendientes/parciales.",
            "info",
        )
        return models_to_execute

    def _rerun_info_text() -> str:
        """
        Obtiene el texto informativo para la reejecución.
        
        Returns:
            str: Texto informativo para la reejecución.
        """
        lines: list[str] = []
        lines.append("Resumen actual por modelo:")
        lines.extend(
            [
                "  - " + snapshot_summary_line(
                    kit.style,
                    display_label_for(model),
                    snapshot_for(model),
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
            """
            Renderiza el encabezado para la reejecución.
            
            Args:
                kit ("UIKit"): Kit de UI.
                app ("AppContext"): Contexto de la aplicación.
                schema_exec_name (str): Nombre de ejecución del esquema.
            """
            kit.clear()
            app.print_banner()
            kit.subtitle(f"BATCH RUNNER · {schema_exec_name} · REEJECUTAR MODELOS")

        model_items = [
            kit.MenuItem(
                display_label_for(model_tag),
                description=(
                    "Estado actual: COMPLETO (se limpiará y reejecutará)."
                    if model_tag in completed_models
                    else "Estado actual: PENDIENTE/PARCIAL (se ejecutará)."
                ),
            )
            for model_tag in queued_models
        ]
        for item, model_tag in zip(model_items, queued_models):
            setattr(item, "_execution_id", model_tag)
        for model_item in model_items:
            model_label = str(getattr(model_item, "_execution_id", "") or "").strip()
            model_snapshot = snapshot_for(model_label)

            def _dynamic_model_label(
                _is_selected_row: bool,
                _label: str = model_label,
                _snapshot: dict[str, Any] = model_snapshot,
            ) -> str:
                """
                Función dinámica para el label del modelo.
                
                Args:
                    _is_selected_row (bool): Indica si la fila está seleccionada.
                    _label (str): Etiqueta del modelo.
                    _snapshot (dict[str, Any]): Snapshot del modelo.
                    
                Returns:
                    str: Label dinámico del modelo.
                """
                return snapshot_summary_line(kit.style, _label, _snapshot)

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
                str(getattr(item, "_execution_id", "") or "").strip()
                for item in selected_models
                if str(getattr(item, "_execution_id", "") or "").strip()
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

    return models_to_execute


def _format_accuracy_text(accuracy_payload: dict[str, Any] | None) -> str:
    """
    Formatea aciertos como 'ok/evaluados (xx.x%)' o N/D.
    
    Args:
        accuracy_payload (dict[str, Any] | None): Payload de aciertos.
        
    Returns:
        str: Texto formateado de aciertos.
    """
    if not isinstance(accuracy_payload, dict):
        return "N/D"
    evaluated = int(accuracy_payload.get("evaluated", 0) or 0)
    correct = int(accuracy_payload.get("correct", 0) or 0)
    value = accuracy_payload.get("value")
    if evaluated <= 0 or not isinstance(value, (int, float)):
        return "N/D"
    return f"{correct}/{evaluated} ({float(value) * 100.0:.1f}%)"


def _collect_queue_rows(
    queue_results: list[_BatchQueueResultContext],
) -> tuple[list[tuple[str, str, bool, dict[str, Any], str]], int, int, int, int, int, int, list[str]]:
    """
    Normaliza filas por modelo y acumula totales globales de la cola.
    
    Args:
        queue_results (list[_BatchQueueResultContext]): Resultados de ejecución por variante.
        
    Returns:
        tuple[list[tuple[str, str, bool, dict[str, Any], str]], int, int, int, int, int, int, list[str]]:
            Tupla con filas de cola, totales de OK, items, pendientes, modelos completos,
            modelos parciales, modelos fallidos y lista de modelos en cola.
    """
    total_ok = 0
    total_items = 0
    total_pending = 0
    complete_models = 0
    partial_models = 0
    failed_models = 0
    queue_rows: list[tuple[str, str, bool, dict[str, Any], str]] = []
    for item in queue_results:
        model_name = item.model_label
        model_id = item.model_id
        include_reasoning = item.include_reasoning
        final_snapshot = item.final_snapshot
        output_path = item.output_path
        resolved_output = str(final_snapshot.get("output_path") or output_path or "")
        queue_rows.append((model_name, model_id, include_reasoning, final_snapshot, resolved_output))
        total_ok += int(final_snapshot.get("ok", 0) or 0)
        total_items += int(final_snapshot.get("total", 0) or 0)
        pending_value = int(final_snapshot.get("pending", 0) or 0)
        total_pending += pending_value
        state = normalize_state_value(final_snapshot.get("status"))
        if state == "green":
            complete_models += 1
        elif state == "yellow":
            partial_models += 1
        else:
            failed_models += 1
    output_paths = [
        resolved_output
        for _model_name, _model_id, _include_reasoning, _final_snapshot, resolved_output in queue_rows
    ]
    return (
        queue_rows,
        total_ok,
        total_items,
        total_pending,
        complete_models,
        partial_models,
        failed_models,
        output_paths,
    )


def _resolve_iteration_progress(
    *,
    event: str,
    current_index: int,
    iterations_per_image: int,
    payload_iteration_index: int,
    payload_iteration_total: int,
) -> tuple[int, int]:
    """
    Resuelve progreso/total de iteraciones de la imagen actual.
    
    Args:
        event (str): Evento que disparó la resolución.
        current_index (int): Índice actual.
        iterations_per_image (int): Número de iteraciones por imagen.
        payload_iteration_index (int): Índice de iteración del payload.
        payload_iteration_total (int): Total de iteraciones del payload.
        
    Returns:
        tuple[int, int]: Tupla con el progreso de la iteración y el valor total de iteraciones.
    """
    iterations_value = max(1, int(iterations_per_image or 1))
    iteration_progress = iterations_value
    if event == "complete":
        return iteration_progress, iterations_value

    if payload_iteration_total > 0:
        iterations_value = payload_iteration_total
    if payload_iteration_index > 0:
        iteration_progress = min(iterations_value, payload_iteration_index)
    else:
        iteration_progress = ((max(1, current_index) - 1) % iterations_value) + 1

    return iteration_progress, iterations_value


def _build_extra_progress_lines(
    *,
    kit: "UIKit",
    context: _BatchProgressBarsContext,
) -> list[str]:
    """
    Compone líneas extra de barras de progreso del dashboard vivo.
    
    Args:
        kit ("UIKit"): Kit de UI.
        context (_BatchProgressBarsContext): Contexto de barras de progreso.
        
    Returns:
        list[str]: Líneas extra de barras de progreso.
    """
    show_model_queue_bar = context.show_model_queue_bar
    model_queue_current = context.model_queue_current
    models_target = context.models_target
    compact_model = context.compact_model
    model_completed_total = context.model_completed_total
    current_model_target = context.current_model_target
    iteration_progress = context.iteration_progress
    iterations_value = context.iterations_value

    model_progress_line: str | None = None
    if _should_show_progress_bar(models_target):
        model_progress_line = (
            f"Cola de modelos: {_build_progress_bar(kit, model_queue_current, models_target)}"
        )

    current_images_line: str | None = None
    if _should_show_progress_bar(current_model_target):
        current_images_line = (
            f"Modelo actual ({compact_model}): {_build_progress_bar(kit, model_completed_total, current_model_target)}"
        )

    iterations_line: str | None = None
    if _should_show_progress_bar(iterations_value):
        iterations_line = (
            f"Iteraciones de imagen actual ({iterations_value}x): "
            f"{_build_progress_bar(kit, iteration_progress, iterations_value)}"
        )

    lines: list[str] = []
    if current_images_line:
        lines.append(current_images_line)
    if iterations_line:
        lines.append(iterations_line)

    if show_model_queue_bar and model_progress_line:
        return [model_progress_line, *lines]
    return lines


def _resolve_pending_entries_and_paths(snapshot: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Extrae entradas pendientes y su lista de paths limpios desde snapshot.
    
    Args:
        snapshot (dict[str, Any]): Snapshot del modelo.
        
    Returns:
        tuple[list[dict[str, Any]], list[str]]: Tupla con entradas pendientes y lista de paths limpios.
    """
    pending_entries = cast(list[dict[str, Any]], snapshot.get("pending_entries") or [])
    pending_paths = [
        str(item.get("image_path") or "").strip()
        for item in pending_entries
        if str(item.get("image_path") or "").strip()
    ]
    return pending_entries, pending_paths


def _retry_made_progress(*, pending_before_retry: int, pending_after_retry: int) -> bool:
    """
    Determina si un reintento redujo pendientes y por tanto hubo progreso.
    
    Args:
        pending_before_retry (int): Número de pendientes antes del reintento.
        pending_after_retry (int): Número de pendientes después del reintento.
        
    Returns:
        bool: True si hubo progreso (menos pendientes después), False en caso contrario.
    """
    return pending_after_retry < pending_before_retry


def _build_output_path_by_model(
    *,
    catalog: _BatchExecutionCatalog,
    selected_manifest: str,
    linked_batch_output_path: Callable[..., Any],
) -> dict[str, str]:
    """
    Precalcula la ruta de salida compartida por cada modelo.
    
    Args:
        catalog (_BatchExecutionCatalog): Catálogo de ejecución.
        selected_manifest (str): Manifest seleccionado.
        linked_batch_output_path (Callable[..., Any]): Función para obtener la ruta de salida.
        
    Returns:
        dict[str, str]: Diccionario con la ruta de salida por modelo.
    """
    return {
        model: str(
            linked_batch_output_path(
                manifest_path=selected_manifest,
                schema_name=catalog.schema_by_id[model],
            )
        )
        for model in catalog.queued_models
    }


def _build_snapshot_kwargs_by_model(
    *,
    catalog: _BatchExecutionCatalog,
    selected_manifest: str,
    schema_name_by_model: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """
    Precalcula kwargs de snapshot por modelo para evitar duplicaciones.
    
    Args:
        catalog (_BatchExecutionCatalog): Catálogo de ejecución.
        selected_manifest (str): Manifest seleccionado.
        schema_name_by_model (dict[str, str]): Diccionario de nombres de esquema por modelo.
        
    Returns:
        dict[str, dict[str, Any]]: Diccionario con kwargs de snapshot por modelo.
    """
    return {
        model: {
            "manifest_path": selected_manifest,
            "model_tag": catalog.model_id_by_id[model],
            "schema_name": schema_name_by_model[model],
            "include_reasoning": catalog.include_reasoning_by_id[model],
        }
        for model in catalog.queued_models
    }


def _build_execution_catalog(execution_variants: list[dict[str, Any]]) -> _BatchExecutionCatalog:
    """
    Construye catálogo de ejecución y sus índices por id.
    
    Args:
        execution_variants (list[dict[str, Any]]): Lista de variantes de ejecución.
        
    Returns:
        _BatchExecutionCatalog: Catálogo de ejecución.
    """
    queued_models = [str(item.get("execution_id") or "") for item in execution_variants]
    return _BatchExecutionCatalog(
        queued_models=queued_models,
        label_by_id={
            str(item.get("execution_id") or ""): str(item.get("label") or "")
            for item in execution_variants
        },
        schema_by_id={
            str(item.get("execution_id") or ""): str(item.get("schema_exec_name") or "")
            for item in execution_variants
        },
        model_id_by_id={
            str(item.get("execution_id") or ""): str(item.get("model_id") or "")
            for item in execution_variants
        },
        include_reasoning_by_id={
            str(item.get("execution_id") or ""): bool(item.get("include_reasoning"))
            for item in execution_variants
        },
    )


def _build_queue_targets(
    *,
    runnable_models: list[str],
    initial_snapshots: dict[str, dict[str, Any]],
) -> dict[str, int]:
    """
    Calcula objetivo de imágenes por modelo runnable desde snapshot inicial.
    
    Args:
        runnable_models (list[str]): Lista de modelos ejecutables.
        initial_snapshots (dict[str, dict[str, Any]]): Snapshot inicial por modelo.
        
    Returns:
        dict[str, int]: Diccionario con el objetivo de imágenes por modelo.
    """
    return {
        model: int(cast(dict[str, Any], initial_snapshots.get(model) or {}).get("total", 0))
        for model in runnable_models
    }


@dataclass(frozen=True)
class _BatchModelExecutionContext:
    """
    Contexto inmutable para ejecutar una variante de modelo con menos parámetros.
    
    Args:
        selected_manifest (str): Manifest seleccionado.
        schema_exec_name (str): Nombre de ejecución del esquema.
        schema_base_name (str): Nombre base del esquema.
        include_reasoning (bool): Indica si se debe incluir razonamiento.
        model_id (str): ID del modelo.
        model_label (str): Etiqueta del modelo.
        output_path (str): Ruta de salida.
        manifest_snapshot (dict[str, Any]): Snapshot del manifiesto.
        rerun_only_pending (bool): Indica si se debe ejecutar solo pendientes.
        completed_global_images_before_model (int): Imágenes completadas globalmente antes del modelo.
        total_global_target (int): Objetivo global de imágenes.
        current_model_target (int): Objetivo actual del modelo.
        models_done_before (int): Modelos completados antes.
        models_target (int): Objetivo de modelos.
        iterations_per_image (int): Iteraciones por imagen.
        baseline_completed_for_model (int): Completados de baseline para el modelo.
    """

    selected_manifest: str
    schema_exec_name: str
    schema_base_name: str
    include_reasoning: bool
    model_id: str
    model_label: str
    output_path: str
    manifest_snapshot: dict[str, Any]
    rerun_only_pending: bool
    completed_global_images_before_model: int
    total_global_target: int
    current_model_target: int
    models_done_before: int
    models_target: int
    iterations_per_image: int
    baseline_completed_for_model: int


def _render_batch_final_summary(
    *,
    kit: "UIKit",
    app: "AppContext",
    context: _BatchFinalRenderContext,
    upsert_batch_execution_summary: Callable[..., dict[str, Any] | None],
) -> Callable[[int], None]:
    """
    Construye el renderer del dashboard final del batch runner y realiza un primer render.
    
    Args:
        kit ("UIKit"): Kit de UI.
        app ("AppContext"): Contexto de la aplicación.
        context (_BatchFinalRenderContext): Contexto de renderizado final.
        upsert_batch_execution_summary (Callable[..., dict[str, Any] | None]): Función para actualizar el resumen del batch.
    Returns:
        Callable[[int], None]: Callback de re-render para cambios de ancho.
    """
    def _format_metric_triplet(stats: dict[str, Any], *, suffix: str = "") -> str:
        """
        Formatea min/media/max para una métrica agregada.
        
        Args:
            stats (dict[str, Any]): Estadísticas de la métrica.
            suffix (str): Sufijo a añadir al valor.
            
        Returns:
            str: Texto formateado con min/media/max.
        """
        return (
            f"min={_format_metric_value(stats.get('min'), suffix=suffix)} | "
            f"media={_format_metric_value(stats.get('avg'), suffix=suffix)} | "
            f"max={_format_metric_value(stats.get('max'), suffix=suffix)}"
        )

    queue_results = context.queue_results
    selected_manifest = context.selected_manifest
    summary_schema_base_name = context.summary_schema_base_name
    schema_exec_name = context.schema_exec_name
    selected_model_ids = context.selected_model_ids

    (
        queue_rows,
        total_ok,
        total_items,
        total_pending,
        complete_models,
        partial_models,
        failed_models,
        output_paths,
    ) = _collect_queue_rows(queue_results)

    unique_outputs = sorted({path for path in output_paths if path})
    aggregate_summaries: dict[str, dict[str, Any]] = {}
    for output_item in unique_outputs:
        aggregate_summary = upsert_batch_execution_summary(
            output_path=output_item,
            schema_name=summary_schema_base_name,
            model_ids=selected_model_ids,
        )
        if aggregate_summary:
            aggregate_summaries[output_item] = aggregate_summary
    primary_output = unique_outputs[0] if unique_outputs else ""
    aggregate_models_accuracy: dict[str, str] = {}
    global_warn = "OK" if total_pending == 0 else "WARN"
    model_count_text = (
        f"{len(queue_rows)} variantes (modelos base={len(selected_model_ids)}, "
        f"completos={complete_models}, parciales={partial_models}, error={failed_models})"
    )

    execution_rows: list[tuple[str, str, str]] = [
        ("Manifiesto", rel_path(selected_manifest), "OK"),
        ("Schema", schema_exec_name, "OK"),
        ("Modelos", model_count_text, "OK" if failed_models == 0 else "WARN"),
        ("Cobertura global", f"{total_ok}/{total_items} imágenes exportadas", global_warn),
        ("Pendientes globales", str(total_pending), global_warn),
    ]
    if primary_output:
        execution_rows.append(("Salida JSONL", rel_path(primary_output), "OK"))
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
            execution_rows.append(("Duración global", _format_metric_triplet(duration_stats, suffix=" s"), "OK"))
            execution_rows.append(("TTFT global", _format_metric_triplet(ttft_stats, suffix=" s"), "OK"))
            execution_rows.append(("TPS global", _format_metric_triplet(tps_stats), "OK"))

    execution_lines = [f"  {name:<18} : {value}" for name, value, _status in execution_rows]

    colored_state_lines: list[str] = []
    for model_name, _model_id, _include_reasoning, final_snapshot, _resolved_output in queue_rows:
        status_plain = snapshot_status_text(final_snapshot)
        colored_model_name = colorize_model(kit.style, model_name, final_snapshot)
        colored_state_lines.append(f"  - {colored_model_name} · {status_plain}")

    def _redraw_final_batch_screen(ui_width: int) -> None:
        """Re-renderiza la pantalla final del batch runner para el ancho indicado."""
        safe_width = max(80, int(ui_width))
        model_w = max(14, min(24, int(safe_width * 0.24)))
        variant_w = 16
        state_w = 10
        progress_w = 8
        acc_w = 15
        pending_w = 6
        table_padding = 28
        jsonl_w = max(
            14,
            safe_width - (model_w + variant_w + state_w + progress_w + acc_w + pending_w + table_padding),
        )

        header = (
            f"  {'Modelo':<{model_w}} │ "
            f"{'Variante':<{variant_w}} │ "
            f"{'Estado':<{state_w}} │ "
            f"{'OK/TOT':>{progress_w}} │ "
            f"{'ACC':>{acc_w}} │ "
            f"{'Pend':>{pending_w}} │ "
            f"{'JSONL':<{jsonl_w}}"
        )
        divider = (
            f"  {'─' * model_w}─┼─"
            f"{'─' * variant_w}─┼─"
            f"{'─' * state_w}─┼─"
            f"{'─' * progress_w}─┼─"
            f"{'─' * acc_w}─┼─"
            f"{'─' * pending_w}─┼─"
            f"{'─' * jsonl_w}"
        )

        model_table_lines: list[str] = [header, divider]
        for model_name, model_id, include_reasoning, final_snapshot, resolved_output in queue_rows:
            status_plain = snapshot_status_text(final_snapshot)
            ok_value = int(final_snapshot.get("ok", 0) or 0)
            total_value = int(final_snapshot.get("total", 0) or 0)
            pending_value = int(final_snapshot.get("pending", 0) or 0)
            variant_name = variant_label(model_id, include_reasoning)
            variant_mode = "con razonamiento" if include_reasoning else "sin razonamiento"
            accuracy_text = aggregate_models_accuracy.get(variant_name, "N/D")
            jsonl_text = rel_path(resolved_output)
            model_table_lines.append(
                f"  {truncate_middle(model_id, model_w):<{model_w}} │ "
                f"{truncate_middle(variant_mode, variant_w):<{variant_w}} │ "
                f"{status_plain:<{state_w}} │ "
                f"{f'{ok_value}/{total_value}':>{progress_w}} │ "
                f"{truncate_middle(accuracy_text, acc_w):>{acc_w}} │ "
                f"{str(pending_value):>{pending_w}} │ "
                f"{truncate_middle(jsonl_text, jsonl_w):<{jsonl_w}}"
            )

        if len(unique_outputs) > 1:
            model_table_lines.append("")
            model_table_lines.append("  Rutas de salida detectadas:")
            for path_item in unique_outputs:
                model_table_lines.append(f"  - {rel_path(path_item)}")

        _render_final_sections_screen(
            kit,
            app,
            subtitle=f"BATCH RUNNER · {schema_exec_name}",
            intro=_standard_final_intro(),
            ui_width=kit.width(),
            sections=[
                ("Resumen de ejecución", execution_lines),
                ("Estado por modelo (colores)", colored_state_lines),
                ("Resultados por modelo", model_table_lines),
            ],
        )

    _redraw_final_batch_screen(kit.width())
    return _redraw_final_batch_screen


def _execute_model_with_retries(
    *,
    kit: "UIKit",
    app: "AppContext",
    run_batch_job: Callable[..., dict[str, Any]],
    manifest_execution_snapshot: Callable[..., dict[str, Any]],
    execution: _BatchModelExecutionContext,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """
    Ejecuta un modelo de la cola con progreso en vivo y reintentos de pendientes.
    
    Args:
        kit ("UIKit"): Kit de UI.
        app ("AppContext"): Contexto de la aplicación.
        run_batch_job (Callable[..., dict[str, Any]]): Función para ejecutar el batch.
        manifest_execution_snapshot (Callable[..., dict[str, Any]]): Función para obtener el snapshot del manifiesto.
        execution (_BatchModelExecutionContext): Contexto de ejecución del modelo.
        
    Returns:
        tuple[dict[str, Any], dict[str, Any], bool]: Tupla con el resumen del modelo, el snapshot final y un flag de progreso.
    """
    model_label = execution.model_label.strip() or execution.model_id
    kit.clear()
    app.print_banner()
    kit.subtitle(f"BATCH RUNNER · {execution.schema_exec_name} · {model_label}")
    kit.log(
        f"Exportando resultados incrementales en JSONL desde manifiesto: {execution.selected_manifest}",
        "step",
    )

    summary: dict[str, Any] = {
        "ok": 0,
        "invalid": 0,
        "fail": 0,
        "output_path": execution.output_path,
    }
    show_model_queue_bar = execution.models_target > 1
    compact_model = compact_model_label(model_label, ui_width=kit.width())
    snapshot_kwargs = {
        "manifest_path": execution.selected_manifest,
        "model_tag": execution.model_id,
        "schema_name": execution.schema_base_name,
        "include_reasoning": execution.include_reasoning,
    }

    try:
        recent_records = _make_recent_records(limit=2)
        panel = _make_live_panel(
            kit,
            app,
            subtitle=f"BATCH RUNNER · {execution.schema_exec_name} · {model_label}",
            intro=f"Exportando resultados incrementales en JSONL desde manifiesto: {execution.selected_manifest}",
        )
        global_target_default = (
            execution.total_global_target
            if execution.total_global_target > 0
            else execution.current_model_target
        )
        live_progress_summary: dict[str, object] = {
            "ok": 0,
            "invalid": 0,
            "fail": 0,
            "processed": 0,
        }
        live_state: dict[str, int | str] = {
            "global_target": global_target_default,
            "global_completed": min(
                global_target_default,
                execution.completed_global_images_before_model + execution.baseline_completed_for_model,
            ),
            "model_queue_current": execution.models_done_before,
            "model_completed_total": execution.baseline_completed_for_model,
            "iteration_progress": 1,
            "iterations_value": max(1, execution.iterations_per_image),
            "total_current": execution.current_model_target,
            "status_line": f"Estado actual: preparando ejecución · muestra {execution.current_model_target}",
        }

        def _render_live_batch_dashboard(*, force_full: bool | None = None) -> None:
            """Renderiza el dashboard de batch con el último estado incremental."""
            invalid_count_live = _coerce_int(live_progress_summary.get("invalid"))
            fail_count_live = _coerce_int(live_progress_summary.get("fail"))
            _render_live_dashboard(
                kit,
                panel,
                current=_coerce_int(live_state.get("global_completed")),
                total=_coerce_int(live_state.get("global_target")),
                stats_line=(
                    f"Resumen del modelo actual: OK={live_progress_summary.get('ok', 0)} | "
                    f"Inválidas={invalid_count_live} | Errores={fail_count_live}"
                ),
                status_line=str(live_state.get("status_line") or ""),
                metrics_line=_build_partial_metrics_line(live_progress_summary),
                extra_lines=_build_extra_progress_lines(
                    kit=kit,
                    context=_BatchProgressBarsContext(
                        show_model_queue_bar=show_model_queue_bar,
                        model_queue_current=_coerce_int(live_state.get("model_queue_current")),
                        models_target=execution.models_target,
                        compact_model=compact_model,
                        model_completed_total=_coerce_int(live_state.get("model_completed_total")),
                        current_model_target=execution.current_model_target,
                        iteration_progress=_coerce_int(live_state.get("iteration_progress")),
                        iterations_value=max(1, _coerce_int(live_state.get("iterations_value"))),
                    ),
                ),
                recent_title="Últimos registros exportados:",
                recent_records=list(recent_records),
                force_full=force_full,
            )

        live_renderer = ReactiveTerminalRenderer(
            kit=kit,
            render_fn=_render_live_batch_dashboard,
        )
        base_batch_kwargs: dict[str, Any] = {
            "model_id": execution.model_id,
            "manifest": execution.selected_manifest,
            "schema_name": execution.schema_base_name,
            "include_reasoning": execution.include_reasoning,
            "output_path": execution.output_path,
            "max_images": None,
            # Respetar orden del manifiesto para ejecutar iteraciones por imagen en bloque.
            "shuffle": False,
        }

        def on_batch_progress(payload: dict[str, object]) -> None:
            """
            Renderiza progreso en vivo durante el batch actual.
            
            Args:
                payload (dict[str, object]): Payload con el progreso del batch.
            """
            summary_progress = cast(dict[str, object], payload.get("summary") or {})
            live_progress_summary.clear()
            live_progress_summary.update(summary_progress)
            total = _coerce_int(payload.get("total"))
            event = str(payload.get("event") or "")
            current_index = _coerce_int(payload.get("index"))
            image_path = _rel_probe_path(cast(str | None, payload.get("image_path")))
            status_value = str(payload.get("status") or "")
            record_payload = payload.get("record")
            last_record = record_payload if isinstance(record_payload, dict) else None
            completed = _coerce_int(summary_progress.get("processed"))
            global_target = (
                execution.total_global_target
                if execution.total_global_target > 0
                else execution.current_model_target
            )
            model_completed_total = min(
                execution.current_model_target,
                execution.baseline_completed_for_model + completed,
            )
            global_completed = min(
                global_target,
                execution.completed_global_images_before_model + model_completed_total,
            )

            model_queue_current = execution.models_done_before
            if event == "complete" and execution.models_target > 0:
                model_queue_current = min(execution.models_target, execution.models_done_before + 1)

            payload_iteration_index = _coerce_int(payload.get("run_iteration_index"))
            payload_iteration_total = _coerce_int(payload.get("run_iteration_total"))
            iteration_progress, iterations_value = _resolve_iteration_progress(
                event=event,
                current_index=current_index,
                iterations_per_image=execution.iterations_per_image,
                payload_iteration_index=payload_iteration_index,
                payload_iteration_total=payload_iteration_total,
            )

            status_line = build_live_status_line(
                event=event,
                current_index=current_index,
                total=total,
                item_label=image_path,
                status=status_value,
                completed=completed,
                on_start="Estado actual: procesando imagen {current_index}/{total} · {item}",
                on_done_ok="Estado actual: exportada imagen {current_index}/{total} · {item}",
                on_done_invalid="Estado actual: respuesta inválida en {current_index}/{total} · {item}",
                on_complete="Estado actual: finalizado · {completed}/{total} imágenes exportadas",
                on_prepare="Estado actual: preparando ejecución · muestra {total}",
            )

            if event == "image_done" and last_record is not None:
                _append_recent_record(recent_records, last_record)

            live_state.update(
                {
                    "global_target": global_target,
                    "global_completed": global_completed,
                    "model_queue_current": model_queue_current,
                    "model_completed_total": model_completed_total,
                    "iteration_progress": iteration_progress,
                    "iterations_value": iterations_value,
                    "total_current": total,
                    "status_line": status_line,
                }
            )

            live_renderer.render()

        def execute_batch_once(
            *,
            pending_image_paths: list[str] | None = None,
            pending_entries: list[dict[str, Any]] | None = None,
        ) -> dict[str, Any]:
            """
            Ejecuta una iteración de batch (completa o reanudada).
            
            Args:
                pending_image_paths (list[str] | None): Lista de paths de imágenes pendientes.
                pending_entries (list[dict[str, Any]] | None): Lista de entradas pendientes.
                
            Returns:
                dict[str, Any]: Resumen del batch.
            """
            try:
                live_renderer.start()
                return run_batch_job(
                    **base_batch_kwargs,
                    pending_image_paths=pending_image_paths,
                    pending_entries=pending_entries,
                    on_progress=on_batch_progress,
                )
            except TypeError as error:
                if "on_progress" not in str(error):
                    raise
                return run_batch_job(
                    **base_batch_kwargs,
                    pending_image_paths=pending_image_paths,
                    pending_entries=pending_entries,
                )
            finally:
                live_renderer.stop()

        pending_image_paths: list[str] | None = None
        pending_entries: list[dict[str, Any]] | None = None
        if execution.rerun_only_pending:
            pending_entries, pending_image_paths = _resolve_pending_entries_and_paths(execution.manifest_snapshot)

        summary = execute_batch_once(
            pending_image_paths=pending_image_paths,
            pending_entries=pending_entries,
        )

        while True:
            refreshed = manifest_execution_snapshot(**snapshot_kwargs)
            pending_before_retry = int(refreshed.get("pending", 0))
            if pending_before_retry <= 0:
                break
            retry_now = as_yes(
                kit.ask(
                    (
                        f"[{execution.model_id}] Quedan imágenes pendientes o con error. "
                        "¿Quieres reejecutarlas ahora sin salir del menú?"
                    ),
                    "y",
                )
            )
            if not retry_now:
                break
            refreshed_entries, retry_pending_paths = _resolve_pending_entries_and_paths(refreshed)
            if not retry_pending_paths:
                break
            summary = execute_batch_once(
                pending_image_paths=retry_pending_paths,
                pending_entries=refreshed_entries,
            )

            refreshed_after_retry = manifest_execution_snapshot(**snapshot_kwargs)
            pending_after_retry = int(refreshed_after_retry.get("pending", 0))
            if not _retry_made_progress(
                pending_before_retry=pending_before_retry,
                pending_after_retry=pending_after_retry,
            ):
                kit.log(
                    (
                        f"[{execution.model_id}] Reintento sin progreso "
                        f"({pending_after_retry} pendientes). Se detiene para evitar bucle."
                    ),
                    "warning",
                )
                break

        final_snapshot = manifest_execution_snapshot(**snapshot_kwargs)
        return final_snapshot, summary, True
    except Exception as error:
        final_snapshot = manifest_execution_snapshot(**snapshot_kwargs)
        kit.log(f"[{model_label}] Batch Runner terminó con error: {error}", "error")
        return final_snapshot, summary, False


def run_batch_runner_wrapper(
    kit: "UIKit",
    app: "AppContext",
    *,
    select_manifest_for_batch: Callable[[], dict[str, Any] | None],
    execution_schema_name: Callable[[str, bool], str],
    linked_batch_output_path: Callable[..., Any],
    manifest_execution_snapshot: Callable[..., dict[str, Any]],
    prune_output_records_for_model: Callable[..., bool],
) -> None:
    """Ejecuta trabajos de inferencia por lotes en cola definidos por manifiestos autosuficientes.

    Args:
        kit: Toolkit de interfaz de usuario de terminal.
        app: Contexto de la aplicación utilizado para renderizar el cromo de la UI compartida.
        select_manifest_for_batch: Callback que selecciona o crea el paquete del manifiesto.
        execution_schema_name: Callback que deriva el nombre de visualización del esquema de ejecución.
        linked_batch_output_path: Callback que calcula la ruta JSONL de salida.
        manifest_execution_snapshot: Callback que calcula el estado de ejecución actual.
        prune_output_records_for_model: Callback que elimina filas para un modelo en el JSONL compartido.
    """
    from src.scripts.batch_runner import run_batch_job, seed_batch_meta_header, upsert_batch_execution_summary

    initial_snapshots: dict[str, dict[str, Any]] = {}
    execution_label_by_id: dict[str, str] = {}

    def _snapshot_for(execution_id: str) -> dict[str, Any]:
        """
        Obtiene snapshot inicial por ejecución (modelo + modo).
        
        Args:
            execution_id (str): ID de ejecución.
            
        Returns:
            dict[str, Any]: Snapshot inicial.
        """
        return cast(dict[str, Any], initial_snapshots.get(execution_id) or {})

    def _label_for(execution_id: str) -> str:
        """
        Resuelve etiqueta de ejecución para mostrar en menús y logs.
        
        Args:
            execution_id (str): ID de ejecución.
            
        Returns:
            str: Etiqueta de ejecución.
        """
        return str(execution_label_by_id.get(execution_id) or execution_id)

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
        iterations_per_image = int(manifest_config.get("iterations_per_image", 1) or 1)
        model_variants_raw = cast(list[dict[str, Any]], manifest_config.get("model_variants") or [])
        normalized_variants = normalize_model_variants(model_variants_raw)

        execution_variants: list[dict[str, Any]] = []
        for raw_variant in normalized_variants:
            model_id = str(raw_variant.get("model_id") or "").strip()
            include_reasoning = bool(raw_variant.get("include_reasoning"))
            schema_exec_name = execution_schema_name(schema_base_name, include_reasoning)
            execution_id = f"{model_id}|{'r' if include_reasoning else 'n'}"
            execution_variants.append(
                {
                    "execution_id": execution_id,
                    "model_id": model_id,
                    "include_reasoning": include_reasoning,
                    "schema_exec_name": schema_exec_name,
                    "label": variant_label(model_id, include_reasoning),
                }
            )

        if not schema_base_name or not execution_variants:
            kit.log("Configuración incompleta en manifiesto (faltan schema/modelos).", "error")
            kit.wait("Press any key to return to tests manager...")
            continue

        queue_schema_name = schema_base_name

        catalog = _build_execution_catalog(execution_variants)
        queued_models = catalog.queued_models
        execution_label_by_id = catalog.label_by_id
        schema_name_by_execution = catalog.schema_by_id
        model_id_by_execution = catalog.model_id_by_id
        include_reasoning_by_execution = catalog.include_reasoning_by_id

        kit.clear()
        app.print_banner()
        kit.subtitle(f"BATCH RUNNER · {queue_schema_name}")
        kit.log(f"Manifiesto seleccionado: {selected_manifest}", "step")
        kit.log(cast(str, manifest_overview.get("description") or "Sin descripción disponible."), "info")
        kit.log("Ejecuciones en cola: " + " | ".join(_label_for(item) for item in queued_models), "info")

        output_to_models: dict[str, set[str]] = {}
        output_to_schemas: dict[str, set[str]] = {}
        for execution_id in queued_models:
            model_id = model_id_by_execution[execution_id]
            schema_exec_name = schema_name_by_execution[execution_id]
            output_path = str(
                linked_batch_output_path(
                    manifest_path=selected_manifest,
                    schema_name=schema_exec_name,
                )
            )
            output_to_models.setdefault(output_path, set()).add(model_id)
            output_to_schemas.setdefault(output_path, set()).add(schema_base_name)

        for output_path, model_ids in output_to_models.items():
            for schema_name in sorted(output_to_schemas.get(output_path, set()) or {queue_schema_name}):
                seed_batch_meta_header(
                    output_path=output_path,
                    model_ids=sorted(model_ids),
                    schema_name=schema_name,
                    input_source="manifest",
                    manifest_path=selected_manifest,
                )

        output_path_by_model = _build_output_path_by_model(
            catalog=catalog,
            selected_manifest=selected_manifest,
            linked_batch_output_path=linked_batch_output_path,
        )
        snapshot_kwargs_by_model = _build_snapshot_kwargs_by_model(
            catalog=catalog,
            selected_manifest=selected_manifest,
            schema_name_by_model={model: schema_base_name for model in queued_models},
        )

        queue_results: list[_BatchQueueResultContext] = []
        initial_snapshots = {
            model: manifest_execution_snapshot(**snapshot_kwargs_by_model[model])
            for model in queued_models
        }
        colored_queue_models = [
            colorize_model(kit.style, _label_for(model), _snapshot_for(model))
            for model in queued_models
        ]
        kit.log("Modelos por estado: " + " | ".join(colored_queue_models), "info")
        completed_models = [
            model for model in queued_models if is_model_snapshot_complete(_snapshot_for(model))
        ]
        pending_or_partial_models = [model for model in queued_models if model not in completed_models]
        models_to_execute: set[str] = set(pending_or_partial_models)
        models_to_execute = _select_models_to_execute(
            kit=kit,
            app=app,
            context=_BatchModelSelectionContext(
                schema_exec_name=queue_schema_name,
                queued_models=queued_models,
                completed_models=completed_models,
                snapshot_for=_snapshot_for,
                default_models=models_to_execute,
                display_label_for=_label_for,
            ),
        )

        runnable_models = [
            model
            for model in queued_models
            if model in models_to_execute
        ]
        runnable_models_count = len(runnable_models)
        runnable_index_by_model = {model: index for index, model in enumerate(runnable_models)}

        queue_targets = _build_queue_targets(
            runnable_models=runnable_models,
            initial_snapshots=initial_snapshots,
        )
        total_global_target = sum(queue_targets.values())
        completed_global_images = 0

        for queued_model in queued_models:
            model_label = _label_for(queued_model)
            model_snapshot_kwargs = snapshot_kwargs_by_model[queued_model]
            output_path = output_path_by_model[queued_model]
            manifest_snapshot = manifest_execution_snapshot(**model_snapshot_kwargs)
            schema_exec_name = schema_name_by_execution[queued_model]
            model_id = model_id_by_execution[queued_model]
            include_reasoning = include_reasoning_by_execution[queued_model]

            if queued_model not in models_to_execute:
                kit.log(f"[{model_label}] Omitido por selección de ejecución.", "info")
                queue_results.append(
                    _BatchQueueResultContext(
                        model_label=model_label,
                        model_id=model_id,
                        include_reasoning=include_reasoning,
                        initial_snapshot=manifest_snapshot,
                        final_snapshot=manifest_snapshot,
                        output_path=output_path,
                    )
                )
                continue

            if is_model_snapshot_complete(manifest_snapshot):
                ok_prune = prune_output_records_for_model(
                    output_path=output_path,
                    model_tag=model_id,
                    schema_name=schema_base_name,
                    include_reasoning=include_reasoning,
                )
                if not ok_prune:
                    kit.log(f"[{model_label}] No se pudo limpiar el JSONL compartido para este modelo.", "error")
                    queue_results.append(
                        _BatchQueueResultContext(
                            model_label=model_label,
                            model_id=model_id,
                            include_reasoning=include_reasoning,
                            initial_snapshot=manifest_snapshot,
                            final_snapshot=manifest_snapshot,
                            output_path=output_path,
                        )
                    )
                    continue

            rerun_only_pending = str(manifest_snapshot.get("status") or "") == "yellow"
            current_model_target = int(manifest_snapshot.get("total", queue_targets.get(queued_model, 0)) or 0)
            models_done_before = int(runnable_index_by_model.get(queued_model, 0))
            models_target = runnable_models_count
            baseline_completed = int(manifest_snapshot.get("ok", 0) or 0) if rerun_only_pending else 0
            final_snapshot, summary, succeeded = _execute_model_with_retries(
                kit=kit,
                app=app,
                run_batch_job=run_batch_job,
                manifest_execution_snapshot=manifest_execution_snapshot,
                execution=_BatchModelExecutionContext(
                    selected_manifest=selected_manifest,
                    schema_exec_name=schema_exec_name,
                    schema_base_name=schema_base_name,
                    include_reasoning=include_reasoning,
                    model_id=model_id,
                    model_label=model_label,
                    output_path=output_path,
                    manifest_snapshot=manifest_snapshot,
                    rerun_only_pending=rerun_only_pending,
                    completed_global_images_before_model=completed_global_images,
                    total_global_target=total_global_target,
                    current_model_target=current_model_target,
                    models_done_before=models_done_before,
                    models_target=models_target,
                    iterations_per_image=iterations_per_image,
                    baseline_completed_for_model=baseline_completed,
                ),
            )
            completed_global_images += int(final_snapshot.get("ok", 0))
            queue_results.append(
                _BatchQueueResultContext(
                    model_label=model_label,
                    model_id=model_id,
                    include_reasoning=include_reasoning,
                    initial_snapshot=manifest_snapshot,
                    final_snapshot=final_snapshot,
                    output_path=output_path,
                )
            )
            if succeeded:
                level = "success" if int(final_snapshot.get("pending", 0)) == 0 else "warning"
                kit.log(
                    f"[{model_label}] Batch completado: {summary['ok']} OK, {summary['invalid']} inválidas, "
                    f"{summary['fail']} errores. Salida: {summary['output_path']}",
                    level,
                )

        summary_model_ids = sorted({model_id_by_execution[item] for item in queued_models})

        final_batch_renderer = _render_batch_final_summary(
            kit=kit,
            app=app,
            context=_BatchFinalRenderContext(
                selected_manifest=selected_manifest,
                summary_schema_base_name=schema_base_name,
                schema_exec_name=queue_schema_name,
                selected_model_ids=summary_model_ids,
                queue_results=queue_results,
            ),
            upsert_batch_execution_summary=upsert_batch_execution_summary,
        )
        kit.render_and_wait_responsive(
            render_fn=final_batch_renderer,
            message="Press any key to return to tests manager...",
            initial_render=False,
        )
        return
