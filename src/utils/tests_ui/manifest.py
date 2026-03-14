from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, cast

from ..setup_ui_io import ask_text

if TYPE_CHECKING:
    from ..menu_kit import AppContext, UIKit


_MANIFEST_META_KEY = "__manifest_meta__"


def _parse_models(raw_models: Any) -> list[str]:
    """Normalize run model list from manifest values."""
    if isinstance(raw_models, list):
        return [str(item).strip() for item in raw_models if str(item).strip()]
    if isinstance(raw_models, str):
        return [item.strip() for item in raw_models.split(",") if item.strip()]
    return []


def _safe_positive_int(value: Any, default: int = 1) -> int:
    """Parsea enteros positivos con fallback seguro."""
    try:
        return max(1, int(value or default))
    except (TypeError, ValueError):
        return default


def _read_manifest_meta(manifest_path: str) -> dict[str, Any] | None:
    """Read global manifest metadata if present in the first JSONL lines."""
    path = Path(manifest_path)
    if not path.exists() or not path.is_file():
        return None
    for _line, payload in _iter_jsonl_lines_with_payload(path):
        if payload is None:
            continue
        meta = payload.get(_MANIFEST_META_KEY)
        if isinstance(meta, dict):
            return meta
        return None
    return None


def _iter_jsonl_lines_with_payload(path: Path) -> Iterator[tuple[str, dict[str, Any] | None]]:
    """Itera líneas JSONL devolviendo cada línea y su payload dict parseado.

    Args:
        path: Ruta al archivo JSONL.

    Yields:
        Tuplas ``(linea_sin_salto, payload_dict_o_none)``.
    """
    if not path.exists() or not path.is_file():
        return

    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                payload: dict[str, Any] | None = None
                try:
                    parsed = json.loads(line)
                    if isinstance(parsed, dict):
                        payload = parsed
                except Exception:
                    payload = None
                yield line, payload
    except Exception:
        return


def _load_manifest_class_counts(csv_path: Path) -> dict[str, int]:
    """Carga conteos por clase para estimar el reparto estratificado.

    Args:
        csv_path: Ruta al CSV base de entrenamiento.

    Returns:
        Diccionario ``clase -> número de filas``. Vacío si no se puede cargar.
    """
    if not csv_path.exists() or not csv_path.is_file():
        return {}

    try:
        import pandas as pd
    except Exception:
        return {}

    try:
        frame = pd.read_csv(csv_path)
        if "cls" not in frame.columns:
            return {}
        counts = frame["cls"].value_counts(dropna=False).to_dict()
    except Exception:
        return {}

    normalized: dict[str, int] = {}
    for key, value in counts.items():
        normalized[str(key)] = int(value)
    return normalized


def _select_manifest_sample_size(kit: "UIKit") -> int | str:
    """Solicita tamaño de muestra con previsualización por subgrupos.

    Args:
        kit: Terminal UI toolkit.

    Returns:
        Positive integer sample size, or `"BACK"` if cancelled.
    """
    class_counts = _load_manifest_class_counts(Path("data/raw/m_train/train.csv"))
    total_available = sum(class_counts.values())

    from .manifest_generation import _compute_target_counts
    total_distribution = " | ".join(
        f"{cls_name}={class_counts[cls_name]}" for cls_name in sorted(class_counts.keys())
    )

    def _validate_sample_size(raw_value: str) -> str | None:
        """Valida tamaño de muestra escrito por el usuario.

        Args:
            raw_value: Texto confirmado en el campo de entrada.

        Returns:
            Mensaje de error si no es válido; `None` si es correcto.
        """
        if not raw_value:
            return "Debes indicar un número entero positivo."
        if not raw_value.isdigit():
            return "Entrada inválida. Usa solo dígitos."
        if int(raw_value) <= 0:
            return "El tamaño debe ser mayor que cero."
        return None

    def _render_preview_lines(current_value: str, _ui_width: int) -> list[str]:
        """Construye líneas dinámicas del preview estratificado.

        Args:
            current_value: Valor actual del input.
            _ui_width: Ancho actual de terminal (no requerido en este layout).

        Returns:
            Líneas para render dinámico bajo el campo de entrada.
        """
        style = kit.style
        sample_size = int(current_value) if current_value.isdigit() else 0
        preview_counts: dict[str, int] = {}
        if sample_size > 0 and class_counts:
            preview_counts = _compute_target_counts(class_counts, sample_size)

        lines: list[str] = []
        if class_counts:
            lines.append(
                f" {style.DIM}Dataset base: {total_available} imágenes · {len(class_counts)} clases{style.ENDC}"
            )
            if total_distribution:
                lines.append(f" {style.DIM}Distribución total: {total_distribution}{style.ENDC}")
            lines.append(" " + kit.divider(_ui_width))

            if sample_size > 0:
                if sample_size > total_available > 0:
                    lines.append(
                        f" {style.WARNING}Solicitud {sample_size}: se usará el máximo disponible ({total_available}).{style.ENDC}"
                    )
                lines.append(f" {style.BOLD}{style.OKCYAN}Distribución estimada del subgrupo:{style.ENDC}")
                for cls_name in sorted(preview_counts.keys()):
                    estimated = int(preview_counts[cls_name])
                    pct = (estimated / sample_size * 100.0) if sample_size > 0 else 0.0
                    lines.append(f"   {style.DIM}•{style.ENDC} {cls_name:<8} {estimated:>4} ({pct:>5.1f}%)")
                lines.append(" " + kit.divider(_ui_width))
                lines.append(
                    f" {style.DIM}Total estimado: {sum(preview_counts.values())} / solicitado: {sample_size}{style.ENDC}"
                )
            else:
                lines.append(
                    f" {style.DIM}Empieza a escribir un número para ver el reparto estimado.{style.ENDC}"
                )
        else:
            lines.append(
                f" {style.WARNING}No se pudo leer data/raw/m_train/train.csv para previsualizar subgrupos.{style.ENDC}"
            )
        return lines

    value = ask_text(
        kit=kit,
        title="BATCH RUNNER · MANIFEST SAMPLE SIZE",
        intro_lines=[
            f"{kit.style.DIM}Define el tamaño del subgrupo y valida en vivo el reparto estratificado por clase (cls).{kit.style.ENDC}",
        ],
        prompt_label="Entrada:",
        help_line="[ENTER] confirmar · [Backspace] borrar · [ESC] cancelar",
        allow_char_fn=lambda ch: ch.isdigit(),
        normalize_on_submit_fn=lambda text: text.strip(),
        validate_on_submit_fn=_validate_sample_size,
        render_extra_lines_fn=_render_preview_lines,
        force_full_on_update=True,
    )
    if value is None:
        return "BACK"
    return int(value)


def _select_manifest_iterations_per_image(kit: "UIKit") -> int | str:
    """Solicita cuántas iteraciones ejecutar por imagen/modelo.

    Args:
        kit: Terminal UI toolkit.

    Returns:
        Entero positivo o "BACK" si se cancela.
    """

    def _validate_iterations(raw_value: str) -> str | None:
        if not raw_value:
            return "Debes indicar un número entero positivo."
        if not raw_value.isdigit():
            return "Entrada inválida. Usa solo dígitos."
        if int(raw_value) <= 0:
            return "El número de iteraciones debe ser mayor que cero."
        if int(raw_value) > 100:
            return "Por seguridad, el máximo permitido es 100 iteraciones por imagen."
        return None

    def _render_iterations_preview(current_value: str, _ui_width: int) -> list[str]:
        style = kit.style
        value = int(current_value) if current_value.isdigit() else 0
        lines: list[str] = [
            (
                f" {style.DIM}Este valor multiplica el número de filas del manifiesto por imagen."
                f"{style.ENDC}"
            )
        ]
        if value > 0:
            lines.append(f" {style.OKCYAN}Iteraciones por imagen: {value}x{style.ENDC}")
        else:
            lines.append(f" {style.DIM}Introduce un entero para continuar.{style.ENDC}")
        return lines

    value = ask_text(
        kit=kit,
        title="BATCH RUNNER · MANIFEST ITERATIONS",
        intro_lines=[
            (
                f"{kit.style.DIM}Define cuántas veces se ejecutará cada imagen por modelo"
                f" para estudiar variabilidad de respuestas.{kit.style.ENDC}"
            ),
        ],
        prompt_label="Entrada:",
        help_line="[ENTER] confirmar · [Backspace] borrar · [ESC] cancelar",
        allow_char_fn=lambda ch: ch.isdigit(),
        normalize_on_submit_fn=lambda text: text.strip(),
        validate_on_submit_fn=_validate_iterations,
        render_extra_lines_fn=_render_iterations_preview,
        force_full_on_update=True,
    )
    if value is None:
        return "BACK"
    return int(value)


def discover_experiment_manifests() -> list[str]:
    """Discover available experiment manifests under data/experiments.

    Returns:
        Absolute paths to manifest JSONL files ordered by most recent first.
    """
    experiments_dir = Path("data/experiments")
    if not experiments_dir.exists() or not experiments_dir.is_dir():
        return []

    manifests = [
        path.resolve()
        for path in sorted(
            experiments_dir.glob("*.jsonl"),
            key=lambda candidate: candidate.stat().st_mtime,
            reverse=True,
        )
        if path.is_file() and "manifest" in path.name.lower()
    ]
    return [str(path) for path in manifests]


def _sanitize_name(name: str) -> str:
    """Normalize strings so they are safe for filenames.

    Args:
        name: Raw token to normalize.

    Returns:
        Filename-safe token.
    """
    allowed = []
    for char in name:
        if char.isalnum() or char in ("-", "_"):
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_") or "run"


def linked_batch_output_path(*, manifest_path: str, model_tag: str, schema_name: str) -> Path:
    """Build stable output JSONL path bound to manifest+schema.

    Args:
        manifest_path: Manifest JSONL path.
        model_tag: LM Studio model tag (unused, kept for compatibility).
        schema_name: Effective execution schema name.

    Returns:
        Absolute output path for batch JSONL results.
    """
    manifest_stem = _sanitize_name(Path(manifest_path).stem)
    schema_token = _sanitize_name(schema_name)
    output_dir = Path("data/processed/batch_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    return (output_dir / f"batch_{manifest_stem}_{schema_token}.jsonl").resolve()


def load_manifest_entries(manifest_path: str) -> list[dict[str, Any]]:
    """Load valid rows from a manifest JSONL file.

    Args:
        manifest_path: Path to JSONL manifest.

    Returns:
        Parsed valid row dictionaries containing at least `image_path`.
    """
    entries: list[dict[str, Any]] = []
    for _line, payload in _iter_jsonl_lines_with_payload(Path(manifest_path)):
        if payload is None:
            continue
        image_path = payload.get("image_path")
        if image_path is None:
            continue
        entries.append(payload)
    return entries


def _entry_execution_key(entry: dict[str, Any]) -> tuple[str, int]:
    """Genera una clave estable por entrada soportando iteraciones repetidas."""
    image_path = str(entry.get("image_path") or "").strip()
    iteration_index = _safe_positive_int(entry.get("run_iteration_index"))
    return image_path, iteration_index


def _load_last_records_by_entry_for_model(
    output_path: str,
    *,
    model_tag: str,
) -> dict[tuple[str, int], dict[str, Any]]:
    """Index latest result record by entry key filtered by model id.

    Args:
        output_path: Path to batch output JSONL.

    Returns:
        Mapping `(image_path, run_iteration_index) -> last seen record`.
    """
    records_by_entry: dict[tuple[str, int], dict[str, Any]] = {}
    output_file = Path(output_path)
    if not output_file.exists() or not output_file.is_file():
        return records_by_entry

    for _line, payload in _iter_jsonl_lines_with_payload(output_file):
        if payload is None:
            continue
        if isinstance(payload.get("__batch_meta__"), dict):
            continue
        if str(payload.get("model_id") or "").strip() != str(model_tag).strip():
            continue
        image_path = payload.get("image_path")
        if not isinstance(image_path, str) or not image_path.strip():
            continue
        iteration_index = _safe_positive_int(payload.get("run_iteration_index"))
        record_key = (image_path.strip(), iteration_index)
        records_by_entry[record_key] = payload

    return records_by_entry


def prune_output_records_for_model(*, output_path: str, model_tag: str) -> bool:
    """Remove records for one model from a shared batch output JSONL.

    Args:
        output_path: Shared batch output path.
        model_tag: Model id whose rows should be removed.

    Returns:
        `True` when rewrite succeeds.
    """
    out_file = Path(output_path)
    if not out_file.exists() or not out_file.is_file():
        return True

    kept_lines: list[str] = []
    try:
        for line, payload in _iter_jsonl_lines_with_payload(out_file):
            if payload is None:
                kept_lines.append(line)
                continue
            if isinstance(payload.get("__batch_meta__"), dict):
                kept_lines.append(line)
                continue
            if str(payload.get("model_id") or "").strip() == str(model_tag).strip():
                continue
            kept_lines.append(line)
        with out_file.open("w", encoding="utf-8", newline="\n") as handle:
            for entry in kept_lines:
                handle.write(entry + "\n")
        return True
    except Exception:
        return False


def _is_success_record(record: dict[str, Any]) -> bool:
    """Determine whether an output record is considered successful.

    Args:
        record: Parsed result row.

    Returns:
        `True` when status is `ok` and payload contains meaningful content.
    """
    if str(record.get("status") or "") != "ok":
        return False
    payload = record.get("payload")
    if isinstance(payload, dict):
        return bool(payload)
    return payload is not None


def manifest_execution_snapshot(*, manifest_path: str, model_tag: str, schema_name: str) -> dict[str, Any]:
    """Compute execution semaphore snapshot for a manifest/model/schema run.

    Args:
        manifest_path: Manifest JSONL path.
        model_tag: LM Studio model tag.
        schema_name: Effective execution schema name.

    Returns:
        Snapshot dictionary with status, counters, pending entries and output path.
    """
    entries = load_manifest_entries(manifest_path)
    output_path = str(linked_batch_output_path(manifest_path=manifest_path, model_tag=model_tag, schema_name=schema_name))

    if not os.path.exists(output_path):
        try:
            with open(output_path, "w", encoding="utf-8", newline="\n"):
                pass
        except Exception:
            pass

    records_by_entry = _load_last_records_by_entry_for_model(output_path, model_tag=model_tag)

    total = len(entries)
    if total <= 0:
        return {
            "status": "red",
            "total": 0,
            "ok": 0,
            "errors": 0,
            "pending": 0,
            "output_path": output_path,
            "pending_entries": [],
        }

    ok_count = 0
    error_count = 0
    pending_entries: list[dict[str, Any]] = []
    for entry in entries:
        record = records_by_entry.get(_entry_execution_key(entry))
        if record is None:
            pending_entries.append(entry)
            continue
        if _is_success_record(record):
            ok_count += 1
            continue
        error_count += 1
        pending_entries.append(entry)

    if ok_count == total and error_count == 0:
        status = "green"
    elif ok_count == 0 and error_count == 0:
        status = "red"
    else:
        status = "yellow"

    return {
        "status": status,
        "total": total,
        "ok": ok_count,
        "errors": error_count,
        "pending": len(pending_entries),
        "output_path": output_path,
        "pending_entries": pending_entries,
    }


def status_label(kit: "UIKit", snapshot: dict[str, Any]) -> str:
    """Render colored label for manifest snapshot status.

    Args:
        kit: Terminal UI toolkit.
        snapshot: Snapshot dictionary produced by `manifest_execution_snapshot`.

    Returns:
        ANSI-colored status label.
    """
    okgreen = getattr(kit.style, "OKGREEN", "")
    warning = getattr(kit.style, "WARNING", "")
    fail = getattr(kit.style, "FAIL", "")
    endc = getattr(kit.style, "ENDC", "")
    status = str(snapshot.get("status") or "red")
    if status == "green":
        return f"{okgreen}● COMPLETO{endc}"
    if status == "yellow":
        return f"{warning}● PARCIAL{endc}"
    return f"{fail}● SIN EJECUTAR{endc}"


def create_pending_manifest(*, pending_entries: list[dict[str, Any]], source_manifest_path: str) -> str | None:
    """Write temporary pending-only manifest for retry execution.

    Args:
        pending_entries: Entries still pending or failed.
        source_manifest_path: Original manifest path.

    Returns:
        Temporary manifest path, or `None` when nothing is pending.
    """
    if not pending_entries:
        return None
    source = Path(source_manifest_path)
    temp_manifest = source.with_name(f"{source.stem}.pending.jsonl")
    with temp_manifest.open("w", encoding="utf-8", newline="\n") as handle:
        meta = _read_manifest_meta(source_manifest_path)
        if isinstance(meta, dict):
            handle.write(json.dumps({_MANIFEST_META_KEY: meta}, ensure_ascii=False) + "\n")
        for entry in pending_entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return str(temp_manifest.resolve())


def _format_class_distribution(summary: dict[str, Any]) -> str:
    """Format compact class distribution text from generation summary.

    Args:
        summary: Manifest generation summary dictionary.

    Returns:
        Human-readable class distribution string.
    """
    counts = cast(dict[str, Any], summary.get("written_by_class") or {})
    if not counts:
        return "sin datos"
    parts = [f"{cls}={counts[cls]}" for cls in sorted(counts.keys())]
    return " | ".join(parts)


def extract_manifest_run_config(manifest_path: str) -> dict[str, Any] | None:
    """Extract embedded run configuration from manifest first row.

    Args:
        manifest_path: Manifest JSONL path.

    Returns:
        Config dictionary with models and schema settings, or `None`.
    """
    meta = _read_manifest_meta(manifest_path)
    if isinstance(meta, dict):
        models = _parse_models(meta.get("run_models"))
        schema_name = str(meta.get("run_schema_name") or "").strip()
        include_reasoning = bool(meta.get("run_include_reasoning"))
        iterations_per_image = _safe_positive_int(meta.get("run_iterations_per_image"))
        if models and schema_name:
            return {
                "models": models,
                "schema_name": schema_name,
                "include_reasoning": include_reasoning,
                "iterations_per_image": iterations_per_image,
            }

    entries = load_manifest_entries(manifest_path)
    if not entries:
        return None

    # Fallback compatible con manifiestos antiguos: agregamos modelos de todas las filas.
    models: list[str] = []
    seen_models: set[str] = set()
    schema_name = ""
    include_reasoning = False
    iterations_candidates: list[int] = []

    for entry in entries:
        for model in _parse_models(entry.get("run_models")):
            if model not in seen_models:
                seen_models.add(model)
                models.append(model)

        if not schema_name:
            schema_name = str(entry.get("run_schema_name") or "").strip()

        include_reasoning = include_reasoning or bool(entry.get("run_include_reasoning"))

        iterations_candidates.append(
            _safe_positive_int(entry.get("run_iterations_per_image") or entry.get("run_iteration_total"))
        )

    iterations_per_image = max(iterations_candidates) if iterations_candidates else 1
    if not models or not schema_name:
        return None

    return {
        "models": models,
        "schema_name": schema_name,
        "include_reasoning": include_reasoning,
        "iterations_per_image": iterations_per_image,
    }


def execution_schema_name(schema_name: str, include_reasoning: bool) -> str:
    """Build public schema name used in status and reporting.

    Args:
        schema_name: Base schema name.
        include_reasoning: Whether reasoning variant is enabled.

    Returns:
        Effective schema name.
    """
    if include_reasoning and not schema_name.endswith("WithReasoning"):
        return f"{schema_name}WithReasoning"
    return schema_name


def _select_models_for_manifest(
    kit: "UIKit",
    app: "AppContext",
    *,
    make_header: Callable[[str], Callable[[], None]],
) -> list[str] | None:
    """Select multiple models to embed in a generated manifest.

    Args:
        kit: Terminal UI toolkit.
        app: Application context with model inventory.
        make_header: Header factory for interactive menus.

    Returns:
        Selected model tags or `None` if cancelled.
    """
    installed = app.get_installed_lms_models()
    if not installed:
        kit.log(
            "No hay modelos disponibles en LM Studio. Carga uno desde 'Manage/Pull LM Studio Models...'.",
            "warning",
        )
        return None

    model_items = [
        kit.MenuItem(tag, description="Modelo candidato para ejecutar este manifiesto.")
        for tag in installed
    ]
    selected = kit.menu(
        model_items,
        header_func=make_header("MANIFEST CONFIG · SELECT MODELS"),
        menu_id="manifest_models_selector",
        multi_select=True,
        nav_hint_text="↑/↓ navegar · SPACE seleccionar · ENTER confirmar · ESC cancelar",
    )
    if not selected:
        return None
    if not isinstance(selected, list):
        selected = [selected]
    tags = [str(item.label).strip() for item in selected if str(item.label).strip()]
    return tags or None


def _select_schema_config_for_manifest(
    select_schema_variant: Callable[[str, str], tuple[str, Any] | None],
) -> tuple[str, bool] | None:
    """Select schema base name and reasoning flag for manifest config.

    Args:
        select_schema_variant: Callback that returns selected schema variant.

    Returns:
        Tuple `(base_schema_name, include_reasoning)` or `None`.
    """
    schema_variant = select_schema_variant("manifest", "MANIFEST CONFIG")
    if schema_variant is None:
        return None
    public_name, _schema_cls = schema_variant
    include_reasoning = public_name.endswith("WithReasoning")
    base_schema_name = public_name.removesuffix("WithReasoning") if include_reasoning else public_name
    return base_schema_name, include_reasoning


def manifest_overview(manifest_path: str) -> dict[str, Any]:
    """Aggregate manifest status across all configured queue models.

    Args:
        manifest_path: Manifest JSONL path.

    Returns:
        Overview dictionary with global status, per-model runs and description.
    """
    config = extract_manifest_run_config(manifest_path)
    if config is None:
        return {
            "status": "red",
            "runs": [],
            "config": None,
            "description": "sin configuración de ejecución embebida",
        }

    schema_exec_name = execution_schema_name(
        cast(str, config["schema_name"]),
        bool(config["include_reasoning"]),
    )
    runs: list[dict[str, Any]] = []
    statuses: list[str] = []
    total_pending = 0
    for model in cast(list[str], config["models"]):
        snapshot = manifest_execution_snapshot(
            manifest_path=manifest_path,
            model_tag=model,
            schema_name=schema_exec_name,
        )
        statuses.append(str(snapshot.get("status") or "red"))
        total_pending += int(snapshot.get("pending", 0))
        runs.append({"model": model, "snapshot": snapshot})

    if statuses and all(status == "green" for status in statuses):
        status = "green"
    elif statuses and all(status == "red" for status in statuses):
        status = "red"
    else:
        status = "yellow"

    return {
        "status": status,
        "runs": runs,
        "config": config,
        "description": (
            f"schema={config['schema_name']}"
            f"{' + reasoning' if config['include_reasoning'] else ''} · "
            f"modelos={len(config['models'])} · iteraciones={int(config.get('iterations_per_image', 1))}x · "
            f"pendientes={total_pending}"
        ),
    }


def _generate_manifest_with_prompt(
    kit: "UIKit",
    app: "AppContext",
    *,
    subtitle: str,
    make_header: Callable[[str], Callable[[], None]],
    select_schema_variant: Callable[[str, str], tuple[str, Any] | None],
) -> dict[str, Any] | None:
    """Interactively generate a new autosufficient execution manifest.

    Args:
        kit: Terminal UI toolkit.
        app: Application context used by selectors.
        subtitle: Subtitle for generation screen.
        make_header: Header factory for interactive menus.
        select_schema_variant: Callback for schema variant selection.

    Returns:
        Manifest generation summary or `None` when cancelled.
    """
    from .manifest_generation import generate_manifest

    sample_size = _select_manifest_sample_size(kit)
    if sample_size == "BACK":
        return None

    iterations_per_image = _select_manifest_iterations_per_image(kit)
    if iterations_per_image == "BACK":
        return None

    models = _select_models_for_manifest(kit, app, make_header=make_header)
    if not models:
        kit.log("Generación cancelada: debes seleccionar al menos un modelo.", "warning")
        return None

    schema_config = _select_schema_config_for_manifest(select_schema_variant)
    if schema_config is None:
        return None
    base_schema_name, include_reasoning = schema_config

    selected_size = cast(int, sample_size)
    selected_iterations = cast(int, iterations_per_image)
    output_dir = Path("data/experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = (output_dir / f"zeroshot_manifest_{timestamp}.jsonl").resolve()

    kit.clear()
    app.print_banner()
    kit.subtitle(subtitle)
    kit.log(
        f"Generando manifiesto estratificado (n={selected_size}) listo para ejecución automática...",
        "step",
    )

    summary = generate_manifest(
        input_csv=Path("data/raw/m_train/train.csv").resolve(),
        images_dir=Path("data/processed/m_train/images").resolve(),
        output_path=output_path,
        sample_size=selected_size,
        seed=42,
        stratify_col="cls",
        id_col="image_id",
        label_col="cls",
        relative_paths=False,
        run_models=models,
        run_schema_name=base_schema_name,
        run_include_reasoning=include_reasoning,
        run_iterations_per_image=selected_iterations,
    )
    kit.log(f"Manifiesto generado: {summary.get('output_path')}", "success")
    kit.log(f"Iteraciones por imagen: {summary.get('run_iterations_per_image', 1)}x", "info")
    kit.log("Distribución del subconjunto: " + _format_class_distribution(summary), "info")
    return summary


def select_manifest_for_batch(
    kit: "UIKit",
    app: "AppContext",
    *,
    make_header: Callable[[str], Callable[[], None]],
    select_schema_variant: Callable[[str, str], tuple[str, Any] | None],
) -> dict[str, Any] | None:
    """Select an existing autosufficient manifest or generate a new one.

    Args:
        kit: Terminal UI toolkit.
        app: Application context used by model selector and UI rendering.
        make_header: Header factory for interactive menus.
        select_schema_variant: Callback for schema variant selection.

    Returns:
        Dictionary with selected manifest path, overview and run config, or `None`.
    """
    manifests = discover_experiment_manifests()
    if not manifests:
        kit.log("No hay manifiestos detectados. Se iniciará la generación automática.", "warning")
        generated_summary = _generate_manifest_with_prompt(
            kit,
            app,
            subtitle="BATCH RUNNER · GENERATE MANIFEST",
            make_header=make_header,
            select_schema_variant=select_schema_variant,
        )
        if generated_summary is None:
            return None
        manifest_path = str(generated_summary.get("output_path") or "")
        if not manifest_path:
            return None
        overview = manifest_overview(manifest_path)
        return {
            "manifest_path": manifest_path,
            "overview": overview,
            "config": overview.get("config"),
        }

    options = []
    overviews = {path: manifest_overview(path) for path in manifests}
    for path in manifests:
        overview = overviews[path]
        options.append(
            kit.MenuItem(
                f"Usar {os.path.basename(path)} · {status_label(kit, overview)}",
                description=f"{path} · {overview.get('description', '-')}",
            )
        )
    options.append(
        kit.MenuItem(
            "Generar manifiesto nuevo",
            description="Crea un nuevo manifiesto con subset, modelos y schema embebidos.",
        )
    )
    options.append(kit.MenuItem("Cancel", lambda: None, description="Vuelve al selector anterior."))

    selection = kit.menu(
        options,
        header_func=make_header("BATCH RUNNER · SELECT MANIFEST"),
        menu_id="batch_runner_manifest_selector",
        nav_hint_text="↑/↓ elegir manifiesto · ENTER confirmar · ESC volver",
        description_slot_rows=10,
    )
    if not selection or selection.label.strip() == "Cancel":
        return None
    if selection.label.startswith("Generar manifiesto nuevo"):
        generated_summary = _generate_manifest_with_prompt(
            kit,
            app,
            subtitle="BATCH RUNNER · GENERATE MANIFEST",
            make_header=make_header,
            select_schema_variant=select_schema_variant,
        )
        if generated_summary is None:
            return None
        manifest_path = str(generated_summary.get("output_path") or "")
        if not manifest_path:
            return None
        overview = manifest_overview(manifest_path)
        return {
            "manifest_path": manifest_path,
            "overview": overview,
            "config": overview.get("config"),
        }

    selected_label = selection.label.replace("Usar ", "", 1).split(" · ", 1)[0].strip()
    for manifest_path in manifests:
        if os.path.basename(manifest_path) == selected_label:
            overview = overviews[manifest_path]
            return {
                "manifest_path": manifest_path,
                "overview": overview,
                "config": overview.get("config"),
            }
    return None
