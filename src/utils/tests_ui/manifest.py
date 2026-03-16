from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, cast

from ..setup_ui_io import ask_text
from .shared import (
    normalize_model_variants,
    parse_required_positive_int,
    safe_positive_int,
    select_model_variants,
    snapshot_status_text,
    variant_label,
)

if TYPE_CHECKING:
    from ..menu_kit import AppContext, UIKit


_MANIFEST_META_KEY = "__manifest_meta__"


def _normalize_manifest_run_config(
    *,
    schema_name: str,
    iterations_per_image: int,
    seed: int,
    model_variants: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Build a canonical run config payload used by the batch wrapper."""
    if not schema_name:
        return None

    normalized_variants = normalize_model_variants(model_variants)
    if not normalized_variants:
        return None

    return {
        "model_variants": normalized_variants,
        "schema_name": schema_name,
        "iterations_per_image": safe_positive_int(iterations_per_image),
        "seed": safe_positive_int(seed, default=42),
    }


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


def _output_schema_name(schema_name: str) -> str:
    """Normaliza schema para nombre de archivo de salida compartido."""
    value = str(schema_name or "").strip()
    suffix = "WithReasoning"
    if value.endswith(suffix) and len(value) > len(suffix):
        return value[: -len(suffix)]
    return value


def linked_batch_output_path(*, manifest_path: str, schema_name: str) -> Path:
    """Build stable output JSONL path bound to manifest+schema.

    Args:
        manifest_path: Manifest JSONL path.
        schema_name: Effective execution schema name.

    Returns:
        Absolute output path for batch JSONL results.
    """
    manifest_stem = _sanitize_name(Path(manifest_path).stem)
    schema_token = _sanitize_name(_output_schema_name(schema_name))
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
    iteration_index = safe_positive_int(entry.get("run_iteration_index"))
    return image_path, iteration_index


def _load_last_records_by_entry_for_model(
    output_path: str,
    *,
    model_tag: str,
    schema_name: str,
    include_reasoning: bool,
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
        if str(payload.get("schema_name") or "").strip() != str(schema_name).strip():
            continue
        if bool(payload.get("include_reasoning")) != bool(include_reasoning):
            continue
        image_path = payload.get("image_path")
        if not isinstance(image_path, str) or not image_path.strip():
            continue
        iteration_index = safe_positive_int(payload.get("run_iteration_index"))
        record_key = (image_path.strip(), iteration_index)
        records_by_entry[record_key] = payload

    return records_by_entry


def prune_output_records_for_model(
    *,
    output_path: str,
    model_tag: str,
    schema_name: str,
    include_reasoning: bool,
) -> bool:
    """Remove records for one model from a shared batch output JSONL.

    Args:
        output_path: Shared batch output path.
        model_tag: Model id whose rows should be removed.
        schema_name: Base schema name shared by variants.
        include_reasoning: Variant mode flag used to isolate rows.

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
            if (
                str(payload.get("model_id") or "").strip() == str(model_tag).strip()
                and str(payload.get("schema_name") or "").strip() == str(schema_name).strip()
                and bool(payload.get("include_reasoning")) == bool(include_reasoning)
            ):
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


def manifest_execution_snapshot(
    *,
    manifest_path: str,
    model_tag: str,
    schema_name: str,
    include_reasoning: bool,
) -> dict[str, Any]:
    """Compute execution semaphore snapshot for a manifest/model/schema run.

    Args:
        manifest_path: Manifest JSONL path.
        model_tag: LM Studio model tag.
        schema_name: Base schema name.
        include_reasoning: Variant mode flag.

    Returns:
        Snapshot dictionary with status, counters, pending entries and output path.
    """
    entries = load_manifest_entries(manifest_path)
    output_path = str(linked_batch_output_path(manifest_path=manifest_path, schema_name=schema_name))

    if not os.path.exists(output_path):
        try:
            with open(output_path, "w", encoding="utf-8", newline="\n"):
                pass
        except Exception:
            pass

    records_by_entry = _load_last_records_by_entry_for_model(
        output_path,
        model_tag=model_tag,
        schema_name=schema_name,
        include_reasoning=include_reasoning,
    )

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
    style = kit.style
    color_by_status = {
        "COMPLETO": str(getattr(style, "OKGREEN", "")),
        "PARCIAL": str(getattr(style, "WARNING", "")),
        "ERROR": str(getattr(style, "FAIL", "")),
    }
    plain = snapshot_status_text(snapshot)
    color = color_by_status.get(plain, "")
    endc = str(getattr(style, "ENDC", "")) if color else ""
    visible = "SIN EJECUTAR" if plain == "ERROR" else plain
    return f"{color}● {visible}{endc}" if color else f"● {visible}"


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


def _estimate_manifest_sample_size(*, manifest_path: str, iterations_per_image: int) -> int:
    """Estimate original sample size from manifest rows and iteration metadata."""
    entries = load_manifest_entries(manifest_path)
    if not entries:
        return 1

    iterations = safe_positive_int(iterations_per_image)
    unique_entry_keys = {_entry_execution_key(entry) for entry in entries}
    unique_images = {
        str(entry.get("image_id") or entry.get("image_path") or "").strip()
        for entry in entries
        if str(entry.get("image_id") or entry.get("image_path") or "").strip()
    }
    if unique_images:
        return max(1, len(unique_images))
    return max(1, int(len(unique_entry_keys) / iterations))


def _select_manifest_sample_size_with_default(kit: "UIKit", *, initial_value: int | None = None) -> int | str:
    """Solicita tamaño de muestra con un valor inicial precargado."""
    class_counts = _load_manifest_class_counts(Path("data/raw/m_train/train.csv"))
    total_available = sum(class_counts.values())

    from .manifest_generation import _compute_target_counts
    total_distribution = " | ".join(
        f"{cls_name}={class_counts[cls_name]}" for cls_name in sorted(class_counts.keys())
    )

    def _validate_sample_size(raw_value: str) -> str | None:
        if not raw_value:
            return "Debes indicar un número entero positivo."
        if not raw_value.isdigit():
            return "Entrada inválida. Usa solo dígitos."
        if int(raw_value) <= 0:
            return "El tamaño debe ser mayor que cero."
        return None

    def _render_preview_lines(current_value: str, _ui_width: int) -> list[str]:
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
        initial_value=str(safe_positive_int(initial_value, default=1)) if initial_value is not None else "",
        allow_char_fn=lambda ch: ch.isdigit(),
        normalize_on_submit_fn=lambda text: text.strip(),
        validate_on_submit_fn=_validate_sample_size,
        render_extra_lines_fn=_render_preview_lines,
        force_full_on_update=True,
    )
    if value is None:
        return "BACK"
    return int(value)


def _select_manifest_iterations_per_image_with_default(
    kit: "UIKit",
    *,
    initial_value: int | None = None,
) -> int | str:
    """Solicita iteraciones con valor inicial precargado."""

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
        initial_value=str(safe_positive_int(initial_value, default=1)) if initial_value is not None else "",
        allow_char_fn=lambda ch: ch.isdigit(),
        normalize_on_submit_fn=lambda text: text.strip(),
        validate_on_submit_fn=_validate_iterations,
        render_extra_lines_fn=_render_iterations_preview,
        force_full_on_update=True,
    )
    if value is None:
        return "BACK"
    return int(value)


def extract_manifest_run_config(manifest_path: str) -> dict[str, Any] | None:
    """Extract embedded run configuration from manifest first row.

    Args:
        manifest_path: Manifest JSONL path.

    Returns:
        Config dictionary with models and schema settings, or `None`.
    """
    meta = _read_manifest_meta(manifest_path)
    if not isinstance(meta, dict):
        return None

    schema_name = str(meta.get("run_schema_name") or "").strip()
    iterations_per_image = parse_required_positive_int(meta.get("run_iterations_per_image"))
    seed = parse_required_positive_int(meta.get("run_seed"))
    model_variants = normalize_model_variants(meta.get("run_model_variants"))

    if iterations_per_image is None or seed is None:
        return None

    return _normalize_manifest_run_config(
        schema_name=schema_name,
        iterations_per_image=iterations_per_image,
        seed=seed,
        model_variants=model_variants,
    )


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


def _select_source_manifest_to_modify(
    kit: "UIKit",
    *,
    manifests: list[str],
    overviews: dict[str, dict[str, Any]],
    make_header: Callable[[str], Callable[[], None]],
) -> str | None:
    """Prompt source manifest selection for derivation flow."""
    options = [
        kit.MenuItem(
            f"{os.path.basename(path)} · {status_label(kit, overviews[path])}",
            description=f"{path} · {overviews[path].get('description', '-')}",
        )
        for path in manifests
    ]
    options.append(kit.MenuItem("Cancel", lambda: None, description="Vuelve al menú anterior."))
    selected = kit.menu(
        options,
        header_func=make_header("BATCH RUNNER · SELECT SOURCE MANIFEST"),
        menu_id="batch_runner_manifest_modify_selector",
        nav_hint_text="↑/↓ elegir manifiesto origen · ENTER confirmar · ESC volver",
        description_slot_rows=10,
    )
    if not selected or selected.label.strip() == "Cancel":
        return None
    selected_label = selected.label.split(" · ", 1)[0].strip()
    for manifest_path in manifests:
        if os.path.basename(manifest_path) == selected_label:
            return manifest_path
    return None


def _select_source_manifest_to_delete(
    kit: "UIKit",
    *,
    manifests: list[str],
    overviews: dict[str, dict[str, Any]],
    make_header: Callable[[str], Callable[[], None]],
) -> str | None:
    """Prompt manifest selection for deletion flow."""
    options = [
        kit.MenuItem(
            f"{os.path.basename(path)} · {status_label(kit, overviews[path])}",
            description=f"{path} · {overviews[path].get('description', '-')}",
        )
        for path in manifests
    ]
    options.append(kit.MenuItem("Cancel", lambda: None, description="Vuelve al menú anterior."))
    selected = kit.menu(
        options,
        header_func=make_header("BATCH RUNNER · DELETE MANIFEST"),
        menu_id="batch_runner_manifest_delete_selector",
        nav_hint_text="↑/↓ elegir manifiesto a eliminar · ENTER confirmar · ESC volver",
        description_slot_rows=10,
    )
    if not selected or selected.label.strip() == "Cancel":
        return None
    selected_label = selected.label.split(" · ", 1)[0].strip()
    for manifest_path in manifests:
        if os.path.basename(manifest_path) == selected_label:
            return manifest_path
    return None


def _confirm_manifest_deletion(
    kit: "UIKit",
    *,
    manifest_path: str,
    make_header: Callable[[str], Callable[[], None]],
) -> bool:
    """Render an explicit confirmation step before deleting a manifest file."""
    manifest_name = os.path.basename(manifest_path)
    linked_outputs = _linked_shared_jsonl_outputs(manifest_path)
    linked_names = [str(path) for path in linked_outputs]
    if linked_names:
        linked_summary = " | ".join(linked_names[:2])
        if len(linked_names) > 2:
            linked_summary += f" | +{len(linked_names) - 2} más"
    else:
        linked_summary = "No se detectaron JSONL compartidos vinculados."

    kit.log(f"Manifest a eliminar: {manifest_path}", "warning")
    kit.log(f"JSONL compartidos que se eliminarán: {linked_summary}", "warning")
    options = [
        kit.MenuItem(
            f"Eliminar definitivamente {manifest_name}",
            description=(
                "Esta acción borra el manifiesto y sus JSONL compartidos vinculados. "
                f"Archivos detectados: {len(linked_outputs)}"
            ),
        ),
        kit.MenuItem("Cancelar", lambda: None, description="No borrar y volver al selector."),
    ]
    selected = kit.menu(
        options,
        header_func=make_header("BATCH RUNNER · CONFIRM DELETE"),
        menu_id="batch_runner_manifest_delete_confirm",
        nav_hint_text="↑/↓ elegir acción · ENTER confirmar · ESC volver",
        description_slot_rows=8,
    )
    if not selected:
        return False
    return selected.label.startswith("Eliminar definitivamente")


def _linked_shared_jsonl_outputs(manifest_path: str) -> list[Path]:
    """Resolve shared batch JSONL outputs linked to a manifest path."""
    manifest_stem = _sanitize_name(Path(manifest_path).stem)
    batch_dir = Path("data/processed/batch_results")
    if not batch_dir.exists() or not batch_dir.is_dir():
        return []
    return sorted(
        [
            path
            for path in batch_dir.glob(f"batch_{manifest_stem}_*.jsonl")
            if path.is_file()
        ],
        key=lambda value: value.name,
    )


def _delete_manifest_file(manifest_path: str) -> tuple[bool, str | None]:
    """Delete a manifest file and linked shared batch JSONL outputs."""
    target = Path(manifest_path)
    if not target.exists() or not target.is_file():
        return False, "El archivo no existe o no es un manifiesto válido."

    linked_outputs = _linked_shared_jsonl_outputs(manifest_path)

    try:
        target.unlink()
        for output_file in linked_outputs:
            if output_file.is_file():
                output_file.unlink()
    except OSError as error:
        return False, str(error)
    return True, None


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

    model_variants = cast(list[dict[str, Any]], config.get("model_variants") or [])
    model_count = len({str(item.get("model_id") or "").strip() for item in model_variants if str(item.get("model_id") or "").strip()})
    schema_name = cast(str, config["schema_name"])
    runs: list[dict[str, Any]] = []
    statuses: list[str] = []
    total_pending = 0
    for variant in model_variants:
        model = str(variant.get("model_id") or "").strip()
        include_reasoning = bool(variant.get("include_reasoning"))
        if not model:
            continue
        schema_exec_name = execution_schema_name(schema_name, include_reasoning)
        snapshot = manifest_execution_snapshot(
            manifest_path=manifest_path,
            model_tag=model,
            schema_name=schema_name,
            include_reasoning=include_reasoning,
        )
        statuses.append(str(snapshot.get("status") or "red"))
        total_pending += int(snapshot.get("pending", 0))
        runs.append(
            {
                "model": model,
                "include_reasoning": include_reasoning,
                "schema_exec_name": schema_exec_name,
                "snapshot": snapshot,
            }
        )

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
            f"schema={config['schema_name']} · "
            f"modelos={model_count} · variantes={len(model_variants)} · "
            f"iteraciones={int(config.get('iterations_per_image', 1))}x · "
            f"pendientes={total_pending}"
        ),
    }


def _generate_manifest_with_prompt(
    kit: "UIKit",
    app: "AppContext",
    *,
    subtitle: str,
    make_header: Callable[[str], Callable[[], None]],
    select_schema_base: Callable[..., str | None],
    source_manifest_path: str | None = None,
    source_seed: int = 42,
    initial_sample_size: int | None = None,
    initial_iterations_per_image: int | None = None,
    initial_model_variants: list[dict[str, Any]] | None = None,
    initial_schema_name: str | None = None,
) -> dict[str, Any] | None:
    """Interactively generate a new autosufficient execution manifest.

    Args:
        kit: Terminal UI toolkit.
        app: Application context used by selectors.
        subtitle: Subtitle for generation screen.
        make_header: Header factory for interactive menus.
        select_schema_base: Callback for base schema selection.
        source_manifest_path: Optional source manifest for derivation.
        source_seed: Seed inherited from source manifest.

    Returns:
        Manifest generation summary or `None` when cancelled.
    """
    from .manifest_generation import generate_manifest

    sample_size = _select_manifest_sample_size_with_default(kit, initial_value=initial_sample_size)
    if sample_size == "BACK":
        return None

    iterations_per_image = _select_manifest_iterations_per_image_with_default(
        kit,
        initial_value=initial_iterations_per_image,
    )
    if iterations_per_image == "BACK":
        return None

    model_selection = select_model_variants(
        kit,
        app,
        menu_id="manifest_models_selector",
        subtitle="MANIFEST CONFIG · SELECT MODEL VARIANTS",
        make_header_fn=make_header,
        initial_model_variants=initial_model_variants,
    )
    if not model_selection:
        kit.log("Generación cancelada: debes seleccionar al menos una variante de modelo.", "warning")
        return None
    _models, model_variants = model_selection

    try:
        base_schema_name = select_schema_base(
            "manifest",
            "MANIFEST CONFIG",
            initial_schema_name=initial_schema_name,
        )
    except TypeError:
        base_schema_name = select_schema_base("manifest", "MANIFEST CONFIG")
    if base_schema_name is None:
        return None

    selected_size = cast(int, sample_size)
    selected_iterations = cast(int, iterations_per_image)
    output_dir = Path("data/experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if source_manifest_path:
        source_stem = Path(source_manifest_path).stem
        output_path = (output_dir / f"zeroshot_manifest_{timestamp}_from_{source_stem}.jsonl").resolve()
    else:
        output_path = (output_dir / f"zeroshot_manifest_{timestamp}.jsonl").resolve()
    seed_value = safe_positive_int(source_seed, default=42)

    kit.clear()
    app.print_banner()
    kit.subtitle(subtitle)
    if source_manifest_path:
        kit.log(f"Modo derivado desde: {source_manifest_path}", "info")
        kit.log(f"Semilla heredada (fija): {seed_value}", "info")
    kit.log(
        f"Generando manifiesto estratificado (n={selected_size}) listo para ejecución automática...",
        "step",
    )

    summary = generate_manifest(
        input_csv=Path("data/raw/m_train/train.csv").resolve(),
        images_dir=Path("data/processed/m_train/images").resolve(),
        output_path=output_path,
        sample_size=selected_size,
        seed=seed_value,
        stratify_col="cls",
        id_col="image_id",
        label_col="cls",
        relative_paths=False,
        run_schema_name=base_schema_name,
        run_model_variants=model_variants,
        run_iterations_per_image=selected_iterations,
        run_derived_from=source_manifest_path,
    )
    kit.log(f"Manifiesto generado: {summary.get('output_path')}", "success")
    kit.log(f"Iteraciones por imagen: {summary.get('run_iterations_per_image', 1)}x", "info")
    kit.log(f"Semilla de muestreo: {summary.get('run_seed', 42)}", "info")
    variant_labels = [
        variant_label(str(item.get("model_id") or ""), bool(item.get("include_reasoning")))
        for item in cast(list[dict[str, Any]], summary.get("run_model_variants") or [])
    ]
    if variant_labels:
        kit.log("Variantes seleccionadas: " + " | ".join(variant_labels), "info")
    kit.log("Distribución del subconjunto: " + _format_class_distribution(summary), "info")
    return summary


def select_manifest_for_batch(
    kit: "UIKit",
    app: "AppContext",
    *,
    make_header: Callable[[str], Callable[[], None]],
    select_schema_base: Callable[..., str | None],
) -> dict[str, Any] | None:
    """Select an existing autosufficient manifest or generate a new one.

    Args:
        kit: Terminal UI toolkit.
        app: Application context used by model selector and UI rendering.
        make_header: Header factory for interactive menus.
        select_schema_base: Callback for schema base selection.

    Returns:
        Dictionary with selected manifest path, overview and run config, or `None`.
    """
    while True:
        manifests = discover_experiment_manifests()
        if not manifests:
            kit.log("No hay manifiestos detectados. Se iniciará la generación automática.", "warning")
            generated_summary = _generate_manifest_with_prompt(
                kit,
                app,
                subtitle="BATCH RUNNER · GENERATE MANIFEST",
                make_header=make_header,
                select_schema_base=select_schema_base,
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
        options.append(
            kit.MenuItem(
                "Modificar manifiesto existente",
                description=(
                    "Regenera un manifiesto reutilizando la semilla del origen y permitiendo "
                    "cambiar cantidad, iteraciones, modelo y variantes de razonamiento."
                ),
            )
        )
        options.append(
            kit.MenuItem(
                "Eliminar manifiesto",
                description="Borra un manifiesto JSONL existente de data/experiments.",
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
                select_schema_base=select_schema_base,
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

        if selection.label.startswith("Modificar manifiesto existente"):
            source_manifest = _select_source_manifest_to_modify(
                kit,
                manifests=manifests,
                overviews=overviews,
                make_header=make_header,
            )
            if not source_manifest:
                return None
            source_config = extract_manifest_run_config(source_manifest)
            if source_config is None:
                kit.log("No se pudo extraer configuración del manifiesto origen.", "error")
                kit.wait("Press any key to return to tests manager...")
                return None
            source_seed = safe_positive_int(source_config.get("seed"), default=42)
            source_iterations = safe_positive_int(source_config.get("iterations_per_image"), default=1)
            source_model_variants = normalize_model_variants(source_config.get("model_variants"))
            source_schema_name = str(source_config.get("schema_name") or "").strip() or None
            source_sample_size = _estimate_manifest_sample_size(
                manifest_path=source_manifest,
                iterations_per_image=source_iterations,
            )
            generated_summary = _generate_manifest_with_prompt(
                kit,
                app,
                subtitle="BATCH RUNNER · DERIVE MANIFEST",
                make_header=make_header,
                select_schema_base=select_schema_base,
                source_manifest_path=source_manifest,
                source_seed=source_seed,
                initial_sample_size=source_sample_size,
                initial_iterations_per_image=source_iterations,
                initial_model_variants=source_model_variants,
                initial_schema_name=source_schema_name,
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

        if selection.label.startswith("Eliminar manifiesto"):
            source_manifest = _select_source_manifest_to_delete(
                kit,
                manifests=manifests,
                overviews=overviews,
                make_header=make_header,
            )
            if not source_manifest:
                continue
            if not _confirm_manifest_deletion(
                kit,
                manifest_path=source_manifest,
                make_header=make_header,
            ):
                kit.log("Eliminación cancelada.", "info")
                continue
            deleted, error_message = _delete_manifest_file(source_manifest)
            if not deleted:
                kit.log(f"No se pudo eliminar el manifiesto: {error_message or 'error desconocido'}", "error")
                kit.wait("Press any key to continue...")
                continue
            kit.log(f"Manifiesto eliminado: {source_manifest}", "success")
            kit.wait("Press any key to refresh manifest list...")
            continue

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
