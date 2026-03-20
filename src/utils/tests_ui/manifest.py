from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, cast

from ..setup_ui_io import ask_choice, ask_text
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
    """
    Construye un payload de configuración de ejecución canónico utilizado por el wrapper de lotes.
    
    Args:
        schema_name: Nombre del esquema.
        iterations_per_image: Número de iteraciones por imagen.
        seed: Semilla aleatoria.
        model_variants: Variantes del modelo.
        
    Returns:
        Diccionario con la configuración normalizada o `None` si falta información crítica.
    """
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
    """
    Lee los metadatos globales del manifiesto si están presentes en las primeras líneas del JSONL.

    Args:
        manifest_path: Ruta al archivo JSONL del manifiesto.

    Returns:
        Diccionario de metadatos o `None` si no se encuentran.
    """
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
    """
    Itera líneas JSONL devolviendo cada línea y su payload dict parseado.

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
    """
    Carga conteos por clase para estimar el reparto estratificado.

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


def discover_experiment_manifests() -> list[str]:
    """
    Descubre los manifiestos de experimentos disponibles en data/experiments.

    Returns:
        Rutas absolutas a los archivos JSONL de los manifiestos, ordenadas por los más recientes.
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
    """
    Normaliza cadenas para que sean seguras para nombres de archivo.

    Args:
        name: Token en crudo a normalizar.

    Returns:
        Token seguro para nombres de archivo.
    """
    allowed = []
    for char in name:
        if char.isalnum() or char in ("-", "_"):
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_") or "run"


def _output_schema_name(schema_name: str) -> str:
    """
    Normaliza schema para nombre de archivo de salida compartido.

    Args:
        schema_name: Nombre del esquema a normalizar.

    Returns:
        Nombre del esquema normalizado.
    """
    value = str(schema_name or "").strip()
    suffix = "WithReasoning"
    if value.endswith(suffix) and len(value) > len(suffix):
        return value[: -len(suffix)]
    return value


def linked_batch_output_path(*, manifest_path: str, schema_name: str) -> Path:
    """
    Construye una ruta JSONL de salida estable vinculada al manifiesto y al esquema.

    Args:
        manifest_path: Ruta al JSONL del manifiesto.
        schema_name: Nombre del esquema de ejecución efectivo.

    Returns:
        Ruta absoluta de salida para los resultados JSONL por lotes.
    """
    manifest_stem = _sanitize_name(Path(manifest_path).stem)
    schema_token = _sanitize_name(_output_schema_name(schema_name))
    output_dir = Path("data/processed/batch_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    return (output_dir / f"batch_{manifest_stem}_{schema_token}.jsonl").resolve()


def load_manifest_entries(manifest_path: str) -> list[dict[str, Any]]:
    """
    Carga filas válidas de un archivo JSONL de manifiesto.

    Args:
        manifest_path: Ruta al JSONL del manifiesto.

    Returns:
        Lista de diccionarios de filas válidas parseadas que contienen al menos `image_path`.
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
    """
    Genera una clave estable por entrada soportando iteraciones repetidas.
    
    Args:
        entry: Entrada del manifiesto.
        
    Returns:
        Clave estable por entrada.
    """
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
    """
    Indexa el último registro de resultado por clave de entrada filtrado por ID de modelo.

    Args:
        output_path: Ruta al JSONL de salida por lotes.
        model_tag: Identificador del modelo para filtrar.
        schema_name: Nombre del esquema para filtrar.
        include_reasoning: Si se incluye el razonamiento en la variante.

    Returns:
        Mapeo de `(image_path, run_iteration_index) -> último registro visto`.
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
    """
    Elimina los registros de un modelo específico de un JSONL de salida por lotes compartido.

    Args:
        output_path: Ruta del JSONL de salida compartido.
        model_tag: ID del modelo cuyas filas deben ser eliminadas.
        schema_name: Nombre base del esquema compartido por las variantes.
        include_reasoning: Flag de modo de variante utilizado para aislar filas.

    Returns:
        `True` cuando la reescritura tiene éxito.
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
    """
    Determina si un registro de salida se considera exitoso.

    Args:
        record: Fila de resultado parseada.

    Returns:
        `True` cuando el estado es `ok` y el payload contiene contenido significativo.
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
    """
    Calcula la instantánea del semáforo de ejecución para una ejecución de manifiesto/modelo/esquema.

    Args:
        manifest_path: Ruta al JSONL del manifiesto.
        model_tag: Etiqueta del modelo de LM Studio.
        schema_name: Nombre base del esquema.
        include_reasoning: Flag de modo de variante.

    Returns:
        Diccionario de instantánea con estado, contadores, entradas pendientes y ruta de salida.
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
    """
    Renderiza una etiqueta coloreada para el estado de la instantánea del manifiesto.

    Args:
        kit: Toolkit de interfaz de usuario de terminal.
        snapshot: Diccionario de instantánea producido por `manifest_execution_snapshot`.

    Returns:
        Etiqueta de estado coloreada con ANSI.
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


def _manifest_dashboard_metrics(overview: dict[str, Any]) -> dict[str, int | float]:
    """
    Calcula métricas agregadas para pintar badges de dashboard por manifiesto.

    Args:
        overview: Resumen devuelto por ``manifest_overview``.

    Returns:
        Diccionario con ``ok``, ``total``, ``pending`` y ``completion_pct``.
    """
    runs = cast(list[dict[str, Any]], overview.get("runs") or [])
    total_ok = 0
    total_total = 0
    total_pending = 0
    for run in runs:
        snapshot = cast(dict[str, Any], run.get("snapshot") or {})
        total_ok += int(snapshot.get("ok", 0))
        total_total += int(snapshot.get("total", 0))
        total_pending += int(snapshot.get("pending", 0))

    completion_pct = 0.0
    if total_total > 0:
        completion_pct = max(0.0, min(100.0, (float(total_ok) / float(total_total)) * 100.0))

    return {
        "ok": total_ok,
        "total": total_total,
        "pending": total_pending,
        "completion_pct": completion_pct,
    }


def _manifest_dashboard_badges(kit: "UIKit", overview: dict[str, Any]) -> str:
    """
    Construye badges compactos de progreso para mostrar junto a cada manifiesto.

    Args:
        kit: Toolkit de UI para estilos ANSI.
        overview: Resumen del manifiesto.

    Returns:
        Cadena con badges de ``ok``, ``pending`` y ``% completado``.
    """
    style = kit.style
    metrics = _manifest_dashboard_metrics(overview)
    ok_value = int(metrics["ok"])
    pending_value = int(metrics["pending"])
    total_value = int(metrics["total"])
    pct_value = float(metrics["completion_pct"])

    ok_color = str(getattr(style, "OKGREEN", ""))
    pending_color = str(getattr(style, "WARNING", "")) if pending_value > 0 else str(getattr(style, "DIM", ""))
    if pct_value >= 100.0 and total_value > 0:
        pct_color = str(getattr(style, "OKGREEN", ""))
    elif pct_value <= 0.0:
        pct_color = str(getattr(style, "FAIL", ""))
    else:
        pct_color = str(getattr(style, "OKCYAN", ""))
    endc = str(getattr(style, "ENDC", ""))

    return (
        f"{ok_color}[ok {ok_value}/{total_value}]{endc} "
        f"{pending_color}[pending {pending_value}]{endc} "
        f"{pct_color}[{pct_value:>5.1f}%]{endc}"
    )


def _truncate_plain_text(text: str, max_len: int) -> str:
    """
    Recorta texto plano (sin ANSI) para adaptar labels a anchos reducidos.

    Args:
        text: Texto original.
        max_len: Longitud máxima visible.

    Returns:
        Texto recortado con elipsis cuando sea necesario.
    """
    value = str(text or "")
    if max_len <= 0:
        return ""
    if len(value) <= max_len:
        return value
    if max_len <= 1:
        return value[:max_len]
    return value[: max_len - 1] + "…"


def _build_manifest_row_label(
    *,
    kit: "UIKit",
    manifest_path: str,
    overview: dict[str, Any],
) -> str:
    """
    Construye un label de fila dinámico y compacto según el ancho de terminal.

    Args:
        kit: Toolkit de interfaz con acceso a ancho actual.
        manifest_path: Ruta del manifiesto.
        overview: Resumen del manifiesto.

    Returns:
        Label una-línea ajustado al ancho disponible.
    """
    metrics = _manifest_dashboard_metrics(overview)
    status = snapshot_status_text(overview)

    manifest_name = os.path.basename(manifest_path)
    # Reserva para indicadores compactos de dashboard al final de la fila.
    right_plain = (
        f"OK {int(metrics['ok'])}/{int(metrics['total'])} · "
        f"P {int(metrics['pending'])} · "
        f"{float(metrics['completion_pct']):.1f}%"
    )

    # Ajuste dinámico: margen para prefijos del engine (> , indent, etc.).
    content_budget = max(36, int(kit.width()) - 14)
    left_budget = max(14, content_budget - len(right_plain) - 3)
    left_plain = _truncate_plain_text(f"{manifest_name} · {status}", left_budget)

    style = kit.style
    ok_color = str(getattr(style, "OKGREEN", ""))
    pending_color = (
        str(getattr(style, "WARNING", ""))
        if int(metrics["pending"]) > 0
        else str(getattr(style, "DIM", ""))
    )
    pct_value = float(metrics["completion_pct"])
    if pct_value >= 100.0 and int(metrics["total"]) > 0:
        pct_color = str(getattr(style, "OKGREEN", ""))
    elif pct_value <= 0.0:
        pct_color = str(getattr(style, "FAIL", ""))
    else:
        pct_color = str(getattr(style, "OKCYAN", ""))
    endc = str(getattr(style, "ENDC", ""))

    right_colored = (
        f"{ok_color}OK {int(metrics['ok'])}/{int(metrics['total'])}{endc} · "
        f"{pending_color}P {int(metrics['pending'])}{endc} · "
        f"{pct_color}{float(metrics['completion_pct']):.1f}%{endc}"
    )
    return f"{left_plain} · {right_colored}"


def create_pending_manifest(*, pending_entries: list[dict[str, Any]], source_manifest_path: str) -> str | None:
    """
    Escribe un manifiesto temporal solo con las entradas pendientes para una reejecución.

    Args:
        pending_entries: Entradas que aún están pendientes o fallidas.
        source_manifest_path: Ruta original del manifiesto.

    Returns:
        Ruta del manifiesto temporal, o `None` cuando no hay nada pendiente.
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
    """
    Formatea el texto compacto de distribución de clases a partir del resumen de generación.

    Args:
        summary: Diccionario de resumen de generación de manifiesto.

    Returns:
        Cadena de distribución de clases legible por humanos.
    """
    counts = cast(dict[str, Any], summary.get("written_by_class") or {})
    if not counts:
        return "sin datos"
    parts = [f"{cls}={counts[cls]}" for cls in sorted(counts.keys())]
    return " | ".join(parts)


def _estimate_manifest_sample_size(*, manifest_path: str, iterations_per_image: int) -> int:
    """
    Estima el tamaño original de la muestra a partir de las filas del manifiesto y los metadatos de iteración.
    
    Args:
        manifest_path: Ruta al JSONL del manifiesto.
        iterations_per_image: Número de iteraciones por imagen.
        
    Returns:
        Tamaño estimado de la muestra original.
    """
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
    """
    Solicita tamaño de muestra con un valor inicial precargado.
    
    Args:
        kit: Toolkit de interfaz de usuario de terminal.
        initial_value: Valor inicial para el campo de entrada.
        
    Returns:
        Tamaño de muestra confirmado o mensaje de error.
    """
    class_counts = _load_manifest_class_counts(Path("data/raw/m_train/train.csv"))
    total_available = sum(class_counts.values())

    from .manifest_generation import _compute_target_counts
    total_distribution = " | ".join(
        f"{cls_name}={class_counts[cls_name]}" for cls_name in sorted(class_counts.keys())
    )

    def _validate_sample_size(raw_value: str) -> str | None:
        """
        Valida el tamaño de muestra introducido por el usuario.
        
        Args:
            raw_value: Valor introducido por el usuario.
            
        Returns:
            Mensaje de error si la validación falla, None en caso contrario.
        """
        if not raw_value:
            return "Debes indicar un número entero positivo."
        if not raw_value.isdigit():
            return "Entrada inválida. Usa solo dígitos."
        if int(raw_value) <= 0:
            return "El tamaño debe ser mayor que cero."
        return None

    def _render_preview_lines(current_value: str, _ui_width: int) -> list[str]:
        """
        Renderiza las líneas de preview para el tamaño de muestra.
        
        Args:
            current_value: Valor actual del campo de entrada.
            _ui_width: Ancho de la interfaz de usuario.
            
        Returns:
            Lista de líneas de preview.
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
    """
    Solicita iteraciones con valor inicial precargado.
    
    Args:
        kit: Toolkit de interfaz de usuario de terminal.
        initial_value: Valor inicial para el campo de entrada.
        
    Returns:
        Número de iteraciones confirmado o mensaje de error.
    """

    def _validate_iterations(raw_value: str) -> str | None:
        """
        Valida el número de iteraciones introducido por el usuario.
        
        Args:
            raw_value: Valor introducido por el usuario.
            
        Returns:
            Mensaje de error si la validación falla, None en caso contrario.
        """
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
        """
        Renderiza las líneas de preview para el número de iteraciones.
        
        Args:
            current_value: Valor actual del campo de entrada.
            _ui_width: Ancho de la interfaz de usuario.
            
        Returns:
            Lista de líneas de preview.
        """
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
    """
    Extrae la configuración de ejecución embebida de la primera fila del manifiesto.

    Args:
        manifest_path: Ruta del JSONL del manifiesto.

    Returns:
        Diccionario de configuración con ajustes de modelos y esquema, o `None`.
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
    """
    Construye el nombre público del esquema utilizado en el estado y los informes.

    Args:
        schema_name: Nombre base del esquema.
        include_reasoning: Si está habilitada la variante con razonamiento.

    Returns:
        Nombre de esquema efectivo.
    """
    if include_reasoning and not schema_name.endswith("WithReasoning"):
        return f"{schema_name}WithReasoning"
    return schema_name


def _find_manifest_by_selected_label(*, manifests: list[str], selected_label: str) -> str | None:
    """
    Resuelve la ruta de manifiesto a partir de la etiqueta visible en el menú.

    Args:
        manifests: Lista de rutas candidatas.
        selected_label: Etiqueta seleccionada en UI.

    Returns:
        Ruta completa del manifiesto o ``None`` si no coincide.
    """
    base_label = selected_label.replace("Usar ", "", 1).split(" · ", 1)[0].strip()
    for manifest_path in manifests:
        if os.path.basename(manifest_path) == base_label:
            return manifest_path
    return None


def _show_manifest_overview_screen(
    kit: "UIKit",
    app: "AppContext",
    *,
    manifest_path: str,
    overview: dict[str, Any],
    make_header: Callable[[str], Callable[[], None]],
) -> None:
    """
    Muestra una vista resumida del manifiesto seleccionado y su estado por variante.

    Args:
        kit: Toolkit de interfaz de usuario.
        app: Contexto de aplicación para banner.
        manifest_path: Ruta del manifiesto seleccionado.
        overview: Resumen calculado por ``manifest_overview``.
        make_header: Factoría de cabecera visual.
    """
    style = kit.style

    def _table_lines(
        headers: list[str],
        rows: list[list[str]],
        *,
        width: int,
        status_col: int | None = None,
    ) -> list[str]:
        """Renderiza tabla estática con ancho reactivo para el overview."""
        if not rows:
            return ["  Sin datos."]

        table_menu_fn = getattr(kit, "table_menu", None)
        table_column_cls = getattr(kit, "TableColumn", None)
        table_row_cls = getattr(kit, "TableRow", None)
        if not (callable(table_menu_fn) and table_column_cls is not None and table_row_cls is not None):
            return ["  Sin datos."]

        # Improve column distribution heuristics:
        # - Special-case common 2-column tables (Campo|Valor) to give the value column most space
        # - Give Variante and Schema larger min_width and higher weight
        # - Keep numeric/short columns compact
        two_col_value_like = len(headers) == 2 and headers[1].lower() in ("valor", "value")

        # Base weight mapping for known headers
        weight_map: dict[str, int] = {
            "Variante": 4,
            "Schema": 3,
            "%": 1,
            "Estado": 1,
            "OK/TOTAL": 1,
            "Pend": 1,
            "Err": 1,
        }

        if two_col_value_like:
            weights = [1, 5]
        else:
            weights = [weight_map.get(h, 1) for h in headers]

        total_weight = max(1, sum(weights))

        def _min_width_for(header: str, weight: int) -> int:
            if two_col_value_like:
                return 10 if header == headers[0] else max(20, int(kit.width() * 0.5))
            if header in ("Variante", "Schema"):
                return 22 if header == "Variante" else 18
            if header in ("OK/TOTAL", "Pend", "Err", "%"):
                return 8
            return 10

        columns = [
            table_column_cls(
                label=header,
                width_ratio=(weight / total_weight),
                min_width=_min_width_for(header, weight),
            )
            for header, weight in zip(headers, weights)
        ]
        table_rows: list[Any] = []
        for row in rows:
            normalized = [str(cell) for cell in row]
            colors: list[str | None] | None = None
            if status_col is not None and 0 <= status_col < len(normalized):
                status_token = normalized[status_col].upper()
                if "COMPLETO" in status_token or "OK" in status_token:
                    color_name = "OKGREEN"
                elif "PARCIAL" in status_token or "WARN" in status_token:
                    color_name = "WARNING"
                else:
                    color_name = "FAIL"
                typed_colors: list[str | None] = [None for _ in normalized]
                typed_colors[status_col] = color_name
                colors = typed_colors
            row_cells: list[Any] = [str(cell) for cell in normalized]
            table_rows.append(table_row_cls(cells=row_cells, cell_colors=colors))

        # Allow cell wrapping for value-like tables so long text wraps in the value column
        rendered = table_menu_fn(
            columns,
            table_rows,
            interactive=False,
            return_lines=True,
            width=width,
            max_cell_lines=True,
        )
        if isinstance(rendered, list):
            return [str(line) for line in rendered]
        return ["  Sin datos."]


    def _render_overview_panel(ui_width: int) -> None:
        """
        Renderiza el contenido completo del overview con el ancho indicado.

        Args:
            ui_width: Ancho actual de la terminal.
        """
        divider = " " + kit.divider(ui_width)

        print(divider)
        overview_rows = [
            ["Manifest", os.path.basename(manifest_path)],
            ["Estado", snapshot_status_text(overview)],
            ["Ruta", str(manifest_path)],
            ["Resumen", str(overview.get("description", "-"))],
        ]
        for line in _table_lines(["Campo", "Valor"], overview_rows, width=ui_width):
            print(line)
        print(divider)

        config = cast(dict[str, Any] | None, overview.get("config"))
        if config:
            schema_name = str(config.get("schema_name") or "-")
            iterations_value = safe_positive_int(config.get("iterations_per_image"), default=1)
            seed_value = safe_positive_int(config.get("seed"), default=42)
            config_rows = [
                ["Schema", schema_name],
                ["Iteraciones", f"{iterations_value}x"],
                ["Seed", str(seed_value)],
                ["Variantes", str(len(cast(list[dict[str, Any]], config.get('model_variants') or [])))],
            ]
            for line in _table_lines(["Config", "Valor"], config_rows, width=ui_width):
                print(line)
            print(divider)

        runs = cast(list[dict[str, Any]], overview.get("runs") or [])
        if runs:
            print(f" {style.BOLD}{style.OKCYAN}Estado por variante{style.ENDC}")

            variant_rows: list[list[str]] = []
            for run in runs:
                model = str(run.get("model") or "-")
                schema_exec = str(run.get("schema_exec_name") or "-")
                snapshot = cast(dict[str, Any], run.get("snapshot") or {})
                total = int(snapshot.get("total", 0))
                ok_count = int(snapshot.get("ok", 0))
                pending = int(snapshot.get("pending", 0))
                errors = int(snapshot.get("errors", 0))
                completion_pct = f"{(100.0 * ok_count / total):.1f}%" if total > 0 else "0.0%"
                variant_rows.append(
                    [
                        variant_label(model, bool(run.get("include_reasoning"))),
                        schema_exec,
                        snapshot_status_text(snapshot),
                        f"{ok_count}/{total}",
                        str(pending),
                        str(errors),
                        completion_pct,
                    ]
                )

            for line in _table_lines(
                ["Variante", "Schema", "Estado", "OK/TOTAL", "Pend", "Err", "%"],
                variant_rows,
                width=ui_width,
                status_col=2,
            ):
                print(line)

            print(divider)
        else:
            print(f" {style.WARNING}No hay variantes configuradas para este manifiesto.{style.ENDC}")
            print(divider)

        print(
            f" {style.DIM}Press any key para volver. Ajusta el ancho de la terminal "
            f"para reflow automático.{style.ENDC}"
        )

    def _redraw_for_width(current_width: int) -> None:
        """
        Redibuja cabecera y panel completo para el ancho actual.

        Args:
            current_width: Ancho actual de terminal.
        """
        try:
            header_fn = make_header("BATCH RUNNER · MANIFEST OVERVIEW")
            kit.clear()
            header_fn()
        except Exception:
            kit.clear()
            app.print_banner()
            kit.subtitle(" BATCH RUNNER · MANIFEST OVERVIEW")
        _render_overview_panel(current_width)

    _redraw_for_width(max(60, kit.width()))
    kit.render_and_wait_responsive(
        render_fn=_redraw_for_width,
        message="Press any key para volver al selector de acciones...",
        initial_render=False,
    )


def _select_manifest_action_horizontal(
    kit: "UIKit",
    *,
    manifest_path: str,
    overview: dict[str, Any],
) -> str | None:
    """
    Solicita la acción para un manifiesto usando selector horizontal.

    Args:
        kit: Toolkit de interfaz de usuario.
        manifest_path: Ruta del manifiesto seleccionado.
        overview: Resumen de estado del manifiesto.

    Returns:
        Acción elegida (``execute``, ``view``, ``modify``, ``delete``, ``back``)
        o ``None`` si se cancela.
    """
    actions = ["EJECUTAR", "VISUALIZAR", "MODIFICAR", "ELIMINAR", "VOLVER"]
    metrics = _manifest_dashboard_metrics(overview)
    selected_index = ask_choice(
        question=f"Manifest seleccionado: {os.path.basename(manifest_path)}",
        options=actions,
        default_index=0,
        style=kit.style,
        read_key_fn=kit.read_key,
        clear_screen_fn=kit.clear,
        info_text=(
            f"Estado: {snapshot_status_text(overview)}\n"
            f"OK/TOTAL: {int(metrics['ok'])}/{int(metrics['total'])} · "
            f"PENDING: {int(metrics['pending'])} · "
            f"COMPLETADO: {float(metrics['completion_pct']):.1f}%\n"
            f"{overview.get('description', '-')}\n"
            "Acciones horizontales: ←/→ cambiar · ENTER confirmar · ESC cancelar"
        ),
    )
    if selected_index is None:
        return None
    return {
        0: "execute",
        1: "view",
        2: "modify",
        3: "delete",
        4: "back",
    }.get(selected_index, "back")


def _confirm_manifest_deletion(
    kit: "UIKit",
    *,
    manifest_path: str,
    make_header: Callable[[str], Callable[[], None]],
) -> bool:
    """
    Muestra un paso de confirmación explícito antes de eliminar un archivo de manifiesto.
    
    Args:
        kit: Toolkit de interfaz de usuario de terminal.
        manifest_path: Ruta del archivo de manifiesto a eliminar.
        make_header: Función para crear el encabezado del menú.
        
    Returns:
        True si se confirma la eliminación, False en caso contrario.
    """
    manifest_name = os.path.basename(manifest_path)
    linked_output = _linked_shared_jsonl_output(manifest_path)
    linked_summary = (
        str(linked_output) if linked_output is not None else "No se detectó JSONL compartido vinculado."
    )

    kit.log(f"Manifest a eliminar: {manifest_path}", "warning")
    kit.log(f"JSONL compartido que se eliminará: {linked_summary}", "warning")

    # Renderizar encabezado del menú antes de preguntar
    try:
        header_fn = make_header("BATCH RUNNER · CONFIRM DELETE")
        kit.clear()
        header_fn()
    except Exception:
        # Si la factoría de cabecera no está disponible, continuar silenciosamente
        pass

    question = f"Eliminar definitivamente {manifest_name}?"
    info_text = linked_summary

    confirmed = kit.ask(question=question, default="n", info_text=info_text)
    return bool(confirmed)


def _linked_shared_jsonl_output(manifest_path: str) -> Path | None:
    """
    Resuelve la salida JSONL de lotes compartida vinculada a una ruta de manifiesto.
    
    Args:
        manifest_path: Ruta del archivo de manifiesto.
        
    Returns:
        Ruta del archivo JSONL compartido vinculado, o `None` si no existe.
    """
    manifest_stem = _sanitize_name(Path(manifest_path).stem)
    batch_dir = Path("data/processed/batch_results")
    if not batch_dir.exists() or not batch_dir.is_dir():
        return None
    candidates = sorted(
        [
            path
            for path in batch_dir.glob(f"batch_{manifest_stem}_*.jsonl")
            if path.is_file()
        ],
        key=lambda value: value.name,
    )
    return candidates[0] if candidates else None


def _delete_manifest_file(manifest_path: str) -> tuple[bool, str | None]:
    """
    Elimina un archivo de manifiesto y las salidas JSONL de lotes compartidas vinculadas.
    
    Args:
        manifest_path: Ruta del archivo de manifiesto a eliminar.
        
    Returns:
        Tupla de (éxito, error_mensaje).
    """
    target = Path(manifest_path)
    if not target.exists() or not target.is_file():
        return False, "El archivo no existe o no es un manifiesto válido."

    linked_output = _linked_shared_jsonl_output(manifest_path)

    try:
        target.unlink()
        if linked_output is not None and linked_output.is_file():
            linked_output.unlink()
    except OSError as error:
        return False, str(error)
    return True, None


def manifest_overview(manifest_path: str) -> dict[str, Any]:
    """
    Agrega el estado del manifiesto a través de todos los modelos de cola configurados.

    Args:
        manifest_path: Ruta del JSONL del manifiesto.

    Returns:
        Diccionario de resumen con estado global, ejecuciones por modelo y descripción.
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
    """
    Genera interactivamente un nuevo manifiesto de ejecución autosuficiente.

    Args:
        kit: Toolkit de interfaz de usuario de terminal.
        app: Contexto de la aplicación utilizado por los selectores.
        subtitle: Subtítulo para la pantalla de generación.
        make_header: Factoría de cabeceras para menús interactivos.
        select_schema_base: Callback para la selección del esquema base.
        source_manifest_path: Manifiesto de origen opcional para la derivación.
        source_seed: Semilla heredada del manifiesto original.

    Returns:
        Resumen de generación de manifiesto o `None` cuando se cancela.
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
    """
    Selecciona un manifiesto autosuficiente existente o genera uno nuevo.

    Args:
        kit: Toolkit de interfaz de usuario de terminal.
        app: Contexto de la aplicación utilizado por el selector de modelos y el renderizado de la UI.
        make_header: Factoría de cabeceras para menús interactivos.
        select_schema_base: Callback para la selección de la base del esquema.

    Returns:
        Diccionario con la ruta del manifiesto seleccionado, resumen y configuración de ejecución, o `None`.
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
            dashboard_badges = _manifest_dashboard_badges(kit, overview)
            item = kit.MenuItem(
                "",
                description=(
                    f"{path}\n"
                    f"{dashboard_badges}\n"
                    f"{overview.get('description', '-')}"
                ),
            )
            setattr(item, "manifest_path", path)
            item.dynamic_label = (
                lambda _is_selected, _path=path, _overview=overview:
                _build_manifest_row_label(kit=kit, manifest_path=_path, overview=_overview)
            )
            options.append(item)
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
            nav_hint_text=(
                "↑/↓ elegir manifiesto · ENTER abrir acciones horizontales · ESC volver"
            ),
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

        selected_manifest = str(getattr(selection, "manifest_path", "") or "").strip()
        if not selected_manifest:
            selected_manifest = _find_manifest_by_selected_label(
                manifests=manifests,
                selected_label=selection.label,
            ) or ""
        if not selected_manifest:
            return None

        selected_overview = overviews[selected_manifest]
        selected_action = _select_manifest_action_horizontal(
            kit,
            manifest_path=selected_manifest,
            overview=selected_overview,
        )
        if selected_action is None or selected_action == "back":
            continue

        if selected_action == "execute":
            return {
                "manifest_path": selected_manifest,
                "overview": selected_overview,
                "config": selected_overview.get("config"),
            }

        if selected_action == "view":
            _show_manifest_overview_screen(
                kit,
                app,
                manifest_path=selected_manifest,
                overview=selected_overview,
                make_header=make_header,
            )
            continue

        if selected_action == "modify":
            source_config = extract_manifest_run_config(selected_manifest)
            if source_config is None:
                kit.log("No se pudo extraer configuración del manifiesto origen.", "error")
                kit.wait("Press any key to return to tests manager...")
                return None
            source_seed = safe_positive_int(source_config.get("seed"), default=42)
            source_iterations = safe_positive_int(source_config.get("iterations_per_image"), default=1)
            source_model_variants = normalize_model_variants(source_config.get("model_variants"))
            source_schema_name = str(source_config.get("schema_name") or "").strip() or None
            source_sample_size = _estimate_manifest_sample_size(
                manifest_path=selected_manifest,
                iterations_per_image=source_iterations,
            )
            generated_summary = _generate_manifest_with_prompt(
                kit,
                app,
                subtitle="BATCH RUNNER · DERIVE MANIFEST",
                make_header=make_header,
                select_schema_base=select_schema_base,
                source_manifest_path=selected_manifest,
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

        if selected_action == "delete":
            if not _confirm_manifest_deletion(
                kit,
                manifest_path=selected_manifest,
                make_header=make_header,
            ):
                kit.log("Eliminación cancelada.", "info")
                continue
            deleted, error_message = _delete_manifest_file(selected_manifest)
            if not deleted:
                kit.log(f"No se pudo eliminar el manifiesto: {error_message or 'error desconocido'}", "error")
                kit.wait("Press any key to continue...")
                continue
            kit.log(f"Manifiesto eliminado: {selected_manifest}", "success")
            kit.wait("Press any key to refresh manifest list...")
            continue

        continue
