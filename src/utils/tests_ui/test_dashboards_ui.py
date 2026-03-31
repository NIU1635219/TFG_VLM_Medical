"""Helpers visuales reutilizables para los dashboards del menu de tests."""

from __future__ import annotations

from collections import deque
import json
import os
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from ..menu_kit import AppContext, UIKit


def _format_metric_value(value: object, *, suffix: str = "") -> str:
    """
    Representa metricas opcionales de forma amigable para la TUI.

    Args:
        value: Valor de la métrica (puede ser None).
        suffix: Sufijo a añadir al valor (ej. " s").

    Returns:
        Representación en string de la métrica.
    """
    if value is None:
        return "N/D"
    if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
        return f"{int(value)}{suffix}"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".") + suffix
    return f"{value}{suffix}"


def _build_partial_metrics_line(summary: dict[str, object]) -> str:
    """
    Construye línea estándar de promedios parciales TTFT/TPS/latencia.

    Args:
        summary: Resumen de métricas.

    Returns:
        Línea con los promedios parciales.
    """
    ttft = cast(dict[str, object], summary.get("ttft") or {})
    tps = cast(dict[str, object], summary.get("tps") or {})
    duration = cast(dict[str, object], summary.get("total_duration") or {})
    return (
        "Promedios parciales: "
        f"TTFT={_format_metric_value(ttft.get('avg'), suffix=' s')} │ "
        f"TPS={_format_metric_value(tps.get('avg'))} │ "
        f"latencia={_format_metric_value(duration.get('avg'), suffix=' s')}"
    )


def _style_token(kit: "UIKit", name: str) -> str:
    """
    Devuelve un token ANSI si existe en el estilo actual.

    Args:
        kit: Toolkit de interfaz de usuario de terminal.
        name: Nombre del token de estilo.

    Returns:
        Token ANSI si existe, string vacío si no.
    """
    return str(getattr(kit.style, name, ""))


def _wrap_text(kit: "UIKit", text: str, width: int) -> list[str]:
    """
    Envuelve texto con el helper del kit o usa un fallback simple en tests.

    Args:
        kit: Toolkit de interfaz de usuario de terminal.
        text: Texto a envolver.
        width: Ancho máximo por línea.

    Returns:
        Lista de líneas envueltas.
    """
    wrap_fn = getattr(kit, "wrap", None)
    if callable(wrap_fn):
        wrapped = wrap_fn(text, width)
        if isinstance(wrapped, list) and wrapped:
            return [str(line) for line in wrapped]
    if len(text) <= width:
        return [text]
    return [text[index:index + width] for index in range(0, len(text), width)]


def _stringify_payload_value(value: object) -> str:
    """
    Normaliza valores JSON para mostrarlos en líneas legibles.

    Args:
        value: Valor a convertir a string.

    Returns:
        String normalizado.
    """
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)
    text = str(value)
    if "\n" in text or "\r" in text:
        text = " ".join(text.replace("\r", "\n").split())
    return text


def _format_payload_lines(kit: "UIKit", payload: object, width: int) -> list[str]:
    """
    Convierte un payload en líneas con wrapping conservando claves.

    Args:
        kit: Toolkit de interfaz de usuario de terminal.
        payload: Payload a formatear.
        width: Ancho máximo por línea.

    Returns:
        Lista de líneas formateadas.
    """
    if payload is None:
        return[]

    if isinstance(payload, dict):
        lines: list[str] =[]
        for key, value in payload.items():
            value_text = _stringify_payload_value(value)
            key_prefix = f"{key}="
            available_width = max(10, width - len(key_prefix))
            wrapped = _wrap_text(kit, value_text, available_width)
            if not wrapped:
                lines.append(key_prefix)
                continue
            lines.append(key_prefix + wrapped[0])
            for chunk in wrapped[1:]:
                lines.append(" " * len(key_prefix) + chunk)
        return lines

    return _wrap_text(kit, _stringify_payload_value(payload), width)


def _recent_record_separator(kit: "UIKit", *, ui_width: int | None = None) -> str:
    """
    Crea una linea divisoria para separar registros recientes.

    Args:
        kit: Toolkit de interfaz de usuario de terminal.
        ui_width: Ancho de terminal ya capturado para el frame actual.

    Returns:
        Línea divisoria. 
    """
    dim = _style_token(kit, "DIM")
    endc = _style_token(kit, "ENDC")
    width = kit.width() if ui_width is None else int(ui_width)
    return f"  {dim}{'─' * max(2, width - 4)}{endc}"


def _cap_detail_lines(lines: list[str], *, max_lines: int) -> list[str]:
    """
    Recorta lineas muy largas para evitar cortes a mitad de registro.

    Args:
        lines: Líneas a recortar.
        max_lines: Número máximo de líneas a mantener.

    Returns:
        Líneas recortadas.
    """
    if max_lines <= 0 or len(lines) <= max_lines:
        return lines
    if max_lines == 1:
        return [lines[0]]
    trimmed = lines[: max_lines - 1]
    trimmed.append("... (truncado)")
    return trimmed


def _coerce_int(value: object) -> int:
    """
    Convierte valores numericos opcionales del callback de progreso a enteros.

    Args:
        value: Valor a convertir.

    Returns:
        Valor entero.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _telemetry_available(availability: dict[str, object], key: str) -> bool:
    """
    Indica si una metrica tiene al menos un registro valido en la ejecucion.

    Args:
        availability: Disponibilidad de métricas.
        key: Clave de la métrica.

    Returns:
        True si la métrica está disponible, False en caso contrario.
    """
    return _coerce_int(availability.get(key)) > 0


def _coverage_fragments(availability: dict[str, object]) -> list[str]:
    """
    Construye una linea compacta de cobertura omitiendo metricas ausentes.

    Args:
        availability: Disponibilidad de métricas.

    Returns:
        Línea compacta de cobertura.
    """
    ok_records = _coerce_int(availability.get("ok_records"))
    if ok_records <= 0:
        return []

    fragments: list[str] =[]
    metric_specs =[
        ("TTFT", "ttft_records"),
        ("TPS", "tps_records"),
        ("GPU", "gpu_layer_records"),
    ]
    for label, key in metric_specs:
        count = _coerce_int(availability.get(key))
        if count > 0:
            fragments.append(f"{label} {count}/{ok_records}")
    return fragments


def _build_probe_detail_parts(record: dict[str, object], availability: dict[str, object]) -> list[str]:
    """
    Genera solo las columnas utiles para una inferencia concreta del probe.

    Args:
        record: Registro de inferencia.
        availability: Disponibilidad de métricas.

    Returns:
        Lista de partes de la línea de detalle.
    """
    parts =[f"TTFT={_format_metric_value(record.get('ttft_seconds'), suffix=' s')}"]

    if _telemetry_available(availability, "tps_records"):
        parts.append(f"TPS={_format_metric_value(record.get('tokens_per_second'))}")
    if record.get("generation_duration_seconds") is not None:
        parts.append(f"gen={_format_metric_value(record.get('generation_duration_seconds'), suffix=' s')}")
    if _telemetry_available(availability, "prompt_token_records"):
        parts.append(f"prompt={_format_metric_value(record.get('prompt_tokens'))}")
    if _telemetry_available(availability, "completion_token_records"):
        parts.append(f"output={_format_metric_value(record.get('completion_tokens'))}")
    if _telemetry_available(availability, "reasoning_records"):
        parts.append(f"reasoning_json={_format_metric_value(record.get('reasoning_tokens'))}")
    if record.get("stop_reason") is not None:
        parts.append(f"stop={_format_metric_value(record.get('stop_reason'))}")
    if _telemetry_available(availability, "gpu_layer_records"):
        parts.append(f"gpu_layers={_format_metric_value(record.get('gpu_layers'))}")
    if _telemetry_available(availability, "total_duration_records"):
        parts.append(f"total={_format_metric_value(record.get('total_duration_seconds'), suffix=' s')}")

    return parts


def _rel_probe_path(image_path: str | None) -> str:
    """
    Normaliza rutas de imagenes para mostrarlas de forma compacta.

    Args:
        image_path: Ruta de la imagen.

    Returns:
        Ruta relativa de la imagen.
    """
    if not image_path:
        return "N/D"
    return os.path.relpath(image_path, ".")


def _build_progress_bar(kit: "UIKit", current: int, total: int) -> str:
    """
    Construye una barra de progreso coloreada y adaptada al ancho disponible.

    Args:
        kit: Toolkit de interfaz de usuario de terminal.
        current: Progreso actual.
        total: Progreso total.

    Returns:
        Barra de progreso en formato string.
    """
    if total <= 0:
        return f"{_style_token(kit, 'DIM')}[sin muestra]{_style_token(kit, 'ENDC')}"
    safe_current = max(0, min(int(current), int(total)))
    ui_width = max(60, kit.width())
    bar_width = max(20, min(40, ui_width - 44))
    pct = max(0, min(100, int((safe_current / total) * 100)))
    filled = int((pct / 100) * bar_width)
    pct_color = (
        _style_token(kit, "OKGREEN")
        if pct >= 100
        else _style_token(kit, "OKCYAN")
        if pct >= 50
        else _style_token(kit, "WARNING")
    )
    bar = (
        f"{_style_token(kit, 'OKCYAN')}{'█' * filled}"
        f"{_style_token(kit, 'DIM')}{'░' * (bar_width - filled)}{_style_token(kit, 'ENDC')}"
    )
    return (
        f"[{bar}]  "
        f"{pct_color}{_style_token(kit, 'BOLD')}{pct:>3}%{_style_token(kit, 'ENDC')}"
        f"  {_style_token(kit, 'DIM')}({safe_current} / {total}){_style_token(kit, 'ENDC')}"
    )


def _should_show_progress_bar(total: int) -> bool:
    """
    Regla global: ocultar barras triviales (0/0 o 1/1).

    Args:
        total: Progreso total.

    Returns:
        True si se debe mostrar la barra de progreso, False en caso contrario.
    """
    return total > 1


def _status_icon(status: str) -> str:
    """
    Devuelve un símbolo Unicode indicativo del estado para la TUI.

    Args:
        status: Estado del registro.

    Returns:
        Símbolo Unicode indicativo del estado.
    """
    return {"OK": "✓", "WARN": "⚠", "FAIL": "✗"}.get(status.upper(), "·")


def _status_color(kit: "UIKit", status: str) -> str:
    """
    Devuelve el color ANSI asociado a un estado textual.

    Args:
        kit: Toolkit de interfaz de usuario de terminal.
        status: Estado del registro.

    Returns:
        Color ANSI asociado al estado.
    """
    if status == "OK":
        return _style_token(kit, "OKGREEN")
    if status == "WARN":
        return _style_token(kit, "WARNING")
    if status == "FAIL":
        return _style_token(kit, "FAIL")
    return _style_token(kit, "OKCYAN")


def _section_header(kit: "UIKit", title: str, *, width: int | None = None) -> str:
    """
    Genera un encabezado de sección con título integrado en la línea divisora.

    Args:
        kit: Toolkit de interfaz de usuario de terminal.
        title: Título de la sección.
        width: Ancho de la línea (opcional).

    Returns:
        Encabezado de sección en formato string.
    """
    w = (width if width is not None else max(60, kit.width())) - 4
    fill = max(2, w - len(title) - 5)
    dim = _style_token(kit, "DIM")
    bold = _style_token(kit, "BOLD")
    endc = _style_token(kit, "ENDC")
    return f"  {dim}──{endc} {bold}{title}{endc} {dim}{'─' * fill}{endc}"


def _make_recent_records(*, limit: int = 5) -> deque[dict[str, object]]:
    """
    Crea un buffer para guardar los registros crudos (NO strings formateados).

    Args:
        limit: Número máximo de registros a guardar.

    Returns:
        Buffer de registros recientes.
    """
    return deque(maxlen=limit)


def _append_recent_record(buffer: deque[dict[str, object]], record: dict[str, object]) -> None:
    """
    Añade el diccionario crudo al buffer. El formato se hará en caliente en el render.

    Args:
        buffer: Buffer de registros recientes.
        record: Registro de inferencia.
    """
    buffer.append(record)


def _format_recent_status_lines(
    kit: "UIKit",
    record: dict[str, object],
    *,
    truncate: bool = True,
) -> list[str]:
    """
    Formatea un registro adaptando los anchos al tamaño EXACTO actual del terminal.

    Args:
        kit: Toolkit de interfaz de usuario de terminal.
        record: Registro de inferencia.
        truncate: Si es True, trunca las líneas largas.

    Returns:
        Lista de líneas formateadas.
    """
    status = str(record.get("status") or "unknown")
    if status == "ok":
        color = _style_token(kit, "OKGREEN")
        icon = "✓"
    elif status == "invalid":
        color = _style_token(kit, "WARNING")
        icon = "⚠"
    else:
        color = _style_token(kit, "FAIL")
        icon = "✗"
    badge = f"{color}{_style_token(kit, 'BOLD')}{icon}{_style_token(kit, 'ENDC')}"

    image_name = str(record.get("image_name") or os.path.basename(str(record.get("image_path") or "N/D")))
    
    # --- CÁLCULO DINÁMICO DE ANCHOS DE COLUMNA ---
    ui_width = kit.width()
    # Overhead: 2 espacios iniciales + badge (1) + 1 espacio + name_width + 2 espacios + pipe (1) + 2 espacios = 9 fijos
    content_budget = max(40, ui_width - 10) 
    
    # La columna del nombre de la imagen será como máximo 24 chars, o el 35% del presupuesto.
    name_width = min(24, max(10, int(content_budget * 0.35)))
    detail_width = max(20, content_budget - name_width)
    
    if len(image_name) > name_width:
        image_name = image_name[:name_width - 1] + "…"

    detail_lines: list[str] =[]
    payload = cast(dict[str, object] | None, record.get("payload"))
    if status == "ok":
        if payload:
            detail_lines.extend(_format_payload_lines(kit, payload, detail_width))
        if record.get("ttft_seconds") is not None:
            detail_lines.append(f"TTFT={_format_metric_value(record.get('ttft_seconds'), suffix=' s')}")
        if record.get("tokens_per_second") is not None:
            detail_lines.append(f"TPS={_format_metric_value(record.get('tokens_per_second'))}")
        if record.get("total_duration_seconds") is not None:
            detail_lines.append(f"total={_format_metric_value(record.get('total_duration_seconds'), suffix=' s')}")
        if record.get("response_preview") is not None and not detail_lines:
            detail_lines.append(str(record.get("response_preview")))
        if not detail_lines:
            detail_lines.append("Resultado válido")
    elif status == "invalid":
        detail_lines.append(str(record.get("validation_error") or record.get("message") or "JSON inválido"))
        if payload:
            detail_lines.extend(_format_payload_lines(kit, payload, detail_width))
    else:
        detail_lines.append(str(record.get("error") or "Error desconocido"))

    if truncate:
        detail_lines = _cap_detail_lines(detail_lines, max_lines=12)
        
    dim = _style_token(kit, "DIM")
    endc = _style_token(kit, "ENDC")
    
    base_prefix = (
        f"  {badge} "
        f"{_style_token(kit, 'BOLD')}{image_name:<{name_width}}{endc}"
        f"  {dim}│{endc}  "
    )
    cont_prefix = f"  {' ' * 2}{' ' * name_width}  {dim}│{endc}  "
    
    if not detail_lines:
        detail_lines.append("")
        
    formatted_lines = [base_prefix + detail_lines[0]]
    for line in detail_lines[1:]:
        formatted_lines.append(cont_prefix + line)
        
    return formatted_lines


def _build_recent_record_lines(
    kit: "UIKit",
    records: list[dict[str, object]],
    *,
    empty_message: str = "  Sin registros.",
    truncate: bool = True,
    ui_width: int | None = None,
) -> list[str]:
    """
    Convierte registros crudos en líneas de actividad formateándolos al momento.

    Args:
        kit: Toolkit de interfaz de usuario de terminal.
        records: Lista de registros de inferencia.
        empty_message: Mensaje a mostrar si no hay registros.
        truncate: Si es True, trunca las líneas largas.
        ui_width: Ancho de terminal ya capturado para el frame actual.

    Returns:
        Lista de líneas de actividad formateadas.
    """
    if not records:
        return [empty_message]

    width = kit.width() if ui_width is None else int(ui_width)
    table_menu_fn = getattr(kit, "table_menu", None)
    table_column_cls = getattr(kit, "TableColumn", None)
    table_row_cls = getattr(kit, "TableRow", None)
    table_cell_cls = getattr(kit, "TableCell", None)

    def _clip_inline(text: str, max_len: int) -> str:
        value = str(text or "").replace("\n", " ").strip()
        if max_len <= 0 or len(value) <= max_len:
            return value
        if max_len <= 1:
            return value[:max_len]
        return value[: max_len - 1] + "…"

    def _status_cell(status_value: str) -> tuple[str, str]:
        if status_value == "ok":
            return "✓ OK", "OKGREEN"
        if status_value == "invalid":
            return "⚠ WARN", "WARNING"
        return "✗ FAIL", "FAIL"

    def _is_bbox_payload(payload: dict[str, object]) -> bool:
        return (
            "detected_subjects_count" in payload
            or "object_count_reasoning" in payload
            or any(str(key).startswith("bbox_") for key in payload.keys())
        )

    def _ordered_payload_items(payload: dict[str, object]) -> list[str]:
        if _is_bbox_payload(payload):
            lines: list[str] = []
            count = payload.get("detected_subjects_count")
            reasoning = payload.get("object_count_reasoning")
            if count not in (None, ""):
                lines.append(f"Subjects detectados: {count}")
            if reasoning not in (None, ""):
                lines.append(f"Resumen: {_stringify_payload_value(reasoning)}")

            bbox_entries: list[tuple[int, str]] = []
            for key, value in payload.items():
                key_str = str(key)
                if not key_str.startswith("bbox_"):
                    continue
                if key_str == "bbox_more":
                    continue
                index_text = key_str.replace("bbox_", "")
                try:
                    index = int(index_text)
                except ValueError:
                    continue
                bbox_entries.append((index, _stringify_payload_value(value)))
            bbox_entries.sort(key=lambda item: item[0])
            for index, value_text in bbox_entries:
                lines.append(f"Detección {index}: {value_text}")

            bbox_more = payload.get("bbox_more")
            if bbox_more not in (None, ""):
                lines.append(f"Más detecciones: {_stringify_payload_value(bbox_more)}")

            return lines

        ordered_keys = [
            "polyp_detected",
            "prediction",
            "label",
            "class",
            "confidence_score",
            "score",
            "reasoning",
            "justification",
            "analysis",
            "response_preview",
            "message",
        ]
        lines: list[str] = []
        seen: set[str] = set()
        for key in ordered_keys:
            value = payload.get(key)
            if value in (None, ""):
                continue
            seen.add(key)
            if key == "polyp_detected":
                lines.append(f"{key}={'Sí' if bool(value) else 'No'}")
            else:
                lines.append(f"{key}={_stringify_payload_value(value)}")

        for key, value in payload.items():
            if key in seen or value in (None, ""):
                continue
            lines.append(f"{key}={_stringify_payload_value(value)}")
        return lines

    metric_keys = {
        "ttft_seconds",
        "tokens_per_second",
        "total_duration_seconds",
        "generation_duration_seconds",
        "confidence_score",
        "score",
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
    }

    def _is_metric_line(value: str) -> bool:
        key = value.split("=", 1)[0].strip().lower()
        return key in metric_keys

    def _allocate_non_metric_line_budget(
        values: list[str],
        *,
        max_total_lines: int,
        max_lines_per_field: int,
        wrap_width: int,
    ) -> list[int]:
        """Distribuye de forma justa un presupuesto global de líneas no métricas.

        Estrategia:
        1) Asigna 1 línea inicial por campo (si hay presupuesto).
        2) Reparte el resto en rondas (round-robin) hasta cubrir necesidades
           o agotar presupuesto.
        """
        if not values:
            return []

        needed = [min(max_lines_per_field, max(1, len(_wrap_text(kit, value, wrap_width)))) for value in values]
        allocated = [0 for _ in values]
        budget = max(0, max_total_lines)

        for idx, need in enumerate(needed):
            if budget <= 0:
                break
            if need > 0:
                allocated[idx] = 1
                budget -= 1

        while budget > 0:
            progressed = False
            for idx, need in enumerate(needed):
                if allocated[idx] < need and budget > 0:
                    allocated[idx] += 1
                    budget -= 1
                    progressed = True
                if budget <= 0:
                    break
            if not progressed:
                break

        return allocated

    def _clip_to_lines(value: str, *, max_width: int, max_lines: int) -> str:
        """Recorta texto a un número máximo de líneas envueltas."""
        wrapped = _wrap_text(kit, value, max_width)
        if len(wrapped) <= max_lines:
            return "\n".join(wrapped)
        kept = wrapped[:max_lines]
        if kept:
            last = kept[-1]
            if len(last) >= max_width:
                kept[-1] = (last[: max(1, max_width - 1)] + "…")[:max_width]
            else:
                kept[-1] = last + "…"
        return "\n".join(kept)

    merged_rows: list[tuple[str, str, str, str]] = []
    detailed_rows: list[tuple[str, list[tuple[str, bool, int | None, str | None]], str]] = []

    def _recent_payload_summary(payload: dict[str, object], *, limit: int = 4) -> str:
        if _is_bbox_payload(payload):
            count = payload.get("detected_subjects_count")
            reasoning = payload.get("object_count_reasoning")
            first_box = payload.get("bbox_1")
            parts: list[str] = []
            if count not in (None, ""):
                parts.append(f"subjects={count}")
            if reasoning not in (None, ""):
                parts.append(f"resumen={_stringify_payload_value(reasoning)}")
            if first_box not in (None, ""):
                parts.append(f"detección_1={_stringify_payload_value(first_box)}")
            return " · ".join(parts[:limit]) if parts else "Resultado válido"

        summary_order = [
            "polyp_detected",
            "prediction",
            "label",
            "class",
            "confidence_score",
            "score",
            "ttft_seconds",
            "tokens_per_second",
            "generation_duration_seconds",
            "total_duration_seconds",
        ]
        summary_parts: list[str] = []
        used_keys: set[str] = set()
        for key in summary_order:
            value = payload.get(key)
            if value in (None, ""):
                continue
            used_keys.add(key)
            if key == "polyp_detected":
                value_text = "Sí" if bool(value) else "No"
            elif key in {"ttft_seconds", "tokens_per_second", "generation_duration_seconds", "total_duration_seconds"}:
                suffix = " s" if key != "tokens_per_second" else ""
                value_text = _format_metric_value(value, suffix=suffix)
            else:
                value_text = _stringify_payload_value(value)
            summary_parts.append(f"{key}={value_text}")
            if len(summary_parts) >= limit:
                break

        if summary_parts:
            return " · ".join(summary_parts)

        for key, value in payload.items():
            if value in (None, ""):
                continue
            if key in used_keys:
                continue
            return f"{key}={_stringify_payload_value(value)}"
        return "Resultado válido"

    def _recent_payload_detail(record: dict[str, object], payload: dict[str, object] | None, *, status: str) -> str:
        if status == "invalid":
            message = str(record.get("validation_error") or record.get("message") or "JSON inválido")
            if payload:
                return f"{message} · {_recent_payload_summary(payload, limit=2)}"
            return message
        if status != "ok":
            return str(record.get("error") or "Error desconocido")

        if payload:
            if _is_bbox_payload(payload):
                detail_parts: list[str] = []
                reasoning = payload.get("object_count_reasoning")
                if reasoning not in (None, ""):
                    detail_parts.append(f"resumen={_stringify_payload_value(reasoning)}")
                for key in ("bbox_1", "bbox_2"):
                    value = payload.get(key)
                    if value in (None, ""):
                        continue
                    detail_parts.append(f"{key}={_stringify_payload_value(value)}")
                bbox_more = payload.get("bbox_more")
                if bbox_more not in (None, ""):
                    detail_parts.append(f"bbox_more={_stringify_payload_value(bbox_more)}")
                if detail_parts:
                    return " · ".join(detail_parts)

            verbose_parts: list[str] = []
            for key in ("reasoning", "justification", "analysis", "response_preview", "message"):
                value = payload.get(key)
                if value in (None, ""):
                    continue
                verbose_parts.append(f"{key}={_stringify_payload_value(value)}")

            extra_parts: list[str] = []
            for key, value in payload.items():
                if key in {"polyp_detected", "prediction", "label", "class", "confidence_score", "score", "ttft_seconds", "tokens_per_second", "generation_duration_seconds", "total_duration_seconds", "reasoning", "justification", "analysis", "response_preview", "message"}:
                    continue
                if value in (None, ""):
                    continue
                extra_parts.append(f"{key}={_stringify_payload_value(value)}")
                if len(extra_parts) >= 2:
                    break

            if verbose_parts:
                if extra_parts:
                    verbose_parts.append(" · ".join(extra_parts))
                return "\n".join(verbose_parts)

            if extra_parts:
                return " · ".join(extra_parts)

        if record.get("response_preview") is not None:
            return str(record.get("response_preview"))
        return "Resultado válido"

    for index, record in enumerate(records):
        status = str(record.get("status") or "unknown")
        status_cell, status_color = _status_cell(status)

        image_name = str(record.get("image_name") or os.path.basename(str(record.get("image_path") or "N/D")))
        payload = cast(dict[str, object] | None, record.get("payload"))

        summary_text = _recent_payload_summary(payload or {}) if payload else "Resultado válido"
        detail_text = _recent_payload_detail(record, payload, status=status)
        if status == "ok" and summary_text == "Resultado válido":
            metric_parts: list[str] = []
            if record.get("ttft_seconds") is not None:
                metric_parts.append(f"TTFT={_format_metric_value(record.get('ttft_seconds'), suffix=' s')}")
            if record.get("tokens_per_second") is not None:
                metric_parts.append(f"TPS={_format_metric_value(record.get('tokens_per_second'))}")
            if record.get("total_duration_seconds") is not None:
                metric_parts.append(
                    f"total={_format_metric_value(record.get('total_duration_seconds'), suffix=' s')}"
                )
            if metric_parts:
                summary_text = " · ".join(metric_parts)
                if detail_text == "Resultado válido":
                    detail_text = summary_text

        if truncate:
            summary_text = _clip_inline(summary_text, max(28, width // 4))
            detail_text = _clip_inline(detail_text, max(48, width // 2))

        merged_rows.append((f"{status_cell} {image_name}", summary_text, detail_text, status_color))

        field_rows: list[str] = []
        if status == "ok":
            if payload:
                field_rows.extend(_ordered_payload_items(payload))
            if record.get("ttft_seconds") is not None:
                field_rows.append(f"ttft_seconds={_format_metric_value(record.get('ttft_seconds'), suffix=' s')}")
            if record.get("tokens_per_second") is not None:
                field_rows.append(f"tokens_per_second={_format_metric_value(record.get('tokens_per_second'))}")
            if record.get("total_duration_seconds") is not None:
                field_rows.append(
                    f"total_duration_seconds={_format_metric_value(record.get('total_duration_seconds'), suffix=' s')}"
                )
        elif status == "invalid":
            field_rows.append(f"error={str(record.get('validation_error') or record.get('message') or 'JSON inválido')}")
            if payload:
                field_rows.extend(_ordered_payload_items(payload))
        else:
            field_rows.append(f"error={str(record.get('error') or 'Error desconocido')}")

        if truncate and len(field_rows) > 14:
            field_rows = field_rows[:14]
            field_rows.append("…=Se ocultaron campos adicionales")

        response_wrap_width = max(28, int(width * 0.66))
        non_metric_indices = [idx for idx, value in enumerate(field_rows) if not _is_metric_line(value)]
        non_metric_values = [field_rows[idx] for idx in non_metric_indices]
        non_metric_alloc = (
            _allocate_non_metric_line_budget(
                non_metric_values,
                max_total_lines=15,
                max_lines_per_field=5,
                wrap_width=response_wrap_width,
            )
            if truncate
            else [5 for _ in non_metric_values]
        )
        per_field_non_metric_budget = {
            non_metric_indices[idx]: non_metric_alloc[idx]
            for idx in range(len(non_metric_indices))
        }

        normalized_fields: list[tuple[str, bool, int | None, str | None]] = []
        has_metric = False
        has_non_metric = False
        for idx, value in enumerate(field_rows):
            is_metric = _is_metric_line(value)
            if is_metric:
                has_metric = True
                if truncate:
                    normalized_fields.append((_clip_inline(value, max(32, width - 48)), True, 1, None))
                else:
                    normalized_fields.append((value, True, None, None))
                continue

            has_non_metric = True
            allowed_lines = per_field_non_metric_budget.get(idx, 0)
            if truncate:
                if allowed_lines <= 0:
                    normalized_fields.append(("…", False, 1, "DIM"))
                else:
                    normalized_fields.append((value, False, min(5, allowed_lines), None))
            else:
                normalized_fields.append((value, False, None, None))

        has_bbox_payload = bool(payload) and _is_bbox_payload(cast(dict[str, object], payload))

        # Subfilas para separar claramente contenido y métricas en Respuesta.
        non_metric_fields = [item for item in normalized_fields if not item[1]]
        metric_fields = [item for item in normalized_fields if item[1]]
        grouped_fields: list[tuple[str, bool, int | None, str | None]] = []
        if has_bbox_payload:
            grouped_fields.extend(non_metric_fields)
            grouped_fields.extend(metric_fields)
        else:
            if non_metric_fields:
                grouped_fields.append(("─── contenido ───", False, 1, "DIM"))
                grouped_fields.extend(non_metric_fields)
            if metric_fields:
                grouped_fields.append(("─── métricas ───", False, 1, "DIM"))
                grouped_fields.extend(metric_fields)
        normalized_fields = grouped_fields

        # Zebra striping blanco/gris para filas de respuesta (None/DIM)
        zebra_idx = 0
        zebra_fields: list[tuple[str, bool, int | None, str | None]] = []
        for value, is_metric, max_lines, row_color in normalized_fields:
            if value in ("─── contenido ───", "─── métricas ───"):
                zebra_fields.append((value, is_metric, max_lines, "DIM"))
                continue
            color = None if (zebra_idx % 2 == 0) else "DIM"
            zebra_fields.append((value, is_metric, max_lines, color))
            zebra_idx += 1
        normalized_fields = zebra_fields

        # Si no hay payload ni métricas, conserva una fila mínima visible.
        if not normalized_fields:
            fallback_value = detail_text if str(detail_text or "").strip() else summary_text
            if not str(fallback_value or "").strip():
                fallback_value = "Resultado válido"
            normalized_fields = [(str(fallback_value), False, 1 if truncate else None, None)]

        detailed_rows.append((f"{status_cell} {image_name}", normalized_fields, status_color))

    if callable(table_menu_fn) and table_column_cls is not None and table_row_cls is not None:
        # Preferred view: one block per image using merged rows (rowspan)
        columns = [
            table_column_cls(label="Imagen", width_ratio=0.30, min_width=20),
            table_column_cls(label="Respuesta", width_ratio=0.70, min_width=40),
        ]

        table_rows: list[Any] = []
        if table_cell_cls is not None:
            for image_cell_text, fields, image_color in detailed_rows:
                row_span = max(1, len(fields))
                first_value, _first_is_metric, first_max_lines, first_color = fields[0]
                first_image_cell = table_cell_cls(text=image_cell_text, rowspan=row_span, color=image_color)
                first_response_cell = table_cell_cls(
                    text=first_value,
                    max_lines=first_max_lines,
                    color=first_color,
                )
                table_rows.append(table_row_cls(cells=[first_image_cell, first_response_cell]))
                for value, _is_metric, max_lines, row_color in fields[1:]:
                    response_cell = table_cell_cls(
                        text=value,
                        max_lines=max_lines,
                        color=row_color,
                    )
                    table_rows.append(table_row_cls(cells=[response_cell]))
                table_rows.append(table_row_cls(cells=["", ""]))
            if table_rows:
                table_rows.pop()
        else:
            for image_cell_text, fields, image_color in detailed_rows:
                fallback_wrap_width = max(28, int(width * 0.66))
                for idx, (value, is_metric, max_lines, row_color) in enumerate(fields):
                    if truncate and max_lines is not None:
                        rendered_value = _clip_to_lines(value, max_width=fallback_wrap_width, max_lines=max(1, max_lines))
                    elif truncate and is_metric:
                        rendered_value = _clip_inline(value, max(32, width - 48))
                    else:
                        rendered_value = value
                    table_rows.append(
                        table_row_cls(
                            cells=[image_cell_text if idx == 0 else "", rendered_value],
                            cell_colors=[image_color if idx == 0 else None, row_color],
                        )
                    )
                table_rows.append(table_row_cls(cells=["", ""]))
            if table_rows:
                table_rows.pop()

        rendered = table_menu_fn(
            columns,
            table_rows,
            interactive=False,
            return_lines=True,
            width=width,
            max_cell_lines=(5 if truncate else False),
        )
        if isinstance(rendered, list):
            return [str(line) for line in rendered]

    fallback_lines: list[str] = []
    for left, summary, detail, _color in merged_rows:
        fallback_lines.append(f"  {left} │ {summary} │ {detail}")
    return fallback_lines


def _render_table_lines(
    kit: "UIKit",
    headers: list[str],
    rows: list[list[str]],
    *,
    ui_width: int | None = None,
    status_column_index: int | None = None,
) -> list[str]:
    """Renderiza tablas no interactivas reutilizando el motor de menu_kit."""
    if not rows:
        return ["  Sin datos."]

    width = kit.width() if ui_width is None else int(ui_width)
    table_menu_fn = getattr(kit, "table_menu", None)
    table_column_cls = getattr(kit, "TableColumn", None)
    table_row_cls = getattr(kit, "TableRow", None)

    if not (callable(table_menu_fn) and table_column_cls is not None and table_row_cls is not None):
        label_width = max(12, min(30, max(len(h) for h in headers)))
        value_width = max(20, width - label_width - 10)
        fallback_lines: list[str] = []
        for row in rows:
            left = str(row[0]) if row else ""
            right = " │ ".join(str(cell) for cell in row[1:])
            wrapped = _wrap_text(kit, right, value_width)
            fallback_lines.append(f"  {left:<{label_width}} : {wrapped[0] if wrapped else ''}")
            for extra in wrapped[1:]:
                fallback_lines.append(f"  {'':<{label_width}}   {extra}")
        return fallback_lines

    ratio = 1.0 / max(1, len(headers))
    columns = [
        table_column_cls(label=header, width_ratio=ratio, min_width=10)
        for header in headers
    ]

    table_rows: list = []
    for row in rows:
        normalized = [str(cell) for cell in row]
        colors: list[str | None] | None = None
        if status_column_index is not None and 0 <= status_column_index < len(normalized):
            status_token = normalized[status_column_index].upper()
            if "OK" in status_token:
                color_name = "OKGREEN"
            elif "WARN" in status_token:
                color_name = "WARNING"
            else:
                color_name = "FAIL"
            typed_colors: list[str | None] = [None for _ in normalized]
            typed_colors[status_column_index] = color_name
            colors = typed_colors
        table_rows.append(table_row_cls(cells=normalized, cell_colors=colors))

    rendered = table_menu_fn(
        columns,
        table_rows,
        interactive=False,
        return_lines=True,
        width=width,
        max_cell_lines=False,
    )
    if isinstance(rendered, list):
        return [str(line) for line in rendered]
    return ["  Sin datos."]


def _make_live_panel(
    kit: "UIKit",
    app: "AppContext",
    *,
    subtitle: str,
    intro: str,
    static_lines: list[str] | None = None,
):
    """
    Crea un panel incremental con banner estatico y subtitulo reutilizable.

    Args:
        kit: Toolkit de interfaz de usuario de terminal.
        app: Contexto de la aplicación.
        subtitle: Subtítulo del panel.
        intro: Introducción del panel.
        static_lines: Líneas estáticas del panel.

    Returns:
        Panel incremental.
    """
    def render_static() -> None:
        """Pinta cabecera, subtítulo y bloque estático del panel en vivo."""
        app.print_banner()
        kit.subtitle(subtitle)
        if static_lines:
            print()
            for line in static_lines:
                print(line)
            print()
        kit.log(intro, "step")

    return kit.IncrementalPanelRenderer(
        clear_screen_fn=kit.clear,
        render_static_fn=render_static,
    )


def _render_live_dashboard(
    kit: "UIKit",
    panel: Any,
    *,
    current: int,
    total: int,
    stats_line: str,
    status_line: str,
    metrics_line: str | None = None,
    coverage_line: str | None = None,
    recent_title: str = "Últimas imágenes procesadas:",
    recent_records: list[dict[str, object]] | None = None, # AHORA RECIBE RECORDS CRUDOS
    recent_limit: int | None = 5,
    extra_lines: list[str] | None = None,
    force_full: bool | None = None,
) -> None:
    """
    Compone un frame dinámico homogéneo calculando todos los anchos al vuelo.

    Args:
        kit: Toolkit de interfaz de usuario de terminal.
        panel: Panel incremental.
        current: Progreso actual.
        total: Progreso total.
        stats_line: Línea de estadísticas.
        status_line: Línea de estado.
        metrics_line: Línea de métricas.
        coverage_line: Línea de cobertura.
        recent_title: Título de la sección de registros recientes.
        recent_records: Lista de registros crudos.
        recent_limit: Número máximo de registros recientes.
        extra_lines: Líneas adicionales.
        force_full: Si es True, fuerza el render completo.
    """
    ui_width = kit.width()
    dim = _style_token(kit, "DIM")
    endc = _style_token(kit, "ENDC")
    section_title = recent_title.rstrip(":")
    
    dynamic_lines: list[str] =[
        "",
        _section_header(kit, "Progreso"),
        f"  {stats_line}",
        f"  {dim}{status_line}{endc}",
    ]
    if _should_show_progress_bar(total):
        dynamic_lines.insert(2, f"  {_build_progress_bar(kit, current, total)}")
    
    if metrics_line:
        dynamic_lines.append(f"  {metrics_line}")
    if coverage_line:
        dynamic_lines.append(f"  {dim}{coverage_line}{endc}")
    if extra_lines:
        dynamic_lines.extend(f"  {line}" for line in extra_lines)
        
    dynamic_lines.append("")
    dynamic_lines.append(_section_header(kit, section_title))
    
    if recent_records:
        # Truncamos a las últimas entradas si se solicita para evitar listas
        # excesivamente largas en el panel en vivo.
        if recent_limit is not None and len(recent_records) > recent_limit:
            display_records = recent_records[-recent_limit:]
        else:
            display_records = recent_records
        # Se formatean AHORA, usando el kit.width() actual del terminal
        dynamic_lines.extend(_build_recent_record_lines(kit, display_records, ui_width=ui_width))
    else:
        dynamic_lines.append(f"  {dim}Sin actividad todavía.{endc}")
        
    dynamic_lines.append("")
    dynamic_lines.append(f"  {dim}{'─' * max(2, ui_width - 4)}{endc}")
    
    if force_full is None:
        force_full = getattr(kit, "os_module", None) is not None and kit.os_module.name == "nt"
        
    panel.render(dynamic_lines, force_full=bool(force_full))


def _build_summary_lines(
    kit: "UIKit",
    rows: list[tuple[str, str, str]],
    *,
    ui_width: int | None = None,
) -> list[str]:
    """
    Convierte filas tipo tabla en lineas compactas para el panel final.

    Args:
        kit: Toolkit de interfaz de usuario de terminal.
        rows: Lista de filas tipo tabla.
        ui_width: Ancho de terminal ya capturado para el frame actual.

    Returns:
        Lista de líneas compactas.
    """
    table_rows = []
    for name, value, status in rows:
        table_rows.append([str(name), str(value), f"{_status_icon(status)} {status}"])
    return _render_table_lines(
        kit,
        ["Campo", "Valor", "Estado"],
        table_rows,
        ui_width=ui_width,
        status_column_index=2,
    )


def _build_named_value_lines(
    kit: "UIKit",
    rows: list[tuple[str, object]],
    *,
    ui_width: int | None = None,
) -> list[str]:
    """
    Convierte pares clave/valor en lineas compactas para paneles finales.

    Args:
        kit: Toolkit de interfaz de usuario de terminal.
        rows: Lista de pares clave/valor.
        ui_width: Ancho de terminal ya capturado para el frame actual.

    Returns:
        Lista de líneas compactas.
    """
    visible_rows = [[str(name), str(value)] for name, value in rows if value not in (None, "")]
    return _render_table_lines(
        kit,
        ["Campo", "Valor"],
        visible_rows,
        ui_width=ui_width,
    )


def _normalize_final_section_title(title: str) -> str:
    """
    Homogeneiza titulos de secciones entre distintos runners del menu.

    Args:
        title: Título de la sección.

    Returns:
        Título normalizado.
    """
    normalized = title.strip().upper()
    title_map = {
        "LM STUDIO RESPONSE INSPECTOR": "Resumen",
        "SDK ATTRIBUTES": "Atributos SDK",
        "SDK STATS": "Estadísticas SDK",
        "PARSED PAYLOAD": "Payload estructurado",
        "REST RESOURCES": "Recursos REST",
        "DETALLE POR IMAGEN": "Actividad reciente",
        "DETALLE POR CASO": "Actividad reciente",
        "ULTIMOS REGISTROS EXPORTADOS": "Actividad reciente",
        "ULTIMAS VALIDACIONES": "Actividad reciente",
        "ÚLTIMOS REGISTROS EXPORTADOS": "Actividad reciente",
        "ÚLTIMAS VALIDACIONES": "Actividad reciente",
    }
    return title_map.get(normalized, title)


def _standard_final_intro() -> str:
    """
    Mensaje final homogeneo para los resumenes del menu de tests.

    Returns:
        Mensaje final en formato string.
    """
    return "Proceso finalizado. Revisa el resumen y las observaciones."


def _render_final_sections_screen(
    kit: "UIKit",
    app: "AppContext",
    *,
    subtitle: str,
    intro: str,
    sections: list[tuple[str, list[str]]],
    ui_width: int | None = None,
) -> None:
    """
    Pinta una pantalla final compacta reutilizable sin salida cruda dispersa.

    Args:
        kit: Toolkit de interfaz de usuario de terminal.
        app: Contexto de la aplicación.
        subtitle: Subtítulo del panel.
        intro: Introducción del panel.
        sections: Lista de secciones con título y líneas.
        ui_width: Ancho de terminal ya capturado para el frame actual.
    """
    dim = _style_token(kit, "DIM")
    endc = _style_token(kit, "ENDC")
    width = kit.width() if ui_width is None else int(ui_width)
    panel = _make_live_panel(kit, app, subtitle=subtitle, intro=intro)
    dynamic_lines: list[str] = [""]
    for title, lines in sections:
        dynamic_lines.append(_section_header(kit, _normalize_final_section_title(title)))
        dynamic_lines.extend(lines if lines else[f"  {dim}Sin datos.{endc}"])
        dynamic_lines.append("")
    dynamic_lines.append(f"  {dim}{'─' * max(2, width - 4)}{endc}")
    panel.render(dynamic_lines, force_full=True)


def _format_batch_summary_rows(summary: dict[str, object]) -> list[tuple[str, str, str]]:
    """
    Prepara filas compactas para el resumen final del batch runner.

    Args:
        summary: Resumen del batch runner.

    Returns:
        Lista de filas compactas.
    """
    processed = _coerce_int(summary.get("processed"))
    sample_size = _coerce_int(summary.get("sample_size")) or processed
    ok_count = _coerce_int(summary.get("ok"))
    invalid_count = _coerce_int(summary.get("invalid"))
    fail_count = _coerce_int(summary.get("fail"))
    return[
        ("Modelo", str(summary.get("model_id") or "N/D"), "OK"),
        ("Esquema", str(summary.get("schema_name") or "N/D"), "OK"),
        ("Muestra", f"{processed} de {sample_size}", "OK"),
        ("OK", str(ok_count), "OK" if ok_count > 0 else "WARN"),
        ("Invalidas", str(invalid_count), "OK" if invalid_count == 0 else "WARN"),
        ("Errores", str(fail_count), "OK" if fail_count == 0 else "WARN"),
        ("TTFT medio (s)", _format_metric_value(cast(dict[str, object], summary.get("ttft") or {}).get("avg")), "OK"),
        ("TPS medio", _format_metric_value(cast(dict[str, object], summary.get("tps") or {}).get("avg")), "OK"),
        ("Latencia media (s)", _format_metric_value(cast(dict[str, object], summary.get("total_duration") or {}).get("avg")), "OK"),
        ("Salida", str(summary.get("output_path") or "N/D"), "OK"),
    ]


def _format_schema_summary_rows(summary: dict[str, object]) -> list[tuple[str, str, str]]:
    """
    Prepara filas compactas para el resumen final del schema tester.

    Args:
        summary: Resumen del schema tester.

    Returns:
        Lista de filas compactas.
    """
    return[
        ("Modelo", str(summary["model_id"]), "OK"),
        ("Esquema", str(summary["schema_name"]), "OK"),
        ("Muestra", f"{summary['sample_size']} de {summary['total_available']}", "OK"),
        ("Validas", str(summary["ok"]), "OK" if _coerce_int(summary.get("ok")) > 0 else "WARN"),
        ("Invalidas", str(summary["invalid"]), "OK" if _coerce_int(summary.get("invalid")) == 0 else "WARN"),
        ("Errores", str(summary["fail"]), "OK" if _coerce_int(summary.get("fail")) == 0 else "WARN"),
    ]