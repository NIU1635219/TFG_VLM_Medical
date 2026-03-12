"""Helpers visuales reutilizables para los dashboards del menu de tests."""

from __future__ import annotations

from collections import deque
import json
import os
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from .menu_kit import AppContext, UIKit


def _format_metric_value(value: object, *, suffix: str = "") -> str:
    """Representa metricas opcionales de forma amigable para la TUI."""
    if value is None:
        return "N/D"
    if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
        return f"{int(value)}{suffix}"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".") + suffix
    return f"{value}{suffix}"


def _style_token(kit: "UIKit", name: str) -> str:
    """Devuelve un token ANSI si existe en el estilo actual."""
    return str(getattr(kit.style, name, ""))


def _wrap_text(kit: "UIKit", text: str, width: int) -> list[str]:
    """Envuelve texto con el helper del kit o usa un fallback simple en tests."""
    wrap_fn = getattr(kit, "wrap", None)
    if callable(wrap_fn):
        wrapped = wrap_fn(text, width)
        if isinstance(wrapped, list) and wrapped:
            return [str(line) for line in wrapped]
    if len(text) <= width:
        return [text]
    return [text[index:index + width] for index in range(0, len(text), width)]


def _stringify_payload_value(value: object) -> str:
    """Normaliza valores JSON para mostrarlos en líneas legibles."""
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
    """Convierte un payload en líneas con wrapping conservando claves."""
    if payload is None:
        return []

    if isinstance(payload, dict):
        lines: list[str] = []
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


def _recent_record_separator(kit: "UIKit") -> str:
    """Crea una linea divisoria para separar registros recientes."""
    dim = _style_token(kit, "DIM")
    endc = _style_token(kit, "ENDC")
    return f"  {dim}{'─' * max(2, kit.width() - 4)}{endc}"


def _cap_detail_lines(lines: list[str], *, max_lines: int) -> list[str]:
    """Recorta lineas muy largas para evitar cortes a mitad de registro."""
    if max_lines <= 0 or len(lines) <= max_lines:
        return lines
    if max_lines == 1:
        return [lines[0]]
    trimmed = lines[: max_lines - 1]
    trimmed.append("... (truncado)")
    return trimmed


def _coerce_int(value: object) -> int:
    """Convierte valores numericos opcionales del callback de progreso a enteros."""
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
    """Indica si una metrica tiene al menos un registro valido en la ejecucion."""
    return _coerce_int(availability.get(key)) > 0


def _coverage_fragments(availability: dict[str, object]) -> list[str]:
    """Construye una linea compacta de cobertura omitiendo metricas ausentes."""
    ok_records = _coerce_int(availability.get("ok_records"))
    if ok_records <= 0:
        return []

    fragments: list[str] = []
    metric_specs = [
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
    """Genera solo las columnas utiles para una inferencia concreta del probe."""
    parts = [f"TTFT={_format_metric_value(record.get('ttft_seconds'), suffix=' s')}"]

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
    """Normaliza rutas de imagenes para mostrarlas de forma compacta."""
    if not image_path:
        return "N/D"
    return os.path.relpath(image_path, ".")


def _build_progress_bar(kit: "UIKit", current: int, total: int) -> str:
    """Construye una barra de progreso coloreada y adaptada al ancho disponible."""
    if total <= 0:
        return f"{_style_token(kit, 'DIM')}[sin muestra]{_style_token(kit, 'ENDC')}"
    ui_width = max(60, kit.width())
    bar_width = max(20, min(40, ui_width - 44))
    pct = max(0, min(100, int((current / total) * 100)))
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
        f"  {_style_token(kit, 'DIM')}({current} / {total}){_style_token(kit, 'ENDC')}"
    )


def _status_icon(status: str) -> str:
    """Devuelve un símbolo Unicode indicativo del estado para la TUI."""
    return {"OK": "✓", "WARN": "⚠", "FAIL": "✗"}.get(status.upper(), "·")


def _status_color(kit: "UIKit", status: str) -> str:
    """Devuelve el color ANSI asociado a un estado textual."""
    if status == "OK":
        return _style_token(kit, "OKGREEN")
    if status == "WARN":
        return _style_token(kit, "WARNING")
    if status == "FAIL":
        return _style_token(kit, "FAIL")
    return _style_token(kit, "OKCYAN")


def _section_header(kit: "UIKit", title: str, *, width: int | None = None) -> str:
    """Genera un encabezado de sección con título integrado en la línea divisora.

    Ejemplo: ``  ── Últimas validaciones ─────────────────────────────``
    """
    w = (width if width is not None else max(60, kit.width())) - 4
    fill = max(2, w - len(title) - 5)
    dim = _style_token(kit, "DIM")
    bold = _style_token(kit, "BOLD")
    endc = _style_token(kit, "ENDC")
    return f"  {dim}──{endc} {bold}{title}{endc} {dim}{'─' * fill}{endc}"


def _append_recent_line(buffer: deque[str], line: str, *, limit: int = 30) -> None:
    """Mantiene una ventana corta de actividad reciente para el dashboard."""
    if buffer.maxlen != limit:
        while len(buffer) >= limit:
            buffer.popleft()
    buffer.append(line)


def _make_recent_lines(*, limit: int = 30) -> deque[str]:
    """Crea un buffer homogéneo para actividad reciente en los dashboards."""
    return deque(maxlen=limit)


def _append_recent_record(kit: "UIKit", buffer: deque[str], record: dict[str, object], *, limit: int = 30) -> None:
    """Añade un registro formateado al buffer de actividad reciente."""
    lines = _format_recent_status_lines(kit, record, truncate=True)
    lines.append(_recent_record_separator(kit))
    if len(lines) >= limit:
        buffer.clear()
        buffer.extend(lines[-limit:])
        return
    while len(buffer) + len(lines) > limit:
        buffer.popleft()
    buffer.extend(lines)


def _build_recent_record_lines(
    kit: "UIKit",
    records: list[dict[str, object]],
    *,
    empty_message: str = "  Sin registros.",
) -> list[str]:
    """Convierte registros crudos en líneas de actividad reutilizables para paneles finales."""
    if not records:
        return [empty_message]
    lines: list[str] = []
    for record in records:
        lines.extend(_format_recent_status_lines(kit, record, truncate=False))
        lines.append(_recent_record_separator(kit))
    return lines


def _make_live_panel(
    kit: "UIKit",
    app: "AppContext",
    *,
    subtitle: str,
    intro: str,
    static_lines: list[str] | None = None,
):
    """Crea un panel incremental con banner estatico y subtitulo reutilizable."""
    def render_static() -> None:
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
    recent_title: str = "Ultimas imagenes procesadas:",
    recent_lines: list[str] | None = None,
    extra_lines: list[str] | None = None,
    force_full: bool | None = None,
) -> None:
    """Compone un frame dinamico homogeneo para ejecuciones largas."""
    dim = _style_token(kit, "DIM")
    endc = _style_token(kit, "ENDC")
    section_title = recent_title.rstrip(":")
    dynamic_lines: list[str] = [
        "",
        _section_header(kit, "Progreso"),
        f"  {_build_progress_bar(kit, current, total)}",
        f"  {stats_line}",
        f"  {dim}{status_line}{endc}",
    ]
    if metrics_line:
        dynamic_lines.append(f"  {metrics_line}")
    if coverage_line:
        dynamic_lines.append(f"  {dim}{coverage_line}{endc}")
    if extra_lines:
        dynamic_lines.extend(f"  {line}" for line in extra_lines)
    dynamic_lines.append("")
    dynamic_lines.append(_section_header(kit, section_title))
    if recent_lines:
        dynamic_lines.extend(recent_lines)
    else:
        dynamic_lines.append(f"  {dim}Sin actividad todavía.{endc}")
    dynamic_lines.append("")
    dynamic_lines.append(f"  {dim}{'─' * max(2, kit.width() - 4)}{endc}")
    if force_full is None:
        force_full = getattr(kit, "os_module", None) is not None and kit.os_module.name == "nt"
    panel.render(dynamic_lines, force_full=bool(force_full))


def _format_recent_status_lines(
    kit: "UIKit",
    record: dict[str, object],
    *,
    truncate: bool = True,
) -> list[str]:
    """Resume un registro reciente para mostrarlo en el dashboard."""
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
    detail_lines: list[str] = []
    payload = cast(dict[str, object] | None, record.get("payload"))
    if status == "ok":
        detail_width = max(30, min(120, kit.width() - 40))
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
            detail_width = max(30, min(120, kit.width() - 40))
            detail_lines.extend(_format_payload_lines(kit, payload, detail_width))
    else:
        detail_lines.append(str(record.get("error") or "Error desconocido"))

    if truncate:
        detail_lines = _cap_detail_lines(detail_lines, max_lines=12)
    dim = _style_token(kit, "DIM")
    endc = _style_token(kit, "ENDC")
    name_width = 24
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


def _build_summary_lines(kit: "UIKit", rows: list[tuple[str, str, str]]) -> list[str]:
    """Convierte filas tipo tabla en lineas compactas para el panel final."""
    if not rows:
        return ["  Sin datos."]
    label_width = min(28, max(len(str(name)) for name, _, _ in rows))
    value_width = max(20, min(kit.width() - label_width - 16, 60))
    lines: list[str] = []
    dim = _style_token(kit, "DIM")
    endc = _style_token(kit, "ENDC")
    for name, value, status in rows:
        name_lines = _wrap_text(kit, str(name), label_width)
        value_lines = _wrap_text(kit, str(value), value_width)
        row_count = max(len(name_lines), len(value_lines))
        icon = _status_icon(status)
        status_text = (
            f"{_status_color(kit, status)}{_style_token(kit, 'BOLD')}{icon}{endc}"
            f" {_status_color(kit, status)}{status:<4}{endc}"
        )
        for index in range(row_count):
            left = name_lines[index] if index < len(name_lines) else ""
            right = value_lines[index] if index < len(value_lines) else ""
            suffix = status_text if index == 0 else "      "
            lines.append(
                f"  {dim}{left:<{label_width}}{endc}"
                f"  {dim}:{endc}  "
                f"{right:<{value_width}}  {suffix}"
            )
    return lines


def _build_named_value_lines(kit: "UIKit", rows: list[tuple[str, object]]) -> list[str]:
    """Convierte pares clave/valor en lineas compactas para paneles finales."""
    visible_rows = [(str(name), str(value)) for name, value in rows if value not in (None, "")]
    if not visible_rows:
        return ["  Sin datos."]
    label_width = min(28, max(len(name) for name, _ in visible_rows))
    value_width = max(20, min(kit.width() - label_width - 8, 70))
    lines: list[str] = []
    dim = _style_token(kit, "DIM")
    endc = _style_token(kit, "ENDC")
    for name, value in visible_rows:
        value_lines = _wrap_text(kit, value, value_width)
        for index, chunk in enumerate(value_lines):
            if index == 0:
                prefix = f"  {dim}{name:<{label_width}}{endc}  {dim}:{endc}  "
            else:
                prefix = f"  {'':<{label_width}}     "
            lines.append(prefix + chunk)
    return lines


def _normalize_final_section_title(title: str) -> str:
    """Homogeneiza titulos de secciones entre distintos runners del menu."""
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
    """Mensaje final homogeneo para los resumenes del menu de tests."""
    return "Proceso finalizado. Revisa el resumen y las observaciones."


def _render_final_sections_screen(
    kit: "UIKit",
    app: "AppContext",
    *,
    subtitle: str,
    intro: str,
    sections: list[tuple[str, list[str]]],
) -> None:
    """Pinta una pantalla final compacta reutilizable sin salida cruda dispersa."""
    dim = _style_token(kit, "DIM")
    endc = _style_token(kit, "ENDC")
    panel = _make_live_panel(kit, app, subtitle=subtitle, intro=intro)
    dynamic_lines: list[str] = [""]
    for title, lines in sections:
        dynamic_lines.append(_section_header(kit, _normalize_final_section_title(title)))
        dynamic_lines.extend(lines if lines else [f"  {dim}Sin datos.{endc}"])
        dynamic_lines.append("")
    dynamic_lines.append(f"  {dim}{'─' * max(2, kit.width() - 4)}{endc}")
    panel.render(dynamic_lines, force_full=True)


def _format_batch_summary_rows(summary: dict[str, object]) -> list[tuple[str, str, str]]:
    """Prepara filas compactas para el resumen final del batch runner."""
    processed = _coerce_int(summary.get("processed"))
    sample_size = _coerce_int(summary.get("sample_size")) or processed
    ok_count = _coerce_int(summary.get("ok"))
    invalid_count = _coerce_int(summary.get("invalid"))
    fail_count = _coerce_int(summary.get("fail"))
    return [
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
    """Prepara filas compactas para el resumen final del schema tester."""
    return [
        ("Modelo", str(summary["model_id"]), "OK"),
        ("Esquema", str(summary["schema_name"]), "OK"),
        ("Muestra", f"{summary['sample_size']} de {summary['total_available']}", "OK"),
        ("Validas", str(summary["ok"]), "OK" if _coerce_int(summary.get("ok")) > 0 else "WARN"),
        ("Invalidas", str(summary["invalid"]), "OK" if _coerce_int(summary.get("invalid")) == 0 else "WARN"),
        ("Errores", str(summary["fail"]), "OK" if _coerce_int(summary.get("fail")) == 0 else "WARN"),
    ]