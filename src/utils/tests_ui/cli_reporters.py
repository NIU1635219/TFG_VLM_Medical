"""
Utilidades para imprimir reportes y resúmenes de scripts en CLI.

Este módulo concentra toda la presentación (prints, reportes y formateo visual de consola)
que fue extraída de los componentes lógicos en src/scripts.
"""

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SUMMARY_WIDTH = 76


def print_section(title: str, *, stream: Any = None) -> None:
    """
    Imprime una sección ASCII consistente para el informe CLI.
    
    Args:
        title (str): Título de la sección.
        stream (Any, optional): Stream de salida. Por defecto sys.stdout.
    """
    target = stream or sys.stdout
    divider = "=" * SUMMARY_WIDTH
    print(divider, file=target)
    print(title, file=target)
    print(divider, file=target)


def has_display_value(value: Any) -> bool:
    """
    Indica si un valor merece mostrarse en el resumen visual.
    
    Args:
        value (Any): Valor a evaluar.
        
    Returns:
        bool: True si el valor merece mostrarse en el resumen visual, False en caso contrario.
    """
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return any(has_display_value(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(has_display_value(item) for item in value)
    return True


def format_scalar(value: Any) -> str:
    """
    Convierte un valor a texto compacto para tablas de resumen.
    
    Args:
        value (Any): Valor a convertir.
        
    Returns:
        str: Valor convertido a texto compacto.
    """
    if not has_display_value(value):
        return "N/D"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def truncate_text(value: Any, width: int = SUMMARY_WIDTH) -> str:
    """
    Recorta textos largos para el resumen por consola.
    
    Args:
        value (Any): Valor a recortar.
        width (int, optional): Ancho máximo del texto. Por defecto SUMMARY_WIDTH.
        
    Returns:
        str: Texto recortado.
    """
    text = format_scalar(value).replace("\n", " ").strip()
    if len(text) <= width:
        return text
    return f"{text[: max(0, width - 3)]}..."


def print_rows(rows: list[tuple[str, Any]], *, stream: Any = None) -> bool:
    """
    Imprime filas clave/valor con alineación simple, omitiendo valores vacíos.
    
    Args:
        rows (list[tuple[str, Any]]): Filas clave/valor a imprimir.
        stream (Any, optional): Stream de salida. Por defecto sys.stdout.
        
    Returns:
        bool: True si se imprimieron filas, False en caso contrario.
    """
    target = stream or sys.stdout
    visible_rows = [(label, value) for label, value in rows if has_display_value(value)]
    if not visible_rows:
        return False
    label_width = max((len(label) for label, _ in visible_rows), default=10)

    for label, value in visible_rows:
        if isinstance(value, (dict, list)):
            print(f"{label:<{label_width}} :", file=target)
        else:
            print(f"{label:<{label_width}} : {truncate_text(value)}", file=target)
    print(file=target)
    return True


# --- FORMATEADORES ESPECÍFICOS DE TEST TELEMETRY ---

def format_metric(value: float | None, *, suffix: str = "") -> str:
    """
    Formatea una métrica numérica o devuelve N/D si no está disponible.
    
    Args:
        value (float | None): Métrica numérica a formatear.
        suffix (str, optional): Sufijo a añadir al valor. Por defecto "".
        
    Returns:
        str: Métrica formateada.
    """
    if value is None:
        return "N/D"
    if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
        return f"{int(value)}{suffix}"
    return f"{value:.3f}".rstrip("0").rstrip(".") + suffix


def _status_tag(status: str) -> str:
    """
    Normaliza el estado de una inferencia para mostrarlo en consola.
    
    Args:
        status (str): Estado de la inferencia.
        
    Returns:
        str: Estado normalizado.
    """
    return "OK" if status == "ok" else "ERR"


def _format_metric_window(summary: dict[str, float | None], *, suffix: str = "") -> str:
    """
    Representa avg/min/max de una métrica en una sola línea.
    
    Args:
        summary (dict[str, float | None]): Resumen de la métrica.
        suffix (str, optional): Sufijo a añadir al valor. Por defecto "".
        
    Returns:
        str: Métrica formateada.
    """
    avg = summary.get("avg")
    if avg is None:
        return "N/D"
    return (
        f"{format_metric(avg, suffix=suffix)} "
        f"(min {format_metric(summary.get('min'), suffix=suffix)} / "
        f"max {format_metric(summary.get('max'), suffix=suffix)})"
    )

def print_telemetry_report(summary: dict[str, Any], *, stream: Any = None) -> None:
    """
    Renderiza el informe final parametrizado de telemetría devuelto por el script.
    
    Args:
        summary (dict[str, Any]): Resumen de la telemetría.
        stream (Any, optional): Stream de salida. Por defecto sys.stdout.
    """
    target = stream or sys.stdout
    static_info = summary.get("static_model_info") or {}
    availability = summary.get("telemetry_availability") or {}
    ok_records = int(availability.get("ok_records") or 0)

    print_section("TELEMETRY PROBE", stream=target)
    print_rows(
        [
            ("Modelo", summary.get("model_id")),
            ("Esquema", summary.get("schema_name")),
            ("Muestra", summary.get("sample_size")),
            ("Inferencias OK", summary.get("ok")),
            ("Errores", summary.get("fail")),
            ("Modelo resuelto", static_info.get("resolved_model_id")),
            ("Arquitectura", static_info.get("architecture") or "N/D"),
            ("Stop reason", static_info.get("stop_reason") or "N/D"),
        ],
        stream=target,
    )

    print_section("RENDIMIENTO", stream=target)
    print_rows(
        [
            ("TTFT", _format_metric_window(summary.get("ttft") or {}, suffix=" s")),
            ("TPS", _format_metric_window(summary.get("tps") or {})),
            ("Generación", _format_metric_window(summary.get("generation_duration") or {}, suffix=" s")),
            ("Latencia total", _format_metric_window(summary.get("total_duration") or {}, suffix=" s")),
        ],
        stream=target,
    )

    print_section("TOKENS Y RECURSOS", stream=target)
    print_rows(
        [
            ("Prompt tokens", _format_metric_window(summary.get("prompt_tokens") or {})),
            ("Output tokens", _format_metric_window(summary.get("completion_tokens") or {})),
            ("Total tokens", _format_metric_window(summary.get("total_tokens") or {})),
            ("Reasoning JSON", _format_metric_window(summary.get("reasoning_tokens") or {})),
            ("GPU layers", _format_metric_window(summary.get("gpu_layers") or {})),
        ],
        stream=target,
    )

    print_section("COBERTURA", stream=target)
    print_rows(
        [
            ("TTFT", f"{availability.get('ttft_records', 0)}/{ok_records}"),
            ("TPS", f"{availability.get('tps_records', 0)}/{ok_records}"),
            ("GPU layers", f"{availability.get('gpu_layer_records', 0)}/{ok_records}"),
        ],
        stream=target,
    )

    notes = summary.get("notes") or {}
    ttft_note = notes.get("ttft")
    tps_note = notes.get("tps")
    if ttft_note or tps_note:
        print_section("OBSERVACIONES", stream=target)
        if ttft_note:
            print(f"- {ttft_note}", file=target)
        if tps_note:
            print(f"- {tps_note}", file=target)
        print(file=target)

    print_section("DETALLE POR IMAGEN", stream=target)
    records = summary.get("records") or []
    if not records:
        print("Sin registros.", file=target)
        print(file=target)
    else:
        for record in records:
            image_name = Path(str(record.get("image_path") or "N/D")).name
            status = str(record.get("status") or "error")
            if status == "ok":
                detail = (
                    f"TTFT={format_metric(record.get('ttft_seconds'), suffix=' s')} | "
                    f"TPS={format_metric(record.get('tokens_per_second'))} | "
                    f"total={format_metric(record.get('total_duration_seconds'), suffix=' s')}"
                )
            else:
                detail = str(record.get("error") or "unknown error")
            print(f"[{_status_tag(status)}] {image_name} | {detail}", file=target)
        print(file=target)

    prompt = summary.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        from pprint import pformat
        print_section("PROMPT USADO", stream=target)
        print(pformat(prompt.strip(), width=SUMMARY_WIDTH), file=target)
        print(file=target)

def print_telemetry_progress(payload: dict[str, Any], *, stream: Any = None) -> None:
    """
    Imprime el avance línea por línea de la telemetría para ejecución CLI interactiva.
    
    Args:
        payload (dict[str, Any]): Payload de la telemetría.
        stream (Any, optional): Stream de salida. Por defecto sys.stdout.
    """
    target = stream or sys.stdout
    event = str(payload.get("event") or "")
    index = int(payload.get("index") or 0)
    total = int(payload.get("total") or 0)
    image_path = payload.get("image_path")
    image_name = Path(str(image_path)).name if image_path else "N/D"
    record = payload.get("record")
    summary = payload.get("summary") or {}

    if event == "start":
        print(f"[telemetry] Preparando muestra de {total} imágenes...", file=target)
        return

    if event == "image_start":
        print(f"[{index}/{total}] Procesando {image_name}...", file=target)
        return

    if event == "image_done" and isinstance(record, dict):
        status = str(record.get("status") or "error")
        if status == "ok":
            print(
                (
                    f"[{index}/{total}] [{_status_tag(status)}] {image_name} | "
                    f"TTFT={format_metric(record.get('ttft_seconds'), suffix=' s')} | "
                    f"TPS={format_metric(record.get('tokens_per_second'))} | "
                    f"latencia={format_metric(record.get('total_duration_seconds'), suffix=' s')}"
                ),
                file=target,
            )
        else:
            print(f"[{index}/{total}] [{_status_tag(status)}] {image_name} | {record.get('error', 'unknown error')}", file=target)
        return

    if event == "complete":
        availability = summary.get("telemetry_availability") or {}
        ok_records = int(availability.get("ok_records") or 0)
        print(
            (
                f"[telemetry] Completado: {summary.get('ok', 0)} OK / {summary.get('fail', 0)} errores | "
                f"TTFT {availability.get('ttft_records', 0)}/{ok_records} | "
                f"TPS {availability.get('tps_records', 0)}/{ok_records} | "
                f"VRAM {availability.get('vram_records', 0)}/{ok_records}"
            ),
            file=target,
        )


# --- FORMATEADORES ESPECÍFICOS DE BATCH RUNNER ---

def print_batch_summary(summary: dict[str, Any], *, stream: Any = None) -> None:
    """
    Imprime resumen estándar de ejecución por lotes devuelto por batch_runner.
    
    Args:
        summary (dict[str, Any]): Resumen de la ejecución por lotes.
        stream (Any, optional): Stream de salida. Por defecto sys.stdout.
    """
    target = stream or sys.stdout
    print("\n=== BATCH RUNNER ===", file=target)
    print(f"Modelo        : {summary.get('model_id')}", file=target)
    print(f"Esquema       : {summary.get('schema_name')}", file=target)
    print(f"Procesadas    : {summary.get('processed')} de {summary.get('total_available')}", file=target)
    print(f"OK            : {summary.get('ok')}", file=target)
    print(f"Inválidas     : {summary.get('invalid')}", file=target)
    print(f"Errores       : {summary.get('fail')}", file=target)
    print(f"Origen entrada: {summary.get('input_source')}", file=target)
    if summary.get("input_source") == "manifest":
        print(f"Descartadas   : {summary.get('discarded_manifest_rows')}", file=target)
    print(f"Salida        : {summary.get('output_path')}", file=target)


# --- FORMATEADORES ESPECÍFICOS DE TEST RESPONSE ---

def build_summary_sections(payload: dict[str, Any]) -> list[tuple[str, list[tuple[str, Any]]]]:
    """
    Extrae secciones y filas visibles para CLI o TUI basado en response inspector.
    
    Args:
        payload (dict[str, Any]): Payload de la telemetría.
        
    Returns:
        list[tuple[str, list[tuple[str, Any]]]]: Secciones y filas visibles.
    """
    request = payload["request"]
    response = payload["response"]
    stats = response.get("stats") or {}
    model_info = response.get("model_info") or {}
    parsed = response.get("parsed") or {}

    has_parsed = isinstance(parsed, dict) and any(has_display_value(v) for v in parsed.values())

    sections: list[tuple[str, list[tuple[str, Any]]]] = [
        (
            "LM STUDIO RESPONSE INSPECTOR",
            [
                ("Model", request.get("model_id")),
                ("Image", request.get("image_path")),
                ("Schema", request.get("schema_name")),
                ("Response type", response.get("python_type")),
                ("Structured", response.get("structured")),
            ],
        ),
    ]

    attr_rows = [
        ("Model key", model_info.get("model_key")),
        ("Architecture", model_info.get("architecture")),
        ("Params", model_info.get("params_string")),
        ("Display name", model_info.get("display_name")),
    ]
    if not has_parsed:
        attr_rows.insert(0, ("Public attrs", ", ".join(response.get("public_attributes", {}).keys()) or None))
        attr_rows.append(("Text preview", response.get("text_extracted")))

    sections.append(("SDK ATTRIBUTES", attr_rows))

    stats_rows = [
        ("stop_reason", stats.get("stop_reason")),
        ("tokens_per_second", stats.get("tokens_per_second")),
        ("time_to_first_token_sec", stats.get("time_to_first_token_sec")),
        ("prompt_tokens_count", stats.get("prompt_tokens_count")),
        ("predicted_tokens_count", stats.get("predicted_tokens_count")),
        ("total_tokens_count", stats.get("total_tokens_count")),
        ("num_gpu_layers", stats.get("num_gpu_layers")),
    ]
    if any(has_display_value(value) for _, value in stats_rows):
        sections.append(("SDK STATS", stats_rows))

    if has_parsed:
        sections.append(("PARSED PAYLOAD", [(key, value) for key, value in parsed.items()]))
    elif has_display_value(parsed):
        sections.append(("PARSED PAYLOAD", [("parsed", parsed)]))

    return sections

def print_response_summary(payload: dict[str, Any], *, stream: Any = None) -> None:
    """
    Imprime un resumen visual tabular solo con la información que realmente existe (inspector).
    
    Args:
        payload (dict[str, Any]): Payload de la telemetría.
        stream (Any, optional): Stream de salida. Por defecto sys.stdout.
    """
    for title, rows in build_summary_sections(payload):
        visible_rows = [(label, value) for label, value in rows if has_display_value(value)]
        if not visible_rows:
            continue
        print_section(title, stream=stream)
        print_rows(visible_rows, stream=stream)


# --- FORMATEADORES ESPECÍFICOS DE TEST INFERENCE ---

def print_smoke_progress(payload: dict[str, Any], *, stream: Any = None) -> None:
    """
    Imprime el stream de salida simple para Test Inference durante uso CLI directo.
    
    Args:
        payload (dict[str, Any]): Payload de la telemetría.
        stream (Any, optional): Stream de salida. Por defecto sys.stdout.
    """
    event = payload.get("event")
    target = stream or sys.stdout

    if event == "start":
        print("\n=== TAREA: SMOKE TEST VLM (LM STUDIO) ===", file=target)
        print("1. Inicializando modelo...", file=target)

    elif event == "model_ready":
        print("✅ Modelo listo para inferir.", file=target)

    elif event == "case_start":
        index = payload.get("index")
        case = payload.get("case", {})
        print(f"\n--- Caso {index}: {case.get('id', 'N/D')} ({case.get('label', 'N/D')}) ---", file=target)
        print(f"Imagen: {case.get('path')}", file=target)

    elif event == "case_done":
        record = payload.get("record", {})
        status = record.get("status")
        msg = record.get("message")
        err = record.get("error")
        if status == "ok":
            print(f"OK: {msg}", file=target)
        elif status == "invalid":
            print(f"FAILED (invalid): {msg}", file=target)
        else:
            print(f"ERROR: {err}", file=target)

    elif event == "complete":
        summary = payload.get("summary", {})
        passed = summary.get("passed", 0)
        failed = summary.get("failed", 0)
        if failed == 0 and passed > 0:
            print("\n✅ TEST COMPLETADO CON ÉXITO.", file=target)
        else:
            print("\n❌ TEST FALLIDO: al menos una validación no pasó.", file=target)

# --- FORMATEADORES ESPECÍFICOS DE TEST SCHEMA ---

def print_schema_progress(payload: dict[str, Any], *, stream: Any = None) -> None:
    """
    Imprime el avance línea por línea de la inferencia de esquemas (test_schema) para CLI.
    
    Args:
        payload (dict[str, Any]): Payload de la telemetría.
        stream (Any, optional): Stream de salida. Por defecto sys.stdout.
    """
    target = stream or sys.stdout
    event = str(payload.get("event") or "")
    index = int(payload.get("index") or 0)
    total = int(payload.get("total") or 0)
    image_path = payload.get("image_path")
    record = payload.get("record") or {}
    summary = payload.get("summary") or {}

    if event == "start":
        print(f"\nModelo  : {summary.get('model_id')}", file=target)
        print(f"Esquema : {summary.get('schema_name')}", file=target)
        print(f"Imágenes: {summary.get('sample_size')} (de {summary.get('total_available')} disponibles)", file=target)
        
        prompt = summary.get('prompt', '')
        print(f"Prompt  : {prompt[:100]}{'...' if len(prompt) > 100 else ''}", file=target)
        print("-" * 60, file=target)
        return

    if event == "image_start":
        import os
        try:
            rel = os.path.relpath(image_path, os.getcwd()) if image_path else "N/D"
        except ValueError:
            rel = str(image_path) if image_path else "N/D"
        print(f"\n  ▶ [{index}/{total}] {rel}", file=target)
        return

    if event == "image_done":
        status = str(record.get("status") or "error")
        if status == "ok":
            print("  ✔ Schema OK", file=target)
        elif status == "invalid":
            reason = record.get("validation_error", "Unknown validation error")
            print(f"  ⚠ Schema INVALID: {reason}", file=target)
        else:
            err = record.get("error", "Unknown error")
            print(f"  ✗ {err}", file=target)
        return

    if event == "complete":
        print("\n" + "-" * 60, file=target)
        print(f"  ✔ {summary.get('ok', 0)} válidas  ⚠ {summary.get('invalid', 0)} inválidas  ✗ {summary.get('fail', 0)} errores  (de {total} imágenes)\n", file=target)

