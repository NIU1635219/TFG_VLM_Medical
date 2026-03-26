"""Prueba de telemetría basada en `response.stats` del SDK y metadatos del modelo."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable, cast

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pydantic import BaseModel

from src.inference.schemas import SCHEMA_REGISTRY, get_schema_variant
from src.inference.vlm_runner import VLMLoader
from src.scripts.test_schema import build_prompt_for_schema, find_images
from src.utils.tests_ui.metrics import summarize_numeric as summarize_numeric_values


def build_parser() -> argparse.ArgumentParser:
    """
    Construye el parser CLI para la prueba de telemetría.

    Returns:
        ArgumentParser con los argumentos de la CLI.
    """
    parser = argparse.ArgumentParser(description="Mide TTFT y TPS de un modelo VLM en LM Studio.")
    parser.add_argument("--model", required=True, help="Identificador del modelo cargado en LM Studio.")
    parser.add_argument("--schema", required=True, help="Nombre del esquema base registrado.")
    parser.add_argument("--with-reasoning", action="store_true", help="Usa la variante del esquema con reasoning.")
    parser.add_argument("--max-images", type=int, default=5, help="Número máximo de imágenes a muestrear.")
    parser.add_argument("--seed", type=int, default=None, help="Semilla de muestreo aleatorio.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperatura de inferencia.")
    parser.add_argument("--prompt", help="Prompt manual opcional.")
    parser.add_argument("--server-api-host", help="Host del servidor LM Studio si no es el local por defecto.")
    parser.add_argument("--api-token", help="Token de API si aplica.")
    return parser


def resolve_schema(schema_name: str, include_reasoning: bool) -> tuple[str, type[BaseModel]]:
    """
    Resuelve el esquema a partir del nombre CLI.

    Args:
        schema_name: Nombre del esquema.
        include_reasoning: Incluye razonamiento.

    Returns:
        Tupla con el nombre del esquema y la clase del esquema.
    """
    normalized_name = schema_name.strip()
    if normalized_name.endswith("WithReasoning"):
        normalized_name = normalized_name.removesuffix("WithReasoning")
        include_reasoning = True

    if normalized_name not in SCHEMA_REGISTRY:
        available = ", ".join(sorted(SCHEMA_REGISTRY.keys()))
        raise ValueError(f"Esquema desconocido '{schema_name}'. Disponibles: {available}")

    return get_schema_variant(normalized_name, include_reasoning)


def select_sample(images: list[str], max_images: int, seed: int | None = None) -> list[str]:
    """
    Selecciona una muestra aleatoria reproducible de imágenes.

    Args:
        images: Lista de imágenes.
        max_images: Número máximo de imágenes a muestrear.
        seed: Semilla de muestreo aleatorio.

    Returns:
        Lista de imágenes muestreadas.
    """
    if max_images <= 0:
        raise ValueError("max_images debe ser mayor que cero")
    if len(images) <= max_images:
        return list(images)
    return random.Random(seed).sample(images, max_images)


def summarize_numeric(values: list[float]) -> dict[str, float | None]:
    """
    Resume una lista numérica con media, mínimo y máximo.

    Args:
        values: Lista de valores numéricos.

    Returns:
        Diccionario con la media, mínimo y máximo.
    """
    return summarize_numeric_values(values)


def first_available_value(records: list[dict[str, Any]], field_name: str) -> Any:
    """
    Devuelve el primer valor no vacío encontrado en una lista de registros.

    Args:
        records: Lista de registros.
        field_name: Nombre del campo.

    Returns:
        Primer valor no vacío encontrado.
    """
    for record in records:
        value = record.get(field_name)
        if value not in (None, "", []):
            return value
    return None


def _build_telemetry_summary(
    *,
    model_id: str,
    schema_name: str,
    sample_size: int,
    records: list[dict[str, Any]],
    ok: int,
    fail: int,
    prompt: str,
) -> dict[str, Any]:
    """
    Construye un resumen parcial o final a partir de los registros acumulados.

    Args:
        model_id: ID del modelo.
        schema_name: Nombre del esquema.
        sample_size: Tamaño de la muestra.
        records: Lista de registros.
        ok: Número de inferencias correctas.
        fail: Número de errores de inferencia.
        prompt: Prompt usado.

    Returns:
        Resumen parcial o final.
    """
    ttft_values = [record["ttft_seconds"] for record in records if record.get("status") == "ok" and isinstance(record.get("ttft_seconds"), (int, float))]
    tps_values = [record["tokens_per_second"] for record in records if record.get("status") == "ok" and isinstance(record.get("tokens_per_second"), (int, float))]
    generation_values = [record["generation_duration_seconds"] for record in records if record.get("status") == "ok" and isinstance(record.get("generation_duration_seconds"), (int, float))]
    reasoning_values = [record["reasoning_tokens"] for record in records if record.get("status") == "ok" and isinstance(record.get("reasoning_tokens"), (int, float))]
    gpu_layer_values = [record["gpu_layers"] for record in records if record.get("status") == "ok" and isinstance(record.get("gpu_layers"), (int, float))]
    prompt_token_values = [record["prompt_tokens"] for record in records if record.get("status") == "ok" and isinstance(record.get("prompt_tokens"), (int, float))]
    completion_token_values = [record["completion_tokens"] for record in records if record.get("status") == "ok" and isinstance(record.get("completion_tokens"), (int, float))]
    total_token_values = [record["total_tokens"] for record in records if record.get("status") == "ok" and isinstance(record.get("total_tokens"), (int, float))]
    total_values = [record["total_duration_seconds"] for record in records if record.get("status") == "ok" and isinstance(record.get("total_duration_seconds"), (int, float))]

    tps_note: str | None = None
    ttft_note: str | None = None
    if ok > 0 and not ttft_values:
        ttft_note = "LM Studio no devolvio TTFT en response.stats; el probe no lo estima a partir de la latencia total."
    if ok > 0 and not tps_values:
        tps_note = "LM Studio no devolvio tokens_per_second en response.stats; TPS no disponible en esta ejecucion."

    summary = {
        "model_id": model_id,
        "schema_name": schema_name,
        "sample_size": sample_size,
        "ok": ok,
        "fail": fail,
        "records": list(records),
        "ttft": summarize_numeric(ttft_values),
        "tps": summarize_numeric(tps_values),
        "generation_duration": summarize_numeric(generation_values),
        "reasoning_tokens": summarize_numeric(reasoning_values),
        "gpu_layers": summarize_numeric(gpu_layer_values),
        "prompt_tokens": summarize_numeric(prompt_token_values),
        "completion_tokens": summarize_numeric(completion_token_values),
        "total_tokens": summarize_numeric(total_token_values),
        "total_duration": summarize_numeric(total_values),
        "static_model_info": {
            "resolved_model_id": first_available_value(records, "model_id") or model_id,
            "architecture": first_available_value(records, "architecture"),
            "stop_reason": first_available_value(records, "stop_reason"),
        },
        "telemetry_availability": {
            "ttft_records": len(ttft_values),
            "tps_records": len(tps_values),
            "generation_records": len(generation_values),
            "reasoning_records": len(reasoning_values),
            "gpu_layer_records": len(gpu_layer_values),
            "prompt_token_records": len(prompt_token_values),
            "completion_token_records": len(completion_token_values),
            "total_token_records": len(total_token_values),
            "total_duration_records": len(total_values),
            "ok_records": ok,
        },
        "notes": {
            "ttft": ttft_note,
            "tps": tps_note,
        },
        "prompt": prompt,
    }
    return summary


def run_telemetry_batch(
    model_id: str,
    schema_name: str,
    schema_cls: type[BaseModel],
    images: list[str],
    *,
    max_images: int = 5,
    seed: int | None = None,
    temperature: float = 0.0,
    prompt: str | None = None,
    server_api_host: str | None = None,
    api_token: str | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """
    Ejecuta una prueba de telemetría sobre una muestra de imágenes.

    Args:
        model_id: ID del modelo.
        schema_name: Nombre del esquema.
        schema_cls: Clase del esquema.
        images: Lista de imágenes.
        max_images: Número máximo de imágenes a muestrear.
        seed: Semilla de muestreo aleatorio.
        temperature: Temperatura de inferencia.
        prompt: Prompt usado.
        server_api_host: Host del servidor LM Studio.
        api_token: Token de API.
        on_progress: Callback de progreso.

    Returns:
        Resumen de la prueba de telemetría.
    """
    sample = select_sample(images, max_images=max_images, seed=seed)
    selected_prompt = prompt.strip() if prompt and prompt.strip() else build_prompt_for_schema(schema_name, schema_cls)

    loader = VLMLoader(
        model_path=model_id,
        verbose=False,
        server_api_host=server_api_host,
        api_token=api_token,
    )

    records: list[dict[str, Any]] = []
    ok = 0
    fail = 0

    def emit_progress(event: str, *, index: int = 0, image_path: str | None = None, status: str | None = None, record: dict[str, Any] | None = None) -> None:
        """
        Emite un evento de progreso.

        Args:
            event: Evento de progreso.
            index: Índice del evento.
            image_path: Ruta de la imagen.
            status: Estado del evento.
            record: Registro del evento.
        """
        if on_progress is None:
            return
        summary = _build_telemetry_summary(
            model_id=model_id,
            schema_name=schema_name,
            sample_size=len(sample),
            records=records,
            ok=ok,
            fail=fail,
            prompt=selected_prompt,
        )
        on_progress(
            {
                "event": event,
                "index": index,
                "total": len(sample),
                "image_path": image_path,
                "status": status,
                "record": dict(record) if record is not None else None,
                "summary": summary,
            }
        )

    emit_progress("start")

    try:
        loader.load_model()
        for index, image_path in enumerate(sample, start=1):
            emit_progress("image_start", index=index, image_path=image_path, status="running")
            try:
                result = loader.inference(
                    image_path=image_path,
                    prompt=selected_prompt,
                    schema=schema_cls,
                    temperature=temperature,
                    include_telemetry=True,
                )
                telemetry = result.telemetry
                records.append(
                    {
                        "image_path": image_path,
                        "status": "ok",
                        "ttft_seconds": telemetry.ttft_seconds,
                        "generation_duration_seconds": telemetry.generation_duration_seconds,
                        "tokens_per_second": telemetry.tokens_per_second,
                        "reasoning_tokens": telemetry.reasoning_tokens,
                        "stop_reason": telemetry.stop_reason,
                        "model_id": telemetry.model_id,
                        "architecture": telemetry.architecture,
                        "gpu_layers": telemetry.gpu_layers,
                        "prompt_tokens": telemetry.prompt_tokens,
                        "completion_tokens": telemetry.completion_tokens,
                        "total_tokens": telemetry.total_tokens,
                        "total_duration_seconds": telemetry.total_duration_seconds,
                    }
                )
                ok += 1
                emit_progress("image_done", index=index, image_path=image_path, status="ok", record=records[-1])
            except Exception as error:
                records.append(
                    {
                        "image_path": image_path,
                        "status": "error",
                        "error": str(error),
                    }
                )
                fail += 1
                emit_progress("image_done", index=index, image_path=image_path, status="error", record=records[-1])
    finally:
        loader.unload_model()

    summary = _build_telemetry_summary(
        model_id=model_id,
        schema_name=schema_name,
        sample_size=len(sample),
        records=records,
        ok=ok,
        fail=fail,
        prompt=selected_prompt,
    )
    emit_progress("complete", index=len(sample), status="complete")
    return summary


def main(argv: list[str] | None = None) -> int:
    """
    CLI de la prueba de telemetría.

    Args:
        argv: Argumentos de línea de comandos.

    Returns:
        Código de salida.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    schema_name, schema_cls = resolve_schema(args.schema, args.with_reasoning)
    images = find_images()
    if not images:
        raise RuntimeError("No se encontraron imágenes en los directorios del proyecto.")

    from src.utils.tests_ui.cli_reporters import print_telemetry_progress, print_telemetry_report

    summary = run_telemetry_batch(
        model_id=args.model,
        schema_name=schema_name,
        schema_cls=schema_cls,
        images=images,
        max_images=args.max_images,
        seed=args.seed,
        temperature=args.temperature,
        prompt=args.prompt,
        server_api_host=args.server_api_host,
        api_token=args.api_token,
        on_progress=print_telemetry_progress,
    )

    print()
    print_telemetry_report(summary)

    return 0 if summary["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())