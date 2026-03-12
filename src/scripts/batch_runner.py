"""Orquestador CLI para inferencia masiva sobre directorios de imágenes."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pydantic import BaseModel

from src.inference.schemas import SCHEMA_REGISTRY, get_schema_variant
from src.inference.vlm_runner import VLMLoader
from src.scripts.test_schema import build_prompt_for_schema

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable: Any, **_: Any) -> Any:
        return iterable


_IMG_EXT: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
_TELEMETRY_RECORD_FIELDS: dict[str, str] = {
    "total_duration_seconds": "total_duration_seconds",
    "ttft_seconds": "ttft_seconds",
    "generation_duration_seconds": "generation_duration_seconds",
    "tokens_per_second": "tokens_per_second",
    "reasoning_tokens": "reasoning_tokens",
    "stop_reason": "stop_reason",
    "resolved_model_id": "model_id",
    "architecture": "architecture",
    "gpu_layers": "gpu_layers",
    "prompt_tokens": "prompt_tokens",
    "completion_tokens": "completion_tokens",
    "total_tokens": "total_tokens",
}


def build_parser() -> argparse.ArgumentParser:
    """Construye el parser CLI del batch runner."""
    parser = argparse.ArgumentParser(
        description=(
            "Procesa un directorio de imágenes con un VLM en LM Studio y guarda "
            "resultados incrementales en JSONL."
        )
    )
    parser.add_argument("--model", required=True, help="Identificador del modelo en LM Studio.")
    parser.add_argument("--image-dir", required=True, help="Directorio raíz de imágenes a procesar.")
    parser.add_argument(
        "--schema",
        required=True,
        help="Nombre del esquema base registrado (por ejemplo, PolypDetection).",
    )
    parser.add_argument(
        "--with-reasoning",
        action="store_true",
        help="Usa la variante del esquema que exige reasoning explícito.",
    )
    parser.add_argument(
        "--output",
        help="Ruta del archivo de salida. Si se omite, se genera en data/processed/batch_results/.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Número máximo de imágenes a procesar. Si se omite, procesa todas.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Baraja las imágenes antes de seleccionar el subconjunto a procesar.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semilla aleatoria usada cuando --shuffle está activo.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperatura de inferencia. Por defecto 0.0 para mayor determinismo.",
    )
    parser.add_argument("--prompt", help="Prompt manual. Si se omite, se construye desde el esquema.")
    parser.add_argument("--server-api-host", help="Host del servidor LM Studio si no es el local por defecto.")
    parser.add_argument("--api-token", help="Token de API para LM Studio si aplica.")
    parser.add_argument("--verbose", action="store_true", help="Activa logs verbosos del loader.")
    return parser


def iter_image_paths(image_dir: Path) -> list[Path]:
    """Descubre imágenes compatibles bajo un directorio o archivo concreto."""
    normalized_path = image_dir.expanduser().resolve()
    if not normalized_path.exists():
        raise FileNotFoundError(f"No existe la ruta indicada: {normalized_path}")

    if normalized_path.is_file():
        if normalized_path.suffix.lower() not in _IMG_EXT:
            raise ValueError(f"El archivo no tiene una extensión de imagen soportada: {normalized_path}")
        return [normalized_path]

    image_paths = [
        path.resolve()
        for path in sorted(normalized_path.rglob("*"))
        if path.is_file() and path.suffix.lower() in _IMG_EXT
    ]
    if not image_paths:
        raise RuntimeError(f"No se encontraron imágenes compatibles dentro de: {normalized_path}")
    return image_paths


def resolve_schema(schema_name: str, include_reasoning: bool) -> tuple[str, type[BaseModel]]:
    """Resuelve el esquema final a partir del nombre solicitado por CLI."""
    normalized_name = schema_name.strip()
    if normalized_name.endswith("WithReasoning"):
        normalized_name = normalized_name.removesuffix("WithReasoning")
        include_reasoning = True

    if normalized_name not in SCHEMA_REGISTRY:
        available = ", ".join(sorted(SCHEMA_REGISTRY.keys()))
        raise ValueError(f"Esquema desconocido '{schema_name}'. Disponibles: {available}")

    return get_schema_variant(normalized_name, include_reasoning)


def select_images(
    image_paths: list[Path],
    *,
    max_images: int | None = None,
    shuffle: bool = False,
    seed: int | None = None,
) -> list[Path]:
    """Selecciona el subconjunto final de imágenes a procesar."""
    selected = list(image_paths)
    if shuffle:
        random.Random(seed).shuffle(selected)

    if max_images is not None:
        if max_images <= 0:
            raise ValueError("max_images debe ser mayor que cero")
        selected = selected[:max_images]

    return selected


def build_default_output_path(schema_name: str) -> Path:
    """Genera una ruta JSONL por defecto bajo data/processed/batch_results."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path(_PROJECT_ROOT) / "data" / "processed" / "batch_results" / f"batch_{schema_name}_{timestamp}.jsonl"


def _sync_file(file_handle: Any) -> None:
    """Fuerza el volcado a disco tras cada append para minimizar pérdida de datos."""
    file_handle.flush()
    os.fsync(file_handle.fileno())


def append_jsonl_record(output_path: Path, record: dict[str, Any]) -> None:
    """Añade un registro JSONL y lo fuerza a disco inmediatamente."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8", newline="\n") as file_handle:
        file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        _sync_file(file_handle)


def classify_error(error: BaseException) -> str:
    """Clasifica un error para la telemetría del lote."""
    message = str(error).lower()
    if "no cumple el esquema" in message or "json válido" in message:
        return "invalid"
    return "error"


def _prune_missing_fields(value: Any) -> Any:
    """Elimina claves opcionales ausentes en salidas JSON-like sin tocar 0 o False."""
    if isinstance(value, dict):
        pruned: dict[str, Any] = {}
        for key, item in value.items():
            cleaned = _prune_missing_fields(item)
            if cleaned is None:
                continue
            if isinstance(cleaned, dict) and not cleaned:
                continue
            if isinstance(cleaned, list) and not cleaned:
                continue
            pruned[key] = cleaned
        return pruned
    if isinstance(value, list):
        items = [_prune_missing_fields(item) for item in value]
        return [item for item in items if item is not None]
    return value


def build_record(
    *,
    model_id: str,
    schema_name: str,
    image_path: Path,
    duration_seconds: float,
    status: str,
    payload: dict[str, Any] | None = None,
    telemetry: dict[str, Any] | None = None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    """Construye un registro homogéneo para JSONL."""
    record: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "schema_name": schema_name,
        "image_path": str(image_path),
        "image_name": image_path.name,
        "status": status,
        "duration_seconds": round(duration_seconds, 6),
        "error_type": type(error).__name__ if error is not None else None,
        "error_message": str(error) if error is not None else None,
    }
    if telemetry:
        record.update(
            {
                record_key: telemetry.get(telemetry_key)
                for record_key, telemetry_key in _TELEMETRY_RECORD_FIELDS.items()
            }
        )
    if payload:
        record["payload"] = payload
    return _prune_missing_fields(record)


def _empty_metric_summary() -> dict[str, float | None]:
    """Construye una estructura homogénea para métricas opcionales."""
    return {"avg": None}


def _compute_metric_summary(total: float, count: int) -> dict[str, float | None]:
    """Calcula una media incremental o marca la métrica como no disponible."""
    if count <= 0:
        return _empty_metric_summary()
    return {"avg": total / count}


def _build_batch_summary(
    *,
    model_id: str,
    schema_name: str,
    prompt: str,
    output_path: Path,
    processed: int,
    ok: int,
    invalid: int,
    fail: int,
    total_available: int,
    sample_size: int,
    ttft_total: float,
    ttft_count: int,
    tps_total: float,
    tps_count: int,
    total_duration_total: float,
    total_duration_count: int,
) -> dict[str, Any]:
    """Construye el resumen parcial o final del batch runner."""
    return {
        "model_id": model_id,
        "schema_name": schema_name,
        "prompt": prompt,
        "output_path": str(output_path),
        "processed": processed,
        "ok": ok,
        "invalid": invalid,
        "fail": fail,
        "total_available": total_available,
        "sample_size": sample_size,
        "ttft": _compute_metric_summary(ttft_total, ttft_count),
        "tps": _compute_metric_summary(tps_total, tps_count),
        "total_duration": _compute_metric_summary(total_duration_total, total_duration_count),
    }


def run_batch_job(
    *,
    model_id: str,
    image_dir: str | Path,
    schema_name: str,
    include_reasoning: bool = False,
    output_path: str | Path | None = None,
    max_images: int | None = None,
    shuffle: bool = False,
    seed: int | None = None,
    temperature: float = 0.0,
    prompt: str | None = None,
    server_api_host: str | None = None,
    api_token: str | None = None,
    verbose: bool = False,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Ejecuta el lote completo y devuelve un resumen final."""
    public_schema_name, schema_cls = resolve_schema(schema_name, include_reasoning)
    selected_prompt = prompt.strip() if prompt and prompt.strip() else build_prompt_for_schema(public_schema_name, schema_cls)
    discovered_images = iter_image_paths(Path(image_dir))
    images = select_images(discovered_images, max_images=max_images, shuffle=shuffle, seed=seed)

    resolved_output_path = Path(output_path) if output_path is not None else build_default_output_path(public_schema_name)
    if resolved_output_path.suffix.lower() != ".jsonl":
        resolved_output_path = resolved_output_path.with_suffix(".jsonl")

    loader = VLMLoader(
        model_path=model_id,
        verbose=verbose,
        server_api_host=server_api_host,
        api_token=api_token,
    )

    ok = 0
    invalid = 0
    fail = 0
    processed = 0
    ttft_total = 0.0
    ttft_count = 0
    tps_total = 0.0
    tps_count = 0
    total_duration_total = 0.0
    total_duration_count = 0

    def build_summary() -> dict[str, Any]:
        return _build_batch_summary(
            model_id=model_id,
            schema_name=public_schema_name,
            prompt=selected_prompt,
            output_path=resolved_output_path,
            processed=processed,
            ok=ok,
            invalid=invalid,
            fail=fail,
            total_available=len(discovered_images),
            sample_size=len(images),
            ttft_total=ttft_total,
            ttft_count=ttft_count,
            tps_total=tps_total,
            tps_count=tps_count,
            total_duration_total=total_duration_total,
            total_duration_count=total_duration_count,
        )

    def emit_progress(
        event: str,
        *,
        index: int = 0,
        image_path: str | None = None,
        status: str | None = None,
        record: dict[str, Any] | None = None,
    ) -> None:
        if on_progress is None:
            return
        on_progress(
            {
                "event": event,
                "index": index,
                "total": len(images),
                "image_path": image_path,
                "status": status,
                "record": dict(record) if record is not None else None,
                "summary": build_summary(),
            }
        )

    emit_progress("start")

    try:
        loader.load_model()
        progress_iter = tqdm(images, desc="Procesando imágenes", unit="img") if on_progress is None else images
        for index, image_path in enumerate(progress_iter, start=1):
            emit_progress("image_start", index=index, image_path=str(image_path), status="running")
            started_at = time.perf_counter()
            try:
                result = loader.inference(
                    image_path=image_path,
                    prompt=selected_prompt,
                    schema=schema_cls,
                    temperature=temperature,
                    include_telemetry=True,
                )
                duration_seconds = time.perf_counter() - started_at
                payload = result.data.model_dump()
                telemetry_payload = (
                    dataclasses.asdict(result.telemetry)
                    if dataclasses.is_dataclass(result.telemetry)
                    else dict(getattr(result.telemetry, "__dict__", {}))
                )
                record = build_record(
                    model_id=model_id,
                    schema_name=public_schema_name,
                    image_path=image_path,
                    duration_seconds=duration_seconds,
                    status="ok",
                    payload=payload,
                    telemetry=telemetry_payload,
                )
                ok += 1
                ttft_value = telemetry_payload.get("ttft_seconds")
                if isinstance(ttft_value, (int, float)):
                    ttft_total += float(ttft_value)
                    ttft_count += 1
                tps_value = telemetry_payload.get("tokens_per_second")
                if isinstance(tps_value, (int, float)):
                    tps_total += float(tps_value)
                    tps_count += 1
                total_duration_value = telemetry_payload.get("total_duration_seconds")
                if isinstance(total_duration_value, (int, float)):
                    total_duration_total += float(total_duration_value)
                    total_duration_count += 1
            except Exception as error:
                duration_seconds = time.perf_counter() - started_at
                status = classify_error(error)
                record = build_record(
                    model_id=model_id,
                    schema_name=public_schema_name,
                    image_path=image_path,
                    duration_seconds=duration_seconds,
                    status=status,
                    error=error,
                )
                if status == "invalid":
                    invalid += 1
                else:
                    fail += 1

            append_jsonl_record(resolved_output_path, record)
            processed += 1
            emit_progress(
                "image_done",
                index=index,
                image_path=str(image_path),
                status=str(record.get("status") or "error"),
                record=record,
            )
    finally:
        loader.unload_model()

    summary = build_summary()
    emit_progress("complete", index=processed, status="complete")
    return summary


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada CLI del batch runner."""
    parser = build_parser()
    args = parser.parse_args(argv)

    summary = run_batch_job(
        model_id=args.model,
        image_dir=args.image_dir,
        schema_name=args.schema,
        include_reasoning=args.with_reasoning,
        output_path=args.output,
        max_images=args.max_images,
        shuffle=args.shuffle,
        seed=args.seed,
        temperature=args.temperature,
        prompt=args.prompt,
        server_api_host=args.server_api_host,
        api_token=args.api_token,
        verbose=args.verbose,
    )

    print("\n=== BATCH RUNNER ===")
    print(f"Modelo        : {summary['model_id']}")
    print(f"Esquema       : {summary['schema_name']}")
    print(f"Procesadas    : {summary['processed']} de {summary['total_available']}")
    print(f"OK            : {summary['ok']}")
    print(f"Inválidas     : {summary['invalid']}")
    print(f"Errores       : {summary['fail']}")
    print(f"Salida        : {summary['output_path']}")

    return 0 if summary["fail"] == 0 and summary["invalid"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())