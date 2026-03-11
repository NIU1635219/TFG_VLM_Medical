"""Orquestador CLI para inferencia masiva sobre directorios de imágenes."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
_CSV_META_FIELDS: list[str] = [
    "timestamp_utc",
    "model_id",
    "schema_name",
    "image_path",
    "image_name",
    "status",
    "duration_seconds",
    "total_duration_seconds",
    "ttft_seconds",
    "generation_duration_seconds",
    "prompt_tokens_per_second",
    "tokens_per_second",
    "reasoning_tokens",
    "stop_reason",
    "resolved_model_id",
    "vram_usage_mb",
    "total_params",
    "architecture",
    "quantization",
    "gpu_layers",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "error_type",
    "error_message",
]


def build_parser() -> argparse.ArgumentParser:
    """Construye el parser CLI del batch runner."""
    parser = argparse.ArgumentParser(
        description=(
            "Procesa un directorio de imágenes con un VLM en LM Studio y guarda "
            "resultados incrementales en CSV o JSONL."
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
        "--output-format",
        choices=("csv", "jsonl"),
        help="Formato de salida. Si se omite, se infiere de la extensión o se usa csv.",
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


def infer_output_format(output_path: Path, explicit_format: str | None) -> str:
    """Determina el formato de salida efectivo."""
    if explicit_format:
        return explicit_format

    suffix = output_path.suffix.lower()
    if suffix == ".jsonl":
        return "jsonl"
    return "csv"


def build_default_output_path(schema_name: str, output_format: str) -> Path:
    """Genera una ruta de salida por defecto bajo data/processed/batch_results."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    extension = "jsonl" if output_format == "jsonl" else "csv"
    return Path(_PROJECT_ROOT) / "data" / "processed" / "batch_results" / f"batch_{schema_name}_{timestamp}.{extension}"


def _serialize_scalar(value: Any) -> Any:
    """Convierte valores complejos a una representación persistible."""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


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


def append_csv_record(output_path: Path, record: dict[str, Any], fieldnames: list[str]) -> None:
    """Añade un registro CSV conservando cabecera y compatibilidad entre reintentos."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header_exists = output_path.exists() and output_path.stat().st_size > 0

    if header_exists:
        with output_path.open("r", encoding="utf-8", newline="") as existing_file:
            reader = csv.reader(existing_file)
            existing_header = next(reader, [])
        if existing_header and existing_header != fieldnames:
            raise RuntimeError(
                "La cabecera del CSV existente no coincide con el esquema actual. "
                f"Archivo: {output_path}"
            )

    row = {field_name: _serialize_scalar(record.get(field_name, "")) for field_name in fieldnames}
    with output_path.open("a", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        if not header_exists:
            writer.writeheader()
        writer.writerow(row)
        _sync_file(file_handle)


def persist_record(
    output_path: Path,
    output_format: str,
    record: dict[str, Any],
    *,
    fieldnames: list[str],
) -> None:
    """Persiste un registro en el formato solicitado."""
    if output_format == "jsonl":
        append_jsonl_record(output_path, record)
        return
    append_csv_record(output_path, record, fieldnames)


def classify_error(error: BaseException) -> str:
    """Clasifica un error para la telemetría del lote."""
    message = str(error).lower()
    if "no cumple el esquema" in message or "json válido" in message:
        return "invalid"
    return "error"


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
    """Construye un registro homogéneo para CSV o JSONL."""
    record: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "schema_name": schema_name,
        "image_path": str(image_path),
        "image_name": image_path.name,
        "status": status,
        "duration_seconds": round(duration_seconds, 6),
        "total_duration_seconds": telemetry.get("total_duration_seconds") if telemetry else None,
        "ttft_seconds": telemetry.get("ttft_seconds") if telemetry else None,
        "generation_duration_seconds": telemetry.get("generation_duration_seconds") if telemetry else None,
        "prompt_tokens_per_second": telemetry.get("prompt_tokens_per_second") if telemetry else None,
        "tokens_per_second": telemetry.get("tokens_per_second") if telemetry else None,
        "reasoning_tokens": telemetry.get("reasoning_tokens") if telemetry else None,
        "stop_reason": telemetry.get("stop_reason") if telemetry else None,
        "resolved_model_id": telemetry.get("model_id") if telemetry else None,
        "vram_usage_mb": telemetry.get("vram_usage_mb") if telemetry else None,
        "total_params": telemetry.get("total_params") if telemetry else None,
        "architecture": telemetry.get("architecture") if telemetry else None,
        "quantization": telemetry.get("quantization") if telemetry else None,
        "gpu_layers": telemetry.get("gpu_layers") if telemetry else None,
        "prompt_tokens": telemetry.get("prompt_tokens") if telemetry else None,
        "completion_tokens": telemetry.get("completion_tokens") if telemetry else None,
        "total_tokens": telemetry.get("total_tokens") if telemetry else None,
        "error_type": type(error).__name__ if error is not None else "",
        "error_message": str(error) if error is not None else "",
    }
    if payload:
        record.update(payload)
        record["payload"] = payload
    else:
        record["payload"] = None
    return record


def run_batch_job(
    *,
    model_id: str,
    image_dir: str | Path,
    schema_name: str,
    include_reasoning: bool = False,
    output_path: str | Path | None = None,
    output_format: str | None = None,
    max_images: int | None = None,
    shuffle: bool = False,
    seed: int | None = None,
    temperature: float = 0.0,
    prompt: str | None = None,
    server_api_host: str | None = None,
    api_token: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Ejecuta el lote completo y devuelve un resumen final."""
    public_schema_name, schema_cls = resolve_schema(schema_name, include_reasoning)
    selected_prompt = prompt.strip() if prompt and prompt.strip() else build_prompt_for_schema(public_schema_name, schema_cls)
    discovered_images = iter_image_paths(Path(image_dir))
    images = select_images(discovered_images, max_images=max_images, shuffle=shuffle, seed=seed)

    resolved_output_path = Path(output_path) if output_path is not None else build_default_output_path(
        public_schema_name,
        output_format or "csv",
    )
    resolved_output_format = infer_output_format(resolved_output_path, output_format)
    csv_fieldnames = [*_CSV_META_FIELDS, *schema_cls.model_fields.keys()]

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

    try:
        loader.load_model()
        progress_iter = tqdm(images, desc="Procesando imágenes", unit="img")
        for image_path in progress_iter:
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

            persist_record(
                resolved_output_path,
                resolved_output_format,
                record,
                fieldnames=csv_fieldnames,
            )
            processed += 1
    finally:
        loader.unload_model()

    return {
        "model_id": model_id,
        "schema_name": public_schema_name,
        "prompt": selected_prompt,
        "output_path": str(resolved_output_path),
        "output_format": resolved_output_format,
        "processed": processed,
        "ok": ok,
        "invalid": invalid,
        "fail": fail,
        "total_available": len(discovered_images),
    }


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
        output_format=args.output_format,
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