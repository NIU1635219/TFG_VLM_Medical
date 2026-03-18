"""Inspecciona la respuesta real del SDK de LM Studio para una petición multimodal."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pydantic import BaseModel

from src.inference.schemas import SCHEMA_REGISTRY, get_schema_variant
from src.inference.vlm_runner import VLMLoader
from src.scripts.test_schema import build_prompt_for_schema, find_images
from src.utils.models_ui.lms_models import get_installed_lms_models, get_installed_models, list_loaded_llm_model_keys
from src.utils.tests_ui.cli_reporters import (
    build_summary_sections,
    print_response_summary,
    print_section,
)

DEFAULT_RAW_PROMPT = (
    "Analiza esta imagen y responde con el máximo detalle posible para fines de depuración. "
    "Si puedes, describe el contenido visible y la confianza de tu respuesta."
)
DEFAULT_STRUCTURED_SCHEMA = "GenericObjectDetection"


def build_parser() -> argparse.ArgumentParser:
    """
    Construye la CLI del inspector de respuestas.

    Returns:
        ArgumentParser con los argumentos de la CLI.
    """
    parser = argparse.ArgumentParser(
        description="Inspecciona los campos útiles devueltos por LM Studio ante una petición multimodal."
    )
    parser.add_argument("--model", help="Identificador del modelo cargado en LM Studio. Si se omite, se intenta autodetectar.")
    parser.add_argument("--image", help="Ruta a la imagen a enviar al modelo. Si se omite, se usa la primera imagen disponible del proyecto.")
    parser.add_argument("--prompt", help="Prompt manual. Si no se indica y hay schema, se usa el prompt por defecto del proyecto.")
    parser.add_argument("--schema", help="Schema base registrado para probar response_format.")
    parser.add_argument("--structured", action="store_true", help="Usa GenericObjectDetection como schema base si no se especifica otro.")
    parser.add_argument("--with-reasoning", action="store_true", help="Usa la variante WithReasoning del schema indicado.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperatura de inferencia.")
    parser.add_argument("--server-api-host", help="Host del servidor LM Studio si no es el local por defecto.")
    parser.add_argument("--api-token", help="Token de API si aplica.")
    parser.add_argument("--print-json", action="store_true", help="Además del resumen visual, imprime el JSON completo en consola.")
    parser.add_argument(
        "--output",
        help="Ruta opcional de salida JSON. Si no se indica, se guarda en data/processed/debug/.",
    )
    return parser


def resolve_schema(schema_name: str | None, include_reasoning: bool) -> tuple[str | None, type[BaseModel] | None]:
    """
    Resuelve un schema opcional desde el registro del proyecto.

    Args:
        schema_name: Nombre del schema.
        include_reasoning: Si se debe incluir razonamiento.

    Returns:
        Nombre del schema y clase del schema.
    """
    if include_reasoning and not schema_name:
        schema_name = DEFAULT_STRUCTURED_SCHEMA

    if not schema_name:
        return None, None

    normalized_name = schema_name.strip()
    if normalized_name.endswith("WithReasoning"):
        normalized_name = normalized_name.removesuffix("WithReasoning")
        include_reasoning = True

    if normalized_name not in SCHEMA_REGISTRY:
        available = ", ".join(sorted(SCHEMA_REGISTRY.keys()))
        raise ValueError(f"Esquema desconocido '{schema_name}'. Disponibles: {available}")

    return get_schema_variant(normalized_name, include_reasoning)


def resolve_default_model(explicit_model: str | None) -> tuple[str, bool]:
    """
    Resuelve el modelo a usar, priorizando el valor explícito y luego la autodetección.

    Args:
        explicit_model: ID del modelo.

    Returns:
        ID del modelo y si fue autodetectado.
    """
    if explicit_model and explicit_model.strip():
        return explicit_model.strip(), False

    loaded_models = sorted(list_loaded_llm_model_keys())
    if loaded_models:
        return loaded_models[0], True

    installed_models = get_installed_lms_models() or get_installed_models()
    if installed_models:
        return installed_models[0], True

    raise RuntimeError("No se pudo autodetectar un modelo de LM Studio. Usa --model para indicarlo manualmente.")


def resolve_default_image(explicit_image: str | None) -> tuple[str, bool]:
    """
    Resuelve la imagen a inspeccionar, priorizando el valor explícito y luego el dataset del proyecto.

    Args:
        explicit_image: Ruta de la imagen.

    Returns:
        Ruta de la imagen y si fue autodetectada.
    """
    if explicit_image and explicit_image.strip():
        image_path = os.path.abspath(explicit_image.strip())
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Imagen no encontrada: {image_path}")
        return image_path, False

    images = find_images()
    if not images:
        raise RuntimeError("No se encontraron imágenes del proyecto. Usa --image para indicar una manualmente.")

    return os.path.abspath(images[0]), True


def deep_serialize(value: Any, *, max_depth: int = 8, _depth: int = 0, _seen: set[int] | None = None) -> Any:
    """
    Convierte objetos del SDK a estructuras JSON-friendly sin perder demasiada información.

    Args:
        value: Objeto a serializar.
        max_depth: Profundidad máxima de serialización.
        _depth: Profundidad actual.
        _seen: Conjunto de objetos ya serializados.

    Returns:
        Objeto serializado.
    """
    if _seen is None:
        _seen = set()

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if _depth >= max_depth:
        return repr(value)

    obj_id = id(value)
    if obj_id in _seen:
        return "<recursive-reference>"

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return deep_serialize(model_dump(), max_depth=max_depth, _depth=_depth + 1, _seen=_seen)
        except Exception:
            pass

    to_dict = getattr(value, "dict", None)
    if callable(to_dict):
        try:
            return deep_serialize(to_dict(), max_depth=max_depth, _depth=_depth + 1, _seen=_seen)
        except Exception:
            pass

    if isinstance(value, Mapping):
        _seen.add(obj_id)
        return {
            str(key): deep_serialize(item, max_depth=max_depth, _depth=_depth + 1, _seen=_seen)
            for key, item in value.items()
        }

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        _seen.add(obj_id)
        return [
            deep_serialize(item, max_depth=max_depth, _depth=_depth + 1, _seen=_seen)
            for item in value
        ]

    value_dict = getattr(value, "__dict__", None)
    if isinstance(value_dict, dict):
        _seen.add(obj_id)
        return {
            str(key): deep_serialize(item, max_depth=max_depth, _depth=_depth + 1, _seen=_seen)
            for key, item in value_dict.items()
            if not key.startswith("_")
        }

    return repr(value)


def safe_getattr(value: Any, attr_name: str) -> Any:
    """
    Lee un atributo sin propagar errores del SDK.

    Args:
        value: Objeto a inspeccionar.
        attr_name: Nombre del atributo.

    Returns:
        Valor del atributo.
    """
    try:
        return getattr(value, attr_name)
    except Exception as error:
        return f"<error leyendo atributo {attr_name}: {error}>"


def collect_public_attributes(value: Any) -> dict[str, Any]:
    """
    Recoge atributos públicos simples del objeto de respuesta.

    Args:
        value: Objeto a inspeccionar.

    Returns:
        Diccionario con atributos públicos.
    """
    attrs: dict[str, Any] = {}
    for attr_name in sorted(dir(value)):
        if attr_name.startswith("_"):
            continue
        attr_value = safe_getattr(value, attr_name)
        if callable(attr_value):
            continue
        attrs[attr_name] = deep_serialize(attr_value, max_depth=4)
    return attrs


def prune_absent_response_fields(value: Any) -> Any:
    """
    Elimina claves ausentes serializadas como null en payloads de inspeccion.

    Args:
        value: Valor a inspeccionar.

    Returns:
        Valor con claves ausentes eliminadas.
    """
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            pruned = prune_absent_response_fields(item)
            if pruned is None:
                continue
            if isinstance(pruned, Mapping) and not pruned:
                continue
            if isinstance(pruned, list) and not pruned:
                continue
            cleaned[str(key)] = pruned
        return cleaned
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = [prune_absent_response_fields(item) for item in value]
        return [item for item in items if item is not None]
    return value


def default_output_path(model_id: str) -> str:
    """
    Construye una ruta por defecto para guardar la inspección.

    Args:
        model_id: ID del modelo.

    Returns:
        Ruta por defecto.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = "".join(char if char.isalnum() or char in ("-", "_", "@") else "_" for char in model_id)
    output_dir = os.path.join(_PROJECT_ROOT, "data", "processed", "debug")
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"response_inspector_{safe_model}_{stamp}.json")


def save_inspection_payload(payload: dict[str, Any], output_path: str | None = None) -> str:
    """
    Guarda la inspección en disco y devuelve la ruta final usada.

    Args:
        payload: Payload de la inspección.
        output_path: Ruta de salida.

    Returns:
        Ruta final usada.
    """
    resolved_output_path = os.path.abspath(output_path) if output_path else default_output_path(str(payload["request"]["model_id"]))
    os.makedirs(os.path.dirname(resolved_output_path), exist_ok=True)
    with open(resolved_output_path, "w", encoding="utf-8") as file_handler:
        json.dump(payload, file_handler, indent=2, ensure_ascii=False)
    return resolved_output_path


def respond_without_schema(model_handle: Any, payload: Any, temperature: float) -> Any:
    """
    Lanza una petición cruda sin response_format para inspeccionar la respuesta real del SDK.

    Args:
        model_handle: Handle del modelo cargado.
        payload: Payload de la petición.
        temperature: Temperatura de la petición.

    Returns:
        Respuesta real del SDK.
    """
    kwargs: dict[str, Any] = {"config": {"temperature": temperature}}

    while True:
        try:
            return model_handle.respond(payload, **kwargs)
        except TypeError as error:
            error_text = str(error)
            if "unexpected keyword argument" in error_text and "config" in error_text and "config" in kwargs:
                kwargs.pop("config", None)
                kwargs["temperature"] = temperature
                continue
            if "unexpected keyword argument" in error_text and "temperature" in error_text and "temperature" in kwargs:
                kwargs.pop("temperature", None)
                continue
            raise


def run_inspection(args: argparse.Namespace) -> dict[str, Any]:
    """
    Ejecuta la petición real y construye un payload de inspección completo.

    Args:
        args: Argumentos de línea de comandos.

    Returns:
        Payload de inspección completo.
    """
    model_id, auto_model = resolve_default_model(args.model)
    image_path, auto_image = resolve_default_image(args.image)

    schema_request = args.schema
    if args.structured and not schema_request:
        schema_request = DEFAULT_STRUCTURED_SCHEMA

    schema_name, schema_cls = resolve_schema(schema_request, args.with_reasoning)
    prompt = (args.prompt or "").strip()
    if not prompt:
        if schema_name and schema_cls is not None:
            prompt = build_prompt_for_schema(schema_name, schema_cls)
        else:
            prompt = DEFAULT_RAW_PROMPT

    loader = VLMLoader(
        model_path=model_id,
        verbose=True,
        server_api_host=args.server_api_host,
        api_token=args.api_token,
    )

    response: Any = None
    structured_prompt = prompt
    try:
        loader.load_model()
        structured_prompt = loader._build_structured_instruction(prompt, schema=schema_cls) if schema_cls is not None else prompt
        multimodal_input = loader._build_multimodal_history(structured_prompt, image_path)
        model_handle = loader._loaded_model
        if model_handle is None:
            raise RuntimeError("El modelo no quedó cargado correctamente en LM Studio.")

        if schema_cls is not None:
            response = loader._respond_with_config_compat(
                model_handle,
                multimodal_input,
                float(args.temperature),
                schema=schema_cls,
            )
        else:
            response = respond_without_schema(
                model_handle,
                multimodal_input,
                float(args.temperature),
            )

        result_payload = loader._object_to_dict(getattr(response, "result", None))
        stats_payload = loader._object_to_dict(getattr(response, "stats", None))
        if not stats_payload and result_payload:
            stats_payload = loader._object_to_dict(result_payload.get("stats"))
        parsed_payload = safe_getattr(response, "parsed")
        model_info_payload = loader._object_to_dict(getattr(response, "model_info", None))
        prediction_config_payload = loader._object_to_dict(getattr(response, "prediction_config", None))
        response_payload = {
            "python_type": type(response).__name__,
            "repr": repr(response),
            "public_attributes": collect_public_attributes(response),
            "model_info": deep_serialize(model_info_payload),
            "prediction_config": deep_serialize(prediction_config_payload),
            "parsed": deep_serialize(parsed_payload),
            "stats": deep_serialize(stats_payload),
            "result": deep_serialize(result_payload),
            "structured": deep_serialize(safe_getattr(response, "structured")),
            "text_extracted": loader._extract_response_text(response),
            "object_dump": deep_serialize(response),
        }
        return {
            "request": {
                "model_id": model_id,
                "model_auto_selected": auto_model,
                "image_path": image_path,
                "image_auto_selected": auto_image,
                "schema_name": schema_name,
                "structured_requested": bool(schema_cls is not None),
                "with_reasoning": bool(args.with_reasoning),
                "temperature": float(args.temperature),
                "server_api_host": args.server_api_host,
                "used_prompt": prompt,
                "used_structured_prompt": structured_prompt,
                "multimodal_payload": deep_serialize(multimodal_input, max_depth=6),
            },
            "response": prune_absent_response_fields(response_payload),
        }
    finally:
        loader.unload_model()


def main() -> int:
    """Función de entrada para ejecución CLI."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        payload = run_inspection(args)
        output_path = save_inspection_payload(payload, args.output)

        from src.utils.tests_ui.cli_reporters import print_response_summary, print_section

        print_response_summary(payload)
        if args.print_json:
            print_section("FULL JSON")
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            print()
        print()
        print(f"Inspección guardada en: {output_path}")
        return 0
    except Exception as error:
        print(f"\n❌ Error durante la inspección: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())