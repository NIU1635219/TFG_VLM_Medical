"""Helpers puros para menús LM Studio en setup_env.

Este módulo concentra lógica reutilizable de:
- Normalización/igualdad de tags de modelo.
- Resolución de metadata desde registry.
- Cálculo de capacidades (GPU/RAM) para cuantizaciones.
- Orden y detección de opciones locales.
"""

from __future__ import annotations

import re
import subprocess
from typing import Any


def normalize_model_tag(model_tag: Any) -> str:
    """Normaliza un tag de modelo (lower + trim)."""
    if not model_tag:
        return ""
    return str(model_tag).strip().lower()


def split_model_tag(model_tag: Any) -> tuple[str, str | None]:
    """Divide `modelo:tag` en `(modelo, tag|None)` con normalización."""
    normalized = normalize_model_tag(model_tag)
    if not normalized:
        return "", None

    if ":" not in normalized:
        return normalized, None

    model_name, tag = normalized.rsplit(":", 1)
    return model_name, tag


def is_latest_alias_equivalent(model_a: Any, model_b: Any) -> bool:
    """Equivalencia limitada: `model` <-> `model:latest`."""
    name_a, tag_a = split_model_tag(model_a)
    name_b, tag_b = split_model_tag(model_b)

    if not name_a or not name_b or name_a != name_b:
        return False

    if tag_a == tag_b:
        return True

    a_latest_or_none = tag_a in (None, "latest")
    b_latest_or_none = tag_b in (None, "latest")
    return a_latest_or_none and b_latest_or_none


def contains_equivalent_model(model_tag: Any, model_list: list[Any]) -> bool:
    """Comprueba si un modelo equivalente ya existe en una lista de modelos."""
    for installed in model_list:
        if is_latest_alias_equivalent(model_tag, installed):
            return True
    return False


def is_model_installed(model_tag: Any, installed_models: list[Any]) -> bool:
    """Comprueba instalación sin mezclar tags distintos (8b != latest)."""
    if not model_tag:
        return False

    target = normalize_model_tag(model_tag)
    installed_normalized = [normalize_model_tag(m) for m in installed_models if m]

    if target in installed_normalized:
        return True

    return contains_equivalent_model(target, installed_normalized)


def resolve_model_tag(model_ref: Any, models_registry: dict[str, dict[str, Any]]) -> str:
    """Resuelve key de registry o devuelve referencia de modelo directa."""
    if model_ref in models_registry:
        return str(models_registry[model_ref].get("name", "")).strip()
    return str(model_ref or "").strip()


def resolve_model_description(model_ref: Any, models_registry: dict[str, dict[str, Any]]) -> str:
    """Resuelve descripción legible para logs de acciones de modelo."""
    if model_ref in models_registry:
        return str(models_registry[model_ref].get("description", "Registry model"))
    return "Custom/Detected model"


def bytes_to_gb(value_bytes: Any) -> float:
    """Convierte bytes a GB con protección de tipos."""
    if not isinstance(value_bytes, (int, float)) or value_bytes <= 0:
        return 0.0
    return float(value_bytes) / (1024**3)


def detect_gpu_memory_bytes() -> int:
    """Intenta detectar memoria total de GPU NVIDIA en bytes (0 si no disponible)."""
    try:
        raw = subprocess.check_output(
            "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits",
            shell=True,
            stderr=subprocess.DEVNULL,
        ).decode("utf-8", errors="ignore")
        first_line = next((line.strip() for line in raw.splitlines() if line.strip()), "")
        mib = float(first_line)
        return int(mib * 1024 * 1024)
    except Exception:
        return 0


def detect_ram_memory_bytes(psutil_module: Any) -> int:
    """Devuelve RAM total en bytes (0 si psutil no está disponible)."""
    if psutil_module is None:
        return 0
    try:
        return int(psutil_module.virtual_memory().total)
    except Exception:
        return 0


def format_size(size_bytes: Any) -> str:
    """Formatea bytes a unidad legible (B/KB/MB/GB/TB)."""
    if not isinstance(size_bytes, int) or size_bytes <= 0:
        return "?"

    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    idx = 0
    while size >= 1024.0 and idx < len(units) - 1:
        size /= 1024.0
        idx += 1
    return f"{size:.1f} {units[idx]}"


def model_hint_from_ref(model_ref: Any) -> str:
    """Genera un hint de modelo para matching parcial de cuantizaciones."""
    model_hint = str(model_ref or "").strip().lower()
    if "/" in model_hint:
        model_hint = model_hint.split("/")[-1]
    return model_hint.replace("-gguf", "")


def option_is_local(entry: dict[str, Any], normalized_local: set[str], model_hint: str = "") -> bool:
    """Decide si una opción está en local con matching estricto (sin falsos positivos)."""
    indexed = str(entry.get("indexed_model_identifier") or "").strip().lower()
    indexed_file = indexed.rsplit("/", 1)[-1] if indexed else ""

    if indexed and indexed in normalized_local:
        return True

    for local_key in normalized_local:
        if not local_key:
            continue

        normalized_local_key = local_key.replace("\\", "/")
        local_file = normalized_local_key.rsplit("/", 1)[-1]

        if indexed_file and indexed_file == local_file:
            return True

    return False


def capacity_status(size_bytes: Any, gpu_mem_bytes: int, ram_mem_bytes: int) -> str:
    """Clasifica capacidad como gpu, hybrid, no_fit o unknown."""
    if not isinstance(size_bytes, int) or size_bytes <= 0:
        return "unknown"

    if size_bytes <= gpu_mem_bytes and gpu_mem_bytes > 0:
        return "gpu"
    if size_bytes <= (gpu_mem_bytes + ram_mem_bytes) and (gpu_mem_bytes + ram_mem_bytes) > 0:
        return "hybrid"
    return "no_fit"


def sort_download_options(options: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ordena cuantizaciones de menor a mayor tamaño; desempata por nombre."""
    return sorted(
        options,
        key=lambda item: (
            item.get("size_bytes") if isinstance(item.get("size_bytes"), int) else 10**20,
            str(item.get("quantization") or "zzz").lower(),
        ),
    )


def extract_quantization_from_text(value: Any) -> str:
    """Extrae cuantización desde un texto libre y devuelve etiqueta normalizada."""
    text = str(value or "").strip()
    if not text:
        return "unknown"

    match = re.search(r"(Q\d(?:_K)?(?:_[MSL])?|IQ\d(?:_[MSL])?)", text, re.IGNORECASE)
    if not match:
        return "unknown"

    return match.group(1).upper()


def detect_local_model_quantization(model_entry: dict[str, Any]) -> str:
    """Detecta cuantización de un modelo local usando model_key, display_name y path."""
    candidates = [
        model_entry.get("model_key"),
        model_entry.get("display_name"),
        model_entry.get("path"),
    ]

    for candidate in candidates:
        quant = extract_quantization_from_text(candidate)
        if quant != "unknown":
            return quant

    return "unknown"


def build_local_signatures(local_models: list[dict[str, Any]]) -> set[str]:
    """Construye firmas normalizadas de modelos locales para matching estricto."""
    signatures: set[str] = set()
    for model in local_models:
        for field in ("model_key", "display_name", "path"):
            value = str(model.get(field) or "").strip().lower()
            if not value:
                continue
            signatures.add(value)
            signatures.add(value.replace("\\", "/"))
    return signatures


def dedupe_options_by_quantization(options: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Elimina duplicados de descarga manteniendo una opción por cuantización."""
    unique_by_quant: dict[str, dict[str, Any]] = {}
    for entry in options:
        quant = str(entry.get("quantization") or "unknown").upper()
        if quant not in unique_by_quant:
            unique_by_quant[quant] = entry
    return list(unique_by_quant.values())


def option_visual_state(
    entry: dict[str, Any],
    local_signatures: set[str],
    model_ref: str,
    gpu_mem_bytes: int,
    ram_mem_bytes: int,
) -> tuple[str, str, bool]:
    """Devuelve estado visual `(label, color, blocked)` para una cuantización."""
    model_hint = model_hint_from_ref(model_ref)
    is_local = option_is_local(entry, local_signatures, model_hint)
    cap_status = capacity_status(entry.get("size_bytes"), gpu_mem_bytes, ram_mem_bytes)

    if is_local:
        return "LOCAL", "OKBLUE", True
    if cap_status == "gpu":
        return "INSTALL", "OKGREEN", False
    if cap_status == "hybrid":
        return "INSTALL+RAM", "WARNING", False
    if cap_status == "no_fit":
        return "NO-FIT", "FAIL", True
    return "INSTALL?", "WARNING", False


def shift_horizontal_index(current_idx: int, direction: str, total: int) -> int:
    """Desplaza índice horizontal en bucle para navegación LEFT/RIGHT."""
    if total <= 0:
        return 0
    step = -1 if direction == "LEFT" else 1
    return (current_idx + step) % total
