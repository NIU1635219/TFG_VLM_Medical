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
    """
    Normaliza un tag de modelo a minúsculas y sin espacios.
    
    Args:
        model_tag (Any): El tag del modelo (str o similar).
        
    Returns:
        str: Tag normalizado o cadena vacía si la entrada es nula/vacía.
    """
    if not model_tag:
        return ""
    return str(model_tag).strip().lower()


def split_model_tag(model_tag: Any) -> tuple[str, str | None]:
    """
    Divide un identificador `modelo:tag` en componentes.
    
    Args:
        model_tag (Any): El identificador completo (ej. 'modelo:v1').
        
    Returns:
        tuple[str, str | None]: Tupla (nombre_modelo, tag). Tag es None si no existe.
    """
    normalized = normalize_model_tag(model_tag)
    if not normalized:
        return "", None

    if ":" not in normalized:
        return normalized, None

    model_name, tag = normalized.rsplit(":", 1)
    return model_name, tag


def is_latest_alias_equivalent(model_a: Any, model_b: Any) -> bool:
    """
    Comprueba si dos modelos son equivalentes considerando el tag 'latest'.
    
    Trata los tags 'latest' y None como equivalentes entre sí.
    
    Args:
        model_a (Any): Primer modelo.
        model_b (Any): Segundo modelo.
        
    Returns:
        bool: True si representan el mismo modelo base y versión compatible.
    """
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
    """
    Verifica si existe un modelo equivalente en una lista.
    
    Args:
        model_tag (Any): Tag del modelo a buscar.
        model_list (list[Any]): Lista de modelos disponibles.
        
    Returns:
        bool: True si se encuentra una coincidencia equivalente.
    """
    for installed in model_list:
        if is_latest_alias_equivalent(model_tag, installed):
            return True
    return False


def is_model_installed(model_tag: Any, installed_models: list[Any]) -> bool:
    """
    Comprueba si un modelo está instalado, manejando normalización.
    
    Distingue entre versiones específicas (ej. 8b vs latest) pero maneja equivalencias básicas.
    
    Args:
        model_tag (Any): Tag del modelo buscado.
        installed_models (list[Any]): Lista de modelos instalados.
        
    Returns:
        bool: True si el modelo se considera instalado.
    """
    if not model_tag:
        return False

    target = normalize_model_tag(model_tag)
    installed_normalized = [normalize_model_tag(m) for m in installed_models if m]

    if target in installed_normalized:
        return True

    return contains_equivalent_model(target, installed_normalized)


def resolve_model_tag(model_ref: Any, models_registry: dict[str, dict[str, Any]]) -> str:
    """
    Resuelve el nombre legible de un modelo desde un registro.
    
    Args:
        model_ref (Any): Clave o referencia del modelo.
        models_registry (dict): Registro de metadatos de modelos.
        
    Returns:
        str: Nombre del modelo resuelto o la referencia original.
    """
    if model_ref in models_registry:
        return str(models_registry[model_ref].get("name", "")).strip()
    return str(model_ref or "").strip()


def resolve_model_description(model_ref: Any, models_registry: dict[str, dict[str, Any]]) -> str:
    """
    Obtiene la descripción de un modelo desde el registro.
    
    Args:
        model_ref (Any): Referencia del modelo.
        models_registry (dict): Registro de modelos.
        
    Returns:
        str: Descripción del modelo o 'Custom/Detected model' si no está en registro.
    """
    if model_ref in models_registry:
        return str(models_registry[model_ref].get("description", "Registry model"))
    return "Custom/Detected model"


def bytes_to_gb(value_bytes: Any) -> float:
    """
    Convierte un valor de bytes a Gigabytes (GB).
    
    Args:
        value_bytes (Any): Valor numérico en bytes.
        
    Returns:
        float: Valor en GB. Retorna 0.0 si la entrada no es válida.
    """
    if not isinstance(value_bytes, (int, float)) or value_bytes <= 0:
        return 0.0
    return float(value_bytes) / (1024**3)


def detect_gpu_memory_bytes() -> int:
    """
    Detecta la VRAM total disponible en GPU NVIDIA.
    
    Usa `nvidia-smi` para consultar la memoria.
    
    Returns:
        int: Memoria total en bytes, o 0 si falla o no hay GPU NVIDIA.
    """
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
    """
    Obtiene la memoria RAM total del sistema.
    
    Args:
        psutil_module (Any): Módulo `psutil` inyectado.
        
    Returns:
        int: Memoria RAM total en bytes, o 0 si psutil no está disponible.
    """
    if psutil_module is None:
        return 0
    try:
        return int(psutil_module.virtual_memory().total)
    except Exception:
        return 0


def format_size(size_bytes: Any) -> str:
    """
    Formatea un valor en bytes a una cadena legible con unidades (B, KB, MB, GB).
    
    Args:
        size_bytes (Any): El tamaño en bytes.
        
    Returns:
        str: Cadena formateada (ej. '1.5 GB'). Retorna '?' si el valor no es válido.
    """
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
    """
    Genera una pista de modelo simplificada para comparaciones.
    
    Elimina rutas y sufijos comunes como '-gguf' para facilitar el matching.
    
    Args:
        model_ref (Any): Referencia o nombre de modelo.
        
    Returns:
        str: String simplificado para búsqueda/comparación.
    """
    model_hint = str(model_ref or "").strip().lower()
    if "/" in model_hint:
        model_hint = model_hint.split("/")[-1]
    return model_hint.replace("-gguf", "")


def option_is_local(entry: dict[str, Any], normalized_local: set[str], model_hint: str = "") -> bool:
    """
    Determina si una opción de descarga corresponde a un archivo ya existente localmente.
    
    Realiza una comprobación estricta contra un conjunto de firmas de archivos locales.
    
    Args:
        entry (dict): Entrada de opción de descarga.
        normalized_local (set[str]): Conjunto de firmas de archivos locales normalizadas.
        model_hint (str, optional): Pista de modelo para ayudar en la desambiguación.
        
    Returns:
        bool: True si el archivo parece estar localmente presente.
    """
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
    """
    Evalúa si un modelo cabe en memoria (GPU o RAM).
    
    Args:
        size_bytes (Any): Tamaño del modelo en bytes.
        gpu_mem_bytes (int): Memoria GPU disponible en bytes.
        ram_mem_bytes (int): Memoria RAM disponible en bytes.
        
    Returns:
        str: 'gpu' (cabe en VRAM), 'hybrid' (cabe en RAM+VRAM), 'no_fit' (no cabe), 'unknown'.
    """
    if not isinstance(size_bytes, int) or size_bytes <= 0:
        return "unknown"

    if size_bytes <= gpu_mem_bytes and gpu_mem_bytes > 0:
        return "gpu"
    if size_bytes <= (gpu_mem_bytes + ram_mem_bytes) and (gpu_mem_bytes + ram_mem_bytes) > 0:
        return "hybrid"
    return "no_fit"


def sort_download_options(options: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Ordena opciones de descarga por tamaño (ascendente) y nombre.
    
    Args:
        options (list[dict]): Lista de opciones de descarga.
        
    Returns:
        list[dict]: Lista ordenada.
    """
    return sorted(
        options,
        key=lambda item: (
            item.get("size_bytes") if isinstance(item.get("size_bytes"), int) else 10**20,
            str(item.get("quantization") or "zzz").lower(),
        ),
    )


def extract_quantization_from_text(value: Any) -> str:
    """
    Intenta extraer el nivel de cuantización (ej. Q4_K_M) de un texto arbitrario.
    
    Args:
        value (Any): Texto donde buscar (nombre de archivo, tag, etc.).
        
    Returns:
        str: Nivel de cuantización en mayúsculas (ej. 'Q4_K_M') o 'unknown'.
    """
    text = str(value or "").strip()
    if not text:
        return "unknown"

    match = re.search(r"(Q\d(?:_K)?(?:_[MSL])?|IQ\d(?:_[MSL])?)", text, re.IGNORECASE)
    if not match:
        return "unknown"

    return match.group(1).upper()


def detect_local_model_quantization(model_entry: dict[str, Any]) -> str:
    """
    Infiere la cuantización de una entrada de modelo local.
    
    Busca patrones de cuantización en model_key, display_name y path.
    
    Args:
        model_entry (dict): Entrada de modelo local.
        
    Returns:
        str: Cuantización detectada o 'unknown'.
    """
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
    """
    Genera un conjunto de firmas únicas para identificar modelos locales.
    
    Combina claves de modelo, nombres y rutas normalizadas.
    
    Args:
        local_models (list[dict]): Lista de modelos locales.
        
    Returns:
        set[str]: Conjunto de strings identificadores únicos.
    """
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
    """
    Filtra opciones de descarga duplicadas, conservando una por nivel de cuantización.
    
    Args:
        options (list[dict]): Lista de opciones de descarga.
        
    Returns:
        list[dict]: Lista filtrada con opciones únicas por cuantización.
    """
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
    """
    Calcula el estado visual (etiqueta, color, bloqueo) de una opción de menú.
    
    Args:
        entry (dict): Entrada de opción (modelo).
        local_signatures (set[str]): Firmas de modelos locales para detección.
        model_ref (str): Referencia del modelo base.
        gpu_mem_bytes (int): Memoria VRAM disponible.
        ram_mem_bytes (int): Memoria RAM disponible.
        
    Returns:
        tuple[str, str, bool]: (Etiqueta corta, Estilo de color, Bloqueo de opción).
    """
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
    """
    Calcula el próximo índice en una navegación horizontal cíclica.
    
    Args:
        current_idx (int): Índice actual.
        direction (str): Dirección ('LEFT' o 'RIGHT').
        total (int): Total de elementos.
        
    Returns:
        int: Nuevo índice.
    """
    if total <= 0:
        return 0
    step = -1 if direction == "LEFT" else 1
    return (current_idx + step) % total
