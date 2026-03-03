"""Operaciones de modelos LM Studio con prioridad en SDK `lmstudio`."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Any, Callable

try:
    import lmstudio as lms
except ImportError:
    lms = None


# ==========================================================
# SDK availability / shared normalization helpers
# ==========================================================

def sdk_available() -> bool:
    """
    Indica si el SDK oficial de LM Studio (`lmstudio`) está disponible.
    
    Returns:
        bool: True si el módulo `lmstudio` se importó correctamente.
    """
    return lms is not None


def _normalize_model_ref(model_ref: str) -> str:
    """
    Normaliza referencias de modelos de Hugging Face para búsquedas.
    
    Elimina prefijos 'huggingface.co/' y extrae el identificador 'usuario/repo'.
    
    Args:
        model_ref (str): Referencia en crudo (URL, ID parcial, etc.).
        
    Returns:
        str: Identificador normalizado o el texto original limpio.
    """
    text = (model_ref or "").strip()
    if not text:
        return ""

    match = re.search(r"huggingface\.co/([^/]+/[^/?#]+)", text)
    if match:
        return match.group(1)

    return text.rstrip("/")


def _extract_quantization(name: str, fallback: str | None = None) -> str | None:
    """
    Extrae la etiqueta de cuantización desde un nombre de archivo o modelo.
    
    Busca patrones como Q4_K_M, IQ4_M, etc.
    
    Args:
        name (str): Cadena donde buscar el patrón.
        fallback (str, optional): Valor a devolver si no se encuentra nada.
        
    Returns:
        str | None: La cuantización en mayúsculas o el valor de fallback.
    """
    text = (name or "").strip()
    if not text:
        return fallback
    match = re.search(r"(Q\d(?:_K)?(?:_[MSL])?|IQ\d(?:_[MSL])?)", text, re.IGNORECASE)
    if not match:
        return fallback
    return match.group(1).upper()


# ==========================================================
# CLI (`lms`) fallback helpers
# ==========================================================

def _run_lms_command(args: list[str], check: bool = False) -> subprocess.CompletedProcess[str]:
    """
    Ejecuta un comando del CLI `lms`.
    
    Args:
        args (list[str]): Argumentos del comando.
        check (bool, optional): Si True, lanza excepción en caso de error.
        
    Returns:
        subprocess.CompletedProcess: Resultado de la ejecución con stdout/stderr capturados.
    """
    return subprocess.run(
        args,
        text=True,
        capture_output=True,
        check=check,
    )


def check_lms() -> bool:
    """
    Verifica si la herramienta de línea de comandos `lms` está instalada y accesible.
    
    Intenta ejecutar `lms version`.
    
    Returns:
        bool: True si `lms` responde correctamente.
    """
    try:
        _run_lms_command(["lms", "version"], check=True)
        return True
    except Exception:
        return False


def parse_lms_table_models(raw_output: str) -> list[str]:
    """
    Parsea la salida tabular de comandos `lms ls` o `lms ps`.
    
    Args:
        raw_output (str): Salida de texto cruda del comando CLI.
        
    Returns:
        list[str]: Lista de identificadores o nombres de modelos extraídos.
    """
    lines = [line.strip() for line in str(raw_output).splitlines() if line.strip()]
    models: list[str] = []

    for line in lines:
        upper = line.upper()
        if "MODEL" in upper and "SIZE" in upper:
            continue
        if "MODEL" in upper and "STATUS" in upper:
            continue

        parts = line.split()
        if not parts:
            continue

        candidate = parts[0].strip()
        if candidate.lower() in ("id", "model", "name"):
            continue
        models.append(candidate)

    return models


def get_installed_models() -> list[str]:
    """
    Obtiene la lista de modelos descargados usando el CLI `lms ls`.
    
    Returns:
        list[str]: Lista de identificadores de modelos instalados.
        Devuelve lista vacía si `lms` no está disponible o falla.
    """
    if not check_lms():
        return []
    try:
        result = _run_lms_command(["lms", "ls"], check=True)
        return parse_lms_table_models(result.stdout)
    except Exception:
        return []


def get_loaded_models() -> list[str]:
    """
    Obtiene la lista de modelos actualmente cargados en memoria vía `lms ps`.
    
    Returns:
        list[str]: Lista de identificadores de modelos cargados.
    """
    if not check_lms():
        return []
    try:
        result = _run_lms_command(["lms", "ps"], check=True)
        return parse_lms_table_models(result.stdout)
    except Exception:
        return []


def get_server_status() -> tuple[bool, str]:
    """
    Consulta el estado del servidor LM Studio (`lms server status`).
    
    Returns:
        tuple[bool, str]: (Está corriendo?, Detalle/Mensaje de salida).
    """
    if not check_lms():
        return False, "lms CLI not found"

    try:
        result = _run_lms_command(["lms", "server", "status"], check=False)
        output = (result.stdout or result.stderr or "").strip()
    except Exception as error:
        return False, str(error)

    normalized = output.lower()
    running_markers = ("running", "online", "listening", "ready")
    is_running = any(marker in normalized for marker in running_markers)
    return is_running, output


def start_server() -> bool:
    """
    Inicia el servidor local de LM Studio (`lms server start`).
    
    Returns:
        bool: True si el comando de inicio fue exitoso.
    """
    if not check_lms():
        return False
    try:
        _run_lms_command(["lms", "server", "start"], check=True)
        return True
    except Exception:
        return False


def stop_server() -> bool:
    """
    Detiene el servidor local de LM Studio (`lms server stop`).
    
    Returns:
        bool: True si el comando de parada fue exitoso.
    """
    if not check_lms():
        return False
    try:
        _run_lms_command(["lms", "server", "stop"], check=True)
        return True
    except Exception:
        return False


def _cli_get_model(model_tag: str) -> bool:
    """
    Descarga un modelo usando el CLI (`lms get`) como mecanismo de respaldo.
    
    Args:
        model_tag (str): Tag o ID del modelo a descargar.
        
    Returns:
        bool: True si la descarga fue exitosa.
    """
    if not check_lms():
        return False
    try:
        _run_lms_command(["lms", "get", model_tag], check=True)
        return True
    except Exception:
        return False


def _cli_load_model(model_tag: str) -> bool:
    """
    Carga un modelo usando el CLI (`lms load`) como respaldo.
    
    Args:
        model_tag (str): Tag del modelo a cargar.
        
    Returns:
        bool: True si la carga fue exitosa.
    """
    if not check_lms():
        return False
    try:
        _run_lms_command(["lms", "load", model_tag], check=True)
        return True
    except Exception:
        return False


def _cli_unload_model(model_tag: str) -> bool:
    """
    Descarga un modelo de la memoria usando el CLI (`lms unload`).
    
    Args:
        model_tag (str): Tag del modelo a descargar de RAM.
        
    Returns:
        bool: True si la operación fue exitosa.
    """
    if not check_lms():
        return False
    try:
        _run_lms_command(["lms", "unload", model_tag], check=True)
        return True
    except Exception:
        return False


# ==========================================================
# SDK-first model operations (with CLI fallback when needed)
# ==========================================================

def list_local_llm_models() -> list[dict[str, Any]]:
    """
    Lista los modelos LLM descargados en local.
    
    Intenta usar el SDK `lmstudio` para obtener metadatos detallados.
    
    Returns:
        list[dict]: Lista de diccionarios con claves 'model_key', 'display_name', 'path', 'size_bytes'.
    """
    if lms is None:
        return []

    try:
        downloaded = lms.list_downloaded_models("llm")
    except Exception:
        return []

    models: list[dict[str, Any]] = []
    for item in downloaded:
        model_key = getattr(item, "model_key", None)
        if not isinstance(model_key, str) or not model_key.strip():
            continue

        models.append(
            {
                "model_key": model_key.strip(),
                "display_name": getattr(item, "display_name", model_key),
                "path": getattr(item, "path", ""),
                "size_bytes": getattr(item, "size_bytes", None),
            }
        )

    models.sort(key=lambda m: str(m["model_key"]).lower())
    return models


def list_loaded_llm_model_keys() -> set[str]:
    """
    Recupera las claves de los modelos LLM cargados en memoria usando el SDK.
    
    Returns:
        set[str]: Conjunto de claves de modelos cargados.
    """
    if lms is None:
        return set()

    try:
        loaded = lms.list_loaded_models("llm")
    except Exception:
        return set()

    loaded_keys: set[str] = set()
    for handle in loaded:
        try:
            info = handle.get_info()
            model_key = getattr(info, "model_key", None)
            if isinstance(model_key, str) and model_key.strip():
                loaded_keys.add(model_key.strip())
        except Exception:
            continue

    return loaded_keys


def load_model(model_key: str) -> bool:
    """
    Carga un modelo en memoria.
    
    Prioriza el SDK `lms.llm()`, con fallback al CLI `lms load`.
    
    Args:
        model_key (str): Clave del modelo a cargar.
        
    Returns:
        bool: True si tuvo éxito.
    """
    if lms is None:
        return _cli_load_model(model_key)
    try:
        lms.llm(model_key)
        return True
    except Exception:
        return _cli_load_model(model_key)


def unload_model(model_key: str) -> bool:
    """
    Descarga un modelo de la memoria.
    
    Prioriza el SDK, intentando encontrar el handle del modelo y llamar `unload()`.
    
    Args:
        model_key (str): Clave del modelo a descargar.
        
    Returns:
        bool: True si tuvo éxito.
    """
    if lms is None:
        return _cli_unload_model(model_key)

    unloaded_any = False
    try:
        loaded = lms.list_loaded_models("llm")
    except Exception:
        loaded = []

    for handle in loaded:
        try:
            info = handle.get_info()
            loaded_key = getattr(info, "model_key", None)
            if isinstance(loaded_key, str) and loaded_key.strip() == model_key:
                handle.unload()
                unloaded_any = True
        except Exception:
            continue

    if unloaded_any:
        return True

    try:
        lms.get_default_client().llm.unload(model_key)
        return True
    except Exception:
        return _cli_unload_model(model_key)


def preload_model(model_key: str) -> bool:
    """
    Realiza un calentamiento (warmup) del modelo cargado mediante una inferencia trivial.
    
    Args:
        model_key (str): Clave del modelo.
        
    Returns:
        bool: True si el modelo respondió correctamente.
    """
    if lms is None:
        return False
    try:
        handle = lms.llm(model_key)
        handle.respond("Warmup mínimo. Responde solo: ok.")
        return True
    except Exception:
        return False


# ==========================================================
# Repository lookup / quantization options
# ==========================================================

def _select_search_result(model_ref: str, candidates: list[Any]) -> Any | None:
    """
    Selecciona el mejor resultado de búsqueda dado un criterio.
    
    Busca coincidencia exacta de nombre, o contención de cadena.
    
    Args:
        model_ref (str): Referencia buscada.
        candidates (list[Any]): Resultados de búsqueda del SDK.
        
    Returns:
        Any | None: El objeto resultado seleccionado o None.
    """
    normalized_ref = _normalize_model_ref(model_ref).lower()

    if not candidates:
        return None

    exact = []
    contains = []
    for entry in candidates:
        try:
            name = str(entry.search_result.name).strip().lower()
        except Exception:
            name = ""

        if name == normalized_ref:
            exact.append(entry)
        elif normalized_ref and normalized_ref in name:
            contains.append(entry)

    if exact:
        return exact[0]
    if contains:
        return contains[0]
    return candidates[0]


def get_download_options(model_ref: str) -> list[dict[str, Any]]:
    """
    Busca opciones de descarga (cuantizaciones) en el repositorio de LM Studio.
    
    Args:
        model_ref (str): Nombre o referencia del modelo.
        
    Returns:
        list[dict]: Lista de opciones de descarga con metadatos.
    """
    if lms is None:
        return []

    normalized_ref = _normalize_model_ref(model_ref)
    if not normalized_ref:
        return []

    client = lms.get_default_client()

    search_terms = [normalized_ref]
    if "/" in normalized_ref:
        search_terms.append(normalized_ref.split("/")[-1])
    if normalized_ref.lower().endswith("-gguf"):
        search_terms.append(normalized_ref[:-5])

    search_results: list[Any] = []
    for term in search_terms:
        try:
            current_results = list(client.repository.search_models(search_term=term, limit=25))
            if current_results:
                search_results = current_results
                break
        except Exception:
            continue

    if not search_results:
        return []

    selected = _select_search_result(normalized_ref, search_results)
    if selected is None:
        return []

    try:
        options = list(selected.get_download_options())
    except Exception:
        return []

    parsed: list[dict[str, Any]] = []
    for option in options:
        try:
            info = option.info
            parsed.append(
                {
                    "name": getattr(info, "name", "unknown"),
                    "quantization": _extract_quantization(
                        str(getattr(info, "name", "")),
                        fallback=getattr(info, "quantization", None),
                    ),
                    "recommended": bool(getattr(info, "recommended", False)),
                    "size_bytes": getattr(info, "size_bytes", None),
                    "indexed_model_identifier": getattr(info, "indexed_model_identifier", ""),
                    "option_obj": option,
                }
            )
        except Exception:
            continue

    parsed.sort(
        key=lambda item: (
            0 if item.get("recommended") else 1,
            str(item.get("quantization") or "zzz").lower(),
            str(item.get("name") or "").lower(),
        )
    )
    return parsed


# ==========================================================
# Download / local artifact management
# ==========================================================

def download_option(
    option_entry: dict[str, Any],
    on_progress: Callable[[Any], None] | None = None,
    on_finalize: Callable[[Any], None] | None = None,
) -> tuple[bool, str]:
    """
    Descarga una variante de modelo específica mediante el SDK.
    
    Args:
        option_entry (dict): Entrada de opción obtenida de `get_download_options`.
        on_progress (Callable, optional): Callback de progreso (float 0.0-1.0).
        on_finalize (Callable, optional): Callback al finalizar.
        
    Returns:
        tuple[bool, str]: (Éxito, Key del modelo descargado o mensaje de error).
    """
    if lms is None:
        return False, "SDK lmstudio no disponible"

    option_obj = option_entry.get("option_obj")
    if option_obj is None:
        return False, "Opción de descarga inválida"

    try:
        downloaded_path = str(option_obj.download(on_progress=on_progress, on_finalize=on_finalize))
    except Exception as error:
        return False, str(error)

    local_models = list_local_llm_models()
    for model in local_models:
        if str(model.get("path", "")).strip() == downloaded_path:
            return True, str(model.get("model_key", ""))

    indexed = str(option_entry.get("indexed_model_identifier", "")).strip()
    if indexed:
        return True, indexed

    return True, downloaded_path


def download_by_model_and_filename(model_ref: str, file_name: str) -> tuple[bool, str]:
    """
    Descarga un archivo específico de un modelo buscando por nombre de archivo.
    
    Args:
        model_ref (str): Referencia del modelo.
        file_name (str): Nombre del archivo (ej. 'modelo-q4_k_m.gguf').
        
    Returns:
        tuple[bool, str]: (Éxito, Key del modelo o error).
    """
    file_name_clean = (file_name or "").strip().lower()
    if not file_name_clean:
        return False, "file_name vacío"

    options = get_download_options(model_ref)
    for option in options:
        option_name = str(option.get("name") or "").strip().lower()
        indexed_identifier = str(option.get("indexed_model_identifier") or "").strip().lower()
        if file_name_clean == option_name or file_name_clean in indexed_identifier:
            return download_option(option)

    return False, f"No se encontró opción de descarga para '{file_name}'"


def download_model(model_ref: str) -> tuple[bool, str]:
    """
    Descarga un modelo automáticamente (opción recomendada).
    
    Args:
        model_ref (str): Referencia del modelo.
        
    Returns:
        tuple[bool, str]: (Éxito, Key del modelo o error).
    """
    normalized_ref = _normalize_model_ref(model_ref)
    if not normalized_ref:
        return False, "Referencia de modelo inválida"

    options = get_download_options(normalized_ref)
    if options:
        selected = next((opt for opt in options if opt.get("recommended")), options[0])
        return download_option(selected)

    if _cli_get_model(normalized_ref):
        return True, normalized_ref

    return False, f"No se pudo descargar el modelo: {normalized_ref}"


def remove_local_model(model_key: str) -> tuple[bool, str]:
    """
    Elimina físicamente un modelo local y sus archivos.
    
    Args:
        model_key (str): Clave del modelo a borrar.
        
    Returns:
        tuple[bool, str]: (Éxito, Ruta eliminada o error).
    """
    local_models = list_local_llm_models()
    target = next((item for item in local_models if item.get("model_key") == model_key), None)
    if target is None:
        return False, "Modelo no encontrado en lista local"

    _ = unload_model(model_key)

    target_path = str(target.get("path") or "").strip()
    if not target_path:
        return False, "No se encontró ruta local para el modelo"

    if not os.path.exists(target_path):
        return False, f"La ruta del modelo no existe: {target_path}"

    try:
        if os.path.isdir(target_path):
            shutil.rmtree(target_path)
        else:
            os.remove(target_path)
        return True, target_path
    except Exception as error:
        return False, str(error)


def get_installed_lms_models() -> list[str]:
    """Devuelve la lista de modelos LM Studio instalados localmente.

    Intenta primero el SDK local (``list_local_llm_models``) y cae al CLI
    (``get_installed_models``) como fallback.

    Returns:
        list[str]: Identificadores de los modelos instalados.
    """
    local_models = list_local_llm_models()
    if local_models:
        return [
            str(item.get("model_key", "")).strip()
            for item in local_models
            if item.get("model_key")
        ]
    return get_installed_models()
