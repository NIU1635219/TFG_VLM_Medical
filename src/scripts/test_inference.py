"""Smoke test VLM con LM Studio usando SDK oficial lmstudio."""

import argparse
import json
import os
import re
import sys
import time
import unicodedata
import urllib.request
import urllib.error
from typing import Dict, List, Tuple

try:
    import lmstudio as lms
except ImportError:
    lms = None

try:
    from PIL import Image
except ImportError:
    Image = None

# Add project root to sys.path to allow importing src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.inference.vlm_runner import VLMLoader, GenericObjectDetection
from src.utils.lms_models import (
    get_installed_lms_models as _lms_get_installed_models,
    get_installed_models as _lms_cli_installed_models,
    list_loaded_llm_model_keys as _lms_loaded_model_keys,
    parse_lms_table_models as parse_lms_ls_output,
)

PROMPT = (
    "Identifica el animal principal de la imagen y responde en español. "
    "Incluye primero la clase del animal y luego una breve descripción."
)
TEST_IMAGES_DIR = "data/smoke_test"

TEST_CASES = [
    {
        "id": "sample_01",
        "label": "cat",
        "path": os.path.join(TEST_IMAGES_DIR, "sample_01.jpg"),
        "urls": [
            "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg",
            "https://cataas.com/cat?type=square",
            "https://placekitten.com/640/480",
        ],
    },
    {
        "id": "sample_02",
        "label": "cat",
        "path": os.path.join(TEST_IMAGES_DIR, "sample_02.jpg"),
        "urls": [
            "https://cataas.com/cat?width=640&height=480",
            "https://placekitten.com/600/400",
            "https://cataas.com/cat?type=small",
        ],
    },
    {
        "id": "sample_03",
        "label": "dog",
        "path": os.path.join(TEST_IMAGES_DIR, "sample_03.jpg"),
        "urls": [
            "https://upload.wikimedia.org/wikipedia/commons/6/6e/Golde33443.jpg",
            "https://placedog.net/640/480",
            "https://images.dog.ceo/breeds/retriever-golden/n02099601_3004.jpg",
        ],
    },
    {
        "id": "sample_04",
        "label": "dog",
        "path": os.path.join(TEST_IMAGES_DIR, "sample_04.jpg"),
        "urls": [
            "https://placedog.net/600/400",
            "https://images.dog.ceo/breeds/pembroke/n02113023_4378.jpg",
            "https://images.dog.ceo/breeds/labrador/n02099712_7428.jpg",
        ],
    },
    {
        "id": "sample_05",
        "label": "none",
        "path": os.path.join(TEST_IMAGES_DIR, "sample_05.jpg"),
        "urls": [
            "https://upload.wikimedia.org/wikipedia/commons/3/3f/Fronalpstock_big.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/8/84/Example.jpg",
            "https://picsum.photos/640/480",
        ],
    },
]

KEYWORDS_BY_LABEL = {
    "cat": ["gato", "gata", "gatito", "gatita", "cat", "kitty", "feline", "felino"],
    "dog": ["perro", "perra", "perrito", "perrita", "dog", "canine", "canino", "cachorro"],
    "none": [],
}


def get_installed_models() -> List[str]:
    """
    Obtiene la lista de modelos disponibles en LM Studio (incluidas todas las variantes).

    Usa ``get_installed_lms_models`` como fuente primaria (vía ``lms ls --json``).
    Cae a los modelos cargados en SDK y, por último, al CLI ``lms ls`` básico.

    Returns:
        List[str]: Lista de identificadores de modelos (con sufijo ``@variante``).
    """
    installed = _lms_get_installed_models()
    if installed:
        return installed

    loaded = list(_lms_loaded_model_keys())
    if loaded:
        return loaded

    return _lms_cli_installed_models()


def _download_bytes_from_url(image_url: str) -> bytes:
    """
    Descarga el contenido binario de una URL.
    
    Args:
        image_url (str): La URL del recurso a descargar.
        
    Returns:
        bytes: El contenido descargado.
        
    Raises:
        urllib.error.URLError: Si hay un error en la conexión.
    """
    request = urllib.request.Request(
        image_url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "Referer": "https://www.google.com/",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def sanitize_image_file(image_path: str) -> None:
    """
    Normaliza y limpia una imagen guardada localmente.
    
    Abre la imagen, la convierte a RGB y la vuelve a guardar como JPEG
    para eliminar metadatos y asegurar un formato estándar.
    
    Args:
        image_path (str): Ruta al archivo de imagen.
    """
    if Image is None:
        return

    with Image.open(image_path) as image_obj:
        clean = image_obj.convert("RGB")
        clean.save(image_path, format="JPEG", quality=95)


def download_image_if_missing(image_path: str, image_urls: List[str]) -> None:
    """
    Descarga una imagen desde una lista de URLs de respaldo si no existe localmente.
    
    Intenta descargar secuencialmente desde la lista de URLs hasta tener éxito.
    Verifica que el archivo descargado tenga un tamaño mínimo válido.
    
    Args:
        image_path (str): Ruta local donde guardar la imagen.
        image_urls (List[str]): Lista de URLs desde donde intentar la descarga.
        
    Raises:
        RuntimeError: Si no se puede descargar la imagen desde ninguna URL.
    """
    if os.path.exists(image_path):
        try:
            if os.path.getsize(image_path) >= 1024:
                return
            print(f"⚠️ Imagen local inválida (muy pequeña), reintentando descarga: {image_path}")
        except OSError:
            pass

    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    print(f"⚠️ Imagen no encontrada: {image_path}")

    errors = []
    for image_url in image_urls:
        print(f"   Descargando desde: {image_url}")
        try:
            data = _download_bytes_from_url(image_url)
            if len(data) < 1024:
                raise RuntimeError("archivo demasiado pequeño")

            with open(image_path, "wb") as file_handler:
                file_handler.write(data)

            sanitize_image_file(image_path)

            print(f"✅ Imagen descargada: {image_path}")
            return
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, RuntimeError) as error:
            errors.append(f"{image_url} -> {error}")
            continue

    details = "\n - " + "\n - ".join(errors) if errors else ""
    raise RuntimeError(
        f"No se pudo descargar imagen de prueba para {image_path}."
        f" Intentadas {len(image_urls)} URL(s).{details}"
    )


def ensure_test_images() -> List[Dict[str, str]]:
    """
    Asegura que todas las imágenes de prueba necesarias están disponibles localmente.
    
    Itera sobre los casos de prueba definidos y descarga las imágenes faltantes.
    
    Returns:
        List[Dict[str, str]]: Lista de casos de prueba resueltos con rutas locales válidas.
    """
    resolved_cases: List[Dict[str, str]] = []
    for case in TEST_CASES:
        image_path = case["path"]
        download_image_if_missing(image_path, case["urls"])
        sanitize_image_file(image_path)
        resolved_cases.append(
            {
                "id": case["id"],
                "label": case["label"],
                "path": image_path,
            }
        )
    return resolved_cases


def normalize_text(text: str) -> str:
    """
    Normaliza texto eliminando acentos y convirtiendo a minúsculas.
    
    Args:
        text (str): El texto de entrada.
        
    Returns:
        str: Texto normalizado (NFD, sin diacríticos, lowercase).
    """
    normalized = unicodedata.normalize("NFD", text)
    without_accents = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
    return without_accents.lower().strip()


def contains_any_keyword(response_text: str, keywords: List[str]) -> bool:
    """
    Verifica si alguna de las palabras clave aparece en el texto como palabra completa.

    Usa límites de palabra (\\b) para evitar falsos positivos por substring
    (p.ej. "cat" dentro de "significativa").

    Args:
        response_text (str): Texto donde buscar.
        keywords (List[str]): Lista de palabras clave a buscar.

    Returns:
        bool: True si alguna keyword está presente como palabra completa, False en caso contrario.
    """
    normalized_response = normalize_text(response_text)
    return any(
        re.search(r"\b" + re.escape(normalize_text(kw)) + r"\b", normalized_response)
        for kw in keywords
    )


def parse_structured_response(response_payload: str | Dict[str, object] | GenericObjectDetection) -> Dict[str, object]:
    """
    Convierte la respuesta del modelo VLM en un diccionario uniforme.
    
    Acepta objetos GenericObjectDetection, diccionarios o cadenas JSON.
    Valida que contenga los campos requeridos.
    
    Args:
        response_payload: Respuesta cruda del modelo.
        
    Returns:
        Dict[str, object]: Diccionario con los datos analizados.
        
    Raises:
        ValueError: Si el formato es inválido o faltan campos obligatorios.
    """
    if isinstance(response_payload, GenericObjectDetection):
        parsed = response_payload.model_dump()
    elif isinstance(response_payload, dict):
        parsed = response_payload
    else:
        parsed = json.loads(response_payload)

    if not isinstance(parsed, dict):
        raise ValueError("La salida estructurada no es un objeto JSON")

    # GenericObjectDetection genérico usa 'object_detected'; los esquemas médicos usan sus propios campos.
    # Verificamos que al menos existan confidence_score y justification, más algún campo de detección.
    detection_keys = {"object_detected", "polyp_detected", "agrees_with_user", "is_blurry"}
    has_detection = bool(detection_keys & set(parsed.keys()))
    has_base = {"confidence_score", "justification"} <= set(parsed.keys())
    if not has_detection or not has_base:
        missing = (detection_keys | {"confidence_score", "justification"}) - set(parsed.keys())
        raise ValueError(f"Faltan campos obligatorios en JSON: {', '.join(sorted(missing))}")

    return parsed


def validate_response(label: str, response_payload: str | Dict[str, object] | GenericObjectDetection) -> Tuple[bool, str]:
    """
    Valida si la respuesta del modelo es correcta para la etiqueta dada.
    
    Verifica que la respuesta no sea vacía, que sea un JSON válido y que
    la justificación contenga (o no) palabras clave según lo esperado.
    
    Args:
        label (str): La etiqueta esperada (ej. 'cat', 'dog', 'none').
        response_payload: La respuesta devuelta por el modelo.
        
    Returns:
        Tuple[bool, str]: (Success, Mensaje descriptivo del resultado).
    """
    if isinstance(response_payload, str) and (not response_payload or not response_payload.strip()):
        return False, (
            f"❌ Validación FALLIDA para {label}: respuesta vacía del modelo. "
            "Posibles causas: modelo sin capacidad VLM, timeout interno, "
            "o formato de entrada no aceptado por ese modelo/tag."
        )

    try:
        payload = parse_structured_response(response_payload)
    except Exception as error:
        return False, f"❌ Validación FALLIDA para {label}: salida JSON inválida ({error})."

    justification = str(payload.get("justification", ""))
    detection_value = str(payload.get("object_detected", "") or "")
    # Combina la justificación y el campo de detección para cubrir casos donde
    # el modelo pone la clase correcta en object_detected pero no la repite en justification.
    full_text = justification + " " + detection_value

    # Si la etiqueta es neutral o no tiene keywords esperadas, consideramos
    # válida la salida cuando NO aparece ninguna keyword de gato/perro.
    expected_keywords = KEYWORDS_BY_LABEL.get(label, [])
    if not expected_keywords:
        # comprobar ausencia de keywords de gato/perro
        combined = KEYWORDS_BY_LABEL.get("cat", []) + KEYWORDS_BY_LABEL.get("dog", [])
        if contains_any_keyword(full_text, combined):
            return False, (
                f"❌ Validación FALLIDA para {label}: se detectaron menciones de animales "
                "(gato/perro) en la respuesta, pero se esperaba ausencia."
            )
        return True, f"✅ Validación OK para {label}: salida válida y no se mencionan gato/perro."

    is_valid = contains_any_keyword(full_text, expected_keywords)
    if is_valid:
        return True, f"✅ Validación OK para {label}: JSON válido y justificación alineada con keywords esperadas."

    joined_keywords = ", ".join(expected_keywords)
    return False, (
        f"❌ Validación FALLIDA para {label}: no se detectaron keywords esperadas "
        f"({joined_keywords})."
    )


def run_smoke_test(model_tag: str, test_cases: List[Dict[str, str]]) -> int:
    """
    Ejecuta un test de humo para validar que el modelo puede procesar imágenes.
    
    Args:
        model_tag (str): Etiqueta del modelo a probar.
        test_cases (List[Dict[str, str]]): Lista de casos de prueba.
        
    Returns:
        int: Cantidad de casos de prueba exitosos.
    """
    print("\n=== TAREA 4: SMOKE TEST VLM (LM STUDIO) ===")
    print(f"👉 Modelo objetivo: {model_tag}")

    print("\n1. Inicializando VLMLoader...")
    loader = VLMLoader(model_path=model_tag, verbose=True)

    print("2. Validando modelo en LM Studio...")
    start_time = time.time()
    loader.preload_model()
    print(f"✅ Modelo precargado en {time.time() - start_time:.2f} s")

    print(f"3. Ejecutando inferencia para {len(test_cases)} imágenes de prueba...")
    all_passed = True

    try:
        for case in test_cases:
            label = case["label"]
            image_path = case["path"]
            case_id = case["id"]

            print(f"\n--- Caso: {case_id.upper()} ({label.upper()}) ---")
            print(f"Imagen: {image_path}")
            response = loader.inference(image_path, PROMPT)
            print("Respuesta estructurada del modelo:")
            if isinstance(response, GenericObjectDetection):
                print(response.model_dump_json(indent=2, ensure_ascii=False))
            else:
                print(response)

            valid, message = validate_response(label, response)
            print(message)
            if not valid:
                all_passed = False
    finally:
        try:
            loader.unload_model()
        except Exception as unload_error:
            print(f"⚠️ No se pudo liberar el modelo: {unload_error}")

    if all_passed:
        print("\n✅ TEST COMPLETADO CON ÉXITO (múltiples imágenes validadas).")
        return 0

    print("\n❌ TEST FALLIDO: al menos una validación no pasó.")
    return 1


def resolve_model(model_arg: str | None) -> str:
    """
    Resuelve el modelo objetivo para el test de humo.
    
    Args:
        model_arg (str | None): Etiqueta del modelo proporcionada por el usuario.
        
    Returns:
        str: Etiqueta del modelo seleccionado.
    """
    if model_arg:
        return model_arg

    installed_models = get_installed_models()

    if not installed_models:
        raise RuntimeError(
            "No se detectaron modelos disponibles en LM Studio. "
            "Carga uno en LM Studio o usa --model_path."
        )

    selected_model = installed_models[0]
    print(f"👉 Modo automático: usando primer modelo detectado: {selected_model}")
    return selected_model


def main(model_path: str | None = None) -> int:
    """
    Ejecuta el test de humo para validar que el modelo puede procesar imágenes.
    
    Args:
        model_path (str | None): Ruta al modelo a probar.
        
    Returns:
        int: Cantidad de casos de prueba exitosos.
    """
    try:
        model_tag = resolve_model(model_path)
        test_cases = ensure_test_images()
        return run_smoke_test(model_tag, test_cases)
    except Exception as error:
        print(f"\n❌ ERROR CRÍTICO: {error}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke Test VLM LM Studio (4 imágenes + validación automática)")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="ID/tag del modelo disponible en LM Studio (compatibilidad con scripts existentes)",
    )
    arguments = parser.parse_args()
    sys.exit(main(model_path=arguments.model_path))
