"""Smoke test VLM con LM Studio usando SDK oficial lmstudio."""

import argparse
import json
import os
import subprocess
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
    import msvcrt
except ImportError:
    msvcrt = None

try:
    from PIL import Image
except ImportError:
    Image = None

# Add project root to sys.path to allow importing src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.inference.vlm_runner import VLMLoader, VLMStructuredResponse

PROMPT = (
    "Identifica el animal principal de la imagen y responde en español. "
    "Incluye primero la clase del animal y luego una breve descripción."
)
TEST_IMAGES_DIR = "data/raw/smoke_test"

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


def parse_lms_ls_output(raw_output: str) -> List[str]:
    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
    if not lines:
        return []

    models: List[str] = []
    for line in lines:
        upper = line.upper()
        if "MODEL" in upper and "SIZE" in upper:
            continue

        parts = line.split()
        if not parts:
            continue

        candidate = parts[0].strip()
        if candidate and candidate.lower() not in {"id", "model", "name"}:
            models.append(candidate)

    return models


def _extract_model_key(item) -> str | None:
    if item is None:
        return None
    if isinstance(item, str):
        text = item.strip()
        return text or None
    if isinstance(item, dict):
        for key_name in ("key", "id", "model"):
            value = item.get(key_name)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None
    for attr_name in ("key", "id", "model"):
        value = getattr(item, attr_name, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _get_models_from_sdk() -> List[str]:
    if lms is None:
        return []

    try:
        list_fn = getattr(lms, "list_loaded_models", None)
        if not callable(list_fn):
            return []

        response = list_fn()
        keys: List[str] = []

        def collect(items):
            if items is None:
                return
            if isinstance(items, (list, tuple)):
                for entry in items:
                    key = _extract_model_key(entry)
                    if key:
                        keys.append(key)
                return
            if isinstance(items, dict):
                if "data" in items:
                    collect(items.get("data"))
                    return
                if "models" in items:
                    collect(items.get("models"))
                    return
                key = _extract_model_key(items)
                if key:
                    keys.append(key)
                return

            key = _extract_model_key(items)
            if key:
                keys.append(key)

            data_attr = getattr(items, "data", None)
            if data_attr is not None:
                collect(data_attr)
            models_attr = getattr(items, "models", None)
            if models_attr is not None:
                collect(models_attr)

        collect(response)
        return list(dict.fromkeys(keys))
    except Exception:
        return []


def get_installed_models() -> List[str]:
    """Obtiene modelos disponibles en LM Studio (SDK primero, CLI como fallback)."""
    sdk_models = _get_models_from_sdk()
    if sdk_models:
        return sdk_models

    try:
        result = subprocess.check_output(["lms", "ls"], text=True, stderr=subprocess.STDOUT)
        return parse_lms_ls_output(result)
    except Exception:
        return []


def select_model_interactive(installed_models: List[str]) -> str:
    print("\n=== SELECCIÓN DE MODELO LM STUDIO ===")

    if installed_models:
        options = installed_models + ["Introducir tag manualmente"]

        if os.name == "nt" and msvcrt:
            selected_idx = 0
            while True:
                os.system("cls")
                print("=== SELECCIÓN DE MODELO LM STUDIO ===")
                print("Modelos instalados (flechas + ENTER, ESC para cancelar):\n")

                for idx, option in enumerate(options):
                    pointer = "👉" if idx == selected_idx else "  "
                    print(f"{pointer} {option}")

                key = msvcrt.getch()
                if key in (b"\xe0", b"\x00"):
                    key2 = msvcrt.getch()
                    if key2 == b"H":  # UP
                        selected_idx = (selected_idx - 1) % len(options)
                    elif key2 == b"P":  # DOWN
                        selected_idx = (selected_idx + 1) % len(options)
                    continue

                if key in (b"\r", b"\n"):
                    selected = options[selected_idx]
                    if selected == "Introducir tag manualmente":
                        break
                    print(f"👉 Seleccionado: {selected}")
                    return selected

                if key == b"\x1b":  # ESC
                    return ""

        # Fallback no-Windows o terminal sin msvcrt
        print("Modelos instalados (detectados con API local o `lms ls`):")
        for idx, model_name in enumerate(installed_models, start=1):
            print(f"  {idx}. {model_name}")
        print("  0. Introducir tag manualmente")

        while True:
            selected = input(f"\nSelecciona una opción (0-{len(installed_models)}): ").strip()
            if not selected:
                continue
            if selected == "0":
                break
            try:
                selected_idx = int(selected) - 1
                if 0 <= selected_idx < len(installed_models):
                    selected_model = installed_models[selected_idx]
                    print(f"👉 Seleccionado: {selected_model}")
                    return selected_model
                print("❌ Selección inválida.")
            except ValueError:
                print("❌ Introduce un número válido.")

    manual_tag = input("Introduce el id/tag del modelo cargado en LM Studio: ").strip()
    return manual_tag


def _download_bytes_from_url(image_url: str) -> bytes:
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
    """Re-guarda imagen para limpiar metadatos y dejar formato uniforme."""
    if Image is None:
        return

    with Image.open(image_path) as image_obj:
        clean = image_obj.convert("RGB")
        clean.save(image_path, format="JPEG", quality=95)


def download_image_if_missing(image_path: str, image_urls: List[str]) -> None:
    """Descarga imagen desde lista de URLs fallback si no existe localmente."""
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
    """Asegura disponibilidad local de múltiples imágenes con nombres neutros."""
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
    normalized = unicodedata.normalize("NFD", text)
    without_accents = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
    return without_accents.lower().strip()


def contains_any_keyword(response_text: str, keywords: List[str]) -> bool:
    normalized_response = normalize_text(response_text)
    return any(normalize_text(keyword) in normalized_response for keyword in keywords)


def parse_structured_response(response_payload: str | Dict[str, object] | VLMStructuredResponse) -> Dict[str, object]:
    if isinstance(response_payload, VLMStructuredResponse):
        parsed = response_payload.model_dump()
    elif isinstance(response_payload, dict):
        parsed = response_payload
    else:
        parsed = json.loads(response_payload)

    if not isinstance(parsed, dict):
        raise ValueError("La salida estructurada no es un objeto JSON")

    required_keys = {"polyp_detected", "confidence_score", "justification"}
    missing = required_keys - set(parsed.keys())
    if missing:
        raise ValueError(f"Faltan campos obligatorios en JSON: {', '.join(sorted(missing))}")

    return parsed


def validate_response(label: str, response_payload: str | Dict[str, object] | VLMStructuredResponse) -> Tuple[bool, str]:
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

    # Si la etiqueta es neutral o no tiene keywords esperadas, consideramos
    # válida la salida cuando NO aparece ninguna keyword de gato/perro.
    expected_keywords = KEYWORDS_BY_LABEL.get(label, [])
    if not expected_keywords:
        # comprobar ausencia de keywords de gato/perro
        combined = KEYWORDS_BY_LABEL.get("cat", []) + KEYWORDS_BY_LABEL.get("dog", [])
        if contains_any_keyword(justification, combined):
            return False, (
                f"❌ Validación FALLIDA para {label}: se detectaron menciones de animales "
                "(gato/perro) en la justificación, pero se esperaba ausencia."
            )
        return True, f"✅ Validación OK para {label}: salida válida y no se mencionan gato/perro."

    is_valid = contains_any_keyword(justification, expected_keywords)
    if is_valid:
        return True, f"✅ Validación OK para {label}: JSON válido y justificación alineada con keywords esperadas."

    joined_keywords = ", ".join(expected_keywords)
    return False, (
        f"❌ Validación FALLIDA para {label}: no se detectaron keywords esperadas "
        f"({joined_keywords})."
    )


def run_smoke_test(model_tag: str, test_cases: List[Dict[str, str]]) -> int:
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
            if isinstance(response, VLMStructuredResponse):
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


def resolve_model(model_arg: str | None, interactive: bool) -> str:
    if model_arg:
        return model_arg

    installed_models = get_installed_models()

    if interactive:
        selected = select_model_interactive(installed_models)
        if not selected:
            raise RuntimeError("No se seleccionó ningún modelo.")
        return selected

    if not installed_models:
        raise RuntimeError(
            "No se detectaron modelos disponibles en LM Studio. "
            "Carga uno en LM Studio o usa --interactive/--model_path."
        )

    selected_model = installed_models[0]
    print(f"👉 Modo automático: usando primer modelo detectado: {selected_model}")
    return selected_model


def main(model_path: str | None = None, interactive: bool = False) -> int:
    try:
        model_tag = resolve_model(model_path, interactive)
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
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Activa selección interactiva del modelo.",
    )
    arguments = parser.parse_args()
    sys.exit(main(model_path=arguments.model_path, interactive=arguments.interactive))
