"""
Script de 'Smoke Test' para validar inferencia VLM con Ollama usando 2 imágenes
(gato/perro) y verificación automática por palabras clave.
"""

import argparse
import os
import subprocess
import sys
import time
import unicodedata
import urllib.request
import urllib.error
from typing import Dict, List, Tuple

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

from src.inference.vlm_runner import VLMLoader

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
]

KEYWORDS_BY_LABEL = {
    "cat": ["gato", "gata", "gatito", "gatita", "cat", "kitty", "feline", "felino"],
    "dog": ["perro", "perra", "perrito", "perrita", "dog", "canine", "canino", "cachorro"],
}


def parse_ollama_list_output(raw_output: str) -> List[str]:
    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
    if not lines:
        return []

    models = []
    for line in lines[1:]:
        parts = line.split()
        if not parts:
            continue
        tag = parts[0].strip()
        if tag and tag.upper() != "NAME":
            models.append(tag)
    return models


def get_installed_models() -> List[str]:
    """Obtiene la lista de modelos instalados mediante `ollama list`."""
    try:
        result = subprocess.check_output(["ollama", "list"], text=True, stderr=subprocess.STDOUT)
        return parse_ollama_list_output(result)
    except Exception:
        return []


def select_model_interactive(installed_models: List[str]) -> str:
    print("\n=== SELECCIÓN DE MODELO OLLAMA ===")

    if installed_models:
        options = installed_models + ["Introducir tag manualmente"]

        if os.name == "nt" and msvcrt:
            selected_idx = 0
            while True:
                os.system("cls")
                print("=== SELECCIÓN DE MODELO OLLAMA ===")
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
        print("Modelos instalados (detectados con `ollama list`):")
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

    manual_tag = input("Introduce el tag del modelo de Ollama: ").strip()
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


def validate_response(label: str, response_text: str) -> Tuple[bool, str]:
    if not response_text or not response_text.strip():
        return False, (
            f"❌ Validación FALLIDA para {label}: respuesta vacía del modelo. "
            "Posibles causas: modelo sin capacidad VLM, timeout interno, "
            "o formato de entrada no aceptado por ese modelo/tag."
        )

    expected_keywords = KEYWORDS_BY_LABEL[label]
    is_valid = contains_any_keyword(response_text, expected_keywords)
    if is_valid:
        return True, f"✅ Validación OK para {label}: se detectó keyword esperada."

    joined_keywords = ", ".join(expected_keywords)
    return False, (
        f"❌ Validación FALLIDA para {label}: no se detectaron keywords esperadas "
        f"({joined_keywords})."
    )


def run_smoke_test(model_tag: str, test_cases: List[Dict[str, str]]) -> int:
    print("\n=== TAREA 4: SMOKE TEST VLM (OLLAMA) ===")
    print(f"👉 Modelo objetivo: {model_tag}")

    print("\n1. Inicializando VLMLoader...")
    loader = VLMLoader(model_path=model_tag, verbose=True)

    print("2. Precargando modelo en Ollama...")
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
            print("Respuesta del modelo:")
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
            "No se detectaron modelos instalados con `ollama list`. "
            "Instala uno o usa --interactive/--model_path."
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
    parser = argparse.ArgumentParser(description="Smoke Test VLM (2 imágenes + validación automática)")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Tag del modelo en Ollama (compatibilidad con scripts existentes)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Activa selección interactiva del modelo.",
    )
    arguments = parser.parse_args()
    sys.exit(main(model_path=arguments.model_path, interactive=arguments.interactive))
