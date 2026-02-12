"""
Script de 'Smoke Test' (Prueba de Humo) para validar la inferencia VLM con Ollama.
Tarea 4: Verificar modelo (pull si necesario), pasar imagen, verificar texto.
"""

import os
import sys
import time
import subprocess

# Add project root to sys.path to allow importing src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.inference.vlm_runner import VLMLoader

# Configuración
DEFAULT_MODEL_TAG = "openbmb/minicpm-v4.5:8b"
IMAGE_PATH = "data/raw/test_image.jpg" # Imagen de prueba
PROMPT = "Describe detalladamente esta imagen (o lo que veas)."

def get_installed_models():
    """Obtiene la lista de modelos instalados en Ollama."""
    try:
        # Usamos subprocess para no depender de que la lib 'ollama' esté en este script (aunque vlm_runner la usa)
        res = subprocess.check_output("ollama list", shell=True).decode()
        lines = res.strip().split('\n')[1:] # Skip header
        models = [line.split()[0] for line in lines if line.strip()]
        return models
    except:
        return []

def select_model():
    """Permite al usuario seleccionar un modelo de Ollama."""
    models = get_installed_models()
    
    if not models:
        print("❌ No se detectaron modelos en 'ollama list'. Asegúrate de que Ollama está corriendo.")
        manual = input(f"Introduce el tag del modelo manualmente (default: {DEFAULT_MODEL_TAG}): ").strip()
        return manual if manual else DEFAULT_MODEL_TAG
    
    print("\n=== SELECCIÓN DE MODELO OLLAMA ===")
    print("Modelos disponibles:")
    for i, m in enumerate(models):
        print(f"  {i+1}. {m}")
    
    print(f"  0. Introducir manualmente")

    while True:
        try:
            selection = input(f"\nSelecciona una opción (1-{len(models)}): ").strip()
            if not selection:
                continue
            
            if selection == '0':
                 return input("Introduce el tag del modelo: ").strip()

            idx = int(selection) - 1
            if 0 <= idx < len(models):
                selected_model = models[idx]
                print(f"👉 Seleccionado: {selected_model}")
                return selected_model
            print("❌ Selección inválida.")
        except ValueError:
            print("❌ Por favor ingresa un número válido.")

def ensure_test_image():
    """Crea una imagen de prueba dummy si no existe."""
    if not os.path.exists(IMAGE_PATH):
        print(f"⚠️ Imagen {IMAGE_PATH} no encontrada. Generando imagen de prueba dummy...")
        os.makedirs(os.path.dirname(IMAGE_PATH), exist_ok=True)
        try:
            from PIL import Image, ImageDraw
            img = Image.new('RGB', (512, 512), color = (73, 109, 137))
            d = ImageDraw.Draw(img)
            d.text((10,10), "Hola VLM (Ollama)", fill=(255,255,0))
            img.save(IMAGE_PATH)
            print("✅ Imagen de prueba generada.")
        except ImportError:
            # Fallback si no hay PIL, copia manual requerida o error
            print("❌ No se pudo generar la imagen (falta Pillow).") 
            # Intentamos crear un archivo vacío solo para que no pete el loader check inicial
            # pero el loader fallará al leerlo si no es imagen válida.
            pass

def main(model_path=None):
    print("=== TAREA 4: SMOKE TEST VLM (OLLAMA) ===")
    
    # 1. Seleccionar Modelo
    # Nota: model_path aquí viene del argumento --model_path, que reusamos como model_tag
    if not model_path:
        model_tag = select_model()
    else:
        model_tag = model_path
        print(f"👉 Usando modelo pre-seleccionado: {model_tag}")

    if not model_tag:
        print("Cancelando test.")
        sys.exit(1)
    
    # 2. Verificar/Crear Imagen
    ensure_test_image()

    try:
        # 3. Inicializar Loader
        print(f"\n1. Inicializando VLMLoader con modelo: {model_tag}...")
        loader = VLMLoader(model_path=model_tag, verbose=True)

        # 4. "Cargar" Modelo (Verificar disponibilidad)
        print("2. Verificando modelo en Ollama...")
        start_time = time.time()
        loader.load_model() 
        print(f"✅ Modelo verificado en {time.time() - start_time:.2f} segundos.")

        # 5. Inferencia
        print(f"3. Ejecutando inferencia sobre {IMAGE_PATH}...")
        print(f"   Prompt: '{PROMPT}'")
        resp = loader.inference(IMAGE_PATH, PROMPT)

        print("\n" + "="*40)
        print("RESPUESTA DEL MODELO:")
        print("="*40)
        print(resp)
        print("="*40 + "\n")
        print("✅ TEST COMPLETADO CON ÉXITO.")

    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Smoke Test for VLM Inference with Ollama")
    parser.add_argument("--model_path", type=str, help="Ollama Model Tag (e.g. minicpm-v:latest)", default=None)
    args = parser.parse_args()
    main(model_path=args.model_path)
