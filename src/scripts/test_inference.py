"""
Script de 'Smoke Test' (Prueba de Humo) para validar la inferencia VLM.
Tarea 4: Cargar modelo pequeño, pasar imagen, verificar texto y VRAM.
"""

import os
import sys
import time

# Add project root to sys.path to allow importing src
# archivo en src/scripts/ -> subir 2 niveles para llegar a root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.inference.vlm_runner import VLMLoader

# Configuración
DEFAULT_MODEL_PATH = "models/MiniCPM-V-2_6-Q8_0.gguf"
IMAGE_PATH = "data/raw/test_image.jpg" # Imagen de prueba (generada o existente)
PROMPT = "Describe detalladamente esta imagen (o lo que veas)."

def select_model():
    """Permite al usuario seleccionar un modelo de la carpeta models/ (recursivo)."""
    models_dir = "models"
    if not os.path.exists(models_dir):
        print(f"❌ La carpeta '{models_dir}' no existe.")
        return None
    
    # Listar archivos .gguf recursivamente, excluyendo los proyectores (mmproj)
    found_models = []
    for root, dirs, files in os.walk(models_dir):
        for f in files:
            if f.endswith(".gguf") and "mmproj" not in f:
                full_path = os.path.join(root, f)
                found_models.append(full_path)
    
    if not found_models:
        print(f"❌ No se encontraron modelos .gguf en '{models_dir}' (ni subcarpetas).")
        return None
    
    print("\n=== SELECCIÓN DE MODELO ===")
    print("Modelos disponibles:")
    for i, m in enumerate(found_models):
        # Mostrar path relativo para claridad
        rel_path = os.path.relpath(m, start=os.getcwd())
        print(f"  {i+1}. {rel_path}")
    
    while True:
        try:
            selection = input(f"\nSelecciona el número del modelo a probar (1-{len(found_models)}): ")
            if not selection.strip():
                continue
                
            idx = int(selection) - 1
            if 0 <= idx < len(found_models):
                selected_model = found_models[idx]
                print(f"👉 Seleccionado: {selected_model}")
                return selected_model
            print("❌ Selección inválida.")
        except ValueError:
            print("❌ Por favor ingresa un número válido.")

def ensure_test_image():
    """Crea una imagen de prueba dummy si no existe, solo para validar el pipeline."""
    if not os.path.exists(IMAGE_PATH):
        print(f"⚠️ Imagen {IMAGE_PATH} no encontrada. Generando imagen de prueba dummy...")
        os.makedirs(os.path.dirname(IMAGE_PATH), exist_ok=True)
        try:
            from PIL import Image, ImageDraw
            img = Image.new('RGB', (512, 512), color = (73, 109, 137))
            d = ImageDraw.Draw(img)
            d.text((10,10), "Hola VLM", fill=(255,255,0))
            img.save(IMAGE_PATH)
            print("✅ Imagen de prueba generada.")
        except ImportError:
            print("❌ No se pudo generar la imagen (falta Pillow). Por favor coloca una imagen jpg en data/raw/test_image.jpg")
            sys.exit(1)

def main(model_path=None):
    print("=== TAREA 4: SMOKE TEST VLM ===")
    
    # 1. Seleccionar Modelo (CLI arg o Interactivo)
    if not model_path:
        if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
            model_path = sys.argv[1]
            print(f"👉 Usando modelo pre-seleccionado: {model_path}")
        else:
            model_path = select_model()

    if not model_path:
        print("Cancelando test.")
        sys.exit(1)
    
    # 2. Verificar/Crear Imagen
    ensure_test_image()

    try:
        # 3. Inicializar Loader
        print(f"\n1. Inicializando VLMLoader con modelo: {model_path}...")
        loader = VLMLoader(model_path=model_path, verbose=True)

        # 4. Cargar Modelo (Watcher de VRAM aquí)
        print("2. Cargando modelo en memoria (Mira tu VRAM ahora!)...")
        start_time = time.time()
        # n_gpu_layers=-1 para mover todo a la GPU (RTX 4060 Ti tiene 16GB, Q8 ocupa ~8-10GB + KV Cache)
        print("   (Usando aceleración GPU total)")
        loader.load_model(n_ctx=2048, n_gpu_layers=-1) 
        print(f"✅ Modelo cargado en {time.time() - start_time:.2f} segundos.")

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
    parser = argparse.ArgumentParser(description="Smoke Test for VLM Inference")
    parser.add_argument("--model_path", type=str, help="Path to the GGUF model file", default=None)
    args = parser.parse_args()
    main(model_path=args.model_path)
