import subprocess
import sys
import os
import platform
import shutil
import time
from typing import Callable, Optional

# --- Configuration: Model Registry (solo para descargas/pull) ---
# Añade aquí nuevos modelos siguiendo el formato.
MODELS_REGISTRY = {
    "minicpm_v_2_6_8b": {
        "name": "openbmb/minicpm-v2.6:8b",
        "description": "MiniCPM-V 2.6 (8B) - Versión Estable Compatible",
    },
    "minicpm_v_4_5_8b": {
        "name": "openbmb/minicpm-v4.5:8b", 
        "description": "MiniCPM-V 4.5 (8B) - SOTA OpenBMB",
    },
    "qwen3_vl_8b": {
        "name": "qwen3-vl:8b", 
        "description": "Qwen3-VL 8B (SOTA Razonamiento 2026)"
    },
    "internvl3_5_8b": {
        "name": "blaifa/InternVL3_5:8b",
        "description": "InternVL 3.5 (8B) - Blaifa"
    }
}

# Manejo de MSVCRT para inputs en Windows
try:
    import msvcrt
except ImportError:
    msvcrt = None

# Intento de importar psutil para métricas de RAM, manejo el error si no existe
try:
    import psutil  # type: ignore
except ImportError:
    psutil = None

# Intentar importar librerías de utilidad (requests, tqdm)
try:
    import requests
    from tqdm import tqdm
except ImportError:
    requests = None
    tqdm = None

# Habilitar secuencias ANSI en Windows (Simple & Robust)
if os.name == 'nt':
    os.system('') # Activa VT100 en Windows 10/11 sin deps extra

# --- Visual Styles ---
# Clases de colores ANSI para la interfaz de terminal
class Style:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    # Estilo visual para selección (Fondo blanco texto negro o invertido)
    SELECTED = '\033[7m' 

# --- Diagnostic Issue Class ---
# Objeto que representa un problema detectado durante el diagnóstico
class DiagnosticIssue:
    def __init__(self, description, fix_func, fix_name):
        self.description = description  # Descripción del problema para el usuario
        self.fix_func = fix_func        # La función Python que repara este problema
        self.fix_name = fix_name        # Nombre corto de la reparación para el menú
        self.label = fix_name           # Alias para el menú genérico
        self.action = fix_func          # Compatibilidad con MenuItem para ejecución directa
        self.children = []
        self.is_selected = False

# --- Menu State Storage ---
MENU_CURSOR_MEMORY: dict[str, int] = {}

# Generic Menu Item
class MenuItem:
    def __init__(self, label, action=None, description="", children=None):
        self.label = label
        self.action = action
        self.description = description
        self.children = children or []
        self.is_selected = False
        self.dynamic_label: Optional[Callable] = None


# --- Configuration: Dependencies ---
# Single source of truth for required python libraries
REQUIRED_LIBS = [
    "ollama",
    "pillow",   # Import name: PIL
    "opencv-python",
    "pandas",
    "matplotlib",
    "jupyter",
    "pytest",
    "tqdm"
]

# Mapping for import checks (Library Name -> Import Name)
# Only needed when import name differs from package name
LIB_IMPORT_MAP = {
    "pillow": "PIL",
    "opencv-python": "cv2",
    "pyyaml": "yaml",
    "pytest-mock": "pytest_mock"
}

# --- Helper Functions ---

def clear_screen_ansi():
    """
    Limpia pantalla usando CLS en Windows para asegurar limpieza.
    """
    if os.name == 'nt':
        os.system('cls')
    else:
        sys.stdout.write("\033[H\033[2J")
        sys.stdout.flush()

def clear_screen():
    """Wrapper compatible."""
    clear_screen_ansi()

def read_key():
    """Lee una tecla y devuelve un código unificado (UP, DOWN, ENTER, SPACE, ESC)."""
    if os.name == 'nt' and msvcrt:
        key = msvcrt.getch()
        if key == b'\xe0':  # Teclas de dirección
            key = msvcrt.getch()
            if key == b'H': return 'UP'
            if key == b'P': return 'DOWN'
            if key == b'K': return 'LEFT'
            if key == b'M': return 'RIGHT'
        elif key == b'\r': return 'ENTER'
        elif key == b' ': return 'SPACE'
        elif key == b'\x1b': return 'ESC'
        elif key == b'\x03': raise KeyboardInterrupt # Ctrl+C
    else:
        # Fallback para input simple o Linux (si se implementa luego)
        pass
    return None

def log(msg, level="info"):
    """
    Imprime mensajes con formato y color.
    level: info, success, error, warning, step
    """
    if level == "info":
        print(f"{Style.OKCYAN} ℹ {msg}{Style.ENDC}")
    elif level == "success":
        print(f"{Style.OKGREEN} ✔ {msg}{Style.ENDC}")
    elif level == "error":
        print(f"{Style.FAIL} ✖ {msg}{Style.ENDC}")
    elif level == "warning":
        print(f"{Style.WARNING} ⚠ {msg}{Style.ENDC}")
    elif level == "step":
        print(f"\n{Style.BOLD}➤ {msg}{Style.ENDC}")

def ask_user(question, default="y"):
    """Pide confirmación con interfaz visual (flechas + ENTER, ESC cancela)."""
    normalized_default = (default or "y").strip().lower()
    if normalized_default not in ("y", "n"):
        normalized_default = "y"

    # Opción 0 = Yes, opción 1 = No
    selected = 0 if normalized_default == "y" else 1

    while True:
        clear_screen_ansi()
        print(f"{Style.BOLD}{Style.WARNING} {question} {Style.ENDC}")
        print(f"{Style.DIM} Usa flechas izquierda/derecha (o arriba/abajo), ENTER para confirmar, ESC para cancelar.{Style.ENDC}\n")

        yes_label = " Sí "
        no_label = " No "

        if selected == 0:
            yes_render = f"{Style.SELECTED}{yes_label}{Style.ENDC}"
            no_render = no_label
        else:
            yes_render = yes_label
            no_render = f"{Style.SELECTED}{no_label}{Style.ENDC}"

        print(f"   {yes_render}    {no_render}\n")

        key = read_key()
        if key in ('LEFT', 'UP'):
            selected = 0
        elif key in ('RIGHT', 'DOWN'):
            selected = 1
        elif key == 'ENTER':
            return selected == 0
        elif key == 'ESC':
            return False


def input_with_esc(prompt):
    """Entrada de texto que permite cancelar con ESC en Windows. Retorna None si se cancela."""
    if os.name == 'nt' and msvcrt:
        print(prompt, end="", flush=True)
        buffer = []
        while True:
            key = msvcrt.getch()
            if key == b'\x1b':  # ESC
                print()
                return None
            if key in (b'\r', b'\n'):
                print()
                return "".join(buffer).strip()
            if key == b'\x08':  # Backspace
                if buffer:
                    buffer.pop()
                    print("\b \b", end="", flush=True)
                continue
            if key in (b'\xe0', b'\x00'):
                # Special key prefix (arrows/function keys), consume next byte and ignore.
                _ = msvcrt.getch()
                continue
            try:
                char = key.decode("utf-8")
            except UnicodeDecodeError:
                continue
            buffer.append(char)
            print(char, end="", flush=True)

    value = input(prompt).strip()
    if value.lower() == "esc":
        return None
    return value


def wait_for_any_key(message="Press any key to return..."):
    """Pausa breve: espera cualquier tecla y retorna al menú anterior."""
    print(f"\n{Style.DIM}{message}{Style.ENDC}", end="", flush=True)
    if os.name == 'nt' and msvcrt:
        _ = msvcrt.getch()
        print()
        return

    # Fallback fuera de Windows
    input()

def run_cmd(cmd, critical=True):
    """
    Ejecuta un comando de shell e imprime el comando en gris.
    Si falla y critical=True, pregunta al usuario si reintentar.
    """
    print(f"{Style.DIM}$ {cmd}{Style.ENDC}")
    try:
        subprocess.check_call(cmd, shell=True)
        return True
    except subprocess.CalledProcessError:
        log(f"Command failed: {cmd}", "error")
        if critical:
            if ask_user("Critical command failed. Retry?", "y"):
                return run_cmd(cmd, critical)
            return False
        return False

def get_sys_info():
    """Recopila información básica del hardware (OS, Python, CPU, GPU, RAM)."""
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
        "cpu_cores": os.cpu_count(),
        "gpu": "N/A",
        "ram": "N/A"
    }

    # RAM via psutil
    if psutil:
        try:
            mem = psutil.virtual_memory()
            info["ram"] = f"{mem.total / (1024**3):.1f} GB"
        except:
             info["ram"] = "Unknown"
    else:
        info["ram"] = "Unknown (psutil missing)"

    # GPU via nvidia-smi
    try:
        res = subprocess.check_output("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True)
        info["gpu"] = res.decode("utf-8").strip()
    except:
        info["gpu"] = "Not Detected (CPU Mode)"
    
    return info

def check_ollama():
    """Verifica si 'ollama' está instalado y el servicio corriendo."""
    try:
        subprocess.check_call("ollama --version", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except:
        return False

def get_installed_ollama_models():
    """Obtiene modelos instalados desde `ollama list` (sin depender de registros estáticos)."""
    if not check_ollama():
        return []

    try:
        res = subprocess.check_output("ollama list", shell=True).decode()
        lines = [line.strip() for line in res.splitlines() if line.strip()]
        if len(lines) <= 1:
            return []

        models = []
        for line in lines[1:]:  # skip header
            parts = line.split()
            if parts:
                model_tag = parts[0].strip()
                if model_tag and model_tag.upper() != "NAME":
                    models.append(model_tag)
        return models
    except:
        return []

def _normalize_model_tag(model_tag):
    """Normaliza un tag de modelo (lower + trim)."""
    if not model_tag:
        return ""
    return str(model_tag).strip().lower()


def _split_model_tag(model_tag):
    """Divide `modelo:tag` en (modelo, tag|None)."""
    normalized = _normalize_model_tag(model_tag)
    if not normalized:
        return "", None

    if ":" not in normalized:
        return normalized, None

    model_name, tag = normalized.rsplit(":", 1)
    return model_name, tag


def _is_latest_alias_equivalent(model_a, model_b):
    """Equivalencia limitada: `model` <-> `model:latest`."""
    name_a, tag_a = _split_model_tag(model_a)
    name_b, tag_b = _split_model_tag(model_b)

    if not name_a or not name_b or name_a != name_b:
        return False

    # exacto
    if tag_a == tag_b:
        return True

    # alias latest explícito/implícito
    a_latest_or_none = tag_a in (None, "latest")
    b_latest_or_none = tag_b in (None, "latest")
    return a_latest_or_none and b_latest_or_none


def _contains_equivalent_model(model_tag, model_list):
    """Comprueba si model_tag ya existe en model_list con equivalencia exacta/latest."""
    for installed in model_list:
        if _is_latest_alias_equivalent(model_tag, installed):
            return True
    return False

def is_model_installed(model_tag, installed_models):
    """Comprueba si un modelo está instalado sin mezclar tags diferentes (ej: 8b != latest)."""
    if not model_tag:
        return False

    target = _normalize_model_tag(model_tag)
    installed_normalized = [_normalize_model_tag(m) for m in installed_models if m]

    if target in installed_normalized:
        return True

    return _contains_equivalent_model(target, installed_normalized)

def _resolve_model_tag(model_ref):
    """Resuelve una referencia de modelo (key de registro o tag directo) a su tag real."""
    if model_ref in MODELS_REGISTRY:
        return MODELS_REGISTRY[model_ref]["name"]
    return str(model_ref).strip()

def _resolve_model_description(model_ref):
    """Resuelve una descripción legible para logs de acciones de modelo."""
    if model_ref in MODELS_REGISTRY:
        return MODELS_REGISTRY[model_ref].get("description", "Registry model")
    return "Custom/Detected model"

def pull_ollama_model(model_ref):
    """Descarga (pull) un modelo de Ollama (registro o tag directo)."""
    model_tag = _resolve_model_tag(model_ref)
    if not model_tag:
        log("Invalid model reference.", "error")
        return False

    model_description = _resolve_model_description(model_ref)
    
    log(f"Pulling {model_tag} ({model_description})...", "step")
    
    # Check if ollama is running first (simple check)
    if not check_ollama():
        log("Ollama binary not found or not in PATH.", "error")
        log("Please install Ollama from https://ollama.com/", "warning")
        return False

    try:
        # Use subprocess to stream output properly if possible, or just wait
        # "ollama pull" writes to stderr usually for progress bars
        subprocess.check_call(f"ollama pull {model_tag}", shell=True)
        log(f"Successfully pulled {model_tag}", "success")
        return True
    except subprocess.CalledProcessError:
        log(f"Failed to pull {model_tag}. Is Ollama server running?", "error")
        return False

def remove_ollama_model(model_ref):
    """Elimina (rm) un modelo de Ollama con confirmación simple."""
    model_tag = _resolve_model_tag(model_ref)
    if not model_tag:
        log("Invalid model reference.", "error")
        return False

    if not check_ollama():
        log("Ollama binary not found or not in PATH.", "error")
        log("Please install Ollama from https://ollama.com/", "warning")
        return False

    if not ask_user(f"Confirm remove model '{model_tag}'?", "n"):
        log("Delete cancelled by user.", "warning")
        return False

    log(f"Removing model {model_tag}...", "step")
    try:
        subprocess.check_call(f"ollama rm {model_tag}", shell=True)
        log(f"Successfully removed {model_tag}", "success")
        return True
    except subprocess.CalledProcessError:
        log(f"Failed to remove {model_tag}.", "error")
        return False

def toggle_ollama_model(model_ref, installed_models):
    """Si el modelo está instalado lo elimina, si no está instalado lo descarga."""
    model_tag = _resolve_model_tag(model_ref)
    if not model_tag:
        log("Invalid model reference.", "error")
        return False

    is_installed = is_model_installed(model_tag, installed_models)

    if is_installed:
        return remove_ollama_model(model_ref)
    return pull_ollama_model(model_ref)

def get_manageable_models(installed_models):
    """Construye lista de modelos a mostrar: registro + todos los detectados por `ollama list`."""
    entries = []
    tracked_tags = []

    for key, data in MODELS_REGISTRY.items():
        model_tag = data["name"]
        tracked_tags.append(_normalize_model_tag(model_tag))
        entries.append({
            "model_ref": key,
            "model_tag": model_tag,
            "description": data.get("description", "Registry model"),
            "from_registry": True,
        })

    for model_tag in installed_models:
        normalized_installed = _normalize_model_tag(model_tag)
        if _contains_equivalent_model(normalized_installed, tracked_tags):
            continue
        tracked_tags.append(normalized_installed)
        entries.append({
            "model_ref": model_tag,
            "model_tag": model_tag,
            "description": "Detected from ollama list",
            "from_registry": False,
        })

    return entries

def list_test_files():
    """Escanea la carpeta tests/ y devuelve los archivos .py válidos."""
    test_dir = "tests"
    if not os.path.exists(test_dir):
        return []
    
    files = [f for f in os.listdir(test_dir) if f.startswith("test_") and f.endswith(".py")]
    return sorted(files)


def _get_item_description(item):
    """Devuelve la descripción del item si existe y no está vacía."""
    description = getattr(item, "description", "")
    if not description:
        return ""
    return str(description).strip()

def manage_models_menu_ui():
    """Menú dinámico para gestionar descargas de modelos OLLAMA."""
    def pull_custom_model_ui():
        clear_screen_ansi()
        print_banner()
        print(f"{Style.BOLD} ADD CUSTOM MODEL {Style.ENDC}")
        custom_tag = input_with_esc(f"{Style.WARNING}Model tag (e.g. qwen2.5vl:7b, ESC cancel): {Style.ENDC}")
        if custom_tag is None:
            log("Operation cancelled by user (ESC).", "warning")
            wait_for_any_key()
            return None

        if not custom_tag:
            log("Empty model tag. Operation cancelled.", "warning")
            wait_for_any_key()
            return None

        pull_ollama_model(custom_tag)
        wait_for_any_key()
        return None

    def header():
        print_banner()
        print(f"{Style.BOLD} OLLAMA MODEL MANAGER {Style.ENDC}")
        print(" Select a model (all downloaded detected via ollama list):")

    while True:
        options = []
        
        # Check if ollama is running
        ollama_status = "RUNNING" if check_ollama() else "NOT FOUND/STOPPED"
        ollama_color = Style.OKGREEN if ollama_status == "RUNNING" else Style.FAIL
        
        # Try to get list of installed models
        installed_models = get_installed_ollama_models()

        model_entries = get_manageable_models(installed_models)

        for entry in model_entries:
            model_ref = entry["model_ref"]
            model_tag = entry["model_tag"]
            is_installed = is_model_installed(model_tag, installed_models)

            # Action wrapper using closure for key capture
            action = lambda ref=model_ref, m=installed_models: (toggle_ollama_model(ref, m), wait_for_any_key())

            menu_item = MenuItem("", action, description=entry["description"])

            def dynamic_label(is_selected_row, tag=model_tag, installed=is_installed):
                if installed and is_selected_row:
                    status_icon = f"{Style.FAIL}🗑 REMOVE{Style.ENDC}"
                elif installed:
                    status_icon = f"{Style.OKGREEN}✔ INSTALLED{Style.ENDC}"
                else:
                    status_icon = f"{Style.OKBLUE}☁ PULL{Style.ENDC}"
                return f" {tag:<30} | {status_icon}"

            menu_item.dynamic_label = dynamic_label
            options.append(menu_item)

        options.append(MenuItem(" Add custom model by tag...", pull_custom_model_ui, description="Introduce un tag manual y descarga ese modelo en Ollama."))
        options.append(MenuItem(" Back", lambda: "BACK", description="Vuelve al menú de Tests & Models."))
    
        choice = interactive_menu(options, header_func=header, menu_id="manage_models_menu")
        
        if not choice:
            break
            
        # choice es un objeto MenuItem, hay que ejecutar su acción
        if hasattr(choice, 'action') and callable(choice.action):
            clear_screen_ansi()
            res = choice.action()
            if res == "BACK":
                break    

def run_tests_menu():
    """Menú para ejecutar tests (Unitarios y Smoke Tests)."""

    def run_smoke_test_in_process(model_tag):
        """Ejecuta smoke test sin subprocess para mantener navegación de menú estable."""
        try:
            from src.scripts.test_inference import main as smoke_test_main
        except Exception as error:
            log(f"Could not import smoke test script: {error}", "error")
            return 1

        try:
            return int(smoke_test_main(model_path=model_tag, interactive=False))
        except Exception as error:
            log(f"Smoke test crashed: {error}", "error")
            return 1
    
    def run_all_unit_tests():
        log("Running All Unit Tests...", "step")
        run_cmd("uv run python -m pytest tests/")
        wait_for_any_key()

    def run_specific_test():
        while True:
            files = list_test_files()
            if not files:
                log("No tests found in tests/ folder.", "warning")
                time.sleep(1)
                return

            test_opts = [MenuItem(f, description="Ejecuta solo este archivo de tests con pytest.") for f in files]
            test_opts.append(MenuItem("Cancel", lambda: None, description="Vuelve al menú anterior sin ejecutar pruebas."))

            def t_header():
                print_banner()
                print(f"{Style.BOLD} SELECT TEST FILE {Style.ENDC}")

            selection = interactive_menu(test_opts, header_func=t_header, menu_id="run_specific_test_selector")

            # ESC o Cancel => volver un nivel atrás
            if not selection or selection.label.strip() == "Cancel":
                return

            fname = selection.label
            clear_screen_ansi()
            log(f"Running {fname}...", "step")
            run_cmd(f"uv run python -m pytest tests/{fname}")
            wait_for_any_key("Finished. Press any key to return to test selector...")

    def run_smoke_test_wrapper():
        """Lanza smoke test y mantiene el retorno al selector de modelo (un nivel atrás)."""
        while True:
            # Selección dinámica desde `ollama list`.
            model_opts = []
            installed_models = get_installed_ollama_models()

            if not installed_models:
                log("No hay modelos instalados en Ollama. Instala uno desde 'Manage/Pull Ollama Models...'.", "warning")
                wait_for_any_key("Press any key to return to tests menu...")
                return

            for model_tag in installed_models:
                model_opts.append(MenuItem(model_tag, description="Ejecuta el smoke test usando este modelo instalado."))

            model_opts.append(MenuItem("Cancel", lambda: None, description="Vuelve al menú anterior sin ejecutar smoke test."))

            def m_header():
                print_banner()
                print(f"{Style.BOLD} SELECT INFERENCE MODEL {Style.ENDC}")

            selection = interactive_menu(model_opts, header_func=m_header, menu_id="run_smoke_model_selector")

            # ESC o Cancel => volver un nivel atrás (TEST & MODEL MANAGER)
            if not selection or selection.label.strip() == "Cancel":
                return

            model_tag = selection.label

            if not model_tag:
                log("No model selected.", "warning")
                wait_for_any_key("Press any key to return to model selector...")
                continue

            clear_screen_ansi()
            log(f"Launching Inference with {model_tag}...", "step")
            result_code = run_smoke_test_in_process(model_tag)
            if result_code != 0:
                log("Smoke test failed (Exit Code 1). Check output above.", "error")
            wait_for_any_key("Press any key to return to model selector...")


    def header():
        print_banner()
        print(f"{Style.BOLD} TEST & MODEL MANAGER {Style.ENDC}")

    options = [
        MenuItem(" Run All Unit Tests (pytest)", run_all_unit_tests, description="Ejecuta todos los tests dentro de la carpeta tests/."),
        MenuItem(" Run Specific Test File...", run_specific_test, description="Abre un selector para ejecutar un único archivo de test."),
        MenuItem(" Run Smoke Test (Inference Demo)", run_smoke_test_wrapper, description="Lanza una inferencia rápida para validar flujo modelo+imagen."),
        MenuItem(" Manage/Pull Ollama Models...", manage_models_menu_ui, description="Instala, elimina o gestiona modelos de Ollama."),
        MenuItem(" Return to Main Menu", lambda: "BACK", description="Vuelve al menú principal.")
    ]

    while True:
        choice = interactive_menu(options, header_func=header, multi_select=False, menu_id="tests_manager_menu")
        if choice == "BACK" or not choice:
            break
        
        # Ensure choice is a single item, not a list
        if isinstance(choice, list):
            choice = choice[0] if len(choice) > 0 else None
        
        if choice and hasattr(choice, 'action') and callable(choice.action):
            clear_screen_ansi()
            res = choice.action()
            if res == "BACK": break

def print_banner():
    """Muestra el banner principal con el estado del sistema."""
    clear_screen()
    info = get_sys_info()
    
    print(f"{Style.HEADER}{Style.BOLD}")
    print("╔═════════════════════════════════════════════════════════════════╗")
    print("║              🩺 TFG VLM Medical - Manager Tool v5.0             ║")
    print("╠═════════════════════════════════════════════════════════════════╣")
    print(f"║ {Style.OKCYAN}OS     : {info['os']:<46}{Style.HEADER}         ║")
    print(f"║ {Style.OKCYAN}Python : {info['python']:<46}{Style.HEADER}         ║")
    print(f"║ {Style.OKCYAN}CPU    : {info['cpu_cores']} Cores{' ':<39}{Style.HEADER}        ║")
    print(f"║ {Style.OKCYAN}RAM    : {info['ram']:<46}{Style.HEADER}         ║")
    print(f"║ {Style.OKCYAN}GPU    : {info['gpu']:<46}{Style.HEADER}         ║")
    print("╚═════════════════════════════════════════════════════════════════╝")

def restart_program():
    """Reinicia el script actual de forma limpia."""
    print(f"\n{Style.BOLD}{Style.WARNING}🔴 CRITICAL ENVIRONMENT UPDATE DETECTED 🔴{Style.ENDC}")
    print(f"{Style.WARNING}The application must restart to load the new libraries correctly.{Style.ENDC}")
    print(f"{Style.WARNING}Restarting in 2 seconds...{Style.ENDC}")
    time.sleep(2)
    
    # Restore terminal state before restart (Evita que la terminal se 'rompa')
    print("\033[?25h", end="") # Show cursor explicitly
    sys.stdout.flush()
    
    python = sys.executable
    # En Windows, os.execl no reemplaza realmente el proceso, lo que causa el "ghosting".
    # Usamos subprocess para lanzar el hijo y esperar a que termine, manteniendo la terminal ocupada.
    try:
        subprocess.check_call([python] + sys.argv)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error restarting: {e}")
    
    sys.exit(0)


# --- Core Utility Functions ---

def detect_gpu():
    """Detecta si hay una GPU NVIDIA disponible usando nvidia-smi."""
    try:
        subprocess.check_call("nvidia-smi", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except:
        return False

def check_uv():
    """Verifica si 'uv' está instalado y accesible en el PATH."""
    try:
        subprocess.check_call("uv --version", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except:
        return False

def create_project_structure(verbose=True):
    """Crea la estructura de carpetas necesaria si no existe."""
    folders = [
        "data/raw",
        "data/processed",
        "models",
        "src/preprocessing",
        "src/inference",
        "notebooks"
    ]
    
    created = 0
    for folder in folders:
        if not os.path.exists(folder):
            try:
                os.makedirs(folder, exist_ok=True)
                if verbose: log(f"Created missing directory: {folder}", "success")
                created += 1
            except Exception as e:
                log(f"Failed to create {folder}: {e}", "error")
    
    if verbose and created == 0:
        log("All directories verified.", "success")

# --- Modular Fixers ---
# Estas funciones son llamadas automáticamente por el Smart Fix Menu

def fix_folders():
    """Reparación: Regenera carpetas faltantes."""
    log("Regenerating folder structure...", "step")
    create_project_structure(verbose=True)

def fix_uv():
    """Reparación: Instala la herramienta uv."""
    log("Installing 'uv' package manager...", "step")
    run_cmd("pip install uv")

def fix_libs(libs_to_install=None):
    """Reparación: Reinstala una lista de librerías en modo Soft (sin romper dependencias)."""
    if libs_to_install is None:
        if os.path.exists("pyproject.toml"):
            log("Syncing dependencies from pyproject.toml/uv.lock...", "step")
            run_cmd("uv sync")
            return

        # Fallback si no existe pyproject
        libs_to_install = REQUIRED_LIBS
    
    # Aseguramos que sea lista
    if isinstance(libs_to_install, str):
        libs_to_install = [libs_to_install]

    libs_str = " ".join(libs_to_install)
    log(f"Installing libraries: {libs_str}...", "step")
    # Eliminamos --no-deps para asegurar que se instalen las dependencias faltantes (ej: requests)
    run_cmd(f"uv pip install --force-reinstall {libs_str}")



# --- Diagnostic System ---

def perform_diagnostics():
    """
    Ejecuta comprobaciones del sistema.
    Retorna:
    - report: Lista de tuplas (Nombre, Estado, ColorStatus) para mostrar.
    - issues: Lista de objetos DiagnosticIssue con soluciones vinculadas.
    """
    report = []
    issues = []
    
    # 1. Folder Structure Check
    folders = ["data/raw", "data/processed", "models", "src/preprocessing", "src/inference", "notebooks"]
    missing_folders = False
    for folder in folders:
        if os.path.exists(folder):
             report.append((f"Dir: {folder}", "Exists", "OK"))
        else:
             report.append((f"Dir: {folder}", "MISSING", "FAIL"))
             missing_folders = True
    
    if missing_folders:
        issues.append(DiagnosticIssue("Missing Folders", fix_folders, "Regenerate Folders"))

    # 2. UV Method Check
    if check_uv():
        report.append(("Tool: uv", "Installed", "OK"))
    else:
        report.append(("Tool: uv", "MISSING", "FAIL"))
        issues.append(DiagnosticIssue("Missing uv", fix_uv, "Install uv"))



    # 4. Key Libraries
    # Verificamos importación de librerías clave una por una
    for pkg_name in REQUIRED_LIBS:
        # Determine import name
        import_name = LIB_IMPORT_MAP.get(pkg_name, pkg_name)
        
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "Unknown")
            report.append((f"Lib: {pkg_name}", version, "OK"))
        except ImportError:
            report.append((f"Lib: {pkg_name}", "MISSING", "FAIL"))
            fix_lambda = lambda p=[pkg_name]: fix_libs(p)
            issues.append(DiagnosticIssue(f"Missing {pkg_name}", fix_lambda, f"Install {pkg_name}"))
        except Exception as e:
            report.append((f"Lib: {pkg_name}", "ERROR", "FAIL"))
            fix_lambda = lambda p=[pkg_name]: fix_libs(p)
            issues.append(DiagnosticIssue(f"Error {pkg_name}", fix_lambda, f"Reinstall {pkg_name}"))

    # Ollama Service Check
    if check_ollama():
        report.append(("Service: Ollama", "Running", "OK"))
    else:
        report.append(("Service: Ollama", "NOT DETECTED", "FAIL"))
        issues.append(DiagnosticIssue("Ollama not finding", lambda: log("Install Ollama from https://ollama.com", "warning"), "Install Ollama"))

    # Ollama Python Lib
    try:
        import ollama
        report.append(("Lib: ollama", "Installed", "OK"))
    except ImportError:
        report.append(("Lib: ollama", "MISSING", "FAIL"))
        issues.append(DiagnosticIssue("Missing ollama lib", lambda: fix_libs("ollama"), "Install olama lib"))

    # 5. Storage
    free_gb = shutil.disk_usage(".").free / (1024**3)
    if free_gb < 10: 
        report.append(("Disk Space", f"{free_gb:.1f} GB", "WARN"))
    else:
        report.append(("Disk Space", f"{free_gb:.1f} GB (Free)", "OK"))

    return report, issues


def print_report_table(report):
    """Imprime la tabla de diagnóstico de forma formateada."""
    print(f"\n{Style.HEADER}┌──────────────────────────┬───────────────────────────────┬──────┐{Style.ENDC}")
    print(f"{Style.HEADER}│ Component                │ Status/Value                  │ Res  │{Style.ENDC}")
    print(f"{Style.HEADER}├──────────────────────────┼───────────────────────────────┼──────┤{Style.ENDC}")
    
    for name, value, status in report:
        color = Style.OKGREEN if status == "OK" else (Style.WARNING if status == "WARN" else Style.FAIL)
        # Limpiar nombres para visualización
        display_name = name.replace("cv2", "OpenCV").replace("llama_cpp", "LlamaCPP")
        print(f"│ {display_name:<24} │ {str(value):<29} │ {color}{status:<4}{Style.ENDC} │")

    print(f"{Style.HEADER}└──────────────────────────┴───────────────────────────────┴──────┘{Style.ENDC}")


def interactive_menu(options, header_func=None, multi_select=False, info_text="", menu_id=None):
    """
    Menú interactivo genérico controlado por teclado.
    Soporta circularidad y navegación por sub-niveles.
    """
    current_row = MENU_CURSOR_MEMORY.get(menu_id, 0) if menu_id else 0
    in_sub_nav = False

    def _persist_cursor(flat_rows, row_index):
        if not menu_id or not flat_rows:
            return

        safe_index = row_index % len(flat_rows)
        row = flat_rows[safe_index]

        # Si estamos en hijo, memoriza el padre para restaurar navegación estable.
        if row.get('level', 0) > 0 and row.get('parent') is not None:
            parent = row.get('parent')
            for idx, candidate in enumerate(flat_rows):
                if candidate.get('obj') == parent and candidate.get('level') == 0:
                    MENU_CURSOR_MEMORY[menu_id] = idx
                    return

        MENU_CURSOR_MEMORY[menu_id] = safe_index
    
    # Hide cursor
    print("\033[?25l", end="")
    
    # Limpieza inicial completa con CLS
    if os.name == 'nt':
        os.system('cls')
    else:
        sys.stdout.write("\033[H\033[2J")

    try:
        while True:
            # 1. Construir lista plana de filas visibles
            flat_rows = []
            for opt in options:
                # El elemento principal
                flat_rows.append({'obj': opt, 'level': 0})
                # Hijos (siempre visibles según el usuario)
                if hasattr(opt, 'children') and opt.children:
                    for child in opt.children:
                        flat_rows.append({'obj': child, 'level': 1, 'parent': opt})

            # 2. Asegurar circularidad inicial y clamping
            if not flat_rows:
                if menu_id:
                    MENU_CURSOR_MEMORY[menu_id] = 0
                return None
            current_row = current_row % len(flat_rows)

            # 3. Renderizado
            clear_screen_ansi()
            if header_func: header_func()
            if info_text: print(f"{info_text}\n")
            
            # Ayuda contextual
            if in_sub_nav:
                print(f"{Style.DIM} [SUB-NAV] SPACE: Seleccionar/Deseleccionar, ESC: Volver atrás, ENTER: Confirmar todo.{Style.ENDC}")
            else:
                print(f"{Style.DIM} Arriba/Abajo: Navegar, SPACE: Entrar en Submenú / Seleccionar, ENTER: Confirmar Todo.{Style.ENDC}")

            # Viewport Calculation (Keep cursor centered)
            term_height = 24
            if shutil:
                term_height = shutil.get_terminal_size().lines
            
            # Lines reserved roughly for: (Banner/Header ~10) + (Info Text ~2) + (Footer ~3) + Space
            reserved_lines = 15 
            max_visible_items = max(5, term_height - reserved_lines)
            
            # Calculate scroll offset to keep current_row centered
            half_view = max_visible_items // 2
            start_row = max(0, current_row - half_view)
            end_row = start_row + max_visible_items
            
            # Adjustment if we are near the end (but for circular menus this is tricky, let's keep it simple mostly linear view)
            if end_row > len(flat_rows):
                end_row = len(flat_rows)
                start_row = max(0, end_row - max_visible_items)

            visible_slice = flat_rows[start_row:end_row]

            # Determine current level for filtering indicators
            cur_level = flat_rows[current_row]['level']
            
            # Top Indicator
            if start_row > 0:
                hidden_above = sum(1 for r in flat_rows[:start_row] if r['level'] == cur_level)
                msg = f"▲ ({hidden_above}) ▲" if hidden_above > 0 else "▲"
                print(f"  {Style.DIM}{msg}{Style.ENDC}")
            else:
                print(" ")

            # Render Loop
            for i, row in enumerate(visible_slice):
                idx = start_row + i # Actual index in flat_rows
                is_selected_row = (idx == current_row)
                item = row['obj']
                level = row['level']
                
                # Checkbox
                checkbox = ""
                if multi_select:
                    is_checked = getattr(item, 'is_selected', False)
                    checkbox = f"[{'x' if is_checked else ' '}] "
                
                pointer = "👉" if is_selected_row else "  "
                indent = "    " * level
                suffix = " >" if level == 0 and hasattr(item, 'children') and item.children else ""
                if hasattr(item, 'dynamic_label') and callable(item.dynamic_label):
                    label = item.dynamic_label(is_selected_row)
                else:
                    label = getattr(item, 'label', getattr(item, 'fix_name', str(item)))
                
                line_content = f"{pointer} {indent}{checkbox}{label}{suffix}"
                if is_selected_row:
                    print(f"{Style.SELECTED} {line_content:<60} {Style.ENDC}")
                else:
                    print(f"  {line_content}")
            
            # Bottom Indicator
            if end_row < len(flat_rows):
                hidden_below = sum(1 for r in flat_rows[end_row:] if r['level'] == cur_level)
                msg = f"▼ ({hidden_below}) ▼" if hidden_below > 0 else "▼"
                print(f"  {Style.DIM}{msg}{Style.ENDC}")
            else:
                print(" ")

            # Footer
            selected_item = flat_rows[current_row]['obj']
            selected_description = _get_item_description(selected_item)

            if selected_description:
                print(f"{Style.DIM} Descripción: {selected_description}{Style.ENDC}")

            print("\n" + "─"*65)
            if multi_select:
                # Contar seleccionados totales
                res = []
                def collect(items):
                    for i in items:
                        if i.is_selected: res.append(i)
                        if hasattr(i, 'children'): collect(i.children)
                collect(options)
                print(f" {Style.BOLD}[ENTER]: Confirmar selección ({len(res)} elementos).{Style.ENDC}")
            else:
                print(f" {Style.BOLD}[ENTER]: Seleccionar opción.{Style.ENDC}")
            
            # 4. Input Handling
            key = read_key()
            if key == 'UP':
                if in_sub_nav:
                     # Rotar solo dentro del submenú actual (hijos del mismo padre)
                    current_parent = flat_rows[current_row].get('parent')
                    if current_parent:
                        # Encontrar rango de índices de este padre
                        siblings = [i for i, r in enumerate(flat_rows) if r.get('parent') == current_parent]
                        if siblings:
                            # Encontrar índice relativo actual
                            try:
                                rel_idx = siblings.index(current_row)
                                new_rel_idx = (rel_idx - 1) % len(siblings)
                                current_row = siblings[new_rel_idx]
                            except ValueError:
                                pass # Should not happen
                else:
                    # Navegación nivel superior
                    next_row = (current_row - 1) % len(flat_rows)
                    while flat_rows[next_row]['level'] > 0:
                        next_row = (next_row - 1) % len(flat_rows)
                    current_row = next_row

            elif key == 'DOWN':
                if in_sub_nav:
                    # Rotar solo dentro del submenú actual
                    current_parent = flat_rows[current_row].get('parent')
                    if current_parent:
                        siblings = [i for i, r in enumerate(flat_rows) if r.get('parent') == current_parent]
                        if siblings:
                            rel_idx = siblings.index(current_row)
                            new_rel_idx = (rel_idx + 1) % len(siblings)
                            current_row = siblings[new_rel_idx]
                else:
                    # Navegación nivel superior
                    next_row = (current_row + 1) % len(flat_rows)
                    while flat_rows[next_row]['level'] > 0:
                        next_row = (next_row + 1) % len(flat_rows)
                    current_row = next_row
            
            elif key == 'SPACE' and multi_select:
                row = flat_rows[current_row]
                item = row['obj']
                
                if row['level'] == 0:
                     # Es un padre -> Entrar en Submenú (Si tiene hijos)
                    if hasattr(item, 'children') and item.children:
                        in_sub_nav = True
                        # Buscar primer hijo
                        for i, r in enumerate(flat_rows):
                            if r['level'] == 1 and r.get('parent') == item:
                                current_row = i
                                break
                    else:
                        # Si no tiene hijos, actúa como toggle normal
                        item.is_selected = not item.is_selected
                else:
                    # Es un hijo -> Toggle selección
                    item.is_selected = not item.is_selected
            
            elif key == 'ENTER':
                # ENTER SIEMPRE confirma todo (o selecciona si no es multi)
                row = flat_rows[current_row]
                item = row['obj']
                
                if multi_select:
                    selected_objs = []
                    def collect(items):
                        for i in items:
                            if i.is_selected: selected_objs.append(i)
                            if hasattr(i, 'children'): collect(i.children)
                    collect(options)
                    _persist_cursor(flat_rows, current_row)
                    return selected_objs
                
                # Single select logic usually returns the item
                _persist_cursor(flat_rows, current_row)
                return item
            
            elif key == 'ESC':
                if in_sub_nav:
                    in_sub_nav = False
                    # Volver al padre
                    if flat_rows[current_row]['level'] > 0:
                        parent = flat_rows[current_row].get('parent')
                        if parent:
                            for i, r in enumerate(flat_rows):
                                if r['obj'] == parent:
                                    current_row = i
                                    break
                else:
                    _persist_cursor(flat_rows, current_row)
                    return None
    finally:
        # Show cursor again
        print("\033[?25h", end="")


def smart_fix_menu(issues, report=None):
    """
    Menú gráfico para seleccionar arreglos.
    """
    # 1. Preparar lista única
    seen_funcs = set()
    display_options = []
    
    for issue in issues:
        # Usamos fix_func como clave única de la operación
        func_key = issue.fix_func
        if func_key not in seen_funcs:
            # Asignamos label para el menú (Visual)
            issue.label = issue.description 
            # Asignamos la acción directa para que interactive_menu pueda ejecutarla si se deseara (aunque aquí lo hacemos en batch)
            issue.action = issue.fix_func
            
            display_options.append(issue)
            seen_funcs.add(func_key)
    
    # Header dinámico (Reporte + Lista de errores)
    def draw_header():
        if report:
             print(f"\n{Style.BOLD}--- SYSTEM DIAGNOSTICS (CACHED) ---{Style.ENDC}")
             print_report_table(report)
        print(f"{Style.BOLD}{Style.FAIL} DIAGNOSTIC ISSUES DETECTED ({len(issues)}) {Style.ENDC}")
        # Listar errores brevemente
        for i in issues:
            print(f" {Style.FAIL}●{Style.ENDC} {i.description}")

    # 2. Ejecutar menú
    selected_fixes = interactive_menu(
        display_options, 
        header_func=draw_header,
        multi_select=True,
        info_text="",
        menu_id="smart_fix_menu"
    )
    
    # 3. Procesar
    if selected_fixes:
        clear_screen_ansi()
        print(f"\n{Style.BOLD} APPLYING FIXES... {Style.ENDC}")
        time.sleep(0.5)
        
        # Asegurar que iteramos sobre una lista (compatibilidad linter)
        fixes_list = selected_fixes if isinstance(selected_fixes, list) else [selected_fixes]
        
        # Orden estable para aplicar fixes en el orden seleccionado
        fixes_list = list(fixes_list)

        for fix in fixes_list:
            fix.action() 
        
        log("All fixes applied. Refreshing diagnostics...", "success")
        time.sleep(1.0)
        return True # Indicate rerun needed
    
    return False # Cancelled 


def run_diagnostics_ui():
    """Wrapper UI para ejecutar diagnósticos y mostrar resultados."""
    while True:
        clear_screen()
        print(f"\n{Style.BOLD}--- SYSTEM DIAGNOSTICS & VERIFICATION ---{Style.ENDC}")
        
        report, issues = perform_diagnostics()

        # Print Report Table
        print_report_table(report)
        
        if issues:
            print(f"\n{Style.FAIL}Issues found! Entering Smart Fix Menu...{Style.ENDC}")
            time.sleep(1)
            # Pass report so menu can re-print it
            should_rerun = smart_fix_menu(issues, report)
            if should_rerun:
                continue # Re-run checks and refresh table
            else:
                break # Cancelled by user
        else:
            print(f"\n{Style.OKGREEN}System looks healthy!{Style.ENDC}")
            wait_for_any_key()
            break # Return to main menu

# --- Install & Menus ---

def reinstall_library_menu():
    """Menú manual para forzar reinstalaciones limpias."""
    def header():
        print_banner()
        print(f"{Style.BOLD} REINSTALL LIBRARIES {Style.ENDC}")
        print(" Select libraries to force re-install (clean install).")

    # Definición de librerías para el bloque core (Lightweight)
    # Definición de librerías para el bloque core (Lightweight)
    core_children = [MenuItem(lib, description=f"Reinstala la librería '{lib}'.") for lib in REQUIRED_LIBS]

    # Opciones principales
    opts = [
        MenuItem("Reinstall Core Dependencies", children=core_children, description="Selecciona una o varias dependencias para reinstalar."),
        MenuItem("Reinstall 'uv' Tool", lambda: fix_uv(), description="Reinstala la herramienta uv en el entorno actual.")
    ]
    
    selected = interactive_menu(opts, header_func=header, multi_select=True, menu_id="reinstall_menu")
    
    if selected:

        clear_screen_ansi()
        
        # Agrupar librerías core para instalarlas en un solo comando uv
        selected_core_libs = [s.label for s in selected if s in core_children]
        other_tasks = [s for s in selected if s not in core_children and callable(s.action)]
        
        if selected_core_libs:
            fix_libs(selected_core_libs)
            
        for task in other_tasks:
            log(f"Ejecutando {task.label}...", "step")
            task.action()
        
        # Detectar necesidad de reinicio manual para este menú
        restart_needed = False
        # Si se instalaron librerías core, recomendamos reinicio
        if selected_core_libs:
            restart_needed = True


        if restart_needed:
            restart_program()

        log("Operaciones completadas.", "success")
        wait_for_any_key("Press any key to continue...")

def perform_install(full_reinstall=False):
    """
    Flujo de instalación inicial o reseteo de fábrica.
    Crea el venv y lanza las instalaciones secuenciales.
    """
    force_flags = ""

    if full_reinstall:
        if os.path.exists(".venv"):
            # Check if running inside the venv we are trying to delete
            current_exe = os.path.normpath(sys.executable)
            venv_path = os.path.abspath(".venv")
            
            if venv_path in current_exe:
                log("⚠️  Cannot delete active .venv (File in use).", "warning")
                log("🔄 Switching to 'Force Reinstall' mode (Overwriting libraries)...", "step")
                force_flags = "--force-reinstall"
                time.sleep(2)
            else:
                log("Removing existing environment (.venv)...", "warning")
                try:
                    shutil.rmtree(".venv")
                    log("Environment removed.", "success")
                    time.sleep(1)
                except Exception as e:
                    log(f"Failed to delete .venv: {e}", "error")
                    log("Please delete the folder manually.", "warning")
                    return

    # 1. Project Structure
    create_project_structure(verbose=False)

    # 2. UV Check
    if not check_uv():
        log("uv not found. Attempting to install via pip...", "step")
        if not run_cmd("pip install uv"):
            log("Could not install uv. Please install it manually.", "error")
            return

    # 3. Hardware Strategy
    use_gpu = detect_gpu()
    if not use_gpu:
        log("No NVIDIA GPU detected. Installing in CPU mode.", "warning")
    else:
        log("NVIDIA GPU detected. Using High Performance configuration.", "success")
    
    # 4. Virtual Env
    log("Configuring Virtual Environment...", "step")
    if not os.path.exists(".venv"):
        run_cmd("uv venv .venv --python 3.12")
    else:
        log(".venv exists.", "info")

    # 5. Libraries
    if os.path.exists("pyproject.toml"):
        log("Syncing dependencies from pyproject.toml/uv.lock...", "step")
        sync_flag = " --reinstall" if full_reinstall else ""
        run_cmd(f"uv sync{sync_flag}")
    else:
        log("Installing fallback dependencies (no pyproject.toml found)...", "step")
        light_libs = " ".join(REQUIRED_LIBS)
        run_cmd(f"uv pip install {force_flags} {light_libs}")

def show_menu():
    """Bucle principal del menú."""
    
    def header():
        print_banner()
        print(f"{Style.BOLD} MAIN MENU {Style.ENDC}")

    # Actions wrappers
    def action_diag(): run_diagnostics_ui()
    def action_reinstall(): reinstall_library_menu()
    def action_regen(): 
        create_project_structure(verbose=True)
        wait_for_any_key()
    def action_reset():
        if ask_user("This will delete .venv and reinstall everything. Sure?", "n"):
            perform_install(full_reinstall=True)

    options = [
        MenuItem(" Run System Diagnostics", action_diag, description="Analiza el sistema y ofrece correcciones automáticas."),
        MenuItem(" Tests & Models Manager", run_tests_menu, description="Ejecuta tests, smoke test y gestiona modelos Ollama."), # New Menu
        MenuItem(" Manual Reinstall Menu", action_reinstall, description="Reinstala dependencias concretas manualmente."),
        MenuItem(" Regenerate Folders", action_regen, description="Crea carpetas del proyecto que falten."),
        MenuItem(" Factory Reset", action_reset, description="Reinstala el entorno del proyecto desde cero (confirmación por defecto en NO)."),
        MenuItem(" Exit", lambda: sys.exit(0), description="Cierra la herramienta de gestión.")
    ]

    while True:
        # Single selection menu
        choice = interactive_menu(options, header_func=header, multi_select=False, menu_id="main_menu")
        
        if choice:
            # Si el linter detecta una lista (por ser el retorno posible del menú genérico), 
            # nos aseguramos de tratarlo como un solo objeto.
            item = choice[0] if isinstance(choice, list) else choice
            
            # Execute action
            # We clear before action to give space for logs
            clear_screen_ansi()
            item.action()
            # Loop continues...
        else:
            # ESC or Enter on nothing -> Exit
            clear_screen_ansi()
            print("Goodbye!")
            sys.exit(0)

def main():
    venv_dir = ".venv"
    venv_python = os.path.join(venv_dir, "Scripts", "python.exe") if os.name == 'nt' else os.path.join(venv_dir, "bin", "python")
    
    # Check if we are running inside the venv
    # We compare the current executable path with the expected venv python path
    running_in_venv = False
    try:
        if os.path.exists(venv_python) and os.path.samefile(sys.executable, venv_python):
            running_in_venv = True
    except:
        pass # Handle potential errors in path comparison

    if not running_in_venv:
        # If venv exists, switch to it immediately
        if os.path.exists(venv_python):
            # print(f"🔄 Switching to virtual environment: {venv_python}")
            try:
                subprocess.check_call([venv_python] + sys.argv)
                sys.exit(0)
            except Exception as e:
                print(f"❌ Failed to switch to venv: {e}")
                sys.exit(1)
        
        # If venv does NOT exist (First Run)
        print_banner()
        log("First run detected. Starting setup...", "info")
        perform_install()
        
        # After install, we MUST restart to load the new venv
        if os.path.exists(venv_python):
            log("Restarting in new environment...", "success")
            time.sleep(1)
            subprocess.check_call([venv_python] + sys.argv)
            sys.exit(0)
        else:
            log("Error: .venv was not created properly.", "error")
            sys.exit(1)

    # If we are here, we are running INSIDE the venv (or failed to switch)
    show_menu()

if __name__ == "__main__":
    main()
