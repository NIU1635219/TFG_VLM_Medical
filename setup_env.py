import subprocess
import sys
import os
import platform
import shutil
import time
import atexit
from src.utils import (
    lms_models,
    lms_menu_helpers,
    setup_ui_io,
    setup_menu_engine,
    setup_models_ui,
    setup_diagnostics,
    setup_install_flow,
    setup_tests_ui,
    setup_reinstall_ui,
)

# --- Configuration: Model Registry (solo para descargas/pull) ---
# Añade aquí nuevos modelos siguiendo el formato.
MODELS_REGISTRY = {
    "minicpm_v_2_6_8b": {
        "name": "https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf",
        "description": "MiniCPM-V 2.6 (8B) - Versión Estable Compatible",
    },
    "minicpm_v_4_5_8b": {
        "name": "https://huggingface.co/openbmb/MiniCPM-V-4_5-gguf",
        "description": "MiniCPM-V 4.5 (8B) - SOTA OpenBMB",
    },
    "qwen3_vl_8b": {
        "name": "https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF",
        "description": "Qwen3-VL 8B (SOTA Razonamiento 2026)"
    },
    "internvl3_5_8b": {
        "name": "https://huggingface.co/bartowski/OpenGVLab_InternVL3_5-8B-GGUF",
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
MenuItem = setup_menu_engine.MenuItem


# --- Configuration: Dependencies ---
# Single source of truth for required python libraries
REQUIRED_LIBS = [
    "lmstudio",
    "pydantic",
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
    "lmstudio": "lmstudio",
    "pillow": "PIL",
    "opencv-python": "cv2",
    "pyyaml": "yaml",
    "pytest-mock": "pytest_mock"
}

LMS_SERVER_STARTED_BY_THIS_SESSION = False
_SYS_INFO_CACHE = None

# --- Helper Functions ---

def clear_screen_ansi():
    """
    Limpia pantalla usando CLS en Windows para asegurar limpieza.
    """
    setup_ui_io.clear_screen_ansi(os_module=os, sys_module=sys)

def clear_screen():
    """Wrapper compatible."""
    clear_screen_ansi()

def read_key():
    """Lee una tecla y devuelve un código unificado (UP, DOWN, ENTER, SPACE, ESC)."""
    return setup_ui_io.read_key(os_module=os, msvcrt_module=msvcrt)

def log(msg, level="info"):
    """
    Imprime mensajes con formato y color.
    level: info, success, error, warning, step
    """
    setup_ui_io.log(style=Style, msg=msg, level=level)

def ask_user(question, default="y"):
    """Pide confirmación con interfaz visual (flechas + ENTER, ESC cancela)."""
    return setup_ui_io.ask_user(
        question=question,
        default=default,
        style=Style,
        read_key_fn=read_key,
        clear_screen_fn=clear_screen_ansi,
    )


def input_with_esc(prompt):
    """Entrada de texto que permite cancelar con ESC en Windows. Retorna None si se cancela."""
    return setup_ui_io.input_with_esc(prompt=prompt, os_module=os, msvcrt_module=msvcrt)


def wait_for_any_key(message="Press any key to return..."):
    """Pausa breve: espera cualquier tecla y retorna al menú anterior."""
    setup_ui_io.wait_for_any_key(
        message=message,
        style=Style,
        os_module=os,
        msvcrt_module=msvcrt,
    )

def run_cmd(cmd, critical=True):
    """
    Ejecuta un comando de shell e imprime el comando en gris.
    Si falla y critical=True, pregunta al usuario si reintentar.
    """
    return setup_ui_io.run_cmd(
        cmd=cmd,
        critical=critical,
        style=Style,
        ask_user_fn=ask_user,
        log_fn=log,
    )

def get_sys_info(refresh=False):
    """Recopila información básica del hardware (OS, Python, CPU, GPU, RAM)."""
    global _SYS_INFO_CACHE

    if _SYS_INFO_CACHE is not None and not refresh:
        return dict(_SYS_INFO_CACHE)

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

    _SYS_INFO_CACHE = dict(info)
    return dict(info)

def check_lms():
    """Verifica si 'lms' (LM Studio CLI) está instalado y accesible."""
    return lms_models.check_lms()


def get_lms_server_status():
    """Obtiene estado del servidor LM Studio vía CLI (`lms server status`)."""
    return lms_models.get_server_status()


def ensure_lms_server_running():
    """Arranca `lms server` si no está activo. Registra ownership de esta sesión."""
    global LMS_SERVER_STARTED_BY_THIS_SESSION

    is_running, detail = get_lms_server_status()

    if is_running:
        log("LM Studio server ya estaba activo.", "info")
        LMS_SERVER_STARTED_BY_THIS_SESSION = False
        return True

    log("LM Studio server no está activo. Iniciando `lms server start`...", "step")
    if not lms_models.start_server():
        log("No se pudo iniciar `lms server start`.", "error")
        return False

    for _ in range(12):
        time.sleep(0.5)
        running, _ = get_lms_server_status()
        if running:
            LMS_SERVER_STARTED_BY_THIS_SESSION = True
            log("LM Studio server iniciado automáticamente para esta sesión.", "success")
            return True

    log("El servidor no quedó en estado RUNNING tras el arranque automático.", "warning")
    return False


def stop_lms_server_if_owned():
    """Detiene `lms server` solo si fue iniciado por esta sesión."""
    global LMS_SERVER_STARTED_BY_THIS_SESSION

    if not LMS_SERVER_STARTED_BY_THIS_SESSION:
        return

    try:
        log("Deteniendo LM Studio server iniciado por esta sesión...", "step")
        if not lms_models.stop_server():
            raise RuntimeError("stop failed")
        log("LM Studio server detenido correctamente.", "success")
    except Exception:
        log("No se pudo detener `lms server stop` automáticamente.", "warning")
    finally:
        LMS_SERVER_STARTED_BY_THIS_SESSION = False


atexit.register(stop_lms_server_if_owned)
def get_installed_lms_models():
    """Obtiene modelos locales LLM (SDK primero, CLI como fallback)."""
    local_models = lms_models.list_local_llm_models()
    if local_models:
        return [str(item.get("model_key", "")).strip() for item in local_models if item.get("model_key")]
    return lms_models.get_installed_models()

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
    """Menú de gestión de modelos LM Studio (delegado a módulo UI)."""
    return setup_models_ui.manage_models_menu_ui(
        {
            "Style": Style,
            "MenuItem": MenuItem,
            "MenuStaticItem": setup_menu_engine.MenuStaticItem,
            "MenuSeparator": setup_menu_engine.MenuSeparator,
            "IncrementalPanelRenderer": setup_ui_io.IncrementalPanelRenderer,
            "lms_models": lms_models,
            "lms_menu_helpers": lms_menu_helpers,
            "psutil": psutil,
            "MODELS_REGISTRY": MODELS_REGISTRY,
            "MENU_CURSOR_MEMORY": MENU_CURSOR_MEMORY,
            "print_banner": print_banner,
            "clear_screen_ansi": clear_screen_ansi,
            "input_with_esc": input_with_esc,
            "wait_for_any_key": wait_for_any_key,
            "log": log,
            "ask_user": ask_user,
            "interactive_menu": interactive_menu,
            "get_installed_lms_models": get_installed_lms_models,
            "read_key": read_key,
        }
    )

def run_tests_menu():
    """Menú para ejecutar tests (delegado a módulo UI)."""
    return setup_tests_ui.run_tests_menu(
        ctx={
            "Style": Style,
            "MenuItem": MenuItem,
            "print_banner": print_banner,
            "log": log,
            "run_cmd": run_cmd,
            "wait_for_any_key": wait_for_any_key,
            "interactive_menu": interactive_menu,
            "list_test_files": list_test_files,
            "get_installed_lms_models": get_installed_lms_models,
            "clear_screen_ansi": clear_screen_ansi,
            "manage_models_menu_ui": manage_models_menu_ui,
            "time": time,
        }
    )

def print_banner():
    """Muestra el banner principal con el estado del sistema."""
    clear_screen()
    info = get_sys_info()
    box_inner = 65
    label_width = 7

    def truncate(text, width):
        value = str(text)
        if len(value) <= width:
            return value
        if width <= 1:
            return value[:width]
        return value[: width - 1] + "…"

    def line(content=""):
        text = truncate(content, box_inner)
        return f"║{text:<{box_inner}}║"

    def metric_row(label, value):
        value_width = box_inner - (1 + label_width + 3)
        value_text = truncate(value, value_width)
        return line(f" {label:<{label_width}} │ {value_text}")

    title = "TFG VLM Medical · Manager Tool v5.0"
    print(f"{Style.HEADER}{Style.BOLD}")
    print("╔" + "═" * box_inner + "╗")
    print(line(title.center(box_inner)))
    print("╠" + "═" * box_inner + "╣")
    print(metric_row("OS", info["os"]))
    print(metric_row("Python", info["python"]))
    print(metric_row("CPU", f"{info['cpu_cores']} Cores"))
    print(metric_row("RAM", info["ram"]))
    print(metric_row("GPU", info["gpu"]))
    print("╚" + "═" * box_inner + "╝")
    print(Style.ENDC)

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
    return setup_diagnostics.perform_diagnostics(
        {
            "DiagnosticIssue": DiagnosticIssue,
            "fix_folders": fix_folders,
            "fix_uv": fix_uv,
            "fix_libs": fix_libs,
            "check_uv": check_uv,
            "check_lms": check_lms,
            "log": log,
            "REQUIRED_LIBS": REQUIRED_LIBS,
            "LIB_IMPORT_MAP": LIB_IMPORT_MAP,
        }
    )


def print_report_table(report):
    """Imprime la tabla de diagnóstico de forma formateada."""
    return setup_diagnostics.print_report_table(report, style=Style)


def interactive_menu(
    options,
    header_func=None,
    multi_select=False,
    info_text="",
    menu_id=None,
    nav_hint=True,
    left_margin=0,
    nav_hint_text=None,
    sub_nav_hint_text=None,
    footer_hint_text=None,
    repaint_strategy="auto",
):
    """
    Menú interactivo genérico controlado por teclado.
    Soporta circularidad y navegación por sub-niveles.
    """
    return setup_menu_engine.interactive_menu(
        options,
        style=Style,
        clear_screen_fn=clear_screen_ansi,
        read_key_fn=read_key,
        get_item_description_fn=_get_item_description,
        cursor_memory=MENU_CURSOR_MEMORY,
        os_module=os,
        sys_module=sys,
        shutil_module=shutil,
        header_func=header_func,
        multi_select=multi_select,
        info_text=info_text,
        menu_id=menu_id,
        nav_hint=nav_hint,
        left_margin=left_margin,
        nav_hint_text=nav_hint_text,
        sub_nav_hint_text=sub_nav_hint_text,
        footer_hint_text=footer_hint_text,
        repaint_strategy=repaint_strategy,
    )


def smart_fix_menu(issues, report=None):
    """
    Menú gráfico para seleccionar arreglos.
    """
    return setup_diagnostics.smart_fix_menu(
        issues,
        report,
        {
            "Style": Style,
            "interactive_menu": interactive_menu,
            "print_report_table": print_report_table,
            "clear_screen_ansi": clear_screen_ansi,
            "log": log,
        },
    )


def run_diagnostics_ui():
    """Wrapper UI para ejecutar diagnósticos y mostrar resultados."""
    return setup_diagnostics.run_diagnostics_ui(
        {
            "Style": Style,
            "clear_screen": clear_screen,
            "wait_for_any_key": wait_for_any_key,
            "perform_diagnostics": perform_diagnostics,
            "print_report_table": print_report_table,
            "smart_fix_menu": smart_fix_menu,
        }
    )

# --- Install & Menus ---

def reinstall_library_menu():
    """Menú manual para forzar reinstalaciones limpias (delegado)."""
    return setup_reinstall_ui.reinstall_library_menu(
        ctx={
            "Style": Style,
            "MenuItem": MenuItem,
            "REQUIRED_LIBS": REQUIRED_LIBS,
            "print_banner": print_banner,
            "interactive_menu": interactive_menu,
            "clear_screen_ansi": clear_screen_ansi,
            "fix_uv": fix_uv,
            "fix_libs": fix_libs,
            "log": log,
            "restart_program": restart_program,
            "wait_for_any_key": wait_for_any_key,
        }
    )

def perform_install(full_reinstall=False):
    """
    Flujo de instalación inicial o reseteo de fábrica.
    Crea el venv y lanza las instalaciones secuenciales.
    """
    return setup_install_flow.perform_install(
        ctx={
            "log": log,
            "create_project_structure": create_project_structure,
            "check_uv": check_uv,
            "run_cmd": run_cmd,
            "detect_gpu": detect_gpu,
            "REQUIRED_LIBS": REQUIRED_LIBS,
        },
        full_reinstall=full_reinstall,
    )

def show_menu():
    """Bucle principal del menú."""
    return setup_install_flow.show_menu(
        ctx={
            "MenuItem": MenuItem,
            "Style": Style,
            "print_banner": print_banner,
            "run_diagnostics_ui": run_diagnostics_ui,
            "run_tests_menu": run_tests_menu,
            "reinstall_library_menu": reinstall_library_menu,
            "create_project_structure": create_project_structure,
            "wait_for_any_key": wait_for_any_key,
            "ask_user": ask_user,
            "perform_install": perform_install,
            "interactive_menu": interactive_menu,
            "clear_screen_ansi": clear_screen_ansi,
        }
    )

def main():
    return setup_install_flow.main(
        ctx={
            "os": os,
            "sys": sys,
            "subprocess": subprocess,
            "time": time,
            "print_banner": print_banner,
            "log": log,
            "perform_install": perform_install,
            "ensure_lms_server_running": ensure_lms_server_running,
            "show_menu": show_menu,
            "stop_lms_server_if_owned": stop_lms_server_if_owned,
        }
    )

if __name__ == "__main__":
    main()
