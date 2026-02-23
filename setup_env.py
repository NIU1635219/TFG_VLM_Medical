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
    """
    Clase que define códigos de escape ANSI para dar estilo al texto en la terminal.
    
    Proporciona constantes para colores (azul, cian, verde, amarillo, rojo),
    estilos de texto (negrita, atenuado) y colores de fondo para selecciones.
    """
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
    """
    Representa un problema detectado durante el diagnóstico del entorno.
    
    Almacena la descripción del problema, la función que lo soluciona y el nombre
    de la solución para mostrar en la interfaz de usuario.
    """
    def __init__(self, description, fix_func, fix_name):
        """
        Inicializa una nueva instancia de DiagnosticIssue.
        
        Args:
            description (str): Descripción detallada del problema detectado.
            fix_func (callable): Función que se ejecutará para intentar solucionar el problema.
            fix_name (str): Nombre corto o etiqueta de la solución para mostrar en el menú.
        """
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
    "numpy",
    "rarfile",
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
    Limpia la pantalla de la terminal usando secuencias ANSI o comandos del sistema.
    
    Utiliza el módulo `os` para ejecutar el comando de limpieza adecuado según
    el sistema operativo (cls para Windows, clear para Unix).
    """
    setup_ui_io.clear_screen_ansi(os_module=os, sys_module=sys)

def clear_screen():
    """
    Wrapper compatible para limpiar la pantalla.
    
    Llama a `clear_screen_ansi()` para mantener la compatibilidad con código existente.
    """
    clear_screen_ansi()

def read_key():
    """
    Lee una tecla presionada por el usuario y devuelve un código unificado.
    
    Returns:
        str: Un código de tecla unificado ('UP', 'DOWN', 'ENTER', 'SPACE', 'ESC', o el carácter).
    """
    return setup_ui_io.read_key(os_module=os, msvcrt_module=msvcrt)

def log(msg, level="info"):
    """
    Imprime mensajes en la terminal con formato y color según el nivel.
    
    Args:
        msg (str): El mensaje a imprimir.
        level (str, optional): El nivel del mensaje ('info', 'success', 'error', 'warning', 'step').
            Por defecto es 'info'.
    """
    setup_ui_io.log(style=Style, msg=msg, level=level)

def ask_user(question, default="y", info_text=""):
    """
    Pide confirmación al usuario con una interfaz visual interactiva.
    
    Permite al usuario seleccionar 'Yes' o 'No' usando las flechas del teclado
    y confirmar con ENTER, o cancelar con ESC.
    
    Args:
        question (str): La pregunta a mostrar al usuario.
        default (str, optional): La opción por defecto ('y' o 'n'). Por defecto es 'y'.
        
    Returns:
        bool: True si el usuario selecciona 'Yes', False si selecciona 'No' o cancela.
    """
    return setup_ui_io.ask_user(
        question=question,
        default=default,
        style=Style,
        read_key_fn=read_key,
        clear_screen_fn=clear_screen_ansi,
        info_text=info_text,
    )


def input_with_esc(prompt):
    """
    Solicita entrada de texto al usuario, permitiendo cancelar con la tecla ESC.
    
    Args:
        prompt (str): El texto a mostrar antes de la entrada.
        
    Returns:
        str or None: El texto introducido por el usuario, o None si se canceló con ESC.
    """
    return setup_ui_io.input_with_esc(prompt=prompt, os_module=os, msvcrt_module=msvcrt)


def wait_for_any_key(message="Press any key to return..."):
    """
    Pausa la ejecución y espera a que el usuario presione cualquier tecla.
    
    Args:
        message (str, optional): El mensaje a mostrar durante la pausa.
            Por defecto es "Press any key to return...".
    """
    setup_ui_io.wait_for_any_key(
        message=message,
        style=Style,
        os_module=os,
        msvcrt_module=msvcrt,
    )

def run_cmd(cmd, critical=True):
    """
    Ejecuta un comando de shell e imprime el comando en la terminal.
    
    Si el comando falla y `critical` es True, pregunta al usuario si desea reintentar.
    
    Args:
        cmd (str): El comando de shell a ejecutar.
        critical (bool, optional): Si es True, permite reintentar en caso de fallo.
            Por defecto es True.
            
    Returns:
        bool: True si el comando se ejecutó con éxito, False en caso contrario.
    """
    return setup_ui_io.run_cmd(
        cmd=cmd,
        critical=critical,
        style=Style,
        ask_user_fn=ask_user,
        log_fn=log,
    )

def get_sys_info(refresh=False):
    """
    Recopila información básica del hardware y sistema operativo.
    
    Obtiene detalles sobre el SO, versión de Python, núcleos de CPU,
    modelo de GPU (si está disponible vía nvidia-smi) y memoria RAM total.
    
    Args:
        refresh (bool, optional): Si es True, fuerza la recolección de datos
            ignorando la caché. Por defecto es False.
            
    Returns:
        dict: Un diccionario con la información del sistema ('os', 'python',
            'cpu_cores', 'gpu', 'ram').
    """
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
    """
    Verifica si la interfaz de línea de comandos de LM Studio ('lms') está instalada.
    
    Returns:
        bool: True si 'lms' está disponible en el PATH, False en caso contrario.
    """
    return lms_models.check_lms()


def get_lms_server_status():
    """
    Obtiene el estado actual del servidor local de LM Studio.
    
    Ejecuta el comando `lms server status` para comprobar si el servidor
    está en ejecución.
    
    Returns:
        str: El estado del servidor (ej. 'running', 'stopped', 'error').
    """
    return lms_models.get_server_status()


def ensure_lms_server_running():
    """
    Asegura que el servidor de LM Studio esté en ejecución.
    
    Si el servidor no está activo, intenta iniciarlo. Si lo inicia con éxito,
    registra que esta sesión es la propietaria del servidor para poder
    detenerlo al salir.
    """
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
    """
    Detiene el servidor de LM Studio solo si fue iniciado por esta sesión.
    
    Verifica la variable global `LMS_SERVER_STARTED_BY_THIS_SESSION`. Si es True,
    intenta detener el servidor y registra el resultado.
    """
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
    """
    Obtiene la lista de identificadores de los modelos LM Studio instalados localmente.
    
    Intenta primero usar la API del SDK local y, como fallback, usa la CLI.
    
    Returns:
        list: Lista de strings con los keys/IDs de los modelos instalados.
    """
    local_models = lms_models.list_local_llm_models()
    if local_models:
        return [str(item.get("model_key", "")).strip() for item in local_models if item.get("model_key")]
    return lms_models.get_installed_models()

def list_test_files():
    """
    Escanea la carpeta `tests/` en busca de archivos de prueba.
    
    Busca archivos que comiencen con 'test_' y terminen en '.py'.
    
    Returns:
        list: Lista ordenada alfabéticamente de los nombres de archivos encontrados.
    """
    test_dir = "tests"
    if not os.path.exists(test_dir):
        return []
    
    files = [f for f in os.listdir(test_dir) if f.startswith("test_") and f.endswith(".py")]
    return sorted(files)


def _get_item_description(item):
    """
    Obtiene la descripción de un item de menú, si existe.
    
    Args:
        item: Objeto que puede tener un atributo 'description'.
             
    Returns:
        str: La descripción del item o una cadena vacía si no existe.
    """
    description = getattr(item, "description", "")
    if not description:
        return ""
    return str(description).strip()

def manage_models_menu_ui():
    """
    Muestra el menú interactivo para la gestión de modelos de LM Studio.
    
    Delega la lógica al módulo `setup_models_ui`, pasando el contexto necesario.
    """
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
            "os_module": os,
            "msvcrt_module": msvcrt,
        }
    )

def run_tests_menu():
    """
    Ejecuta el menú para la selección y ejecución de pruebas automatizadas.
    
    Delega la lógica al módulo `setup_tests_ui`.
    """
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
    """
    Imprime el banner principal de la aplicación mostrando información del sistema.
    
    Limpia la pantalla y muestra una tabla con SO, Python, CPU, RAM y GPU.
    """
    clear_screen()
    info = get_sys_info()
    box_inner = setup_ui_io.get_full_ui_width(shutil_module=shutil)
    label_width = 7

    def truncate(text, width):
        """Trunca el texto para ajustarlo al ancho especificado con elipsis."""
        value = str(text)
        if len(value) <= width:
            return value
        if width <= 1:
            return value[:width]
        return value[: width - 1] + "…"

    def line(content=""):
        """Formatea una línea de contenido para que se ajuste al recuadro."""
        text = truncate(content, box_inner)
        return f"║{text:<{box_inner}}║"

    def metric_row(label, value):
        """Genera una fila de métricas con etiqueta y valor."""
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
    """
    Reinicia el programa actual completamente.
    
    Útil cuando se instalan nuevas dependencias y es necesario recargar el entorno.
    Muestra un aviso, espera 2 segundos y reinicia el proceso usando `subprocess`.
    """
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
    """
    Detecta si el sistema tiene una GPU NVIDIA disponible.
    
    Ejecuta `nvidia-smi` y devuelve True si el comando tiene éxito.
    
    Returns:
        bool: True si se detecta GPU NVIDIA, False en caso contrario.
    """
    try:
        subprocess.check_call("nvidia-smi", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except:
        return False

def check_uv():
    """
    Verifica si el gestor de paquetes 'uv' está instalado.
    
    Returns:
        bool: True si 'uv --version' se ejecuta correctamente.
    """
    try:
        subprocess.check_call("uv --version", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except:
        return False

def create_project_structure(verbose=True):
    """
    Crea la estructura de carpetas necesaria para el proyecto.
    
    Verifica la existencia de directorios clave (data, src, notebooks) y
    los crea si faltan.
    
    Args:
        verbose (bool, optional): Si es True, imprime mensajes de progreso.
    """
    folders = [
        "data/raw",
        "data/processed",
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
    """
    Reparación automática: Regenera carpetas faltantes en el proyecto.
    
    Llama a `create_project_structure` para asegurar que los directorios
    esenciales existen.
    """
    log("Regenerating folder structure...", "step")
    create_project_structure(verbose=True)

def fix_uv():
    """
    Reparación automática: Instala el gestor de paquetes 'uv' usando pip.
    
    Se utiliza como prerrequisito para la instalación rápida de dependencias.
    """
    log("Installing 'uv' package manager...", "step")
    run_cmd("pip install uv")

def fix_libs(libs_to_install=None):
    """
    Reparación automática: Sincroniza o reinstala las dependencias del proyecto.
    
    Si `libs_to_install` es None y existe un archivo de proyecto, usa `uv sync`.
    De lo contrario, reinstala las librerías especificadas o las por defecto.
    
    Args:
        libs_to_install (list or str, optional): Librería(s) específica(s) a instalar.
            Si es None, intenta instalar todas las requeridas.
    """
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
    Ejecuta una serie de comprobaciones para diagnosticar el estado del entorno.
    
    Verifica la estructura de carpetas, la instalación de 'uv', las librerías
    requeridas y el estado del servidor LM Studio.
    
    Returns:
        tuple: Una tupla conteniendo:
            - report (list): Lista de resultados para mostrar en tabla UI.
            - issues (list): Lista de objetos `DiagnosticIssue` detectados.
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
    """
    Imprime una tabla formateada con los resultados del diagnóstico.
    
    Args:
        report (list): Lista de tuplas (Nombre, Estado, Color) generada por `perform_diagnostics`.
    """
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
    dynamic_info_top=False,
):
    """
    Muestra un menú interactivo controlado por teclado.
    
    Soporta navegación con flechas, selección múltiple, secciones plegables (si
    está soportado) y renderizado personalizado.
    
    Args:
        options (list): Lista de objetos `MenuItem` o separadores.
        header_func (callable, optional): Función que imprime un encabezado antes del menú.
        multi_select (bool, optional): Permite seleccionar múltiples opciones.
        ... (otros parámetros de configuración visual y comportamiento)
        
    Returns:
        object: El item seleccionado (o lista de items si multi_select=True),
        o None si se cancela.
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
        dynamic_info_top=dynamic_info_top,
    )


def smart_fix_menu(issues, report=None):
    """
    Muestra un menú interactivo para aplicar soluciones a problemas detectados.
    
    Permite al usuario seleccionar qué problemas corregir automáticamente.
    
    Args:
        issues (list): Lista de objetos `DiagnosticIssue`.
        report (list, optional): Reporte de diagnóstico previo para contexto.
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
    """
    Ejecuta el flujo completo de diagnóstico y reparación (UI).
    
    Realiza el diagnóstico, muestra el reporte y, si hay problemas,
    lanza el menú de reparación inteligente.
    """
    return setup_diagnostics.run_diagnostics_ui(
        {
            "Style": Style,
            "clear_screen": clear_screen,
            "wait_for_any_key": wait_for_any_key,
            "read_key": read_key,
            "perform_diagnostics": perform_diagnostics,
            "print_report_table": print_report_table,
            "smart_fix_menu": smart_fix_menu,
        }
    )

# --- Install & Menus ---

def reinstall_library_menu():
    """
    Muestra un menú para reinstalar librerías específicas manualmente.
    
    Delega al módulo `setup_reinstall_ui`.
    """
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
    Ejecuta el flujo de instalación de dependencias del proyecto.
    
    Verifica prerrequisitos (uv, carpetas) e instala las librerías necesarias.
    Puede realizar una reinstalación completa si se solicita.
    
    Args:
        full_reinstall (bool, optional): Si es True, fuerza la reinstalación
            de todo el entorno. Por defecto es False.
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
    """
    Muestra el menú principal de la aplicación de configuración.
    
    Punto de entrada para todas las funcionalidades (test, install, diagnose, etc.).
    Delega al módulo `setup_install_flow`.
    """
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
    """
    Función principal del script setup_env.py.
    
    Inicializa el contexto y lanza el menú principal.
    """
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
