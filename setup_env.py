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
    setup_models_ui,
    setup_diagnostics,
    setup_install_flow,
    setup_tests_ui,
    setup_reinstall_ui,
)
from src.utils.menu_kit import UIKit, AppContext
from src.utils.app_config import REQUIRED_LIBS, LIB_IMPORT_MAP, MODELS_REGISTRY
from src.utils.setup_install_flow import detect_gpu, check_uv, create_project_structure
from src.utils.setup_tests_ui import list_test_files
from src.utils.lms_models import get_installed_lms_models

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
# Style ahora vive en src.utils.setup_ui_io — re-exportada aquí
from src.utils.setup_ui_io import Style  # noqa: E402

# DiagnosticIssue ahora vive en src.utils.setup_diagnostics (importada al comienzo)

# --- Menu State Storage ---
MENU_CURSOR_MEMORY: dict[str, int] = {}

_kit = UIKit(
    style=Style,
    os_module=os,
    sys_module=sys,
    shutil_module=shutil,
    msvcrt_module=msvcrt,
    cursor_memory=MENU_CURSOR_MEMORY,
)

# --- Runtime State ---
LMS_SERVER_STARTED_BY_THIS_SESSION = False
_SYS_INFO_CACHE = None


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
        _kit.log("LM Studio server ya estaba activo.", "info")
        LMS_SERVER_STARTED_BY_THIS_SESSION = False
        return True

    _kit.log("LM Studio server no está activo. Iniciando `lms server start`...", "step")
    if not lms_models.start_server():
        _kit.log("No se pudo iniciar `lms server start`.", "error")
        return False

    for _ in range(12):
        time.sleep(0.5)
        running, _ = get_lms_server_status()
        if running:
            LMS_SERVER_STARTED_BY_THIS_SESSION = True
            _kit.log("LM Studio server iniciado automáticamente para esta sesión.", "success")
            return True

    _kit.log("El servidor no quedó en estado RUNNING tras el arranque automático.", "warning")
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
        _kit.log("Deteniendo LM Studio server iniciado por esta sesión...", "step")
        if not lms_models.stop_server():
            raise RuntimeError("stop failed")
        _kit.log("LM Studio server detenido correctamente.", "success")
    except Exception:
        _kit.log("No se pudo detener `lms server stop` automáticamente.", "warning")
    finally:
        LMS_SERVER_STARTED_BY_THIS_SESSION = False


atexit.register(stop_lms_server_if_owned)


# --- UI Flows ---

def manage_models_menu_ui():
    """
    Muestra el menú interactivo para la gestión de modelos de LM Studio.
    """
    return setup_models_ui.manage_models_menu_ui(_kit, _app)

def run_tests_menu():
    """
    Ejecuta el menú para la selección y ejecución de pruebas automatizadas.
    """
    return setup_tests_ui.run_tests_menu(_kit, _app)

def print_banner():
    """
    Imprime el banner principal de la aplicación mostrando información del sistema.
    
    Limpia la pantalla y muestra una tabla con SO, Python, CPU, RAM y GPU.
    """
    _kit.clear()
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

# --- Modular Fixers ---
# Estas funciones son llamadas automáticamente por el Smart Fix Menu

def fix_folders():
    """
    Reparación automática: Regenera carpetas faltantes en el proyecto.
    
    Llama a `create_project_structure` para asegurar que los directorios
    esenciales existen.
    """
    _kit.log("Regenerating folder structure...", "step")
    create_project_structure(verbose=True, log_fn=_kit.log)

def fix_uv():
    """
    Reparación automática: Instala el gestor de paquetes 'uv' usando pip.
    
    Se utiliza como prerrequisito para la instalación rápida de dependencias.
    """
    _kit.log("Installing 'uv' package manager...", "step")
    _kit.run_cmd("pip install uv")

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
            _kit.log("Syncing dependencies from pyproject.toml/uv.lock...", "step")
            _kit.run_cmd("uv sync")
            return

        # Fallback si no existe pyproject
        libs_to_install = REQUIRED_LIBS
    
    # Aseguramos que sea lista
    if isinstance(libs_to_install, str):
        libs_to_install = [libs_to_install]

    libs_str = " ".join(libs_to_install)
    _kit.log(f"Installing libraries: {libs_str}...", "step")
    # Eliminamos --no-deps para asegurar que se instalen las dependencias faltantes (ej: requests)
    _kit.run_cmd(f"uv pip install --force-reinstall {libs_str}")


# ---------------------------------------------------------------------------
# Objetos centrales: _kit (UI) y _app (dominio)
# ---------------------------------------------------------------------------

_app = AppContext(
    REQUIRED_LIBS=REQUIRED_LIBS,
    LIB_IMPORT_MAP=LIB_IMPORT_MAP,
    MODELS_REGISTRY=MODELS_REGISTRY,
    lms_models=lms_models,
    lms_menu_helpers=lms_menu_helpers,
    psutil=psutil,
    os_module=os,
    sys_module=sys,
    subprocess_module=subprocess,
    time_module=time,
    print_banner=print_banner,
    fix_folders=fix_folders,
    fix_uv=fix_uv,
    fix_libs=fix_libs,
    check_lms=check_lms,
    restart_program=restart_program,
    get_installed_lms_models=get_installed_lms_models,
    list_test_files=list_test_files,
    ensure_lms_server_running=ensure_lms_server_running,
    stop_lms_server_if_owned=stop_lms_server_if_owned,
    MENU_CURSOR_MEMORY=MENU_CURSOR_MEMORY,
)


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
    return setup_diagnostics.perform_diagnostics(_kit, _app)


def smart_fix_menu(issues, report=None):
    """
    Muestra un menú interactivo para aplicar soluciones a problemas detectados.
    
    Permite al usuario seleccionar qué problemas corregir automáticamente.
    
    Args:
        issues (list): Lista de objetos `DiagnosticIssue`.
        report (list, optional): Reporte de diagnóstico previo para contexto.
    """
    return setup_diagnostics.smart_fix_menu(_kit, _app, issues, report)


def run_diagnostics_ui():
    """
    Ejecuta el flujo completo de diagnóstico y reparación (UI).
    
    Realiza el diagnóstico, muestra el reporte y, si hay problemas,
    lanza el menú de reparación inteligente.
    """
    return setup_diagnostics.run_diagnostics_ui(_kit, _app)

# --- Install & Menus ---

def reinstall_library_menu():
    """
    Muestra un menú para reinstalar librerías específicas manualmente.
    
    Delega al módulo `setup_reinstall_ui`.
    """
    return setup_reinstall_ui.reinstall_library_menu(_kit, _app)

def perform_install(full_reinstall=False):
    """
    Ejecuta el flujo de instalación de dependencias del proyecto.
    
    Verifica prerrequisitos (uv, carpetas) e instala las librerías necesarias.
    Puede realizar una reinstalación completa si se solicita.
    
    Args:
        full_reinstall (bool, optional): Si es True, fuerza la reinstalación
            de todo el entorno. Por defecto es False.
    """
    return setup_install_flow.perform_install(_kit, _app, full_reinstall=full_reinstall)

def show_menu():
    """
    Muestra el menú principal de la aplicación de configuración.
    
    Punto de entrada para todas las funcionalidades (test, install, diagnose, etc.).
    Delega al módulo `setup_install_flow`.
    """
    return setup_install_flow.show_menu(_kit, _app)

def main():
    """
    Función principal del script setup_env.py.
    
    Inicializa el contexto y lanza el menú principal.
    """
    return setup_install_flow.main(_kit, _app)

if __name__ == "__main__":
    main()
