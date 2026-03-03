"""Flujo de instalación/bootstrap y menú principal para setup_env."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .menu_kit import AppContext, UIKit


# ---------------------------------------------------------------------------
# Utilidades de sistema puras (sin dependencias de UI)
# ---------------------------------------------------------------------------


def detect_gpu() -> bool:
    """Detecta si hay una GPU NVIDIA disponible en el sistema.

    Returns:
        bool: ``True`` si ``nvidia-smi`` responde correctamente.
    """
    try:
        subprocess.check_call(
            "nvidia-smi",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def check_uv() -> bool:
    """Verifica si el gestor de paquetes ``uv`` está instalado.

    Returns:
        bool: ``True`` si ``uv --version`` se ejecuta correctamente.
    """
    try:
        subprocess.check_call(
            "uv --version",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def create_project_structure(verbose: bool = True, *, log_fn=None) -> None:
    """Crea la estructura de carpetas necesaria para el proyecto.

    Args:
        verbose (bool): Si ``True``, emite mensajes de progreso via ``log_fn``.
        log_fn: Callable ``(msg, level)`` para logging. Si es ``None`` los
            mensajes se descartan silenciosamente.
    """
    _log = log_fn if log_fn is not None else (lambda *a, **k: None)

    folders = [
        "data/raw",
        "data/processed",
        "src/preprocessing",
        "src/inference",
        "notebooks",
    ]

    created = 0
    for folder in folders:
        if not os.path.exists(folder):
            try:
                os.makedirs(folder, exist_ok=True)
                if verbose:
                    _log(f"Created missing directory: {folder}", "success")
                created += 1
            except Exception as e:
                _log(f"Failed to create {folder}: {e}", "error")

    if verbose and created == 0:
        _log("All directories verified.", "success")


# ---------------------------------------------------------------------------
# Flujo de instalación
# ---------------------------------------------------------------------------

def perform_install(kit: "UIKit", app: "AppContext", *, full_reinstall: bool = False) -> None:
    """
    Ejecuta el flujo de instalación inicial o reinstalación completa del entorno.

    Gestiona la creación de ``.venv``, la instalación de ``uv`` y la
    sincronización de dependencias base.

    Args:
        kit (UIKit): Interfaz de UI de terminal.
        app (AppContext): Contexto de dominio.
        full_reinstall (bool): Si ``True``, elimina el entorno virtual existente antes.
    """
    force_flags = ""

    if full_reinstall and os.path.exists(".venv"):
        current_exe = os.path.normpath(sys.executable)
        venv_path = os.path.abspath(".venv")

        if venv_path in current_exe:
            kit.log("⚠️  Cannot delete active .venv (File in use).", "warning")
            kit.log("🔄 Switching to 'Force Reinstall' mode (Overwriting libraries)...", "step")
            force_flags = "--force-reinstall"
            time.sleep(2)
        else:
            kit.log("Removing existing environment (.venv)...", "warning")
            try:
                import shutil as _shutil
                _shutil.rmtree(".venv")
                kit.log("Environment removed.", "success")
                time.sleep(1)
            except Exception as error:
                kit.log(f"Failed to delete .venv: {error}", "error")
                kit.log("Please delete the folder manually.", "warning")
                return

    create_project_structure(verbose=False)

    if not check_uv():
        kit.log("uv not found. Attempting to install via pip...", "step")
        if not kit.run_cmd("pip install uv"):
            kit.log("Could not install uv. Please install it manually.", "error")
            return

    use_gpu = detect_gpu()
    if not use_gpu:
        kit.log("No NVIDIA GPU detected. Installing in CPU mode.", "warning")
    else:
        kit.log("NVIDIA GPU detected. Using High Performance configuration.", "success")

    kit.log("Configuring Virtual Environment...", "step")
    if not os.path.exists(".venv"):
        kit.run_cmd("uv venv .venv --python 3.12")
    else:
        kit.log(".venv exists.", "info")

    if os.path.exists("pyproject.toml"):
        kit.log("Syncing dependencies from pyproject.toml/uv.lock...", "step")
        sync_flag = " --reinstall" if full_reinstall else ""
        kit.run_cmd(f"uv sync{sync_flag}")
    else:
        kit.log("Installing fallback dependencies (no pyproject.toml found)...", "step")
        light_libs = " ".join(app.REQUIRED_LIBS)
        kit.run_cmd(f"uv pip install {force_flags} {light_libs}")


def show_menu(kit: "UIKit", app: "AppContext") -> None:
    """
    Despliega el menú principal interactivo de la herramienta de setup.

    Args:
        kit (UIKit): Interfaz de UI de terminal.
        app (AppContext): Contexto de dominio.
    """
    from . import setup_diagnostics, setup_reinstall_ui, setup_tests_ui

    def header() -> None:
        app.print_banner()
        kit.subtitle("MAIN MENU")

    def action_diag() -> None:
        setup_diagnostics.run_diagnostics_ui(kit, app)

    def action_reinstall() -> None:
        setup_reinstall_ui.reinstall_library_menu(kit, app)

    def action_tests() -> None:
        setup_tests_ui.run_tests_menu(kit, app)

    def action_regen() -> None:
        create_project_structure(verbose=True, log_fn=kit.log)
        kit.wait()

    def action_reset() -> None:
        if kit.ask("This will delete .venv and reinstall everything. Sure?", "n"):
            perform_install(kit, app, full_reinstall=True)

    options = [
        kit.MenuItem(" Run System Diagnostics", action_diag, description="Analiza el sistema y ofrece correcciones automáticas."),
        kit.MenuItem(" Tests & Models Manager", action_tests, description="Ejecuta tests, smoke test y gestiona modelos LM Studio."),
        kit.MenuItem(" Manual Reinstall Menu", action_reinstall, description="Reinstala dependencias concretas manualmente."),
        kit.MenuItem(" Regenerate Folders", action_regen, description="Crea carpetas del proyecto que falten."),
        kit.MenuItem(" Factory Reset", action_reset, description="Reinstala el entorno del proyecto desde cero (confirmación por defecto en NO)."),
        kit.MenuItem(" Exit", lambda: sys.exit(0), description="Cierra la herramienta de gestión."),
    ]

    while True:
        choice = kit.menu(
            options,
            header_func=header,
            multi_select=False,
            menu_id="main_menu",
            nav_hint_text="↑/↓ navegar opciones · ENTER ejecutar acción · ESC salir",
        )

        if choice:
            item = choice[0] if isinstance(choice, list) else choice
            kit.clear()
            item.action()
        else:
            kit.clear()
            print("Goodbye!")
            sys.exit(0)


def main(kit: "UIKit", app: "AppContext") -> None:
    """
    Punto de entrada principal del script de configuración.

    1. Detecta si corre dentro del ``.venv``.
    2. Si no, crea el entorno, instala dependencias y se relanza dentro de él.
    3. Una vez en ``.venv``, arranca el servidor LM Studio si es necesario.
    4. Muestra el menú principal.

    Args:
        kit (UIKit): Interfaz de UI de terminal.
        app (AppContext): Contexto de dominio.
    """
    os_module = app.os_module
    sys_module = app.sys_module
    subprocess_module = app.subprocess_module
    time_module = app.time_module

    venv_dir = ".venv"
    if os_module.name == "nt":
        venv_python = os_module.path.join(venv_dir, "Scripts", "python.exe")
    else:
        venv_python = os_module.path.join(venv_dir, "bin", "python")

    running_in_venv = False
    try:
        if os_module.path.exists(venv_python) and os_module.path.samefile(sys_module.executable, venv_python):
            running_in_venv = True
    except Exception:
        pass

    if not running_in_venv:
        if os_module.path.exists(venv_python):
            try:
                subprocess_module.check_call([venv_python] + sys_module.argv)
                sys_module.exit(0)
            except Exception as error:
                print(f"❌ Failed to switch to venv: {error}")
                sys_module.exit(1)

        app.print_banner()
        kit.log("First run detected. Starting setup...", "info")
        perform_install(kit, app)

        if os_module.path.exists(venv_python):
            kit.log("Restarting in new environment...", "success")
            time_module.sleep(1)
            subprocess_module.check_call([venv_python] + sys_module.argv)
            sys_module.exit(0)

        kit.log("Error: .venv was not created properly.", "error")
        sys_module.exit(1)

    app.ensure_lms_server_running()
    try:
        show_menu(kit, app)
    finally:
        app.stop_lms_server_if_owned()
