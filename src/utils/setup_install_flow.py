"""Flujo de instalación/bootstrap y menú principal para setup_env."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Any


def perform_install(*, ctx: dict[str, Any], full_reinstall: bool = False) -> None:
    """
    Ejecuta el flujo de instalación inicial o reinstalación completa del entorno.
    
    Gestiona la creación de `.venv`, la instalación de `uv`, y la sincronización 
    de dependencias base y de GPU/CPU según detección.
    
    Args:
        ctx (dict): Contexto de ejecución con herramientas inyectadas.
        full_reinstall (bool, optional): Si True, elimina el entorno virtual existente antes.
    """
    log = ctx["log"]
    create_project_structure = ctx["create_project_structure"]
    check_uv = ctx["check_uv"]
    run_cmd = ctx["run_cmd"]
    detect_gpu = ctx["detect_gpu"]
    REQUIRED_LIBS = ctx["REQUIRED_LIBS"]

    force_flags = ""

    if full_reinstall and os.path.exists(".venv"):
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
                import shutil

                shutil.rmtree(".venv")
                log("Environment removed.", "success")
                time.sleep(1)
            except Exception as error:
                log(f"Failed to delete .venv: {error}", "error")
                log("Please delete the folder manually.", "warning")
                return

    create_project_structure(verbose=False)

    if not check_uv():
        log("uv not found. Attempting to install via pip...", "step")
        if not run_cmd("pip install uv"):
            log("Could not install uv. Please install it manually.", "error")
            return

    use_gpu = detect_gpu()
    if not use_gpu:
        log("No NVIDIA GPU detected. Installing in CPU mode.", "warning")
    else:
        log("NVIDIA GPU detected. Using High Performance configuration.", "success")

    log("Configuring Virtual Environment...", "step")
    if not os.path.exists(".venv"):
        run_cmd("uv venv .venv --python 3.12")
    else:
        log(".venv exists.", "info")

    if os.path.exists("pyproject.toml"):
        log("Syncing dependencies from pyproject.toml/uv.lock...", "step")
        sync_flag = " --reinstall" if full_reinstall else ""
        run_cmd(f"uv sync{sync_flag}")
    else:
        log("Installing fallback dependencies (no pyproject.toml found)...", "step")
        light_libs = " ".join(REQUIRED_LIBS)
        run_cmd(f"uv pip install {force_flags} {light_libs}")


def show_menu(*, ctx: dict[str, Any]) -> None:
    """
    Despliega el menú principal interactivo de la herramienta de setup.
    
    Ofrece opciones para diagnóstico, gestión de modelos/tests, reinstalación 
    y regeneración de estructura.
    
    Args:
        ctx (dict): Contexto de aplicación.
    """
    MenuItem = ctx["MenuItem"]
    Style = ctx["Style"]
    print_banner = ctx["print_banner"]
    run_diagnostics_ui = ctx["run_diagnostics_ui"]
    run_tests_menu = ctx["run_tests_menu"]
    reinstall_library_menu = ctx["reinstall_library_menu"]
    create_project_structure = ctx["create_project_structure"]
    wait_for_any_key = ctx["wait_for_any_key"]
    ask_user = ctx["ask_user"]
    perform_install_fn = ctx["perform_install"]
    interactive_menu = ctx["interactive_menu"]
    clear_screen_ansi = ctx["clear_screen_ansi"]

    def header():
        """
        Renderiza el encabezado del menú principal.
        """
        print_banner()
        print(f"{Style.BOLD} MAIN MENU {Style.ENDC}")

    def action_diag():
        """
        Ejecuta el menú de diagnóstico.
        """
        run_diagnostics_ui()

    def action_reinstall():
        """
        Ejecuta el menú de reinstalación de dependencias.
        """
        reinstall_library_menu()

    def action_regen():
        """
        Regenera la estructura del proyecto.
        """
        create_project_structure(verbose=True)
        wait_for_any_key()

    def action_reset():
        """
        Ejecuta el menú de reinicio del entorno.
        """
        if ask_user("This will delete .venv and reinstall everything. Sure?", "n"):
            perform_install_fn(full_reinstall=True)

    options = [
        MenuItem(" Run System Diagnostics", action_diag, description="Analiza el sistema y ofrece correcciones automáticas."),
        MenuItem(" Tests & Models Manager", run_tests_menu, description="Ejecuta tests, smoke test y gestiona modelos LM Studio."),
        MenuItem(" Manual Reinstall Menu", action_reinstall, description="Reinstala dependencias concretas manualmente."),
        MenuItem(" Regenerate Folders", action_regen, description="Crea carpetas del proyecto que falten."),
        MenuItem(" Factory Reset", action_reset, description="Reinstala el entorno del proyecto desde cero (confirmación por defecto en NO)."),
        MenuItem(" Exit", lambda: sys.exit(0), description="Cierra la herramienta de gestión."),
    ]

    while True:
        choice = interactive_menu(
            options,
            header_func=header,
            multi_select=False,
            menu_id="main_menu",
            nav_hint_text="↑/↓ navegar opciones · ENTER ejecutar acción · ESC salir",
        )

        if choice:
            item = choice[0] if isinstance(choice, list) else choice
            clear_screen_ansi()
            item.action()
        else:
            clear_screen_ansi()
            print("Goodbye!")
            sys.exit(0)


def main(*, ctx: dict[str, Any]) -> None:
    """
    Punto de entrada principal del script de configuración.
    
    Se encarga de:
    1. Detectar si corre dentro del entorno virtual `.venv`.
    2. Si no, intenta crearlo, instalar dependencias y relanzarse dentro de él.
    3. Una vez en `.venv`, asegura que el servidor LM Studio esté corriendo.
    4. Muestra el menú principal.
    5. Limpia procesos (servidor LMS) al salir.
    
    Args:
        ctx (dict): Contexto global de dependencias.
    """
    os_module = ctx["os"]
    sys_module = ctx["sys"]
    subprocess_module = ctx["subprocess"]
    time_module = ctx["time"]
    print_banner = ctx["print_banner"]
    log = ctx["log"]
    perform_install_fn = ctx["perform_install"]
    ensure_lms_server_running = ctx["ensure_lms_server_running"]
    show_menu_fn = ctx["show_menu"]
    stop_lms_server_if_owned = ctx["stop_lms_server_if_owned"]

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

        print_banner()
        log("First run detected. Starting setup...", "info")
        perform_install_fn()

        if os_module.path.exists(venv_python):
            log("Restarting in new environment...", "success")
            time_module.sleep(1)
            subprocess_module.check_call([venv_python] + sys_module.argv)
            sys_module.exit(0)

        log("Error: .venv was not created properly.", "error")
        sys_module.exit(1)

    ensure_lms_server_running()
    try:
        show_menu_fn()
    finally:
        stop_lms_server_if_owned()
