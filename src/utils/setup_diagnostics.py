"""Diagnóstico y smart-fix del entorno para setup_env."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import TYPE_CHECKING

from .setup_install_flow import check_uv  # sin circular: setup_install_flow no importa setup_diagnostics a nivel de módulo

if TYPE_CHECKING:
    from .menu_kit import AppContext, UIKit


def _get_project_python_executable() -> str:
    """Resuelve el intérprete Python del entorno del proyecto si existe."""
    if os.name == "nt":
        candidate = os.path.join(".venv", "Scripts", "python.exe")
    else:
        candidate = os.path.join(".venv", "bin", "python")

    if os.path.exists(candidate):
        return candidate
    return sys.executable


def _can_import_package_in_project_env(import_name: str) -> bool:
    """Comprueba importabilidad de un módulo usando el Python del proyecto."""
    if not import_name:
        return False

    python_exe = _get_project_python_executable()
    code = (
        "import importlib, sys; "
        "importlib.import_module(sys.argv[1]); "
        "print('ok')"
    )
    try:
        result = subprocess.run(
            [python_exe, "-c", code, import_name],
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception:
        return False

    return result.returncode == 0


def _get_module_version_in_project_env(import_name: str) -> str:
    """Obtiene la versión de un módulo usando el Python del proyecto."""
    python_exe = _get_project_python_executable()
    code = (
        "import importlib, sys; "
        "m=importlib.import_module(sys.argv[1]); "
        "print(getattr(m, '__version__', 'Unknown'))"
    )
    try:
        result = subprocess.run(
            [python_exe, "-c", code, import_name],
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception:
        return "Unknown"

    if result.returncode != 0:
        return "Unknown"
    return (result.stdout or "Unknown").strip() or "Unknown"


def _is_lmstudio_server_reachable() -> bool:
    """Detecta si el servidor LM Studio responde en el puerto local por defecto."""
    urls = [
        "http://127.0.0.1:1234/v1/models",
        "http://localhost:1234/v1/models",
    ]

    configured_host = (os.environ.get("LMSTUDIO_HOST", "") or "").strip()
    if configured_host:
        if configured_host.startswith(("http://", "https://")):
            urls.insert(0, configured_host.rstrip("/") + "/v1/models")
        else:
            urls.insert(0, f"http://{configured_host}/v1/models")

    # En WSL, LM Studio puede vivir en Windows host en vez de localhost Linux.
    try:
        with open("/etc/resolv.conf", "r", encoding="utf-8", errors="ignore") as resolv:
            for line in resolv:
                line = line.strip()
                if line.startswith("nameserver "):
                    host_ip = line.split(maxsplit=1)[1].strip()
                    if host_ip:
                        urls.append(f"http://{host_ip}:1234/v1/models")
                    break
    except Exception:
        pass

    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=1.5) as response:
                status_code = int(getattr(response, "status", 200))
                if status_code < 500:
                    return True
        except urllib.error.HTTPError as error:
            if int(getattr(error, "code", 500)) < 500:
                return True
        except Exception:
            continue
    return False


def _render_diagnostics_report_table(kit: "UIKit", report: list[tuple[str, str, str]], *, width: int) -> None:
    """Renderiza el reporte de diagnósticos con el motor de tablas unificado."""
    columns = [
        kit.TableColumn(label="Component", width_ratio=0.45, min_width=16),
        kit.TableColumn(label="Status/Value", width_ratio=0.45, min_width=16),
        kit.TableColumn(label="Res", fixed_width=6, min_width=6),
    ]
    rows = [
        kit.TableRow(
            cells=[
                str(name).replace("cv2", "OpenCV").replace("llama_cpp", "LlamaCPP"),
                str(value),
                str(status),
            ],
            cell_colors=[
                None,
                None,
                "OKGREEN" if str(status) == "OK" else ("WARNING" if str(status) == "WARN" else "FAIL"),
            ],
        )
        for name, value, status in report
    ]
    kit.table_menu(columns, rows, width=width, interactive=False)


# ---------------------------------------------------------------------------
# DiagnosticIssue
# ---------------------------------------------------------------------------

class DiagnosticIssue:
    """
    Representa un problema detectado durante el diagnóstico del entorno.

    Almacena la descripción del problema, la función que lo soluciona y el
    nombre de la solución para mostrar en la interfaz de usuario.
    """

    def __init__(self, description: str, fix_func, fix_name: str) -> None:
        self.description = description
        self.name = description          # alias usado en tests
        self.fix_func = fix_func
        self.fix_name = fix_name
        self.label = fix_name
        self.action = fix_func
        self.children: list = []
        self.is_selected: bool = False


# ---------------------------------------------------------------------------
# Diagnóstico del entorno
# ---------------------------------------------------------------------------

def perform_diagnostics(kit: "UIKit", app: "AppContext"):
    """
    Ejecuta un análisis completo del estado del sistema y dependencias.

    Verifica carpetas críticas, herramientas (uv, lms), librerías requeridas
    y espacio en disco.

    Args:
        kit (UIKit): Interfaz de UI de terminal.
        app (AppContext): Contexto de dominio de la aplicación.

    Returns:
        tuple[list, list]: (Reporte tabular, Lista de DiagnosticIssue detectados).
    """
    fix_folders = app.fix_folders
    fix_uv = app.fix_uv
    fix_libs = app.fix_libs
    check_lms = app.check_lms
    REQUIRED_LIBS = app.REQUIRED_LIBS
    LIB_IMPORT_MAP = app.LIB_IMPORT_MAP

    report = []
    issues = []

    folders = ["data/raw", "data/processed", "src/preprocessing", "src/inference", "notebooks"]
    missing_folders = False
    for folder in folders:
        if os.path.exists(folder):
            report.append((f"Dir: {folder}", "Exists", "OK"))
        else:
            report.append((f"Dir: {folder}", "MISSING", "FAIL"))
            missing_folders = True

    if missing_folders:
        issues.append(DiagnosticIssue("Missing Folders", fix_folders, "Regenerate Folders"))

    if check_uv():
        report.append(("Tool: uv", "Installed", "OK"))
    else:
        report.append(("Tool: uv", "MISSING", "FAIL"))
        issues.append(DiagnosticIssue("Missing uv", fix_uv, "Install uv"))

    for pkg_name in REQUIRED_LIBS:
        import_name = LIB_IMPORT_MAP.get(pkg_name, pkg_name)
        if _can_import_package_in_project_env(import_name):
            version = _get_module_version_in_project_env(import_name)
            report.append((f"Lib: {pkg_name}", version, "OK"))
        else:
            report.append((f"Lib: {pkg_name}", "MISSING", "FAIL"))
            fix_lambda = lambda p=[pkg_name]: fix_libs(p)
            issues.append(DiagnosticIssue(f"Missing {pkg_name}", fix_lambda, f"Install {pkg_name}"))

    lms_cli_ok = bool(check_lms())
    lms_server_ok = _is_lmstudio_server_reachable()

    if lms_cli_ok:
        report.append(("Tool: lms CLI", "Installed", "OK"))
    else:
        report.append(("Tool: lms CLI", "NOT DETECTED", "WARN"))

    if lms_server_ok:
        report.append(("LM Studio Server", "Reachable (localhost:1234)", "OK"))
    else:
        report.append(("LM Studio Server", "NOT REACHABLE", "FAIL"))
        issues.append(
            DiagnosticIssue(
                "LM Studio server not reachable",
                lambda: kit.log("Open LM Studio and ensure local server is running on port 1234", "warning"),
                "Check LM Studio Server",
            )
        )

    if not lms_cli_ok and not lms_server_ok:
        issues.append(
            DiagnosticIssue(
                "LM Studio CLI not found",
                lambda: kit.log("Install LM Studio and ensure lms is on PATH", "warning"),
                "Install LM Studio CLI",
            )
        )

    free_gb = shutil.disk_usage(".").free / (1024**3)
    if free_gb < 10:
        report.append(("Disk Space", f"{free_gb:.1f} GB", "WARN"))
    else:
        report.append(("Disk Space", f"{free_gb:.1f} GB (Free)", "OK"))

    return report, issues


def smart_fix_menu(kit: "UIKit", app: "AppContext", issues: list, report: list | None = None) -> bool:
    """
    Despliega un menú interactivo para aplicar correcciones automáticas.

    Args:
        kit (UIKit): Interfaz de UI de terminal.
        app (AppContext): Contexto de dominio.
        issues (list): Lista de problemas detectados (DiagnosticIssue).
        report (list | None): Reporte del diagnóstico previo para contexto.

    Returns:
        bool: True si se aplicaron correcciones.
    """
    seen_funcs: set = set()
    display_options = []
    for issue in issues:
        func_key = issue.fix_func
        if func_key not in seen_funcs:
            issue.label = issue.description
            issue.action = issue.fix_func
            display_options.append(issue)
            seen_funcs.add(func_key)

    def draw_header() -> None:
        """Renderiza el encabezado del menú de correcciones."""
        w = kit.width()
        app.print_banner()
        if report:
            kit.banner("SYSTEM DIAGNOSTICS (CACHED)", width=w)
            _render_diagnostics_report_table(kit, report, width=w)
        kit.banner(f"DIAGNOSTIC ISSUES DETECTED ({len(issues)})", width=w)
        for item in issues:
            print(f" {kit.style.FAIL}●{kit.style.ENDC} {item.description}")

    selected_fixes = kit.menu(
        display_options,
        header_func=draw_header,
        multi_select=True,
        info_text="",
        menu_id="smart_fix_menu",
        nav_hint_text="↑/↓ navegar problemas · SPACE seleccionar fix · ENTER aplicar seleccionados · ESC cancelar",
    )

    if selected_fixes:
        kit.clear()
        print(f"\n{kit.style.BOLD} APPLYING FIXES... {kit.style.ENDC}")
        time.sleep(0.5)

        fixes_list = selected_fixes if isinstance(selected_fixes, list) else [selected_fixes]
        for fix in list(fixes_list):
            fix.action()

        kit.log("All fixes applied. Refreshing diagnostics...", "success")
        time.sleep(1.0)
        return True

    return False


def run_diagnostics_ui(kit: "UIKit", app: "AppContext") -> None:
    """
    Controlador principal de la interfaz de diagnósticos.

    Ejecuta el ciclo diagnóstico → reporte → smart fix → re-diagnóstico.

    Args:
        kit (UIKit): Interfaz de UI de terminal.
        app (AppContext): Contexto de dominio.
    """

    def _show_diagnostics_summary(*, report: list, issues: list) -> bool:
        """Renderiza el resumen de diagnóstico con resize en tiempo real.

        Returns:
            bool: ``True`` si el usuario confirma abrir Smart Fix.
        """

        def _render_static() -> None:
            w = kit.width()
            print()
            app.print_banner()
            kit.banner("SYSTEM DIAGNOSTICS & VERIFICATION", width=w)
            _render_diagnostics_report_table(kit, report, width=w)

        panel = kit.IncrementalPanelRenderer(
            clear_screen_fn=kit.clear,
            render_static_fn=_render_static,
        )

        _last_w: int | None = None

        while True:
            w = kit.width()
            if w != _last_w:
                div = kit.divider(w)
                dynamic_lines: list[str] = ["", div]
                if issues:
                    dynamic_lines.append(f"{kit.style.FAIL}Issues found: {len(issues)}{kit.style.ENDC}")
                    dynamic_lines.append(f"{kit.style.DIM}[ENTER] abrir Smart Fix · [ESC] volver{kit.style.ENDC}")
                else:
                    dynamic_lines.append(f"{kit.style.OKGREEN}System looks healthy!{kit.style.ENDC}")
                    dynamic_lines.append(f"{kit.style.DIM}[ENTER/ESC] volver{kit.style.ENDC}")
                dynamic_lines.append(div)
                panel.render(dynamic_lines)
                _last_w = w

            key = kit.read_key()
            if key is None:
                time.sleep(0.05)
                continue

            if issues and key == "ENTER":
                return True
            if key in ("ENTER", "ESC"):
                return False

            time.sleep(0.01)

    while True:
        report, issues = perform_diagnostics(kit, app)

        if issues:
            open_smart_fix = _show_diagnostics_summary(report=report, issues=issues)
            if not open_smart_fix:
                break
            should_rerun = smart_fix_menu(kit, app, issues, report)
            if should_rerun:
                continue
            break

        _show_diagnostics_summary(report=report, issues=issues)
        break
