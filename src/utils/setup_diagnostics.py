"""Diagnóstico y smart-fix del entorno para setup_env."""

from __future__ import annotations

import os
import shutil
import time
from typing import TYPE_CHECKING

from .setup_install_flow import check_uv  # sin circular: setup_install_flow no importa setup_diagnostics a nivel de módulo

if TYPE_CHECKING:
    from .menu_kit import AppContext, UIKit


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
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "Unknown")
            report.append((f"Lib: {pkg_name}", version, "OK"))
        except ImportError:
            report.append((f"Lib: {pkg_name}", "MISSING", "FAIL"))
            fix_lambda = lambda p=[pkg_name]: fix_libs(p)
            issues.append(DiagnosticIssue(f"Missing {pkg_name}", fix_lambda, f"Install {pkg_name}"))
        except Exception:
            report.append((f"Lib: {pkg_name}", "ERROR", "FAIL"))
            fix_lambda = lambda p=[pkg_name]: fix_libs(p)
            issues.append(DiagnosticIssue(f"Error {pkg_name}", fix_lambda, f"Reinstall {pkg_name}"))

    if check_lms():
        report.append(("Tool: lms", "Installed", "OK"))
    else:
        report.append(("Tool: lms", "NOT DETECTED", "FAIL"))
        issues.append(
            DiagnosticIssue(
                "LM Studio CLI not found",
                lambda: kit.log("Install LM Studio and ensure lms is on PATH", "warning"),
                "Install LM Studio CLI",
            )
        )

    try:
        import lmstudio

        _ = lmstudio
        report.append(("Lib: lmstudio", "Installed", "OK"))
    except ImportError:
        report.append(("Lib: lmstudio", "MISSING", "FAIL"))
        issues.append(DiagnosticIssue("Missing lmstudio lib", lambda: fix_libs("lmstudio"), "Install lmstudio lib"))

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
        if report:
            kit.banner("SYSTEM DIAGNOSTICS (CACHED)", width=w)
            kit.table(report, width=w)
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
            kit.banner("SYSTEM DIAGNOSTICS & VERIFICATION", width=w)
            kit.table(report, width=w)

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
