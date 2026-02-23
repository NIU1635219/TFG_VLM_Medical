"""Diagnóstico y smart-fix del entorno para setup_env."""

from __future__ import annotations

import os
import shutil
import time
from typing import Any

from .setup_ui_io import IncrementalPanelRenderer, get_ui_width, render_title_banner, wrap_plain_text


def _dynamic_ui_width() -> int:
    """Calcula ancho de UI adaptado al terminal actual sin tope fijo de 86."""
    try:
        cols = int(shutil.get_terminal_size(fallback=(120, 30)).columns)
    except Exception:
        cols = 120
    max_width = max(60, cols - 4)
    return get_ui_width(shutil_module=shutil, max_width=max_width)


def perform_diagnostics(ctx: dict[str, Any]):
    """
    Ejecuta un análisis completo del estado del sistema y dependencias.
    
    Verifica:
    - Existencia de carpetas críticas.
    - Instalación de herramientas (uv, lms).
    - Importación de librerías requeridas.
    - Espacio en disco disponible.
    
    Args:
        ctx (dict): Contexto de ejecución.
        
    Returns:
        tuple[list, list]: (Reporte tabular, Lista de objetos DiagnosticIssue detectados).
    """
    DiagnosticIssue = ctx["DiagnosticIssue"]
    fix_folders = ctx["fix_folders"]
    fix_uv = ctx["fix_uv"]
    fix_libs = ctx["fix_libs"]
    check_uv = ctx["check_uv"]
    check_lms = ctx["check_lms"]
    log = ctx["log"]
    REQUIRED_LIBS = ctx["REQUIRED_LIBS"]
    LIB_IMPORT_MAP = ctx["LIB_IMPORT_MAP"]

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
                lambda: log("Install LM Studio and ensure lms is on PATH", "warning"),
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


def print_report_table(report, *, style: Any, width: int | None = None):
    """
    Imprime una tabla formateada con los resultados del diagnóstico.
    
    Args:
        report (list): Lista de tuplas (Componente, Valor, Estado).
        style: Objeto de estilos ANSI.
        width (int, optional): Ancho de UI a usar para alinear con el banner.
    """
    ui_width = int(width) if isinstance(width, int) and width > 0 else _dynamic_ui_width()
    result_col = 6
    content_budget = max(32, ui_width - result_col - 6)
    component_col = max(16, int(content_budget * 0.45))
    value_col = max(16, content_budget - component_col)

    top = "┌" + "─" * (component_col + 2) + "┬" + "─" * (value_col + 2) + "┬" + "─" * result_col + "┐"
    mid = "├" + "─" * (component_col + 2) + "┼" + "─" * (value_col + 2) + "┼" + "─" * result_col + "┤"
    bottom = "└" + "─" * (component_col + 2) + "┴" + "─" * (value_col + 2) + "┴" + "─" * result_col + "┘"

    print(f"\n{style.HEADER}{top}{style.ENDC}")
    print(
        f"{style.HEADER}│ {'Component':<{component_col}} │ {'Status/Value':<{value_col}} │ {'Res':<{result_col - 2}} │{style.ENDC}"
    )
    print(f"{style.HEADER}{mid}{style.ENDC}")

    for name, value, status in report:
        color = style.OKGREEN if status == "OK" else (style.WARNING if status == "WARN" else style.FAIL)
        display_name = str(name).replace("cv2", "OpenCV").replace("llama_cpp", "LlamaCPP")
        value_text = str(value)

        name_lines = wrap_plain_text(display_name, component_col)
        value_lines = wrap_plain_text(value_text, value_col)
        row_count = max(len(name_lines), len(value_lines))

        for row_idx in range(row_count):
            left = name_lines[row_idx] if row_idx < len(name_lines) else ""
            right = value_lines[row_idx] if row_idx < len(value_lines) else ""
            if row_idx == 0:
                res_text = f"{color}{status:<{result_col - 2}}{style.ENDC}"
            else:
                res_text = " " * (result_col - 2)

            print(f"│ {left:<{component_col}} │ {right:<{value_col}} │ {res_text} │")

    print(f"{style.HEADER}{bottom}{style.ENDC}")


def smart_fix_menu(issues, report, ctx: dict[str, Any]) -> bool:
    """
    Despliega un menú interactivo para aplicar correcciones automáticas (Smart Fix).
    
    Args:
        issues (list): Lista de problemas detectados (DiagnosticIssue).
        report (list): Reporte del diagnóstico previo para mostrar contexto.
        ctx (dict): Contexto de la aplicación.
        
    Returns:
        bool: True si se aplicaron correcciones (sugiere re-ejecutar diagnóstico), False si no.
    """
    style = ctx["Style"]
    interactive_menu = ctx["interactive_menu"]
    print_report_table_fn = ctx["print_report_table"]
    clear_screen_ansi = ctx["clear_screen_ansi"]
    log = ctx["log"]

    seen_funcs = set()
    display_options = []
    for issue in issues:
        func_key = issue.fix_func
        if func_key not in seen_funcs:
            issue.label = issue.description
            issue.action = issue.fix_func
            display_options.append(issue)
            seen_funcs.add(func_key)

    def draw_header():
        if report:
            width = _dynamic_ui_width()
            render_title_banner(title="SYSTEM DIAGNOSTICS (CACHED)", style=style, width=width)
            print_report_table_fn(report)
        width = _dynamic_ui_width()
        render_title_banner(title=f"DIAGNOSTIC ISSUES DETECTED ({len(issues)})", style=style, width=width)
        for item in issues:
            print(f" {style.FAIL}●{style.ENDC} {item.description}")

    selected_fixes = interactive_menu(
        display_options,
        header_func=draw_header,
        multi_select=True,
        info_text="",
        menu_id="smart_fix_menu",
        nav_hint_text="↑/↓ navegar problemas · SPACE seleccionar fix · ENTER aplicar seleccionados · ESC cancelar",
    )

    if selected_fixes:
        clear_screen_ansi()
        print(f"\n{style.BOLD} APPLYING FIXES... {style.ENDC}")
        time.sleep(0.5)

        fixes_list = selected_fixes if isinstance(selected_fixes, list) else [selected_fixes]
        for fix in list(fixes_list):
            fix.action()

        log("All fixes applied. Refreshing diagnostics...", "success")
        time.sleep(1.0)
        return True

    return False


def run_diagnostics_ui(ctx: dict[str, Any]) -> None:
    """
    Controlador principal de la interfaz de diagnósticos.
    
    Ejecuta el ciclo de diagnóstico -> reporte -> smart fix -> re-diagnóstico.
    
    Args:
        ctx (dict): Contexto de aplicación.
    """
    style = ctx["Style"]
    clear_screen = ctx["clear_screen"]
    wait_for_any_key = ctx["wait_for_any_key"]
    perform_diagnostics_fn = ctx["perform_diagnostics"]
    print_report_table_fn = ctx["print_report_table"]
    smart_fix_menu_fn = ctx["smart_fix_menu"]
    read_key_fn = ctx.get("read_key")

    def _show_diagnostics_summary(*, report, issues) -> bool:
        """Renderiza el resumen de diagnóstico y permite resize en tiempo real.

        Returns:
            bool: True si el usuario confirma abrir Smart Fix (solo cuando hay issues).
        """

        def _render_static() -> None:
            width = _dynamic_ui_width()
            print()
            render_title_banner(title="SYSTEM DIAGNOSTICS & VERIFICATION", style=style, width=width)
            try:
                print_report_table_fn(report, width=width)
            except TypeError:
                print_report_table_fn(report)

        panel = IncrementalPanelRenderer(clear_screen_fn=clear_screen, render_static_fn=_render_static)

        if read_key_fn is None:
            clear_screen()
            _render_static()
            if issues:
                print(f"\n{style.FAIL}Issues found! Entering Smart Fix Menu...{style.ENDC}")
                time.sleep(1)
                return True
            print(f"\n{style.OKGREEN}System looks healthy!{style.ENDC}")
            wait_for_any_key()
            return False

        while True:
            dynamic_lines: list[str] = []
            width = _dynamic_ui_width()
            divider = "─" * width

            dynamic_lines.append("")
            dynamic_lines.append(divider)
            if issues:
                dynamic_lines.append(f"{style.FAIL}Issues found: {len(issues)}{style.ENDC}")
                dynamic_lines.append(f"{style.DIM}[ENTER] abrir Smart Fix · [ESC] volver{style.ENDC}")
            else:
                dynamic_lines.append(f"{style.OKGREEN}System looks healthy!{style.ENDC}")
                dynamic_lines.append(f"{style.DIM}[ENTER/ESC] volver{style.ENDC}")
            dynamic_lines.append(divider)

            panel.render(dynamic_lines)

            key = read_key_fn()
            if key is None:
                time.sleep(0.05)
                continue

            if issues and key == "ENTER":
                return True
            if key in ("ENTER", "ESC"):
                return False

            time.sleep(0.01)

    while True:
        report, issues = perform_diagnostics_fn()

        if issues:
            open_smart_fix = _show_diagnostics_summary(report=report, issues=issues)
            if not open_smart_fix:
                break
            should_rerun = smart_fix_menu_fn(issues, report)
            if should_rerun:
                continue
            break

        _show_diagnostics_summary(report=report, issues=issues)
        break
