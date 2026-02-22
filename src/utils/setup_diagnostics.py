"""Diagnóstico y smart-fix del entorno para setup_env."""

from __future__ import annotations

import os
import shutil
import time
from typing import Any


def perform_diagnostics(ctx: dict[str, Any]):
    """Ejecuta comprobaciones del sistema y devuelve (report, issues)."""
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


def print_report_table(report, *, style: Any):
    """Imprime la tabla de diagnóstico de forma formateada."""
    print(f"\n{style.HEADER}┌──────────────────────────┬───────────────────────────────┬──────┐{style.ENDC}")
    print(f"{style.HEADER}│ Component                │ Status/Value                  │ Res  │{style.ENDC}")
    print(f"{style.HEADER}├──────────────────────────┼───────────────────────────────┼──────┤{style.ENDC}")

    for name, value, status in report:
        color = style.OKGREEN if status == "OK" else (style.WARNING if status == "WARN" else style.FAIL)
        display_name = name.replace("cv2", "OpenCV").replace("llama_cpp", "LlamaCPP")
        print(f"│ {display_name:<24} │ {str(value):<29} │ {color}{status:<4}{style.ENDC} │")

    print(f"{style.HEADER}└──────────────────────────┴───────────────────────────────┴──────┘{style.ENDC}")


def smart_fix_menu(issues, report, ctx: dict[str, Any]) -> bool:
    """Menú gráfico para seleccionar y ejecutar arreglos."""
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
            print(f"\n{style.BOLD}--- SYSTEM DIAGNOSTICS (CACHED) ---{style.ENDC}")
            print_report_table_fn(report)
        print(f"{style.BOLD}{style.FAIL} DIAGNOSTIC ISSUES DETECTED ({len(issues)}) {style.ENDC}")
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
    """UI loop de diagnósticos con smart fix."""
    style = ctx["Style"]
    clear_screen = ctx["clear_screen"]
    wait_for_any_key = ctx["wait_for_any_key"]
    perform_diagnostics_fn = ctx["perform_diagnostics"]
    print_report_table_fn = ctx["print_report_table"]
    smart_fix_menu_fn = ctx["smart_fix_menu"]

    while True:
        clear_screen()
        print(f"\n{style.BOLD}--- SYSTEM DIAGNOSTICS & VERIFICATION ---{style.ENDC}")

        report, issues = perform_diagnostics_fn()
        print_report_table_fn(report)

        if issues:
            print(f"\n{style.FAIL}Issues found! Entering Smart Fix Menu...{style.ENDC}")
            time.sleep(1)
            should_rerun = smart_fix_menu_fn(issues, report)
            if should_rerun:
                continue
            break

        print(f"\n{style.OKGREEN}System looks healthy!{style.ENDC}")
        wait_for_any_key()
        break
