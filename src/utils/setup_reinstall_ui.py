"""UI de reinstalación manual de librerías para setup_env."""

from __future__ import annotations

from typing import Any


def reinstall_library_menu(*, ctx: dict[str, Any]) -> None:
    """Menú manual para forzar reinstalaciones limpias."""
    Style = ctx["Style"]
    MenuItem = ctx["MenuItem"]
    REQUIRED_LIBS = ctx["REQUIRED_LIBS"]
    print_banner = ctx["print_banner"]
    interactive_menu = ctx["interactive_menu"]
    clear_screen_ansi = ctx["clear_screen_ansi"]
    fix_uv = ctx["fix_uv"]
    fix_libs = ctx["fix_libs"]
    log = ctx["log"]
    restart_program = ctx["restart_program"]
    wait_for_any_key = ctx["wait_for_any_key"]

    def header() -> None:
        print_banner()
        print(f"{Style.BOLD} REINSTALL LIBRARIES {Style.ENDC}")
        print(" Select libraries to force re-install (clean install).")

    core_children = [MenuItem(lib, description=f"Reinstala la librería '{lib}'.") for lib in REQUIRED_LIBS]

    opts = [
        MenuItem("Reinstall Core Dependencies", children=core_children, description="Selecciona una o varias dependencias para reinstalar."),
        MenuItem("Reinstall 'uv' Tool", lambda: fix_uv(), description="Reinstala la herramienta uv en el entorno actual."),
    ]

    selected = interactive_menu(
        opts,
        header_func=header,
        multi_select=True,
        menu_id="reinstall_menu",
        nav_hint_text="↑/↓ navegar · SPACE seleccionar · ENTER confirmar selección · ESC volver",
        sub_nav_hint_text="[DEPENDENCIAS] ↑/↓ navegar librerías · SPACE marcar/desmarcar · ESC volver",
    )

    if selected:
        clear_screen_ansi()

        selected_core_libs = [s.label for s in selected if s in core_children]
        other_tasks = [s for s in selected if s not in core_children and callable(s.action)]

        if selected_core_libs:
            fix_libs(selected_core_libs)

        for task in other_tasks:
            log(f"Ejecutando {task.label}...", "step")
            task.action()

        restart_needed = False
        if selected_core_libs:
            restart_needed = True

        if restart_needed:
            restart_program()

        log("Operaciones completadas.", "success")
        wait_for_any_key("Press any key to continue...")
