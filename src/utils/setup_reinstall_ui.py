"""UI de reinstalación manual de librerías para setup_env."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .menu_kit import AppContext, UIKit


def reinstall_library_menu(kit: "UIKit", app: "AppContext") -> None:
    """
    Despliega el menú para la reinstalación manual de librerías y herramientas.

    Permite seleccionar dependencias individuales para forzar su reinstalación
    limpia mediante ``uv``.

    Args:
        kit (UIKit): Interfaz de UI de terminal.
        app (AppContext): Contexto de dominio de la aplicación.
    """

    def header() -> None:
        app.print_banner()
        kit.subtitle("REINSTALL LIBRARIES")
        print(" Select libraries to force re-install (clean install).")

    core_children = [
        kit.MenuItem(lib, description=f"Reinstala la librería '{lib}'.")
        for lib in app.REQUIRED_LIBS
    ]

    opts = [
        kit.MenuItem(
            "Reinstall Core Dependencies",
            children=core_children,
            description="Selecciona una o varias dependencias para reinstalar.",
        ),
        kit.MenuItem(
            "Reinstall 'uv' Tool",
            lambda: app.fix_uv(),
            description="Reinstala la herramienta uv en el entorno actual.",
        ),
    ]

    selected = kit.menu(
        opts,
        header_func=header,
        multi_select=True,
        menu_id="reinstall_menu",
        nav_hint_text="↑/↓ navegar · SPACE seleccionar · ENTER confirmar selección · ESC volver",
        sub_nav_hint_text="[DEPENDENCIAS] ↑/↓ navegar librerías · SPACE marcar/desmarcar · ESC volver",
    )

    if selected:
        kit.clear()

        selected_core_libs = [s.label for s in selected if s in core_children]
        other_tasks = [s for s in selected if s not in core_children and callable(s.action)]

        if selected_core_libs:
            app.fix_libs(selected_core_libs)

        for task in other_tasks:
            kit.log(f"Ejecutando {task.label}...", "step")
            task.action()

        restart_needed = bool(selected_core_libs)

        if restart_needed:
            app.restart_program()

        kit.log("Operaciones completadas.", "success")
        kit.wait("Press any key to continue...")
