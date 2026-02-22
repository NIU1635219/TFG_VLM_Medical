"""Utilidades de UI/IO para el asistente interactivo de setup.

Este módulo concentra helpers de terminal y entrada de usuario para reducir
acoplamiento y tamaño del módulo principal `setup_env.py`.
"""

from __future__ import annotations

import subprocess
from typing import Any, Callable


class IncrementalPanelRenderer:
    """Renderer incremental para paneles de terminal con bloque estático + bloque dinámico."""

    def __init__(self, *, clear_screen_fn: Callable[[], None], render_static_fn: Callable[[], None]):
        self.clear_screen_fn = clear_screen_fn
        self.render_static_fn = render_static_fn
        self.static_rendered = False
        self.prev_dynamic_line_count = 0

    def reset(self) -> None:
        """Invalida el caché visual para forzar repaint completo en próximo render."""
        self.static_rendered = False
        self.prev_dynamic_line_count = 0

    def render(self, dynamic_lines: list[str], *, force_full: bool = False) -> None:
        """Renderiza solo líneas dinámicas, con repaint completo opcional."""
        if force_full or not self.static_rendered:
            self.clear_screen_fn()
            self.render_static_fn()
            self.static_rendered = True
            self.prev_dynamic_line_count = 0

        if self.prev_dynamic_line_count > 0:
            print(f"\033[{self.prev_dynamic_line_count}F", end="")

        for line in dynamic_lines:
            print(f"\033[2K{line}")

        if self.prev_dynamic_line_count > len(dynamic_lines):
            for _ in range(self.prev_dynamic_line_count - len(dynamic_lines)):
                print("\033[2K")

        self.prev_dynamic_line_count = len(dynamic_lines)


def clear_screen_ansi(*, os_module: Any, sys_module: Any) -> None:
    """Limpia pantalla con `cls` en Windows y ANSI en otros SO."""
    if os_module.name == "nt":
        os_module.system("cls")
    else:
        sys_module.stdout.write("\033[H\033[2J")
        sys_module.stdout.flush()


def read_key(*, os_module: Any, msvcrt_module: Any) -> str | None:
    """Lee tecla y devuelve código unificado (UP/DOWN/LEFT/RIGHT/ENTER/SPACE/ESC)."""
    if os_module.name == "nt" and msvcrt_module:
        key = msvcrt_module.getch()
        if key == b"\xe0":
            key = msvcrt_module.getch()
            if key == b"H":
                return "UP"
            if key == b"P":
                return "DOWN"
            if key == b"K":
                return "LEFT"
            if key == b"M":
                return "RIGHT"
        elif key == b"\r":
            return "ENTER"
        elif key == b" ":
            return "SPACE"
        elif key == b"\x1b":
            return "ESC"
        elif key == b"\x03":
            raise KeyboardInterrupt
    return None


def log(*, style: Any, msg: str, level: str = "info") -> None:
    """Imprime mensajes de estado con color/estilo."""
    if level == "info":
        print(f"{style.OKCYAN} ℹ {msg}{style.ENDC}")
    elif level == "success":
        print(f"{style.OKGREEN} ✔ {msg}{style.ENDC}")
    elif level == "error":
        print(f"{style.FAIL} ✖ {msg}{style.ENDC}")
    elif level == "warning":
        print(f"{style.WARNING} ⚠ {msg}{style.ENDC}")
    elif level == "step":
        print(f"\n{style.BOLD}➤ {msg}{style.ENDC}")


def ask_user(
    *,
    question: str,
    default: str,
    style: Any,
    read_key_fn: Callable[[], str | None],
    clear_screen_fn: Callable[[], None],
) -> bool:
    """Pide confirmación usando navegación por flechas + ENTER."""
    normalized_default = (default or "y").strip().lower()
    if normalized_default not in ("y", "n"):
        normalized_default = "y"

    selected = 0 if normalized_default == "y" else 1
    divider = "─" * 79

    def _render_static() -> None:
        print(f"{style.BOLD}{style.WARNING}{question}{style.ENDC}")
        print(divider)
        print(
            f"{style.DIM}Acciones: ←/→ (o ↑/↓) cambiar opción · ENTER confirmar · ESC cancelar.{style.ENDC}"
        )
        print(divider)

    panel = IncrementalPanelRenderer(clear_screen_fn=clear_screen_fn, render_static_fn=_render_static)

    while True:
        yes_label = " Sí "
        no_label = " No "

        if selected == 0:
            yes_render = f"{style.SELECTED}{yes_label}{style.ENDC}"
            no_render = no_label
        else:
            yes_render = yes_label
            no_render = f"{style.SELECTED}{no_label}{style.ENDC}"

        panel.render([
            "",
            f"   {yes_render}    {no_render}",
            "",
            divider,
        ])

        key = read_key_fn()
        if key in ("LEFT", "UP"):
            selected = 0
        elif key in ("RIGHT", "DOWN"):
            selected = 1
        elif key == "ENTER":
            return selected == 0
        elif key == "ESC":
            return False


def input_with_esc(*, prompt: str, os_module: Any, msvcrt_module: Any) -> str | None:
    """Entrada de texto con cancelación por ESC en Windows."""
    if os_module.name == "nt" and msvcrt_module:
        print(prompt, end="", flush=True)
        buffer: list[str] = []
        while True:
            key = msvcrt_module.getch()
            if key == b"\x1b":
                print()
                return None
            if key in (b"\r", b"\n"):
                print()
                return "".join(buffer).strip()
            if key == b"\x08":
                if buffer:
                    buffer.pop()
                    print("\b \b", end="", flush=True)
                continue
            if key in (b"\xe0", b"\x00"):
                _ = msvcrt_module.getch()
                continue
            try:
                char = key.decode("utf-8")
            except UnicodeDecodeError:
                continue
            buffer.append(char)
            print(char, end="", flush=True)

    value = input(prompt).strip()
    if value.lower() == "esc":
        return None
    return value


def wait_for_any_key(*, message: str, style: Any, os_module: Any, msvcrt_module: Any) -> None:
    """Pausa breve hasta recibir una tecla."""
    print(f"\n{style.DIM}{message}{style.ENDC}", end="", flush=True)
    if os_module.name == "nt" and msvcrt_module:
        _ = msvcrt_module.getch()
        print()
        return
    input()


def run_cmd(
    *,
    cmd: str,
    critical: bool,
    style: Any,
    ask_user_fn: Callable[[str, str], bool],
    log_fn: Callable[[str, str], None],
) -> bool:
    """Ejecuta comando shell con política de reintento para fallos críticos."""
    print(f"{style.DIM}$ {cmd}{style.ENDC}")
    try:
        subprocess.check_call(cmd, shell=True)
        return True
    except subprocess.CalledProcessError:
        log_fn(f"Command failed: {cmd}", "error")
        if critical:
            if ask_user_fn("Critical command failed. Retry?", "y"):
                return run_cmd(
                    cmd=cmd,
                    critical=critical,
                    style=style,
                    ask_user_fn=ask_user_fn,
                    log_fn=log_fn,
                )
            return False
        return False
