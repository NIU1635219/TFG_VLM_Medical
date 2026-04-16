"""Utilidades de UI/IO para el asistente interactivo de setup.

Este módulo concentra helpers de terminal y entrada de usuario para reducir
acoplamiento y tamaño del módulo principal `setup_env.py`.
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
import time
import os
import sys
from typing import Any, Callable

try:
    import select
except Exception:  # pragma: no cover
    select = None

try:
    import termios
    import tty
except Exception:  # pragma: no cover
    termios = None
    tty = None


_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
DEFAULT_UI_WIDTH = 86


def _get_posix_tty_api() -> tuple[Any, Any, Any, int] | None:
    """Obtiene APIs POSIX de terminal de forma segura para entornos sin soporte."""
    if termios is None or tty is None:
        return None
    tcgetattr = getattr(termios, "tcgetattr", None)
    tcsetattr = getattr(termios, "tcsetattr", None)
    setcbreak = getattr(tty, "setcbreak", None)
    tcsadrain = int(getattr(termios, "TCSADRAIN", 1))
    if not callable(tcgetattr) or not callable(tcsetattr) or not callable(setcbreak):
        return None
    return tcgetattr, tcsetattr, setcbreak, tcsadrain


# ---------------------------------------------------------------------------
# Style — códigos de escape ANSI
# ---------------------------------------------------------------------------


class Style:
    """Constantes de escape ANSI para estilizar texto en la terminal.

    Re-exportada en ``setup_env.py`` para mantener compatibilidad con módulos
    que la referencian a través del punto de entrada principal.
    """

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    # Fondo invertido para ítems seleccionados en menús
    SELECTED = "\033[7m"
    # Heatmap live: combinaciones alto contraste fg/bg para celdas
    HM_CELL_LOW = "\033[97;44m"        # texto claro en fondo azul oscuro
    HM_CELL_MID = "\033[30;103m"       # texto oscuro en fondo amarillo
    HM_CELL_HIGH = "\033[30;101m"      # texto oscuro en fondo rojo claro
    HM_CELL_DIAG = "\033[30;102m"      # texto oscuro en fondo verde claro (diagonal correcta)
    HM_CELL_TOTAL = "\033[97;100m"     # texto claro en gris oscuro para totales


# ---------------------------------------------------------------------------
# Helpers de ancho visible (respetando secuencias ANSI)
# ---------------------------------------------------------------------------


def _visible_len(text: str) -> int:
    """Longitud visible de una cadena ignorando códigos ANSI.

    Args:
        text (str): Texto con o sin secuencias de escape ANSI.

    Returns:
        int: Longitud visual en caracteres.
    """
    return len(_ANSI_RE.sub("", str(text or "")))


def _wrap_colored_chunks(chunks: list[str], *, prefix: str, max_width: int) -> list[str]:
    """Envuelve segmentos coloreados sin romper secuencias ANSI.

    Args:
        chunks (list[str]): Segmentos de texto con posibles códigos ANSI.
        prefix (str): Prefijo añadido al comienzo de cada nueva línea.
        max_width (int): Ancho visible máximo por línea.

    Returns:
        list[str]: Líneas compuestas que respetan el ancho visible.
    """
    if not chunks:
        return [prefix]

    prefix_len = _visible_len(prefix)
    separator = "  "
    separator_len = _visible_len(separator)

    lines: list[str] = []
    current = prefix
    current_len = prefix_len

    for chunk in chunks:
        chunk_len = _visible_len(chunk)
        extra = chunk_len if current_len == prefix_len else (separator_len + chunk_len)
        if current_len + extra > max_width and current_len > prefix_len:
            lines.append(current)
            current = prefix + chunk
            current_len = prefix_len + chunk_len
            continue

        if current_len == prefix_len:
            current += chunk
            current_len += chunk_len
        else:
            current += separator + chunk
            current_len += separator_len + chunk_len

    lines.append(current)
    return lines


# ---------------------------------------------------------------------------
# Helpers internos de geometría
# ---------------------------------------------------------------------------


def _visual_rows(text: str, terminal_width: int) -> int:
    """
    Calcula cuántas filas visuales ocupa un texto considerando wrap automático.
    
    Args:
        text (str): Texto a analizar.
        terminal_width (int): Ancho del terminal.
    
    Returns:
        int: Número de filas visuales.
    """
    clean_text = _ANSI_RE.sub("", str(text or ""))
    chunks = clean_text.splitlines() or [""]
    width = max(1, int(terminal_width or 1))

    rows = 0
    for chunk in chunks:
        chunk_len = len(chunk)
        rows += max(1, (chunk_len + width - 1) // width)
    return rows


def get_ui_width(*, shutil_module: Any = None, max_width: int = DEFAULT_UI_WIDTH, min_width: int = 60) -> int:
    """
    Calcula un ancho consistente de UI adaptado al tamaño de terminal.
    
    Args:
        shutil_module (Any, optional): Módulo shutil inyectado para tests.
        max_width (int, optional): Ancho máximo de seguridad.
        min_width (int, optional): Ancho mínimo de seguridad.
    
    Returns:
        int: Ancho utilizable en el frame actual sin tope fijo.
    """
    width = int(max_width)
    try:
        module = shutil_module or shutil
        cols = int(module.get_terminal_size(fallback=(max_width + 4, 30)).columns)
        width = max(min_width, min(max_width, cols - 4))
    except Exception:
        width = int(max_width)
    return max(min_width, width)


def get_full_ui_width(*, shutil_module: Any = None, min_width: int = 60, side_padding: int = 4) -> int:
    """
    Calcula el ancho de UI ocupando todo el terminal disponible.
    
    Args:
        shutil_module (Any, optional): Módulo shutil inyectado para tests.
        min_width (int, optional): Ancho mínimo de seguridad.
        side_padding (int, optional): Margen horizontal reservado.

    Returns:
        int: Ancho utilizable en el frame actual sin tope fijo.
    """
    try:
        module = shutil_module or shutil
        cols = int(module.get_terminal_size(fallback=(120, 30)).columns)
    except Exception:
        cols = 120

    max_width = max(min_width, cols - max(0, int(side_padding)))
    return get_ui_width(shutil_module=shutil_module or shutil, max_width=max_width, min_width=min_width)


def render_title_banner(*, title: str, style: Any, width: int, left_margin: int = 1) -> None:
    """
    Imprime un banner rectangular alineado al ancho de separadores.
    
    Args:
        title (str): Título del banner.
        style (Any): Estilo ANSI para el banner.
        width (int): Ancho del banner.
        left_margin (int, optional): Margen izquierdo.
    """
    margin = " " * max(0, int(left_margin))
    content = str(title or "").strip()
    if len(content) > width:
        content = content[: width - 1] + "…"
    print(f"{margin}{style.BOLD}╔{'═' * width}╗{style.ENDC}")
    print(f"{margin}{style.BOLD}║{content:^{width}}║{style.ENDC}")
    print(f"{margin}{style.BOLD}╚{'═' * width}╝{style.ENDC}")


def wrap_plain_text(text: str, width: int) -> list[str]:
    """
    Envuelve texto plano sin códigos ANSI para evitar overflow visual.
    
    Args:
        text (str): Texto a envolver.
        width (int): Ancho máximo de la línea.
    
    Returns:
        list[str]: Lista de líneas envueltas.
    """
    raw = str(text or "")
    if width <= 4 or len(raw) <= width:
        return [raw]

    words = raw.split(" ")
    lines: list[str] = []
    current = ""

    for word in words:
        if not word:
            continue
        if len(word) > width:
            if current:
                lines.append(current)
                current = ""
            start = 0
            while start < len(word):
                lines.append(word[start : start + width])
                start += width
            continue

        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines or [""]


def compute_render_decision(
    *,
    dynamic_lines: list[str],
    terminal_width: int,
    prev_terminal_width: int | None,
    static_rendered: bool,
    force_full: bool = False,
) -> dict[str, int | bool]:
    """
    Calcula la estrategia de repaint para paneles dinámicos en terminal.

    Args:
        dynamic_lines (list[str]): Líneas a pintar en el bloque dinámico.
        terminal_width (int): Ancho actual de terminal en columnas.
        prev_terminal_width (int | None): Ancho previo conocido o None si no hay historial.
        static_rendered (bool): Indica si la parte estática ya fue renderizada.
        force_full (bool, optional): Solicita repaint completo explícito.

    Returns:
        dict[str, int | bool]:
            - width_changed (bool): True si cambió el ancho desde el frame anterior.
            - has_wrapped_lines (bool): True si alguna línea ocupa más de una fila visual.
            - current_dynamic_rows (int): Número total de filas visuales dinámicas.
            - effective_force_full (bool): True si debe forzarse repaint completo.
            - use_incremental_frame (bool): True si puede usarse reposicionamiento incremental.
    """
    width_changed = prev_terminal_width is not None and prev_terminal_width != terminal_width
    has_wrapped_lines = any(_visual_rows(line, terminal_width) > 1 for line in dynamic_lines)
    current_dynamic_rows = sum(_visual_rows(line, terminal_width) for line in dynamic_lines)

    # Si el ancho cambia, forzamos repaint completo (hard clear) para evitar basura en pantalla
    effective_force_full = force_full or not static_rendered or width_changed
    
    # Solo usamos incremental si el ancho NO ha cambiado y NO forzamos repaint completo
    use_incremental_frame = not effective_force_full

    return {
        "width_changed": width_changed,
        "has_wrapped_lines": has_wrapped_lines,
        "current_dynamic_rows": current_dynamic_rows,
        "effective_force_full": effective_force_full,
        "use_incremental_frame": use_incremental_frame,
    }


def should_repaint_static(
    *,
    use_incremental: bool,
    static_rendered: bool,
    force_full: bool = False,
    width_changed: bool = False,
    height_changed: bool = False,
    signature_changed: bool = False,
    has_wrapped_lines: bool = False,
) -> bool:
    """
    Determina si debe repintarse el bloque estático del panel actual.

    Args:
        use_incremental (bool): Si el flujo usa estrategia incremental.
        static_rendered (bool): Si el bloque estático ya fue pintado.
        force_full (bool, optional): Solicita repaint completo explícito.
        width_changed (bool, optional): Cambio de ancho de terminal.
        height_changed (bool, optional): Cambio de alto de terminal.
        signature_changed (bool, optional): Cambio en contenido estático (header/hints).
        has_wrapped_lines (bool, optional): Presencia de líneas envueltas en bloque dinámico.

    Returns:
        bool: True si corresponde limpiar y volver a dibujar el bloque estático.
    """
    if not use_incremental:
        return True
    if not static_rendered:
        return True
    # Si cambió el tamaño, SIEMPRE repintamos todo para limpiar la basura
    if force_full or width_changed or height_changed or signature_changed or has_wrapped_lines:
        return True
    return False


def paint_dynamic_lines(
    *,
    dynamic_lines: list[str],
    prev_dynamic_visual_rows: int,
    current_dynamic_rows: int,
    use_incremental_frame: bool,
) -> int:
    """
    Pinta el bloque dinámico del panel con limpieza de línea y backtracking opcional.

    Args:
        dynamic_lines (list[str]): Líneas dinámicas a imprimir.
        prev_dynamic_visual_rows (int): Filas dinámicas visuales del frame previo.
        current_dynamic_rows (int): Filas dinámicas visuales del frame actual.
        use_incremental_frame (bool): Si True, aplica backtracking incremental.

    Returns:
        int: Nuevo valor de `prev_dynamic_visual_rows`.
    """
    if use_incremental_frame and prev_dynamic_visual_rows > 0:
        # Movemos el cursor hacia arriba
        print(f"\033[{prev_dynamic_visual_rows}F", end="")

    for line in dynamic_lines:
        # Limpiamos la línea entera antes de pintar (crucial si el ancho ha cambiado ligeramente)
        print(f"\033[2K{line}")

    if use_incremental_frame and prev_dynamic_visual_rows > current_dynamic_rows:
        # Si el frame anterior era más largo, limpiamos las líneas sobrantes de abajo
        for _ in range(prev_dynamic_visual_rows - current_dynamic_rows):
            print("\033[2K")

    return current_dynamic_rows


class IncrementalPanelRenderer:
    """
    Renderer incremental para paneles de terminal con bloque estático + bloque dinámico.
    
    Optimiza el redibujado de la consola evitando limpiar todo el contenido cuando
    solo cambian ciertas líneas, reduciendo el parpadeo.
    
    Attributes:
        clear_screen_fn (Callable): Función para limpiar la pantalla.
        render_static_fn (Callable): Función para renderizar el contenido que no cambia.
    """

    def __init__(self, *, clear_screen_fn: Callable[[], None], render_static_fn: Callable[[], None]):
        """
        Inicializa el renderer incremental.

        Args:
            clear_screen_fn (Callable): Función sin argumentos para limpiar la pantalla.
            render_static_fn (Callable): Función sin argumentos que imprime la parte estática.
        """
        self.clear_screen_fn = clear_screen_fn
        self.render_static_fn = render_static_fn
        self.static_rendered = False
        self.prev_dynamic_visual_rows = 0
        self.prev_terminal_width: int | None = None
        self.prev_terminal_height: int | None = None

    def reset(self) -> None:
        """
        Invalida el caché visual para forzar repaint completo en próximo render.
        """
        self.static_rendered = False
        self.prev_dynamic_visual_rows = 0
        self.prev_terminal_width = None
        self.prev_terminal_height = None

    def render(self, dynamic_lines: list[str], *, force_full: bool = False) -> None:
        """
        Renderiza solo las líneas dinámicas, con repaint completo opcional.
        
        Args:
            dynamic_lines (list[str]): Líneas de texto que componen la parte dinámica.
            force_full (bool, optional): Si es True, fuerza un redibujado completo (clrs + static + dynamic).
        """
        terminal_size = shutil.get_terminal_size(fallback=(120, 30))
        terminal_width = terminal_size.columns
        terminal_height = terminal_size.lines
        height_changed = self.prev_terminal_height is not None and self.prev_terminal_height != terminal_height

        decision = compute_render_decision(
            dynamic_lines=dynamic_lines,
            terminal_width=terminal_width,
            prev_terminal_width=self.prev_terminal_width,
            static_rendered=self.static_rendered,
            force_full=force_full,
        )
        self.prev_terminal_width = terminal_width
        self.prev_terminal_height = terminal_height

        has_wrapped_lines = bool(decision["has_wrapped_lines"])
        current_dynamic_rows = int(decision["current_dynamic_rows"])
        effective_force_full = bool(decision["effective_force_full"])
        use_incremental_frame = bool(decision["use_incremental_frame"])

        # Si el renderer detecta que hay que limpiar la pantalla (por cambio de ancho, etc)
        if should_repaint_static(
            use_incremental=True,
            static_rendered=self.static_rendered,
            force_full=effective_force_full,
            width_changed=bool(decision["width_changed"]), # AÑADIDO: Pasamos el cambio de ancho
            height_changed=height_changed,
            has_wrapped_lines=has_wrapped_lines,
        ):
            self.clear_screen_fn()
            self.render_static_fn()
            self.static_rendered = True
            # Como acabamos de limpiar la pantalla, no hay filas previas que sobrescribir
            self.prev_dynamic_visual_rows = 0
            # Forzamos a que pinte secuencialmente sin subir el cursor
            use_incremental_frame = False

        self.prev_dynamic_visual_rows = paint_dynamic_lines(
            dynamic_lines=dynamic_lines,
            prev_dynamic_visual_rows=self.prev_dynamic_visual_rows,
            current_dynamic_rows=current_dynamic_rows,
            use_incremental_frame=use_incremental_frame,
        )


def clear_screen_ansi(*, os_module: Any, sys_module: Any) -> None:
    """
    Limpia la pantalla usando comandos del sistema operativo.
    
    Usa 'cls' en Windows y secuencias ANSI ('\033[H\033[2J') en otros sistemas.
    
    Args:
        os_module (Any): Módulo `os` inyectado.
        sys_module (Any): Módulo `sys` inyectado.
    """
    # En Windows priorizamos `cls` porque es el camino más estable entre
    # hosts de consola (PowerShell, Terminal integrado, etc.).
    if os_module.name == "nt":
        try:
            os_module.system("cls")
            return
        except Exception:
            # Fallback ANSI si `cls` no está disponible.
            pass

    try:
        # 2J limpia pantalla visible, 3J limpia scrollback, H reposiciona cursor.
        sys_module.stdout.write("\033[2J\033[3J\033[H")
        sys_module.stdout.flush()
        return
    except Exception:
        pass

    # Último recurso en POSIX si ANSI falla por algún motivo.
    if os_module.name != "nt":
        try:
            os_module.system("clear")
        except Exception:
            pass


def read_key(*, os_module: Any, msvcrt_module: Any) -> str | None:
    """
    Lee una tecla presionada por el usuario y devuelve un código unificado.
    
    Soporta teclas de dirección, ENTER, SPACE, ESC y detección de Ctrl+C.
    Específico para Windows usando `msvcrt`.
    
    Args:
        os_module (Any): Módulo `os` inyectado.
        msvcrt_module (Any): Módulo `msvcrt` inyectado (puede ser None en Linux/Mac).
        
    Returns:
        str | None: Código de tecla ('UP', 'DOWN', 'LEFT', 'RIGHT', 'ENTER', 'SPACE', 'ESC') o None.
        
    Raises:
        KeyboardInterrupt: Si se detecta Ctrl+C (byte \x03).
    """
    if os_module.name == "nt" and msvcrt_module:
        if hasattr(msvcrt_module, "kbhit") and callable(msvcrt_module.kbhit):
            if not msvcrt_module.kbhit():
                return None

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

    # POSIX path (Linux/macOS): non-blocking ANSI key parsing.
    if (
        os_module.name != "nt"
        and select is not None
        and _get_posix_tty_api() is not None
        and hasattr(sys.stdin, "isatty")
        and sys.stdin.isatty()
    ):
        tcgetattr, tcsetattr, _setcbreak, tcsadrain = _get_posix_tty_api() or (None, None, None, 1)
        if not callable(tcgetattr) or not callable(tcsetattr):
            return None
        try:
            fd = sys.stdin.fileno()
            old_settings_raw = tcgetattr(fd)
        except Exception:
            return None

        if not isinstance(old_settings_raw, (list, tuple)):
            return None

        old_settings = list(old_settings_raw)

        try:
            # Modo raw no-canonical + sin echo para evitar que flechas se impriman como ^[[A.
            new_settings = list(old_settings)
            if len(new_settings) >= 4:
                lflag = int(new_settings[3])
                if termios is not None:
                    lflag &= ~int(getattr(termios, "ICANON", 0))
                    lflag &= ~int(getattr(termios, "ECHO", 0))
                    lflag &= ~int(getattr(termios, "IEXTEN", 0))
                    lflag &= ~int(getattr(termios, "ISIG", 0))
                new_settings[3] = lflag
            if len(new_settings) >= 7:
                cc = list(new_settings[6])
                if termios is not None:
                    vmin_idx = int(getattr(termios, "VMIN", 6))
                    vtime_idx = int(getattr(termios, "VTIME", 5))
                    if 0 <= vmin_idx < len(cc):
                        cc[vmin_idx] = 0
                    if 0 <= vtime_idx < len(cc):
                        cc[vtime_idx] = 0
                new_settings[6] = cc
            tcsetattr(fd, tcsadrain, new_settings)
            readable, _, _ = select.select([sys.stdin], [], [], 0)
            if not readable:
                return None

            key = os.read(fd, 1)
            if not key:
                return None

            if key == b"\x03":
                raise KeyboardInterrupt
            if key in (b"\r", b"\n"):
                return "ENTER"
            if key == b" ":
                return "SPACE"
            if key == b"\x1b":
                seq = b""
                # Pequeña ventana para completar la secuencia ANSI de flechas.
                for _ in range(3):
                    more, _, _ = select.select([sys.stdin], [], [], 0.01)
                    if not more:
                        break
                    try:
                        next_chunk = os.read(fd, 1)
                    except Exception:
                        break
                    if not next_chunk:
                        break
                    seq += next_chunk
                if seq == b"[A":
                    return "UP"
                if seq == b"[B":
                    return "DOWN"
                if seq == b"[D":
                    return "LEFT"
                if seq == b"[C":
                    return "RIGHT"
                if seq == b"OA":
                    return "UP"
                if seq == b"OB":
                    return "DOWN"
                if seq == b"OD":
                    return "LEFT"
                if seq == b"OC":
                    return "RIGHT"
                return "ESC"

            try:
                char = key.decode("utf-8", errors="ignore")
            except Exception:
                return None
            if char:
                return char.upper() if char == " " else None
            return None
        finally:
            try:
                tcsetattr(fd, tcsadrain, old_settings)
            except Exception:
                pass
    return None


def log(*, style: Any, msg: str, level: str = "info") -> None:
    """
    Imprime mensajes de estado con formato y color.
    
    Args:
        style (Any): Clase con definiciones de estilos ANSI.
        msg (str): El mensaje a imprimir.
        level (str, optional): Nivel del mensaje ('info', 'success', 'error', 'warning', 'step').
    """
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


def ask_text(
    *,
    kit: Any,
    title: str,
    intro_lines: list[str] | None = None,
    prompt_label: str = "Entrada:",
    help_line: str = "[ENTER] confirmar · [Backspace] borrar · [ESC] cancelar",
    initial_value: str = "",
    allow_char_fn: Callable[[str], bool] | None = None,
    normalize_on_submit_fn: Callable[[str], str] | None = None,
    validate_on_submit_fn: Callable[[str], str | None] | None = None,
    render_extra_lines_fn: Callable[[str, int], list[str]] | None = None,
    render_static_top_fn: Callable[[int], None] | None = None,
    max_length: int | None = None,
    force_full_on_update: bool = False,
) -> str | None:
    """
    Renderiza un panel reactivo de entrada de texto reutilizable.

    Args:
        kit: Instancia de ``UIKit`` o compatible.
        title: Título del panel.
        intro_lines: Líneas introductorias bajo el título.
        prompt_label: Etiqueta del campo de entrada.
        help_line: Línea de ayuda de teclas.
        initial_value: Valor inicial del campo.
        allow_char_fn: Filtro opcional por carácter.
        normalize_on_submit_fn: Normalización previa a validar/devolver.
        validate_on_submit_fn: Validador; devuelve error o ``None``.
        render_extra_lines_fn: Callback de líneas dinámicas adicionales.
        render_static_top_fn: Callback estático opcional antes del título.
        max_length: Longitud maxima aceptada para el campo (sin truncar valor confirmado).
        force_full_on_update: Fuerza repaint completo para evitar ghosting.

    Returns:
        Texto confirmado o ``None`` si se cancela con ESC.
    """
    intro = list(intro_lines or [])

    def _clip_value_for_render(value: str, ui_width: int) -> str:
        """Recorta visualmente el valor para evitar wrap y repintado excesivo."""
        safe_width = max(8, int(ui_width) - 10)
        visible_limit = safe_width if max_length is None else min(safe_width, max(1, int(max_length)))
        if len(value) <= visible_limit:
            return value
        return "..." + value[-(visible_limit - 3):]

    def _apply_normalize(value: str) -> str:
        if normalize_on_submit_fn is None:
            return value
        return normalize_on_submit_fn(value)

    def _validate(value: str) -> str | None:
        if validate_on_submit_fn is None:
            return None
        return validate_on_submit_fn(value)

    if getattr(kit.os_module, "name", "") != "nt" or kit.msvcrt_module is None:
        while True:
            raw_value = kit.input(f"{prompt_label} ")
            if raw_value is None:
                return None
            value = str(raw_value)
            if allow_char_fn is not None and any(not allow_char_fn(ch) for ch in value):
                kit.log("Entrada invalida para este campo.", "warning")
                continue
            value = _apply_normalize(value)
            error = _validate(value)
            if error:
                kit.log(error, "warning")
                continue
            return value

    style = kit.style
    panel = kit.IncrementalPanelRenderer(clear_screen_fn=kit.clear, render_static_fn=lambda: None)
    current_value = str(initial_value or "")
    inline_error = ""
    previous_width: int | None = None
    needs_render = True

    while True:
        ui_width = kit.width()
        if previous_width is None:
            previous_width = ui_width
            needs_render = True
        elif previous_width != ui_width:
            panel.reset()
            previous_width = ui_width
            needs_render = True

        if needs_render:

            def _render_static() -> None:
                """Pinta cabecera fija del panel de entrada reactiva."""
                if render_static_top_fn is not None:
                    render_static_top_fn(ui_width)
                print(f"{style.BOLD} {title} {style.ENDC}")
                print(" " + kit.divider(ui_width))
                for line in intro:
                    print(" " + line)
                print(" " + kit.divider(ui_width))

            panel.render_static_fn = _render_static

            rendered_value = _clip_value_for_render(current_value, ui_width)
            dynamic_lines: list[str] = [
                f" {style.BOLD}{prompt_label}{style.ENDC} {style.WARNING}{rendered_value}{style.ENDC}",
                f" {style.DIM}{help_line}{style.ENDC}",
                kit.divider(ui_width),
            ]
            if inline_error:
                dynamic_lines.append(f" {style.FAIL}{inline_error}{style.ENDC}")
                dynamic_lines.append(kit.divider(ui_width))

            if render_extra_lines_fn is not None:
                dynamic_lines.extend(render_extra_lines_fn(current_value, ui_width))

            panel.render(dynamic_lines, force_full=force_full_on_update)
            needs_render = False

        if not hasattr(kit.msvcrt_module, "kbhit") or not kit.msvcrt_module.kbhit():
            time.sleep(0.03)
            continue

        try:
            key = kit.msvcrt_module.getwch()
        except Exception:
            time.sleep(0.03)
            continue

        inline_error = ""

        if key in ("\r", "\n"):
            candidate = _apply_normalize(current_value)
            error = _validate(candidate)
            if error:
                inline_error = error
                needs_render = True
                continue
            return candidate

        if key == "\x1b":
            return None

        if key in ("\x08", "\x7f"):
            new_value = current_value[:-1]
            if new_value != current_value:
                current_value = new_value
                needs_render = True
            continue

        if key in ("\x00", "\xe0"):
            try:
                _ = kit.msvcrt_module.getwch()
            except Exception:
                pass
            continue

        if key == "\x03":
            raise KeyboardInterrupt

        if key.isprintable():
            if allow_char_fn is not None and not allow_char_fn(key):
                inline_error = "Caracter no permitido para este campo."
                needs_render = True
                continue
            if max_length is not None and len(current_value) >= max(0, int(max_length)):
                inline_error = "Has alcanzado la longitud maxima permitida."
                needs_render = True
                continue
            current_value += key
            needs_render = True


def ask_choice(
    *,
    question: str,
    options: list[str],
    default_index: int,
    style: Any,
    read_key_fn: Callable[[], str | None],
    clear_screen_fn: Callable[[], None],
    info_text: str | Callable[..., str] = "",
    nav_hint_text: str = "Acciones: ←/→ cambiar opción · ENTER confirmar · ESC cancelar.",
) -> int | None:
    """
    Solicita al usuario elegir una opción horizontal con teclado.

    Args:
        question: Pregunta principal.
        options: Opciones renderizadas en formato "chips".
        default_index: Índice inicial seleccionado.
        style: Clase/objeto con estilos ANSI.
        read_key_fn: Lector de tecla no bloqueante.
        clear_screen_fn: Función para limpiar pantalla.
        info_text: Texto adicional fijo o dinámico.
        nav_hint_text: Ayuda de navegación mostrada en cabecera.

    Returns:
        Índice de opción confirmada o ``None`` si se cancela con ESC.
    """
    if not options:
        return None

    selected = max(0, min(int(default_index), len(options) - 1))
    info_scroll_offset = 0

    def _render_static() -> None:
        """Renderiza cabecera estática del selector."""
        content_width = get_full_ui_width(shutil_module=shutil)
        divider = "─" * (content_width + 2)
        for q_line in wrap_plain_text(str(question or ""), max(16, content_width)):
            print(f"{style.BOLD}{style.WARNING}{q_line}{style.ENDC}")
        print(divider)
        for hint_line in wrap_plain_text(nav_hint_text, max(16, content_width)):
            print(f"{style.DIM}{hint_line}{style.ENDC}")
        print(divider)

    panel = IncrementalPanelRenderer(clear_screen_fn=clear_screen_fn, render_static_fn=_render_static)
    last_render_signature: tuple[Any, ...] | None = None
    has_scrollable_info = False
    can_scroll_down = False

    ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')

    def get_real_visual_rows(text: str, width: int) -> int:
        """Calcula cuántas líneas físicas ocupa un texto ignorando los códigos de color."""
        clean_text = ANSI_ESCAPE.sub('', text)
        length = len(clean_text)
        if length == 0:
            return 1
        return (length + width - 10) // width

    while True:
        content_width = get_full_ui_width(shutil_module=shutil)
        
        # Obtener el tamaño real de la terminal (soluciona el bug de Windows que devuelve 30 por defecto)
        try:
            terminal_height = os.get_terminal_size(sys.stdout.fileno()).lines
        except OSError:
            terminal_height = int(shutil.get_terminal_size(fallback=(120, 30)).lines)
            
        divider = "─" * (content_width + 2)

        static_rows = (
            len(wrap_plain_text(str(question or ""), max(16, content_width)))
            + len(wrap_plain_text(nav_hint_text, max(16, content_width)))
            + 2  # divisores estáticos
        )

        rendered_options: list[str] =[]
        for idx, option in enumerate(options):
            label = f" {option.strip()} "
            if idx == selected:
                rendered_options.append(f"{style.SELECTED}{label}{style.ENDC}")
            else:
                rendered_options.append(label)

        dynamic_lines: list[str] =["", "   " + "    ".join(rendered_options), ""]

        resolved_info_text = ""
        if callable(info_text):
            try:
                resolved_info_text = str(info_text(content_width) or "")
            except TypeError:
                try:
                    resolved_info_text = str(info_text() or "")
                except Exception:
                    resolved_info_text = ""
            except Exception:
                resolved_info_text = ""
        else:
            resolved_info_text = str(info_text or "")

        info_lines =[line for line in resolved_info_text.splitlines() if line.strip()]
        if info_lines:
            rendered_info_lines: list[str] = []
            for info_line in info_lines:
                if "\x1b[" in info_line:
                    rendered_info_lines.append(info_line)
                else:
                    for wrapped in wrap_plain_text(info_line, max(16, content_width)):
                        rendered_info_lines.append(f"{style.DIM}{wrapped}{style.ENDC}")

            option_rows = sum(get_real_visual_rows(line, content_width) for line in dynamic_lines)
            max_dynamic_rows = max(4, terminal_height - static_rows - 1)
            info_budget_rows = max(0, max_dynamic_rows - option_rows - 2)

            total_info_rows = sum(get_real_visual_rows(line, content_width) for line in rendered_info_lines)
            has_scrollable_info = info_budget_rows > 0 and total_info_rows > info_budget_rows

            if has_scrollable_info:
                max_offset = max(0, len(rendered_info_lines) - 1)
                info_scroll_offset = max(0, min(info_scroll_offset, max_offset))
                actual_budget = max(0, info_budget_rows - 1)
            else:
                info_scroll_offset = 0
                actual_budget = info_budget_rows

            clipped_info_lines: list[str] =[]
            used_rows = 0
            start_idx = info_scroll_offset if has_scrollable_info else 0
            for line in rendered_info_lines[start_idx:]:
                line_rows = get_real_visual_rows(line, content_width)
                if used_rows + line_rows > actual_budget:
                    break
                clipped_info_lines.append(line)
                used_rows += line_rows

            if has_scrollable_info and info_budget_rows > 0:
                at_top = info_scroll_offset == 0
                consumed = start_idx + len(clipped_info_lines)
                at_bottom = consumed >= len(rendered_info_lines)
                can_scroll_down = not at_bottom
                scroll_hint = (
                    f"{style.DIM}Scroll resumen: {'↑' if not at_top else '·'} / {'↓' if not at_bottom else '·'}"
                    f"  (usa ↑/↓ para mover, ←/→ para Si/No){style.ENDC}"
                )
                clipped_info_lines.append(scroll_hint)
            else:
                can_scroll_down = False

            dynamic_lines.append(divider)
            dynamic_lines.extend(clipped_info_lines)
        else:
            has_scrollable_info = False
            can_scroll_down = False

        # --- RELLENO PARA ALTURA COMPLETA DE TERMINAL ---
        current_dynamic_rows = sum(get_real_visual_rows(line, content_width) for line in dynamic_lines)
        target_dynamic_rows = max(0, terminal_height - static_rows - 1)
        
        padding_rows = max(0, target_dynamic_rows - current_dynamic_rows - 1)
        
        for _ in range(padding_rows):
            dynamic_lines.append("")
        # --------------------------------------------------------------

        dynamic_lines.append(divider)

        render_signature: tuple[Any, ...] = (
            selected,
            content_width,
            tuple(options),
            tuple(info_lines),
            info_scroll_offset,
            terminal_height, # Añadido por si redimensionas la pantalla, para forzar redibujado
        )
        if last_render_signature != render_signature:
            panel.render(dynamic_lines, force_full=bool(info_lines))
            last_render_signature = render_signature

        key = read_key_fn()
        if key == "UP":
            if has_scrollable_info and info_scroll_offset > 0:
                info_scroll_offset -= 1
        elif key == "DOWN":
            if has_scrollable_info and can_scroll_down:
                info_scroll_offset += 1
        elif key == "LEFT":
            selected = (selected - 1) % len(options)
        elif key == "RIGHT":
            selected = (selected + 1) % len(options)
        elif key == "ENTER":
            clear_screen_fn()
            return selected
        elif key == "ESC":
            clear_screen_fn()
            return None
        elif key is None:
            time.sleep(0.03)


def ask_user(
    *,
    question: str,
    default: str,
    style: Any,
    read_key_fn: Callable[[], str | None],
    clear_screen_fn: Callable[[], None],
    info_text: str | Callable[..., str] = "",
) -> bool:
    """
    Solicita confirmación al usuario mediante una interfaz interactiva de selección.
    
    Permite navegar entre opciones Sí/No usando flechas y confirmar con ENTER.
    
    Args:
        question (str): La pregunta a mostrar al usuario.
        default (str): Opción por defecto ('y' o 'n').
        style (Any): Clase con definiciones de estilos ANSI.
        read_key_fn (Callable): Función para leer entrada de teclado.
        clear_screen_fn (Callable): Función para limpiar la pantalla.
        info_text (str | Callable, optional): Texto adicional dinámico o fijo a mostrar.
        
    Returns:
        bool: True si el usuario selecciona Sí, False en caso contrario o si cancela.
    """
    normalized_default = (default or "y").strip().lower()
    if normalized_default not in ("y", "n"):
        normalized_default = "y"

    selected_index = ask_choice(
        question=question,
        options=["Si", "No"],
        default_index=0 if normalized_default == "y" else 1,
        style=style,
        read_key_fn=read_key_fn,
        clear_screen_fn=clear_screen_fn,
        info_text=info_text,
    )
    if selected_index is None:
        return False
    return selected_index == 0


def input_with_esc(*, prompt: str, os_module: Any, msvcrt_module: Any) -> str | None:
    """
    Solicita entrada de texto con soporte para cancelación mediante ESC.
    
    Proporciona una experiencia de entrada mejorada en Windows, permitiendo
    cancelar la operación presionando la tecla Escape.
    
    Args:
        prompt (str): Texto a mostrar antes de la entrada.
        os_module (Any): Módulo `os` inyectado.
        msvcrt_module (Any): Módulo `msvcrt` inyectado (para Windows).
        
    Returns:
        str | None: El texto ingresado, o None si se canceló con ESC.
    """
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

    if (
        os_module.name != "nt"
        and _get_posix_tty_api() is not None
        and hasattr(sys.stdin, "isatty")
        and sys.stdin.isatty()
    ):
        print(prompt, end="", flush=True)
        buffer: list[str] = []
        tcgetattr, tcsetattr, setcbreak, tcsadrain = _get_posix_tty_api() or (None, None, None, 1)
        if not callable(tcgetattr) or not callable(tcsetattr) or not callable(setcbreak):
            value = input(prompt).strip()
            if value.lower() == "esc":
                return None
            return value
        try:
            fd = sys.stdin.fileno()
            old_settings = tcgetattr(fd)
            setcbreak(fd)
        except Exception:
            value = input(prompt).strip()
            if value.lower() == "esc":
                return None
            return value

        try:
            while True:
                key = os.read(fd, 1)
                if not key:
                    continue
                if key == b"\x1b":
                    print()
                    return None
                if key in (b"\r", b"\n"):
                    print()
                    return "".join(buffer).strip()
                if key in (b"\x08", b"\x7f"):
                    if buffer:
                        buffer.pop()
                        print("\b \b", end="", flush=True)
                    continue
                if key == b"\x03":
                    raise KeyboardInterrupt
                try:
                    char = key.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                if not char.isprintable():
                    continue
                buffer.append(char)
                print(char, end="", flush=True)
        finally:
            try:
                tcsetattr(fd, tcsadrain, old_settings)
            except Exception:
                pass

    value = input(prompt).strip()
    if value.lower() == "esc":
        return None
    return value


def wait_for_any_key(
    *,
    message: str,
    style: Any,
    os_module: Any,
    msvcrt_module: Any,
    get_width_fn: Callable[[], int] | None = None,
    on_resize_fn: Callable[[int], None] | None = None,
    poll_interval_seconds: float = 0.05,
) -> None:
    """
    Pausa la ejecución hasta que el usuario presione cualquier tecla.
    
    Args:
        message (str): Mensaje a mostrar al usuario.
        style (Any): Clase con definiciones de estilos ANSI.
        os_module (Any): Módulo `os` inyectado.
        msvcrt_module (Any): Módulo `msvcrt` inyectado (para Windows).
        get_width_fn (Callable[[], int] | None): Callback para obtener el ancho
            actual de terminal. Si se omite, no se monitoriza resize.
        on_resize_fn (Callable[[int], None] | None): Callback opcional que se
            ejecuta al detectar un cambio de ancho.
        poll_interval_seconds (float): Intervalo de sondeo no bloqueante.
    """
    def _print_wait_message() -> None:
        """Imprime el mensaje de espera en la línea actual."""
        print(f"\n{style.DIM}{message}{style.ENDC}", end="", flush=True)

    def _drain_pending_keys() -> bool:
        """Descarta teclas pendientes del buffer y devuelve si había contenido."""
        drained = False
        if not (os_module.name == "nt" and msvcrt_module):
            return drained
        if not (hasattr(msvcrt_module, "kbhit") and callable(msvcrt_module.kbhit)):
            return drained
        while msvcrt_module.kbhit():
            _ = msvcrt_module.getch()
            drained = True
        return drained

    _print_wait_message()

    if os_module.name == "nt" and msvcrt_module:
        had_pending_keys = _drain_pending_keys()
        debounce_until = time.monotonic() + 0.14 if had_pending_keys else 0.0
        previous_width = get_width_fn() if callable(get_width_fn) else None
        while True:
            if callable(get_width_fn):
                current_width = get_width_fn()
                if previous_width is None:
                    previous_width = current_width
                elif current_width != previous_width:
                    previous_width = current_width
                    if callable(on_resize_fn):
                        on_resize_fn(current_width)
                    _print_wait_message()

            if hasattr(msvcrt_module, "kbhit") and callable(msvcrt_module.kbhit):
                if msvcrt_module.kbhit():
                    if debounce_until and time.monotonic() < debounce_until:
                        _ = msvcrt_module.getch()
                        continue
                    _ = msvcrt_module.getch()
                    print()
                    return
                time.sleep(max(0.01, float(poll_interval_seconds)))
                continue

            _ = msvcrt_module.getch()
            print()
            return

    if (
        os_module.name != "nt"
        and _get_posix_tty_api() is not None
        and hasattr(sys.stdin, "isatty")
        and sys.stdin.isatty()
    ):
        tcgetattr, tcsetattr, setcbreak, tcsadrain = _get_posix_tty_api() or (None, None, None, 1)
        if not callable(tcgetattr) or not callable(tcsetattr) or not callable(setcbreak):
            input()
            return
        fd = None
        old_settings = None
        try:
            fd = sys.stdin.fileno()
            old_settings = tcgetattr(fd)
            setcbreak(fd)
            _ = os.read(fd, 1)
            print()
            return
        except Exception:
            pass
        finally:
            try:
                if old_settings is not None and fd is not None:
                    tcsetattr(fd, tcsadrain, old_settings)
            except Exception:
                pass

    input()


def run_cmd(
    *,
    cmd: str | list[str],
    critical: bool,
    style: Any,
    ask_user_fn: Callable[[str, str], bool],
    log_fn: Callable[[str, str], None],
) -> bool:
    """
    Ejecuta un comando de la shell y maneja errores.
    
    Permite reintentar si el comando es crítico.
    
    Args:
        cmd (str | list[str]): Comando a ejecutar (texto o argv).
        critical (bool): Si es True, preguntará al usuario si quiere reintentar en caso de fallo.
        style (Any): Clase con definiciones de estilos ANSI.
        ask_user_fn (Callable): Función para solicitar confirmación al usuario.
        log_fn (Callable): Función para logging.
        
    Returns:
        bool: True si el comando tuvo éxito, False si falló y no se recuperó.
    """
    print(f"{style.DIM}$ {cmd}{style.ENDC}")
    if isinstance(cmd, str):
        parsed_cmd = shlex.split(cmd, posix=(os.name != "nt"))
    else:
        parsed_cmd = list(cmd)

    if not parsed_cmd:
        log_fn("Empty command.", "error")
        return False

    try:
        subprocess.check_call(parsed_cmd)
        return True
    except FileNotFoundError:
        log_fn(f"Command not found: {parsed_cmd[0]}", "error")
        return False
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
