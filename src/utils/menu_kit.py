"""UIKit y AppContext: capa de abstracción para la UI de terminal.

``UIKit`` centraliza todos los helpers de renderizado: banners, tablas, menús,
separadores, log, input… permitiendo que cualquier módulo de ``utils/`` opere
con un único objeto en lugar de dicts ``ctx``.

``AppContext`` es un dataclass tipado con los callables y datos de dominio de
la aplicación (librerías externas, funciones de fix/check, registros, etc.).

Uso típico::

    from src.utils.menu_kit import UIKit, AppContext

    kit = UIKit(style=Style, os_module=os, sys_module=sys,
                shutil_module=shutil, msvcrt_module=msvcrt,
                cursor_memory=MENU_CURSOR_MEMORY)

    app = AppContext(REQUIRED_LIBS=REQUIRED_LIBS, ...)

    setup_diagnostics.run_diagnostics_ui(kit, app)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from . import setup_menu_engine
from .setup_ui_io import (
    IncrementalPanelRenderer as _IncrementalPanelRenderer,
    ask_user as _ask_user,
    clear_screen_ansi as _clear_screen_ansi,
    get_full_ui_width,
    input_with_esc as _input_with_esc,
    log as _log_fn,
    paint_dynamic_lines,
    read_key as _read_key_fn,
    render_title_banner as _render_title_banner,
    run_cmd as _run_cmd_fn,
    should_repaint_static,
    wait_for_any_key as _wait_for_any_key,
    wrap_plain_text,
    compute_render_decision,
)


# ---------------------------------------------------------------------------
# TableColumn / TableRow — primitivas para tablas simples e interactivas
# ---------------------------------------------------------------------------


@dataclass
class TableColumn:
    """Definición de una columna para tablas interactivas y estáticas.

    Attributes:
        label: Texto de cabecera.
        fixed_width: Ancho fijo en caracteres. Excluyente con ``width_ratio``.
        width_ratio: Fracción del espacio disponible entre columnas sin
            ``fixed_width`` (0.0–1.0). Si es ``None``, reparte equitativamente.
        min_width: Ancho mínimo garantizado independientemente del terminal.
    """

    label: str
    fixed_width: int | None = None
    width_ratio: float | None = None
    min_width: int = 8


@dataclass
class TableRow:
    """Fila de datos para :meth:`UIKit.build_table_items` y :meth:`UIKit.table_menu`.

    Attributes:
        cells: Texto de cada celda (mismo orden que ``columns``).
        action: Callable ejecutado al pulsar ENTER. ``None`` → fila no
            seleccionable (separador visual, título, etc.).
        description: Tooltip mostrado en el panel de descripción del menú.
        selected_cells: Textos alternativos cuando la fila está resaltada.
            ``None`` = usar los mismos ``cells``.
        cell_colors: Nombre del atributo de color en ``style`` por celda
            (p.ej. ``["OKGREEN", None, "DIM"]``). ``None`` = sin color.
        selected_cell_colors: Igual que ``cell_colors`` pero cuando la fila
            está resaltada.
    """

    cells: list[str]
    action: Callable | None = None
    description: str = ""
    selected_cells: list[str] | None = None
    cell_colors: list[str | None] | None = None
    selected_cell_colors: list[str | None] | None = None


def _compute_col_widths(columns: list[TableColumn], total_width: int) -> list[int]:
    """Calcula los anchos de columna ajustados al ancho de terminal.

    Columnas con ``fixed_width`` ocupan exactamente ese valor (respetando
    ``min_width``). Las columnas flexibles reparten el espacio restante
    proporcionalmente a ``width_ratio``; si ninguna lo define, a partes iguales.

    El separador `` │ `` entre columnas ocupa 3 chars; se descuenta del espacio
    disponible antes de asignar anchos.

    Args:
        columns: Definiciones de columna.
        total_width: Ancho total disponible en caracteres.

    Returns:
        list[int]: Ancho asignado a cada columna en el mismo orden.
    """
    n = len(columns)
    if n == 0:
        return []
    sep_overhead = 3 * (n - 1)
    available = max(n * 8, total_width - sep_overhead)
    widths = [0] * n
    remaining = available
    flex_indices: list[int] = []

    for i, col in enumerate(columns):
        if col.fixed_width is not None:
            w = max(col.min_width, col.fixed_width)
            widths[i] = w
            remaining -= w
        else:
            flex_indices.append(i)

    if flex_indices:
        total_ratio = sum(columns[i].width_ratio or 1.0 for i in flex_indices)
        for i in flex_indices:
            ratio = columns[i].width_ratio or 1.0
            w = max(columns[i].min_width, int(remaining * ratio / total_ratio))
            widths[i] = w

    return widths


def _build_table_menu_items(
    columns: list[TableColumn],
    rows: list[TableRow],
    *,
    style: Any,
    get_width_fn: Callable[[], int],
) -> list:
    """Construye la lista de ítems de menú para una tabla interactiva.

    Genera:

    - Un ítem de cabecera (``MenuStaticItem``) con ``dynamic_label`` que
      reajusta los anchos de columna al cambiar el terminal.
    - Un separador de apertura.
    - Un ítem por cada ``TableRow``: con ``MenuItem`` (seleccionable) si tiene
      ``action``, o ``MenuStaticItem`` (no seleccionable) si ``action=None``.
    - Un separador de cierre.

    La ``TableRow`` original se almacena en el atributo ``_table_row`` de cada
    ítem para recuperarla tras la selección en :meth:`UIKit.table_menu`.

    Args:
        columns: Definición de columnas.
        rows: Filas de datos.
        style: Objeto con constantes ANSI (``DIM``, ``ENDC``, etc.).
        get_width_fn: Callable sin argumentos que devuelve el ancho actual.

    Returns:
        list: Ítems listos para pasar a ``interactive_menu`` / ``UIKit.menu``.
    """
    pipe = " │ "

    def _row_str(cells: list[str], widths: list[int], colors: Sequence[str | None] | None) -> str:
        parts = []
        for j, (cell, w) in enumerate(zip(cells, widths)):
            text = str(cell)
            if len(text) > w:
                text = text[: w - 1] + "…"
            color_name = (colors[j] if colors and j < len(colors) else None) or ""
            color_code = getattr(style, color_name, "") if color_name else ""
            endc = style.ENDC if color_code else ""
            parts.append(f"{color_code}{text:<{w}}{endc}")
        return pipe.join(parts)

    # ── encabezado ──────────────────────────────────────────────────────
    header_item = setup_menu_engine.MenuStaticItem(label="", description="")

    def _header_lbl(_sel: bool) -> str:
        widths = _compute_col_widths(columns, get_width_fn())
        return f"{style.DIM}{_row_str([c.label for c in columns], widths, ['DIM'] * len(columns))}{style.ENDC}"

    header_item.dynamic_label = _header_lbl

    # ── separador superior ───────────────────────────────────────────────
    top_sep = setup_menu_engine.MenuStaticItem(label="", description="")

    def _top_lbl(_sel: bool) -> str:
        return f"{style.DIM}{'─' * get_width_fn()}{style.ENDC}"

    top_sep.dynamic_label = _top_lbl

    items: list = [header_item, top_sep]

    # ── filas de datos ───────────────────────────────────────────────────
    for row in rows:
        item: Any
        if row.action is not None:
            item = setup_menu_engine.MenuItem("", action=row.action, description=row.description)
        else:
            item = setup_menu_engine.MenuStaticItem(label="", description=row.description)

        _row = row  # captura en closure

        def _row_lbl(is_sel: bool, _r: TableRow = _row) -> str:
            widths = _compute_col_widths(columns, get_width_fn())
            cells = _r.selected_cells if (is_sel and _r.selected_cells is not None) else _r.cells
            colors = _r.selected_cell_colors if is_sel else _r.cell_colors
            return _row_str(cells, widths, colors)

        item.dynamic_label = _row_lbl
        item._table_row = row  # type: ignore[attr-defined]
        items.append(item)

    # ── separador inferior ───────────────────────────────────────────────
    bot_sep = setup_menu_engine.MenuStaticItem(label="", description="")

    def _bot_lbl(_sel: bool) -> str:
        return f"{style.DIM}{'─' * get_width_fn()}{style.ENDC}"

    bot_sep.dynamic_label = _bot_lbl
    items.append(bot_sep)

    return items


# ---------------------------------------------------------------------------
# UIKit
# ---------------------------------------------------------------------------


class UIKit:
    """Capa de alto nivel para toda la UI de terminal.

    Encapsula estilo ANSI, módulos del SO y helpers de renderizado en un único
    objeto reutilizable.  Todos los módulos de ``utils/`` reciben una instancia
    de esta clase en lugar de un dict ``ctx`` con referencias sueltas.

    Attributes:
        style: Objeto con constantes ANSI (BOLD, DIM, SELECTED, ENDC, …).
        os_module: Módulo ``os`` inyectado.
        sys_module: Módulo ``sys`` inyectado.
        shutil_module: Módulo ``shutil`` inyectado.
        msvcrt_module: Módulo ``msvcrt`` inyectado (puede ser ``None`` en Linux/Mac).
        cursor_memory (dict): Persistencia de posición de cursor por ``menu_id``.
    """

    # ── Referencia a clases de menú ─────────────────────────────────────
    MenuItem = setup_menu_engine.MenuItem
    MenuStaticItem = setup_menu_engine.MenuStaticItem
    MenuSeparator = setup_menu_engine.MenuSeparator
    IncrementalPanelRenderer = _IncrementalPanelRenderer

    # ── Primitivas de tabla re-exportadas ──────────────────────────────
    TableColumn = TableColumn  # type: ignore[assignment]  # re-export para callers que usen kit.TableColumn
    TableRow = TableRow        # type: ignore[assignment]

    # ── Helpers de bajo nivel re-exportados para módulos que los usen ──
    paint_dynamic_lines = staticmethod(paint_dynamic_lines)
    compute_render_decision = staticmethod(compute_render_decision)
    should_repaint_static = staticmethod(should_repaint_static)
    wrap_plain_text = staticmethod(wrap_plain_text)

    def __init__(
        self,
        *,
        style: Any,
        os_module: Any,
        sys_module: Any,
        shutil_module: Any,
        msvcrt_module: Any,
        cursor_memory: dict,
    ) -> None:
        self.style = style
        self.os_module = os_module
        self.sys_module = sys_module
        self.shutil_module = shutil_module
        self.msvcrt_module = msvcrt_module
        self.cursor_memory = cursor_memory

    # ── Geometría de terminal ───────────────────────────────────────────

    def width(self, *, min_width: int = 60, side_padding: int = 4) -> int:
        """Ancho de UI adaptado al terminal actual.

        Returns:
            int: Columnas usables tras restar ``side_padding``.
        """
        return get_full_ui_width(
            shutil_module=self.shutil_module,
            min_width=min_width,
            side_padding=side_padding,
        )

    # ── Ayudantes de texto ──────────────────────────────────────────────

    def wrap(self, text: str, width: int) -> list[str]:
        """Word-wrap sin secuencias ANSI.

        Args:
            text (str): Texto a envolver.
            width (int): Ancho máximo por línea.

        Returns:
            list[str]: Líneas resultantes.
        """
        return wrap_plain_text(text, width)

    # ── Primitivas de renderizado ───────────────────────────────────────

    def divider(self, width: int | None = None) -> str:
        """Cadena de separador horizontal ``─``.

        Args:
            width (int | None): Ancho interno del contenido.
                Si es ``None`` se usa ``self.width()``.

        Returns:
            str: Separador horizontal.
        """
        w = width if width is not None else self.width()
        return "─" * (max(8, int(w)) + 2)

    def print_divider(self, width: int | None = None) -> None:
        """Imprime un separador horizontal."""
        print(self.divider(width))

    def clear(self) -> None:
        """Limpia la pantalla del terminal."""
        _clear_screen_ansi(os_module=self.os_module, sys_module=self.sys_module)

    def log(self, msg: str, level: str = "info") -> None:
        """Imprime un mensaje formateado con icono y color.

        Args:
            msg (str): Texto del mensaje.
            level (str): ``'info'``, ``'success'``, ``'error'``, ``'warning'``
                o ``'step'``.
        """
        _log_fn(style=self.style, msg=msg, level=level)

    def banner(self, title: str, width: int | None = None) -> None:
        """Imprime un banner rectangular ``╔═╗``.

        Args:
            title (str): Título centrado en el banner.
            width (int | None): Ancho del banner. Por defecto ``self.width()``.
        """
        w = width if width is not None else self.width()
        _render_title_banner(title=title, style=self.style, width=w)

    def section_header(
        self,
        title: str,
        hint: str | None = None,
        width: int | None = None,
    ) -> None:
        """Banner + separador + línea DIM de ayuda opcional.

        Patrón muy repetido en los submenús:
        ``render_title_banner → divider → hint line``.

        Args:
            title (str): Título del banner.
            hint (str | None): Texto de ayuda en ``DIM`` bajo el separador.
            width (int | None): Ancho del frame.
        """
        w = width if width is not None else self.width()
        self.banner(title, width=w)
        self.print_divider(w)
        if hint:
            print(f"{self.style.DIM}{hint}{self.style.ENDC}")

    def subtitle(self, text: str) -> None:
        """Imprime un subtítulo en negrita.

        Args:
            text (str): Texto del subtítulo.
        """
        print(f"{self.style.BOLD} {text} {self.style.ENDC}")

    def table(
        self,
        rows: list[tuple[str, str, str]],
        *,
        width: int | None = None,
    ) -> None:
        """Imprime una tabla formateada ``┌┬┐├┼┤└┴┘``.

        Punto de entrada único para generar tablas desde cualquier módulo de
        menús: ``kit.table(rows)``.

        Cada fila es una tupla ``(nombre, valor, estado)`` donde ``estado``
        es ``'OK'``, ``'WARN'`` o ``'FAIL'``.

        Args:
            rows: Lista de tuplas ``(componente, valor, estado)``.
            width (int | None): Ancho de UI para calcular columnas.
        """
        ui_width = width if (width is not None and width > 0) else self.width()
        style = self.style
        _wrap = wrap_plain_text

        result_col = 6
        content_budget = max(32, ui_width - result_col - 6)
        component_col = max(16, int(content_budget * 0.45))
        value_col = max(16, content_budget - component_col)

        top    = "┌" + "─" * (component_col + 2) + "┬" + "─" * (value_col + 2) + "┬" + "─" * result_col + "┐"
        mid    = "├" + "─" * (component_col + 2) + "┼" + "─" * (value_col + 2) + "┼" + "─" * result_col + "┤"
        bottom = "└" + "─" * (component_col + 2) + "┴" + "─" * (value_col + 2) + "┴" + "─" * result_col + "┘"

        print(f"\n{style.HEADER}{top}{style.ENDC}")
        print(
            f"{style.HEADER}│ {'Component':<{component_col}} │ {'Status/Value':<{value_col}} │ {'Res':<{result_col - 2}} │{style.ENDC}"
        )
        print(f"{style.HEADER}{mid}{style.ENDC}")

        for name, value, status in rows:
            color = style.OKGREEN if status == "OK" else (style.WARNING if status == "WARN" else style.FAIL)
            display_name = str(name).replace("cv2", "OpenCV").replace("llama_cpp", "LlamaCPP")
            value_text = str(value)

            name_lines  = _wrap(display_name, component_col)
            value_lines = _wrap(value_text, value_col)
            row_count   = max(len(name_lines), len(value_lines))

            for row_idx in range(row_count):
                left  = name_lines[row_idx]  if row_idx < len(name_lines)  else ""
                right = value_lines[row_idx] if row_idx < len(value_lines) else ""
                if row_idx == 0:
                    res_text = f"{color}{status:<{result_col - 2}}{style.ENDC}"
                else:
                    res_text = " " * (result_col - 2)
                print(f"│ {left:<{component_col}} │ {right:<{value_col}} │ {res_text} │")

        print(f"{style.HEADER}{bottom}{style.ENDC}")

    def build_table_items(
        self,
        columns: list[TableColumn],
        rows: list[TableRow],
        *,
        width: int | None = None,
    ) -> list:
        """Construye ítems de menú para una tabla sin invocar el motor de menús.

        Permite mezclar filas de tabla con ítems de menú normales antes de
        llamar manualmente a ``kit.menu()``.

        Ejemplo::

            cols = [TableColumn("Modelo"), TableColumn("Estado", fixed_width=10)]
            rows = [TableRow(["modelo-x", "OK"], action=lambda: delete())]
            items = kit.build_table_items(cols, rows)
            items += [kit.MenuItem("Descargar…", action=download)]
            kit.menu(items, header_func=header, menu_id="my_table")

        Args:
            columns: Definición de columnas (``TableColumn``).
            rows: Filas de datos (``TableRow``).
            width: Ancho fijo. ``None`` = usa ``self.width()`` en tiempo real.

        Returns:
            list: Ítems listos para pasar a ``kit.menu()``.
        """
        def _get_w() -> int:
            return width if (width is not None and width > 0) else self.width()

        return _build_table_menu_items(columns, rows, style=self.style, get_width_fn=_get_w)

    def table_menu(
        self,
        columns: list[TableColumn],
        rows: list[TableRow],
        *,
        width: int | None = None,
        header_func: Callable[[], None] | None = None,
        menu_id: str | None = None,
        multi_select: bool = False,
        nav_hint_text: str | None = None,
        info_text: str | Callable[[], str] = "",
        dynamic_info_top: bool = False,
    ) -> "TableRow | list[TableRow] | None":
        """Tabla navegable controlada por teclado.

        Combina :meth:`build_table_items` con :meth:`menu` para mostrar una
        tabla interactiva. Las filas con ``action`` son seleccionables; el
        resto actúan como cabeceras o separadores no-seleccionables.

        El valor devuelto es la ``TableRow`` seleccionada (no el ``MenuItem``
        interno), o una lista de ``TableRow``s en modo multi-select.

        Cuando se necesita mezclar filas de tabla con otros ítems de menú
        usa :meth:`build_table_items` directamente y llama a ``kit.menu()``.

        Args:
            columns: Definición de columnas (``TableColumn``).
            rows: Filas de datos (``TableRow``).
            width: Ancho fijo. ``None`` = adapta al terminal en tiempo real.
            header_func: Función que imprime el encabezado estático del menú.
            menu_id: ID para persistir posición del cursor entre llamadas.
            multi_select: Selección múltiple con SPACE.
            nav_hint_text: Texto de ayuda de navegación.
            info_text: Texto informativo o callable.
            dynamic_info_top: Si ``True``, el panel de info aparece arriba.

        Returns:
            ``TableRow`` seleccionada, lista de ``TableRow``s (multi-select)
            o ``None`` si el usuario canceló con ESC.
        """
        items = self.build_table_items(columns, rows, width=width)
        result = self.menu(
            items,
            header_func=header_func,
            multi_select=multi_select,
            info_text=info_text,
            menu_id=menu_id,
            nav_hint=nav_hint_text is not None,
            nav_hint_text=nav_hint_text,
            dynamic_info_top=dynamic_info_top,
        )
        if result is None:
            return None
        if isinstance(result, list):
            return [getattr(r, "_table_row", r) for r in result]
        return getattr(result, "_table_row", result)

    # ── Interacción con el usuario ──────────────────────────────────────

    def ask(
        self,
        question: str,
        default: str = "y",
        info_text: str | Callable[[], str] = "",
    ) -> bool:
        """Diálogo interactivo Sí / No.

        Args:
            question (str): Pregunta a mostrar.
            default (str): ``'y'`` o ``'n'``.
            info_text: Texto adicional o callable que lo genera.

        Returns:
            bool: ``True`` si el usuario elige Sí.
        """
        return _ask_user(
            question=question,
            default=default,
            style=self.style,
            read_key_fn=self.read_key,
            clear_screen_fn=self.clear,
            info_text=info_text,
        )

    def input(self, prompt: str) -> str | None:
        """Solicita texto al usuario, cancelable con ESC.

        Args:
            prompt (str): Texto a mostrar antes del campo de entrada.

        Returns:
            str | None: Texto introducido, o ``None`` si se canceló.
        """
        return _input_with_esc(
            prompt=prompt,
            os_module=self.os_module,
            msvcrt_module=self.msvcrt_module,
        )

    def wait(self, message: str = "Press any key to return...") -> None:
        """Pausa la ejecución hasta que el usuario pulse una tecla.

        Args:
            message (str): Texto a mostrar.
        """
        _wait_for_any_key(
            message=message,
            style=self.style,
            os_module=self.os_module,
            msvcrt_module=self.msvcrt_module,
        )

    def read_key(self) -> str | None:
        """Lee una tecla del usuario (sin bloqueo si no hay entrada).

        Returns:
            str | None: Código de tecla o ``None``.
        """
        return _read_key_fn(os_module=self.os_module, msvcrt_module=self.msvcrt_module)

    def run_cmd(self, cmd: str, critical: bool = False) -> bool:
        """Ejecuta un comando de shell con gestión de errores.

        Args:
            cmd (str): Comando a ejecutar.
            critical (bool): Si es ``True``, pregunta al usuario si reintentar.

        Returns:
            bool: ``True`` si el comando tuvo éxito.
        """
        return _run_cmd_fn(
            cmd=cmd,
            critical=critical,
            style=self.style,
            ask_user_fn=self.ask,
            log_fn=self.log,
        )

    # ── Menú interactivo ────────────────────────────────────────────────

    def menu(
        self,
        options: list,
        *,
        header_func: Callable[[], None] | None = None,
        multi_select: bool = False,
        info_text: str | Callable[[], str] = "",
        menu_id: str | None = None,
        nav_hint: bool = True,
        left_margin: int = 0,
        nav_hint_text: str | None = None,
        sub_nav_hint_text: str | None = None,
        footer_hint_text: str | None = None,
        repaint_strategy: str = "auto",
        dynamic_info_top: bool = False,
        description_slot_rows: int = 2,
    ) -> Any:
        """Despliega un menú interactivo controlado por teclado.

        Delega a ``setup_menu_engine.interactive_menu`` inyectando
        automáticamente el estilo y los módulos del SO.

        Returns:
            El ``MenuItem`` seleccionado, una lista (multi-select) o ``None``.
        """
        return setup_menu_engine.interactive_menu(
            options,
            style=self.style,
            clear_screen_fn=self.clear,
            read_key_fn=self.read_key,
            get_item_description_fn=lambda item: str(getattr(item, "description", "") or "").strip(),
            cursor_memory=self.cursor_memory,
            os_module=self.os_module,
            sys_module=self.sys_module,
            shutil_module=self.shutil_module,
            header_func=header_func,
            multi_select=multi_select,
            info_text=info_text,
            menu_id=menu_id,
            nav_hint=nav_hint,
            left_margin=left_margin,
            nav_hint_text=nav_hint_text,
            sub_nav_hint_text=sub_nav_hint_text,
            footer_hint_text=footer_hint_text,
            repaint_strategy=repaint_strategy,
            dynamic_info_top=dynamic_info_top,
            description_slot_rows=description_slot_rows,
        )


# ---------------------------------------------------------------------------
# AppContext
# ---------------------------------------------------------------------------


@dataclass
class AppContext:
    """Contexto de dominio de la aplicación.

    Agrupa los callables y datos de dominio que los módulos de UI de
    ``utils/`` necesitan, eliminando la necesidad de pasar dicts ``ctx``
    con claves de cadena.

    Todos los campos son obligatorios excepto ``MENU_CURSOR_MEMORY``
    (por compatibilidad, aunque ``UIKit`` ya gestiona la memoria de cursor).
    """

    # ── Configuración estática ─────────────────────────────────────────
    REQUIRED_LIBS: list
    LIB_IMPORT_MAP: dict
    MODELS_REGISTRY: dict

    # ── Módulos externos inyectados ────────────────────────────────────
    lms_models: Any
    lms_menu_helpers: Any
    psutil: Any
    os_module: Any
    sys_module: Any
    subprocess_module: Any
    time_module: Any

    # ── UI de alto nivel ───────────────────────────────────────────────
    print_banner: Callable

    # ── Operaciones de dominio ─────────────────────────────────────────
    fix_folders: Callable
    fix_uv: Callable
    fix_libs: Callable
    check_lms: Callable
    restart_program: Callable
    get_installed_lms_models: Callable
    list_test_files: Callable
    ensure_lms_server_running: Callable
    stop_lms_server_if_owned: Callable

    # ── Campo opcional de compatibilidad ──────────────────────────────
    MENU_CURSOR_MEMORY: dict = field(default_factory=dict)
