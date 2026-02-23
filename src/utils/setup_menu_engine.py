"""Motor de menús interactivos para setup_env.

Contiene un `MenuItem` reutilizable y el render/input loop de `interactive_menu`.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from .setup_ui_io import (
    compute_render_decision,
    get_full_ui_width,
    get_ui_width,
    paint_dynamic_lines,
    should_repaint_static,
    wrap_plain_text,
)


class MenuItem:
    """
    Elemento genérico de menú con acción, descripción y posibles hijos.
    
    Representa una opción seleccionable en el menú interactivo.
    
    Attributes:
        label (str): Texto a mostrar para la opción.
        action (Callable, optional): Función a ejecutar al seleccionar.
        description (str): Texto explicativo mostrado al pie.
        children (list[MenuItem], optional): Submenú o opciones hijas.
        is_selected (bool): Estado de selección (para menús multi-select).
    """

    def __init__(self, label, action=None, description="", children=None):
        """
        Inicializa un elemento de menú.

        Args:
            label (str): Etiqueta visible del ítem.
            action (Callable, optional): Acción a ejecutar. Defaults to None.
            description (str, optional): Descripción extendida. Defaults to "".
            children (list[MenuItem], optional): Lista de sub-items. Defaults to None.
        """
        self.label = label
        self.action = action
        self.description = description
        self.children = children or []
        self.is_selected = False
        self.dynamic_label: Optional[Callable] = None


class MenuStaticItem(MenuItem):
    """
    Fila estática no seleccionable para insertar contenido visual en menús.
    
    Útil para mostrar títulos, separadores o texto informativo dentro de la lista.
    """

    def __init__(self, label: str = "", description: str = ""):
        """
        Inicializa un ítem estático.

        Args:
            label (str): Texto a mostrar. Defaults to "".
            description (str): Descripción (aunque no suele ser visible al no ser seleccionable).
        """
        super().__init__(label=label, action=None, description=description, children=None)
        self.selectable = False


class MenuSeparator(MenuStaticItem):
    """
    Separador horizontal no seleccionable con texto opcional.
    
    Dibuja una línea divisoria o un encabezado de sección.
    
    Attributes:
        text (str): Texto central del separador.
        width (int): Ancho total en caracteres.
        fill (str): Carácter de relleno (ej. '─').
    """

    def __init__(self, text: str = "", width: int = 79, fill: str = "─"):
        """
        Inicializa el separador.

        Args:
            text (str): Texto opcional centrado. Defaults to "".
            width (int): Longitud del separador. Defaults to 79.
            fill (str): Carácter repetido para la línea. Defaults to "─".
        """
        super().__init__(label="", description="")
        self.text = str(text or "").strip()
        self.width = max(8, int(width))
        self.fill = (fill or "─")[0]

        def _dynamic_label(_is_selected_row):
            if not self.text:
                return self.fill * self.width

            content = f" {self.text} "
            if len(content) >= self.width:
                return content[: self.width]

            remaining = self.width - len(content)
            left = remaining // 2
            right = remaining - left
            return f"{self.fill * left}{content}{self.fill * right}"

        self.dynamic_label = _dynamic_label


def interactive_menu(
    options,
    *,
    style: Any,
    clear_screen_fn: Callable[[], None],
    read_key_fn: Callable[[], str | None],
    get_item_description_fn: Callable[[Any], str],
    cursor_memory: dict[str, int],
    os_module: Any,
    sys_module: Any,
    shutil_module: Any,
    header_func=None,
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
) -> Any | list[Any] | None:
    """
    Despliega un menú interactivo controlado por teclado en la terminal.
    
    Soporta navegación jerárquica, selección simple/múltiple, renderizado incremental
    para reducir parpadeo y persistencia de la posición del cursor.
    
    Args:
        options (list[MenuItem]): Lista de objetos MenuItem a mostrar.
        style: Objeto con definiciones de estilo ANSI.
        clear_screen_fn (Callable): Función para limpiar la pantalla.
        read_key_fn (Callable): Función que devuelve pulsaciones de teclas ('UP', 'DOWN', 'ENTER', etc.).
        get_item_description_fn (Callable): Función para extraer descripción de un ítem seleccionado.
        cursor_memory (dict): Diccionario para recordar la última posición del cursor por menu_id.
        os_module: Módulo `os` inyectado.
        sys_module: Módulo `sys` inyectado.
        shutil_module: Módulo `shutil` inyectado (para obtener tamaño de terminal).
        header_func (Callable, optional): Función que imprime un encabezado estático.
        multi_select (bool, optional): Habilita selección múltiple con SPACE. Defaults to False.
        info_text (str, optional): Texto informativo adicional. Defaults to "".
        menu_id (str, optional): ID único para persistir cursor. Defaults to None.
        nav_hint (bool, optional): Mostrar ayuda de navegación. Defaults to True.
        left_margin (int, optional): Espaciado izquierdo. Defaults to 0.
        nav_hint_text (str, optional): Texto personalizado para ayuda de navegación.
        sub_nav_hint_text (str, optional): Texto personalizado para ayuda en submenús.
        footer_hint_text (str, optional): Texto personalizado para pie de página.
        repaint_strategy (str, optional): Estrategia de repintado ('auto', 'incremental').
        dynamic_info_top (bool, optional): Si True, pinta `info_text` en bloque dinámico superior.
        
    Returns:
        Any | list[Any] | None: 
            - El objeto asociado al MenuItem seleccionado (single select).
            - Lista de objetos seleccionados (multi select).
            - None si se cancela con ESC.
    """
    current_row = cursor_memory.pop(menu_id, 0) if menu_id else 0
    in_sub_nav = False

    def _is_selectable(row):
        item = row.get("obj")
        return bool(getattr(item, "selectable", True))

    def _find_next_selectable(flat_rows, start_idx, step):
        if not flat_rows:
            return None
        total = len(flat_rows)
        idx = start_idx % total
        for _ in range(total):
            idx = (idx + step) % total
            if _is_selectable(flat_rows[idx]):
                return idx
        return None

    margin = " " * max(0, int(left_margin))
    divider = margin + ("─" * (get_full_ui_width(shutil_module=shutil_module) + 2))
    use_incremental = repaint_strategy in ("auto", "incremental")
    static_rendered = False
    prev_dynamic_visual_rows = 0
    prev_term_height = None
    prev_term_width = None
    prev_static_signature = None

    def _persist_cursor(flat_rows, row_index):
        if not menu_id or not flat_rows:
            return

        safe_index = row_index % len(flat_rows)
        row = flat_rows[safe_index]

        if row.get("level", 0) > 0 and row.get("parent") is not None:
            parent = row.get("parent")
            for idx, candidate in enumerate(flat_rows):
                if candidate.get("obj") == parent and candidate.get("level") == 0:
                    cursor_memory[menu_id] = idx
                    return

        cursor_memory[menu_id] = safe_index

    print("\033[?25l", end="")

    if os_module.name == "nt":
        os_module.system("cls")
    else:
        sys_module.stdout.write("\033[H\033[2J")

    try:
        while True:
            flat_rows = []
            for opt in options:
                flat_rows.append({"obj": opt, "level": 0})
                if hasattr(opt, "children") and opt.children:
                    for child in opt.children:
                        flat_rows.append({"obj": child, "level": 1, "parent": opt})

            if not flat_rows:
                if menu_id:
                    cursor_memory[menu_id] = 0
                return None
            current_row = current_row % len(flat_rows)

            if not any(_is_selectable(row) for row in flat_rows):
                if menu_id:
                    cursor_memory[menu_id] = 0
                return None

            if not _is_selectable(flat_rows[current_row]):
                next_idx = _find_next_selectable(flat_rows, current_row, 1)
                if next_idx is None:
                    return None
                current_row = next_idx

            term_height = 24
            term_width = 120
            if shutil_module:
                term_size = shutil_module.get_terminal_size()
                term_height = term_size.lines
                term_width = term_size.columns

            content_width = get_full_ui_width(shutil_module=shutil_module)
            divider = margin + ("─" * (content_width + 2))

            height_changed = False
            if prev_term_height is None:
                prev_term_height = term_height
            elif prev_term_height != term_height:
                static_rendered = False
                prev_term_height = term_height
                height_changed = True

            reserved_lines = 15
            max_visible_items = max(5, term_height - reserved_lines)

            half_view = max_visible_items // 2
            start_row = max(0, current_row - half_view)
            end_row = start_row + max_visible_items

            if end_row > len(flat_rows):
                end_row = len(flat_rows)
                start_row = max(0, end_row - max_visible_items)

            visible_slice = flat_rows[start_row:end_row]
            cur_level = flat_rows[current_row]["level"]

            dynamic_lines: list[str] = []

            if start_row > 0:
                hidden_above = sum(1 for r in flat_rows[:start_row] if r["level"] == cur_level)
                msg = f"▲ ({hidden_above}) ▲" if hidden_above > 0 else "▲"
                dynamic_lines.append(f"{margin}{style.DIM}{msg}{style.ENDC}")
            else:
                dynamic_lines.append(margin)

            for i, row in enumerate(visible_slice):
                idx = start_row + i
                is_selected_row = idx == current_row and _is_selectable(flat_rows[idx])
                item = row["obj"]
                level = row["level"]

                checkbox = ""
                if multi_select:
                    is_checked = getattr(item, "is_selected", False)
                    checkbox = f"[{'x' if is_checked else ' '}] "

                pointer = ">" if is_selected_row else " "
                indent = "    " * level
                suffix = " >" if level == 0 and hasattr(item, "children") and item.children else ""
                if hasattr(item, "dynamic_label") and callable(item.dynamic_label):
                    label = item.dynamic_label(is_selected_row)
                else:
                    label = getattr(item, "label", getattr(item, "fix_name", str(item)))

                label_text = str(label)
                selection_width = max(20, content_width - 4)
                base_prefix = f"{pointer} {indent}{checkbox}"
                continuation_prefix = f"  {indent}{' ' * len(checkbox)}"
                available_first = max(12, selection_width - len(base_prefix) - len(suffix))
                available_next = max(12, selection_width - len(continuation_prefix))

                if "\x1b[" in label_text:
                    wrapped_label = [label_text]
                else:
                    wrapped_label = wrap_plain_text(label_text, available_first)

                first_line = f"{base_prefix}{wrapped_label[0]}{suffix}"
                if is_selected_row:
                    dynamic_lines.append(f"{margin}{style.SELECTED}{first_line:<{selection_width}}{style.ENDC}")
                else:
                    dynamic_lines.append(f"{margin}{first_line}")

                for extra_line in wrapped_label[1:]:
                    if "\x1b[" in extra_line:
                        rendered_extra = f"{continuation_prefix}{extra_line}"
                    else:
                        wrapped_extra = wrap_plain_text(extra_line, available_next)
                        for idx_extra, chunk in enumerate(wrapped_extra):
                            rendered_extra = f"{continuation_prefix}{chunk}" if idx_extra == 0 else f"{continuation_prefix}{chunk}"
                            if is_selected_row:
                                dynamic_lines.append(f"{margin}{style.SELECTED}{rendered_extra:<{selection_width}}{style.ENDC}")
                            else:
                                dynamic_lines.append(f"{margin}{rendered_extra}")
                        continue

                    if is_selected_row:
                        dynamic_lines.append(f"{margin}{style.SELECTED}{rendered_extra:<{selection_width}}{style.ENDC}")
                    else:
                        dynamic_lines.append(f"{margin}{rendered_extra}")

            if end_row < len(flat_rows):
                hidden_below = sum(1 for r in flat_rows[end_row:] if r["level"] == cur_level)
                msg = f"▼ ({hidden_below}) ▼" if hidden_below > 0 else "▼"
                dynamic_lines.append(f"{margin}{style.DIM}{msg}{style.ENDC}")
            else:
                dynamic_lines.append(margin)

            selected_item = flat_rows[current_row]["obj"]
            selected_description = get_item_description_fn(selected_item)
            dynamic_lines.append(divider)
            description_slot_rows = 2
            if selected_description:
                desc_prefix = "Descripción: "
                desc_lines = wrap_plain_text(str(selected_description), max(16, content_width - len(desc_prefix) - 2))
                shown_desc_lines = desc_lines[:description_slot_rows]
                dynamic_lines.append(f"{margin}{style.DIM}{desc_prefix}{shown_desc_lines[0]}{style.ENDC}")
                for extra_desc in shown_desc_lines[1:]:
                    dynamic_lines.append(f"{margin}{style.DIM}{' ' * len(desc_prefix)}{extra_desc}{style.ENDC}")
                for _ in range(max(0, description_slot_rows - len(shown_desc_lines))):
                    dynamic_lines.append(margin)
            else:
                for _ in range(description_slot_rows):
                    dynamic_lines.append(margin)

            dynamic_lines.append(divider)
            if multi_select:
                res = []

                def collect(items):
                    for itm in items:
                        if itm.is_selected:
                            res.append(itm)
                        if hasattr(itm, "children"):
                            collect(itm.children)

                collect(options)
                dynamic_lines.append(f"{margin}{style.BOLD}[ENTER]: Confirmar selección ({len(res)} elementos).{style.ENDC}")
            else:
                footer_text = footer_hint_text or "[ENTER]: Seleccionar opción."
                footer_lines = wrap_plain_text(str(footer_text), max(16, content_width - 2))
                for footer_line in footer_lines:
                    dynamic_lines.append(f"{margin}{style.BOLD}{footer_line}{style.ENDC}")

            info_text_callable = callable(info_text)
            if info_text_callable:
                try:
                    resolved_info_text = str(info_text() or "")
                except Exception:
                    resolved_info_text = ""
            else:
                resolved_info_text = str(info_text or "")
            resolved_info_lines = [line for line in resolved_info_text.splitlines() if line.strip()]
            wrapped_info_lines: list[str] = []
            info_width = max(16, content_width - 2)
            for info_line in resolved_info_lines:
                wrapped_info_lines.extend(wrap_plain_text(info_line, info_width))

            dynamic_top_lines: list[str] = []
            if dynamic_info_top and wrapped_info_lines:
                dynamic_top_lines.append(divider)
                for info_line in wrapped_info_lines:
                    dynamic_top_lines.append(f"{margin}{info_line}")
                dynamic_top_lines.append(divider)

            if dynamic_info_top and wrapped_info_lines:
                dynamic_lines = [*dynamic_top_lines, *dynamic_lines]

            show_hint_section = ((bool(wrapped_info_lines) and not dynamic_info_top) or nav_hint)
            static_info_signature: Any
            if wrapped_info_lines and not dynamic_info_top:
                static_info_signature = tuple(wrapped_info_lines)
            elif wrapped_info_lines and dynamic_info_top:
                static_info_signature = "<dynamic-info-top>"
            else:
                static_info_signature = tuple()
            static_signature = (
                bool(show_hint_section),
                static_info_signature,
                bool(nav_hint),
                bool(in_sub_nav),
                str(nav_hint_text or ""),
                str(sub_nav_hint_text or ""),
            )
            if prev_static_signature is None:
                prev_static_signature = static_signature
                signature_changed = False
            elif prev_static_signature != static_signature:
                static_rendered = False
                prev_static_signature = static_signature
                signature_changed = True
            else:
                signature_changed = False

            decision = compute_render_decision(
                dynamic_lines=dynamic_lines,
                terminal_width=term_width,
                prev_terminal_width=prev_term_width,
                static_rendered=static_rendered,
                force_full=False,
            )
            prev_term_width = term_width

            if bool(decision["width_changed"]):
                static_rendered = False
                prev_dynamic_visual_rows = 0

            if should_repaint_static(
                use_incremental=use_incremental,
                static_rendered=static_rendered,
                force_full=bool(decision["effective_force_full"]),
                width_changed=bool(decision["width_changed"]),
                height_changed=height_changed,
                signature_changed=signature_changed,
                has_wrapped_lines=bool(decision["has_wrapped_lines"]),
            ):
                clear_screen_fn()
                if header_func:
                    header_func()
                if show_hint_section:
                    print(divider)
                    if wrapped_info_lines and not dynamic_info_top:
                        for info_line in wrapped_info_lines:
                            print(f"{margin}{style.DIM}{info_line}{style.ENDC}")

                    if nav_hint:
                        if in_sub_nav:
                            hint_text = sub_nav_hint_text or "[SUB-NAV] Arriba/Abajo: Navegar cuantizaciones, ENTER: Seleccionar, ESC: Volver."
                        else:
                            hint_text = nav_hint_text or "Arriba/Abajo: Navegar, SPACE/ENTER: Seleccionar/Entrar, ESC: Volver."
                        for hint_line in wrap_plain_text(str(hint_text), max(16, content_width - 2)):
                            print(f"{margin}{style.DIM}{hint_line}{style.ENDC}")

                    print(divider)

                if use_incremental:
                    static_rendered = True
                    prev_dynamic_visual_rows = 0

            use_incremental_frame = use_incremental and bool(decision["use_incremental_frame"])
            current_dynamic_rows = int(decision["current_dynamic_rows"])

            prev_dynamic_visual_rows = paint_dynamic_lines(
                dynamic_lines=dynamic_lines,
                prev_dynamic_visual_rows=prev_dynamic_visual_rows,
                current_dynamic_rows=current_dynamic_rows,
                use_incremental_frame=use_incremental_frame,
            )

            key = read_key_fn()
            if key is None:
                time.sleep(0.03)
                continue

            if key == "UP":
                if in_sub_nav:
                    current_parent = flat_rows[current_row].get("parent")
                    if current_parent:
                        siblings = [i for i, r in enumerate(flat_rows) if r.get("parent") == current_parent]
                        if siblings:
                            try:
                                rel_idx = siblings.index(current_row)
                                new_rel_idx = (rel_idx - 1) % len(siblings)
                                current_row = siblings[new_rel_idx]
                            except ValueError:
                                pass
                else:
                    next_row = (current_row - 1) % len(flat_rows)
                    while flat_rows[next_row]["level"] > 0 or not _is_selectable(flat_rows[next_row]):
                        next_row = (next_row - 1) % len(flat_rows)
                    current_row = next_row

            elif key == "DOWN":
                if in_sub_nav:
                    current_parent = flat_rows[current_row].get("parent")
                    if current_parent:
                        siblings = [i for i, r in enumerate(flat_rows) if r.get("parent") == current_parent]
                        if siblings:
                            rel_idx = siblings.index(current_row)
                            new_rel_idx = (rel_idx + 1) % len(siblings)
                            current_row = siblings[new_rel_idx]
                else:
                    next_row = (current_row + 1) % len(flat_rows)
                    while flat_rows[next_row]["level"] > 0 or not _is_selectable(flat_rows[next_row]):
                        next_row = (next_row + 1) % len(flat_rows)
                    current_row = next_row

            elif key == "SPACE" and multi_select:
                row = flat_rows[current_row]
                item = row["obj"]

                if row["level"] == 0:
                    if hasattr(item, "children") and item.children:
                        in_sub_nav = True
                        for i, r in enumerate(flat_rows):
                            if r["level"] == 1 and r.get("parent") == item:
                                current_row = i
                                break
                    else:
                        item.is_selected = not item.is_selected
                else:
                    item.is_selected = not item.is_selected

            elif key == "SPACE" and not multi_select:
                row = flat_rows[current_row]
                item = row["obj"]
                if row["level"] == 0 and hasattr(item, "children") and item.children:
                    in_sub_nav = True
                    for i, r in enumerate(flat_rows):
                        if r["level"] == 1 and r.get("parent") == item:
                            current_row = i
                            break

            elif key == "ENTER":
                row = flat_rows[current_row]
                item = row["obj"]

                if not _is_selectable(row):
                    continue

                if multi_select:
                    selected_objs = []

                    def collect(items):
                        for itm in items:
                            if itm.is_selected:
                                selected_objs.append(itm)
                            if hasattr(itm, "children"):
                                collect(itm.children)

                    collect(options)
                    _persist_cursor(flat_rows, current_row)
                    return selected_objs

                if row["level"] == 0 and hasattr(item, "children") and item.children and not in_sub_nav:
                    in_sub_nav = True
                    for i, r in enumerate(flat_rows):
                        if r["level"] == 1 and r.get("parent") == item:
                            current_row = i
                            break
                    continue

                _persist_cursor(flat_rows, current_row)
                return item

            elif key == "ESC":
                if in_sub_nav:
                    in_sub_nav = False
                    if flat_rows[current_row]["level"] > 0:
                        parent = flat_rows[current_row].get("parent")
                        if parent:
                            for i, r in enumerate(flat_rows):
                                if r["obj"] == parent:
                                    current_row = i
                                    break
                else:
                    _persist_cursor(flat_rows, current_row)
                    return None
            else:
                time.sleep(0.01)
    finally:
        print("\033[?25h", end="")
