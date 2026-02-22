"""Motor de menús interactivos para setup_env.

Contiene un `MenuItem` reutilizable y el render/input loop de `interactive_menu`.
"""

from __future__ import annotations

from typing import Any, Callable, Optional


class MenuItem:
    """Elemento genérico de menú con acción, descripción y posibles hijos."""

    def __init__(self, label, action=None, description="", children=None):
        self.label = label
        self.action = action
        self.description = description
        self.children = children or []
        self.is_selected = False
        self.dynamic_label: Optional[Callable] = None


class MenuStaticItem(MenuItem):
    """Fila estática no seleccionable para insertar contenido visual en menús."""

    def __init__(self, label: str = "", description: str = ""):
        super().__init__(label=label, action=None, description=description, children=None)
        self.selectable = False


class MenuSeparator(MenuStaticItem):
    """Separador horizontal no seleccionable con texto opcional."""

    def __init__(self, text: str = "", width: int = 79, fill: str = "─"):
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
    info_text: str = "",
    menu_id: str | None = None,
    nav_hint: bool = True,
    left_margin: int = 0,
    nav_hint_text: str | None = None,
    sub_nav_hint_text: str | None = None,
    footer_hint_text: str | None = None,
    repaint_strategy: str = "auto",
):
    """Menú interactivo genérico controlado por teclado."""
    current_row = cursor_memory.get(menu_id, 0) if menu_id else 0
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
    divider = margin + ("─" * 79)
    use_incremental = repaint_strategy in ("auto", "incremental")
    static_rendered = False
    prev_dynamic_line_count = 0
    prev_term_height = None
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
            if shutil_module:
                term_height = shutil_module.get_terminal_size().lines

            if prev_term_height is None:
                prev_term_height = term_height
            elif prev_term_height != term_height:
                static_rendered = False
                prev_term_height = term_height

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

                line_content = f"{pointer} {indent}{checkbox}{label}{suffix}"
                if is_selected_row:
                    dynamic_lines.append(f"{margin}{style.SELECTED}{line_content:<60}{style.ENDC}")
                else:
                    dynamic_lines.append(f"{margin}{line_content}")

            if end_row < len(flat_rows):
                hidden_below = sum(1 for r in flat_rows[end_row:] if r["level"] == cur_level)
                msg = f"▼ ({hidden_below}) ▼" if hidden_below > 0 else "▼"
                dynamic_lines.append(f"{margin}{style.DIM}{msg}{style.ENDC}")
            else:
                dynamic_lines.append(margin)

            selected_item = flat_rows[current_row]["obj"]
            selected_description = get_item_description_fn(selected_item)
            dynamic_lines.append(divider)
            if selected_description:
                dynamic_lines.append(f"{margin}{style.DIM}Descripción: {selected_description}{style.ENDC}")

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
                dynamic_lines.append(f"{margin}{style.BOLD}{footer_text}{style.ENDC}")

            show_hint_section = bool(info_text) or nav_hint
            static_signature = (
                bool(show_hint_section),
                bool(info_text),
                bool(nav_hint),
                bool(in_sub_nav),
                str(nav_hint_text or ""),
                str(sub_nav_hint_text or ""),
            )
            if prev_static_signature is None:
                prev_static_signature = static_signature
            elif prev_static_signature != static_signature:
                static_rendered = False
                prev_static_signature = static_signature

            if not use_incremental or not static_rendered:
                clear_screen_fn()
                if header_func:
                    header_func()
                if show_hint_section:
                    print(divider)
                    if info_text:
                        print(f"{margin}{info_text}")

                    if nav_hint:
                        if in_sub_nav:
                            hint_text = sub_nav_hint_text or "[SUB-NAV] Arriba/Abajo: Navegar cuantizaciones, ENTER: Seleccionar, ESC: Volver."
                        else:
                            hint_text = nav_hint_text or "Arriba/Abajo: Navegar, SPACE/ENTER: Seleccionar/Entrar, ESC: Volver."
                        print(f"{margin}{style.DIM}{hint_text}{style.ENDC}")

                    print(divider)

                if use_incremental:
                    static_rendered = True
                    prev_dynamic_line_count = 0

            if use_incremental and static_rendered and prev_dynamic_line_count > 0:
                print(f"\033[{prev_dynamic_line_count}F", end="")

            for line in dynamic_lines:
                print(f"\033[2K{line}")

            if use_incremental and prev_dynamic_line_count > len(dynamic_lines):
                for _ in range(prev_dynamic_line_count - len(dynamic_lines)):
                    print("\033[2K")

            prev_dynamic_line_count = len(dynamic_lines)

            key = read_key_fn()
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
    finally:
        print("\033[?25h", end="")
