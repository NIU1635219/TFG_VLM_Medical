"""UI del LM Studio Model Manager extraída de setup_env."""

from __future__ import annotations

import sys
import time
from typing import Any


class _FallbackIncrementalPanelRenderer:
    """Fallback local cuando no se inyecta IncrementalPanelRenderer desde setup_ui_io."""

    def __init__(self, *, clear_screen_fn, render_static_fn):
        self.clear_screen_fn = clear_screen_fn
        self.render_static_fn = render_static_fn
        self.static_rendered = False
        self.prev_dynamic_line_count = 0

    def reset(self):
        self.static_rendered = False
        self.prev_dynamic_line_count = 0

    def render(self, dynamic_lines, *, force_full=False):
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


def manage_models_menu_ui(ctx: dict[str, Any]) -> None:
    """Menú de gestión de modelos LM Studio (limpio + cuantizaciones)."""
    Style = ctx["Style"]
    MenuItem = ctx["MenuItem"]
    MenuStaticItem = ctx.get("MenuStaticItem", MenuItem)
    MenuSeparator = ctx.get("MenuSeparator", MenuItem)
    incremental_panel_renderer_cls = ctx.get("IncrementalPanelRenderer") or _FallbackIncrementalPanelRenderer
    lms_models = ctx["lms_models"]
    lms_menu_helpers = ctx["lms_menu_helpers"]
    psutil = ctx["psutil"]
    MODELS_REGISTRY = ctx["MODELS_REGISTRY"]
    MENU_CURSOR_MEMORY = ctx["MENU_CURSOR_MEMORY"]

    print_banner = ctx["print_banner"]
    clear_screen_ansi = ctx["clear_screen_ansi"]
    input_with_esc = ctx["input_with_esc"]
    wait_for_any_key = ctx["wait_for_any_key"]
    log = ctx["log"]
    ask_user = ctx["ask_user"]
    interactive_menu = ctx["interactive_menu"]
    get_installed_lms_models = ctx["get_installed_lms_models"]
    read_key = ctx["read_key"]

    gpu_mem_bytes = lms_menu_helpers.detect_gpu_memory_bytes()
    ram_mem_bytes = lms_menu_helpers.detect_ram_memory_bytes(psutil)

    def _pick_download_option(options, title, local_keys=None, model_ref=None):
        if not options:
            return None

        normalized_local = set(str(item).strip().lower() for item in (local_keys or []))
        model_hint = lms_menu_helpers.model_hint_from_ref(model_ref)

        def _is_local_option(entry):
            return lms_menu_helpers.option_is_local(entry, normalized_local, model_hint)

        def _capacity_status(entry):
            size_bytes = entry.get("size_bytes")
            return lms_menu_helpers.capacity_status(size_bytes, gpu_mem_bytes, ram_mem_bytes)

        sorted_options = lms_menu_helpers.sort_download_options(options)
        option_items = []
        for entry in sorted_options:
            quant = entry.get("quantization") or "unknown"
            rec = " ★" if entry.get("recommended") else ""
            size_label = lms_menu_helpers.format_size(entry.get("size_bytes"))
            is_local_option = _is_local_option(entry)
            cap_status = _capacity_status(entry)

            if is_local_option:
                status_label = f"{Style.OKBLUE}LOCAL{Style.ENDC}"
            elif cap_status == "gpu":
                status_label = f"{Style.OKGREEN}INSTALL{Style.ENDC}"
            elif cap_status == "hybrid":
                status_label = f"{Style.WARNING}INSTALL+RAM{Style.ENDC}"
            elif cap_status == "no_fit":
                status_label = f"{Style.FAIL}NO-FIT{Style.ENDC}"
            else:
                status_label = f"{Style.WARNING}INSTALL?{Style.ENDC}"

            label = f"{quant:<14} | {size_label:<8}{rec} | {status_label}"
            is_blocked = is_local_option or cap_status == "no_fit"

            if is_blocked:
                reason = "Ya descargado" if is_local_option else "No cabe ni en GPU ni en RAM"
                option_items.append(MenuItem(label, action=lambda: None, description=f"{reason}: {str(entry.get('name') or '')}"))
            else:
                option_items.append(MenuItem(label, action=lambda e=entry: e, description=str(entry.get("name") or "")))

        option_items.append(MenuItem("Back", lambda: None, description="Volver al menú anterior."))

        def q_header():
            print_banner()
            print(f"{Style.BOLD} {title} {Style.ENDC}")

        selected = interactive_menu(
            option_items,
            header_func=q_header,
            menu_id="lms_quant_selector",
            nav_hint_text="↑/↓ navegar cuantizaciones · ENTER seleccionar · ESC volver",
        )
        if not selected or selected.label.strip() == "Back":
            return None
        if callable(getattr(selected, "action", None)):
            picked = selected.action()
            if picked is None:
                log("Esa opción no está disponible para descargar en este equipo.", "warning")
                wait_for_any_key()
                return None
            return picked
        return None

    def _download_with_progress(selected_option):
        started_at = time.time()
        last_render_at = 0.0
        header_drawn = False
        latest_downloaded = 0.0
        latest_total = 0.0
        latest_speed = 0.0
        panel_top_row = 0

        option_name = str(selected_option.get("name") or "").strip()
        option_quant = str(selected_option.get("quantization") or "unknown").strip().upper()
        option_identifier = str(selected_option.get("indexed_model_identifier") or "").strip()
        option_model_name = option_identifier.rsplit("/", 1)[0] if "/" in option_identifier else option_identifier
        if not option_model_name:
            option_model_name = option_name or "unknown-model"
        if not option_quant or option_quant == "UNKNOWN":
            option_quant = lms_menu_helpers.extract_quantization_from_text(option_name)
        if not option_quant or option_quant == "unknown":
            option_quant = lms_menu_helpers.extract_quantization_from_text(option_identifier)
        option_quant = str(option_quant or "UNKNOWN").upper()

        def _format_bytes(value):
            if not isinstance(value, (int, float)) or value < 0:
                return "0 B"
            units = ["B", "KB", "MB", "GB", "TB"]
            amount = float(value)
            idx = 0
            while amount >= 1024.0 and idx < len(units) - 1:
                amount /= 1024.0
                idx += 1
            return f"{amount:.2f} {units[idx]}"

        def _format_duration(seconds_value):
            if not isinstance(seconds_value, (int, float)) or seconds_value < 0:
                return "--:--:--"
            total_secs = int(seconds_value)
            hours = total_secs // 3600
            minutes = (total_secs % 3600) // 60
            seconds = total_secs % 60
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        def _render_progress(downloaded, total, speed):
            nonlocal header_drawn, panel_top_row
            progress_pct = (downloaded / total * 100.0) if total > 0 else 0.0
            remaining_bytes = max(total - downloaded, 0.0)
            eta_seconds = (remaining_bytes / speed) if speed > 0 else -1.0
            elapsed_seconds = max(time.time() - started_at, 0.0)

            bar_width = 42
            filled = int((progress_pct / 100.0) * bar_width) if total > 0 else 0
            filled = max(0, min(filled, bar_width))
            bar = f"{'█' * filled}{'░' * (bar_width - filled)}"

            speed_label = f"{speed / (1024 * 1024):.2f} MB/s" if speed > 0 else "-- MB/s"
            downloaded_label = _format_bytes(downloaded)
            total_label = _format_bytes(total)
            eta_label = _format_duration(eta_seconds) if eta_seconds >= 0 else "--:--:--"
            elapsed_label = _format_duration(elapsed_seconds)

            panel_inner_width = 70

            def _fit_text(value: str) -> str:
                text = str(value)
                if len(text) > panel_inner_width:
                    return text[: panel_inner_width - 3] + "..."
                return text.ljust(panel_inner_width)

            def _box_line(content: str) -> str:
                return f"│{_fit_text(content)}│"

            if not header_drawn:
                clear_screen_ansi()
                print_banner()
                print(f"{Style.BOLD} DOWNLOAD IN PROGRESS {Style.ENDC}")
                print()
                print(f" Modelo: {option_model_name}")
                print(f" Cuantización: {option_quant}")
                if option_name:
                    print(f" Artefacto: {option_name}")
                print()
                panel_top_row = 16
                header_drawn = True

            sys.stdout.write(f"\033[{panel_top_row};1H")
            print("┌" + "─" * panel_inner_width + "┐" + " " * 10)
            print(_box_line(f" Progreso: {progress_pct:6.2f}%") + " " * 10)
            print(_box_line(f" [{bar}]") + " " * 10)
            print(_box_line(f" Descargado: {downloaded_label:<14} / {total_label:<14}") + " " * 10)
            print(_box_line(f" Velocidad: {speed_label:<14} | ETA: {eta_label:<10} | Transcurrido: {elapsed_label:<8}") + " " * 10)
            print("└" + "─" * panel_inner_width + "┘" + " " * 10)
            print(f" {Style.DIM}Espera a que finalice la descarga...{Style.ENDC}" + " " * 20)
            sys.stdout.flush()

        def on_progress(update):
            nonlocal last_render_at, latest_downloaded, latest_total, latest_speed
            downloaded = float(getattr(update, "downloaded_bytes", 0) or 0)
            total = float(getattr(update, "total_bytes", 0) or 0)
            speed = float(getattr(update, "speed_bytes_per_second", 0) or 0)

            latest_downloaded = downloaded
            latest_total = total
            latest_speed = speed

            now = time.time()
            if (now - last_render_at) < 0.25 and downloaded < total:
                return

            _render_progress(downloaded, total, speed)
            last_render_at = now

        def on_finalize(_result=None):
            final_downloaded = latest_downloaded
            final_total = latest_total if latest_total > 0 else latest_downloaded
            _render_progress(final_downloaded, final_total, latest_speed)
            print(f"\n{Style.OKGREEN} ✔ Descarga completada.{Style.ENDC}")

        return lms_models.download_option(selected_option, on_progress=on_progress, on_finalize=on_finalize)

    def _download_registry_model_ui():
        registry_items = list(MODELS_REGISTRY.items())
        model_cursor = MENU_CURSOR_MEMORY.get("lms_registry_selector", 0)
        model_quant_cursor: dict[str, int] = {}
        options_cache: dict[str, list[dict[str, object]]] = {}

        gpu_gb = lms_menu_helpers.bytes_to_gb(gpu_mem_bytes)
        ram_gb = lms_menu_helpers.bytes_to_gb(ram_mem_bytes)

        def _render_registry_static() -> None:
            print_banner()
            print(f"{Style.BOLD} DOWNLOAD FROM MODELS_REGISTRY {Style.ENDC}")
            print(" " + "─" * 86)
            print(
                f" {Style.OKBLUE}LOCAL{Style.ENDC} | "
                f"{Style.OKGREEN}INSTALL (GPU){Style.ENDC} | "
                f"{Style.WARNING}INSTALL+RAM{Style.ENDC} | "
                f"{Style.FAIL}NO-FIT{Style.ENDC}"
            )
            print(f" {Style.DIM}Memoria detectada:{Style.ENDC} GPU {gpu_gb:.1f} GB | RAM {ram_gb:.1f} GB")
            print(f" {Style.DIM}↑/↓ modelo · ←/→ cuantización · ENTER seleccionar/descargar · ESC volver{Style.ENDC}")
            print(" " + "─" * 86)
            print()

        panel = incremental_panel_renderer_cls(clear_screen_fn=clear_screen_ansi, render_static_fn=_render_registry_static)

        def _get_cached_options(model_ref: str) -> list[dict[str, object]]:
            if model_ref in options_cache:
                return options_cache[model_ref]
            raw_options = lms_menu_helpers.sort_download_options(lms_models.get_download_options(model_ref))
            deduped = lms_menu_helpers.dedupe_options_by_quantization(raw_options)
            options_cache[model_ref] = deduped
            return deduped

        def _option_state(entry: dict[str, object], local_signatures: set[str], model_ref: str) -> tuple[str, str, bool]:
            state_text, color_token, blocked = lms_menu_helpers.option_visual_state(
                entry, local_signatures, model_ref, gpu_mem_bytes, ram_mem_bytes
            )
            color = getattr(Style, color_token, Style.WARNING)
            return state_text, color, blocked

        def _clamp_model_cursor(value: int) -> int:
            total_rows = len(registry_items) + 1
            if total_rows <= 0:
                return 0
            return value % total_rows

        while True:
            local_signatures = lms_menu_helpers.build_local_signatures(lms_models.list_local_llm_models())
            model_cursor = _clamp_model_cursor(model_cursor)
            MENU_CURSOR_MEMORY["lms_registry_selector"] = model_cursor

            selected_description = ""
            dynamic_lines: list[str] = []

            for row_idx, (model_key, data) in enumerate(registry_items):
                is_selected_model = row_idx == model_cursor
                pointer = ">" if is_selected_model else " "
                model_ref = str(data.get("name", "")).strip()
                options = _get_cached_options(model_ref)

                model_label = model_key if len(model_key) <= 42 else model_key[:39] + "..."
                dynamic_lines.append(f" {pointer} {Style.BOLD if is_selected_model else ''}{model_label}{Style.ENDC if is_selected_model else ''}")

                if not options:
                    chips_line = f"     {Style.DIM}(sin cuantizaciones detectadas){Style.ENDC}"
                    dynamic_lines.append(chips_line)
                    if is_selected_model:
                        selected_description = data.get("description", "Registry model")
                    continue

                current_q_idx = model_quant_cursor.get(model_key, 0)
                current_q_idx = max(0, min(current_q_idx, len(options) - 1))
                model_quant_cursor[model_key] = current_q_idx

                chip_parts: list[str] = []
                for q_idx, entry in enumerate(options):
                    quant = str(entry.get("quantization") or "unknown").upper()
                    _state_text, color, _blocked = _option_state(entry, local_signatures, model_ref)
                    chip = f"{color}{quant}{Style.ENDC}"
                    if is_selected_model and q_idx == current_q_idx:
                        chip = f"{Style.SELECTED} {chip} {Style.ENDC}"
                    chip_parts.append(chip)

                dynamic_lines.append("    " + "  ".join(chip_parts))

                if is_selected_model:
                    selected_entry = options[current_q_idx]
                    selected_quant = str(selected_entry.get("quantization") or "unknown").upper()
                    size_label = lms_menu_helpers.format_size(selected_entry.get("size_bytes"))
                    _state_text, color, _blocked = _option_state(selected_entry, local_signatures, model_ref)
                    selected_name = str(selected_entry.get("name") or "")
                    is_local_selected = lms_menu_helpers.option_is_local(selected_entry, local_signatures, model_ref)
                    short_state = "LOCAL" if is_local_selected else "INSTALL"
                    short_state_render = f"{color}{short_state}{Style.ENDC}"
                    selected_description = (
                        f"{selected_quant} | {size_label} | {short_state_render}"
                        + (f" | {selected_name}" if selected_name else "")
                    )

            back_row = len(registry_items)
            back_pointer = ">" if model_cursor == back_row else " "
            dynamic_lines.append(f" {back_pointer} Back")
            if model_cursor == back_row:
                selected_description = "Volver al menú principal de modelos."

            dynamic_lines.append("")
            dynamic_lines.append("─" * 86)
            if selected_description:
                dynamic_lines.append(f" {Style.DIM}Descripción: {selected_description}{Style.ENDC}")
            dynamic_lines.append("─" * 86)
            dynamic_lines.append(f" {Style.BOLD}[ENTER]: Seleccionar opción.{Style.ENDC}")

            panel.render(dynamic_lines)

            key = read_key()
            if key == "UP":
                model_cursor = _clamp_model_cursor(model_cursor - 1)
                continue
            if key == "DOWN":
                model_cursor = _clamp_model_cursor(model_cursor + 1)
                continue

            if key in ("LEFT", "RIGHT"):
                if model_cursor >= len(registry_items):
                    continue
                model_key, data = registry_items[model_cursor]
                model_ref = str(data.get("name", "")).strip()
                options = _get_cached_options(model_ref)
                if not options:
                    continue
                q_idx = model_quant_cursor.get(model_key, 0)
                q_idx = lms_menu_helpers.shift_horizontal_index(q_idx, key, len(options))
                model_quant_cursor[model_key] = q_idx
                continue

            if key == "ENTER":
                if model_cursor == len(registry_items):
                    return

                model_key, data = registry_items[model_cursor]
                model_ref = str(data.get("name", "")).strip()
                options = _get_cached_options(model_ref)
                if not options:
                    log("No se encontraron variantes/cuantiaciones para ese modelo en el repositorio.", "warning")
                    wait_for_any_key()
                    panel.reset()
                    continue

                q_idx = model_quant_cursor.get(model_key, 0)
                q_idx = max(0, min(q_idx, len(options) - 1))
                selected_option = options[q_idx]
                state_text, _color, blocked = _option_state(selected_option, local_signatures, model_ref)
                if blocked:
                    log(f"La cuantización seleccionada está bloqueada ({state_text}).", "warning")
                    wait_for_any_key()
                    panel.reset()
                    continue

                ok, info = _download_with_progress(selected_option)
                if ok:
                    log(f"Modelo descargado correctamente: {info}", "success")
                    if model_ref in options_cache:
                        del options_cache[model_ref]
                else:
                    log(f"No se pudo descargar el modelo: {info}", "error")
                wait_for_any_key()
                panel.reset()
                continue

            if key == "ESC":
                return

    def _download_custom_model_ui():
        clear_screen_ansi()
        print_banner()
        print(f"{Style.BOLD} ADD MODEL BY ID/URL {Style.ENDC}")
        print(" " + "─" * 86)
        manual_ref = input_with_esc(f"{Style.WARNING} Model id/url: {Style.ENDC}")
        if manual_ref is None:
            log("Operation cancelled by user (ESC).", "warning")
            wait_for_any_key()
            return

        manual_ref = str(manual_ref).strip()
        if not manual_ref:
            log("Empty model reference. Operation cancelled.", "warning")
            wait_for_any_key()
            return

        options = lms_models.get_download_options(manual_ref)
        local_keys = set(get_installed_lms_models())
        if not options:
            log("No se encontraron variantes/cuantiaciones para ese ID/URL.", "warning")
            wait_for_any_key()
            return

        selected_option = _pick_download_option(options, "SELECT QUANTIZATION", local_keys=local_keys, model_ref=manual_ref)
        if selected_option is None:
            return

        ok, info = _download_with_progress(selected_option)
        if ok:
            log(f"Modelo descargado correctamente: {info}", "success")
        else:
            log(f"No se pudo descargar el modelo: {info}", "error")
        wait_for_any_key()

    def _delete_model_ui(model_key):
        if ask_user(f"Confirm delete local model '{model_key}'?", "n"):
            ok, info = lms_models.remove_local_model(model_key)
            if ok:
                log(f"Modelo eliminado localmente: {info}", "success")
            else:
                log(f"No se pudo eliminar el modelo: {info}", "error")
        else:
            log("Delete cancelled by user.", "warning")
        wait_for_any_key()

    def _menu_header():
        print_banner()
        print(f"{Style.BOLD} LM STUDIO MODEL MANAGER {Style.ENDC}")
        print()

    while True:
        local_models = lms_models.list_local_llm_models()
        loaded_keys = lms_models.list_loaded_llm_model_keys()

        model_col_width = 38
        quant_col_width = 10
        status_col_width = 10
        table_width = model_col_width + quant_col_width + status_col_width + 6

        options = []

        table_header_item = MenuStaticItem(label="", description="")
        table_header_item.selectable = False

        def table_header_dynamic_label(_is_selected):
            return (
                f"{Style.DIM}{'MODEL':<{model_col_width}} | "
                f"{'QUANT':<{quant_col_width}} | {'STATUS':<{status_col_width}}{Style.ENDC}"
            )

        table_header_item.dynamic_label = table_header_dynamic_label
        options.append(table_header_item)

        table_separator_item = MenuSeparator(width=table_width)
        if hasattr(table_separator_item, "dynamic_label") and callable(table_separator_item.dynamic_label):
            base_table_separator_dynamic_label = table_separator_item.dynamic_label

            def colorized_table_separator(_is_selected):
                return f"{Style.DIM}{base_table_separator_dynamic_label(False)}{Style.ENDC}"

            table_separator_item.dynamic_label = colorized_table_separator
        table_separator_item.selectable = False
        options.append(table_separator_item)

        previous_base_name = None

        for model in local_models:
            model_key = str(model.get("model_key", "")).strip()
            if not model_key:
                continue

            quant_label = lms_menu_helpers.detect_local_model_quantization(model)
            base_name = model_key.rsplit("@", 1)[0].strip() if "@" in model_key else model_key
            shown_name = base_name if base_name != previous_base_name else "↳"
            previous_base_name = base_name

            menu_item = MenuItem(model_key, action=lambda mk=model_key: _delete_model_ui(mk), description="Pulsa ENTER para eliminar este modelo local.")

            def dynamic_label(
                is_selected_row,
                key=model_key,
                quant=quant_label,
                loaded_set=loaded_keys,
                name_cell=shown_name,
            ):
                status_plain = "REMOVE" if is_selected_row else ("LOADED" if key in loaded_set else "LOCAL")
                if is_selected_row:
                    status_color = Style.FAIL
                elif key in loaded_set:
                    status_color = Style.OKGREEN
                else:
                    status_color = Style.DIM

                model_display = name_cell if len(name_cell) <= model_col_width else name_cell[: model_col_width - 1] + "…"
                quant_plain = str(quant).upper()
                quant_cell = (
                    f"{quant_plain:<{quant_col_width}}"
                    if len(quant_plain) <= quant_col_width
                    else f"{quant_plain[: quant_col_width - 1]}…"
                )
                status_cell = f"{status_plain:<{status_col_width}}"

                if is_selected_row:
                    status_render = f"{Style.FAIL}{status_cell}{Style.ENDC}"
                    return f"{model_display:<{model_col_width}} | {quant_cell} | {status_render}"

                quant_color = Style.OKCYAN if quant != "unknown" else Style.DIM
                quant_render = f"{quant_color}{quant_cell}{Style.ENDC}"
                status_render = f"{status_color}{status_cell}{Style.ENDC}"
                return f"{model_display:<{model_col_width}} | {quant_render} | {status_render}"

            menu_item.dynamic_label = dynamic_label
            options.append(menu_item)

        separator_item = MenuSeparator(width=table_width)

        if not hasattr(separator_item, "dynamic_label") or not callable(separator_item.dynamic_label):
            separator_item = MenuStaticItem(label="", description="")
            separator_item.selectable = False

            def separator_dynamic_label(_is_selected):
                return f"{Style.DIM}{'─' * table_width}{Style.ENDC}"

            separator_item.dynamic_label = separator_dynamic_label
        else:
            base_dynamic_label = separator_item.dynamic_label

            def colorized_separator(_is_selected):
                return f"{Style.DIM}{base_dynamic_label(False)}{Style.ENDC}"

            separator_item.dynamic_label = colorized_separator

        options.append(separator_item)

        options.append(MenuItem("Download from MODELS_REGISTRY (quantizations)...", _download_registry_model_ui, description="Descarga variantes por cuantización de modelos del registro."))
        options.append(MenuItem("Download by model ID/URL (quantizations)...", _download_custom_model_ui, description="Descarga variantes por cuantización usando id/url manual."))
        options.append(MenuItem("Back", lambda: "BACK", description="Vuelve al menú de Tests & Models."))

        choice = interactive_menu(
            options,
            header_func=_menu_header,
            menu_id="manage_models_menu",
            nav_hint=True,
            nav_hint_text="↑/↓ navegar · ENTER seleccionar · ESC volver",
            left_margin=0,
        )
        if not choice:
            break

        clear_screen_ansi()
        if choice is separator_item:
            continue
        if not callable(getattr(choice, "action", None)):
            continue
        result = choice.action()
        if result == "BACK":
            break
