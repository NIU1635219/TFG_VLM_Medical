"""Pantalla de descarga desde MODELS_REGISTRY."""

from __future__ import annotations

import time
from typing import Any

from ..setup_ui_io import _wrap_colored_chunks
from .shared import download_with_progress


def download_registry_model_ui(ctx: dict[str, Any]) -> None:
    """Renderiza la pantalla interactiva de registry y gestiona descargas."""
    style = ctx["Style"]
    lms_models = ctx["lms_models"]
    lms_menu_helpers = ctx["lms_menu_helpers"]

    registry_items = list(ctx["MODELS_REGISTRY"].items())
    model_cursor = ctx["MENU_CURSOR_MEMORY"].get("lms_registry_selector", 0)
    model_quant_cursor: dict[str, int] = {}
    options_cache: dict[str, list[dict[str, object]]] = {}
    local_signatures_cache = lms_menu_helpers.build_local_signatures(lms_models.list_local_llm_models())
    seen_completed_jobs: set[str] = set()

    gpu_gb = lms_menu_helpers.bytes_to_gb(ctx["gpu_mem_bytes"])
    ram_gb = lms_menu_helpers.bytes_to_gb(ctx["ram_mem_bytes"])

    def render_registry_static() -> None:
        """Pinta la cabecera estática del panel de registry."""
        ui_width = ctx["current_ui_width"]()
        ctx["print_banner"]()
        print(f"{style.BOLD} LM STUDIO MODEL MANAGER {style.ENDC}")
        print()
        ctx["kit"].banner(title="DOWNLOAD FROM MODELS_REGISTRY", width=ui_width)
        print(" " + ctx["divider"](ui_width))
        print(
            f" {style.OKBLUE}LOCAL{style.ENDC} | "
            f"{style.OKGREEN}INSTALL (GPU){style.ENDC} | "
            f"{style.WARNING}INSTALL+RAM{style.ENDC} | "
            f"{style.FAIL}NO-FIT{style.ENDC}"
        )
        print(f" {style.DIM}Memoria detectada:{style.ENDC} GPU {gpu_gb:.1f} GB | RAM {ram_gb:.1f} GB")
        print(f" {style.DIM}↑/↓ modelo · ←/→ cuantización · ENTER seleccionar/descargar · ESC volver{style.ENDC}")
        print(" " + ctx["divider"](ui_width))
        print()

    panel = ctx["incremental_panel_renderer_cls"](
        clear_screen_fn=ctx["clear_screen_ansi"],
        render_static_fn=render_registry_static,
    )

    def get_cached_options(model_ref: str) -> list[dict[str, object]]:
        """Devuelve opciones deduplicadas por cuantización usando caché.

        Args:
            model_ref: Referencia del modelo en el registry.

        Returns:
            Lista de opciones de descarga deduplicadas.
        """
        if model_ref in options_cache:
            return options_cache[model_ref]
        raw_options = lms_menu_helpers.sort_download_options(lms_models.get_download_options(model_ref))
        deduped = lms_menu_helpers.dedupe_options_by_quantization(raw_options)
        options_cache[model_ref] = deduped
        return deduped

    def option_state(
        entry: dict[str, object],
        local_signatures: set[str],
        model_ref: str,
        running_signatures: set[str],
    ) -> tuple[str, str, bool]:
        """Calcula estado visual y bloqueo de una cuantización.

        Args:
            entry: Opción de descarga evaluada.
            local_signatures: Firmas de modelos ya instalados.
            model_ref: Referencia base del modelo.
            running_signatures: Firmas de descargas en curso.

        Returns:
            Tupla con texto de estado, color ANSI y flag de bloqueo.
        """
        option_signature = ctx["download_signature_for_option"](entry, model_ref).lower()
        if option_signature in running_signatures:
            return "DOWNLOADING", style.OKCYAN, True
        state_text, color_token, blocked = lms_menu_helpers.option_visual_state(
            entry, local_signatures, model_ref, ctx["gpu_mem_bytes"], ctx["ram_mem_bytes"]
        )
        color = getattr(style, color_token, style.WARNING)
        return state_text, color, blocked

    def clamp_model_cursor(value: int) -> int:
        """Ajusta el cursor principal para navegación circular.

        Args:
            value: Índice solicitado para el cursor.

        Returns:
            Índice normalizado dentro del rango visible.
        """
        total_rows = len(registry_items) + 1
        if total_rows <= 0:
            return 0
        return value % total_rows

    needs_render = True
    last_footer_download = ""
    prev_ui_width: int | None = None
    while True:
        ui_width = ctx["current_ui_width"]()
        if prev_ui_width is None:
            prev_ui_width = ui_width
        elif prev_ui_width != ui_width:
            needs_render = True
            prev_ui_width = ui_width

        footer_snapshot = ctx["build_download_status_snapshot"](ui_width=ui_width)
        completed_jobs = set(str(job_id) for job_id in footer_snapshot.get("completed_ids", set()))
        if completed_jobs != seen_completed_jobs:
            if completed_jobs - seen_completed_jobs:
                local_signatures_cache = lms_menu_helpers.build_local_signatures(lms_models.list_local_llm_models())
                options_cache.clear()
                needs_render = True
            seen_completed_jobs = completed_jobs

        local_signatures = local_signatures_cache
        running_signatures = ctx["running_download_signatures"](snapshot=footer_snapshot)
        model_cursor = clamp_model_cursor(model_cursor)
        ctx["MENU_CURSOR_MEMORY"]["lms_registry_selector"] = model_cursor

        footer_download = str(footer_snapshot.get("summary") or "")
        footer_running_lines = list(footer_snapshot.get("running_lines") or [])
        footer_signature = str(footer_snapshot.get("signature") or "")
        if not needs_render and footer_signature == last_footer_download:
            key = ctx["read_key"]()
            if key is None:
                time.sleep(0.05)
                continue
        else:
            key = None

        if needs_render or footer_signature != last_footer_download:
            selected_description = ""
            dynamic_lines: list[str] = []

            if footer_download:
                dynamic_lines.append(f" {style.DIM}{footer_download}{style.ENDC}")
                for running_line in footer_running_lines:
                    dynamic_lines.append(f" {running_line}")
                dynamic_lines.append(ctx["divider"](ui_width))
                dynamic_lines.append("")

            for row_idx, (model_key, data) in enumerate(registry_items):
                is_selected_model = row_idx == model_cursor
                pointer = ">" if is_selected_model else " "
                model_ref = str(data.get("name", "")).strip()
                options = get_cached_options(model_ref)

                model_label = model_key if len(model_key) <= 42 else model_key[:39] + "..."
                dynamic_lines.append(f" {pointer} {style.BOLD if is_selected_model else ''}{model_label}{style.ENDC if is_selected_model else ''}")

                if not options:
                    chips_line = f"     {style.DIM}(sin cuantizaciones detectadas){style.ENDC}"
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
                    _state_text, color, _blocked = option_state(entry, local_signatures, model_ref, running_signatures)
                    chip = f"{color}{quant}{style.ENDC}"
                    if is_selected_model and q_idx == current_q_idx:
                        chip = f"{style.SELECTED} {chip} {style.ENDC}"
                    chip_parts.append(chip)

                for chips_line in _wrap_colored_chunks(
                    chip_parts,
                    prefix="    ",
                    max_width=max(16, ui_width - 2),
                ):
                    dynamic_lines.append(chips_line)

                if is_selected_model:
                    selected_entry = options[current_q_idx]
                    selected_quant = str(selected_entry.get("quantization") or "unknown").upper()
                    size_label = lms_menu_helpers.format_size(selected_entry.get("size_bytes"))
                    state_text, color, _blocked = option_state(
                        selected_entry,
                        local_signatures,
                        model_ref,
                        running_signatures,
                    )
                    selected_name = str(selected_entry.get("name") or "")
                    is_local_selected = lms_menu_helpers.option_is_local(selected_entry, local_signatures, model_ref)
                    short_state = "DOWNLOADING" if state_text == "DOWNLOADING" else ("LOCAL" if is_local_selected else "INSTALL")
                    short_state_render = f"{color}{short_state}{style.ENDC}"
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
            dynamic_lines.append(ctx["divider"](ui_width))
            if selected_description:
                dynamic_lines.append(f" {style.DIM}Descripción: {selected_description}{style.ENDC}")
            dynamic_lines.append(ctx["divider"](ui_width))
            dynamic_lines.append(f" {style.BOLD}[ENTER]: Seleccionar opción.{style.ENDC}")

            panel.render(dynamic_lines)
            last_footer_download = footer_signature
            needs_render = False

        if key is None:
            key = ctx["read_key"]()
        if key is None:
            time.sleep(0.05)
            continue

        if key == "UP":
            model_cursor = clamp_model_cursor(model_cursor - 1)
            needs_render = True
            continue
        if key == "DOWN":
            model_cursor = clamp_model_cursor(model_cursor + 1)
            needs_render = True
            continue

        if key in ("LEFT", "RIGHT"):
            if model_cursor >= len(registry_items):
                continue
            model_key, data = registry_items[model_cursor]
            model_ref = str(data.get("name", "")).strip()
            options = get_cached_options(model_ref)
            if not options:
                continue
            q_idx = model_quant_cursor.get(model_key, 0)
            q_idx = lms_menu_helpers.shift_horizontal_index(q_idx, key, len(options))
            model_quant_cursor[model_key] = q_idx
            needs_render = True
            continue

        if key == "ENTER":
            if model_cursor == len(registry_items):
                return

            model_key, data = registry_items[model_cursor]
            model_ref = str(data.get("name", "")).strip()
            options = get_cached_options(model_ref)
            if not options:
                ctx["log"]("No se encontraron variantes/cuantiaciones para ese modelo en el repositorio.", "warning")
                ctx["wait_for_any_key"]()
                panel.reset()
                needs_render = True
                continue

            q_idx = model_quant_cursor.get(model_key, 0)
            q_idx = max(0, min(q_idx, len(options) - 1))
            selected_option = options[q_idx]
            state_text, _color, blocked = option_state(selected_option, local_signatures, model_ref, running_signatures)
            if blocked:
                ctx["log"](f"La cuantización seleccionada está bloqueada ({state_text}).", "warning")
                ctx["wait_for_any_key"]()
                panel.reset()
                needs_render = True
                continue

            ok, info = download_with_progress(ctx, selected_option)
            if ok:
                ctx["log"](str(info), "success")
                if model_ref in options_cache:
                    del options_cache[model_ref]
            else:
                ctx["log"](f"No se pudo descargar el modelo: {info}", "error")
                ctx["wait_for_any_key"]()
            panel.reset()
            needs_render = True
            continue

        if key == "ESC":
            return

        time.sleep(0.01)
