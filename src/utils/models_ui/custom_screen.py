"""Pantalla de descarga manual por ID/URL."""

from __future__ import annotations

from typing import Any

from ..setup_ui_io import ask_text
from .shared import download_with_progress, pick_download_option


def download_custom_model_ui(ctx: dict[str, Any]) -> None:
    """Renderiza flujo de descarga manual y dispara descarga en background."""
    style = ctx["Style"]

    def _render_static_top(ui_width: int) -> None:
        """Renderiza el encabezado superior común del Model Manager."""
        ctx["print_banner"]()
        print(f"{style.BOLD} LM STUDIO MODEL MANAGER {style.ENDC}")
        print()
        ctx["kit"].banner(title="DOWNLOAD BY MODEL ID/URL (QUANTIZATIONS)", width=ui_width)

    def _render_footer(current_value: str, ui_width: int) -> list[str]:
        """Construye líneas dinámicas del estado de descargas para el panel."""
        _ = current_value
        snapshot = ctx["build_download_status_snapshot"](ui_width=ui_width)
        summary = str(snapshot.get("summary") or "")
        running_lines = list(snapshot.get("running_lines") or [])

        lines: list[str] = []
        if summary:
            for summary_line in ctx["kit"].wrap(summary, max(16, ui_width - 2)):
                lines.append(f" {style.DIM}{summary_line}{style.ENDC}")
            for running_line in running_lines:
                lines.append(f" {running_line}")
            lines.append(ctx["divider"](ui_width))
        return lines

    manual_ref = ask_text(
        kit=ctx["kit"],
        title="DOWNLOAD BY MODEL ID/URL (QUANTIZATIONS)",
        intro_lines=[
            f"{style.DIM}Pega una URL de Hugging Face o un ID exacto. ESC para cancelar.{style.ENDC}",
            f"{style.DIM}Ejemplos:{style.ENDC}",
            f"  {style.OKCYAN}https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF{style.ENDC}",
            f"  {style.OKCYAN}Qwen/Qwen3-VL-8B-Instruct-GGUF{style.ENDC}",
        ],
        prompt_label="Model id/url:",
        help_line="[ENTER] confirmar · [Backspace] borrar · [ESC] cancelar",
        normalize_on_submit_fn=lambda value: value.strip(),
        render_extra_lines_fn=_render_footer,
        render_static_top_fn=_render_static_top,
        force_full_on_update=True,
    )
    if manual_ref is None:
        ctx["log"]("Operation cancelled by user (ESC).", "warning")
        ctx["wait_for_any_key"]()
        return

    manual_ref = str(manual_ref)
    if not manual_ref:
        ctx["log"]("Empty model reference. Operation cancelled.", "warning")
        ctx["wait_for_any_key"]()
        return

    options = ctx["lms_models"].get_download_options(manual_ref)
    local_keys = set(ctx["get_installed_lms_models"]())
    if not options:
        ctx["log"]("No se encontraron variantes/cuantiaciones para ese ID/URL.", "warning")
        ctx["wait_for_any_key"]()
        return

    selected_option = pick_download_option(
        ctx,
        options,
        "SELECT QUANTIZATION",
        local_keys=local_keys,
        model_ref=manual_ref,
    )
    if selected_option is None:
        return

    ok, info = download_with_progress(ctx, selected_option)
    if ok:
        ctx["log"](str(info), "success")
    else:
        ctx["log"](f"No se pudo descargar el modelo: {info}", "error")
        ctx["wait_for_any_key"]()
