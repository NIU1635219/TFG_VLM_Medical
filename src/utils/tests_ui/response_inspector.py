from __future__ import annotations

from argparse import Namespace
from typing import TYPE_CHECKING, Callable

from .test_dashboards_ui import (
    _build_named_value_lines,
    _render_final_sections_screen,
    _standard_final_intro,
)

if TYPE_CHECKING:
    from ..menu_kit import AppContext, UIKit


def run_response_inspector_wrapper(
    kit: "UIKit",
    app: "AppContext",
    *,
    select_model: Callable[[str, str], str | None],
    select_response_inspector_mode: Callable[[], tuple[bool, bool] | None],
) -> None:
    """Run the SDK response inspector using interactive defaults.

    Args:
        kit: Terminal UI toolkit.
        app: Application context used to render shared UI chrome.
        select_model: Callback that resolves the target model tag.
        select_response_inspector_mode: Callback that resolves raw/structured mode.
    """
    from src.scripts.test_response import build_summary_sections, run_inspection, save_inspection_payload

    while True:
        model_tag = select_model(
            "response_inspector_model_selector",
            "RESPONSE INSPECTOR · SELECT MODEL",
        )
        if model_tag is None:
            return

        mode = select_response_inspector_mode()
        if mode is None:
            continue
        structured, with_reasoning = mode

        kit.clear()
        app.print_banner()
        kit.subtitle("RESPONSE INSPECTOR")
        kit.log(
            f"Inspeccionando respuesta real del SDK con el modelo {model_tag} y defaults automáticos...",
            "step",
        )
        try:
            payload = run_inspection(
                Namespace(
                    model=model_tag,
                    image=None,
                    prompt=None,
                    schema=None,
                    structured=structured,
                    with_reasoning=with_reasoning,
                    temperature=0.0,
                    server_api_host=None,
                    api_token=None,
                    output=None,
                    print_json=False,
                )
            )
            output_path = save_inspection_payload(payload)
            sections = []
            for title, rows in build_summary_sections(payload):
                sections.append((title, _build_named_value_lines(kit, rows)))
            sections.append(("Salida", [f"  Archivo guardado en: {output_path}"]))
            _render_final_sections_screen(
                kit,
                app,
                subtitle="RESPONSE INSPECTOR",
                intro=_standard_final_intro(),
                sections=sections,
            )
            kit.log(f"Response Inspector completado. Salida: {output_path}", "success")
        except Exception as error:
            kit.log(f"Response Inspector terminó con error: {error}", "error")
        kit.wait("Press any key to return to model selector...")
