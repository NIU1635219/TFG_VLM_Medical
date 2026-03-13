from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from ..setup_ui_io import ask_text

if TYPE_CHECKING:
    from ..menu_kit import AppContext, UIKit


def make_header(kit: "UIKit", app: "AppContext", subtitle: str) -> Callable[[], None]:
    """Create a reusable header renderer for tests menus.

    Args:
        kit: Terminal UI toolkit.
        app: Application context with banner renderer.
        subtitle: Subtitle to show below the banner.

    Returns:
        A callable that renders banner and subtitle when invoked.
    """

    def _hdr() -> None:
        """Renderiza banner principal y subtítulo del menú activo."""
        app.print_banner()
        kit.subtitle(subtitle)

    return _hdr


def select_schema_reasoning_mode(
    kit: "UIKit",
    *,
    menu_id: str,
    subtitle: str,
    make_header_fn: Callable[[str], Callable[[], None]],
) -> bool | str:
    """Select whether schema execution should include reasoning.

    Args:
        kit: Terminal UI toolkit.
        menu_id: Stable menu id for cursor memory.
        subtitle: Menu subtitle.
        make_header_fn: Header factory callable.

    Returns:
        `True` when reasoning is enabled, `False` when disabled, or
        `"BACK"` when user cancels to previous schema menu.
    """
    items = [
        kit.MenuItem(
            "Con razonamiento",
            description="Añade el campo reasoning al JSON y obliga a justificar antes del veredicto.",
        ),
        kit.MenuItem(
            "Sin razonamiento",
            description="Usa el schema base sin campo reasoning para una salida estructurada más compacta.",
        ),
        kit.MenuItem(
            "← Volver al selector de schema",
            lambda: None,
            description="Cancela y vuelve a elegir el schema base.",
        ),
    ]
    sel = kit.menu(
        items,
        header_func=make_header_fn(subtitle),
        menu_id=menu_id,
        nav_hint_text="↑/↓ elegir modo · ENTER confirmar · ESC volver al selector de schema",
    )
    if not sel or "Volver" in sel.label:
        return "BACK"
    return "Con razonamiento" in sel.label


def select_schema_variant(
    kit: "UIKit",
    *,
    menu_prefix: str,
    subtitle_prefix: str,
    make_header_fn: Callable[[str], Callable[[], None]],
) -> tuple[str, Any] | None:
    """Resolve base schema plus reasoning variant via interactive menus.

    Args:
        kit: Terminal UI toolkit.
        menu_prefix: Prefix used to build deterministic menu ids.
        subtitle_prefix: Prefix shown in menu subtitle.
        make_header_fn: Header factory callable.

    Returns:
        Tuple `(schema_name, schema_class)` when selected, or `None` if cancelled.
    """
    from src.inference.schemas import SCHEMA_REGISTRY, get_schema_variant
    from src.scripts.test_schema import format_schema_menu_description

    while True:
        schema_items = [
            kit.MenuItem(name, description=format_schema_menu_description(name, cls))
            for name, cls in SCHEMA_REGISTRY.items()
        ]
        schema_items.append(
            kit.MenuItem(
                "Volver al selector de modelos",
                lambda: None,
                description="Vuelve a la selección de modelo.",
            )
        )

        schema_sel = kit.menu(
            schema_items,
            header_func=make_header_fn(f"{subtitle_prefix} · SELECT SCHEMA"),
            menu_id=f"{menu_prefix}_schema_selector",
            nav_hint_text="↑/↓ elegir esquema · ENTER confirmar · ESC volver",
            description_slot_rows=15,
        )
        if not schema_sel or schema_sel.label.strip() == "Volver al selector de modelos":
            return None

        base_schema_name = schema_sel.label
        reasoning_mode = select_schema_reasoning_mode(
            kit,
            menu_id=f"{menu_prefix}_reasoning_selector",
            subtitle=f"{subtitle_prefix} · SELECT SCHEMA MODE",
            make_header_fn=make_header_fn,
        )
        if reasoning_mode == "BACK":
            continue

        return get_schema_variant(base_schema_name, bool(reasoning_mode))


def select_response_inspector_mode(
    kit: "UIKit",
    *,
    make_header_fn: Callable[[str], Callable[[], None]],
) -> tuple[bool, bool] | None:
    """Select response inspector mode (raw vs structured).

    Args:
        kit: Terminal UI toolkit.
        make_header_fn: Header factory callable.

    Returns:
        Tuple `(structured, with_reasoning)` when selected, otherwise `None`.
    """
    items = [
        kit.MenuItem(
            "Cruda (sin schema)",
            description="Envía una petición multimodal sin response_format para ver la respuesta nativa del SDK.",
        ),
        kit.MenuItem(
            "Estructurada con reasoning",
            description="Usa GenericObjectDetectionWithReasoning para inspeccionar parsed, stats y schema activo.",
        ),
        kit.MenuItem("Cancel", lambda: None, description="Vuelve al selector de modelo."),
    ]
    sel = kit.menu(
        items,
        header_func=make_header_fn("RESPONSE INSPECTOR · SELECT MODE"),
        menu_id="response_inspector_mode_selector",
        nav_hint_text="↑/↓ elegir modo · ENTER confirmar · ESC volver",
    )
    if not sel or sel.label.strip() == "Cancel":
        return None
    if sel.label.startswith("Cruda"):
        return False, False
    return True, True


def select_model(
    kit: "UIKit",
    app: "AppContext",
    *,
    menu_id: str,
    subtitle: str,
    make_header_fn: Callable[[str], Callable[[], None]],
) -> str | None:
    """Prompt user to choose an installed LM Studio model.

    Args:
        kit: Terminal UI toolkit.
        app: Application context used to discover installed models.
        menu_id: Stable menu id for cursor memory.
        subtitle: Menu subtitle.
        make_header_fn: Header factory callable.

    Returns:
        Selected model tag or `None` if user cancels.
    """
    installed = app.get_installed_lms_models()
    if not installed:
        kit.log(
            "No hay modelos disponibles en LM Studio. Carga uno desde 'Manage/Pull LM Studio Models...'.",
            "warning",
        )
        kit.wait("Press any key to return to tests menu...")
        return None

    opts = [
        kit.MenuItem(tag, description="Usa este modelo para la inferencia.")
        for tag in installed
    ]
    opts.append(kit.MenuItem("Cancel", lambda: None, description="Vuelve al menú anterior."))

    sel = kit.menu(
        opts,
        header_func=make_header_fn(subtitle),
        menu_id=menu_id,
        nav_hint_text="↑/↓ elegir modelo · ENTER confirmar · ESC volver",
    )
    if not sel or sel.label.strip() == "Cancel":
        return None

    tag = sel.label.strip()
    if not tag:
        kit.log("No model selected.", "warning")
        kit.wait("Press any key to return to tests menu...")
        return None
    return tag
