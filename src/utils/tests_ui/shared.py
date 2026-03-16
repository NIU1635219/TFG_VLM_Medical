from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Callable

from ..setup_ui_io import ask_text

if TYPE_CHECKING:
    from ..menu_kit import AppContext, UIKit


def normalize_state_value(value: object) -> str:
    """Normaliza el estado textual para comparaciones internas."""
    return str(value or "").strip().lower()


def as_yes(value: object) -> bool:
    """Interpreta respuestas interactivas como afirmativas/negativas explícitas."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"y", "yes", "s", "si", "true", "1"}


def normalize_model_variants(raw_variants: Any) -> list[dict[str, Any]]:
    """Normaliza y deduplica variantes `(model_id, include_reasoning)`.

    Args:
        raw_variants: Lista cruda de variantes proveniente de UI o manifiesto.

    Returns:
        Lista estable de variantes válidas.
    """
    if not isinstance(raw_variants, list):
        return []

    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, bool]] = set()
    for raw_variant in raw_variants:
        if not isinstance(raw_variant, dict):
            continue
        model_id = str(raw_variant.get("model_id") or "").strip()
        if not model_id:
            continue
        include_reasoning = bool(raw_variant.get("include_reasoning"))
        key = (model_id, include_reasoning)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            {
                "model_id": model_id,
                "include_reasoning": include_reasoning,
            }
        )
    return normalized


def variant_label(model_id: str, include_reasoning: bool) -> str:
    """Etiqueta de variante con modo de razonamiento."""
    mode = "con razonamiento" if include_reasoning else "sin razonamiento"
    return f"{model_id} [{mode}]"


def safe_positive_int(value: Any, default: int = 1) -> int:
    """Parsea enteros positivos con fallback seguro."""
    try:
        return max(1, int(value or default))
    except (TypeError, ValueError):
        return default


def parse_required_positive_int(value: Any) -> int | None:
    """Parsea entero positivo obligatorio; devuelve None si falta o es inválido."""
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def build_live_status_line(
    *,
    event: str,
    current_index: int,
    total: int,
    item_label: str,
    status: str,
    completed: int,
    on_start: str,
    on_done_ok: str | None = None,
    on_done_invalid: str | None = None,
    on_done_default: str | None = None,
    on_complete: str,
    on_prepare: str,
    start_event: str = "image_start",
    done_event: str = "image_done",
    complete_event: str = "complete",
) -> str:
    """Construye mensajes de estado en vivo para paneles de progreso.

    Las plantillas aceptan placeholders: {current_index}, {total}, {item}, {status}, {completed}.
    """
    context = {
        "current_index": current_index,
        "total": total,
        "item": item_label,
        "status": str(status or ""),
        "completed": completed,
    }

    if event == start_event:
        return on_start.format(**context)

    if event == done_event:
        status_value = str(status or "").lower()
        if status_value == "ok" and on_done_ok:
            return on_done_ok.format(**context)
        if status_value == "invalid" and on_done_invalid:
            return on_done_invalid.format(**context)
        if on_done_default:
            return on_done_default.format(**context)

    if event == complete_event:
        return on_complete.format(**context)

    return on_prepare.format(**context)


def compact_model_label(model_tag: str, *, ui_width: int) -> str:
    """Compacta el model tag para pantallas estrechas preservando inicio/fin."""
    text = str(model_tag or "").strip()
    if not text:
        return "N/D"
    max_len = max(14, min(30, int(ui_width * 0.33)))
    if len(text) <= max_len:
        return text
    head = max(6, (max_len - 1) // 2)
    tail = max(5, max_len - head - 1)
    return f"{text[:head]}…{text[-tail:]}"


def is_model_snapshot_complete(snapshot: dict[str, Any]) -> bool:
    """Valida de forma estricta si un modelo está realmente completado."""
    status = normalize_state_value(snapshot.get("status"))
    total = int(snapshot.get("total", 0) or 0)
    ok = int(snapshot.get("ok", 0) or 0)
    errors = int(snapshot.get("errors", 0) or 0)
    pending = int(snapshot.get("pending", 0) or 0)
    return status == "green" and total > 0 and ok == total and errors == 0 and pending == 0


def rel_path(path_value: str) -> str:
    """Convierte rutas absolutas a relativas para mejorar legibilidad en terminal."""
    path_text = str(path_value or "").strip()
    if not path_text:
        return "N/D"
    try:
        return os.path.relpath(path_text, ".")
    except ValueError:
        return path_text


def truncate_middle(text: str, limit: int) -> str:
    """Recorta texto largo preservando inicio/fin para columnas estrechas."""
    value = str(text or "")
    if len(value) <= limit:
        return value
    if limit <= 6:
        return value[:limit]
    head = max(3, (limit - 1) // 2)
    tail = max(2, limit - head - 1)
    return f"{value[:head]}…{value[-tail:]}"


def snapshot_status_text(snapshot: dict[str, Any]) -> str:
    """Etiqueta de estado sin ANSI para tabla resumen final."""
    state = normalize_state_value(snapshot.get("status"))
    if state == "green":
        return "COMPLETO"
    if state == "yellow":
        return "PARCIAL"
    return "ERROR"


def snapshot_status_color_code(style: Any, snapshot: dict[str, Any]) -> str:
    """Obtiene color ANSI asociado al estado del snapshot."""
    state = normalize_state_value(snapshot.get("status"))
    if state == "green":
        return str(getattr(style, "OKGREEN", ""))
    if state == "yellow":
        return str(getattr(style, "WARNING", ""))
    return str(getattr(style, "FAIL", ""))


def colorize_model(style: Any, model_tag: str, snapshot: dict[str, Any]) -> str:
    """Pinta el nombre del modelo según estado (verde/amarillo/rojo)."""
    color = snapshot_status_color_code(style, snapshot)
    endc = str(getattr(style, "ENDC", "")) if color else ""
    return f"{color}{model_tag}{endc}" if color else model_tag


def snapshot_summary_line(style: Any, model_tag: str, snapshot: dict[str, Any]) -> str:
    """Resumen compacto del estado actual para prompts interactivos."""
    ok = int(snapshot.get("ok", 0) or 0)
    total = int(snapshot.get("total", 0) or 0)
    pending = int(snapshot.get("pending", 0) or 0)
    errors = int(snapshot.get("errors", 0) or 0)
    status_plain = snapshot_status_text(snapshot)
    return (
        f"{colorize_model(style, model_tag, snapshot)} · {status_plain} · "
        f"OK/TOT={ok}/{total} · Pend={pending} · Err={errors}"
    )


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


def select_schema_base(
    kit: "UIKit",
    *,
    menu_prefix: str,
    subtitle_prefix: str,
    make_header_fn: Callable[[str], Callable[[], None]],
    initial_schema_name: str | None = None,
) -> str | None:
    """Resolve only the base schema without asking reasoning mode.

    Args:
        kit: Terminal UI toolkit.
        menu_prefix: Prefix used to build deterministic menu ids.
        subtitle_prefix: Prefix shown in menu subtitle.
        make_header_fn: Header factory callable.

    Returns:
        Selected base schema name, or `None` if cancelled.
    """
    from src.inference.schemas import SCHEMA_REGISTRY
    from src.scripts.test_schema import format_schema_menu_description

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

    selector_menu_id = f"{menu_prefix}_schema_selector"
    if initial_schema_name:
        try:
            initial_index = next(
                idx
                for idx, item in enumerate(schema_items)
                if str(getattr(item, "label", "")).strip() == str(initial_schema_name).strip()
            )
            kit.cursor_memory[selector_menu_id] = initial_index
        except StopIteration:
            pass

    schema_sel = kit.menu(
        schema_items,
        header_func=make_header_fn(f"{subtitle_prefix} · SELECT SCHEMA"),
        menu_id=selector_menu_id,
        nav_hint_text="↑/↓ elegir esquema · ENTER confirmar · ESC volver",
        description_slot_rows=15,
    )
    if not schema_sel or schema_sel.label.strip() == "Volver al selector de modelos":
        return None
    return str(schema_sel.label).strip() or None


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


def select_model_variants(
    kit: "UIKit",
    app: "AppContext",
    *,
    menu_id: str,
    subtitle: str,
    make_header_fn: Callable[[str], Callable[[], None]],
    initial_model_variants: list[dict[str, Any]] | None = None,
) -> tuple[list[str], list[dict[str, Any]]] | None:
    """Selecciona variantes por modelo (sin/con razonamiento).

    Returns:
        Tupla `(models, model_variants)` o None si se cancela.
    """
    installed = app.get_installed_lms_models()
    if not installed:
        kit.log(
            "No hay modelos disponibles en LM Studio. Carga uno desde 'Manage/Pull LM Studio Models...'.",
            "warning",
        )
        return None

    model_items = []
    for tag in installed:
        child_without = kit.MenuItem(
            "Sin razonamiento",
            description="Ejecuta el schema base para este modelo.",
        )
        setattr(child_without, "_is_reasoning_variant", True)
        setattr(child_without, "_model_id", tag)
        setattr(child_without, "_include_reasoning", False)

        child_with = kit.MenuItem(
            "Con razonamiento",
            description="Ejecuta la variante WithReasoning para este modelo.",
        )
        setattr(child_with, "_is_reasoning_variant", True)
        setattr(child_with, "_model_id", tag)
        setattr(child_with, "_include_reasoning", True)

        parent = kit.MenuItem(
            tag,
            description="Selecciona uno o ambos modos de razonamiento para este modelo.",
            children=[child_without, child_with],
        )
        setattr(parent, "_model_parent", tag)
        model_items.append(parent)

    initial_keys: set[tuple[str, bool]] = set()
    for variant in normalize_model_variants(initial_model_variants or []):
        model_id = str(variant.get("model_id") or "").strip()
        if not model_id:
            continue
        initial_keys.add((model_id, bool(variant.get("include_reasoning"))))

    if initial_keys:
        for parent in model_items:
            for child in getattr(parent, "children", []) or []:
                model_id = str(getattr(child, "_model_id", "") or "").strip()
                include_reasoning = bool(getattr(child, "_include_reasoning", False))
                child.is_selected = (model_id, include_reasoning) in initial_keys

    selected = kit.menu(
        model_items,
        header_func=make_header_fn(subtitle),
        menu_id=menu_id,
        multi_select=True,
        nav_hint_text="↑/↓ navegar · SPACE marcar · ENTER confirmar · ESC cancelar",
    )
    if not selected:
        return None
    if not isinstance(selected, list):
        selected = [selected]

    model_variants: list[dict[str, Any]] = []
    selected_parent_models: set[str] = set()
    seen_keys: set[tuple[str, bool]] = set()
    for item in selected:
        is_variant = bool(getattr(item, "_is_reasoning_variant", False))
        if is_variant:
            model_id = str(getattr(item, "_model_id", "") or "").strip()
            include_reasoning = bool(getattr(item, "_include_reasoning", False))
            if not model_id:
                continue
            variant_key = (model_id, include_reasoning)
            if variant_key in seen_keys:
                continue
            seen_keys.add(variant_key)
            model_variants.append(
                {
                    "model_id": model_id,
                    "include_reasoning": include_reasoning,
                }
            )
            continue

        parent_model = str(getattr(item, "_model_parent", "") or "").strip()
        if parent_model:
            selected_parent_models.add(parent_model)

    for parent_model in selected_parent_models:
        for include_reasoning in (False, True):
            variant_key = (parent_model, include_reasoning)
            if variant_key in seen_keys:
                continue
            seen_keys.add(variant_key)
            model_variants.append(
                {
                    "model_id": parent_model,
                    "include_reasoning": include_reasoning,
                }
            )

    if not model_variants:
        return None

    ordered_models: list[str] = []
    seen_models: set[str] = set()
    for variant in model_variants:
        model_id = str(variant.get("model_id") or "").strip()
        if not model_id or model_id in seen_models:
            continue
        seen_models.add(model_id)
        ordered_models.append(model_id)

    return ordered_models, model_variants
