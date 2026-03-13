"""UI del LM Studio Model Manager extraída de setup_env."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .models_ui.custom_screen import download_custom_model_ui
from .models_ui.registry_screen import download_registry_model_ui
from .models_ui.lms_download_manager import DownloadJobState

if TYPE_CHECKING:
    from .menu_kit import AppContext, UIKit


def manage_models_menu_ui(kit: "UIKit", app: "AppContext") -> None:
    """Despliega la interfaz de gestión de modelos de LM Studio."""
    style = kit.style
    menu_item = kit.MenuItem
    lms_models = app.lms_models
    lms_menu_helpers = app.lms_menu_helpers

    print_banner = app.print_banner
    clear_screen_ansi = kit.clear
    input_with_esc = kit.input
    wait_for_any_key = kit.wait
    log = kit.log
    ask_user = kit.ask
    interactive_menu = kit.menu
    get_installed_lms_models = app.get_installed_lms_models
    read_key = kit.read_key
    os_module = kit.os_module
    msvcrt_module = kit.msvcrt_module

    def current_ui_width() -> int:
        """Obtiene el ancho actual del terminal.

        Returns:
            Número de columnas disponibles para render.
        """
        return kit.width()

    def divider(width: int) -> str:
        """Construye un separador horizontal.

        Args:
            width: Ancho objetivo del separador.

        Returns:
            Cadena con separador adaptado al ancho indicado.
        """
        return kit.divider(width)

    djs = DownloadJobState(style=style, cursor_memory=kit.cursor_memory)

    def create_download_job(*, model: str, quant: str) -> str:
        """Crea un trabajo de descarga en estado running.

        Args:
            model: Referencia del modelo.
            quant: Cuantización seleccionada.

        Returns:
            Identificador único del trabajo creado.
        """
        return djs.create_job(model=model, quant=quant)

    def update_download_job(job_id: str, **updates: Any) -> None:
        """Actualiza estado o progreso de un trabajo.

        Args:
            job_id: Identificador del job a modificar.
            **updates: Campos de estado a sobreescribir.
        """
        djs.update_job(job_id, **updates)

    def snapshot_download_jobs() -> list[dict[str, Any]]:
        """Devuelve instantánea ordenada de trabajos.

        Returns:
            Lista de trabajos activos con su estado actual.
        """
        return djs.snapshot()

    def download_signature_for_option(entry: dict[str, Any], model_ref: str) -> str:
        """Genera firma estable modelo|quant para deduplicación.

        Args:
            entry: Opción de descarga seleccionada.
            model_ref: Referencia base del modelo.

        Returns:
            Firma normalizada para identificar descargas equivalentes.
        """
        quant = str(entry.get("quantization") or "").strip().upper()
        if not quant or quant == "UNKNOWN":
            quant = lms_menu_helpers.extract_quantization_from_text(entry.get("name"))
        indexed = str(entry.get("indexed_model_identifier") or "").strip().lower()
        if indexed and "/" in indexed:
            model_key = indexed.rsplit("/", 1)[0]
        elif indexed:
            model_key = indexed
        else:
            model_key = lms_menu_helpers.model_hint_from_ref(model_ref)
        return f"{model_key}|{str(quant or 'UNKNOWN').upper()}"

    def running_download_signatures(*, snapshot: dict[str, Any] | None = None) -> set[str]:
        """Obtiene firmas de descargas en curso.

        Args:
            snapshot: Snapshot opcional precomputado para evitar recomputar.

        Returns:
            Conjunto de firmas activas.
        """
        if snapshot is not None:
            return set(str(sig) for sig in snapshot.get("running_signatures", set()))
        return djs.running_signatures()

    def build_download_status_snapshot(*, ui_width: int) -> dict[str, Any]:
        """Construye resumen renderizable de descargas.

        Args:
            ui_width: Ancho actual de la UI para ajuste visual.

        Returns:
            Snapshot con resumen, líneas activas y metadatos de estado.
        """
        return djs.build_status_snapshot(ui_width=ui_width)

    def format_download_info_text(*, include_running_lines: bool, snapshot: dict[str, Any] | None = None) -> str:
        """Formatea texto de estado para el panel de información.

        Args:
            include_running_lines: Si `True`, incluye detalle de descargas activas.
            snapshot: Snapshot opcional para reutilizar estado ya calculado.

        Returns:
            Texto listo para `info_text` del menú.
        """
        if snapshot is not None:
            summary = str(snapshot.get("summary") or "")
            if not summary:
                return ""
            if not include_running_lines:
                return summary
            lines = list(snapshot.get("running_lines") or [])
            return "\n".join([summary, *lines]) if lines else summary
        return djs.format_info_text(include_running_lines=include_running_lines, ui_width=current_ui_width())

    gpu_mem_bytes = lms_menu_helpers.detect_gpu_memory_bytes()
    ram_mem_bytes = lms_menu_helpers.detect_ram_memory_bytes(app.psutil)

    ctx: dict[str, Any] = {
        "kit": kit,
        "Style": style,
        "MenuItem": menu_item,
        "lms_models": lms_models,
        "lms_menu_helpers": lms_menu_helpers,
        "MODELS_REGISTRY": app.MODELS_REGISTRY,
        "MENU_CURSOR_MEMORY": kit.cursor_memory,
        "print_banner": print_banner,
        "clear_screen_ansi": clear_screen_ansi,
        "input_with_esc": input_with_esc,
        "wait_for_any_key": wait_for_any_key,
        "log": log,
        "ask_user": ask_user,
        "interactive_menu": interactive_menu,
        "get_installed_lms_models": get_installed_lms_models,
        "read_key": read_key,
        "os_module": os_module,
        "msvcrt_module": msvcrt_module,
        "incremental_panel_renderer_cls": kit.IncrementalPanelRenderer,
        "current_ui_width": current_ui_width,
        "divider": divider,
        "create_download_job": create_download_job,
        "update_download_job": update_download_job,
        "snapshot_download_jobs": snapshot_download_jobs,
        "download_signature_for_option": download_signature_for_option,
        "running_download_signatures": running_download_signatures,
        "build_download_status_snapshot": build_download_status_snapshot,
        "format_download_info_text": format_download_info_text,
        "gpu_mem_bytes": gpu_mem_bytes,
        "ram_mem_bytes": ram_mem_bytes,
    }

    def delete_model_ui(model_key: str) -> None:
        """Elimina un modelo local con confirmación previa."""
        if ask_user(
            f"Confirm delete local model '{model_key}'?",
            "n",
            info_text=lambda: format_download_info_text(include_running_lines=True),
        ):
            ok, info = lms_models.remove_local_model(model_key)
            if ok:
                log(f"Modelo eliminado localmente: {info}", "success")
            else:
                log(f"No se pudo eliminar el modelo: {info}", "error")
        else:
            log("Delete cancelled by user.", "warning")
        wait_for_any_key()

    def menu_header() -> None:
        """Imprime el encabezado del menú principal de modelos."""
        print_banner()
        print(f"{style.BOLD} LM STUDIO MODEL MANAGER {style.ENDC}")
        print()

    while True:
        local_models = lms_models.list_installed_variants_flat()
        loaded_keys = lms_models.list_loaded_llm_model_keys()

        columns = [
            kit.TableColumn("MODEL", min_width=24),
            kit.TableColumn("QUANT", fixed_width=12),
            kit.TableColumn("STATUS", fixed_width=10),
        ]

        previous_base_name = None
        table_rows = []
        for model in local_models:
            model_key = str(model.get("model_key", "")).strip()
            if not model_key:
                continue

            quant_label = lms_menu_helpers.detect_local_model_quantization(model)
            base_name = model_key.rsplit("@", 1)[0].strip() if "@" in model_key else model_key
            shown_name = base_name if base_name != previous_base_name else "\u21b3"
            previous_base_name = base_name

            in_loaded = model_key in loaded_keys
            quant_color = "OKCYAN" if quant_label != "unknown" else "DIM"
            status_color = "OKGREEN" if in_loaded else "DIM"
            status_text = "LOADED" if in_loaded else "LOCAL"

            table_rows.append(
                kit.TableRow(
                    cells=[shown_name, str(quant_label).upper(), status_text],
                    action=lambda mk=model_key: delete_model_ui(mk),
                    description="Pulsa ENTER para eliminar este modelo local.",
                    selected_cells=[shown_name, str(quant_label).upper(), "REMOVE"],
                    cell_colors=[None, quant_color, status_color],
                    selected_cell_colors=[None, None, "FAIL"],
                )
            )

        options = [
            *kit.build_table_items(columns, table_rows),
            menu_item(
                "Download from MODELS_REGISTRY (quantizations)...",
                lambda: download_registry_model_ui(ctx),
                description="Descarga variantes por cuantización de modelos del registro.",
            ),
            menu_item(
                "Download by model ID/URL (quantizations)...",
                lambda: download_custom_model_ui(ctx),
                description="Descarga variantes por cuantización usando id/url manual.",
            ),
            menu_item("Back", lambda: "BACK", description="Vuelve al menú de Tests & Models."),
        ]

        choice = interactive_menu(
            options,
            header_func=menu_header,
            menu_id="manage_models_menu",
            info_text=lambda: format_download_info_text(include_running_lines=True),
            nav_hint=True,
            nav_hint_text="↑/↓ navegar · ENTER seleccionar · ESC volver",
            left_margin=0,
            dynamic_info_top=True,
        )
        if not choice:
            break

        clear_screen_ansi()
        if not callable(getattr(choice, "action", None)):
            continue
        result = choice.action()
        if result == "BACK":
            break
