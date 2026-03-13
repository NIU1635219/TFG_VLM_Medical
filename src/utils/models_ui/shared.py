"""Helpers compartidos para pantallas del Model Manager."""

from __future__ import annotations

import threading
from typing import Any


def pick_download_option(
    ctx: dict[str, Any],
    options: list[dict[str, Any]],
    title: str,
    local_keys: set[str] | None = None,
    model_ref: str | None = None,
) -> dict[str, Any] | None:
    """Muestra selector de cuantizaciones y devuelve la opción elegida."""
    if not options:
        return None

    style = ctx["Style"]
    menu_item = ctx["MenuItem"]
    interactive_menu = ctx["interactive_menu"]
    lms_menu_helpers = ctx["lms_menu_helpers"]
    log = ctx["log"]
    wait_for_any_key = ctx["wait_for_any_key"]

    normalized_local = set(str(item).strip().lower() for item in (local_keys or []))
    model_hint = lms_menu_helpers.model_hint_from_ref(model_ref)

    def is_local_option(entry: dict[str, Any]) -> bool:
        """Indica si una opción ya existe en local.

        Args:
            entry: Opción de descarga candidata.

        Returns:
            `True` si la opción ya está instalada.
        """
        return lms_menu_helpers.option_is_local(entry, normalized_local, model_hint)

    def capacity_status(entry: dict[str, Any]) -> str:
        """Evalúa el encaje en memoria para la opción.

        Args:
            entry: Opción de descarga candidata.

        Returns:
            Estado de capacidad (`gpu`, `hybrid`, `no_fit`, `unknown`).
        """
        return lms_menu_helpers.capacity_status(entry.get("size_bytes"), ctx["gpu_mem_bytes"], ctx["ram_mem_bytes"])

    sorted_options = lms_menu_helpers.sort_download_options(options)
    snapshot = ctx["build_download_status_snapshot"](ui_width=ctx["current_ui_width"]())
    running_signatures = ctx["running_download_signatures"](snapshot=snapshot)

    option_items = []
    for entry in sorted_options:
        quant = entry.get("quantization") or "unknown"
        rec = " ★" if entry.get("recommended") else ""
        size_label = lms_menu_helpers.format_size(entry.get("size_bytes"))
        is_local = is_local_option(entry)
        cap = capacity_status(entry)
        is_downloading = ctx["download_signature_for_option"](entry, str(model_ref or "")).lower() in running_signatures

        if is_local:
            status_label = f"{style.OKBLUE}LOCAL{style.ENDC}"
        elif is_downloading:
            status_label = f"{style.OKCYAN}DOWNLOADING{style.ENDC}"
        elif cap == "gpu":
            status_label = f"{style.OKGREEN}INSTALL{style.ENDC}"
        elif cap == "hybrid":
            status_label = f"{style.WARNING}INSTALL+RAM{style.ENDC}"
        elif cap == "no_fit":
            status_label = f"{style.FAIL}NO-FIT{style.ENDC}"
        else:
            status_label = f"{style.WARNING}INSTALL?{style.ENDC}"

        label = f"{quant:<14} | {size_label:<8}{rec} | {status_label}"
        blocked = is_local or is_downloading or cap == "no_fit"
        if blocked:
            if is_local:
                reason = "Ya descargado"
            elif is_downloading:
                reason = "Ya se está descargando"
            else:
                reason = "No cabe ni en GPU ni en RAM"
            option_items.append(menu_item(label, action=lambda: None, description=f"{reason}: {str(entry.get('name') or '')}"))
        else:
            option_items.append(menu_item(label, action=lambda e=entry: e, description=str(entry.get("name") or "")))

    option_items.append(menu_item("Back", lambda: None, description="Volver al menú anterior."))

    def q_header() -> None:
        """Renderiza cabecera del selector de cuantización."""
        ui_width = ctx["current_ui_width"]()
        ctx["kit"].banner(title=title, width=ui_width)
        print(" " + ctx["divider"](ui_width))
        print(f" {style.DIM}Selecciona una cuantización compatible y confirma con ENTER.{style.ENDC}")
        print(" " + ctx["divider"](ui_width))

    selected = interactive_menu(
        option_items,
        header_func=q_header,
        menu_id="lms_quant_selector",
        info_text=lambda: ctx["format_download_info_text"](include_running_lines=True),
        nav_hint_text="↑/↓ navegar cuantizaciones · ENTER seleccionar · ESC volver",
        dynamic_info_top=True,
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


def download_with_progress(ctx: dict[str, Any], selected_option: dict[str, Any]) -> tuple[bool, str]:
    """Lanza una descarga en segundo plano y actualiza estado incremental."""
    lms_models = ctx["lms_models"]
    lms_menu_helpers = ctx["lms_menu_helpers"]

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
    job_id = ctx["create_download_job"](model=option_model_name, quant=option_quant)

    def on_progress(update: Any) -> None:
        """Actualiza progreso incremental de descarga.

        Args:
            update: Objeto de progreso emitido por el SDK de LM Studio.
        """
        downloaded = float(getattr(update, "downloaded_bytes", 0) or 0)
        total = float(getattr(update, "total_bytes", 0) or 0)
        speed = float(getattr(update, "speed_bytes_per_second", 0) or 0)
        ctx["update_download_job"](job_id, downloaded=downloaded, total=total, speed=speed)

    def on_finalize(_result: Any = None) -> None:
        """Normaliza valores finales al cerrar la descarga.

        Args:
            _result: Resultado final opcional del SDK (no utilizado).
        """
        snapshot = next((item for item in ctx["snapshot_download_jobs"]() if item.get("job_id") == job_id), None)
        if snapshot is None:
            return
        final_downloaded = float(snapshot.get("downloaded") or 0.0)
        final_total = float(snapshot.get("total") or 0.0)
        if final_total <= 0:
            final_total = final_downloaded
        final_total = max(final_total, final_downloaded)
        ctx["update_download_job"](job_id, downloaded=final_total, total=final_total, speed=0.0)

    def worker() -> None:
        """Ejecuta la descarga en segundo plano y cierra estado del job."""
        ok, info = lms_models.download_option(selected_option, on_progress=on_progress, on_finalize=on_finalize)
        snapshot = next((item for item in ctx["snapshot_download_jobs"]() if item.get("job_id") == job_id), None)
        if snapshot is None:
            return
        final_downloaded = float(snapshot.get("downloaded") or 0.0)
        final_total = float(snapshot.get("total") or 0.0)
        if ok:
            final_total = max(final_total, final_downloaded, 1.0)
            ctx["update_download_job"](
                job_id,
                status="completed",
                downloaded=final_total,
                total=final_total,
                speed=0.0,
                message=str(info),
            )
        else:
            normalized = str(info or "error desconocido")
            status = "cancelled" if "cancel" in normalized.lower() else "failed"
            ctx["update_download_job"](job_id, status=status, speed=0.0, message=normalized)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return True, f"Descarga iniciada en segundo plano: {option_model_name} [{option_quant}]"
