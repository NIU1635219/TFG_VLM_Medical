"""UI del LM Studio Model Manager extraída de setup_env."""

from __future__ import annotations

import re
import shutil
import threading
import time
from typing import Any

from .setup_ui_io import (
    compute_render_decision,
    get_ui_width,
    paint_dynamic_lines,
    render_title_banner,
    should_repaint_static,
    wrap_plain_text,
)


_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _visible_len(text: str) -> int:
    """Calcula la longitud visible de una cadena ignorando códigos ANSI.

    Args:
        text (str): Texto con o sin estilos ANSI.

    Returns:
        int: Longitud visible en caracteres.
    """
    return len(_ANSI_RE.sub("", str(text or "")))


def _wrap_colored_chunks(chunks: list[str], *, prefix: str, max_width: int) -> list[str]:
    """Envuelve segmentos coloreados sin romper secuencias ANSI.

    Args:
        chunks (list[str]): Segmentos de texto potencialmente coloreados.
        prefix (str): Prefijo fijo al inicio de cada línea.
        max_width (int): Ancho máximo visible permitido por línea.

    Returns:
        list[str]: Líneas compuestas que respetan el ancho visible.
    """
    if not chunks:
        return [prefix]

    prefix_len = _visible_len(prefix)
    separator = "  "
    separator_len = _visible_len(separator)

    lines: list[str] = []
    current = prefix
    current_len = prefix_len

    for chunk in chunks:
        chunk_len = _visible_len(chunk)
        extra = chunk_len if current_len == prefix_len else (separator_len + chunk_len)
        if current_len + extra > max_width and current_len > prefix_len:
            lines.append(current)
            current = prefix + chunk
            current_len = prefix_len + chunk_len
            continue

        if current_len == prefix_len:
            current += chunk
            current_len += chunk_len
        else:
            current += separator + chunk
            current_len += separator_len + chunk_len

    lines.append(current)
    return lines


class _FallbackIncrementalPanelRenderer:
    """
    Renderizador fallback simple para cuando no se inyecta el IncrementalPanelRenderer completo.
    
    Implementa la misma interfaz (reset, render) pero con una implementación mínima.
    """

    def __init__(self, *, clear_screen_fn, render_static_fn):
        """
        Inicializa el renderer fallback.

        Args:
            clear_screen_fn (Callable): Función para limpiar pantalla.
            render_static_fn (Callable): Función para renderizar parte estática.
        """
        self.clear_screen_fn = clear_screen_fn
        self.render_static_fn = render_static_fn
        self.static_rendered = False
        self.prev_dynamic_visual_rows = 0
        self.prev_terminal_width: int | None = None

    def reset(self):
        """Resetea el estado del renderer."""
        self.static_rendered = False
        self.prev_dynamic_visual_rows = 0
        self.prev_terminal_width = None

    def render(self, dynamic_lines, *, force_full=False):
        """
        Renderiza líneas dinámicas sobre la parte estática.

        Args:
            dynamic_lines (list[str]): Líneas a mostrar.
            force_full (bool, optional): Forzar limpieza completa. Defaults to False.
        """
        terminal_width = shutil.get_terminal_size(fallback=(120, 30)).columns
        decision = compute_render_decision(
            dynamic_lines=dynamic_lines,
            terminal_width=terminal_width,
            prev_terminal_width=self.prev_terminal_width,
            static_rendered=self.static_rendered,
            force_full=force_full,
        )
        self.prev_terminal_width = terminal_width

        has_wrapped_lines = bool(decision["has_wrapped_lines"])
        current_dynamic_rows = int(decision["current_dynamic_rows"])
        effective_force_full = bool(decision["effective_force_full"])
        use_incremental_frame = bool(decision["use_incremental_frame"])

        if should_repaint_static(
            use_incremental=True,
            static_rendered=self.static_rendered,
            force_full=effective_force_full,
            has_wrapped_lines=has_wrapped_lines,
        ):
            self.clear_screen_fn()
            self.render_static_fn()
            self.static_rendered = True
            self.prev_dynamic_visual_rows = 0

        self.prev_dynamic_visual_rows = paint_dynamic_lines(
            dynamic_lines=dynamic_lines,
            prev_dynamic_visual_rows=self.prev_dynamic_visual_rows,
            current_dynamic_rows=current_dynamic_rows,
            use_incremental_frame=use_incremental_frame,
        )


def manage_models_menu_ui(ctx: dict[str, Any]) -> None:
    """
    Despliega la interfaz de usuario para la pro de modelos de LM Studio.
    
    Permite:
    - Listar modelos instalados y su estado (cargado/local).
    - Eliminar modelos locales.
    - Navegar por el registro de modelos (MODELS_REGISTRY).
    - Buscar y descargar cuantizaciones específicas.
    - Descargar modelos por ID/URL personalizada.
    
    Args:
        ctx (dict): Contexto de la aplicación con dependencias inyectadas 
                   (estilos, helpers, funciones de UI, configuraciones).
    """
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
    os_module = ctx.get("os_module")
    msvcrt_module = ctx.get("msvcrt_module")

    def _current_ui_width() -> int:
        """Obtiene el ancho de UI actual para render dinámico.

        Returns:
            int: Ancho usable de UI para el frame actual.
        """
        try:
            cols = int(shutil.get_terminal_size(fallback=(120, 30)).columns)
        except Exception:
            cols = 120
        max_width = max(60, cols - 4)
        return get_ui_width(shutil_module=shutil, max_width=max_width)

    def _divider(width: int) -> str:
        """Construye una línea separadora alineada con banners.

        Args:
            width (int): Ancho interno del contenido.

        Returns:
            str: Separador horizontal con ancho visual consistente.
        """
        return "─" * (max(8, int(width)) + 2)

    shared_download_state = MENU_CURSOR_MEMORY.get("__lms_download_state__")
    if not isinstance(shared_download_state, dict):
        shared_download_state = {
            "lock": threading.Lock(),
            "jobs": {},
            "counter": 0,
        }
        MENU_CURSOR_MEMORY["__lms_download_state__"] = shared_download_state

    downloads_lock = shared_download_state.get("lock")
    if downloads_lock is None:
        downloads_lock = threading.Lock()
        shared_download_state["lock"] = downloads_lock

    download_jobs = shared_download_state.get("jobs")
    if not isinstance(download_jobs, dict):
        download_jobs = {}
        shared_download_state["jobs"] = download_jobs

    FINISHED_STATUS_TTL_SECONDS = 5.0

    def _new_job_id() -> str:
        """
        Genera un nuevo ID para un nuevo trabajo de descarga.
        
        Returns:
            str: ID único para el nuevo trabajo.
        """
        with downloads_lock:
            counter = int(shared_download_state.get("counter", 0)) + 1
            shared_download_state["counter"] = counter
            return f"job-{counter}"

    def _create_download_job(*, model: str, quant: str) -> str:
        """
        Crea un nuevo trabajo de descarga para un modelo específico.
        
        Args:
            model (str): ID o URL del modelo.
            quant (str): Tipo de cuantización.
        
        Returns:
            str: ID del nuevo trabajo de descarga.
        """
        job_id = _new_job_id()
        now = time.time()
        with downloads_lock:
            download_jobs[job_id] = {
                "status": "running",
                "model": model,
                "quant": quant,
                "downloaded": 0.0,
                "total": 0.0,
                "speed": 0.0,
                "message": "",
                "created_at": now,
                "updated_at": now,
                "finished_at": None,
                "signature": f"{str(model or '').strip().lower()}|{str(quant or '').strip().upper()}",
            }
        return job_id

    def _update_download_job(job_id: str, **updates: Any) -> None:
        """
        Actualiza el estado de un trabajo de descarga.
        
        Args:
            job_id (str): ID del trabajo de descarga.
            **updates (Any): Actualizaciones para el trabajo de descarga.
        """
        with downloads_lock:
            if job_id not in download_jobs:
                return
            download_jobs[job_id].update(updates)
            if str(download_jobs[job_id].get("status") or "") in ("completed", "failed", "cancelled"):
                if download_jobs[job_id].get("finished_at") is None:
                    download_jobs[job_id]["finished_at"] = time.time()
            else:
                download_jobs[job_id]["finished_at"] = None
            download_jobs[job_id]["updated_at"] = time.time()

    def _prune_finished_jobs_locked(now_ts: float) -> None:
        """
        Elimina trabajos de descarga que han estado en estado finalizado por un tiempo excedido.
        
        Args:
            now_ts (float): Timestamp actual.
        """
        stale_ids = []
        for job_id, job in download_jobs.items():
            status = str(job.get("status") or "")
            finished_at = job.get("finished_at")
            if status in ("completed", "failed", "cancelled") and isinstance(finished_at, (int, float)):
                if (now_ts - float(finished_at)) >= FINISHED_STATUS_TTL_SECONDS:
                    stale_ids.append(job_id)
        for job_id in stale_ids:
            download_jobs.pop(job_id, None)

    def _snapshot_download_jobs() -> list[dict[str, Any]]:
        """
        Genera una instantánea de los trabajos de descarga.
        
        Returns:
            list[dict[str, Any]]: Lista de trabajos de descarga ordenada por fecha de creación.
        """
        with downloads_lock:
            _prune_finished_jobs_locked(time.time())
            ordered = sorted(download_jobs.items(), key=lambda item: float(item[1].get("created_at") or 0.0))
            return [{"job_id": jid, **dict(data)} for jid, data in ordered]

    def _download_signature_for_option(entry: dict[str, Any], model_ref: str) -> str:
        """
        Genera una firma única para una opción de descarga.
        
        Args:
            entry (dict[str, Any]): Entrada del registro de modelos.
            model_ref (str): Referencia al modelo.
        
        Returns:
            str: Firma única para la opción de descarga.
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

    def _running_download_signatures(*, snapshot: dict[str, Any] | None = None) -> set[str]:
        """
        Obtiene las firmas de descargas en ejecución.
        
        Args:
            snapshot (dict[str, Any] | None, optional): Instantánea de descargas. Defaults to None.
        
        Returns:
            set[str]: Conjunto de firmas de descargas en ejecución.
        """
        if snapshot is not None:
            return set(str(sig) for sig in snapshot.get("running_signatures", set()))
        built = _build_download_status_snapshot(ui_width=_current_ui_width())
        return set(str(sig) for sig in built.get("running_signatures", set()))

    def _trim_text(text: str, max_len: int) -> str:
        """
        Trunca un texto si excede una longitud máxima.
        
        Args:
            text (str): Texto a truncar.
            max_len (int): Longitud máxima del texto.
        
        Returns:
            str: Texto truncado si es necesario.
        """
        raw = str(text or "")
        if len(raw) <= max_len:
            return raw
        if max_len <= 1:
            return raw[:max_len]
        return raw[: max_len - 1] + "…"

    def _build_download_status_snapshot(*, ui_width: int) -> dict[str, Any]:
        """Construye un snapshot consistente de estado de descargas para un frame.

        Args:
            ui_width (int): Ancho de UI activo para ajustar barras y truncado.

        Returns:
            dict[str, Any]: Resumen, líneas renderizables y metadatos de estado.
        """
        jobs = _snapshot_download_jobs()
        if not jobs:
            return {
                "summary": "",
                "running_lines": [],
                "signature": "",
                "running_signatures": set(),
                "completed_ids": set(),
            }

        running = [job for job in jobs if str(job.get("status")) == "running"]
        done = [job for job in jobs if str(job.get("status")) in ("completed", "failed", "cancelled")]
        recent_done: list[dict[str, Any]] = []
        now_ts = time.time()
        for job in done:
            finished_at = job.get("finished_at")
            if isinstance(finished_at, (int, float)) and (now_ts - float(finished_at)) < FINISHED_STATUS_TTL_SECONDS:
                recent_done.append(job)

        completed_ids = {
            str(job.get("job_id"))
            for job in jobs
            if str(job.get("status") or "") == "completed"
        }
        running_signatures = {
            str(job.get("signature") or "").strip().lower()
            for job in running
            if str(job.get("signature") or "").strip()
        }

        if not running and not recent_done:
            return {
                "summary": "",
                "running_lines": [],
                "signature": "",
                "running_signatures": running_signatures,
                "completed_ids": completed_ids,
            }

        summary = (
            f"Descargas: {len(running)} activas · {len(done)} finalizadas"
            if running
            else "Resultados recientes (auto-ocultar 5s)"
        )
        running_lines: list[str] = []
        bar_width = max(12, min(26, ui_width - 46))
        for job in running:
            model = _trim_text(str(job.get("model") or "model"), max(18, ui_width - 20))
            quant = str(job.get("quant") or "unknown").upper()
            downloaded = float(job.get("downloaded") or 0.0)
            total = float(job.get("total") or 0.0)
            speed = float(job.get("speed") or 0.0)
            pct = int((downloaded / total) * 100.0) if total > 0 else 0
            pct = max(0, min(pct, 100))
            filled = int((pct / 100.0) * bar_width)
            filled = max(0, min(filled, bar_width))
            bar = f"{Style.OKCYAN}{'█' * filled}{Style.DIM}{'░' * (bar_width - filled)}{Style.ENDC}"

            total_mb = total / (1024 * 1024) if total > 0 else 0.0
            downloaded_mb = downloaded / (1024 * 1024)
            speed_mb = speed / (1024 * 1024) if speed > 0 else 0.0

            running_lines.append(f"{Style.OKCYAN}↓ {model} [{quant}]{Style.ENDC}")
            running_lines.append(
                f"  {bar} {Style.BOLD}{pct:>3d}%{Style.ENDC} · {downloaded_mb:.1f}/{total_mb:.1f} MB · {speed_mb:.1f} MB/s"
            )

        for job in recent_done:
            model = _trim_text(str(job.get("model") or "model"), max(18, ui_width - 20))
            quant = str(job.get("quant") or "unknown").upper()
            status = str(job.get("status") or "").lower()
            if status == "completed":
                running_lines.append(f"{Style.OKGREEN}✔ {model} [{quant}] completado{Style.ENDC}")
            elif status == "failed":
                running_lines.append(f"{Style.FAIL}✖ {model} [{quant}] error{Style.ENDC}")
            else:
                running_lines.append(f"{Style.WARNING}⚠ {model} [{quant}] cancelado{Style.ENDC}")
        signature = "\n".join([summary, *running_lines])
        return {
            "summary": summary,
            "running_lines": running_lines,
            "signature": signature,
            "running_signatures": running_signatures,
            "completed_ids": completed_ids,
        }

    def _format_download_info_text(*, include_running_lines: bool, snapshot: dict[str, Any] | None = None) -> str:
        """
        Formatea el texto de información de descargas.
        
        Args:
            include_running_lines (bool): Incluir líneas de descargas en ejecución.
            snapshot (dict[str, Any] | None, optional): Instantánea de descargas. Defaults to None.
        
        Returns:
            str: Texto formateado de información de descargas.
        """
        current_snapshot = snapshot or _build_download_status_snapshot(ui_width=_current_ui_width())
        summary = str(current_snapshot.get("summary") or "")
        running_lines = list(current_snapshot.get("running_lines") or [])
        if not summary:
            return ""
        if not include_running_lines or not running_lines:
            return summary
        return "\n".join([summary, *running_lines])

    gpu_mem_bytes = lms_menu_helpers.detect_gpu_memory_bytes()
    ram_mem_bytes = lms_menu_helpers.detect_ram_memory_bytes(psutil)

    def _pick_download_option(options, title, local_keys=None, model_ref=None):
        """
        Selecciona una opción de descarga.
        
        Args:
            options (list[dict[str, Any]]): Lista de opciones de descarga.
            title (str): Título de la selección.
            local_keys (list[str] | None, optional): Claves locales. Defaults to None.
            model_ref (str | None, optional): Referencia al modelo. Defaults to None.
        
        Returns:
            dict[str, Any] | None: Opción seleccionada o None si no hay opciones.
        """
        if not options:
            return None

        normalized_local = set(str(item).strip().lower() for item in (local_keys or []))
        model_hint = lms_menu_helpers.model_hint_from_ref(model_ref)

        def _is_local_option(entry):
            """
            Verifica si una opción es local.
            
            Args:
                entry (dict[str, Any]): Entrada de la opción.
            
            Returns:
                bool: True si la opción es local, False en caso contrario.
            """
            return lms_menu_helpers.option_is_local(entry, normalized_local, model_hint)

        def _capacity_status(entry):
            """
            Obtiene el estado de capacidad de una opción.
            
            Args:
                entry (dict[str, Any]): Entrada de la opción.
            
            Returns:
                str: Estado de capacidad de la opción.
            """
            size_bytes = entry.get("size_bytes")
            return lms_menu_helpers.capacity_status(size_bytes, gpu_mem_bytes, ram_mem_bytes)

        sorted_options = lms_menu_helpers.sort_download_options(options)
        snapshot = _build_download_status_snapshot(ui_width=_current_ui_width())
        running_signatures = _running_download_signatures(snapshot=snapshot)
        option_items = []
        for entry in sorted_options:
            quant = entry.get("quantization") or "unknown"
            rec = " ★" if entry.get("recommended") else ""
            size_label = lms_menu_helpers.format_size(entry.get("size_bytes"))
            is_local_option = _is_local_option(entry)
            cap_status = _capacity_status(entry)
            is_downloading = _download_signature_for_option(entry, str(model_ref or "")).lower() in running_signatures

            if is_local_option:
                status_label = f"{Style.OKBLUE}LOCAL{Style.ENDC}"
            elif is_downloading:
                status_label = f"{Style.OKCYAN}DOWNLOADING{Style.ENDC}"
            elif cap_status == "gpu":
                status_label = f"{Style.OKGREEN}INSTALL{Style.ENDC}"
            elif cap_status == "hybrid":
                status_label = f"{Style.WARNING}INSTALL+RAM{Style.ENDC}"
            elif cap_status == "no_fit":
                status_label = f"{Style.FAIL}NO-FIT{Style.ENDC}"
            else:
                status_label = f"{Style.WARNING}INSTALL?{Style.ENDC}"

            label = f"{quant:<14} | {size_label:<8}{rec} | {status_label}"
            is_blocked = is_local_option or is_downloading or cap_status == "no_fit"

            if is_blocked:
                if is_local_option:
                    reason = "Ya descargado"
                elif is_downloading:
                    reason = "Ya se está descargando"
                else:
                    reason = "No cabe ni en GPU ni en RAM"
                option_items.append(MenuItem(label, action=lambda: None, description=f"{reason}: {str(entry.get('name') or '')}"))
            else:
                option_items.append(MenuItem(label, action=lambda e=entry: e, description=str(entry.get("name") or "")))

        option_items.append(MenuItem("Back", lambda: None, description="Volver al menú anterior."))

        def q_header():
            """
            Genera el encabezado para la selección de cuantización.
            
            Returns:
                str: Encabezado formateado.
            """
            ui_width = _current_ui_width()
            render_title_banner(title=title, style=Style, width=ui_width)
            print(" " + _divider(ui_width))
            print(f" {Style.DIM}Selecciona una cuantización compatible y confirma con ENTER.{Style.ENDC}")
            print(" " + _divider(ui_width))

        selected = interactive_menu(
            option_items,
            header_func=q_header,
            menu_id="lms_quant_selector",
            info_text=lambda: _format_download_info_text(include_running_lines=True),
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

    def _download_with_progress(selected_option):
        """
        Descarga un modelo con progreso.
        
        Args:
            selected_option (dict[str, Any]): Opción seleccionada para descargar.
        """
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
        job_id = _create_download_job(model=option_model_name, quant=option_quant)

        def on_progress(update):
            """
            Maneja el progreso de la descarga.
            
            Args:
                update (object): Actualización del progreso.
            """
            downloaded = float(getattr(update, "downloaded_bytes", 0) or 0)
            total = float(getattr(update, "total_bytes", 0) or 0)
            speed = float(getattr(update, "speed_bytes_per_second", 0) or 0)

            _update_download_job(job_id, downloaded=downloaded, total=total, speed=speed)

        def on_finalize(_result=None):
            """
            Maneja la finalización de la descarga.
            
            Args:
                _result (object, optional): Resultado de la descarga. Defaults to None.
            """
            snapshot = next((item for item in _snapshot_download_jobs() if item.get("job_id") == job_id), None)
            if snapshot is None:
                return
            final_downloaded = float(snapshot.get("downloaded") or 0.0)
            final_total = float(snapshot.get("total") or 0.0)
            if final_total <= 0:
                final_total = final_downloaded
            final_total = max(final_total, final_downloaded)
            _update_download_job(job_id, downloaded=final_total, total=final_total, speed=0.0)

        def _worker():
            """
            Ejecuta el trabajo de descarga.
            """
            ok, info = lms_models.download_option(selected_option, on_progress=on_progress, on_finalize=on_finalize)
            snapshot = next((item for item in _snapshot_download_jobs() if item.get("job_id") == job_id), None)
            if snapshot is None:
                return
            final_downloaded = float(snapshot.get("downloaded") or 0.0)
            final_total = float(snapshot.get("total") or 0.0)
            if ok:
                final_total = max(final_total, final_downloaded, 1.0)
                _update_download_job(
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
                _update_download_job(job_id, status=status, speed=0.0, message=normalized)

        worker = threading.Thread(target=_worker, daemon=True)
        worker.start()
        return True, f"Descarga iniciada en segundo plano: {option_model_name} [{option_quant}]"

    def _read_line_non_blocking(*, prompt: str) -> str | None:
        """Lee texto con edición en línea y refresco reactivo al resize.

        Args:
            prompt (str): Etiqueta del campo de entrada.

        Returns:
            str | None: Texto introducido o None si se cancela con ESC.
        """
        if getattr(os_module, "name", "") != "nt" or msvcrt_module is None:
            return input_with_esc(f"{Style.WARNING} {prompt}{Style.ENDC}")

        panel = incremental_panel_renderer_cls(clear_screen_fn=clear_screen_ansi, render_static_fn=lambda: None)
        current_value = ""
        prev_ui_width: int | None = None

        while True:
            ui_width = _current_ui_width()
            if prev_ui_width is None:
                prev_ui_width = ui_width
            elif prev_ui_width != ui_width:
                panel.reset()
                prev_ui_width = ui_width

            footer_snapshot = _build_download_status_snapshot(ui_width=ui_width)
            footer = str(footer_snapshot.get("summary") or "")
            footer_lines = list(footer_snapshot.get("running_lines") or [])

            def _render_static() -> None:
                """
                Renderiza el panel estático.
                """
                print_banner()
                print(f"{Style.BOLD} LM STUDIO MODEL MANAGER {Style.ENDC}")
                print()
                render_title_banner(title="DOWNLOAD BY MODEL ID/URL (QUANTIZATIONS)", style=Style, width=ui_width)
                print(" " + _divider(ui_width))
                print(f" {Style.DIM}Pega una URL de Hugging Face o un ID exacto. ESC para cancelar.{Style.ENDC}")
                print(f" {Style.DIM}Ejemplos:{Style.ENDC}")
                print(f"   {Style.OKCYAN}https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF{Style.ENDC}")
                print(f"   {Style.OKCYAN}Qwen/Qwen3-VL-8B-Instruct-GGUF{Style.ENDC}")
                print(" " + _divider(ui_width))

            panel.render_static_fn = _render_static

            dynamic_lines: list[str] = []
            if footer:
                dynamic_lines.append(f" {Style.DIM}{footer}{Style.ENDC}")
                for running_line in footer_lines:
                    dynamic_lines.append(f" {running_line}")
                dynamic_lines.append(_divider(ui_width))

            max_input = max(8, ui_width - len(prompt) - 4)
            shown_value = current_value
            if len(shown_value) > max_input:
                shown_value = "…" + shown_value[-(max_input - 1):]

            dynamic_lines.append(f" {Style.WARNING}{prompt}{Style.ENDC}{shown_value}")
            dynamic_lines.append(f" {Style.DIM}[ENTER] confirmar · [Backspace] borrar · [ESC] cancelar{Style.ENDC}")

            panel.render(dynamic_lines)

            if not hasattr(msvcrt_module, "kbhit") or not msvcrt_module.kbhit():
                time.sleep(0.03)
                continue

            try:
                ch = msvcrt_module.getwch()
            except Exception:
                time.sleep(0.03)
                continue

            if ch in ("\r", "\n"):
                return current_value
            if ch == "\x1b":
                return None
            if ch in ("\x08", "\x7f"):
                current_value = current_value[:-1]
                continue
            if ch in ("\x00", "\xe0"):
                try:
                    _ = msvcrt_module.getwch()
                except Exception:
                    pass
                continue
            if ch == "\x03":
                raise KeyboardInterrupt
            if ch.isprintable():
                current_value += ch

    def _download_registry_model_ui():
        """
        Interfaz de usuario para descargar un modelo de registro.
        """
        registry_items = list(MODELS_REGISTRY.items())
        model_cursor = MENU_CURSOR_MEMORY.get("lms_registry_selector", 0)
        model_quant_cursor: dict[str, int] = {}
        options_cache: dict[str, list[dict[str, object]]] = {}
        local_signatures_cache = lms_menu_helpers.build_local_signatures(lms_models.list_local_llm_models())
        seen_completed_jobs: set[str] = set()

        gpu_gb = lms_menu_helpers.bytes_to_gb(gpu_mem_bytes)
        ram_gb = lms_menu_helpers.bytes_to_gb(ram_mem_bytes)

        def _render_registry_static() -> None:
            """
            Renderiza el panel estático para la interfaz de usuario de descarga de modelos de registro.
            """
            ui_width = _current_ui_width()
            print_banner()
            print(f"{Style.BOLD} LM STUDIO MODEL MANAGER {Style.ENDC}")
            print()
            render_title_banner(title="DOWNLOAD FROM MODELS_REGISTRY", style=Style, width=ui_width)
            print(" " + _divider(ui_width))
            print(
                f" {Style.OKBLUE}LOCAL{Style.ENDC} | "
                f"{Style.OKGREEN}INSTALL (GPU){Style.ENDC} | "
                f"{Style.WARNING}INSTALL+RAM{Style.ENDC} | "
                f"{Style.FAIL}NO-FIT{Style.ENDC}"
            )
            print(f" {Style.DIM}Memoria detectada:{Style.ENDC} GPU {gpu_gb:.1f} GB | RAM {ram_gb:.1f} GB")
            print(f" {Style.DIM}↑/↓ modelo · ←/→ cuantización · ENTER seleccionar/descargar · ESC volver{Style.ENDC}")
            print(" " + _divider(ui_width))
            print()

        panel = incremental_panel_renderer_cls(clear_screen_fn=clear_screen_ansi, render_static_fn=_render_registry_static)

        def _get_cached_options(model_ref: str) -> list[dict[str, object]]:
            """
            Obtiene las opciones de descarga缓存.
            
            Args:
                model_ref (str): Referencia al modelo.
            
            Returns:
                list[dict[str, object]]: Lista de opciones de descarga.
            """
            if model_ref in options_cache:
                return options_cache[model_ref]
            raw_options = lms_menu_helpers.sort_download_options(lms_models.get_download_options(model_ref))
            deduped = lms_menu_helpers.dedupe_options_by_quantization(raw_options)
            options_cache[model_ref] = deduped
            return deduped

        def _option_state(
            entry: dict[str, object],
            local_signatures: set[str],
            model_ref: str,
            running_signatures: set[str],
        ) -> tuple[str, str, bool]:
            """
            Obtiene el estado de una opción de descarga.
            
            Args:
                entry (dict[str, object]): Entrada de la opción.
                local_signatures (set[str]): Firmas locales.
                model_ref (str): Referencia al modelo.
                running_signatures (set[str]): Firmas en ejecución.
            
            Returns:
                tuple[str, str, bool]: Tupla con el estado, el token de color y si está bloqueada.
            """
            option_signature = _download_signature_for_option(entry, model_ref).lower()
            if option_signature in running_signatures:
                return "DOWNLOADING", Style.OKCYAN, True
            state_text, color_token, blocked = lms_menu_helpers.option_visual_state(
                entry, local_signatures, model_ref, gpu_mem_bytes, ram_mem_bytes
            )
            color = getattr(Style, color_token, Style.WARNING)
            return state_text, color, blocked

        def _clamp_model_cursor(value: int) -> int:
            """
            Ajusta el cursor del modelo.
            
            Args:
                value (int): Valor del cursor.
            
            Returns:
                int: Cursor ajustado.
            """
            total_rows = len(registry_items) + 1
            if total_rows <= 0:
                return 0
            return value % total_rows

        needs_render = True
        last_footer_download = ""
        prev_ui_width: int | None = None
        while True:
            ui_width = _current_ui_width()
            if prev_ui_width is None:
                prev_ui_width = ui_width
            elif prev_ui_width != ui_width:
                needs_render = True
                prev_ui_width = ui_width
            footer_snapshot = _build_download_status_snapshot(ui_width=ui_width)
            completed_jobs = set(str(job_id) for job_id in footer_snapshot.get("completed_ids", set()))
            if completed_jobs != seen_completed_jobs:
                if completed_jobs - seen_completed_jobs:
                    local_signatures_cache = lms_menu_helpers.build_local_signatures(lms_models.list_local_llm_models())
                    options_cache.clear()
                    needs_render = True
                seen_completed_jobs = completed_jobs

            local_signatures = local_signatures_cache
            running_signatures = _running_download_signatures(snapshot=footer_snapshot)
            model_cursor = _clamp_model_cursor(model_cursor)
            MENU_CURSOR_MEMORY["lms_registry_selector"] = model_cursor

            footer_download = str(footer_snapshot.get("summary") or "")
            footer_running_lines = list(footer_snapshot.get("running_lines") or [])
            footer_signature = str(footer_snapshot.get("signature") or "")
            if not needs_render and footer_signature == last_footer_download:
                key = read_key()
                if key is None:
                    time.sleep(0.05)
                    continue
            else:
                key = None

            if needs_render or footer_signature != last_footer_download:
                selected_description = ""
                dynamic_lines: list[str] = []

                if footer_download:
                    dynamic_lines.append(f" {Style.DIM}{footer_download}{Style.ENDC}")
                    for running_line in footer_running_lines:
                        dynamic_lines.append(f" {running_line}")
                    dynamic_lines.append(_divider(ui_width))
                    dynamic_lines.append("")

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
                        _state_text, color, _blocked = _option_state(entry, local_signatures, model_ref, running_signatures)
                        chip = f"{color}{quant}{Style.ENDC}"
                        if is_selected_model and q_idx == current_q_idx:
                            chip = f"{Style.SELECTED} {chip} {Style.ENDC}"
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
                        _state_text, color, _blocked = _option_state(
                            selected_entry,
                            local_signatures,
                            model_ref,
                            running_signatures,
                        )
                        selected_name = str(selected_entry.get("name") or "")
                        is_local_selected = lms_menu_helpers.option_is_local(selected_entry, local_signatures, model_ref)
                        if _state_text == "DOWNLOADING":
                            short_state = "DOWNLOADING"
                        else:
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
                dynamic_lines.append(_divider(ui_width))
                if selected_description:
                    dynamic_lines.append(f" {Style.DIM}Descripción: {selected_description}{Style.ENDC}")
                dynamic_lines.append(_divider(ui_width))
                dynamic_lines.append(f" {Style.BOLD}[ENTER]: Seleccionar opción.{Style.ENDC}")

                panel.render(dynamic_lines)
                last_footer_download = footer_signature
                needs_render = False

            if key is None:
                key = read_key()
            if key is None:
                time.sleep(0.05)
                continue

            if key == "UP":
                model_cursor = _clamp_model_cursor(model_cursor - 1)
                needs_render = True
                continue
            if key == "DOWN":
                model_cursor = _clamp_model_cursor(model_cursor + 1)
                needs_render = True
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
                needs_render = True
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
                    needs_render = True
                    continue

                q_idx = model_quant_cursor.get(model_key, 0)
                q_idx = max(0, min(q_idx, len(options) - 1))
                selected_option = options[q_idx]
                state_text, _color, blocked = _option_state(selected_option, local_signatures, model_ref, running_signatures)
                if blocked:
                    log(f"La cuantización seleccionada está bloqueada ({state_text}).", "warning")
                    wait_for_any_key()
                    panel.reset()
                    needs_render = True
                    continue

                ok, info = _download_with_progress(selected_option)
                if ok:
                    log(str(info), "success")
                    if model_ref in options_cache:
                        del options_cache[model_ref]
                else:
                    log(f"No se pudo descargar el modelo: {info}", "error")
                    wait_for_any_key()
                panel.reset()
                needs_render = True
                continue

            if key == "ESC":
                return

            time.sleep(0.01)

    def _download_custom_model_ui():
        """
        Interfaz de usuario para descargar un modelo personalizado.
        """
        clear_screen_ansi()
        print_banner()
        print(f"{Style.BOLD} LM STUDIO MODEL MANAGER {Style.ENDC}")
        print()
        ui_width = _current_ui_width()
        render_title_banner(title="DOWNLOAD BY MODEL ID/URL (QUANTIZATIONS)", style=Style, width=ui_width)
        print(" " + _divider(ui_width))
        print(f" {Style.DIM}Pega una URL de Hugging Face o un ID exacto. ESC para cancelar.{Style.ENDC}")
        print(f" {Style.DIM}Ejemplos:{Style.ENDC}")
        print(f"   {Style.OKCYAN}https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF{Style.ENDC}")
        print(f"   {Style.OKCYAN}Qwen/Qwen3-VL-8B-Instruct-GGUF{Style.ENDC}")
        footer_snapshot = _build_download_status_snapshot(ui_width=ui_width)
        footer = str(footer_snapshot.get("summary") or "")
        if footer:
            print(" " + _divider(ui_width))
            for summary_line in wrap_plain_text(footer, max(16, ui_width - 2)):
                print(f" {Style.DIM}{summary_line}{Style.ENDC}")
            for running_line in list(footer_snapshot.get("running_lines") or []):
                print(f" {running_line}")
        print(" " + _divider(ui_width))
        manual_ref = _read_line_non_blocking(prompt="Model id/url: ")
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
            log(str(info), "success")
        else:
            log(f"No se pudo descargar el modelo: {info}", "error")
            wait_for_any_key()

    def _delete_model_ui(model_key):
        """
        Interfaz de usuario para eliminar un modelo local.
        
        Args:
            model_key (str): Clave del modelo.
        """
        if ask_user(
            f"Confirm delete local model '{model_key}'?",
            "n",
            info_text=lambda: _format_download_info_text(include_running_lines=True),
        ):
            ok, info = lms_models.remove_local_model(model_key)
            if ok:
                log(f"Modelo eliminado localmente: {info}", "success")
            else:
                log(f"No se pudo eliminar el modelo: {info}", "error")
        else:
            log("Delete cancelled by user.", "warning")
        wait_for_any_key()

    def _menu_header():
        """
        Imprime el encabezado del menú.
        """
        print_banner()
        print(f"{Style.BOLD} LM STUDIO MODEL MANAGER {Style.ENDC}")
        print()

    def _table_layout() -> tuple[int, int, int]:
        """
        Calcula el diseño de la tabla.
        
        Returns:
            tuple[int, int, int]: Tupla con el ancho de la tabla, el ancho de la columna de modelos y el ancho de la columna de cuantiaciones.
        """
        quant_col_width = 12
        status_col_width = 10
        table_width = _current_ui_width()
        model_col_width = max(24, table_width - (quant_col_width + status_col_width + 6))
        return table_width, model_col_width, quant_col_width

    while True:
        local_models = lms_models.list_local_llm_models()
        loaded_keys = lms_models.list_loaded_llm_model_keys()
        status_col_width = 10

        options = []

        table_header_item = MenuStaticItem(label="", description="")
        table_header_item.selectable = False

        def table_header_dynamic_label(_is_selected):
            """
            Genera el encabezado dinámico de la tabla.
            
            Args:
                _is_selected (bool): Indica si el encabezado está seleccionado.
            
            Returns:
                str: Encabezado dinámico de la tabla.
            """
            _table_width, model_col_width, quant_col_width = _table_layout()
            return (
                f"{Style.DIM}{'MODEL':<{model_col_width}} | "
                f"{'QUANT':<{quant_col_width}} | {'STATUS':<{status_col_width}}{Style.ENDC}"
            )

        table_header_item.dynamic_label = table_header_dynamic_label
        options.append(table_header_item)

        table_separator_item = MenuStaticItem(label="", description="")
        def colorized_table_separator(_is_selected):
            """
            Genera el separador dinámico de la tabla.
            
            Args:
                _is_selected (bool): Indica si el separador está seleccionado.
            
            Returns:
                str: Separador dinámico de la tabla.
            """
            table_width, _model_col_width, _quant_col_width = _table_layout()
            return f"{Style.DIM}{'─' * table_width}{Style.ENDC}"

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
                """
                Genera el encabezado dinámico de la tabla.
                
                Args:
                    is_selected_row (bool): Indica si la fila está seleccionada.
                    key (str): Clave del modelo.
                    quant (str): Cantidad del modelo.
                    loaded_set (set[str]): Conjunto de claves cargadas.
                    name_cell (str): Celda del nombre.
                
                Returns:
                    str: Encabezado dinámico de la tabla.
                """
                _table_width, model_col_width, quant_col_width = _table_layout()
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

        separator_item = MenuStaticItem(label="", description="")
        separator_item.selectable = False

        def separator_dynamic_label(_is_selected):
            """
            Genera el separador dinámico de la tabla.
            
            Args:
                _is_selected (bool): Indica si el separador está seleccionado.
            
            Returns:
                str: Separador dinámico de la tabla.
            """
            table_width, _model_col_width, _quant_col_width = _table_layout()
            return f"{Style.DIM}{'─' * table_width}{Style.ENDC}"

        separator_item.dynamic_label = separator_dynamic_label

        options.append(separator_item)

        options.append(MenuItem("Download from MODELS_REGISTRY (quantizations)...", _download_registry_model_ui, description="Descarga variantes por cuantización de modelos del registro."))
        options.append(MenuItem("Download by model ID/URL (quantizations)...", _download_custom_model_ui, description="Descarga variantes por cuantización usando id/url manual."))
        options.append(MenuItem("Back", lambda: "BACK", description="Vuelve al menú de Tests & Models."))

        choice = interactive_menu(
            options,
            header_func=_menu_header,
            menu_id="manage_models_menu",
            info_text=lambda: _format_download_info_text(include_running_lines=True),
            nav_hint=True,
            nav_hint_text="↑/↓ navegar · ENTER seleccionar · ESC volver",
            left_margin=0,
            dynamic_info_top=True,
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
