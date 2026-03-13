"""Máquina de estado para trabajos de descarga de modelos LM Studio.

Extrae la lógica de gestión de jobs que antes vivía como closures dentro de
``setup_models_ui.manage_models_menu_ui``.  La clase ``DownloadJobState``
encapsula el estado compartido y ofrece métodos de alto nivel al resto del
menú de modelos.
"""

from __future__ import annotations

import threading
import time
from typing import Any


FINISHED_STATUS_TTL_SECONDS = 5.0


# ---------------------------------------------------------------------------
# Utilidades de texto
# ---------------------------------------------------------------------------


def _trim_text(text: str, max_len: int) -> str:
    """Trunca ``text`` a ``max_len`` caracteres, añadiendo '…' si es necesario.

    Args:
        text (str): Texto a truncar.
        max_len (int): Longitud máxima.

    Returns:
        str: Texto posiblemente truncado.
    """
    raw = str(text or "")
    if len(raw) <= max_len:
        return raw
    if max_len <= 1:
        return raw[:max_len]
    return raw[: max_len - 1] + "…"


# ---------------------------------------------------------------------------
# DownloadJobState
# ---------------------------------------------------------------------------


class DownloadJobState:
    """Gestiona el ciclo de vida de los trabajos de descarga de modelos.

    El estado se persiste opcionalmente en un dict externo (``cursor_memory``)
    para sobrevivir entre llamadas al menú.  Si ``cursor_memory`` es ``None``
    el estado es local a la instancia.

    Args:
        style: Objeto con constantes ANSI (BOLD, OKCYAN, OKGREEN, FAIL, etc.).
        cursor_memory (dict | None): Dict compartido donde almacenar el estado
            bajo la clave ``"__lms_download_state__"``.  Si es ``None`` se crea
            un dict interno privado.
    """

    _STATE_KEY = "__lms_download_state__"

    def __init__(self, *, style: Any, cursor_memory: dict | None = None) -> None:
        """Inicializa el estado compartido de descargas.

        Args:
            style: Objeto de estilo ANSI usado para renderizar estado.
            cursor_memory: Diccionario compartido para persistir jobs entre menús.
        """
        self.style = style

        # Localizar o crear el bloque de estado persistente
        store = cursor_memory if cursor_memory is not None else {}
        state = store.get(self._STATE_KEY)
        if not isinstance(state, dict):
            state = {"lock": threading.Lock(), "jobs": {}, "counter": 0}
            store[self._STATE_KEY] = state

        # Asegurar que el lock existe (puede haberse serializado sin él)
        if not isinstance(state.get("lock"), type(threading.Lock())):
            state["lock"] = threading.Lock()
        if not isinstance(state.get("jobs"), dict):
            state["jobs"] = {}

        self._state = state
        self._lock: threading.Lock = state["lock"]
        self._jobs: dict[str, dict[str, Any]] = state["jobs"]

    # ── Generación de IDs ───────────────────────────────────────────────

    def _new_job_id(self) -> str:
        """Genera un nuevo ID único para un trabajo de descarga."""
        with self._lock:
            counter = int(self._state.get("counter", 0)) + 1
            self._state["counter"] = counter
            return f"job-{counter}"

    # ── CRUD de trabajos ────────────────────────────────────────────────

    def create_job(self, *, model: str, quant: str) -> str:
        """Registra un nuevo trabajo de descarga en estado ``running``.

        Args:
            model (str): Identificador o URL del modelo.
            quant (str): Etiqueta de cuantización.

        Returns:
            str: ID del nuevo trabajo.
        """
        job_id = self._new_job_id()
        now = time.time()
        with self._lock:
            self._jobs[job_id] = {
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

    def update_job(self, job_id: str, **updates: Any) -> None:
        """Actualiza campos de un trabajo existente.

        Marca automáticamente ``finished_at`` cuando el estado es terminal
        (``completed``, ``failed`` o ``cancelled``).

        Args:
            job_id (str): ID del trabajo a actualizar.
            **updates: Campos a sobreescribir.
        """
        with self._lock:
            if job_id not in self._jobs:
                return
            self._jobs[job_id].update(updates)
            terminal = ("completed", "failed", "cancelled")
            if str(self._jobs[job_id].get("status") or "") in terminal:
                if self._jobs[job_id].get("finished_at") is None:
                    self._jobs[job_id]["finished_at"] = time.time()
            else:
                self._jobs[job_id]["finished_at"] = None
            self._jobs[job_id]["updated_at"] = time.time()

    def _prune_stale_jobs(self, now_ts: float) -> None:
        """Elimina trabajos terminados cuyo TTL ha expirado (llamar con lock)."""
        stale = [
            jid for jid, job in self._jobs.items()
            if str(job.get("status") or "") in ("completed", "failed", "cancelled")
            and isinstance(job.get("finished_at"), (int, float))
            and (now_ts - float(job["finished_at"])) >= FINISHED_STATUS_TTL_SECONDS
        ]
        for jid in stale:
            self._jobs.pop(jid, None)

    def snapshot(self) -> list[dict[str, Any]]:
        """Devuelve una copia de los trabajos activos ordenada por fecha de creación.

        Los trabajos terminados cuyo TTL ha expirado se eliminan antes de la copia.

        Returns:
            list[dict]: Lista de trabajos, cada uno como dict independiente.
        """
        with self._lock:
            self._prune_stale_jobs(time.time())
            ordered = sorted(self._jobs.items(), key=lambda kv: float(kv[1].get("created_at") or 0.0))
            return [{"job_id": jid, **dict(data)} for jid, data in ordered]

    # ── Firmas ─────────────────────────────────────────────────────────

    def running_signatures(self) -> set[str]:
        """Devuelve el conjunto de firmas de trabajos activos."""
        jobs = self.snapshot()
        return {
            str(job.get("signature") or "").strip().lower()
            for job in jobs
            if str(job.get("status") or "") == "running"
            and str(job.get("signature") or "").strip()
        }

    # ── Renderización de estado ─────────────────────────────────────────

    def build_status_snapshot(self, *, ui_width: int) -> dict[str, Any]:
        """Construye un snapshot listo para renderizar en el menú.

        Args:
            ui_width (int): Ancho de UI para calcular barras de progreso.

        Returns:
            dict: Claves ``summary``, ``running_lines``, ``signature``,
                  ``running_signatures``, ``completed_ids``.
        """
        Style = self.style
        jobs = self.snapshot()
        empty: dict[str, Any] = {
            "summary": "",
            "running_lines": [],
            "signature": "",
            "running_signatures": set(),
            "completed_ids": set(),
        }
        if not jobs:
            return empty

        running = [j for j in jobs if str(j.get("status")) == "running"]
        done = [j for j in jobs if str(j.get("status")) in ("completed", "failed", "cancelled")]

        now_ts = time.time()
        recent_done = [
            j for j in done
            if isinstance(j.get("finished_at"), (int, float))
            and (now_ts - float(j["finished_at"])) < FINISHED_STATUS_TTL_SECONDS
        ]

        completed_ids = {
            str(j.get("job_id"))
            for j in jobs
            if str(j.get("status") or "") == "completed"
        }
        running_sigs = {
            str(j.get("signature") or "").strip().lower()
            for j in running
            if str(j.get("signature") or "").strip()
        }

        if not running and not recent_done:
            return {**empty, "running_signatures": running_sigs, "completed_ids": completed_ids}

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
            pct = max(0, min(int((downloaded / total) * 100.0) if total > 0 else 0, 100))
            filled = max(0, min(int((pct / 100.0) * bar_width), bar_width))
            bar = f"{Style.OKCYAN}{'█' * filled}{Style.DIM}{'░' * (bar_width - filled)}{Style.ENDC}"
            total_mb = total / (1024 * 1024) if total > 0 else 0.0
            downloaded_mb = downloaded / (1024 * 1024)
            speed_mb = speed / (1024 * 1024) if speed > 0 else 0.0

            running_lines.append(f"{Style.OKCYAN}↓ {model} [{quant}]{Style.ENDC}")
            running_lines.append(
                f"  {bar} {Style.BOLD}{pct:>3d}%{Style.ENDC}"
                f" · {downloaded_mb:.1f}/{total_mb:.1f} MB"
                f" · {speed_mb:.1f} MB/s"
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
            "running_signatures": running_sigs,
            "completed_ids": completed_ids,
        }

    def format_info_text(self, *, include_running_lines: bool, ui_width: int) -> str:
        """Formatea el texto de estado para mostrarlo en ``info_text``.

        Args:
            include_running_lines (bool): Si ``True`` incluye las barras de progreso.
            ui_width (int): Ancho de UI para el snapshot interno.

        Returns:
            str: Texto listo para ``info_text`` en el menú, o ``""`` si no hay nada.
        """
        snap = self.build_status_snapshot(ui_width=ui_width)
        summary = str(snap.get("summary") or "")
        if not summary:
            return ""
        if not include_running_lines:
            return summary
        lines = list(snap.get("running_lines") or [])
        return "\n".join([summary, *lines]) if lines else summary
