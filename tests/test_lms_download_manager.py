"""Tests para src/utils/lms_download_manager.py — DownloadJobState y utilidades."""

import time
import pytest
from unittest.mock import patch

from src.utils.lms_download_manager import DownloadJobState, _trim_text, FINISHED_STATUS_TTL_SECONDS


# ---------------------------------------------------------------------------
# Estilo dummy
# ---------------------------------------------------------------------------

class _Style:
    BOLD = ENDC = DIM = OKCYAN = OKGREEN = WARNING = FAIL = HEADER = SELECTED = ""


def _make_djs(**kw) -> DownloadJobState:
    """Crea un DownloadJobState aislado (sin cursor_memory compartido)."""
    return DownloadJobState(style=_Style, **kw)


# ---------------------------------------------------------------------------
# _trim_text
# ---------------------------------------------------------------------------

class TestTrimText:
    def test_short_text_unchanged(self):
        assert _trim_text("hello", 10) == "hello"

    def test_exact_length_unchanged(self):
        assert _trim_text("hello", 5) == "hello"

    def test_truncates_with_ellipsis(self):
        result = _trim_text("hello world", 8)
        assert len(result) == 8
        assert result.endswith("…")

    def test_max_len_zero_returns_empty(self):
        assert _trim_text("abc", 0) == ""

    def test_max_len_one_returns_single_char(self):
        result = _trim_text("abc", 1)
        assert len(result) == 1

    def test_none_coerced_to_empty(self):
        assert _trim_text(None, 10) == ""


# ---------------------------------------------------------------------------
# DownloadJobState.create_job / _new_job_id
# ---------------------------------------------------------------------------

class TestCreateJob:
    def test_returns_string_id(self):
        djs = _make_djs()
        jid = djs.create_job(model="test-model", quant="Q4_K_M")
        assert isinstance(jid, str)
        assert jid.startswith("job-")

    def test_consecutive_ids_are_unique(self):
        djs = _make_djs()
        ids = {djs.create_job(model="m", quant="Q4") for _ in range(5)}
        assert len(ids) == 5

    def test_job_has_running_status(self):
        djs = _make_djs()
        jid = djs.create_job(model="m", quant="Q4")
        snap = djs.snapshot()
        job = next(j for j in snap if j["job_id"] == jid)
        assert job["status"] == "running"

    def test_job_signature_format(self):
        djs = _make_djs()
        jid = djs.create_job(model="Org/Model", quant="Q4_K_M")
        snap = djs.snapshot()
        job = next(j for j in snap if j["job_id"] == jid)
        assert "org/model" in job["signature"]
        assert "Q4_K_M" in job["signature"]

    def test_job_persists_in_cursor_memory(self):
        memory = {}
        djs = DownloadJobState(style=_Style, cursor_memory=memory)
        djs.create_job(model="m", quant="Q4")
        assert "__lms_download_state__" in memory
        assert len(memory["__lms_download_state__"]["jobs"]) == 1


# ---------------------------------------------------------------------------
# DownloadJobState.update_job
# ---------------------------------------------------------------------------

class TestUpdateJob:
    def test_update_progress_fields(self):
        djs = _make_djs()
        jid = djs.create_job(model="m", quant="Q4")
        djs.update_job(jid, downloaded=512.0, total=1024.0, speed=100.0)
        snap = djs.snapshot()
        job = next(j for j in snap if j["job_id"] == jid)
        assert job["downloaded"] == 512.0
        assert job["total"] == 1024.0
        assert job["speed"] == 100.0

    def test_terminal_status_sets_finished_at(self):
        djs = _make_djs()
        jid = djs.create_job(model="m", quant="Q4")
        djs.update_job(jid, status="completed")
        snap = djs.snapshot()
        job = next(j for j in snap if j["job_id"] == jid)
        assert job["finished_at"] is not None

    def test_running_status_clears_finished_at(self):
        djs = _make_djs()
        jid = djs.create_job(model="m", quant="Q4")
        djs.update_job(jid, status="completed")
        djs.update_job(jid, status="running")  # reactivado
        snap = djs.snapshot()
        job = next(j for j in snap if j["job_id"] == jid)
        assert job["finished_at"] is None

    def test_update_nonexistent_job_is_noop(self):
        djs = _make_djs()
        djs.update_job("job-999", status="completed")  # no debe lanzar

    def test_failed_status_sets_finished_at(self):
        djs = _make_djs()
        jid = djs.create_job(model="m", quant="Q4")
        djs.update_job(jid, status="failed")
        snap = djs.snapshot()
        job = next(j for j in snap if j["job_id"] == jid)
        assert job["finished_at"] is not None


# ---------------------------------------------------------------------------
# DownloadJobState.snapshot / _prune_stale_jobs
# ---------------------------------------------------------------------------

class TestSnapshot:
    def test_empty_returns_empty_list(self):
        djs = _make_djs()
        assert djs.snapshot() == []

    def test_returns_copies_not_originals(self):
        djs = _make_djs()
        jid = djs.create_job(model="m", quant="Q4")
        snap = djs.snapshot()
        snap[0]["model"] = "MODIFIED"
        # El original no debe cambiar
        snap2 = djs.snapshot()
        assert snap2[0]["model"] == "m"

    def test_prunes_expired_jobs(self):
        djs = _make_djs()
        jid = djs.create_job(model="m", quant="Q4")
        # Marcar como completado y con finished_at en el pasado (TTL expirado)
        with djs._lock:
            djs._jobs[jid]["status"] = "completed"
            djs._jobs[jid]["finished_at"] = time.time() - (FINISHED_STATUS_TTL_SECONDS + 1)
        assert djs.snapshot() == []

    def test_does_not_prune_jobs_within_ttl(self):
        djs = _make_djs()
        jid = djs.create_job(model="m", quant="Q4")
        djs.update_job(jid, status="completed")
        assert len(djs.snapshot()) == 1  # aún dentro del TTL

    def test_ordering_by_created_at(self):
        djs = _make_djs()
        jid1 = djs.create_job(model="first", quant="Q4")
        jid2 = djs.create_job(model="second", quant="Q4")
        snap = djs.snapshot()
        assert snap[0]["job_id"] == jid1
        assert snap[1]["job_id"] == jid2


# ---------------------------------------------------------------------------
# DownloadJobState.running_signatures
# ---------------------------------------------------------------------------

class TestRunningSignatures:
    def test_empty_returns_empty_set(self):
        djs = _make_djs()
        assert djs.running_signatures() == set()

    def test_running_job_included(self):
        djs = _make_djs()
        djs.create_job(model="org/model", quant="Q4_K_M")
        sigs = djs.running_signatures()
        assert len(sigs) == 1
        assert "org/model|q4_k_m" in sigs or any("org/model" in s for s in sigs)

    def test_completed_job_excluded(self):
        djs = _make_djs()
        jid = djs.create_job(model="org/model", quant="Q4")
        djs.update_job(jid, status="completed")
        sigs = djs.running_signatures()
        assert len(sigs) == 0


# ---------------------------------------------------------------------------
# DownloadJobState.build_status_snapshot
# ---------------------------------------------------------------------------

class TestBuildStatusSnapshot:
    def test_empty_returns_empty_summary(self):
        djs = _make_djs()
        snap = djs.build_status_snapshot(ui_width=80)
        assert snap["summary"] == ""
        assert snap["running_lines"] == []
        assert snap["running_signatures"] == set()

    def test_running_job_generates_summary(self):
        djs = _make_djs()
        jid = djs.create_job(model="test-model", quant="Q4_K_M")
        djs.update_job(jid, downloaded=256.0, total=1024.0, speed=50.0)
        snap = djs.build_status_snapshot(ui_width=80)
        assert "activas" in snap["summary"] or "Descargas" in snap["summary"]
        assert len(snap["running_lines"]) > 0

    def test_completed_job_included_in_completed_ids(self):
        djs = _make_djs()
        jid = djs.create_job(model="m", quant="Q4")
        djs.update_job(jid, status="completed")
        snap = djs.build_status_snapshot(ui_width=80)
        assert jid in snap["completed_ids"]

    def test_snapshot_has_required_keys(self):
        djs = _make_djs()
        snap = djs.build_status_snapshot(ui_width=80)
        for key in ("summary", "running_lines", "signature", "running_signatures", "completed_ids"):
            assert key in snap, f"Falta clave '{key}'"


# ---------------------------------------------------------------------------
# DownloadJobState.format_info_text
# ---------------------------------------------------------------------------

class TestFormatInfoText:
    def test_empty_returns_empty_string(self):
        djs = _make_djs()
        assert djs.format_info_text(include_running_lines=True, ui_width=80) == ""

    def test_summary_only_when_no_running_lines_requested(self):
        djs = _make_djs()
        jid = djs.create_job(model="m", quant="Q4")
        djs.update_job(jid, downloaded=100.0, total=200.0)
        text = djs.format_info_text(include_running_lines=False, ui_width=80)
        # Debe contener el resumen pero no las barras
        assert text  # no vacío
        assert "\n" not in text or text.count("\n") == 0  # solo una línea de resumen

    def test_full_text_with_running_lines(self):
        djs = _make_djs()
        jid = djs.create_job(model="m", quant="Q4")
        djs.update_job(jid, downloaded=100.0, total=200.0)
        text = djs.format_info_text(include_running_lines=True, ui_width=80)
        assert text


# ---------------------------------------------------------------------------
# Persistencia entre instancias con cursor_memory compartida
# ---------------------------------------------------------------------------

class TestSharedMemoryPersistence:
    def test_state_survives_new_instance(self):
        memory = {}
        djs1 = DownloadJobState(style=_Style, cursor_memory=memory)
        jid = djs1.create_job(model="persistent-model", quant="Q8")

        djs2 = DownloadJobState(style=_Style, cursor_memory=memory)
        snap = djs2.snapshot()
        assert any(j["job_id"] == jid for j in snap)
