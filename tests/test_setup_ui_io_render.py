"""Tests para helpers de render compartido en setup_ui_io."""

from types import SimpleNamespace

import src.utils.setup_ui_io as setup_ui_io
from src.utils.tests_ui.test_dashboards_ui import _build_recent_record_lines
from src.utils.setup_ui_io import (
    compute_render_decision,
    paint_dynamic_lines,
    should_repaint_static,
)


def test_compute_render_decision_base_case():
    decision = compute_render_decision(
        dynamic_lines=["abc", "def"],
        terminal_width=80,
        prev_terminal_width=80,
        static_rendered=True,
        force_full=False,
    )

    assert decision["width_changed"] is False
    assert decision["has_wrapped_lines"] is False
    assert decision["current_dynamic_rows"] == 2
    assert decision["effective_force_full"] is False
    assert decision["use_incremental_frame"] is True


def test_compute_render_decision_detects_resize_and_wrap():
    decision = compute_render_decision(
        dynamic_lines=["\033[92m1234567890\033[0m"],
        terminal_width=5,
        prev_terminal_width=80,
        static_rendered=True,
        force_full=False,
    )

    assert decision["width_changed"] is True
    assert decision["has_wrapped_lines"] is True
    assert decision["current_dynamic_rows"] == 2
    assert decision["effective_force_full"] is True
    assert decision["use_incremental_frame"] is False


def test_compute_render_decision_forces_full_when_static_not_rendered():
    decision = compute_render_decision(
        dynamic_lines=["abc"],
        terminal_width=80,
        prev_terminal_width=80,
        static_rendered=False,
        force_full=False,
    )

    assert decision["effective_force_full"] is True


def test_should_repaint_static_true_when_incremental_disabled():
    assert (
        should_repaint_static(
            use_incremental=False,
            static_rendered=True,
        )
        is True
    )


def test_should_repaint_static_true_on_any_relevant_change():
    assert (
        should_repaint_static(
            use_incremental=True,
            static_rendered=True,
            width_changed=True,
        )
        is True
    )
    assert (
        should_repaint_static(
            use_incremental=True,
            static_rendered=True,
            height_changed=True,
        )
        is True
    )
    assert (
        should_repaint_static(
            use_incremental=True,
            static_rendered=True,
            signature_changed=True,
        )
        is True
    )


def test_should_repaint_static_false_in_stable_incremental_frame():
    assert (
        should_repaint_static(
            use_incremental=True,
            static_rendered=True,
            force_full=False,
            width_changed=False,
            height_changed=False,
            signature_changed=False,
            has_wrapped_lines=False,
        )
        is False
    )


def test_paint_dynamic_lines_non_incremental(capsys):
    result_rows = paint_dynamic_lines(
        dynamic_lines=["line1", "line2"],
        prev_dynamic_visual_rows=0,
        current_dynamic_rows=2,
        use_incremental_frame=False,
    )

    out = capsys.readouterr().out
    assert "\033[2Kline1" in out
    assert "\033[2Kline2" in out
    assert "\033[0F" not in out
    assert result_rows == 2


def test_paint_dynamic_lines_incremental_with_cleanup(capsys):
    result_rows = paint_dynamic_lines(
        dynamic_lines=["line1"],
        prev_dynamic_visual_rows=3,
        current_dynamic_rows=1,
        use_incremental_frame=True,
    )

    out = capsys.readouterr().out
    assert "\033[3F" in out
    assert out.count("\033[2K") >= 3
    assert result_rows == 1


def test_wait_for_any_key_discards_residual_enter_before_return(monkeypatch, capsys):
    fake_clock = {"value": 0.0}
    sleep_calls: list[float] = []

    class FakeMsvcrt:
        def __init__(self) -> None:
            self._pending_residual = True
            self._actual_key_ready = False
            self.getch_calls = 0

        def kbhit(self) -> bool:
            if self._pending_residual:
                return True
            if not self._actual_key_ready and fake_clock["value"] >= 0.16:
                self._actual_key_ready = True
            return self._actual_key_ready

        def getch(self) -> bytes:
            self.getch_calls += 1
            if self._pending_residual:
                self._pending_residual = False
                return b"\r"
            return b"x"

    fake_msvcrt = FakeMsvcrt()

    monkeypatch.setattr(setup_ui_io.time, "monotonic", lambda: fake_clock["value"])
    monkeypatch.setattr(
        setup_ui_io.time,
        "sleep",
        lambda seconds: sleep_calls.append(seconds) or fake_clock.__setitem__("value", fake_clock["value"] + seconds),
    )

    setup_ui_io.wait_for_any_key(
        message="Press any key to return...",
        style=SimpleNamespace(DIM="", ENDC=""),
        os_module=SimpleNamespace(name="nt"),
        msvcrt_module=fake_msvcrt,
        poll_interval_seconds=0.05,
    )

    out = capsys.readouterr().out
    assert "Press any key to return..." in out
    assert fake_msvcrt.getch_calls == 2
    assert sleep_calls
    assert fake_clock["value"] >= 0.16


def test_recent_record_separator_uses_frame_width():
    class _Kit:
        class style:
            DIM = ""
            ENDC = ""

        def width(self) -> int:
            return 120

    lines = _build_recent_record_lines(
        _Kit(),
        [{"image_name": "a.tif", "status": "ok", "payload": None}],
        truncate=False,
        ui_width=40,
    )

    assert any(line == "  " + "─" * 36 for line in lines)
