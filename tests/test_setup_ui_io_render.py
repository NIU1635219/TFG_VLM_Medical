"""Tests para helpers de render compartido en setup_ui_io."""

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
