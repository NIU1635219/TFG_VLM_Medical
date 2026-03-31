from types import SimpleNamespace

from src.utils.tests_ui.test_dashboards_ui import _build_progress_bar


def _make_kit(width: int = 120):
    style = SimpleNamespace(DIM="", ENDC="", OKCYAN="", OKGREEN="", WARNING="", BOLD="")
    return SimpleNamespace(width=lambda: width, style=style)


def test_progress_bar_clamps_current_to_total() -> None:
    kit = _make_kit()
    bar = _build_progress_bar(kit, 101, 100)
    assert "100%" in bar
    assert "(100 / 100)" in bar


def test_progress_bar_shows_normal_values() -> None:
    kit = _make_kit()
    bar = _build_progress_bar(kit, 50, 100)
    assert "50%" in bar or "(50 / 100)" in bar


def test_progress_bar_no_samples_shows_placeholder() -> None:
    kit = _make_kit()
    bar = _build_progress_bar(kit, 0, 0)
    assert "sin muestra" in bar
