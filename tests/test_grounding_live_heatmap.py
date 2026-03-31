from src.utils.tests_ui.grounding_scenarios import (
    _build_live_confusion_heatmap_lines,
    _empty_live_confusion_counts,
    _heatmap_cell_style,
    _heatmap_rgb_from_percentage,
    _summarize_existing_scenario_records,
)
from types import SimpleNamespace


def _make_dummy_kit(width: int = 120):
    return SimpleNamespace(
        width=lambda: width,
        table_menu=None,
        TableColumn=None,
        TableRow=None,
    )


def test_build_live_confusion_heatmap_lines_renders_matrix() -> None:
    confusion = _empty_live_confusion_counts()
    confusion["AD"]["AD"] = 5
    confusion["AD"]["HP"] = 2
    confusion["HP"]["ASS"] = 1

    lines = _build_live_confusion_heatmap_lines(_make_dummy_kit(), confusion)

    assert lines[0] == "Heatmap GT→Pred (live):"
    assert any("GT AD" in line for line in lines)
    assert any("GT HP" in line for line in lines)
    assert any("GT ASS" in line for line in lines)
    assert any("Total" in line for line in lines)
    assert any("celda = conteo / %global / %filaGT · color = %global" in line for line in lines)
    assert any("%/" in line for line in lines)
    assert all("✓" not in line for line in lines)


def test_build_live_confusion_heatmap_lines_uses_global_percentage() -> None:
    confusion = _empty_live_confusion_counts()
    confusion["AD"]["AD"] = 5
    confusion["AD"]["HP"] = 5

    lines = _build_live_confusion_heatmap_lines(_make_dummy_kit(), confusion)

    # Total global=10 -> cada celda con valor 5 debe mostrar 50.0% global
    assert any("50.0%" in line for line in lines)
    assert any("100.0%" in line for line in lines)


def test_heatmap_rgb_gradient_transitions_to_red() -> None:
    low = _heatmap_rgb_from_percentage(0.0)
    high = _heatmap_rgb_from_percentage(100.0)

    assert low[0] >= high[0]
    assert high[1] < low[1]
    assert high[2] < low[2]


def test_heatmap_cell_style_returns_ansi_escape() -> None:
    style_token = _heatmap_cell_style(percentage=72.5)
    assert style_token.startswith("\033[")
    assert "48;2;" in style_token


def test_summarize_existing_records_builds_confusion_counts() -> None:
    records = [
        {
            "status": "ok",
            "ground_truth_cls": "AD",
            "payload": {"final_diagnosis_class": "AD"},
        },
        {
            "status": "ok",
            "ground_truth_cls": "AD",
            "payload": {"final_diagnosis_class": "HP"},
        },
        {
            "status": "ok",
            "ground_truth_cls": "HP",
            "payload": {"final_diagnosis_class": "ASS"},
        },
    ]

    summary = _summarize_existing_scenario_records(records)
    confusion = summary["confusion_counts"]

    assert confusion["AD"]["AD"] == 1
    assert confusion["AD"]["HP"] == 1
    assert confusion["HP"]["ASS"] == 1


def test_summarize_existing_records_ignores_pending_for_progress() -> None:
    records = [
        {
            "status": "pending",
            "ground_truth_cls": "AD",
            "payload": {"final_diagnosis_class": "HP"},
        },
        {
            "status": "ok",
            "ground_truth_cls": "AD",
            "payload": {"final_diagnosis_class": "AD"},
        },
    ]

    summary = _summarize_existing_scenario_records(records)

    assert summary["current"] == 1
    assert summary["ok"] == 1
    assert summary["confusion_counts"]["AD"]["AD"] == 1
    assert summary["confusion_counts"]["AD"]["HP"] == 0
