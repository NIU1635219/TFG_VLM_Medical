from pathlib import Path

from src.scripts.experiment_ab_text import ABInputSample, ABVariantPlan, ModelVariant
from src.utils.tests_ui.ab_experiment import run_ab_experiment_wrapper


class _DummyKit:
    def __init__(self) -> None:
        self.ask_calls: list[dict] = []

    def clear(self) -> None:
        return None

    def subtitle(self, _text: str) -> None:
        return None

    def log(self, _msg: str, _level: str = "info") -> None:
        return None

    def ask(self, question: str, default: str = "y", info_text: str | object = "") -> bool:
        self.ask_calls.append({"question": question, "default": default, "info_text": info_text})
        return False

    class TableColumn:
        def __init__(self, label: str, width_ratio: float | None = None, min_width: int = 8):
            self.label = label
            self.width_ratio = width_ratio
            self.min_width = min_width

    class TableRow:
        def __init__(self, cells: list[str]):
            self.cells = cells

    def table_menu(self, columns, rows, **_kwargs):
        headers = [getattr(column, "label", "") for column in columns]
        output = ["| " + " | ".join(headers) + " |"]
        for row in rows:
            output.append("| " + " | ".join(getattr(row, "cells", [])) + " |")
        return output

    def width(self) -> int:
        return 120

    def render_and_wait_responsive(self, **_kwargs) -> None:
        return None

    def wait(self, _message: str = "") -> None:
        return None


class _DummyApp:
    def print_banner(self) -> None:
        return None


def test_ab_ui_preview_applies_limiter(monkeypatch) -> None:
    samples = [
        ABInputSample(
            image_path=Path(f"/tmp/image_{idx}.tif"),
            image_name=f"image_{idx}.tif",
            ground_truth_cls="AD",
            predicted_cls="AD" if idx % 2 == 0 else "HP",
            previous_status="Acierto" if idx % 2 == 0 else "Fallo",
        )
        for idx in range(10)
    ]

    plans = [
        ABVariantPlan(
            variant=ModelVariant(
                model_id="qwen3_5-9b@q8_0",
                include_reasoning=True,
                schema_name="PolypClassification",
            ),
            samples=samples,
        )
    ]

    monkeypatch.setattr(
        "src.scripts.experiment_ab_text.detect_results_file",
        lambda _rf, _rd: Path("data/processed/batch_results/fake.jsonl"),
    )
    monkeypatch.setattr(
        "src.scripts.experiment_ab_text.build_execution_plans",
        lambda **_kwargs: plans,
    )

    kit = _DummyKit()
    app = _DummyApp()

    run_ab_experiment_wrapper(kit, app)

    assert len(kit.ask_calls) == 1
    info_payload = kit.ask_calls[0]["info_text"]
    info_text = info_payload(None) if callable(info_payload) else str(info_payload)
    assert "Variantes detectadas" in info_text
    assert "Campo" in info_text
    assert "Imagen" in info_text
    assert "image_0.tif" in info_text
    assert "image_7.tif" in info_text
    assert "image_8.tif" not in info_text
    assert "Nota: ... y 2 imagen(es) mas para esta variante." in info_text
