import json
from pathlib import Path

import pytest

from src.scripts.experiment_ab_text import build_execution_plans


def _touch_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")


def _record(*, model_id: str, include_reasoning: bool, schema_name: str, image_path: Path, predicted: str) -> dict:
    return {
        "model_id": model_id,
        "schema_name": schema_name,
        "include_reasoning": include_reasoning,
        "image_path": str(image_path),
        "status": "ok",
        "ground_truth_cls": "AD",
        "payload": {"predicted_class": predicted},
    }


def test_build_execution_plans_detects_variants_and_balances_samples(tmp_path: Path) -> None:
    results_file = tmp_path / "batch_results.jsonl"

    records = []
    for model_id, include_reasoning in (("model-a@q8", False), ("model-b@q8", True)):
        for idx in range(6):
            img = tmp_path / f"{model_id.replace('@', '_')}_ok_{idx}.tif"
            _touch_image(img)
            records.append(
                _record(
                    model_id=model_id,
                    include_reasoning=include_reasoning,
                    schema_name="PolypClassification",
                    image_path=img,
                    predicted="AD",
                )
            )
        for idx in range(6):
            img = tmp_path / f"{model_id.replace('@', '_')}_fail_{idx}.tif"
            _touch_image(img)
            records.append(
                _record(
                    model_id=model_id,
                    include_reasoning=include_reasoning,
                    schema_name="PolypClassification",
                    image_path=img,
                    predicted="HP",
                )
            )

    with results_file.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"__batch_meta__": {"model_ids": ["model-a@q8", "model-b@q8"]}}) + "\n")
        for row in records:
            handle.write(json.dumps(row) + "\n")

    plans = build_execution_plans(
        results_file=results_file,
        n_correct=5,
        n_incorrect=5,
        seed=42,
    )

    assert len(plans) == 2
    assert {plan.variant.model_id for plan in plans} == {"model-a@q8", "model-b@q8"}

    for plan in plans:
        assert len(plan.samples) == 10
        assert sum(1 for sample in plan.samples if sample.previous_status == "Acierto") == 5
        assert sum(1 for sample in plan.samples if sample.previous_status == "Fallo") == 5


def test_build_execution_plans_filters_by_override_model(tmp_path: Path) -> None:
    results_file = tmp_path / "batch_results.jsonl"

    img_ok = tmp_path / "ok.tif"
    img_fail = tmp_path / "fail.tif"
    _touch_image(img_ok)
    _touch_image(img_fail)

    records = [
        _record(
            model_id="model-a@q8",
            include_reasoning=False,
            schema_name="PolypClassification",
            image_path=img_ok,
            predicted="AD",
        ),
        _record(
            model_id="model-a@q8",
            include_reasoning=False,
            schema_name="PolypClassification",
            image_path=img_fail,
            predicted="HP",
        ),
    ]

    with results_file.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row) + "\n")

    with pytest.raises(RuntimeError):
        build_execution_plans(
            results_file=results_file,
            n_correct=5,
            n_incorrect=5,
            seed=7,
            override_model="model-b@q8",
        )
