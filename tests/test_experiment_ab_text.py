import json
from pathlib import Path

import pytest

from src.scripts.experiment_ab_text import (
    ABInputSample,
    ABVariantPlan,
    ModelVariant,
    build_prompt_a,
    build_prompt_b,
    build_parser,
    build_execution_plans,
    build_resume_summary,
)


def _touch_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")


def _record(
    *,
    model_id: str,
    include_reasoning: bool,
    schema_name: str,
    image_path: Path,
    predicted: str,
    ground_truth: str,
) -> dict:
    return {
        "model_id": model_id,
        "schema_name": schema_name,
        "include_reasoning": include_reasoning,
        "image_path": str(image_path),
        "status": "ok",
        "ground_truth_cls": ground_truth,
        "payload": {"predicted_class": predicted},
    }


def test_build_execution_plans_detects_variants_and_balances_samples(tmp_path: Path) -> None:
    results_file = tmp_path / "batch_results.jsonl"

    records = []
    classes = ("AD", "HP", "ASS")
    for model_id, include_reasoning in (("model-a@q8", False), ("model-b@q8", True)):
        for cls in classes:
            for idx in range(2):
                img_ok = tmp_path / f"{model_id.replace('@', '_')}_{cls}_ok_{idx}.tif"
                _touch_image(img_ok)
                records.append(
                    _record(
                        model_id=model_id,
                        include_reasoning=include_reasoning,
                        schema_name="PolypClassification",
                        image_path=img_ok,
                        predicted=cls,
                        ground_truth=cls,
                    )
                )

                wrong = "AD" if cls != "AD" else "HP"
                img_fail = tmp_path / f"{model_id.replace('@', '_')}_{cls}_fail_{idx}.tif"
                _touch_image(img_fail)
                records.append(
                    _record(
                        model_id=model_id,
                        include_reasoning=include_reasoning,
                        schema_name="PolypClassification",
                        image_path=img_fail,
                        predicted=wrong,
                        ground_truth=cls,
                    )
                )

    with results_file.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"__batch_meta__": {"model_ids": ["model-a@q8", "model-b@q8"]}}) + "\n")
        for row in records:
            handle.write(json.dumps(row) + "\n")

    plans = build_execution_plans(
        results_file=results_file,
        n_correct=1,
        n_incorrect=1,
        seed=42,
    )

    assert len(plans) == 2
    assert {plan.variant.model_id for plan in plans} == {"model-a@q8", "model-b@q8"}

    for plan in plans:
        assert len(plan.samples) == 6
        for cls in classes:
            assert (
                sum(
                    1
                    for sample in plan.samples
                    if sample.ground_truth_cls == cls and sample.previous_status == "Acierto"
                )
                == 1
            )
            assert (
                sum(
                    1
                    for sample in plan.samples
                    if sample.ground_truth_cls == cls and sample.previous_status == "Fallo"
                )
                == 1
            )


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
            ground_truth="AD",
        ),
        _record(
            model_id="model-a@q8",
            include_reasoning=False,
            schema_name="PolypClassification",
            image_path=img_fail,
            predicted="HP",
            ground_truth="AD",
        ),
    ]

    with results_file.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row) + "\n")

    with pytest.raises(RuntimeError):
        build_execution_plans(
            results_file=results_file,
            n_correct=1,
            n_incorrect=1,
            seed=7,
            override_model="model-b@q8",
        )


def test_build_execution_plans_auto_adjusts_to_min_available_per_class(tmp_path: Path) -> None:
    results_file = tmp_path / "batch_results_min.jsonl"

    records = []
    model_id = "model-a@q8"
    include_reasoning = False

    # AD: 4 aciertos, 3 fallos -> cuota final 3
    for idx in range(4):
        img_ok = tmp_path / f"AD_ok_{idx}.tif"
        _touch_image(img_ok)
        records.append(
            _record(
                model_id=model_id,
                include_reasoning=include_reasoning,
                schema_name="PolypClassification",
                image_path=img_ok,
                predicted="AD",
                ground_truth="AD",
            )
        )
    for idx in range(3):
        img_fail = tmp_path / f"AD_fail_{idx}.tif"
        _touch_image(img_fail)
        records.append(
            _record(
                model_id=model_id,
                include_reasoning=include_reasoning,
                schema_name="PolypClassification",
                image_path=img_fail,
                predicted="HP",
                ground_truth="AD",
            )
        )

    # HP: 3 aciertos, 4 fallos -> cuota final 3
    for idx in range(3):
        img_ok = tmp_path / f"HP_ok_{idx}.tif"
        _touch_image(img_ok)
        records.append(
            _record(
                model_id=model_id,
                include_reasoning=include_reasoning,
                schema_name="PolypClassification",
                image_path=img_ok,
                predicted="HP",
                ground_truth="HP",
            )
        )
    for idx in range(4):
        img_fail = tmp_path / f"HP_fail_{idx}.tif"
        _touch_image(img_fail)
        records.append(
            _record(
                model_id=model_id,
                include_reasoning=include_reasoning,
                schema_name="PolypClassification",
                image_path=img_fail,
                predicted="AD",
                ground_truth="HP",
            )
        )

    # ASS: 2 aciertos, 2 fallos -> cuota final 2
    for idx in range(2):
        img_ok = tmp_path / f"ASS_ok_{idx}.tif"
        _touch_image(img_ok)
        records.append(
            _record(
                model_id=model_id,
                include_reasoning=include_reasoning,
                schema_name="PolypClassification",
                image_path=img_ok,
                predicted="ASS",
                ground_truth="ASS",
            )
        )
        img_fail = tmp_path / f"ASS_fail_{idx}.tif"
        _touch_image(img_fail)
        records.append(
            _record(
                model_id=model_id,
                include_reasoning=include_reasoning,
                schema_name="PolypClassification",
                image_path=img_fail,
                predicted="AD",
                ground_truth="ASS",
            )
        )

    with results_file.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row) + "\n")

    plans = build_execution_plans(
        results_file=results_file,
        n_correct=5,
        n_incorrect=5,
        seed=11,
    )

    assert len(plans) == 1
    samples = plans[0].samples
    assert len(samples) == (3 + 3) + (3 + 3) + (2 + 2)

    assert sum(1 for s in samples if s.ground_truth_cls == "AD" and s.previous_status == "Acierto") == 3
    assert sum(1 for s in samples if s.ground_truth_cls == "AD" and s.previous_status == "Fallo") == 3
    assert sum(1 for s in samples if s.ground_truth_cls == "HP" and s.previous_status == "Acierto") == 3
    assert sum(1 for s in samples if s.ground_truth_cls == "HP" and s.previous_status == "Fallo") == 3
    assert sum(1 for s in samples if s.ground_truth_cls == "ASS" and s.previous_status == "Acierto") == 2
    assert sum(1 for s in samples if s.ground_truth_cls == "ASS" and s.previous_status == "Fallo") == 2


def test_build_resume_summary_counts_reusable_pending_and_invalid(tmp_path: Path) -> None:
    img_ok = tmp_path / "ok.tif"
    img_bad = tmp_path / "bad.tif"
    _touch_image(img_ok)
    _touch_image(img_bad)

    variant = ModelVariant(
        model_id="model-a@q8",
        include_reasoning=False,
        schema_name="PolypClassification",
    )
    plan = ABVariantPlan(
        variant=variant,
        samples=[
            ABInputSample(
                image_path=img_ok,
                image_name=img_ok.name,
                ground_truth_cls="AD",
                predicted_cls="AD",
                previous_status="Acierto",
            ),
            ABInputSample(
                image_path=img_bad,
                image_name=img_bad.name,
                ground_truth_cls="AD",
                predicted_cls="HP",
                previous_status="Fallo",
            ),
        ],
    )

    checkpoint_file = tmp_path / "resume.checkpoint.json"
    variant_key = "model-a@q8::false::PolypClassification"
    payload = {
        "version": 2,
        "variants": {
            variant_key: {
                "variant": {
                    "model_id": "model-a@q8",
                    "include_reasoning": False,
                    "schema_name": "PolypClassification",
                },
                "items": {
                    str(img_ok.resolve()): {
                        "sample": {
                            "image_path": str(img_ok),
                            "image_name": img_ok.name,
                            "ground_truth_cls": "AD",
                            "predicted_cls": "AD",
                            "previous_status": "Acierto",
                        },
                        "description_a": {
                            "morphology_and_borders": "t",
                            "surface_and_vascular_pattern": "c",
                            "clinical_justification": "m",
                            "final_diagnosis": "AD",
                        },
                        "description_b": {
                            "morphology_and_borders": "t",
                            "surface_and_vascular_pattern": "c",
                            "clinical_justification": "m",
                            "final_diagnosis": "AD",
                        },
                    },
                    str(img_bad.resolve()): {
                        "sample": {
                            "image_path": str(img_bad),
                            "image_name": img_bad.name,
                            "ground_truth_cls": "AD",
                            "predicted_cls": "HP",
                            "previous_status": "Fallo",
                        },
                        "description_a": {
                            "morphology_and_borders": "",
                            "surface_and_vascular_pattern": "",
                            "clinical_justification": "ERROR: fallo",
                            "final_diagnosis": "",
                        },
                    },
                },
            }
        },
    }
    checkpoint_file.write_text(json.dumps(payload), encoding="utf-8")

    rows = build_resume_summary(plans=[plan], checkpoint_file=checkpoint_file)
    assert len(rows) == 1
    row = rows[0]
    assert row.reusable_samples == 1
    assert row.pending_samples == 1
    assert row.invalid_samples == 1
    assert row.total_samples == 2


def test_build_resume_summary_tolerates_corrupted_checkpoint_file(tmp_path: Path) -> None:
    img = tmp_path / "ok.tif"
    _touch_image(img)

    plan = ABVariantPlan(
        variant=ModelVariant(
            model_id="model-a@q8",
            include_reasoning=True,
            schema_name="PolypClassification",
        ),
        samples=[
            ABInputSample(
                image_path=img,
                image_name=img.name,
                ground_truth_cls="AD",
                predicted_cls="AD",
                previous_status="Acierto",
            )
        ],
    )

    checkpoint_file = tmp_path / "broken.checkpoint.json"
    checkpoint_file.write_text("{broken json", encoding="utf-8")

    rows = build_resume_summary(plans=[plan], checkpoint_file=checkpoint_file)
    assert len(rows) == 1
    row = rows[0]
    assert row.reusable_samples == 0
    assert row.pending_samples == 1
    assert row.invalid_samples == 0


def test_parser_supports_force_recompute_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["--force-recompute"])
    assert bool(args.force_recompute) is True


def test_prompt_b_injects_confirmed_class_label() -> None:
    prompt_ad = build_prompt_b("AD")
    prompt_hp = build_prompt_b("HP")
    prompt_ass = build_prompt_b("ASS")

    assert "Adenoma (AD)" in prompt_ad
    assert "Polipo Hiperplasico (HP)" in prompt_hp
    assert "Adenoma Serrado Sesil (ASS)" in prompt_ass


def test_prompt_a_is_zero_shot_neutral() -> None:
    prompt = build_prompt_a()
    assert "hecho irrefutable" not in prompt
    assert "AD, HP o ASS" in prompt


def test_build_resume_summary_ignores_legacy_checkpoint_version(tmp_path: Path) -> None:
    img = tmp_path / "legacy.tif"
    _touch_image(img)

    plan = ABVariantPlan(
        variant=ModelVariant(
            model_id="model-a@q8",
            include_reasoning=False,
            schema_name="PolypClassification",
        ),
        samples=[
            ABInputSample(
                image_path=img,
                image_name=img.name,
                ground_truth_cls="AD",
                predicted_cls="AD",
                previous_status="Acierto",
            )
        ],
    )

    checkpoint_file = tmp_path / "legacy.checkpoint.json"
    checkpoint_file.write_text(
        json.dumps(
            {
                "version": 1,
                "variants": {
                    "model-a@q8::false::PolypClassification": {
                        "variant": {
                            "model_id": "model-a@q8",
                            "include_reasoning": False,
                            "schema_name": "PolypClassification",
                        },
                        "items": {
                            str(img.resolve()): {
                                "sample": {
                                    "image_path": str(img),
                                    "image_name": img.name,
                                    "ground_truth_cls": "AD",
                                    "predicted_cls": "AD",
                                    "previous_status": "Acierto",
                                },
                                "description_a": {"texture": "x"},
                                "description_b": {"texture": "x"},
                            }
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    rows = build_resume_summary(plans=[plan], checkpoint_file=checkpoint_file)
    assert len(rows) == 1
    row = rows[0]
    assert row.reusable_samples == 0
    assert row.pending_samples == 1
