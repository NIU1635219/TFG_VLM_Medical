from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.preprocessing.prepare_dual_validation import DualValidationConfig, run_pipeline


def _write_results_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _make_image(path: Path, *, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[:] = color
    assert cv2.imwrite(str(path), image)


def _payload(*, diagnosis: str, text: str) -> dict[str, str]:
    return {
        "final_diagnosis_class": diagnosis,
        "clinical_justification": text,
    }


def test_dual_validation_exports_shared_cases_for_both_bundles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = Path.cwd()
    monkeypatch.chdir(project_root)

    images_dir = tmp_path / "images"
    for index, color in enumerate([(20, 20, 20), (40, 40, 40), (60, 60, 60), (80, 80, 80)], start=1):
        _make_image(images_dir / f"img_{index}.png", color=color)

    qwen_run_a = tmp_path / "qwen_a"
    qwen_run_f = tmp_path / "qwen_f"
    gemma_run_a = tmp_path / "gemma_a"
    gemma_run_f = tmp_path / "gemma_f"
    common_bbox = [100, 100, 700, 700]

    image_rows = [
        ("img_1", "AD", 0.98, 0.95),
        ("img_2", "AD", 0.94, 0.91),
        ("img_3", "AD", 0.90, 0.87),
        ("img_4", "AD", 0.86, 0.83),
    ]

    _write_results_jsonl(
        qwen_run_f / "results.jsonl",
        [
            {
                "image_id": image_id,
                "image_name": f"{image_id}.png",
                "image_path": str(images_dir / f"{image_id}.png"),
                "status": "ok",
                "model_id": "qwen3_5-9b@q8_0",
                "ground_truth_bbox": common_bbox,
                "ground_truth_cls": gt_class,
                "iou_score": iou_score,
                "proximity_score": proximity_score,
                "payload": _payload(diagnosis=gt_class, text=f"Qwen F {image_id}"),
            }
            for image_id, gt_class, iou_score, proximity_score in image_rows
        ],
    )
    _write_results_jsonl(
        qwen_run_a / "results.jsonl",
        [
            {
                "image_id": image_id,
                "image_name": f"{image_id}.png",
                "image_path": str(images_dir / f"{image_id}.png"),
                "status": "ok",
                "model_id": "qwen3_5-9b@q8_0",
                "ground_truth_bbox": common_bbox,
                "ground_truth_cls": gt_class,
                "payload": _payload(diagnosis=("HP" if image_id == "img_2" else gt_class), text=f"Qwen A {image_id}"),
            }
            for image_id, gt_class, _, _ in image_rows
        ],
    )
    _write_results_jsonl(
        gemma_run_f / "results.jsonl",
        [
            {
                "image_id": image_id,
                "image_name": f"{image_id}.png",
                "image_path": str(images_dir / f"{image_id}.png"),
                "status": "ok",
                "model_id": "google/gemma-4-e4b@q8_0",
                "ground_truth_bbox": common_bbox,
                "ground_truth_cls": gt_class,
                "iou_score": iou_score - 0.01,
                "proximity_score": proximity_score - 0.01,
                "payload": _payload(diagnosis=gt_class, text=f"Gemma F {image_id}"),
            }
            for image_id, gt_class, iou_score, proximity_score in image_rows
        ],
    )
    _write_results_jsonl(
        gemma_run_a / "results.jsonl",
        [
            {
                "image_id": image_id,
                "image_name": f"{image_id}.png",
                "image_path": str(images_dir / f"{image_id}.png"),
                "status": "ok",
                "model_id": "google/gemma-4-e4b@q8_0",
                "ground_truth_bbox": common_bbox,
                "ground_truth_cls": gt_class,
                "payload": _payload(diagnosis=gt_class, text=f"Gemma A {image_id}"),
            }
            for image_id, gt_class, _, _ in image_rows
        ],
    )

    config = DualValidationConfig(
        qwen_run_path_override_a=str(qwen_run_a),
        qwen_run_path_override_f=str(qwen_run_f),
        gemma_run_path_override_a=str(gemma_run_a),
        gemma_run_path_override_f=str(gemma_run_f),
        top_k=2,
        output_dist_dir=tmp_path / "dist",
        source_app_dir=project_root / "clinical_eval_app",
    )

    paths, selection, export_result = run_pipeline(config)

    assert paths.output_dist_dir == (tmp_path / "dist").resolve()
    assert selection.selected[0]["image_id"] == "img_1"
    assert selection.selected[1]["image_id"] == "img_2"
    assert export_result.selected_image_ids == ["img_1", "img_2"]

    cases_a = json.loads(export_result.output_app_a_cases_json.read_text(encoding="utf-8"))
    cases_f = json.loads(export_result.output_app_f_cases_json.read_text(encoding="utf-8"))

    assert len(cases_a) == 4
    assert len(cases_f) == 4
    assert {case["id_imagen"] for case in cases_a} == {"img_1", "img_2"}
    assert {case["id_imagen"] for case in cases_f} == {"img_1", "img_2"}
    assert {case["id_modelo_oculto"] for case in cases_a} == {"qwen3_5-9b@q8_0", "google/gemma-4-e4b@q8_0"}

    expected_root_a = export_result.output_app_a_cases_json.parent.relative_to(tmp_path).as_posix()
    expected_root_f = export_result.output_app_f_cases_json.parent.relative_to(tmp_path).as_posix()

    assert all(case["image_path"].startswith("images/") for case in cases_a)
    assert all(case["image_path"].startswith("images/") for case in cases_f)
    assert {case["dataset_root"] for case in cases_a} == {expected_root_a}
    assert {case["dataset_root"] for case in cases_f} == {expected_root_f}

    cases_a_by_key = {(case["id_imagen"], case["id_modelo_oculto"]): case for case in cases_a}
    cases_f_by_key = {(case["id_imagen"], case["id_modelo_oculto"]): case for case in cases_f}

    assert cases_a_by_key[("img_2", "qwen3_5-9b@q8_0")]["ai_predicted_class"] == "HP"
    assert cases_f_by_key[("img_2", "qwen3_5-9b@q8_0")]["ai_predicted_class"] == "AD"
    assert cases_f_by_key[("img_1", "google/gemma-4-e4b@q8_0")]["ai_predicted_class"] == "AD"

    with export_result.output_app_a_export_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["id_imagen", "id_modelo_oculto", "veredicto", "comentario_medico"]
        csv_rows_a = list(reader)
    with export_result.output_app_f_export_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["id_imagen", "id_modelo_oculto", "veredicto", "comentario_medico"]
        csv_rows_f = list(reader)

    assert len(csv_rows_a) == 4
    assert len(csv_rows_f) == 4

    image_names_a = sorted(item.name for item in export_result.output_app_a_images_dir.iterdir())
    image_names_f = sorted(item.name for item in export_result.output_app_f_images_dir.iterdir())
    assert image_names_a == image_names_f

    root_entries_a = sorted(item.name for item in export_result.output_app_a_cases_json.parent.iterdir())
    root_entries_f = sorted(item.name for item in export_result.output_app_f_cases_json.parent.iterdir())
    assert root_entries_a == ["cases.json", "cases_export_debug.csv", "images"]
    assert root_entries_f == ["cases.json", "cases_export_debug.csv", "images"]


def test_dual_validation_errors_when_selection_is_incomplete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = Path.cwd()
    monkeypatch.chdir(project_root)

    images_dir = tmp_path / "images"
    for index in range(1, 4):
        _make_image(images_dir / f"img_{index}.png", color=(index * 20, index * 20, index * 20))

    run_a = tmp_path / "run_a"
    run_f = tmp_path / "run_f"

    rows = [
        {
            "image_id": f"img_{index}",
            "image_name": f"img_{index}.png",
            "image_path": str(images_dir / f"img_{index}.png"),
            "status": "ok",
            "model_id": "qwen3_5-9b@q8_0",
            "ground_truth_bbox": [100, 100, 700, 700],
            "ground_truth_cls": "AD",
            "iou_score": 0.95 - index * 0.02,
            "proximity_score": 0.92 - index * 0.02,
            "payload": _payload(diagnosis="AD", text=f"Text {index}"),
        }
        for index in range(1, 4)
    ]

    _write_results_jsonl(run_a / "results.jsonl", rows)
    _write_results_jsonl(run_f / "results.jsonl", rows)

    config = DualValidationConfig(
        run_path_override_a=str(run_a),
        run_path_override_f=str(run_f),
        gemma_run_path_override_a=str(run_a),
        gemma_run_path_override_f=str(run_f),
        top_k=4,
        output_dist_dir=tmp_path / "dist",
        source_app_dir=project_root / "clinical_eval_app",
    )

    with pytest.raises(RuntimeError, match="requeridos=4"):
        run_pipeline(config)
