from __future__ import annotations

import json
from pathlib import Path

from src.scripts.grounding_experiments.runner_core import (
    build_annotated_comparison_filename,
    build_class_lookup_from_m_split_csvs,
    build_markdown_records_from_scenario_jsonl,
    collect_processed_image_ids_from_jsonl,
    generate_single_detection_markdown_report,
    has_unfilled_scenario_records,
    summarize_scenario_records_from_jsonl,
)


def test_build_class_lookup_normalizes_labels(tmp_path: Path) -> None:
    split_dir = tmp_path / "m_train"
    images_dir = split_dir / "images"
    images_dir.mkdir(parents=True)
    csv_path = split_dir / "train.csv"
    csv_path.write_text("image_id,cls\n991234,ad\n991235,Hp\n", encoding="utf-8")

    lookup = build_class_lookup_from_m_split_csvs(img_dir=images_dir)

    assert lookup["991234"] == "AD"
    assert lookup["991235"] == "HP"


def test_build_annotated_filename_uses_image_id_for_uniqueness() -> None:
    image_path = Path("data/processed/m_train/images/frame001.jpg")

    name_a = build_annotated_comparison_filename(image_id="17", image_path=image_path)
    name_b = build_annotated_comparison_filename(image_id="18", image_path=image_path)

    assert name_a == "frame001__17_comparison.jpg"
    assert name_b == "frame001__18_comparison.jpg"
    assert name_a != name_b


def test_markdown_records_keep_summary_and_visualization_aligned(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_1"
    annotated_dir = run_dir / "annotated"
    annotated_dir.mkdir(parents=True)
    output_path = run_dir / "results.jsonl"

    image_path = "data/processed/m_train/images/frame001.jpg"
    record_a = {
        "status": "ok",
        "image_id": "17",
        "image_path": image_path,
        "ground_truth_cls": "ad",
        "payload": {
            "final_diagnosis_class": "ass",
            "ymin": 100,
            "xmin": 110,
            "ymax": 220,
            "xmax": 330,
        },
        "ground_truth_bbox": [100, 110, 220, 330],
        "class_match": False,
        "iou_score": 0.2,
    }
    record_b = {
        "status": "ok",
        "image_id": "18",
        "image_path": image_path,
        "ground_truth_cls": "hp",
        "payload": {
            "final_diagnosis_class": "ad",
            "ymin": 120,
            "xmin": 140,
            "ymax": 250,
            "xmax": 360,
        },
        "ground_truth_bbox": [120, 140, 250, 360],
        "class_match": False,
        "iou_score": 0.3,
    }
    output_path.write_text(
        "\n".join(
            [
                json.dumps(record_a, ensure_ascii=False),
                json.dumps(record_b, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    ann_a = annotated_dir / build_annotated_comparison_filename(
        image_id=record_a["image_id"],
        image_path=Path(image_path),
    )
    ann_b = annotated_dir / build_annotated_comparison_filename(
        image_id=record_b["image_id"],
        image_path=Path(image_path),
    )
    ann_a.write_text("a", encoding="utf-8")
    ann_b.write_text("b", encoding="utf-8")

    records = build_markdown_records_from_scenario_jsonl(output_path=output_path, run_dir=run_dir)

    assert len(records) == 2
    assert records[0]["ground_truth_cls"] == "AD"
    assert records[0]["predicted_cls"] == "ASS"
    assert records[1]["ground_truth_cls"] == "HP"
    assert records[1]["predicted_cls"] == "AD"
    assert records[0]["annotated_path"] == str(ann_a)
    assert records[1]["annotated_path"] == str(ann_b)


def test_markdown_records_resolve_legacy_images_dir_for_visualization(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_legacy"
    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True)
    output_path = run_dir / "results.jsonl"

    image_path = "data/processed/m_train/images/frame001.jpg"
    record = {
        "status": "ok",
        "image_id": "21",
        "image_path": image_path,
        "ground_truth_cls": "ad",
        "payload": {
            "final_diagnosis_class": "ad",
            "ymin": 100,
            "xmin": 110,
            "ymax": 220,
            "xmax": 330,
        },
        "ground_truth_bbox": [100, 110, 220, 330],
        "class_match": True,
        "iou_score": 0.9,
    }
    output_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    ann_path = images_dir / build_annotated_comparison_filename(
        image_id=record["image_id"],
        image_path=Path(image_path),
    )
    ann_path.write_text("legacy", encoding="utf-8")

    records = build_markdown_records_from_scenario_jsonl(output_path=output_path, run_dir=run_dir)

    assert len(records) == 1
    assert records[0]["annotated_path"] == str(ann_path)


def test_summarize_scenario_records_from_jsonl_is_cumulative(tmp_path: Path) -> None:
    output_path = tmp_path / "results.jsonl"
    records = [
        {
            "status": "ok",
            "image_id": "1",
            "ground_truth_cls": "AD",
            "class_match": True,
            "iou_score": 0.8,
            "ttft_seconds": 1.0,
            "tokens_per_second": 12.0,
            "total_duration_seconds": 5.0,
            "payload": {"final_diagnosis_class": "AD", "ymin": 10, "xmin": 10, "ymax": 20, "xmax": 20},
            "ground_truth_bbox": [10, 10, 20, 20],
        },
        {
            "status": "ok",
            "image_id": "2",
            "ground_truth_cls": "HP",
            "class_match": False,
            "iou_score": 0.2,
            "ttft_seconds": 2.0,
            "tokens_per_second": 8.0,
            "total_duration_seconds": 7.0,
            "payload": {"final_diagnosis_class": "AD", "ymin": 15, "xmin": 15, "ymax": 30, "xmax": 30},
            "ground_truth_bbox": [15, 15, 30, 30],
        },
    ]
    output_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n",
        encoding="utf-8",
    )

    summary = summarize_scenario_records_from_jsonl(output_path=output_path)

    assert summary["total"] == 2
    assert summary["ok"] == 2
    assert summary["fail"] == 0
    assert summary["skip"] == 0
    assert summary["matched_class"] == 1
    assert summary["mismatched_class"] == 1
    assert summary["avg_iou"] == 0.5
    assert summary["avg_ttft_seconds"] == 1.5
    assert summary["avg_tokens_per_second"] == 10.0
    assert summary["avg_total_duration_seconds"] == 6.0


def test_generate_report_adds_global_accuracy_and_heatmap(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_report"
    run_dir.mkdir(parents=True)
    jsonl_path = run_dir / "results.jsonl"
    jsonl_path.write_text("", encoding="utf-8")

    records = [
        {
            "image_id": "img_1",
            "ground_truth_cls": "AD",
            "predicted_cls": "AD",
            "class_match": True,
            "iou": 0.7,
            "proximity": 0.85,
            "annotated_path": None,
        },
        {
            "image_id": "img_2",
            "ground_truth_cls": "HP",
            "predicted_cls": "AD",
            "class_match": False,
            "iou": 0.3,
            "proximity": 0.45,
            "annotated_path": None,
        },
    ]

    report_path = generate_single_detection_markdown_report(
        run_dir=run_dir,
        report_title="Scenario A Report",
        scenario_name="scenario_A",
        model_id="model_test",
        jsonl_path=jsonl_path,
        records=records,
    )

    markdown = report_path.read_text(encoding="utf-8")
    assert "- **Global Accuracy (class):** 50.00%" in markdown
    assert "- **Macro-F1 (AD/HP/ASS):**" in markdown
    assert "## Mapa de calor de confusión" in markdown
    assert "### Top de errores más frecuentes" in markdown
    assert "HP->AD: 1" in markdown
    assert "Recall por clase (AD/HP/ASS):" in markdown
    assert "### Resumen ejecutivo" in markdown
    assert "### Interpretación del caso" in markdown
    assert "### Lectura clase por clase" in markdown
    assert "Clase AD:" in markdown
    assert "Clase HP:" in markdown
    assert "## Análisis IoU" in markdown
    assert "report_assets/iou_distribution_histogram.png" in markdown
    assert "report_assets/iou_class_summary_grouped_bars.png" in markdown
    assert "report_assets/iou_correctness_comparison.png" in markdown
    assert "report_assets/iou_boxplot_class_correctness.png" in markdown
    assert "report_assets/iou_threshold_cumulative_curve.png" in markdown
    assert "## Análisis Proximity" in markdown
    assert "report_assets/proximity_distribution_histogram.png" in markdown
    assert "report_assets/proximity_class_summary_grouped_bars.png" in markdown
    assert "report_assets/proximity_correctness_comparison.png" in markdown
    assert "report_assets/proximity_boxplot_class_correctness.png" in markdown
    assert "report_assets/proximity_threshold_cumulative_curve.png" in markdown
    assert "report_assets/class_confusion_heatmap.png" in markdown
    assert (run_dir / "report_assets" / "class_confusion_heatmap.png").exists()
    assert (run_dir / "report_assets" / "iou_distribution_histogram.png").exists()
    assert (run_dir / "report_assets" / "iou_class_summary_grouped_bars.png").exists()
    assert (run_dir / "report_assets" / "iou_correctness_comparison.png").exists()
    assert (run_dir / "report_assets" / "iou_boxplot_class_correctness.png").exists()
    assert (run_dir / "report_assets" / "iou_threshold_cumulative_curve.png").exists()
    assert (run_dir / "report_assets" / "proximity_distribution_histogram.png").exists()
    assert (run_dir / "report_assets" / "proximity_class_summary_grouped_bars.png").exists()
    assert (run_dir / "report_assets" / "proximity_correctness_comparison.png").exists()
    assert (run_dir / "report_assets" / "proximity_boxplot_class_correctness.png").exists()
    assert (run_dir / "report_assets" / "proximity_threshold_cumulative_curve.png").exists()


def test_collect_processed_image_ids_excludes_error_records(tmp_path: Path) -> None:
    output_path = tmp_path / "results.jsonl"
    records = [
        {"status": "ok", "image_id": "img_01"},
        {"status": "skip", "image_id": "img_02"},
        {"status": "error", "image_id": "img_03"},
        {"status": "pending", "image_id": "img_04"},
    ]
    output_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n",
        encoding="utf-8",
    )

    processed = collect_processed_image_ids_from_jsonl(output_jsonl_path=output_path)

    assert processed == {"img_01", "img_02"}


def test_has_unfilled_scenario_records_treats_error_as_incomplete(tmp_path: Path) -> None:
    output_path = tmp_path / "results.jsonl"
    records = [
        {"status": "ok", "image_id": "img_01"},
        {"status": "error", "image_id": "img_02"},
    ]
    output_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n",
        encoding="utf-8",
    )

    assert has_unfilled_scenario_records(output_path=output_path) is True
