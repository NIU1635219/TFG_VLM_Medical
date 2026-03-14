import json
from pathlib import Path

from src.utils.tests_ui.manifest import (
    extract_manifest_run_config,
    linked_batch_output_path,
    manifest_execution_snapshot,
)
from src.utils.tests_ui.manifest_generation import generate_manifest


def test_generate_manifest_repeats_rows_per_iteration(tmp_path):
    csv_path = tmp_path / "train.csv"
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    (images_dir / "img_1.tif").write_bytes(b"a")
    (images_dir / "img_2.tif").write_bytes(b"b")

    csv_path.write_text(
        "image_id,cls\n"
        "img_1,AD\n"
        "img_2,HP\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "manifest.jsonl"
    summary = generate_manifest(
        input_csv=csv_path,
        images_dir=images_dir,
        output_path=output_path,
        sample_size=2,
        seed=42,
        stratify_col="cls",
        id_col="image_id",
        label_col="cls",
        relative_paths=False,
        run_models=["model-a"],
        run_schema_name="PolypClassification",
        run_include_reasoning=False,
        run_iterations_per_image=3,
    )

    lines = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    meta = lines[0]["__manifest_meta__"]
    rows = lines[1:]

    assert summary["run_iterations_per_image"] == 3
    assert meta["run_iterations_per_image"] == 3
    assert len(rows) == 6
    assert sorted({int(row.get("run_iteration_index") or 0) for row in rows}) == [1, 2, 3]
    assert sorted({int(row.get("run_iteration_total") or 0) for row in rows}) == [3]


def test_extract_manifest_run_config_reads_iterations_from_meta(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "__manifest_meta__": {
                    "version": 1,
                    "run_models": ["model-a", "model-b"],
                    "run_schema_name": "PolypClassification",
                    "run_include_reasoning": True,
                    "run_iterations_per_image": 5,
                }
            }
        )
        + "\n"
        + json.dumps({"image_path": str(tmp_path / "x.tif")})
        + "\n",
        encoding="utf-8",
    )

    config = extract_manifest_run_config(str(manifest_path))

    assert config is not None
    assert config["models"] == ["model-a", "model-b"]
    assert config["schema_name"] == "PolypClassification"
    assert config["include_reasoning"] is True
    assert config["iterations_per_image"] == 5


def test_manifest_snapshot_supports_multiple_iterations_same_image(tmp_path):
    image_path = tmp_path / "img_1.tif"
    image_path.write_bytes(b"x")

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_rows = [
        {"image_path": str(image_path), "ground_truth_cls": "AD", "run_iteration_index": 1, "run_iteration_total": 2},
        {"image_path": str(image_path), "ground_truth_cls": "AD", "run_iteration_index": 2, "run_iteration_total": 2},
    ]
    manifest_path.write_text("\n".join(json.dumps(row) for row in manifest_rows) + "\n", encoding="utf-8")

    output_path = linked_batch_output_path(
        manifest_path=str(manifest_path),
        model_tag="model-a",
        schema_name="PolypClassification",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_rows = [
        {
            "__batch_meta__": {
                "version": 2,
                "created_at_utc": "2026-03-14T00:00:00+00:00",
                "updated_at_utc": "2026-03-14T00:00:00+00:00",
                "output_mode": "shared_jsonl",
                "model_ids": ["model-a"],
                "schema_names": ["PolypClassification"],
                "input_sources": ["manifest"],
            }
        },
        {
            "model_id": "model-a",
            "schema_name": "PolypClassification",
            "image_path": str(image_path),
            "run_iteration_index": 1,
            "status": "ok",
            "payload": {"predicted_class": "AD"},
            "duration_seconds": 1.0,
        },
    ]
    output_path.write_text("\n".join(json.dumps(row) for row in output_rows) + "\n", encoding="utf-8")

    snapshot = manifest_execution_snapshot(
        manifest_path=str(manifest_path),
        model_tag="model-a",
        schema_name="PolypClassification",
    )

    assert snapshot["total"] == 2
    assert snapshot["ok"] == 1
    assert snapshot["errors"] == 0
    assert snapshot["pending"] == 1
    assert snapshot["status"] == "yellow"
    assert int(snapshot["pending_entries"][0].get("run_iteration_index") or 0) == 2
