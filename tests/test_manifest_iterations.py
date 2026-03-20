import json
from pathlib import Path

import pytest

from src.utils.tests_ui.manifest import (
    _delete_manifest_file,
    extract_manifest_run_config,
    linked_batch_output_path,
    manifest_execution_snapshot,
    prune_output_records_for_model,
)
from src.utils.tests_ui.manifest_generation import generate_manifest


@pytest.fixture(autouse=True)
def _isolate_manifest_tests_fs(tmp_path, monkeypatch):
    """Aísla los tests para no generar artefactos en data/ del workspace."""
    monkeypatch.chdir(tmp_path)


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
        run_schema_name="PolypClassification",
        run_model_variants=[{"model_id": "model-a", "include_reasoning": False}],
        run_iterations_per_image=3,
    )

    lines = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    meta = lines[0]["__manifest_meta__"]
    rows = lines[1:]

    assert summary["run_iterations_per_image"] == 3
    assert summary["run_seed"] == 42
    assert summary["run_model_variants"] == [{"model_id": "model-a", "include_reasoning": False}]
    assert meta["run_iterations_per_image"] == 3
    assert meta["run_seed"] == 42
    assert meta["run_model_variants"] == [{"model_id": "model-a", "include_reasoning": False}]
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
                    "run_schema_name": "PolypClassification",
                    "run_model_variants": [
                        {"model_id": "model-a", "include_reasoning": False},
                        {"model_id": "model-a", "include_reasoning": True},
                        {"model_id": "model-b", "include_reasoning": True},
                    ],
                    "run_iterations_per_image": 5,
                    "run_seed": 77,
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
    assert config["schema_name"] == "PolypClassification"
    assert config["iterations_per_image"] == 5
    assert config["seed"] == 77
    assert config["model_variants"] == [
        {"model_id": "model-a", "include_reasoning": False},
        {"model_id": "model-a", "include_reasoning": True},
        {"model_id": "model-b", "include_reasoning": True},
    ]


def test_extract_manifest_run_config_requires_meta_header(tmp_path):
    manifest_path = tmp_path / "manifest_rows.jsonl"
    rows = [
        {
            "image_path": str(tmp_path / "a.tif"),
            "run_iteration_total": 2,
        },
        {
            "image_path": str(tmp_path / "b.tif"),
            "run_iteration_total": 3,
        },
        {
            "image_path": str(tmp_path / "c.tif"),
            "run_iteration_total": 1,
        },
    ]
    manifest_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    config = extract_manifest_run_config(str(manifest_path))

    assert config is None


def test_extract_manifest_run_config_rejects_incomplete_meta(tmp_path):
    manifest_path = tmp_path / "manifest_incomplete_meta.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "__manifest_meta__": {
                    "version": 1,
                    "run_schema_name": "PolypClassification",
                    "run_model_variants": [{"model_id": "model-a", "include_reasoning": False}],
                }
            }
        )
        + "\n"
        + json.dumps({"image_path": str(tmp_path / "x.tif")})
        + "\n",
        encoding="utf-8",
    )

    config = extract_manifest_run_config(str(manifest_path))

    assert config is None


def test_generate_manifest_accepts_explicit_model_variants(tmp_path):
    csv_path = tmp_path / "train.csv"
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    (images_dir / "img_1.tif").write_bytes(b"a")
    csv_path.write_text(
        "image_id,cls\n"
        "img_1,AD\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "manifest_variants.jsonl"
    summary = generate_manifest(
        input_csv=csv_path,
        images_dir=images_dir,
        output_path=output_path,
        sample_size=1,
        seed=123,
        stratify_col="cls",
        id_col="image_id",
        label_col="cls",
        relative_paths=False,
        run_schema_name="PolypClassification",
        run_model_variants=[
            {"model_id": "model-a", "include_reasoning": False},
            {"model_id": "model-a", "include_reasoning": True},
        ],
        run_iterations_per_image=1,
    )

    lines = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    meta = lines[0]["__manifest_meta__"]
    assert summary["run_model_variants"] == [
        {"model_id": "model-a", "include_reasoning": False},
        {"model_id": "model-a", "include_reasoning": True},
    ]
    assert meta["run_model_variants"] == summary["run_model_variants"]
    assert meta["run_seed"] == 123


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
        include_reasoning=False,
    )

    assert snapshot["total"] == 2
    assert snapshot["ok"] == 1
    assert snapshot["errors"] == 0
    assert snapshot["pending"] == 1
    assert snapshot["status"] == "yellow"
    assert int(snapshot["pending_entries"][0].get("run_iteration_index") or 0) == 2


def test_linked_batch_output_path_merges_reasoning_variants(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text("", encoding="utf-8")

    base_path = linked_batch_output_path(
        manifest_path=str(manifest_path),
        schema_name="PolypClassification",
    )
    reasoning_path = linked_batch_output_path(
        manifest_path=str(manifest_path),
        schema_name="PolypClassificationWithReasoning",
    )

    assert base_path == reasoning_path


def test_manifest_snapshot_and_prune_isolate_include_reasoning_variant(tmp_path):
    image_path = tmp_path / "img_1.tif"
    image_path.write_bytes(b"x")

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps({"image_path": str(image_path), "ground_truth_cls": "AD", "run_iteration_index": 1, "run_iteration_total": 1})
        + "\n",
        encoding="utf-8",
    )

    output_path = linked_batch_output_path(
        manifest_path=str(manifest_path),
        schema_name="PolypClassificationWithReasoning",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "__batch_meta__": {
                "version": 2,
                "created_at_utc": "2026-03-14T00:00:00+00:00",
                "updated_at_utc": "2026-03-14T00:00:00+00:00",
                "output_mode": "shared_jsonl",
                "model_ids": ["model-a"],
                "schema_names": ["PolypClassification", "PolypClassificationWithReasoning"],
                "input_sources": ["manifest"],
            }
        },
        {
            "model_id": "model-a",
            "schema_name": "PolypClassification",
            "include_reasoning": False,
            "image_path": str(image_path),
            "run_iteration_index": 1,
            "status": "ok",
            "payload": {"predicted_class": "AD"},
            "duration_seconds": 1.0,
        },
        {
            "model_id": "model-a",
            "schema_name": "PolypClassification",
            "include_reasoning": True,
            "image_path": str(image_path),
            "run_iteration_index": 1,
            "status": "ok",
            "payload": {"predicted_class": "AD", "reasoning": "x"},
            "duration_seconds": 1.2,
        },
    ]
    output_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    snapshot_base = manifest_execution_snapshot(
        manifest_path=str(manifest_path),
        model_tag="model-a",
        schema_name="PolypClassification",
        include_reasoning=False,
    )
    snapshot_reasoning = manifest_execution_snapshot(
        manifest_path=str(manifest_path),
        model_tag="model-a",
        schema_name="PolypClassification",
        include_reasoning=True,
    )

    assert snapshot_base["ok"] == 1
    assert snapshot_base["pending"] == 0
    assert snapshot_reasoning["ok"] == 1
    assert snapshot_reasoning["pending"] == 0

    assert prune_output_records_for_model(
        output_path=str(output_path),
        model_tag="model-a",
        schema_name="PolypClassification",
        include_reasoning=False,
    ) is True

    remaining = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload_rows = [row for row in remaining if "__batch_meta__" not in row]
    assert len(payload_rows) == 1
    assert payload_rows[0]["schema_name"] == "PolypClassification"
    assert payload_rows[0]["include_reasoning"] is True


def test_delete_manifest_file_removes_linked_shared_jsonl(tmp_path):
    experiments_dir = Path("data/experiments")
    experiments_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = experiments_dir / "manifest_delete_flow.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "__manifest_meta__": {
                    "version": 1,
                    "run_schema_name": "PolypClassification",
                    "run_model_variants": [{"model_id": "model-a", "include_reasoning": False}],
                    "run_iterations_per_image": 1,
                    "run_seed": 42,
                }
            }
        )
        + "\n"
        + json.dumps({"image_path": str(tmp_path / "img_1.tif")})
        + "\n",
        encoding="utf-8",
    )

    shared_output = linked_batch_output_path(
        manifest_path=str(manifest_path),
        schema_name="PolypClassification",
    )
    shared_output.parent.mkdir(parents=True, exist_ok=True)
    shared_output.write_text("{}\n", encoding="utf-8")

    deleted, error = _delete_manifest_file(str(manifest_path))

    assert deleted is True
    assert error is None
    assert not manifest_path.exists()
    assert not shared_output.exists()
