import json
from pathlib import Path

import pytest

from src.scripts.batch_runner import (
    BatchInputItem,
    build_record,
    build_parser,
    iter_image_paths,
    iter_manifest_items,
    run_batch_job,
    upsert_batch_execution_summary,
)


def _read_jsonl_records(path: Path) -> list[dict[str, object]]:
    """Carga un JSONL ignorando cabeceras de metadatos globales."""
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    raw = [json.loads(line) for line in lines]
    return [row for row in raw if "__batch_meta__" not in row]


def test_iter_image_paths_filters_supported_extensions(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    nested_dir = image_dir / "nested"
    nested_dir.mkdir()

    (image_dir / "a.jpg").write_bytes(b"a")
    (image_dir / "b.png").write_bytes(b"b")
    (nested_dir / "c.webp").write_bytes(b"c")
    (nested_dir / "ignore.txt").write_text("x", encoding="utf-8")

    results = iter_image_paths(image_dir)

    assert [path.name for path in results] == ["a.jpg", "b.png", "c.webp"]


def test_build_parser_accepts_required_batch_args():
    parser = build_parser()
    args = parser.parse_args([
        "--model",
        "fake-model",
        "--image-dir",
        "data/raw/m_test/images",
        "--schema",
        "GenericObjectDetection",
    ])

    assert args.model == "fake-model"
    assert args.image_dir == "data/raw/m_test/images"
    assert args.manifest is None
    assert args.schema == "GenericObjectDetection"


def test_build_parser_accepts_manifest_as_input_source(tmp_path):
    parser = build_parser()
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text('{"image_path": "a.jpg"}\n', encoding="utf-8")

    args = parser.parse_args([
        "--model",
        "fake-model",
        "--manifest",
        str(manifest_path),
        "--schema",
        "GenericObjectDetection",
    ])

    assert args.model == "fake-model"
    assert args.image_dir is None
    assert args.manifest == str(manifest_path)


def test_build_parser_rejects_missing_or_duplicate_input_source(tmp_path):
    parser = build_parser()
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text('{"image_path": "a.jpg"}\n', encoding="utf-8")

    with pytest.raises(SystemExit):
        parser.parse_args([
            "--model",
            "fake-model",
            "--schema",
            "GenericObjectDetection",
        ])

    with pytest.raises(SystemExit):
        parser.parse_args([
            "--model",
            "fake-model",
            "--image-dir",
            "data/raw/m_test/images",
            "--manifest",
            str(manifest_path),
            "--schema",
            "GenericObjectDetection",
        ])


def test_iter_manifest_items_reads_valid_rows_and_discards_invalid(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    image_ok = images_dir / "sample_1.tif"
    image_ok.write_bytes(b"ok")

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                json.dumps({
                    "image_path": str(image_ok),
                    "ground_truth_cls": "AD",
                    "image_id": "sample_1",
                }),
                "not-a-json",
                json.dumps({"ground_truth_cls": "HP"}),
                json.dumps({"image_path": str(tmp_path / 'missing.tif')}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    items, discarded = iter_manifest_items(manifest_path)

    assert discarded == 3
    assert items == [
        BatchInputItem(
            image_path=image_ok.resolve(),
            metadata={"ground_truth_cls": "AD", "image_id": "sample_1"},
        )
    ]


def test_run_batch_job_writes_incremental_jsonl_and_unloads(monkeypatch, tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for index in range(5):
        (image_dir / f"sample_{index}.jpg").write_bytes(b"image")

    output_path = tmp_path / "results.jsonl"
    calls = {"load": 0, "unload": 0, "inference": 0}

    class FakeResult:
        def __init__(self, image_name):
            self._image_name = image_name

        def model_dump(self):
            return {
                "object_detected": self._image_name,
                "confidence_score": 90,
                "justification": "ok",
            }

    class FakeTelemetry:
        def __init__(self):
            self.total_duration_seconds = 1.2
            self.ttft_seconds = 0.4
            self.generation_duration_seconds = 0.8
            self.prompt_tokens_per_second = 120.0
            self.tokens_per_second = 22.5
            self.reasoning_tokens = 4
            self.stop_reason = "eos"
            self.model_id = "fake-model"
            self.vram_usage_mb = 100.0
            self.total_params = 9000000000
            self.architecture = "qwen3_vl"
            self.quantization = "Q8_0"
            self.gpu_layers = 28
            self.prompt_tokens = 32
            self.completion_tokens = 18
            self.total_tokens = 50
            self.stats = {"tokens_per_second": 22.5}
            self.resources = {"vram_usage_mb": 100.0}

    class FakeInferenceResult:
        def __init__(self, image_name):
            self.data = FakeResult(image_name)
            self.telemetry = FakeTelemetry()

    class FakeLoader:
        def __init__(self, model_path, verbose=False, server_api_host=None, api_token=None):
            self.model_path = model_path

        def load_model(self):
            calls["load"] += 1

        def inference(self, image_path, prompt, schema, temperature, include_telemetry=False):
            calls["inference"] += 1
            assert include_telemetry is True
            return FakeInferenceResult(image_path.name)

        def unload_model(self):
            calls["unload"] += 1

    monkeypatch.setattr("src.scripts.batch_runner.VLMLoader", FakeLoader)
    monkeypatch.setattr("src.scripts.batch_runner.tqdm", lambda iterable, **kwargs: iterable)

    summary = run_batch_job(
        model_id="fake-model",
        image_dir=image_dir,
        schema_name="GenericObjectDetection",
        output_path=output_path,
        max_images=5,
    )

    assert summary["processed"] == 5
    assert summary["ok"] == 5
    assert summary["invalid"] == 0
    assert summary["fail"] == 0
    assert calls == {"load": 1, "unload": 1, "inference": 5}

    rows = _read_jsonl_records(output_path)

    assert len(rows) == 5
    assert rows[0]["status"] == "ok"
    assert rows[0]["payload"]["object_detected"].startswith("sample_")
    assert rows[0]["total_duration_seconds"] == 1.2
    assert rows[0]["ttft_seconds"] == 0.4
    assert rows[0]["tokens_per_second"] == 22.5
    assert rows[0]["reasoning_tokens"] == 4
    assert rows[0]["architecture"] == "qwen3_vl"


def test_run_batch_job_with_manifest_propagates_metadata(monkeypatch, tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    first_image = image_dir / "sample_0.tif"
    second_image = image_dir / "sample_1.tif"
    first_image.write_bytes(b"image")
    second_image.write_bytes(b"image")

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "image_path": str(first_image),
                        "ground_truth_cls": "AD",
                        "image_id": "0",
                    }
                ),
                json.dumps(
                    {
                        "image_path": str(second_image),
                        "ground_truth_cls": "HP",
                        "image_id": "1",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "results.jsonl"

    class FakeResult:
        def __init__(self, image_name):
            self._image_name = image_name

        def model_dump(self):
            return {
                "object_detected": self._image_name,
                "confidence_score": 93,
                "justification": "ok",
            }

    class FakeTelemetry:
        def __init__(self):
            self.total_duration_seconds = 1.2
            self.ttft_seconds = 0.4
            self.generation_duration_seconds = 0.8
            self.tokens_per_second = 21.0
            self.reasoning_tokens = 3
            self.stop_reason = "eos"
            self.model_id = "fake-model"
            self.architecture = "qwen3_vl"
            self.gpu_layers = 28
            self.prompt_tokens = 10
            self.completion_tokens = 11
            self.total_tokens = 21

    class FakeInferenceResult:
        def __init__(self, image_name):
            self.data = FakeResult(image_name)
            self.telemetry = FakeTelemetry()

    class FakeLoader:
        def __init__(self, model_path, verbose=False, server_api_host=None, api_token=None):
            self.model_path = model_path

        def load_model(self):
            return None

        def inference(self, image_path, prompt, schema, temperature, include_telemetry=False):
            return FakeInferenceResult(image_path.name)

        def unload_model(self):
            return None

    monkeypatch.setattr("src.scripts.batch_runner.VLMLoader", FakeLoader)

    summary = run_batch_job(
        model_id="fake-model",
        manifest=manifest_path,
        schema_name="GenericObjectDetection",
        output_path=output_path,
    )

    assert summary["processed"] == 2
    assert summary["input_source"] == "manifest"
    assert summary["discarded_manifest_rows"] == 0

    rows = _read_jsonl_records(output_path)
    assert rows[0]["ground_truth_cls"] == "AD"
    assert rows[0]["image_id"] == "0"
    assert rows[1]["ground_truth_cls"] == "HP"
    assert rows[1]["image_id"] == "1"


def test_run_batch_job_emits_progress_events(monkeypatch, tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for index in range(2):
        (image_dir / f"sample_{index}.jpg").write_bytes(b"image")

    events = []

    class FakeResult:
        def __init__(self, image_name):
            self._image_name = image_name

        def model_dump(self):
            return {
                "object_detected": self._image_name,
                "confidence_score": 99,
                "justification": "ok",
            }

    class FakeTelemetry:
        def __init__(self):
            self.total_duration_seconds = 1.0
            self.ttft_seconds = 0.2
            self.generation_duration_seconds = 0.8
            self.prompt_tokens_per_second = 111.0
            self.tokens_per_second = 33.0
            self.reasoning_tokens = 2
            self.stop_reason = "eos"
            self.model_id = "fake-model"
            self.vram_usage_mb = 90.0
            self.total_params = 123
            self.architecture = "fake_arch"
            self.quantization = "Q4"
            self.gpu_layers = 12
            self.prompt_tokens = 10
            self.completion_tokens = 20
            self.total_tokens = 30

    class FakeInferenceResult:
        def __init__(self, image_name):
            self.data = FakeResult(image_name)
            self.telemetry = FakeTelemetry()

    class FakeLoader:
        def __init__(self, model_path, verbose=False, server_api_host=None, api_token=None):
            self.model_path = model_path

        def load_model(self):
            return None

        def inference(self, image_path, prompt, schema, temperature, include_telemetry=False):
            return FakeInferenceResult(image_path.name)

        def unload_model(self):
            return None

    monkeypatch.setattr("src.scripts.batch_runner.VLMLoader", FakeLoader)

    summary = run_batch_job(
        model_id="fake-model",
        image_dir=image_dir,
        schema_name="GenericObjectDetection",
        output_path=tmp_path / "results.jsonl",
        on_progress=events.append,
    )

    assert summary["processed"] == 2
    assert [event["event"] for event in events] == [
        "start",
        "image_start",
        "image_done",
        "image_start",
        "image_done",
        "complete",
    ]
    assert events[2]["record"]["status"] == "ok"
    assert events[2]["summary"]["processed"] == 1
    assert events[-1]["summary"]["processed"] == 2
    assert events[-1]["summary"]["ttft"]["avg"] == 0.2


def test_run_batch_job_persists_error_rows_in_jsonl(monkeypatch, tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for index in range(3):
        (image_dir / f"sample_{index}.jpg").write_bytes(b"image")

    output_path = tmp_path / "results.jsonl"

    class FakeResult:
        def model_dump(self):
            return {
                "object_detected": "ok",
                "confidence_score": 95,
                "justification": "ok",
            }

    class FakeTelemetry:
        def __init__(self):
            self.total_duration_seconds = 0.9
            self.ttft_seconds = 0.3
            self.generation_duration_seconds = 0.6
            self.prompt_tokens_per_second = 150.0
            self.tokens_per_second = 30.0
            self.reasoning_tokens = 2
            self.stop_reason = "eos"
            self.model_id = "fake-model"
            self.vram_usage_mb = 80.0
            self.total_params = 9000000000
            self.architecture = "qwen3_vl"
            self.quantization = "Q8_0"
            self.gpu_layers = 24
            self.prompt_tokens = 20
            self.completion_tokens = 18
            self.total_tokens = 38
            self.stats = {"tokens_per_second": 30.0}
            self.resources = {"vram_usage_mb": 80.0}

    class FakeInferenceResult:
        def __init__(self):
            self.data = FakeResult()
            self.telemetry = FakeTelemetry()

    class FakeLoader:
        def __init__(self, model_path, verbose=False, server_api_host=None, api_token=None):
            self.calls = 0

        def load_model(self):
            return None

        def inference(self, image_path, prompt, schema, temperature, include_telemetry=False):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("La respuesta no cumple el esquema 'GenericObjectDetection'")
            assert include_telemetry is True
            return FakeInferenceResult()

        def unload_model(self):
            return None

    monkeypatch.setattr("src.scripts.batch_runner.VLMLoader", FakeLoader)
    monkeypatch.setattr("src.scripts.batch_runner.tqdm", lambda iterable, **kwargs: iterable)

    summary = run_batch_job(
        model_id="fake-model",
        image_dir=image_dir,
        schema_name="GenericObjectDetection",
        output_path=output_path,
    )

    assert summary["processed"] == 3
    assert summary["ok"] == 2
    assert summary["invalid"] == 1
    assert summary["fail"] == 0

    records = _read_jsonl_records(output_path)

    assert len(records) == 3
    assert "object_detected" not in records[0]
    assert records[0]["payload"]["object_detected"] == "ok"
    assert records[1]["status"] == "invalid"
    assert "no cumple el esquema" in records[1]["error_message"].lower()


def test_build_record_omits_absent_optional_fields_in_json_like_payload():
    record = build_record(
        model_id="fake-model",
        schema_name="GenericObjectDetection",
        image_path=Path("demo.jpg"),
        duration_seconds=0.5,
        status="ok",
        payload={"object_detected": "cat"},
        telemetry={"tokens_per_second": 12.0},
    )

    assert record["tokens_per_second"] == 12.0
    assert record["payload"] == {"object_detected": "cat"}
    assert "object_detected" not in record
    assert "prompt_tokens_per_second" not in record
    assert "error_message" not in record
    assert "total_duration_seconds" not in record


def test_build_record_ignores_manifest_metadata_key_collisions():
    record = build_record(
        model_id="fake-model",
        schema_name="GenericObjectDetection",
        image_path=Path("demo.jpg"),
        duration_seconds=0.5,
        status="ok",
        payload={"object_detected": "cat"},
        telemetry={"tokens_per_second": 12.0},
        metadata={"image_path": "should-not-override", "ground_truth_cls": "AD"},
    )

    assert record["image_path"] == "demo.jpg"
    assert record["ground_truth_cls"] == "AD"


def test_build_record_can_keep_payload_nested_without_flattening():
    record = build_record(
        model_id="fake-model",
        schema_name="GenericObjectDetection",
        image_path=Path("demo.jpg"),
        duration_seconds=0.5,
        status="ok",
        payload={"object_detected": "cat"},
        telemetry={"tokens_per_second": 12.0},
    )

    assert record["payload"] == {"object_detected": "cat"}
    assert "object_detected" not in record


def test_batch_meta_header_tracks_multiple_models_and_summary_line(tmp_path):
    output_path = tmp_path / "shared_results.jsonl"

    first_rows = [
        {
            "__batch_meta__": {
                "version": 1,
                "created_at_utc": "2026-03-13T18:00:00+00:00",
                "model_id": "model-a",
                "schema_name": "PolypClassification",
                "input_source": "manifest",
            }
        },
        {
            "timestamp_utc": "2026-03-13T18:00:01+00:00",
            "model_id": "model-a",
            "schema_name": "PolypClassification",
            "image_path": "a.tif",
            "status": "ok",
            "duration_seconds": 1.0,
            "ttft_seconds": 0.2,
            "tokens_per_second": 20.0,
            "total_duration_seconds": 0.8,
        },
        {
            "timestamp_utc": "2026-03-13T18:00:02+00:00",
            "model_id": "model-b",
            "schema_name": "PolypClassification",
            "image_path": "b.tif",
            "status": "ok",
            "duration_seconds": 2.0,
            "ttft_seconds": 0.4,
            "tokens_per_second": 40.0,
            "total_duration_seconds": 1.6,
        },
    ]
    output_path.write_text("\n".join(json.dumps(row) for row in first_rows) + "\n", encoding="utf-8")

    # Simulamos un meta actualizado como lo dejaría _ensure_batch_meta_header versión 2.
    rewritten_meta = {
        "__batch_meta__": {
            "version": 2,
            "created_at_utc": "2026-03-13T18:00:00+00:00",
            "updated_at_utc": "2026-03-13T18:10:00+00:00",
            "output_mode": "shared_jsonl",
            "model_ids": ["model-a", "model-b"],
            "schema_names": ["PolypClassification"],
            "input_sources": ["manifest"],
            "model_id": "model-a",
            "schema_name": "PolypClassification",
            "input_source": "manifest",
        }
    }
    second_rows = [rewritten_meta, *first_rows[1:]]
    output_path.write_text("\n".join(json.dumps(row) for row in second_rows) + "\n", encoding="utf-8")

    summary = upsert_batch_execution_summary(
        output_path=output_path,
        schema_name="PolypClassification",
        model_ids=["model-a", "model-b"],
    )

    assert summary is not None
    assert summary["model_count"] == 2
    assert summary["models"]["model-a"]["metrics"]["ttft_seconds"]["avg"] == pytest.approx(0.2)
    assert summary["models"]["model-b"]["metrics"]["ttft_seconds"]["avg"] == pytest.approx(0.4)
    assert summary["global"]["metrics"]["ttft_seconds"]["min"] == pytest.approx(0.2)
    assert summary["global"]["metrics"]["ttft_seconds"]["max"] == pytest.approx(0.4)
    assert summary["global"]["metrics"]["ttft_seconds"]["avg"] == pytest.approx(0.3)

    all_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert isinstance(all_rows[0].get("__batch_meta__"), dict)
    summary_rows = [row for row in all_rows if isinstance(row.get("__batch_summary__"), dict)]
    assert len(summary_rows) == 1


def test_run_batch_job_updates_shared_meta_model_ids(monkeypatch, tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "sample_0.jpg").write_bytes(b"image")

    output_path = tmp_path / "shared_results.jsonl"

    class FakeResult:
        def model_dump(self):
            return {
                "object_detected": "x",
                "confidence_score": 90,
                "justification": "ok",
            }

    class FakeTelemetry:
        def __init__(self):
            self.total_duration_seconds = 1.0
            self.ttft_seconds = 0.2
            self.generation_duration_seconds = 0.7
            self.tokens_per_second = 20.0

    class FakeInferenceResult:
        def __init__(self):
            self.data = FakeResult()
            self.telemetry = FakeTelemetry()

    class FakeLoader:
        def __init__(self, model_path, verbose=False, server_api_host=None, api_token=None):
            self.model_path = model_path

        def load_model(self):
            return None

        def inference(self, image_path, prompt, schema, temperature, include_telemetry=False):
            return FakeInferenceResult()

        def unload_model(self):
            return None

    monkeypatch.setattr("src.scripts.batch_runner.VLMLoader", FakeLoader)

    run_batch_job(
        model_id="model-a",
        image_dir=image_dir,
        schema_name="GenericObjectDetection",
        output_path=output_path,
    )
    run_batch_job(
        model_id="model-b",
        image_dir=image_dir,
        schema_name="GenericObjectDetection",
        output_path=output_path,
    )

    first_line = output_path.read_text(encoding="utf-8").splitlines()[0]
    header = json.loads(first_line).get("__batch_meta__", {})
    assert header.get("version") == 2
    assert sorted(header.get("model_ids", [])) == ["model-a", "model-b"]
    assert header.get("output_mode") == "shared_jsonl"


def test_upsert_batch_execution_summary_includes_accuracy(tmp_path):
    output_path = tmp_path / "accuracy_results.jsonl"
    rows = [
        {
            "__batch_meta__": {
                "version": 2,
                "created_at_utc": "2026-03-13T18:00:00+00:00",
                "updated_at_utc": "2026-03-13T18:00:00+00:00",
                "output_mode": "shared_jsonl",
                "model_ids": ["model-a", "model-b"],
                "schema_names": ["PolypClassification"],
                "input_sources": ["manifest"],
            }
        },
        {
            "model_id": "model-a",
            "schema_name": "PolypClassification",
            "status": "ok",
            "ground_truth_cls": "AD",
            "payload": {"predicted_class": "AD"},
            "duration_seconds": 1.0,
            "ttft_seconds": 0.2,
            "tokens_per_second": 20.0,
            "total_duration_seconds": 0.8,
        },
        {
            "model_id": "model-a",
            "schema_name": "PolypClassification",
            "status": "ok",
            "ground_truth_cls": "HP",
            "payload": {"predicted_class": "AD"},
            "duration_seconds": 1.1,
            "ttft_seconds": 0.3,
            "tokens_per_second": 21.0,
            "total_duration_seconds": 0.9,
        },
        {
            "model_id": "model-b",
            "schema_name": "PolypClassification",
            "status": "ok",
            "ground_truth_cls": "ASS",
            "payload": {"predicted_class": "ASS"},
            "duration_seconds": 1.2,
            "ttft_seconds": 0.4,
            "tokens_per_second": 22.0,
            "total_duration_seconds": 1.0,
        },
    ]
    output_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    summary = upsert_batch_execution_summary(
        output_path=output_path,
        schema_name="PolypClassification",
        model_ids=["model-a", "model-b"],
    )

    assert summary is not None
    assert summary["models"]["model-a"]["accuracy"]["evaluated"] == 2
    assert summary["models"]["model-a"]["accuracy"]["correct"] == 1
    assert summary["models"]["model-a"]["accuracy"]["value"] == pytest.approx(0.5)
    assert summary["models"]["model-b"]["accuracy"]["value"] == pytest.approx(1.0)
    assert summary["global"]["accuracy"]["evaluated"] == 3
    assert summary["global"]["accuracy"]["correct"] == 2
    assert summary["global"]["accuracy"]["value"] == pytest.approx(2.0 / 3.0)


def test_run_batch_job_filters_pending_entries_by_iteration(monkeypatch, tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "sample_0.tif"
    image_path.write_bytes(b"image")

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_rows = [
        {
            "image_path": str(image_path),
            "ground_truth_cls": "AD",
            "image_id": "0",
            "run_iteration_index": 1,
            "run_iteration_total": 2,
        },
        {
            "image_path": str(image_path),
            "ground_truth_cls": "AD",
            "image_id": "0",
            "run_iteration_index": 2,
            "run_iteration_total": 2,
        },
    ]
    manifest_path.write_text("\n".join(json.dumps(row) for row in manifest_rows) + "\n", encoding="utf-8")

    output_path = tmp_path / "results.jsonl"

    class FakeResult:
        def __init__(self, image_name):
            self._image_name = image_name

        def model_dump(self):
            return {
                "object_detected": self._image_name,
                "confidence_score": 93,
                "justification": "ok",
            }

    class FakeTelemetry:
        def __init__(self):
            self.total_duration_seconds = 1.2
            self.ttft_seconds = 0.4
            self.generation_duration_seconds = 0.8
            self.tokens_per_second = 21.0
            self.reasoning_tokens = 3
            self.stop_reason = "eos"
            self.model_id = "fake-model"
            self.architecture = "qwen3_vl"
            self.gpu_layers = 28
            self.prompt_tokens = 10
            self.completion_tokens = 11
            self.total_tokens = 21

    class FakeInferenceResult:
        def __init__(self, image_name):
            self.data = FakeResult(image_name)
            self.telemetry = FakeTelemetry()

    class FakeLoader:
        def __init__(self, model_path, verbose=False, server_api_host=None, api_token=None):
            self.model_path = model_path

        def load_model(self):
            return None

        def inference(self, image_path, prompt, schema, temperature, include_telemetry=False):
            return FakeInferenceResult(image_path.name)

        def unload_model(self):
            return None

    monkeypatch.setattr("src.scripts.batch_runner.VLMLoader", FakeLoader)

    summary = run_batch_job(
        model_id="fake-model",
        manifest=manifest_path,
        schema_name="GenericObjectDetection",
        output_path=output_path,
        pending_entries=[{"image_path": str(image_path), "run_iteration_index": 2}],
    )

    assert summary["processed"] == 1
    rows = _read_jsonl_records(output_path)
    assert len(rows) == 1
    assert int(rows[0].get("run_iteration_index") or 0) == 2