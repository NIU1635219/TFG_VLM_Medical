import json
from pathlib import Path

from src.scripts.batch_runner import (
    build_record,
    build_parser,
    iter_image_paths,
    run_batch_job,
)


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
    assert args.schema == "GenericObjectDetection"


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

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").strip().splitlines()]

    assert len(rows) == 5
    assert rows[0]["status"] == "ok"
    assert rows[0]["payload"]["object_detected"].startswith("sample_")
    assert rows[0]["total_duration_seconds"] == 1.2
    assert rows[0]["ttft_seconds"] == 0.4
    assert rows[0]["tokens_per_second"] == 22.5
    assert rows[0]["reasoning_tokens"] == 4
    assert rows[0]["stop_reason"] == "eos"
    assert rows[0]["resolved_model_id"] == "fake-model"
    assert rows[0]["architecture"] == "qwen3_vl"
    assert rows[0]["gpu_layers"] == 28


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

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    records = [json.loads(line) for line in lines]

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