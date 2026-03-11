from types import SimpleNamespace

import pytest

from src.inference.schemas import GenericObjectDetection, get_schema_variant
from src.inference.vlm_runner import InferenceResult, VLMLoader
from src.scripts.test_telemetry import (
    build_cli_progress_callback,
    first_available_value,
    format_metric,
    render_telemetry_report,
    run_telemetry_batch,
    summarize_numeric,
)


def test_summarize_numeric_returns_mean_min_max():
    summary = summarize_numeric([1.0, 2.0, 4.0])
    assert summary == {"avg": pytest.approx(7 / 3), "min": 1.0, "max": 4.0}


def test_format_metric_returns_nd_for_missing_values():
    assert format_metric(None) == "N/D"
    assert format_metric(1.25, suffix=" s") == "1.25 s"
    assert format_metric(2.0, suffix=" s") == "2 s"


def test_first_available_value_returns_first_non_empty_record():
    records = [
        {"architecture": None},
        {"architecture": "qwen3"},
        {"architecture": "other"},
    ]
    assert first_available_value(records, "architecture") == "qwen3"


def test_inference_include_telemetry_reads_sdk_stats_and_rest_resources(monkeypatch):
    loader = VLMLoader(model_path="fake-model")

    mock_model = SimpleNamespace(
        respond=lambda *args, **kwargs: SimpleNamespace(
            parsed={
                "object_detected": "cat",
                "confidence_score": 91,
                "justification": "ok",
            },
            stats=SimpleNamespace(
                prompt_tokens_count=12,
                predicted_tokens_count=24,
                prompt_tokens_per_sec=100.0,
                predicted_tokens_per_sec=60.0,
                time_to_first_token_sec=0.35,
                total_duration_ms=900.0,
                stop_reason="eos",
            ),
        )
    )

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": [
                    {
                        "id": "fake-model",
                        "stats": {
                            "vram_usage_bytes": 104857600,
                            "gpu_layers": 28,
                        },
                        "metadata": {
                            "total_params": 9000000000,
                            "architecture": "qwen3_vl",
                            "quantization": "Q8_0",
                        },
                    }
                ]
            }

    monkeypatch.setattr("src.inference.vlm_runner.requests", SimpleNamespace(get=lambda *args, **kwargs: FakeResponse()))
    monkeypatch.setattr(loader, "load_model", lambda: None)
    loader._loaded_model = mock_model
    monkeypatch.setattr("src.inference.vlm_runner.os.path.isfile", lambda *_: True)
    monkeypatch.setattr(loader, "_build_multimodal_history", lambda structured_text, image_path: structured_text)

    perf_values = iter([10.0, 11.0])
    monkeypatch.setattr("src.inference.vlm_runner.time.perf_counter", lambda: next(perf_values))

    result = loader.inference(
        "fake-image.jpg",
        "Prompt",
        schema=GenericObjectDetection,
        include_telemetry=True,
    )

    assert isinstance(result, InferenceResult)
    assert result.telemetry.total_duration_seconds == pytest.approx(0.9)
    assert result.telemetry.ttft_seconds == pytest.approx(0.35)
    assert result.telemetry.generation_duration_seconds == pytest.approx(0.4)
    assert result.telemetry.prompt_tokens_per_second == pytest.approx(100.0)
    assert result.telemetry.tokens_per_second == pytest.approx(60.0)
    assert result.telemetry.prompt_tokens == 12
    assert result.telemetry.completion_tokens == 24
    assert result.telemetry.total_tokens == 36
    assert result.telemetry.stop_reason == "eos"
    assert result.telemetry.model_id == "fake-model"
    assert result.telemetry.vram_usage_mb == pytest.approx(100.0)
    assert result.telemetry.total_params == 9000000000
    assert result.telemetry.architecture == "qwen3_vl"
    assert result.telemetry.quantization == "Q8_0"
    assert result.telemetry.gpu_layers == 28


def test_inference_include_telemetry_counts_reasoning_tokens_from_schema(monkeypatch):
    loader = VLMLoader(model_path="fake-model")
    schema_name, reasoning_schema = get_schema_variant("GenericObjectDetection", True)
    assert schema_name == "GenericObjectDetectionWithReasoning"

    mock_model = SimpleNamespace(
        respond=lambda *args, **kwargs: SimpleNamespace(
            parsed={
                "reasoning": "hallazgo compatible con un objeto felino pequeno",
                "object_detected": "cat",
                "confidence_score": 91,
                "justification": "ok",
            },
            stats=SimpleNamespace(),
        )
    )

    monkeypatch.setattr("src.inference.vlm_runner.requests", None)
    monkeypatch.setattr(loader, "load_model", lambda: None)
    loader._loaded_model = mock_model
    monkeypatch.setattr("src.inference.vlm_runner.os.path.isfile", lambda *_: True)
    monkeypatch.setattr(loader, "_build_multimodal_history", lambda structured_text, image_path: structured_text)

    perf_values = iter([20.0, 21.0])
    monkeypatch.setattr("src.inference.vlm_runner.time.perf_counter", lambda: next(perf_values))

    result = loader.inference(
        "fake-image.jpg",
        "Prompt",
        schema=reasoning_schema,
        include_telemetry=True,
    )

    assert isinstance(result, InferenceResult)
    assert result.telemetry.reasoning_tokens == 7


def test_inference_include_telemetry_uses_wall_time_when_stats_are_missing(monkeypatch):
    loader = VLMLoader(model_path="fake-model")

    mock_model = SimpleNamespace(
        respond=lambda *args, **kwargs: SimpleNamespace(
            parsed={
                "object_detected": "cat",
                "confidence_score": 91,
                "justification": "ok",
            },
        )
    )

    monkeypatch.setattr("src.inference.vlm_runner.requests", None)
    monkeypatch.setattr(loader, "load_model", lambda: None)
    loader._loaded_model = mock_model
    monkeypatch.setattr("src.inference.vlm_runner.os.path.isfile", lambda *_: True)
    monkeypatch.setattr(loader, "_build_multimodal_history", lambda structured_text, image_path: structured_text)

    perf_values = iter([30.0, 32.0])
    monkeypatch.setattr("src.inference.vlm_runner.time.perf_counter", lambda: next(perf_values))

    result = loader.inference(
        "fake-image.jpg",
        "Prompt",
        schema=GenericObjectDetection,
        include_telemetry=True,
    )

    assert result.telemetry.total_duration_seconds == pytest.approx(2.0)
    assert result.telemetry.ttft_seconds is None
    assert result.telemetry.tokens_per_second is None
    assert result.telemetry.vram_usage_mb is None


def test_inference_include_telemetry_uses_sdk_model_info_when_rest_is_missing(monkeypatch):
    loader = VLMLoader(model_path="fake-model")

    mock_model = SimpleNamespace(
        respond=lambda *args, **kwargs: SimpleNamespace(
            parsed={
                "object_detected": "cat",
                "confidence_score": 91,
                "justification": "ok",
            },
            stats=SimpleNamespace(
                prompt_tokens_count=12,
                predicted_tokens_count=24,
                total_tokens_count=36,
                tokens_per_second=60.0,
                time_to_first_token_sec=0.35,
                stop_reason="eosFound",
                num_gpu_layers=-1.0,
            ),
            model_info=SimpleNamespace(
                model_key="fake-model",
                architecture="qwen35",
                params_string="9B",
                path="repo/Qwen3.5-9B-Q8_0.gguf",
            ),
        )
    )

    monkeypatch.setattr("src.inference.vlm_runner.requests", None)
    monkeypatch.setattr(loader, "load_model", lambda: None)
    loader._loaded_model = mock_model
    monkeypatch.setattr("src.inference.vlm_runner.os.path.isfile", lambda *_: True)
    monkeypatch.setattr(loader, "_build_multimodal_history", lambda structured_text, image_path: structured_text)

    perf_values = iter([40.0, 41.0])
    monkeypatch.setattr("src.inference.vlm_runner.time.perf_counter", lambda: next(perf_values))

    result = loader.inference(
        "fake-image.jpg",
        "Prompt",
        schema=GenericObjectDetection,
        include_telemetry=True,
    )

    assert result.telemetry.model_id == "fake-model"
    assert result.telemetry.architecture == "qwen35"
    assert result.telemetry.total_params == "9B"
    assert result.telemetry.quantization == "Q8_0"
    assert result.telemetry.gpu_layers == -1
    assert result.telemetry.tokens_per_second == pytest.approx(60.0)
    assert result.telemetry.total_tokens == 36


def test_run_telemetry_batch_collects_summary(monkeypatch):
    class FakeTelemetry:
        def __init__(self, ttft, tps, total):
            self.total_duration_seconds = total
            self.ttft_seconds = ttft
            self.generation_duration_seconds = 0.5
            self.prompt_tokens_per_second = 120.0
            self.tokens_per_second = tps
            self.reasoning_tokens = 3
            self.stop_reason = "eos"
            self.model_id = "fake-model"
            self.vram_usage_mb = 512.0
            self.total_params = 9000000000
            self.architecture = "qwen3_vl"
            self.quantization = "Q8_0"
            self.gpu_layers = 28
            self.prompt_tokens = 20
            self.completion_tokens = 15
            self.total_tokens = 35
            self.stats = {"predicted_tokens_per_sec": tps}
            self.resources = {"vram_usage_mb": 512.0}

    class FakePayload:
        def __init__(self, ttft, tps, total):
            self.data = GenericObjectDetection(
                object_detected="cat",
                confidence_score=90,
                justification="ok",
            )
            self.telemetry = FakeTelemetry(ttft, tps, total)

    class FakeLoader:
        def __init__(self, model_path, verbose=False, server_api_host=None, api_token=None):
            self.calls = 0

        def load_model(self):
            return None

        def inference(self, image_path, prompt, schema, temperature, include_telemetry=False):
            self.calls += 1
            if self.calls == 3:
                raise RuntimeError("boom")
            return FakePayload(0.4 + self.calls * 0.1, 20.0 + self.calls, 1.0 + self.calls)

        def unload_model(self):
            return None

    monkeypatch.setattr("src.scripts.test_telemetry.VLMLoader", FakeLoader)

    summary = run_telemetry_batch(
        model_id="fake-model",
        schema_name="GenericObjectDetection",
        schema_cls=GenericObjectDetection,
        images=["a.jpg", "b.jpg", "c.jpg"],
        max_images=3,
        seed=123,
    )

    assert summary["ok"] == 2
    assert summary["fail"] == 1
    assert summary["sample_size"] == 3
    assert summary["ttft"]["avg"] == pytest.approx(0.55)
    assert summary["prompt_tokens_per_second"]["avg"] == pytest.approx(120.0)
    assert summary["tps"]["avg"] == pytest.approx(21.5)
    assert summary["generation_duration"]["avg"] == pytest.approx(0.5)
    assert summary["reasoning_tokens"]["avg"] == pytest.approx(3.0)
    assert summary["vram_usage_mb"]["avg"] == pytest.approx(512.0)
    assert summary["gpu_layers"]["avg"] == pytest.approx(28.0)
    assert summary["prompt_tokens"]["avg"] == pytest.approx(20.0)
    assert summary["static_model_info"]["architecture"] == "qwen3_vl"
    assert summary["notes"]["ttft"] is None
    assert summary["notes"]["tps"] is None
    assert summary["notes"]["resources"] is None


def test_run_telemetry_batch_reports_notes_when_stats_and_rest_are_missing(monkeypatch):
    class FakeTelemetry:
        def __init__(self):
            self.total_duration_seconds = 1.1
            self.ttft_seconds = None
            self.generation_duration_seconds = None
            self.prompt_tokens_per_second = None
            self.tokens_per_second = None
            self.reasoning_tokens = None
            self.stop_reason = None
            self.model_id = None
            self.vram_usage_mb = None
            self.total_params = None
            self.architecture = None
            self.quantization = None
            self.gpu_layers = None
            self.prompt_tokens = None
            self.completion_tokens = None
            self.total_tokens = None
            self.stats = {}
            self.resources = {}

    class FakePayload:
        def __init__(self):
            self.data = GenericObjectDetection(
                object_detected="cat",
                confidence_score=90,
                justification="ok",
            )
            self.telemetry = FakeTelemetry()

    class FakeLoader:
        def __init__(self, model_path, verbose=False, server_api_host=None, api_token=None):
            return None

        def load_model(self):
            return None

        def inference(self, image_path, prompt, schema, temperature, include_telemetry=False):
            return FakePayload()

        def unload_model(self):
            return None

    monkeypatch.setattr("src.scripts.test_telemetry.VLMLoader", FakeLoader)

    summary = run_telemetry_batch(
        model_id="fake-model",
        schema_name="GenericObjectDetection",
        schema_cls=GenericObjectDetection,
        images=["a.jpg", "b.jpg"],
        max_images=2,
        seed=123,
    )

    assert summary["ok"] == 2
    assert summary["ttft"]["avg"] is None
    assert summary["tps"]["avg"] is None
    assert summary["vram_usage_mb"]["avg"] is None
    assert "TTFT" in summary["notes"]["ttft"]
    assert "TPS" in summary["notes"]["tps"]
    assert summary["notes"]["resources"] is None


def test_render_telemetry_report_prints_structured_sections(capsys):
    summary = {
        "model_id": "fake-model",
        "schema_name": "GenericObjectDetection",
        "sample_size": 2,
        "ok": 2,
        "fail": 0,
        "records": [
            {
                "image_path": "a.jpg",
                "status": "ok",
                "ttft_seconds": 0.4,
                "tokens_per_second": 22.0,
                "prompt_tokens_per_second": 120.0,
                "total_duration_seconds": 1.1,
            },
            {
                "image_path": "b.jpg",
                "status": "error",
                "error": "boom",
            },
        ],
        "ttft": {"avg": 0.4, "min": 0.4, "max": 0.4},
        "prompt_tokens_per_second": {"avg": 120.0, "min": 120.0, "max": 120.0},
        "tps": {"avg": 22.0, "min": 22.0, "max": 22.0},
        "generation_duration": {"avg": 0.7, "min": 0.7, "max": 0.7},
        "reasoning_tokens": {"avg": 11.0, "min": 11.0, "max": 11.0},
        "vram_usage_mb": {"avg": None, "min": None, "max": None},
        "gpu_layers": {"avg": None, "min": None, "max": None},
        "prompt_tokens": {"avg": 100.0, "min": 100.0, "max": 100.0},
        "completion_tokens": {"avg": 40.0, "min": 40.0, "max": 40.0},
        "total_tokens": {"avg": 140.0, "min": 140.0, "max": 140.0},
        "total_duration": {"avg": 1.1, "min": 1.1, "max": 1.1},
        "static_model_info": {
            "resolved_model_id": "fake-model",
            "architecture": "qwen3_vl",
            "quantization": "Q8_0",
            "total_params": "9B",
            "stop_reason": "eos",
        },
        "telemetry_availability": {
            "ttft_records": 1,
            "prompt_tps_records": 1,
            "tps_records": 1,
            "vram_records": 0,
            "gpu_layer_records": 0,
            "ok_records": 2,
        },
        "notes": {
            "ttft": None,
            "tps": "LM Studio no devolvio predicted_tokens_per_sec en response.stats; TPS no disponible en esta ejecucion.",
            "resources": None,
        },
        "prompt": "Describe la imagen.",
    }

    render_telemetry_report(summary)
    output = capsys.readouterr().out

    assert "TELEMETRY PROBE" in output
    assert "RENDIMIENTO" in output
    assert "TOKENS Y RECURSOS" in output
    assert "COBERTURA" in output
    assert "DETALLE POR IMAGEN" in output
    assert "[OK] a.jpg" in output
    assert "[ERR] b.jpg | boom" in output
    assert "Prompt usado" not in output
    assert "/v1/models" not in output


def test_build_cli_progress_callback_reports_completion_without_rest_warning(capsys):
    callback = build_cli_progress_callback()

    callback({"event": "start", "total": 2})
    callback({"event": "image_start", "index": 1, "total": 2, "image_path": "a.jpg"})
    callback(
        {
            "event": "image_done",
            "index": 1,
            "total": 2,
            "image_path": "a.jpg",
            "record": {
                "status": "ok",
                "ttft_seconds": 0.4,
                "tokens_per_second": 20.0,
                "total_duration_seconds": 1.0,
            },
        }
    )
    callback(
        {
            "event": "complete",
            "summary": {
                "ok": 1,
                "fail": 1,
                "telemetry_availability": {
                    "ttft_records": 1,
                    "tps_records": 1,
                    "vram_records": 0,
                    "ok_records": 1,
                },
            },
        }
    )

    output = capsys.readouterr().out
    assert "Preparando muestra" in output
    assert "[1/2] Procesando a.jpg" in output
    assert "[1/2] [OK] a.jpg" in output
    assert "Completado: 1 OK / 1 errores" in output
    assert "/v1/models" not in output
