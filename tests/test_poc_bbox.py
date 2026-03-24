"""Tests unitarios para `src/scripts/poc_bbox.py`."""

from __future__ import annotations

import urllib.error
from pathlib import Path

import pytest

from src.inference.schemas import BoundingBox, BoundingBoxDetection
from src.scripts import poc_bbox


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_download_with_headers_retries_429(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Si hay HTTP 429, debe reintentar y emitir evento `download_retry`."""
    output = tmp_path / "sample.jpg"
    waits: list[int] = []
    events: list[tuple[str, dict[str, object]]] = []

    def reporter(event: str, payload: dict[str, object]) -> None:
        events.append((event, payload))

    calls = {"count": 0}

    def fake_urlopen(req, timeout=15):  # noqa: ANN001
        calls["count"] += 1
        if calls["count"] == 1:
            raise urllib.error.HTTPError(str(req.full_url), 429, "Too Many Requests", hdrs=None, fp=None)
        return _FakeResponse(b"image-bytes")

    monkeypatch.setattr(poc_bbox.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(poc_bbox.time, "sleep", lambda seconds: waits.append(int(seconds)))

    poc_bbox._download_with_headers("https://example.com/image.jpg", output, reporter=reporter)

    assert output.read_bytes() == b"image-bytes"
    assert calls["count"] == 2
    assert waits == [5]
    retry_events = [payload for event, payload in events if event == "download_retry"]
    assert retry_events and retry_events[0]["wait_seconds"] == 5


def test_ensure_assets_emits_download_events_and_blank(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """`ensure_assets` debe descargar faltantes y generar blank sample con eventos."""
    monkeypatch.setattr(poc_bbox, "TEST_IMAGE_DIR", tmp_path)
    monkeypatch.setattr(poc_bbox, "SAMPLES", {"one.jpg": "https://example.com/one.jpg"})
    monkeypatch.setattr(poc_bbox.time, "sleep", lambda _seconds: None)

    def fake_download(url: str, dest_path: Path, *, reporter=None):  # noqa: ANN001
        dest_path.write_bytes(b"ok")

    monkeypatch.setattr(poc_bbox, "_download_with_headers", fake_download)

    events: list[str] = []

    def reporter(event: str, payload: dict[str, object]) -> None:  # noqa: ARG001
        events.append(event)

    poc_bbox.ensure_assets(reporter=reporter)

    assert (tmp_path / "one.jpg").exists()
    assert (tmp_path / "blank_sample.jpg").exists()
    assert "download_start" in events
    assert "download_ok" in events
    assert "blank_generated" in events


def test_main_success_emits_expected_events(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """`main` debe emitir progreso completo en ejecución exitosa y propagar host/token."""
    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / "cat_1.jpg").write_bytes(b"fake-jpg")

    out_jsonl = tmp_path / "results" / "run_1" / "results.jsonl"
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.write_text("", encoding="utf-8")

    monkeypatch.setattr(poc_bbox, "TEST_IMAGE_DIR", image_dir)
    monkeypatch.setattr(poc_bbox, "ensure_assets", lambda *, reporter=None: None)
    monkeypatch.setattr(poc_bbox, "_save_run_results", lambda *_args, **_kwargs: out_jsonl)

    captured_init: dict[str, object] = {}

    class FakeLoader:
        def __init__(
            self,
            model_path: str,
            verbose: bool = False,
            server_api_host: str | None = None,
            api_token: str | None = None,
        ):
            captured_init.update(
                {
                    "model_path": model_path,
                    "verbose": verbose,
                    "server_api_host": server_api_host,
                    "api_token": api_token,
                }
            )

        def inference(self, image_path: str, prompt: str, schema, system_prompt: str):  # noqa: ANN001
            _ = (image_path, prompt, schema, system_prompt)
            detection = BoundingBoxDetection(
                detected_subject="gato naranja en la mitad izquierda",
                ymin=100,
                xmin=120,
                ymax=800,
                xmax=700,
            )
            return BoundingBox(
                object_count_reasoning="Se observa un solo sujeto principal en la escena.",
                detected_subjects_count=1,
                detections=[detection],
            )

    monkeypatch.setattr(poc_bbox, "VLMLoader", FakeLoader)

    events: list[str] = []

    def reporter(event: str, payload: dict[str, object]) -> None:  # noqa: ARG001
        events.append(event)

    rc = poc_bbox.main(
        ["--model", "fake-model", "--host", "localhost:1234", "--api-token", "secret"],
        reporter=reporter,
    )

    assert rc == 0
    assert captured_init["model_path"] == "fake-model"
    assert captured_init["server_api_host"] == "localhost:1234"
    assert captured_init["api_token"] == "secret"
    assert events.count("image_start") == 1
    assert "image_result" in events
    assert "run_saved" in events


def test_main_emits_image_error_when_inference_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Si falla una inferencia, debe emitir `image_error` y continuar hasta guardar."""
    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / "dog_1.jpg").write_bytes(b"fake-jpg")

    out_jsonl = tmp_path / "results" / "run_2" / "results.jsonl"
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.write_text("", encoding="utf-8")

    monkeypatch.setattr(poc_bbox, "TEST_IMAGE_DIR", image_dir)
    monkeypatch.setattr(poc_bbox, "ensure_assets", lambda *, reporter=None: None)
    monkeypatch.setattr(poc_bbox, "_save_run_results", lambda *_args, **_kwargs: out_jsonl)

    class FailingLoader:
        def __init__(self, model_path: str, verbose: bool = False, server_api_host=None, api_token=None):  # noqa: ANN001
            _ = (model_path, verbose, server_api_host, api_token)

        def inference(self, image_path: str, prompt: str, schema, system_prompt: str):  # noqa: ANN001
            _ = (image_path, prompt, schema, system_prompt)
            raise RuntimeError("boom")

    monkeypatch.setattr(poc_bbox, "VLMLoader", FailingLoader)

    seen: list[str] = []

    def reporter(event: str, payload: dict[str, object]) -> None:  # noqa: ARG001
        seen.append(event)

    rc = poc_bbox.main(["--model", "fake-model"], reporter=reporter)

    assert rc == 0
    assert "image_error" in seen
    assert "run_saved" in seen


def test_main_rejects_inconsistent_bbox_count(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Si count y detections no coinciden, debe emitirse image_error y no image_result."""
    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / "dog_5.jpg").write_bytes(b"fake-jpg")

    out_jsonl = tmp_path / "results" / "run_3" / "results.jsonl"
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.write_text("", encoding="utf-8")

    monkeypatch.setattr(poc_bbox, "TEST_IMAGE_DIR", image_dir)
    monkeypatch.setattr(poc_bbox, "ensure_assets", lambda *, reporter=None: None)
    monkeypatch.setattr(poc_bbox, "_save_run_results", lambda *_args, **_kwargs: out_jsonl)

    class _RawBBoxResult:
        def __init__(self):
            self.object_count_reasoning = "Hay dos perros en escena"
            self.detected_subjects_count = 2
            self.detections = []

    class InconsistentLoader:
        def __init__(self, model_path: str, verbose: bool = False, server_api_host=None, api_token=None):  # noqa: ANN001
            _ = (model_path, verbose, server_api_host, api_token)

        def inference(self, image_path: str, prompt: str, schema, system_prompt: str):  # noqa: ANN001
            _ = (image_path, prompt, schema, system_prompt)
            return _RawBBoxResult()

    monkeypatch.setattr(poc_bbox, "VLMLoader", InconsistentLoader)

    seen: list[str] = []

    def reporter(event: str, payload: dict[str, object]) -> None:  # noqa: ARG001
        seen.append(event)

    rc = poc_bbox.main(["--model", "fake-model"], reporter=reporter)

    assert rc == 0
    assert "image_error" in seen
    assert "image_result" not in seen
    assert "run_saved" in seen
