import pytest

from src.inference.vlm_runner import VLMStructuredResponse

from src.scripts.test_inference import (
    TEST_CASES,
    contains_any_keyword,
    normalize_text,
    parse_lms_ls_output,
    parse_structured_response,
    run_smoke_test,
    resolve_model,
    validate_response,
)


def test_parse_lms_ls_output_extracts_model_tags():
    raw = (
        "MODEL                                SIZE      MODIFIED\n"
        "openbmb/minicpm-v4.5:8b              4.8 GB    2 hours ago\n"
        "qwen3-vl:latest                      5.1 GB    1 day ago\n"
    )
    models = parse_lms_ls_output(raw)
    assert models == ["openbmb/minicpm-v4.5:8b", "qwen3-vl:latest"]


def test_normalize_text_removes_accents_and_lowercases():
    assert normalize_text("GatÓ PERRÓ") == "gato perro"


def test_contains_any_keyword_with_variants():
    response = "En la imagen aparece claramente un felino doméstico."
    assert contains_any_keyword(response, ["gato", "felino"]) is True


def test_validate_response_by_label_success_and_fail():
    ok, _ = validate_response(
        "cat",
        '{"polyp_detected": false, "confidence_score": 74, "justification": "La imagen muestra un gato sobre una silla."}',
    )
    assert ok is True

    fail, message = validate_response(
        "dog",
        '{"polyp_detected": false, "confidence_score": 88, "justification": "Se aprecia un gato durmiendo."}',
    )
    assert fail is False
    assert "FALLIDA" in message


def test_validate_response_empty_output_has_clear_message():
    fail, message = validate_response("cat", "")
    assert fail is False
    assert "respuesta vacía" in message


def test_parse_structured_response_requires_fields():
    with pytest.raises(ValueError):
        parse_structured_response('{"polyp_detected": false}')


def test_parse_structured_response_accepts_typed_payload():
    payload = VLMStructuredResponse(
        polyp_detected=False,
        confidence_score=90,
        justification="Se ve un gato",
    )
    parsed = parse_structured_response(payload)
    assert parsed["polyp_detected"] is False
    assert parsed["confidence_score"] == 90


def test_resolve_model_non_interactive_uses_first_detected(monkeypatch):
    monkeypatch.setattr("src.scripts.test_inference.get_installed_models", lambda: ["model-a", "model-b"])
    assert resolve_model(model_arg=None, interactive=False) == "model-a"


def test_resolve_model_non_interactive_fails_without_models(monkeypatch):
    monkeypatch.setattr("src.scripts.test_inference.get_installed_models", lambda: [])
    with pytest.raises(RuntimeError):
        resolve_model(model_arg=None, interactive=False)


def test_test_cases_has_multiple_images_and_neutral_names():
    assert len(TEST_CASES) >= 4

    labels = [case["label"] for case in TEST_CASES]
    assert labels.count("cat") >= 2
    assert labels.count("dog") >= 2

    for case in TEST_CASES:
        path = case["path"].lower()
        assert "cat" not in path
        assert "dog" not in path


def test_run_smoke_test_preloads_once_and_unloads_once(monkeypatch):
    calls = {"preload": 0, "inference": 0, "unload": 0}

    class FakeLoader:
        def __init__(self, model_path, verbose=False):
            self.model_path = model_path
            self.verbose = verbose

        def preload_model(self):
            calls["preload"] += 1

        def inference(self, image_path, prompt):
            calls["inference"] += 1
            return '{"polyp_detected": false, "confidence_score": 90, "justification": "Se ve un gato"}'

        def unload_model(self):
            calls["unload"] += 1

    monkeypatch.setattr("src.scripts.test_inference.VLMLoader", FakeLoader)

    cases = [
        {"id": "sample_01", "label": "cat", "path": "a.jpg"},
        {"id": "sample_02", "label": "cat", "path": "b.jpg"},
    ]

    code = run_smoke_test("fake-model", cases)
    assert code == 0
    assert calls["preload"] == 1
    assert calls["inference"] == 2
    assert calls["unload"] == 1


def test_run_smoke_test_unloads_even_if_inference_fails(monkeypatch):
    calls = {"preload": 0, "unload": 0}

    class FakeLoader:
        def __init__(self, model_path, verbose=False):
            self.model_path = model_path

        def preload_model(self):
            calls["preload"] += 1

        def inference(self, image_path, prompt):
            raise RuntimeError("boom")

        def unload_model(self):
            calls["unload"] += 1

    monkeypatch.setattr("src.scripts.test_inference.VLMLoader", FakeLoader)

    with pytest.raises(RuntimeError):
        run_smoke_test("fake-model", [{"id": "sample_01", "label": "cat", "path": "a.jpg"}])

    assert calls["preload"] == 1
    assert calls["unload"] == 1


def test_run_smoke_test_executes_model_load_unload_cycle(monkeypatch):
    calls = {"load": 0, "unload": 0, "inference": 0}

    class FakeLoader:
        def __init__(self, model_path, verbose=False):
            self.model_path = model_path
            self.verbose = verbose

        def load_model(self):
            calls["load"] += 1

        def preload_model(self):
            self.load_model()

        def inference(self, image_path, prompt):
            calls["inference"] += 1
            return '{"polyp_detected": false, "confidence_score": 90, "justification": "Se ve un perro"}'

        def unload_model(self):
            calls["unload"] += 1

    monkeypatch.setattr("src.scripts.test_inference.VLMLoader", FakeLoader)

    cases = [{"id": "sample_01", "label": "dog", "path": "a.jpg"}]
    code = run_smoke_test("fake-model", cases)

    assert code == 0
    assert calls["load"] == 1
    assert calls["inference"] == 1
    assert calls["unload"] == 1
