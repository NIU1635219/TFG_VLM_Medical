from argparse import Namespace
from types import SimpleNamespace

from src.scripts import test_response_inspector as inspector


def test_resolve_schema_defaults_to_generic_when_reasoning_is_requested():
    schema_name, schema_cls = inspector.resolve_schema(None, True)

    assert schema_name == "GenericObjectDetectionWithReasoning"
    assert schema_cls is not None


def test_run_inspection_autofills_model_image_and_prompt(monkeypatch):
    monkeypatch.setattr(inspector, "resolve_default_model", lambda explicit_model: ("auto-model", True))
    monkeypatch.setattr(inspector, "resolve_default_image", lambda explicit_image: ("C:/demo/image.jpg", True))
    monkeypatch.setattr(inspector.os.path, "isfile", lambda path: True)

    class FakeLoader:
        def __init__(self, model_path, verbose=False, server_api_host=None, api_token=None):
            self.model_path = model_path
            self._loaded_model = object()

        def load_model(self):
            return None

        def unload_model(self):
            return None

        def _build_structured_instruction(self, prompt, schema=None):
            return f"STRUCTURED::{prompt}"

        def _build_multimodal_history(self, structured_text, image_path):
            return {"history": [structured_text, image_path]}

        def _respond_with_config_compat(self, model_handle, payload, temperature, schema=None):
            return SimpleNamespace(
                content='{"object_detected":"cat"}',
                parsed={"object_detected": "cat", "confidence_score": 88, "justification": "ok", "reasoning": "pasos"},
                stats=SimpleNamespace(
                    stop_reason="eosFound",
                    tokens_per_second=25.0,
                    time_to_first_token_sec=1.5,
                    prompt_tokens_count=120,
                    predicted_tokens_count=30,
                    total_tokens_count=150,
                    num_gpu_layers=-1.0,
                ),
                model_info=SimpleNamespace(
                    model_key="auto-model",
                    architecture="qwen35",
                    params_string="9B",
                    display_name="Qwen Demo",
                ),
                structured=True,
            )

        def _object_to_dict(self, value):
            if value is None:
                return {}
            if isinstance(value, dict):
                return value
            return {
                key: getattr(value, key)
                for key in dir(value)
                if not key.startswith("_") and not callable(getattr(value, key))
            }

        def _fetch_model_resources(self):
            return {}

        def _extract_response_text(self, response):
            return response.content

    monkeypatch.setattr(inspector, "VLMLoader", FakeLoader)

    payload = inspector.run_inspection(
        Namespace(
            model=None,
            image=None,
            prompt=None,
            schema=None,
            structured=True,
            with_reasoning=True,
            temperature=0.0,
            server_api_host=None,
            api_token=None,
            output=None,
            print_json=False,
        )
    )

    assert payload["request"]["model_id"] == "auto-model"
    assert payload["request"]["model_auto_selected"] is True
    assert payload["request"]["image_auto_selected"] is True
    assert payload["request"]["schema_name"] == "GenericObjectDetectionWithReasoning"
    assert payload["response"]["structured"] is True
    assert payload["response"]["stats"]["tokens_per_second"] == 25.0
    assert payload["response"]["model_info"]["model_key"] == "auto-model"


def test_print_summary_renders_visual_sections(capsys):
    payload = {
        "request": {
            "model_id": "auto-model",
            "model_auto_selected": True,
            "image_path": "C:/demo/image.jpg",
            "image_auto_selected": True,
            "schema_name": "GenericObjectDetectionWithReasoning",
        },
        "response": {
            "python_type": "PredictionResult",
            "structured": True,
            "public_attributes": {"content": "...", "stats": "...", "parsed": "..."},
            "text_extracted": "texto de prueba muy largo para validar el resumen visual",
            "model_info": {
                "model_key": "auto-model",
                "architecture": "qwen35",
                "params_string": "9B",
                "display_name": "Qwen Demo",
            },
            "stats": {
                "stop_reason": "eosFound",
                "tokens_per_second": 25.0,
                "time_to_first_token_sec": 1.5,
                "prompt_tokens_count": 120,
                "predicted_tokens_count": 30,
                "total_tokens_count": 150,
                "num_gpu_layers": -1.0,
            },
            "parsed": {
                "object_detected": "cat",
                "confidence_score": 88,
            },
        },
        "resources": {},
    }

    inspector.print_summary(payload)
    output = capsys.readouterr().out

    assert "LM STUDIO RESPONSE INSPECTOR" in output
    assert "SDK STATS" in output
    assert "PARSED PAYLOAD" in output
    assert "tokens_per_second" in output
    assert "object_detected" in output
    assert "REST RESOURCES" not in output


def test_print_summary_omits_empty_rows_and_sections(capsys):
    payload = {
        "request": {
            "model_id": "auto-model",
            "image_path": "C:/demo/image.jpg",
            "schema_name": None,
        },
        "response": {
            "python_type": "PredictionResult",
            "structured": False,
            "public_attributes": {},
            "text_extracted": "texto breve",
            "model_info": {"model_key": "auto-model"},
            "stats": {},
            "parsed": {},
        },
        "resources": {},
    }

    inspector.print_summary(payload)
    output = capsys.readouterr().out

    assert "REST RESOURCES" not in output
    assert "SDK STATS" not in output
    assert "Text preview" in output


def test_save_inspection_payload_writes_expected_file(tmp_path):
    payload = {"request": {"model_id": "demo-model"}, "response": {}, "resources": {}}

    output_path = inspector.save_inspection_payload(payload, str(tmp_path / "out.json"))

    assert output_path.endswith("out.json")
    assert (tmp_path / "out.json").exists()
