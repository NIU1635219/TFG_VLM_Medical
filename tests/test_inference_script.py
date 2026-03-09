import pytest

from src.inference.vlm_runner import GenericObjectDetection

from src.scripts.test_inference import (
    TEST_CASES,
    contains_any_keyword,
    get_installed_models,
    normalize_text,
    parse_lms_ls_output,
    parse_structured_response,
    run_smoke_test,
    resolve_model,
    validate_response,
)


def test_parse_lms_ls_output_extracts_model_tags():
    """Verifica la extracción correcta de tags de modelos desde la salida tabular de `lms ls`."""
    raw = (
        "MODEL                                SIZE      MODIFIED\n"
        "openbmb/minicpm-v4.5:8b              4.8 GB    2 hours ago\n" 
        "qwen3-vl:latest                      5.1 GB    1 day ago\n"
    )
    models = parse_lms_ls_output(raw)
    assert models == ["openbmb/minicpm-v4.5:8b", "qwen3-vl:latest"]


def test_normalize_text_removes_accents_and_lowercases():
    """Valida que la normalización elimine acentos y convierta a minúsculas."""
    assert normalize_text("GatÓ PERRÓ") == "gato perro"


def test_contains_any_keyword_with_variants():
    """Prueba que se detecten palabras clave dentro de un texto, con normalización."""
    response = "En la imagen aparece claramente un felino doméstico."
    assert contains_any_keyword(response, ["gato", "felino"]) is True


def test_contains_any_keyword_no_false_positive_substring():
    """Regresión: 'cat' NO debe coincidir dentro de palabras como 'significativa' o 'indica'."""
    text = "La imagen muestra una montaña con picos cubiertos de nieve y áreas verdes, lo que indica una altitud significativa."
    assert contains_any_keyword(text, ["cat"]) is False
    assert contains_any_keyword(text, ["cat", "gato", "perro"]) is False


def test_validate_response_none_no_false_positive_in_significativa():
    """Regresión: label 'none' no debe fallar por 'cat' dentro de 'significativa'."""
    ok, _ = validate_response(
        "none",
        '{"object_detected": "montaña", "confidence_score": 95, "justification": "La imagen muestra una montaña con picos cubiertos de nieve y áreas verdes, lo que indica una altitud significativa."}',
    )
    assert ok is True


def test_validate_response_dog_via_object_detected_field():
    """Regresión: si object_detected='perro' pero justification no menciona perro, debe PASAR."""
    ok, _ = validate_response(
        "dog",
        '{"object_detected": "perro", "confidence_score": 95, "justification": "El animal tiene pelaje blanco y una forma característica de orejas y hocico."}',
    )
    assert ok is True



    """Confirma que `validate_response` distinga aciertos y fallos basándose en keywords."""
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
    """Asegura que una respuesta vacía del modelo genere un mensaje de error claro."""
    fail, message = validate_response("cat", "")
    assert fail is False
    assert "respuesta vacía" in message


def test_parse_structured_response_requires_fields():
    """Verifica que se lance ValueError si faltan campos obligatorios en el JSON."""
    with pytest.raises(ValueError):
        parse_structured_response('{"object_detected": "cat"}')


def test_parse_structured_response_accepts_typed_payload():
    """Comprueba que `parse_structured_response` acepte directamente objetos Pydantic/dataclass."""
    payload = GenericObjectDetection(
        object_detected="cat",
        confidence_score=90,
        justification="Se ve un gato",
    )
    parsed = parse_structured_response(payload)
    assert parsed["object_detected"] == "cat"
    assert parsed["confidence_score"] == 90


def test_resolve_model_non_interactive_uses_first_detected(monkeypatch):
    """Prueba que, en modo no interactivo, se seleccione el primer modelo disponible."""
    monkeypatch.setattr("src.scripts.test_inference.get_installed_models", lambda: ["model-a", "model-b"])
    assert resolve_model(model_arg=None) == "model-a"


def test_resolve_model_non_interactive_fails_without_models(monkeypatch):
    """Valida que se lance error si no hay modelos y no se permite interacción."""
    monkeypatch.setattr("src.scripts.test_inference.get_installed_models", lambda: [])
    with pytest.raises(RuntimeError):
        resolve_model(model_arg=None)


def test_test_cases_has_multiple_images_and_neutral_names():
    """Verifica la integridad y variedad de los casos de prueba (smoke tests)."""
    assert len(TEST_CASES) >= 4

    labels = [case["label"] for case in TEST_CASES]
    assert labels.count("cat") >= 2
    assert labels.count("dog") >= 2

    for case in TEST_CASES:
        path = case["path"].lower()
        assert "cat" not in path
        assert "dog" not in path


def test_run_smoke_test_preloads_once_and_unloads_once(monkeypatch):
    """Confirma que el ciclo de vida del modelo (cargar -> inferencias -> descargar) sea correcto."""
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
    """Asegura que el modelo se descargue (unload) incluso si ocurre una excepción durante la inferencia."""
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
    """Verifica nuevamente el ciclo de carga/descarga en un escenario simple."""
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


# ---------------------------------------------------------------------------
# Tests: get_installed_models (variant-aware listing)
# ---------------------------------------------------------------------------

def test_get_installed_models_delegates_to_lms_variant_listing(monkeypatch):
    """get_installed_models devuelve las variantes de _lms_get_installed_models cuando hay resultados."""
    variant_keys = ["qwen/qwen3.5-9b@q4_k_m", "qwen/qwen3.5-9b@q8_0", "opengvlab_internvl3_5-14b"]
    monkeypatch.setattr("src.scripts.test_inference._lms_get_installed_models", lambda: variant_keys)
    result = get_installed_models()
    assert result == variant_keys


def test_get_installed_models_falls_back_when_lms_returns_empty(monkeypatch):
    """Si _lms_get_installed_models devuelve vacío, se usa el fallback (modelos cargados vía SDK)."""
    monkeypatch.setattr("src.scripts.test_inference._lms_get_installed_models", lambda: [])
    monkeypatch.setattr("src.scripts.test_inference._lms_loaded_model_keys", lambda: ["loaded-model"])
    result = get_installed_models()
    assert result == ["loaded-model"]


def test_resolve_model_uses_variant_key_from_installed(monkeypatch):
    """resolve_model selecciona el primer modelo con sufijo @variante en modo automático."""
    monkeypatch.setattr(
        "src.scripts.test_inference.get_installed_models",
        lambda: ["opengvlab_internvl3_5-8b@q4_k_m", "opengvlab_internvl3_5-8b@q8_0"],
    )
    result = resolve_model(model_arg=None)
    assert result == "opengvlab_internvl3_5-8b@q4_k_m"


def test_resolve_model_passes_through_explicit_variant_arg():
    """Si se pasa un model_arg con sufijo @variante, se devuelve tal cual sin consultar la lista."""
    result = resolve_model(model_arg="qwen/qwen3.5-9b@q8_0")
    assert result == "qwen/qwen3.5-9b@q8_0"
