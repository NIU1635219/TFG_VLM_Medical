from src.utils.models_ui.lms_menu_helpers import option_is_local


def test_option_is_local_matches_exact_indexed_signature():
    entry = {"indexed_model_identifier": "openbmb/minicpm-v-2_6-gguf/minicpm-v-2_6-q8_0.gguf"}
    local = {
        "openbmb/minicpm-v-2_6-gguf/minicpm-v-2_6-q8_0.gguf",
        "c:/models/minicpm-v-2_6-q8_0.gguf",
    }

    assert option_is_local(entry, local, model_hint="minicpm-v-2_6") is True


def test_option_is_local_does_not_cross_match_generic_filename_between_models():
    entry = {"indexed_model_identifier": "openbmb/minicpm-v-2_6-gguf/ggml-model-q8_0.gguf"}
    local = {
        "openbmb/minicpm-v-4_5-gguf/ggml-model-q8_0.gguf",
        "c:/models/ggml-model-q8_0.gguf",
    }

    assert option_is_local(entry, local, model_hint="minicpm-v-2_6") is False


def test_option_is_local_accepts_generic_filename_when_model_hint_matches():
    entry = {"indexed_model_identifier": "openbmb/minicpm-v-2_6-gguf/ggml-model-q8_0.gguf"}
    local = {
        "openbmb/minicpm-v-2_6-gguf/ggml-model-q8_0.gguf",
        "c:/models/ggml-model-q8_0.gguf",
    }

    assert option_is_local(entry, local, model_hint="minicpm-v-2_6") is True
