import json
import pytest
from unittest.mock import patch, MagicMock
from src.utils.models_ui.lms_models import (
    _normalize_model_ref,
    check_lms,
    get_server_status,
    _cli_ls_json,
    list_downloaded_models_with_variants,
    list_installed_variants_flat,
    get_installed_lms_models,
)

def test_normalize_model_ref():
    assert _normalize_model_ref("huggingface.co/lmstudio-community/model") == "lmstudio-community/model"
    assert _normalize_model_ref("lmstudio-community/model") == "lmstudio-community/model"
    assert _normalize_model_ref("") == ""

@patch('src.utils.models_ui.lms_models._run_lms_command')
def test_check_lms_installed(mock_run):
    mock_run.return_value = MagicMock(returncode=0)
    assert check_lms() is True

@patch('src.utils.models_ui.lms_models._run_lms_command')
def test_check_lms_not_installed(mock_run):
    mock_run.side_effect = Exception("Command not found")
    assert check_lms() is False

@patch('src.utils.models_ui.lms_models.check_lms')
@patch('src.utils.models_ui.lms_models._run_lms_command')
def test_get_server_status_running(mock_run, mock_check_lms):
    mock_check_lms.return_value = True
    mock_result = MagicMock()
    mock_result.stdout = "Server is running on port 1234"
    mock_result.stderr = ""
    mock_run.return_value = mock_result
    
    is_running, output = get_server_status()
    assert is_running is True
    assert "running" in output.lower()

@patch('src.utils.models_ui.lms_models.check_lms')
@patch('src.utils.models_ui.lms_models._run_lms_command')
def test_get_server_status_not_running(mock_run, mock_check_lms):
    mock_check_lms.return_value = True
    mock_result = MagicMock()
    mock_result.stdout = "Server is stopped"
    mock_result.stderr = ""
    mock_run.return_value = mock_result
    
    is_running, output = get_server_status()
    assert is_running is False
    assert "stopped" in output.lower()

@patch('src.utils.models_ui.lms_models.check_lms')
def test_get_server_status_no_lms(mock_check_lms):
    mock_check_lms.return_value = False
    is_running, output = get_server_status()
    assert is_running is False
    assert "lms CLI not found" in output


# ---------------------------------------------------------------------------
# JSON de lms ls real (fixture compartida)
# ---------------------------------------------------------------------------

LMS_LS_JSON = [
    {
        "type": "llm",
        "modelKey": "qwen/qwen3.5-9b",
        "displayName": "Qwen3.5 9B",
        "publisher": "qwen",
        "architecture": "qwen35",
        "paramsString": "9B",
        "quantization": {"name": "Q4_K_M", "bits": 4},
        "sizeBytes": 6548926907,
        "vision": True,
        "variants": ["qwen/qwen3.5-9b@q4_k_m", "qwen/qwen3.5-9b@q8_0"],
        "selectedVariant": "qwen/qwen3.5-9b@q4_k_m",
        "maxContextLength": 262144,
    },
    {
        "type": "llm",
        "modelKey": "minicpm-v-4_5@q4_k_m",
        "displayName": "MiniCPM-V Q4",
        "publisher": "openbmb",
        "architecture": "qwen3",
        "paramsString": "8.2B",
        "quantization": {"name": "Q4_K_M", "bits": 4},
        "sizeBytes": 6121827488,
        "vision": True,
        "variants": [],
        "selectedVariant": "minicpm-v-4_5@q4_k_m",
        "maxContextLength": 40960,
    },
    {
        "type": "llm",
        "modelKey": "opengvlab_internvl3_5-14b",
        "displayName": "InternVL3.5 14B",
        "publisher": "bartowski",
        "architecture": "qwen3",
        "paramsString": "14B",
        "quantization": {"name": "Q4_K_M", "bits": 4},
        "sizeBytes": 9707086976,
        "vision": True,
        "variants": [],
        "selectedVariant": "opengvlab_internvl3_5-14b",
        "maxContextLength": 40960,
    },
    {
        "type": "embedding",
        "modelKey": "text-embedding-nomic-embed-text-v1.5",
        "displayName": "Nomic Embed",
        "architecture": "nomic-bert",
    },
]


# ---------------------------------------------------------------------------
# Tests: _cli_ls_json
# ---------------------------------------------------------------------------

@patch('src.utils.models_ui.lms_models.check_lms', return_value=True)
@patch('src.utils.models_ui.lms_models._run_lms_command')
def test_cli_ls_json_returns_parsed_list(mock_run, _mock_check):
    """_cli_ls_json parsea correctamente la salida JSON de lms ls."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = json.dumps(LMS_LS_JSON)
    mock_run.return_value = mock_result

    result = _cli_ls_json()
    assert isinstance(result, list)
    assert len(result) == 4
    assert result[0]["modelKey"] == "qwen/qwen3.5-9b"


@patch('src.utils.models_ui.lms_models.check_lms', return_value=True)
@patch('src.utils.models_ui.lms_models._run_lms_command')
def test_cli_ls_json_returns_empty_on_nonzero_rc(mock_run, _mock_check):
    """_cli_ls_json devuelve [] si lms retorna código de error."""
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_run.return_value = mock_result

    assert _cli_ls_json() == []


@patch('src.utils.models_ui.lms_models.check_lms', return_value=False)
def test_cli_ls_json_returns_empty_when_no_lms(_mock_check):
    """_cli_ls_json devuelve [] si lms CLI no está disponible."""
    assert _cli_ls_json() == []


# ---------------------------------------------------------------------------
# Tests: list_downloaded_models_with_variants
# ---------------------------------------------------------------------------

@patch('src.utils.models_ui.lms_models._cli_ls_json', return_value=LMS_LS_JSON)
def test_list_downloaded_models_filters_embeddings(_mock_json):
    """list_downloaded_models_with_variants excluye registros de tipo embedding."""
    result = list_downloaded_models_with_variants()
    keys = [m["model_key"] for m in result]
    assert "text-embedding-nomic-embed-text-v1.5" not in keys
    assert len(result) == 3


@patch('src.utils.models_ui.lms_models._cli_ls_json', return_value=LMS_LS_JSON)
def test_list_downloaded_models_expands_variants(_mock_json):
    """qwen/qwen3.5-9b debe tener 2 variantes en su campo 'variants'."""
    result = list_downloaded_models_with_variants()
    qwen = next(m for m in result if m["model_key"] == "qwen/qwen3.5-9b")
    assert len(qwen["variants"]) == 2
    assert "qwen/qwen3.5-9b@q4_k_m" in qwen["variants"]
    assert "qwen/qwen3.5-9b@q8_0" in qwen["variants"]


@patch('src.utils.models_ui.lms_models._cli_ls_json', return_value=LMS_LS_JSON)
def test_list_downloaded_models_selected_variant(_mock_json):
    """selected_variant debe coincidir con el valor del JSON."""
    result = list_downloaded_models_with_variants()
    qwen = next(m for m in result if m["model_key"] == "qwen/qwen3.5-9b")
    assert qwen["selected_variant"] == "qwen/qwen3.5-9b@q4_k_m"


@patch('src.utils.models_ui.lms_models._cli_ls_json', return_value=LMS_LS_JSON)
def test_list_downloaded_models_metadata(_mock_json):
    """Verifica que arquitectura, vision y max_context se mapean correctamente."""
    result = list_downloaded_models_with_variants()
    qwen = next(m for m in result if m["model_key"] == "qwen/qwen3.5-9b")
    assert qwen["architecture"] == "qwen35"
    assert qwen["vision"] is True
    assert qwen["max_context"] == 262144


@patch('src.utils.models_ui.lms_models._cli_ls_json', return_value=[])
def test_list_downloaded_models_returns_empty_on_empty_json(_mock_json):
    """Si _cli_ls_json devuelve lista vacía, list_downloaded_models devuelve []."""
    assert list_downloaded_models_with_variants() == []


# ---------------------------------------------------------------------------
# Tests: get_installed_lms_models
# ---------------------------------------------------------------------------

@patch('src.utils.models_ui.lms_models.list_downloaded_models_with_variants', return_value=[
    {
        "model_key": "qwen/qwen3.5-9b",
        "variants": ["qwen/qwen3.5-9b@q4_k_m", "qwen/qwen3.5-9b@q8_0"],
        "selected_variant": "qwen/qwen3.5-9b@q4_k_m",
    },
    {
        "model_key": "opengvlab_internvl3_5-14b",
        "variants": [],
        "selected_variant": "opengvlab_internvl3_5-14b",
    },
])
def test_get_installed_lms_models_expands_variants(_mock_models):
    """get_installed_lms_models expande variantes y devuelve lista plana sin duplicados."""
    result = get_installed_lms_models()
    assert "qwen/qwen3.5-9b@q4_k_m" in result
    assert "qwen/qwen3.5-9b@q8_0" in result
    assert "opengvlab_internvl3_5-14b" in result
    # Sin duplicados
    assert len(result) == len(set(result))


@patch('src.utils.models_ui.lms_models.list_downloaded_models_with_variants', return_value=[
    {
        "model_key": "qwen/qwen3.5-9b",
        "variants": ["qwen/qwen3.5-9b@q4_k_m", "qwen/qwen3.5-9b@q8_0"],
        "selected_variant": "qwen/qwen3.5-9b@q4_k_m",
    },
])
def test_get_installed_lms_models_no_duplicates(_mock_models):
    """No aparece 'qwen/qwen3.5-9b' sin sufijo cuando ya hay variantes con @quant."""
    result = get_installed_lms_models()
    assert "qwen/qwen3.5-9b" not in result


@patch('src.utils.models_ui.lms_models.list_downloaded_models_with_variants', return_value=[])
@patch('src.utils.models_ui.lms_models.get_installed_models', return_value=["fallback-model"])
def test_get_installed_lms_models_fallback_to_cli(_mock_installed, _mock_variants):
    """Si list_downloaded_models_with_variants devuelve [], usa get_installed_models."""
    result = get_installed_lms_models()
    assert result == ["fallback-model"]


# ---------------------------------------------------------------------------
# Tests: list_installed_variants_flat
# ---------------------------------------------------------------------------

@patch('src.utils.models_ui.lms_models._cli_ls_json', return_value=LMS_LS_JSON)
def test_list_installed_variants_flat_resolves_quant_for_no_suffix_model(_mock_json):
    """opengvlab_internvl3_5-14b sin @ debe tener quantization=Q4_K_M desde JSON."""
    result = list_installed_variants_flat()
    internvl = next((m for m in result if m["model_key"] == "opengvlab_internvl3_5-14b"), None)
    assert internvl is not None
    assert internvl["quantization"] == "Q4_K_M"


@patch('src.utils.models_ui.lms_models._cli_ls_json', return_value=LMS_LS_JSON)
def test_list_installed_variants_flat_no_duplicates(_mock_json):
    """Sin variantes duplicadas en la lista plana."""
    result = list_installed_variants_flat()
    keys = [m["model_key"] for m in result]
    assert len(keys) == len(set(keys))


@patch('src.utils.models_ui.lms_models._cli_ls_json', return_value=LMS_LS_JSON)
def test_list_installed_variants_flat_expands_qwen_variants(_mock_json):
    """qwen/qwen3.5-9b genera dos entradas: una por @q4_k_m y otra por @q8_0."""
    result = list_installed_variants_flat()
    keys = [m["model_key"] for m in result]
    assert "qwen/qwen3.5-9b@q4_k_m" in keys
    assert "qwen/qwen3.5-9b@q8_0" in keys
    assert "qwen/qwen3.5-9b" not in keys


@patch('src.utils.models_ui.lms_models._cli_ls_json', return_value=LMS_LS_JSON)
def test_list_installed_variants_flat_quant_from_suffix_takes_precedence(_mock_json):
    """Para variantes con @q8_0 en el key, quantization es Q8_0 (no Q4_K_M del JSON padre)."""
    result = list_installed_variants_flat()
    q8 = next((m for m in result if m["model_key"] == "qwen/qwen3.5-9b@q8_0"), None)
    assert q8 is not None
    assert q8["quantization"] == "Q8_0"
