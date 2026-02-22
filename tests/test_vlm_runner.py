import pytest
from unittest.mock import MagicMock, mock_open, patch
from src.inference.vlm_runner import VLMLoader, VLMStructuredResponse

# Constants
FAKE_MODEL_TAG = "fake-model:latest"
FAKE_IMAGE_PATH = "data/raw/fake_image.jpg"

@pytest.fixture
def mock_lms_sdk():
    with patch("src.inference.vlm_runner.lms") as mock_lms:
        mock_model = MagicMock()
        mock_client = MagicMock()
        mock_client.llm.model.return_value = mock_model
        mock_lms.Client.return_value = mock_client
        mock_lms.llm.return_value = mock_model

        loaded_model = MagicMock()
        loaded_model.key = FAKE_MODEL_TAG
        mock_lms.list_loaded_models.return_value = [loaded_model]

        yield mock_lms, mock_model

@pytest.fixture
def loader(mock_lms_sdk):
    _mock_lms, _mock_model = mock_lms_sdk
    return VLMLoader(model_path=FAKE_MODEL_TAG, verbose=True)

def test_init_sets_attributes():
    with patch("src.inference.vlm_runner.lms"):
        instance = VLMLoader(model_path=FAKE_MODEL_TAG)
        assert instance.model_tag == FAKE_MODEL_TAG
        assert instance.verbose is False


def test_init_accepts_optional_host_and_token():
    with patch("src.inference.vlm_runner.lms"):
        instance = VLMLoader(
            model_path=FAKE_MODEL_TAG,
            server_api_host="localhost:1234",
            api_token="secret-token",
        )
        assert instance.server_api_host == "localhost:1234"
        assert instance.api_token == "secret-token"

def test_load_model_creates_lms_model_handle(loader, mock_lms_sdk):
    mock_lms, _mock_model = mock_lms_sdk
    loader.load_model()
    mock_lms.Client.assert_called_once()
    mock_lms.Client.return_value.llm.model.assert_called_once_with(FAKE_MODEL_TAG)


def test_load_model_raises_if_sdk_fails(loader, mock_lms_sdk):
    mock_lms, _mock_model = mock_lms_sdk
    mock_lms.Client.side_effect = RuntimeError("client-boom")
    mock_lms.llm.side_effect = RuntimeError("boom")
    mock_lms.llm.load.side_effect = RuntimeError("boom-fallback")
    with pytest.raises(RuntimeError):
        loader.load_model()


def test_inference_returns_typed_structured_response(loader, mock_lms_sdk):
    _mock_lms, mock_model = mock_lms_sdk
    mock_model.respond.return_value = '{"polyp_detected": false, "confidence_score": 91, "justification": "No se observan pólipos."}'

    with patch("os.path.isfile", return_value=True), patch("builtins.open", mock_open(read_data=b"image-bytes")):
        result = loader.inference(FAKE_IMAGE_PATH, "Prompt de prueba")

    assert isinstance(result, VLMStructuredResponse)
    assert result.polyp_detected is False
    assert result.confidence_score == 91


def test_inference_uses_parsed_when_available(loader, mock_lms_sdk):
    _mock_lms, mock_model = mock_lms_sdk
    mock_response = MagicMock()
    mock_response.parsed = {
        "polyp_detected": False,
        "confidence_score": 92,
        "justification": "Sin pólipos.",
    }
    mock_model.respond.return_value = mock_response

    with patch("os.path.isfile", return_value=True), patch("builtins.open", mock_open(read_data=b"image-bytes")):
        result = loader.inference(FAKE_IMAGE_PATH, "Prompt de prueba")

    assert isinstance(result, VLMStructuredResponse)
    assert result.confidence_score == 92


def test_inference_uses_chat_history_with_images(loader, mock_lms_sdk):
    mock_lms, mock_model = mock_lms_sdk
    mock_model.respond.return_value = '{"polyp_detected": false, "confidence_score": 75, "justification": "Sin pólipos."}'

    with patch("os.path.isfile", return_value=True), patch("builtins.open", mock_open(read_data=b"image-bytes")):
        loader.inference(FAKE_IMAGE_PATH, "Prompt de prueba", temperature=0.5)

    first_call_kwargs = mock_model.respond.call_args_list[0].kwargs
    first_call_args = mock_model.respond.call_args_list[0].args

    assert len(first_call_args) == 1
    chat_payload = first_call_args[0]
    assert chat_payload is mock_lms.Chat.return_value
    mock_lms.Chat.assert_called_once()
    mock_lms.Chat.return_value.add_user_message.assert_called_once()
    add_args = mock_lms.Chat.return_value.add_user_message.call_args
    assert add_args.args[0]
    assert add_args.kwargs.get("images")
    assert first_call_kwargs.get("config") == {"temperature": 0.5}
    assert first_call_kwargs.get("response_format") is VLMStructuredResponse


def test_inference_retries_with_temperature_when_config_not_supported(loader, mock_lms_sdk):
    _mock_lms, mock_model = mock_lms_sdk

    def fake_respond(*args, **kwargs):
        if "config" in kwargs:
            raise TypeError("LLM.respond() got an unexpected keyword argument 'config'")
        if "temperature" in kwargs:
            return '{"polyp_detected": false, "confidence_score": 80, "justification": "Sin hallazgos."}'
        return '{"polyp_detected": false, "confidence_score": 80, "justification": "Sin hallazgos."}'

    mock_model.respond.side_effect = fake_respond

    with patch("os.path.isfile", return_value=True), patch("builtins.open", mock_open(read_data=b"image-bytes")):
        result = loader.inference(FAKE_IMAGE_PATH, "Prompt de prueba", temperature=0.3)

    assert isinstance(result, VLMStructuredResponse)
    assert mock_model.respond.call_count == 2
    assert mock_model.respond.call_args_list[0].kwargs.get("config") == {"temperature": 0.3}
    assert mock_model.respond.call_args_list[1].kwargs.get("temperature") == 0.3
    assert mock_model.respond.call_args_list[1].kwargs.get("response_format") is VLMStructuredResponse


def test_inference_retries_without_temperature_when_not_supported(loader, mock_lms_sdk):
    _mock_lms, mock_model = mock_lms_sdk

    def fake_respond(*args, **kwargs):
        if "config" in kwargs:
            raise TypeError("LLM.respond() got an unexpected keyword argument 'config'")
        if "temperature" in kwargs:
            raise TypeError("LLM.respond() got an unexpected keyword argument 'temperature'")
        return '{"polyp_detected": false, "confidence_score": 80, "justification": "Sin hallazgos."}'

    mock_model.respond.side_effect = fake_respond

    with patch("os.path.isfile", return_value=True), patch("builtins.open", mock_open(read_data=b"image-bytes")):
        result = loader.inference(FAKE_IMAGE_PATH, "Prompt de prueba", temperature=0.3)

    assert isinstance(result, VLMStructuredResponse)
    assert mock_model.respond.call_count == 3
    assert mock_model.respond.call_args_list[0].kwargs.get("config") == {"temperature": 0.3}
    assert mock_model.respond.call_args_list[1].kwargs.get("temperature") == 0.3
    assert mock_model.respond.call_args_list[2].kwargs.get("response_format") is VLMStructuredResponse

def test_inference_raises_file_not_found(loader):
    with patch("os.path.isfile", return_value=False):
        with pytest.raises(FileNotFoundError):
            loader.inference("ghost.jpg", "Prompt")


def test_preload_model_warms_up(loader, mock_lms_sdk):
    _mock_lms, mock_model = mock_lms_sdk
    mock_model.respond.return_value = "ok"

    loader.preload_model(keep_alive="10m")

    assert mock_model.respond.call_count >= 1


def test_unload_model_noop(loader):
    loader.load_model()
    loader.unload_model()
    assert loader._loaded_model is None


def test_inference_raises_when_json_is_invalid(loader, mock_lms_sdk):
    _mock_lms, mock_model = mock_lms_sdk
    mock_model.respond.return_value = "texto no json"

    with patch("os.path.isfile", return_value=True), patch("builtins.open", mock_open(read_data=b"image-bytes")):
        with pytest.raises(RuntimeError):
            loader.inference(FAKE_IMAGE_PATH, "Prompt")
