import pytest
from unittest.mock import MagicMock, mock_open, patch
from src.inference.vlm_runner import VLMLoader, GenericObjectDetection

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
    """Verifica que el inicializador establezca correctamente los atributos básicos."""
    with patch("src.inference.vlm_runner.lms"):
        instance = VLMLoader(model_path=FAKE_MODEL_TAG)
        assert instance.model_tag == FAKE_MODEL_TAG
        assert instance.verbose is False


def test_init_accepts_optional_host_and_token():
    """Comprueba que se puedan pasar host y token opcionales al constructor."""
    with patch("src.inference.vlm_runner.lms"):
        instance = VLMLoader(
            model_path=FAKE_MODEL_TAG,
            server_api_host="localhost:1234",
            api_token="secret-token",
        )
        assert instance.server_api_host == "localhost:1234"
        assert instance.api_token == "secret-token"

def test_load_model_creates_lms_model_handle(loader, mock_lms_sdk):
    """Asegura que `load_model` instancie el cliente y cargue el modelo usando el SDK,
    pasando contextLength en la configuración."""
    mock_lms, _mock_model = mock_lms_sdk
    loader.load_model()
    mock_lms.Client.assert_called_once()
    mock_lms.Client.return_value.llm.model.assert_called_once()
    call_args = mock_lms.Client.return_value.llm.model.call_args
    assert call_args.args[0] == FAKE_MODEL_TAG
    assert call_args.kwargs.get("config", {}).get("contextLength", 0) > 0


def test_load_model_passes_custom_context_window(mock_lms_sdk):
    """Verifica que n_ctx se transmita correctamente como contextLength al SDK."""
    mock_lms, _mock_model = mock_lms_sdk
    loader = VLMLoader(model_path=FAKE_MODEL_TAG)
    loader.load_model(n_ctx=16384)
    call_args = mock_lms.Client.return_value.llm.model.call_args
    assert call_args.kwargs.get("config", {}).get("contextLength") == 16384


def test_load_model_raises_if_sdk_fails(loader, mock_lms_sdk):
    """Valida que se propague una excepción si el SDK falla al cargar el modelo."""
    mock_lms, _mock_model = mock_lms_sdk
    mock_lms.Client.side_effect = RuntimeError("client-boom")
    mock_lms.llm.side_effect = RuntimeError("boom")
    mock_lms.llm.load.side_effect = RuntimeError("boom-fallback")
    with pytest.raises(RuntimeError):
        loader.load_model()


def test_inference_returns_typed_structured_response(loader, mock_lms_sdk):
    """Verifica que el método `inference` devuelva un objeto `GenericObjectDetection`."""
    _mock_lms, mock_model = mock_lms_sdk
    mock_model.respond.return_value = '{"object_detected": "cat", "confidence_score": 91, "justification": "Se ve un gato."}'

    with patch("os.path.isfile", return_value=True), patch("builtins.open", mock_open(read_data=b"image-bytes")):
        result = loader.inference(FAKE_IMAGE_PATH, "Prompt de prueba")

    assert isinstance(result, GenericObjectDetection)
    assert result.object_detected == "cat"
    assert result.confidence_score == 91


def test_inference_uses_parsed_when_available(loader, mock_lms_sdk):
    """Comprueba que se use el campo `parsed` de la respuesta si está disponible."""
    _mock_lms, mock_model = mock_lms_sdk
    mock_response = MagicMock()
    mock_response.parsed = {
        "object_detected": "dog",
        "confidence_score": 92,
        "justification": "Se ve un perro.",
    }
    mock_model.respond.return_value = mock_response

    with patch("os.path.isfile", return_value=True), patch("builtins.open", mock_open(read_data=b"image-bytes")):
        result = loader.inference(FAKE_IMAGE_PATH, "Prompt de prueba")

    assert isinstance(result, GenericObjectDetection)
    assert result.confidence_score == 92


def test_inference_uses_chat_history_with_images(loader, mock_lms_sdk):
    """Confirma que se construya correctamente el historial de chat incluyendo imágenes."""
    mock_lms, mock_model = mock_lms_sdk
    mock_model.respond.return_value = '{"object_detected": "cat", "confidence_score": 75, "justification": "Se ve un gato."}'

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
    assert first_call_kwargs.get("response_format") is GenericObjectDetection


def test_inference_retries_with_temperature_when_config_not_supported(loader, mock_lms_sdk):
    """Prueba el mecanismo de reintento si el argumento `config` falla (fallback a `temperature`)."""
    _mock_lms, mock_model = mock_lms_sdk

    def fake_respond(*args, **kwargs):
        if "config" in kwargs:
            raise TypeError("LLM.respond() got an unexpected keyword argument 'config'")
        if "temperature" in kwargs:
            return '{"object_detected": "mountain", "confidence_score": 80, "justification": "Se ve una montaña."}'
        return '{"object_detected": "mountain", "confidence_score": 80, "justification": "Se ve una montaña."}'

    mock_model.respond.side_effect = fake_respond

    with patch("os.path.isfile", return_value=True), patch("builtins.open", mock_open(read_data=b"image-bytes")):
        result = loader.inference(FAKE_IMAGE_PATH, "Prompt de prueba", temperature=0.3)

    assert isinstance(result, GenericObjectDetection)
    assert mock_model.respond.call_count == 2
    assert mock_model.respond.call_args_list[0].kwargs.get("config") == {"temperature": 0.3}
    assert mock_model.respond.call_args_list[1].kwargs.get("temperature") == 0.3
    assert mock_model.respond.call_args_list[1].kwargs.get("response_format") is GenericObjectDetection


def test_inference_retries_without_temperature_when_not_supported(loader, mock_lms_sdk):
    """Prueba el fallback final sin temperatura si tanto `config` como `temperature` fallan."""
    _mock_lms, mock_model = mock_lms_sdk

    def fake_respond(*args, **kwargs):
        if "config" in kwargs:
            raise TypeError("LLM.respond() got an unexpected keyword argument 'config'")
        if "temperature" in kwargs:
            raise TypeError("LLM.respond() got an unexpected keyword argument 'temperature'")
        return '{"object_detected": "river", "confidence_score": 80, "justification": "Se ve un río."}'

    mock_model.respond.side_effect = fake_respond

    with patch("os.path.isfile", return_value=True), patch("builtins.open", mock_open(read_data=b"image-bytes")):
        result = loader.inference(FAKE_IMAGE_PATH, "Prompt de prueba", temperature=0.3)

    assert isinstance(result, GenericObjectDetection)
    assert mock_model.respond.call_count == 3
    assert mock_model.respond.call_args_list[0].kwargs.get("config") == {"temperature": 0.3}
    assert mock_model.respond.call_args_list[1].kwargs.get("temperature") == 0.3
    assert mock_model.respond.call_args_list[2].kwargs.get("response_format") is GenericObjectDetection

def test_inference_raises_file_not_found(loader):
    """Verifica que se lance `FileNotFoundError` si la imagen no existe."""
    with patch("os.path.isfile", return_value=False):
        with pytest.raises(FileNotFoundError):
            loader.inference("ghost.jpg", "Prompt")


def test_preload_model_warms_up(loader, mock_lms_sdk):
    """Asegura que `preload_model` realice una llamada de calentamiento al modelo."""
    _mock_lms, mock_model = mock_lms_sdk
    mock_model.respond.return_value = "ok"

    loader.preload_model(keep_alive="10m")

    assert mock_model.respond.call_count >= 1


def test_unload_model_noop(loader):
    """Confirma que `unload_model` limpie la referencia al modelo cargado."""
    loader.load_model()
    loader.unload_model()
    assert loader._loaded_model is None


def test_inference_raises_when_json_is_invalid(loader, mock_lms_sdk):
    """Valida que se lance error si la respuesta del modelo no es un JSON válido."""
    _mock_lms, mock_model = mock_lms_sdk
    mock_model.respond.return_value = "texto no json"

    with patch("os.path.isfile", return_value=True), patch("builtins.open", mock_open(read_data=b"image-bytes")):
        with pytest.raises(RuntimeError):
            loader.inference(FAKE_IMAGE_PATH, "Prompt")


def test_inference_does_not_send_native_thinking_config(mock_lms_sdk):
    """La inferencia estructurada no debe activar chatTemplateKwargs de thinking nativo."""
    _mock_lms, mock_model = mock_lms_sdk
    mock_model.respond.return_value = '{"object_detected": "polyp", "confidence_score": 88, "justification": "Detectado."}'

    loader = VLMLoader(model_path="any-model")
    loader._loaded_model = mock_model

    with patch("os.path.isfile", return_value=True), patch("builtins.open", mock_open(read_data=b"image-bytes")):
        loader.inference(FAKE_IMAGE_PATH, "Prompt")

    call_config = mock_model.respond.call_args_list[0].kwargs.get("config", {})
    assert "chatTemplateKwargs" not in call_config
