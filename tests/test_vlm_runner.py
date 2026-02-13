import pytest
from unittest.mock import MagicMock, patch, ANY
from src.inference.vlm_runner import VLMLoader

# Constants
FAKE_MODEL_TAG = "fake-model:latest"
FAKE_IMAGE_PATH = "data/raw/fake_image.jpg"

@pytest.fixture
def mock_ollama():
    with patch("src.inference.vlm_runner.ollama") as mock:
        yield mock

@pytest.fixture
def loader(mock_ollama):
    # Setup default behavior for ollama module if needed
    return VLMLoader(model_path=FAKE_MODEL_TAG, verbose=True)

def test_init_sets_attributes():
    loader = VLMLoader(model_path=FAKE_MODEL_TAG)
    assert loader.model_tag == FAKE_MODEL_TAG
    assert loader.verbose is False

def test_load_model_checks_list(loader, mock_ollama):
    # Mock list response
    # Ollama list returns an object with 'models' attribute which is a list of objects with 'model' attribute
    mock_model = MagicMock()
    mock_model.model = FAKE_MODEL_TAG
    
    mock_response = MagicMock()
    mock_response.models = [mock_model]
    
    mock_ollama.list.return_value = mock_response

    loader.load_model()
    
    # Verify list called
    mock_ollama.list.assert_called_once()
    # Verify pull NOT called because model exists
    mock_ollama.pull.assert_not_called()

def test_load_model_pulls_if_missing(loader, mock_ollama):
    # Mock list response checks (empty or different model)
    mock_model = MagicMock()
    mock_model.model = "other-model:latest"
    
    mock_response = MagicMock()
    mock_response.models = [mock_model]
    
    mock_ollama.list.return_value = mock_response

    loader.load_model()
    
    # Verify pull called
    mock_ollama.pull.assert_called_once_with(FAKE_MODEL_TAG, stream=True)

def test_inference_calls_chat(loader, mock_ollama):
    # Setup mock chat response
    mock_ollama.chat.return_value = {
        'message': {'content': 'Descripción de prueba'}
    }
    
    with patch("os.path.exists", return_value=True):
        result = loader.inference(FAKE_IMAGE_PATH, "Prompt de prueba")
    
    assert result == "Descripción de prueba"
    
    # Verify chat call arguments
    mock_ollama.chat.assert_called_once()
    call_args = mock_ollama.chat.call_args
    assert call_args.kwargs['model'] == FAKE_MODEL_TAG
    messages = call_args.kwargs['messages']
    assert len(messages) == 2 # System + User
    assert messages[0]['role'] == 'system'
    assert messages[1]['role'] == 'user'
    assert messages[1]['content'] == "Prompt de prueba"
    assert messages[1]['images'] == [FAKE_IMAGE_PATH]


def test_inference_supports_object_response(loader, mock_ollama):
    mock_message = MagicMock()
    mock_message.content = "Respuesta objeto"

    mock_response = MagicMock()
    mock_response.message = mock_message

    mock_ollama.chat.return_value = mock_response

    with patch("os.path.exists", return_value=True):
        result = loader.inference(FAKE_IMAGE_PATH, "Prompt de prueba")

    assert result == "Respuesta objeto"

def test_inference_raises_file_not_found(loader):
    with patch("os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            loader.inference("ghost.jpg", "Prompt")


def test_preload_model_warms_with_keep_alive(loader, mock_ollama):
    mock_model = MagicMock()
    mock_model.model = FAKE_MODEL_TAG
    mock_response = MagicMock()
    mock_response.models = [mock_model]
    mock_ollama.list.return_value = mock_response
    mock_ollama.chat.return_value = {'message': {'content': 'ok'}}

    loader.preload_model(keep_alive="10m")

    assert mock_ollama.chat.call_count == 1
    chat_kwargs = mock_ollama.chat.call_args.kwargs
    assert chat_kwargs["model"] == FAKE_MODEL_TAG
    assert chat_kwargs["keep_alive"] == "10m"


def test_unload_model_requests_keep_alive_zero(loader, mock_ollama):
    mock_ollama.chat.return_value = {'message': {'content': 'ok'}}

    loader.unload_model()

    mock_ollama.chat.assert_called_once()
    chat_kwargs = mock_ollama.chat.call_args.kwargs
    assert chat_kwargs["model"] == FAKE_MODEL_TAG
    assert chat_kwargs["keep_alive"] == 0
