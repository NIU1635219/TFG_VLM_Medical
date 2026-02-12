import pytest
import os
from unittest.mock import MagicMock, patch
from src.inference.vlm_runner import VLMLoader

# Mockear paths para tests
FAKE_MODEL_PATH = "models/fake_model.gguf"
FAKE_IMAGE_PATH = "data/raw/fake_image.jpg"

@pytest.fixture
def mock_exists():
    with patch("os.path.exists") as mock:
        yield mock

def test_init_raises_error_if_model_not_found(mock_exists):
    """Debe lanzar FileNotFoundError si el modelo no existe."""
    mock_exists.return_value = False
    with pytest.raises(FileNotFoundError):
        VLMLoader(model_path="non_existent.gguf")

def test_init_success(mock_exists):
    """Debe inicializarse correctamente si el modelo existe."""
    mock_exists.return_value = True
    loader = VLMLoader(model_path=FAKE_MODEL_PATH)
    assert loader.model_path == FAKE_MODEL_PATH

def test_inference_raises_error_if_model_not_loaded(mock_exists):
    """Debe lanzar RuntimeError si se llama a inference antes de load_model."""
    mock_exists.return_value = True
    loader = VLMLoader(model_path=FAKE_MODEL_PATH)
    
    with patch("os.path.exists", return_value=True): # Image exists
        with pytest.raises(RuntimeError) as excinfo:
            loader.inference(FAKE_IMAGE_PATH, "Prompt")
    assert "Modelo no cargado" in str(excinfo.value)

def test_inference_raises_error_if_image_not_found(mock_exists):
    """Debe lanzar FileNotFoundError si la imagen no existe."""
    mock_exists.side_effect = lambda path: path == FAKE_MODEL_PATH # Model exists, image doesn't
    
    loader = VLMLoader(model_path=FAKE_MODEL_PATH)
    # Mocking internal loaded state specifically for this test logic if we executed partial code
    # But since strict TDD fails first, we expect implementation to check image before checking model loaded state or vice-versa.
    # We will simulate model is loaded to test image check.
    loader.llm = MagicMock() 

    with pytest.raises(FileNotFoundError):
        loader.inference("ghost_image.jpg", "Prompt")

@patch("src.inference.vlm_runner.MiniCPMv26ChatHandler")
@patch("src.inference.vlm_runner.Llama") # Patch de la clase externa
def test_load_model_calls_llama(mock_llama_class, mock_chat_handler, mock_exists):
    """Verifica que load_model instancie 'Llama' con los parámetros correctos."""
    mock_exists.return_value = True
    loader = VLMLoader(model_path=FAKE_MODEL_PATH)
    
    loader.load_model(n_ctx=4096, n_gpu_layers=33)
    
    mock_llama_class.assert_called_once()
    mock_chat_handler.assert_called_once()
    _, kwargs = mock_llama_class.call_args
    assert kwargs["model_path"] == FAKE_MODEL_PATH
    assert kwargs["n_ctx"] == 4096
    assert kwargs["n_gpu_layers"] == 33

@patch("src.inference.vlm_runner.MiniCPMv26ChatHandler")
@patch("src.inference.vlm_runner.Llama")
def test_inference_flow(mock_llama_class, mock_chat_handler, mock_exists):
    """Verifica que inference llame a create_chat_completion correctamente."""
    mock_exists.return_value = True
    
    # Setup del mock
    mock_instance = mock_llama_class.return_value
    mock_instance.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "Es un pólipo."}}]
    }
    
    loader = VLMLoader(model_path=FAKE_MODEL_PATH)
    loader.load_model()
    
    response = loader.inference(FAKE_IMAGE_PATH, "Describe")
    
    assert response == "Es un pólipo."
    mock_instance.create_chat_completion.assert_called_once()
    
    # Verificar estructura del mensaje (simplificada para MiniCPM-V que usa chat handler interno o standard)
    call_args = mock_instance.create_chat_completion.call_args
    messages = call_args.kwargs['messages']
    assert messages[0]['role'] == 'user'
    # Validar que contiene texto e imagen
    content = messages[0]['content']
    assert isinstance(content, list)
