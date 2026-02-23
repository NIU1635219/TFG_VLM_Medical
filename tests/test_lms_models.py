import pytest
from unittest.mock import patch, MagicMock
from src.utils.lms_models import _normalize_model_ref, check_lms, get_server_status

def test_normalize_model_ref():
    assert _normalize_model_ref("huggingface.co/lmstudio-community/model") == "lmstudio-community/model"
    assert _normalize_model_ref("lmstudio-community/model") == "lmstudio-community/model"
    assert _normalize_model_ref("") == ""

@patch('src.utils.lms_models._run_lms_command')
def test_check_lms_installed(mock_run):
    mock_run.return_value = MagicMock(returncode=0)
    assert check_lms() is True

@patch('src.utils.lms_models._run_lms_command')
def test_check_lms_not_installed(mock_run):
    mock_run.side_effect = Exception("Command not found")
    assert check_lms() is False

@patch('src.utils.lms_models.check_lms')
@patch('src.utils.lms_models._run_lms_command')
def test_get_server_status_running(mock_run, mock_check_lms):
    mock_check_lms.return_value = True
    mock_result = MagicMock()
    mock_result.stdout = "Server is running on port 1234"
    mock_result.stderr = ""
    mock_run.return_value = mock_result
    
    is_running, output = get_server_status()
    assert is_running is True
    assert "running" in output.lower()

@patch('src.utils.lms_models.check_lms')
@patch('src.utils.lms_models._run_lms_command')
def test_get_server_status_not_running(mock_run, mock_check_lms):
    mock_check_lms.return_value = True
    mock_result = MagicMock()
    mock_result.stdout = "Server is stopped"
    mock_result.stderr = ""
    mock_run.return_value = mock_result
    
    is_running, output = get_server_status()
    assert is_running is False
    assert "stopped" in output.lower()

@patch('src.utils.lms_models.check_lms')
def test_get_server_status_no_lms(mock_check_lms):
    mock_check_lms.return_value = False
    is_running, output = get_server_status()
    assert is_running is False
    assert "lms CLI not found" in output
