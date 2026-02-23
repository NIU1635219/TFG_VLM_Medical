import pytest
from unittest.mock import patch, MagicMock
from src.utils.setup_install_flow import perform_install

def test_perform_install_success():
    # Mock context
    ctx = {
        "log": MagicMock(),
        "create_project_structure": MagicMock(),
        "check_uv": MagicMock(),
        "run_cmd": MagicMock(),
        "detect_gpu": MagicMock(return_value="cpu"),
        "REQUIRED_LIBS": ["pytest"]
    }
    
    # Should not raise an exception
    perform_install(ctx=ctx)
    
    assert ctx["log"].called
    assert ctx["create_project_structure"].called
    assert ctx["check_uv"].called
    assert ctx["run_cmd"].called
    assert ctx["detect_gpu"].called

@patch('src.utils.setup_install_flow.os.path.exists')
@patch('src.utils.setup_install_flow.sys.executable', 'dummy_python')
def test_perform_install_full_reinstall(mock_exists):
    mock_exists.return_value = True
    
    ctx = {
        "log": MagicMock(),
        "create_project_structure": MagicMock(),
        "check_uv": MagicMock(),
        "run_cmd": MagicMock(),
        "detect_gpu": MagicMock(return_value="cpu"),
        "REQUIRED_LIBS": ["pytest"]
    }
    
    with patch('shutil.rmtree') as mock_rmtree:
        perform_install(ctx=ctx, full_reinstall=True)
        
        assert mock_exists.called
        assert ctx["run_cmd"].called
