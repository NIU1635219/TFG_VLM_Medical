from unittest.mock import patch, MagicMock
from src.utils.setup_install_flow import perform_install


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kit_app(overrides=None):
    """Construye mocks (kit, app) para tests de setup_install_flow."""
    overrides = overrides or {}

    kit = MagicMock()
    kit.log = overrides.get("log", MagicMock())
    kit.run_cmd = overrides.get("run_cmd", MagicMock(return_value=True))

    app = MagicMock()
    app.REQUIRED_LIBS = overrides.get("REQUIRED_LIBS", ["pytest"])

    return kit, app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@patch("src.utils.setup_install_flow.create_project_structure")
@patch("src.utils.setup_install_flow.detect_gpu", return_value=False)
@patch("src.utils.setup_install_flow.check_uv", return_value=True)
def test_perform_install_success(mock_check_uv, mock_detect_gpu, mock_create_ps):
    kit, app = _make_kit_app()

    perform_install(kit, app)

    assert kit.log.called
    assert mock_create_ps.called
    assert mock_check_uv.called
    assert mock_detect_gpu.called


@patch('src.utils.setup_install_flow.os.path.exists')
@patch('src.utils.setup_install_flow.sys.executable', 'dummy_python')
def test_perform_install_full_reinstall(mock_exists):
    mock_exists.return_value = True

    kit, app = _make_kit_app()

    with patch('shutil.rmtree'):
        with patch("src.utils.setup_install_flow.create_project_structure"):
            with patch("src.utils.setup_install_flow.check_uv", return_value=True):
                with patch("src.utils.setup_install_flow.detect_gpu", return_value=False):
                    perform_install(kit, app, full_reinstall=True)

    assert mock_exists.called
    assert kit.run_cmd.called
