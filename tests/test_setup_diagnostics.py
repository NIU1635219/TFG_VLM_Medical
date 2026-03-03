from unittest.mock import patch, MagicMock
from src.utils.setup_diagnostics import perform_diagnostics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kit_app(overrides=None):
    """Construye mocks (kit, app) para tests de setup_diagnostics."""
    overrides = overrides or {}

    kit = MagicMock()
    kit.log = MagicMock()

    app = MagicMock()
    app.fix_folders = overrides.get("fix_folders", MagicMock())
    app.fix_uv = overrides.get("fix_uv", MagicMock())
    app.fix_libs = overrides.get("fix_libs", MagicMock())
    app.check_lms = overrides.get("check_lms", MagicMock(return_value=True))
    app.REQUIRED_LIBS = overrides.get("REQUIRED_LIBS", ["pytest"])
    app.LIB_IMPORT_MAP = overrides.get("LIB_IMPORT_MAP", {"pytest": "pytest"})

    return kit, app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_perform_diagnostics():
    kit, app = _make_kit_app()

    with patch("src.utils.setup_diagnostics.check_uv", return_value=True):
        with patch("src.utils.setup_diagnostics.os.path.exists", return_value=True):
            with patch("builtins.__import__", return_value=MagicMock()):
                with patch(
                    "src.utils.setup_diagnostics.shutil.disk_usage",
                    return_value=MagicMock(free=100 * 1024 ** 3),
                ):
                    report, issues = perform_diagnostics(kit, app)

    assert len(report) > 0
    assert len(issues) == 0


def test_perform_diagnostics_with_issues():
    kit, app = _make_kit_app(
        {
            "check_lms": MagicMock(return_value=False),
        }
    )

    with patch("src.utils.setup_diagnostics.check_uv", return_value=False):
        with patch("src.utils.setup_diagnostics.os.path.exists", return_value=False):
            with patch("builtins.__import__", side_effect=ImportError):
                with patch(
                    "src.utils.setup_diagnostics.shutil.disk_usage",
                    return_value=MagicMock(free=1 * 1024 ** 3),
                ):
                    report, issues = perform_diagnostics(kit, app)

    assert len(issues) > 0
    issue_names = [i.name for i in issues]
    assert "Missing Folders" in issue_names
    assert "Missing uv" in issue_names
    assert "Missing pytest" in issue_names
