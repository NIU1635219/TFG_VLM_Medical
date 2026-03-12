from types import SimpleNamespace
import sys
import types

import setup_env
from src.utils import setup_reinstall_ui, setup_tests_ui, setup_menu_engine
from unittest.mock import MagicMock

# Alias cómodo para los tests
_MenuItem = setup_menu_engine.MenuItem


class _DummyStyle:
    BOLD = ""
    ENDC = ""


# ---------------------------------------------------------------------------
# Helper: convierte un ctx-dict al nuevo par (kit, app)
# ---------------------------------------------------------------------------

def _make_kit_app(ctx):
    """Adapta un dict ctx (API antigua) a mocks (kit, app) de la nueva API."""
    kit = MagicMock()
    kit.style = ctx.get("Style", _DummyStyle)
    kit.MenuItem = ctx.get("MenuItem", MagicMock())
    kit.menu = ctx.get("interactive_menu", MagicMock())
    kit.log = ctx.get("log", MagicMock())
    kit.clear = ctx.get("clear_screen_ansi", MagicMock())
    kit.wait = ctx.get("wait_for_any_key", MagicMock())
    kit.ask = ctx.get("ask_user", MagicMock())
    kit.input = ctx.get("input_with_esc", MagicMock())
    kit.run_cmd = ctx.get("run_cmd", MagicMock(return_value=True))
    kit.subtitle = MagicMock()
    kit.banner = MagicMock()
    kit.table = MagicMock()
    kit.read_key = ctx.get("read_key", MagicMock())
    kit.width = MagicMock(return_value=80)
    kit.divider = MagicMock(return_value="─" * 80)
    kit.cursor_memory = ctx.get("MENU_CURSOR_MEMORY", {})
    kit.os_module = ctx.get("os_module", MagicMock())
    kit.msvcrt_module = ctx.get("msvcrt_module", None)
    kit.IncrementalPanelRenderer = ctx.get("IncrementalPanelRenderer", MagicMock())

    app = MagicMock()
    app.print_banner = ctx.get("print_banner", MagicMock())
    app.fix_libs = ctx.get("fix_libs", MagicMock())
    app.fix_uv = ctx.get("fix_uv", MagicMock())
    app.fix_folders = ctx.get("fix_folders", MagicMock())
    app.check_uv = ctx.get("check_uv", MagicMock())
    app.check_lms = ctx.get("check_lms", MagicMock())
    app.restart_program = ctx.get("restart_program", MagicMock())
    app.REQUIRED_LIBS = ctx.get("REQUIRED_LIBS", [])
    app.LIB_IMPORT_MAP = ctx.get("LIB_IMPORT_MAP", {})
    app.MODELS_REGISTRY = ctx.get("MODELS_REGISTRY", {})
    app.lms_models = ctx.get("lms_models", MagicMock())
    app.lms_menu_helpers = ctx.get("lms_menu_helpers", MagicMock())
    app.psutil = ctx.get("psutil", MagicMock())
    app.DiagnosticIssue = ctx.get("DiagnosticIssue", MagicMock())
    app.list_test_files = ctx.get("list_test_files", MagicMock())
    app.get_installed_lms_models = ctx.get("get_installed_lms_models", MagicMock())
    app.time_module = ctx.get("time", MagicMock())
    app.create_project_structure = ctx.get("create_project_structure", MagicMock())
    app.detect_gpu = ctx.get("detect_gpu", MagicMock())

    return kit, app


def test_setup_env_wrappers_delegate_to_extracted_modules(monkeypatch):
    """Verifica que los wrappers en `setup_env` deleguen correctamente a los módulos extraídos."""
    calls = {"tests": 0, "reinstall": 0}

    def fake_tests(kit, app):
        calls["tests"] += 1

    def fake_reinstall(kit, app):
        calls["reinstall"] += 1

    monkeypatch.setattr(setup_env.setup_tests_ui, "run_tests_menu", fake_tests)
    monkeypatch.setattr(setup_env.setup_reinstall_ui, "reinstall_library_menu", fake_reinstall)

    setup_env.run_tests_menu()
    setup_env.reinstall_library_menu()

    assert calls["tests"] == 1
    assert calls["reinstall"] == 1


def test_setup_tests_ui_run_all_executes_pytest_command():
    """Valida que la opción 'Run All Unit Tests' ejecute el comando `pytest tests/`."""
    run_cmd_calls = []
    waits = {"count": 0}
    screens = {"count": 0}
    menu_calls = {"count": 0}

    def interactive_menu(options, **kwargs):
        if kwargs.get("menu_id") == "tests_manager_menu":
            menu_calls["count"] += 1
            if menu_calls["count"] == 1:
                return next(item for item in options if item.label == " Run All Unit Tests (pytest)")
            return None
        raise AssertionError("Unexpected nested menu for this test")

    ctx = {
        "Style": _DummyStyle,
        "MenuItem": _MenuItem,
        "print_banner": lambda: None,
        "log": lambda *args, **kwargs: None,
        "run_cmd": lambda command: run_cmd_calls.append(command),
        "wait_for_any_key": lambda *args, **kwargs: waits.__setitem__("count", waits["count"] + 1),
        "interactive_menu": interactive_menu,
        "list_test_files": lambda: [],
        "get_installed_lms_models": lambda: [],
        "clear_screen_ansi": lambda: screens.__setitem__("count", screens["count"] + 1),
        "manage_models_menu_ui": lambda: None,
        "time": SimpleNamespace(sleep=lambda *_: None),
    }

    setup_tests_ui.run_tests_menu(*_make_kit_app(ctx))

    assert run_cmd_calls == ["uv run python -m pytest tests/"]
    assert waits["count"] == 1
    assert screens["count"] >= 1


def test_setup_reinstall_ui_core_selection_installs_and_restarts():
    """Confirma que seleccionar librerías core desencadene instalación y reinicio del programa."""
    fix_libs_calls = []
    fix_uv_calls = {"count": 0}
    restart_calls = {"count": 0}
    waits = {"count": 0}

    def interactive_menu(options, **kwargs):
        assert kwargs.get("menu_id") == "reinstall_menu"
        core_child = options[0].children[0]
        uv_option = options[1]
        return [core_child, uv_option]

    def fix_uv():
        fix_uv_calls["count"] += 1

    ctx = {
        "Style": _DummyStyle,
        "MenuItem": _MenuItem,
        "REQUIRED_LIBS": ["lmstudio", "pytest"],
        "print_banner": lambda: None,
        "interactive_menu": interactive_menu,
        "clear_screen_ansi": lambda: None,
        "fix_uv": fix_uv,
        "fix_libs": lambda libs: fix_libs_calls.append(libs),
        "log": lambda *args, **kwargs: None,
        "restart_program": lambda: restart_calls.__setitem__("count", restart_calls["count"] + 1),
        "wait_for_any_key": lambda *args, **kwargs: waits.__setitem__("count", waits["count"] + 1),
    }

    setup_reinstall_ui.reinstall_library_menu(*_make_kit_app(ctx))

    assert fix_libs_calls == [["lmstudio"]]
    assert fix_uv_calls["count"] == 1
    assert restart_calls["count"] == 1
    assert waits["count"] == 1


def test_setup_reinstall_ui_no_selection_does_nothing():
    """Asegura que si no se selecciona nada en el menú de reinstalación, no se ejecuten acciones."""
    calls = {"fix_libs": 0, "fix_uv": 0, "restart": 0, "wait": 0}

    ctx = {
        "Style": _DummyStyle,
        "MenuItem": _MenuItem,
        "REQUIRED_LIBS": ["lmstudio"],
        "print_banner": lambda: None,
        "interactive_menu": lambda *args, **kwargs: None,
        "clear_screen_ansi": lambda: None,
        "fix_uv": lambda: calls.__setitem__("fix_uv", calls["fix_uv"] + 1),
        "fix_libs": lambda *_: calls.__setitem__("fix_libs", calls["fix_libs"] + 1),
        "log": lambda *args, **kwargs: None,
        "restart_program": lambda: calls.__setitem__("restart", calls["restart"] + 1),
        "wait_for_any_key": lambda *args, **kwargs: calls.__setitem__("wait", calls["wait"] + 1),
    }

    setup_reinstall_ui.reinstall_library_menu(*_make_kit_app(ctx))

    assert calls == {"fix_libs": 0, "fix_uv": 0, "restart": 0, "wait": 0}


def test_setup_tests_ui_run_specific_cancel_returns_without_command():
    """Valida que cancelar la selección de un test específico no ejecute ningún comando."""
    run_cmd_calls = []
    menu_calls = {"tests": 0, "specific": 0}

    def interactive_menu(options, **kwargs):
        menu_id = kwargs.get("menu_id")
        if menu_id == "tests_manager_menu":
            menu_calls["tests"] += 1
            if menu_calls["tests"] == 1:
                return next(item for item in options if item.label == " Run Specific Test File...")
            return None
        if menu_id == "run_specific_test_selector":
            menu_calls["specific"] += 1
            return next(item for item in options if item.label == "Cancel")
        raise AssertionError(f"Unexpected menu_id: {menu_id}")

    ctx = {
        "Style": _DummyStyle,
        "MenuItem": _MenuItem,
        "print_banner": lambda: None,
        "log": lambda *args, **kwargs: None,
        "run_cmd": lambda command: run_cmd_calls.append(command),
        "wait_for_any_key": lambda *args, **kwargs: None,
        "interactive_menu": interactive_menu,
        "list_test_files": lambda: ["test_alpha.py"],
        "get_installed_lms_models": lambda: [],
        "clear_screen_ansi": lambda: None,
        "manage_models_menu_ui": lambda: None,
        "time": SimpleNamespace(sleep=lambda *_: None),
    }

    setup_tests_ui.run_tests_menu(*_make_kit_app(ctx))

    assert run_cmd_calls == []
    assert menu_calls["specific"] == 1


def test_setup_tests_ui_run_specific_executes_selected_file():
    """Comprueba que seleccionar un archivo de test específico lance el comando `pytest` correcto."""
    run_cmd_calls = []
    waits = {"count": 0}
    menu_calls = {"tests": 0, "specific": 0}

    def interactive_menu(options, **kwargs):
        menu_id = kwargs.get("menu_id")
        if menu_id == "tests_manager_menu":
            menu_calls["tests"] += 1
            if menu_calls["tests"] == 1:
                return next(item for item in options if item.label == " Run Specific Test File...")
            return None
        if menu_id == "run_specific_test_selector":
            menu_calls["specific"] += 1
            if menu_calls["specific"] == 1:
                return next(item for item in options if item.label == "test_alpha.py")
            return next(item for item in options if item.label == "Cancel")
        raise AssertionError(f"Unexpected menu_id: {menu_id}")

    ctx = {
        "Style": _DummyStyle,
        "MenuItem": _MenuItem,
        "print_banner": lambda: None,
        "log": lambda *args, **kwargs: None,
        "run_cmd": lambda command: run_cmd_calls.append(command),
        "wait_for_any_key": lambda *args, **kwargs: waits.__setitem__("count", waits["count"] + 1),
        "interactive_menu": interactive_menu,
        "list_test_files": lambda: ["test_alpha.py"],
        "get_installed_lms_models": lambda: [],
        "clear_screen_ansi": lambda: None,
        "manage_models_menu_ui": lambda: None,
        "time": SimpleNamespace(sleep=lambda *_: None),
    }

    setup_tests_ui.run_tests_menu(*_make_kit_app(ctx))

    assert run_cmd_calls == ["uv run python -m pytest tests/test_alpha.py"]
    assert waits["count"] == 1


def test_setup_tests_ui_smoke_without_models_warns_and_returns():
    """Asegura que el smoke test avise si no hay modelos instalados y no intente ejecutarse."""
    warns = []
    waits = {"count": 0}
    menu_calls = {"tests": 0}

    def interactive_menu(options, **kwargs):
        if kwargs.get("menu_id") == "tests_manager_menu":
            menu_calls["tests"] += 1
            if menu_calls["tests"] == 1:
                return next(item for item in options if item.label == " Run Smoke Test (Inference Demo)")
            return None
        raise AssertionError("Unexpected nested menu for this test")

    def log(msg, level="info"):
        if level == "warning":
            warns.append(msg)

    ctx = {
        "Style": _DummyStyle,
        "MenuItem": _MenuItem,
        "print_banner": lambda: None,
        "log": log,
        "run_cmd": lambda *_: None,
        "wait_for_any_key": lambda *args, **kwargs: waits.__setitem__("count", waits["count"] + 1),
        "interactive_menu": interactive_menu,
        "list_test_files": lambda: [],
        "get_installed_lms_models": lambda: [],
        "clear_screen_ansi": lambda: None,
        "manage_models_menu_ui": lambda: None,
        "time": SimpleNamespace(sleep=lambda *_: None),
    }

    setup_tests_ui.run_tests_menu(*_make_kit_app(ctx))

    assert waits["count"] == 1
    assert any("No hay modelos disponibles" in msg for msg in warns)


def test_setup_reinstall_ui_uv_only_does_not_restart_or_install_core():
    """Valida que reinstalar solo 'uv' no requiera reinstalar librerías core ni reiniciar."""
    calls = {"fix_libs": 0, "fix_uv": 0, "restart": 0, "wait": 0}

    def interactive_menu(options, **kwargs):
        assert kwargs.get("menu_id") == "reinstall_menu"
        uv_option = options[1]
        return [uv_option]

    ctx = {
        "Style": _DummyStyle,
        "MenuItem": _MenuItem,
        "REQUIRED_LIBS": ["lmstudio"],
        "print_banner": lambda: None,
        "interactive_menu": interactive_menu,
        "clear_screen_ansi": lambda: None,
        "fix_uv": lambda: calls.__setitem__("fix_uv", calls["fix_uv"] + 1),
        "fix_libs": lambda *_: calls.__setitem__("fix_libs", calls["fix_libs"] + 1),
        "log": lambda *args, **kwargs: None,
        "restart_program": lambda: calls.__setitem__("restart", calls["restart"] + 1),
        "wait_for_any_key": lambda *args, **kwargs: calls.__setitem__("wait", calls["wait"] + 1),
    }

    setup_reinstall_ui.reinstall_library_menu(*_make_kit_app(ctx))

    assert calls["fix_uv"] == 1
    assert calls["fix_libs"] == 0
    assert calls["restart"] == 0
    assert calls["wait"] == 1


def test_setup_tests_ui_smoke_import_error_logs_and_recovers(monkeypatch):
    """Comprueba que errores de importación en el script de smoke test sean manejados suavemente."""
    errors = []
    waits = {"count": 0}
    menu_calls = {"tests": 0, "smoke_model": 0}

    fake_module = types.ModuleType("src.scripts.test_inference")
    monkeypatch.setitem(sys.modules, "src.scripts.test_inference", fake_module)

    def interactive_menu(options, **kwargs):
        menu_id = kwargs.get("menu_id")
        if menu_id == "tests_manager_menu":
            menu_calls["tests"] += 1
            if menu_calls["tests"] == 1:
                return next(item for item in options if item.label == " Run Smoke Test (Inference Demo)")
            return None
        if menu_id == "run_smoke_model_selector":
            menu_calls["smoke_model"] += 1
            if menu_calls["smoke_model"] == 1:
                return next(item for item in options if item.label == "model-a")
            return next(item for item in options if item.label == "Cancel")
        raise AssertionError(f"Unexpected menu_id: {menu_id}")

    def log(msg, level="info"):
        if level == "error":
            errors.append(msg)

    ctx = {
        "Style": _DummyStyle,
        "MenuItem": _MenuItem,
        "print_banner": lambda: None,
        "log": log,
        "run_cmd": lambda *_: None,
        "wait_for_any_key": lambda *args, **kwargs: waits.__setitem__("count", waits["count"] + 1),
        "interactive_menu": interactive_menu,
        "list_test_files": lambda: [],
        "get_installed_lms_models": lambda: ["model-a"],
        "clear_screen_ansi": lambda: None,
        "manage_models_menu_ui": lambda: None,
        "time": SimpleNamespace(sleep=lambda *_: None),
    }

    setup_tests_ui.run_tests_menu(*_make_kit_app(ctx))

    assert any("Could not import smoke test script" in msg for msg in errors)
    assert any("Smoke test failed" in msg for msg in errors)
    assert waits["count"] >= 1


def test_setup_reinstall_ui_multiple_core_libs_grouped_once_and_restart():
    """Confirma que múltiples librerías core se agrupen en una sola llamada de instalación."""
    fix_libs_calls = []
    restart_calls = {"count": 0}

    def interactive_menu(options, **kwargs):
        assert kwargs.get("menu_id") == "reinstall_menu"
        core_children = options[0].children
        return [core_children[0], core_children[1]]

    ctx = {
        "Style": _DummyStyle,
        "MenuItem": _MenuItem,
        "REQUIRED_LIBS": ["lmstudio", "pytest", "pydantic"],
        "print_banner": lambda: None,
        "interactive_menu": interactive_menu,
        "clear_screen_ansi": lambda: None,
        "fix_uv": lambda: None,
        "fix_libs": lambda libs: fix_libs_calls.append(libs),
        "log": lambda *args, **kwargs: None,
        "restart_program": lambda: restart_calls.__setitem__("count", restart_calls["count"] + 1),
        "wait_for_any_key": lambda *args, **kwargs: None,
    }

    setup_reinstall_ui.reinstall_library_menu(*_make_kit_app(ctx))

    assert fix_libs_calls == [["lmstudio", "pytest"]]
    assert restart_calls["count"] == 1


def test_setup_tests_ui_main_menu_back_exits_immediately():
    """Valida la salida inmediata del menú de tests al seleccionar 'BACK'."""
    calls = {"count": 0}

    def interactive_menu(options, **kwargs):
        assert kwargs.get("menu_id") == "tests_manager_menu"
        calls["count"] += 1
        return "BACK"

    ctx = {
        "Style": _DummyStyle,
        "MenuItem": _MenuItem,
        "print_banner": lambda: None,
        "log": lambda *args, **kwargs: None,
        "run_cmd": lambda *_: None,
        "wait_for_any_key": lambda *args, **kwargs: None,
        "interactive_menu": interactive_menu,
        "list_test_files": lambda: [],
        "get_installed_lms_models": lambda: [],
        "clear_screen_ansi": lambda: None,
        "manage_models_menu_ui": lambda: None,
        "time": SimpleNamespace(sleep=lambda *_: None),
    }

    setup_tests_ui.run_tests_menu(*_make_kit_app(ctx))
    assert calls["count"] == 1


def test_setup_tests_ui_empty_list_choice_is_safe_and_exits_next_loop():
    """Prueba que una selección vacía (lista vacía) no rompa el bucle del menú."""
    menu_calls = {"count": 0}

    def interactive_menu(options, **kwargs):
        if kwargs.get("menu_id") == "tests_manager_menu":
            menu_calls["count"] += 1
            if menu_calls["count"] == 1:
                return []
            return None
        raise AssertionError("Unexpected nested menu")

    ctx = {
        "Style": _DummyStyle,
        "MenuItem": _MenuItem,
        "print_banner": lambda: None,
        "log": lambda *args, **kwargs: None,
        "run_cmd": lambda *_: None,
        "wait_for_any_key": lambda *args, **kwargs: None,
        "interactive_menu": interactive_menu,
        "list_test_files": lambda: [],
        "get_installed_lms_models": lambda: [],
        "clear_screen_ansi": lambda: None,
        "manage_models_menu_ui": lambda: None,
        "time": SimpleNamespace(sleep=lambda *_: None),
    }

    setup_tests_ui.run_tests_menu(*_make_kit_app(ctx))
    assert menu_calls["count"] == 1


def test_setup_tests_ui_non_action_choice_does_not_crash():
    """Asegura que seleccionar un item sin acción definida no provoque errores."""
    menu_calls = {"count": 0}

    def interactive_menu(options, **kwargs):
        if kwargs.get("menu_id") == "tests_manager_menu":
            menu_calls["count"] += 1
            if menu_calls["count"] == 1:
                return SimpleNamespace(label="Unknown", action=None)
            return None
        raise AssertionError("Unexpected nested menu")

    ctx = {
        "Style": _DummyStyle,
        "MenuItem": _MenuItem,
        "print_banner": lambda: None,
        "log": lambda *args, **kwargs: None,
        "run_cmd": lambda *_: None,
        "wait_for_any_key": lambda *args, **kwargs: None,
        "interactive_menu": interactive_menu,
        "list_test_files": lambda: [],
        "get_installed_lms_models": lambda: [],
        "clear_screen_ansi": lambda: None,
        "manage_models_menu_ui": lambda: None,
        "time": SimpleNamespace(sleep=lambda *_: None),
    }

    setup_tests_ui.run_tests_menu(*_make_kit_app(ctx))
    assert menu_calls["count"] == 2


def test_setup_tests_ui_run_specific_without_files_warns_and_returns():
    """Valida que si no hay archivos de tests para ejecución específica, se avise al usuario."""
    warns = []
    sleeps = {"count": 0}
    menu_calls = {"tests": 0}

    def interactive_menu(options, **kwargs):
        if kwargs.get("menu_id") == "tests_manager_menu":
            menu_calls["tests"] += 1
            if menu_calls["tests"] == 1:
                return next(item for item in options if item.label == " Run Specific Test File...")
            return None
        raise AssertionError("Unexpected nested menu")

    def log(msg, level="info"):
        if level == "warning":
            warns.append(msg)

    ctx = {
        "Style": _DummyStyle,
        "MenuItem": _MenuItem,
        "print_banner": lambda: None,
        "log": log,
        "run_cmd": lambda *_: None,
        "wait_for_any_key": lambda *args, **kwargs: None,
        "interactive_menu": interactive_menu,
        "list_test_files": lambda: [],
        "get_installed_lms_models": lambda: [],
        "clear_screen_ansi": lambda: None,
        "manage_models_menu_ui": lambda: None,
        "time": SimpleNamespace(sleep=lambda *_: sleeps.__setitem__("count", sleeps["count"] + 1)),
    }

    setup_tests_ui.run_tests_menu(*_make_kit_app(ctx))
    assert any("No tests found in tests/ folder." in msg for msg in warns)
    assert sleeps["count"] == 1


def test_setup_tests_ui_smoke_model_selector_cancel_returns_cleanly():
    """Verifica que cancelar el selector de modelo para el smoke test retorne limpiamente."""
    waits = {"count": 0}
    menu_calls = {"tests": 0, "smoke": 0}

    def interactive_menu(options, **kwargs):
        menu_id = kwargs.get("menu_id")
        if menu_id == "tests_manager_menu":
            menu_calls["tests"] += 1
            if menu_calls["tests"] == 1:
                return next(item for item in options if item.label == " Run Smoke Test (Inference Demo)")
            return None
        if menu_id == "run_smoke_model_selector":
            menu_calls["smoke"] += 1
            return next(item for item in options if item.label == "Cancel")
        raise AssertionError(f"Unexpected menu_id: {menu_id}")

    ctx = {
        "Style": _DummyStyle,
        "MenuItem": _MenuItem,
        "print_banner": lambda: None,
        "log": lambda *args, **kwargs: None,
        "run_cmd": lambda *_: None,
        "wait_for_any_key": lambda *args, **kwargs: waits.__setitem__("count", waits["count"] + 1),
        "interactive_menu": interactive_menu,
        "list_test_files": lambda: [],
        "get_installed_lms_models": lambda: ["model-a"],
        "clear_screen_ansi": lambda: None,
        "manage_models_menu_ui": lambda: None,
        "time": SimpleNamespace(sleep=lambda *_: None),
    }

    setup_tests_ui.run_tests_menu(*_make_kit_app(ctx))
    assert menu_calls["smoke"] == 1
    assert waits["count"] == 0


def test_setup_tests_ui_smoke_empty_model_tag_then_cancel():
    """Valida el manejo de una selección de modelo vacía o inválida en el smoke test."""
    warns = []
    waits = {"count": 0}
    menu_calls = {"tests": 0, "smoke": 0}

    def interactive_menu(options, **kwargs):
        menu_id = kwargs.get("menu_id")
        if menu_id == "tests_manager_menu":
            menu_calls["tests"] += 1
            if menu_calls["tests"] == 1:
                return next(item for item in options if item.label == " Run Smoke Test (Inference Demo)")
            return None
        if menu_id == "run_smoke_model_selector":
            menu_calls["smoke"] += 1
            if menu_calls["smoke"] == 1:
                return SimpleNamespace(label="")
            return next(item for item in options if item.label == "Cancel")
        raise AssertionError(f"Unexpected menu_id: {menu_id}")

    def log(msg, level="info"):
        if level == "warning":
            warns.append(msg)

    ctx = {
        "Style": _DummyStyle,
        "MenuItem": _MenuItem,
        "print_banner": lambda: None,
        "log": log,
        "run_cmd": lambda *_: None,
        "wait_for_any_key": lambda *args, **kwargs: waits.__setitem__("count", waits["count"] + 1),
        "interactive_menu": interactive_menu,
        "list_test_files": lambda: [],
        "get_installed_lms_models": lambda: ["model-a"],
        "clear_screen_ansi": lambda: None,
        "manage_models_menu_ui": lambda: None,
        "time": SimpleNamespace(sleep=lambda *_: None),
    }

    setup_tests_ui.run_tests_menu(*_make_kit_app(ctx))
    assert any("No model selected." in msg for msg in warns)
    assert waits["count"] == 1


def test_setup_reinstall_ui_empty_selection_list_behaves_like_cancel():
    """Comprueba que una lista de selección vacía en reinstalación se comporte como cancelar."""
    calls = {"fix_libs": 0, "fix_uv": 0, "restart": 0, "wait": 0}

    ctx = {
        "Style": _DummyStyle,
        "MenuItem": _MenuItem,
        "REQUIRED_LIBS": ["lmstudio"],
        "print_banner": lambda: None,
        "interactive_menu": lambda *args, **kwargs: [],
        "clear_screen_ansi": lambda: None,
        "fix_uv": lambda: calls.__setitem__("fix_uv", calls["fix_uv"] + 1),
        "fix_libs": lambda *_: calls.__setitem__("fix_libs", calls["fix_libs"] + 1),
        "log": lambda *args, **kwargs: None,
        "restart_program": lambda: calls.__setitem__("restart", calls["restart"] + 1),
        "wait_for_any_key": lambda *args, **kwargs: calls.__setitem__("wait", calls["wait"] + 1),
    }

    setup_reinstall_ui.reinstall_library_menu(*_make_kit_app(ctx))
    assert calls == {"fix_libs": 0, "fix_uv": 0, "restart": 0, "wait": 0}


def test_setup_tests_ui_schema_tester_selects_reasoning_variant(monkeypatch):
    """El Schema Tester debe elegir el modo reasoning a nivel de schema y no vía thinking nativo."""
    menu_calls = {"tests": 0, "schema_model": 0, "schema_selector": 0, "schema_mode": 0}
    run_batch_calls = []
    waits = {"count": 0}

    def interactive_menu(options, **kwargs):
        menu_id = kwargs.get("menu_id")
        if menu_id == "tests_manager_menu":
            menu_calls["tests"] += 1
            if menu_calls["tests"] == 1:
                return next(item for item in options if item.label == " Run Schema Tester (VLM Interactive)")
            return None
        if menu_id == "schema_tester_model_selector":
            menu_calls["schema_model"] += 1
            if menu_calls["schema_model"] == 1:
                return next(item for item in options if item.label == "model-a")
            if menu_calls["schema_model"] == 2:
                return next(item for item in options if item.label == "Cancel")
            raise AssertionError("schema_tester_model_selector llamado más veces de lo esperado")
        if menu_id == "schema_tester_schema_selector":
            menu_calls["schema_selector"] += 1
            if menu_calls["schema_selector"] > 1:
                raise AssertionError("schema_tester_schema_selector llamado más veces de lo esperado")
            return next(item for item in options if item.label == "PolypDetection")
        if menu_id == "schema_tester_reasoning_selector":
            menu_calls["schema_mode"] += 1
            if menu_calls["schema_mode"] > 1:
                raise AssertionError("schema_tester_reasoning_selector llamado más veces de lo esperado")
            return next(item for item in options if item.label == "Con razonamiento")
        raise AssertionError(f"Unexpected menu_id: {menu_id}")

    import src.scripts.test_schema as schema_script
    monkeypatch.setattr(schema_script, "find_images", lambda: ["a.jpg"])
    monkeypatch.setattr(schema_script, "format_schema_info", lambda *args, **kwargs: "Schema info")
    monkeypatch.setattr(
        schema_script,
        "run_batch",
        lambda model_tag, schema_name, schema_cls, images: run_batch_calls.append(
            (model_tag, schema_name, schema_cls.__name__, list(images))
        ) or (1, 0, 0),
    )

    ctx = {
        "Style": _DummyStyle,
        "MenuItem": _MenuItem,
        "print_banner": lambda: None,
        "log": lambda *args, **kwargs: None,
        "run_cmd": lambda *_: None,
        "wait_for_any_key": lambda *args, **kwargs: waits.__setitem__("count", waits["count"] + 1),
        "interactive_menu": interactive_menu,
        "list_test_files": lambda: [],
        "get_installed_lms_models": lambda: ["model-a"],
        "clear_screen_ansi": lambda: None,
        "manage_models_menu_ui": lambda: None,
        "time": SimpleNamespace(sleep=lambda *_: None),
    }

    setup_tests_ui.run_tests_menu(*_make_kit_app(ctx))

    assert run_batch_calls == [("model-a", "PolypDetectionWithReasoning", "PolypDetectionWithReasoning", ["a.jpg"])]
    assert menu_calls["schema_mode"] == 1
    assert waits["count"] == 1


def test_setup_tests_ui_telemetry_probe_runs_with_selected_schema(monkeypatch):
    menu_calls = {"tests": 0, "model": 0, "schema": 0, "mode": 0}
    telemetry_calls = []
    waits = {"count": 0}

    def interactive_menu(options, **kwargs):
        menu_id = kwargs.get("menu_id")
        if menu_id == "tests_manager_menu":
            menu_calls["tests"] += 1
            if menu_calls["tests"] == 1:
                return next(item for item in options if item.label == " Run Telemetry Probe (TTFT/TPS)")
            return None
        if menu_id == "telemetry_probe_model_selector":
            menu_calls["model"] += 1
            if menu_calls["model"] == 1:
                return next(item for item in options if item.label == "model-a")
            return next(item for item in options if item.label == "Cancel")
        if menu_id == "telemetry_probe_schema_selector":
            menu_calls["schema"] += 1
            return next(item for item in options if item.label == "PolypDetection")
        if menu_id == "telemetry_probe_reasoning_selector":
            menu_calls["mode"] += 1
            return next(item for item in options if item.label == "Sin razonamiento")
        raise AssertionError(f"Unexpected menu_id: {menu_id}")

    import src.scripts.test_schema as schema_script
    import src.scripts.test_telemetry as telemetry_script

    monkeypatch.setattr(schema_script, "find_images", lambda: ["a.jpg", "b.jpg"])
    monkeypatch.setattr(
        telemetry_script,
        "run_telemetry_batch",
        lambda **kwargs: telemetry_calls.append(kwargs) or {
            "model_id": kwargs["model_id"],
            "schema_name": kwargs["schema_name"],
            "sample_size": 2,
            "ok": 2,
            "fail": 0,
            "records": [],
            "ttft": {"avg": 0.4, "min": 0.3, "max": 0.5},
            "tps": {"avg": 20.0, "min": 19.0, "max": 21.0},
            "generation_duration": {"avg": 0.8, "min": 0.7, "max": 0.9},
            "reasoning_tokens": {"avg": 12.0, "min": 10.0, "max": 14.0},
            "gpu_layers": {"avg": 28.0, "min": 28.0, "max": 28.0},
            "prompt_tokens": {"avg": 110.0, "min": 100.0, "max": 120.0},
            "completion_tokens": {"avg": 45.0, "min": 40.0, "max": 50.0},
            "total_tokens": {"avg": 155.0, "min": 140.0, "max": 170.0},
            "total_duration": {"avg": 1.1, "min": 1.0, "max": 1.2},
            "static_model_info": {
                "resolved_model_id": "model-a",
                "architecture": "qwen3_vl",
                "stop_reason": "eos",
            },
            "prompt": "demo",
            "telemetry_availability": {},
            "notes": {"ttft": None, "tps": None},
        },
    )

    ctx = {
        "Style": _DummyStyle,
        "MenuItem": _MenuItem,
        "print_banner": lambda: None,
        "log": lambda *args, **kwargs: None,
        "run_cmd": lambda *_: None,
        "wait_for_any_key": lambda *args, **kwargs: waits.__setitem__("count", waits["count"] + 1),
        "interactive_menu": interactive_menu,
        "list_test_files": lambda: [],
        "get_installed_lms_models": lambda: ["model-a"],
        "clear_screen_ansi": lambda: None,
        "manage_models_menu_ui": lambda: None,
        "time": SimpleNamespace(sleep=lambda *_: None),
    }

    setup_tests_ui.run_tests_menu(*_make_kit_app(ctx))

    assert telemetry_calls[0]["model_id"] == "model-a"
    assert telemetry_calls[0]["schema_name"] == "PolypDetection"
    assert waits["count"] == 1


def test_setup_tests_ui_telemetry_probe_warns_when_tps_is_unavailable(monkeypatch):
    menu_calls = {"tests": 0, "model": 0, "schema": 0, "mode": 0}
    warnings = []

    def interactive_menu(options, **kwargs):
        menu_id = kwargs.get("menu_id")
        if menu_id == "tests_manager_menu":
            menu_calls["tests"] += 1
            if menu_calls["tests"] == 1:
                return next(item for item in options if item.label == " Run Telemetry Probe (TTFT/TPS)")
            return None
        if menu_id == "telemetry_probe_model_selector":
            menu_calls["model"] += 1
            if menu_calls["model"] == 1:
                return next(item for item in options if item.label == "model-a")
            return next(item for item in options if item.label == "Cancel")
        if menu_id == "telemetry_probe_schema_selector":
            menu_calls["schema"] += 1
            return next(item for item in options if item.label == "PolypDetection")
        if menu_id == "telemetry_probe_reasoning_selector":
            menu_calls["mode"] += 1
            return next(item for item in options if item.label == "Sin razonamiento")
        raise AssertionError(f"Unexpected menu_id: {menu_id}")

    import src.scripts.test_schema as schema_script
    import src.scripts.test_telemetry as telemetry_script

    monkeypatch.setattr(schema_script, "find_images", lambda: ["a.jpg"])
    monkeypatch.setattr(
        telemetry_script,
        "run_telemetry_batch",
        lambda **kwargs: {
            "model_id": kwargs["model_id"],
            "schema_name": kwargs["schema_name"],
            "sample_size": 1,
            "ok": 1,
            "fail": 0,
            "records": [
                {
                    "image_path": "a.jpg",
                    "status": "ok",
                    "ttft_seconds": 0.4,
                    "tokens_per_second": None,
                    "generation_duration_seconds": 0.7,
                    "prompt_tokens": 100,
                    "completion_tokens": 40,
                    "reasoning_tokens": 11,
                    "stop_reason": "eos",
                    "gpu_layers": 24,
                    "total_duration_seconds": 1.1,
                }
            ],
            "ttft": {"avg": 0.4, "min": 0.4, "max": 0.4},
            "tps": {"avg": None, "min": None, "max": None},
            "generation_duration": {"avg": 0.7, "min": 0.7, "max": 0.7},
            "reasoning_tokens": {"avg": 11.0, "min": 11.0, "max": 11.0},
            "gpu_layers": {"avg": 24.0, "min": 24.0, "max": 24.0},
            "prompt_tokens": {"avg": 100.0, "min": 100.0, "max": 100.0},
            "completion_tokens": {"avg": 40.0, "min": 40.0, "max": 40.0},
            "total_tokens": {"avg": 140.0, "min": 140.0, "max": 140.0},
            "total_duration": {"avg": 1.1, "min": 1.1, "max": 1.1},
            "static_model_info": {
                "resolved_model_id": "model-a",
                "architecture": None,
                "stop_reason": "eos",
            },
            "telemetry_availability": {"ttft_records": 1, "tps_records": 0, "total_duration_records": 1, "ok_records": 1},
            "notes": {"ttft": None, "tps": "LM Studio no devolvio tokens_per_second en response.stats; TPS no disponible en esta ejecucion."},
            "prompt": "demo",
        },
    )

    def log(msg, level="info"):
        if level == "warning":
            warnings.append(msg)

    ctx = {
        "Style": _DummyStyle,
        "MenuItem": _MenuItem,
        "print_banner": lambda: None,
        "log": log,
        "run_cmd": lambda *_: None,
        "wait_for_any_key": lambda *args, **kwargs: None,
        "interactive_menu": interactive_menu,
        "list_test_files": lambda: [],
        "get_installed_lms_models": lambda: ["model-a"],
        "clear_screen_ansi": lambda: None,
        "manage_models_menu_ui": lambda: None,
        "time": SimpleNamespace(sleep=lambda *_: None),
    }

    setup_tests_ui.run_tests_menu(*_make_kit_app(ctx))

    assert any("TPS no disponible" in msg for msg in warnings)


def test_setup_tests_ui_telemetry_probe_hides_metrics_without_coverage(monkeypatch, capsys):
    menu_calls = {"tests": 0, "model": 0, "schema": 0, "mode": 0}
    logs = []

    def interactive_menu(options, **kwargs):
        menu_id = kwargs.get("menu_id")
        if menu_id == "tests_manager_menu":
            menu_calls["tests"] += 1
            if menu_calls["tests"] == 1:
                return next(item for item in options if item.label == " Run Telemetry Probe (TTFT/TPS)")
            return None
        if menu_id == "telemetry_probe_model_selector":
            menu_calls["model"] += 1
            if menu_calls["model"] == 1:
                return next(item for item in options if item.label == "model-a")
            return next(item for item in options if item.label == "Cancel")
        if menu_id == "telemetry_probe_schema_selector":
            menu_calls["schema"] += 1
            return next(item for item in options if item.label == "PolypDetection")
        if menu_id == "telemetry_probe_reasoning_selector":
            menu_calls["mode"] += 1
            return next(item for item in options if item.label == "Sin razonamiento")
        raise AssertionError(f"Unexpected menu_id: {menu_id}")

    import src.scripts.test_schema as schema_script
    import src.scripts.test_telemetry as telemetry_script

    monkeypatch.setattr(schema_script, "find_images", lambda: ["a.jpg"])
    monkeypatch.setattr(
        telemetry_script,
        "run_telemetry_batch",
        lambda **kwargs: {
            "model_id": kwargs["model_id"],
            "schema_name": kwargs["schema_name"],
            "sample_size": 1,
            "ok": 1,
            "fail": 0,
            "records": [
                {
                    "image_path": "a.jpg",
                    "status": "ok",
                    "ttft_seconds": 1.85791,
                    "tokens_per_second": 28.196269274704576,
                    "generation_duration_seconds": 8.4348634,
                    "prompt_tokens": 1488,
                    "completion_tokens": 237.8,
                    "reasoning_tokens": None,
                    "stop_reason": "eosFound",
                    "gpu_layers": -1,
                    "total_duration_seconds": 10.3789271799993,
                }
            ],
            "ttft": {"avg": 1.85791, "min": 1.85791, "max": 1.85791},
            "tps": {"avg": 28.196269274704576, "min": 28.196269274704576, "max": 28.196269274704576},
            "generation_duration": {"avg": 8.4348634, "min": 8.4348634, "max": 8.4348634},
            "reasoning_tokens": {"avg": None, "min": None, "max": None},
            "gpu_layers": {"avg": -1.0, "min": -1.0, "max": -1.0},
            "prompt_tokens": {"avg": 1488.0, "min": 1488.0, "max": 1488.0},
            "completion_tokens": {"avg": 237.8, "min": 237.8, "max": 237.8},
            "total_tokens": {"avg": 1725.8, "min": 1725.8, "max": 1725.8},
            "total_duration": {"avg": 10.3789271799993, "min": 10.3789271799993, "max": 10.3789271799993},
            "static_model_info": {
                "resolved_model_id": "model-a",
                "architecture": "qwen35",
                "stop_reason": "eosFound",
            },
            "telemetry_availability": {
                "ttft_records": 1,
                "tps_records": 1,
                "generation_records": 1,
                "reasoning_records": 0,
                "gpu_layer_records": 1,
                "prompt_token_records": 1,
                "completion_token_records": 1,
                "total_token_records": 1,
                "total_duration_records": 1,
                "ok_records": 1,
            },
            "notes": {"ttft": None, "tps": None},
            "prompt": "demo",
        },
    )

    ctx = {
        "Style": _DummyStyle,
        "MenuItem": _MenuItem,
        "print_banner": lambda: None,
        "log": lambda msg, level="info": logs.append((msg, level)),
        "run_cmd": lambda *_: None,
        "wait_for_any_key": lambda *args, **kwargs: None,
        "interactive_menu": interactive_menu,
        "list_test_files": lambda: [],
        "get_installed_lms_models": lambda: ["model-a"],
        "clear_screen_ansi": lambda: None,
        "manage_models_menu_ui": lambda: None,
        "time": SimpleNamespace(sleep=lambda *_: None),
    }

    kit, app = _make_kit_app(ctx)
    setup_tests_ui.run_tests_menu(kit, app)

    output = capsys.readouterr().out
    render_calls = kit.IncrementalPanelRenderer.return_value.render.call_args_list
    rendered_lines = []
    if render_calls:
        rendered_lines = render_calls[-1][0][0]
    rendered_text = "\n".join(rendered_lines)
    assert "Ingesta prompt media (t/s)" not in output
    assert "VRAM media (MB)" not in output
    assert "Reasoning schema medio" not in output
    assert "prompt_tps=" not in rendered_text
    assert "vram=" not in rendered_text
    assert "reasoning=" not in rendered_text
    assert any(msg == "Cobertura: TTFT 1/1 · TPS 1/1 · GPU 1/1" for msg, _level in logs)
    assert "TTFT=1.858 s" in rendered_text
    assert "TPS=28.196" in rendered_text
    assert "total=10.379 s" in rendered_text


def test_setup_tests_ui_telemetry_probe_reports_incidents_in_final_message(monkeypatch):
    menu_calls = {"tests": 0, "model": 0, "schema": 0, "mode": 0}
    logs = []

    def interactive_menu(options, **kwargs):
        menu_id = kwargs.get("menu_id")
        if menu_id == "tests_manager_menu":
            menu_calls["tests"] += 1
            if menu_calls["tests"] == 1:
                return next(item for item in options if item.label == " Run Telemetry Probe (TTFT/TPS)")
            return None
        if menu_id == "telemetry_probe_model_selector":
            menu_calls["model"] += 1
            if menu_calls["model"] == 1:
                return next(item for item in options if item.label == "model-a")
            return next(item for item in options if item.label == "Cancel")
        if menu_id == "telemetry_probe_schema_selector":
            menu_calls["schema"] += 1
            return next(item for item in options if item.label == "PolypDetection")
        if menu_id == "telemetry_probe_reasoning_selector":
            menu_calls["mode"] += 1
            return next(item for item in options if item.label == "Sin razonamiento")
        raise AssertionError(f"Unexpected menu_id: {menu_id}")

    import src.scripts.test_schema as schema_script
    import src.scripts.test_telemetry as telemetry_script

    monkeypatch.setattr(schema_script, "find_images", lambda: ["a.jpg", "b.jpg"])
    monkeypatch.setattr(
        telemetry_script,
        "run_telemetry_batch",
        lambda **kwargs: {
            "model_id": kwargs["model_id"],
            "schema_name": kwargs["schema_name"],
            "sample_size": 2,
            "ok": 1,
            "fail": 1,
            "records": [
                {"image_path": "a.jpg", "status": "ok", "ttft_seconds": 1.2, "tokens_per_second": 30.0, "generation_duration_seconds": 2.0, "prompt_tokens": 100, "completion_tokens": 50, "stop_reason": "eosFound", "gpu_layers": -1, "total_duration_seconds": 3.4},
                {"image_path": "b.jpg", "status": "error", "error": "boom"},
            ],
            "ttft": {"avg": 1.2, "min": 1.2, "max": 1.2},
            "tps": {"avg": 30.0, "min": 30.0, "max": 30.0},
            "generation_duration": {"avg": 2.0, "min": 2.0, "max": 2.0},
            "reasoning_tokens": {"avg": None, "min": None, "max": None},
            "gpu_layers": {"avg": -1.0, "min": -1.0, "max": -1.0},
            "prompt_tokens": {"avg": 100.0, "min": 100.0, "max": 100.0},
            "completion_tokens": {"avg": 50.0, "min": 50.0, "max": 50.0},
            "total_tokens": {"avg": 150.0, "min": 150.0, "max": 150.0},
            "total_duration": {"avg": 3.4, "min": 3.4, "max": 3.4},
            "static_model_info": {"resolved_model_id": "model-a", "architecture": "qwen35", "stop_reason": "eosFound"},
            "telemetry_availability": {"ttft_records": 1, "tps_records": 1, "gpu_layer_records": 1, "prompt_token_records": 1, "completion_token_records": 1, "total_token_records": 1, "total_duration_records": 1, "ok_records": 1},
            "notes": {"ttft": None, "tps": None},
            "prompt": "demo",
        },
    )

    ctx = {
        "Style": _DummyStyle,
        "MenuItem": _MenuItem,
        "print_banner": lambda: None,
        "log": lambda msg, level="info": logs.append((msg, level)),
        "run_cmd": lambda *_: None,
        "wait_for_any_key": lambda *args, **kwargs: None,
        "interactive_menu": interactive_menu,
        "list_test_files": lambda: [],
        "get_installed_lms_models": lambda: ["model-a"],
        "clear_screen_ansi": lambda: None,
        "manage_models_menu_ui": lambda: None,
        "time": SimpleNamespace(sleep=lambda *_: None),
    }

    setup_tests_ui.run_tests_menu(*_make_kit_app(ctx))

    assert any(msg == "Telemetry Probe completado con incidencias: 1 OK, 1 errores." and level == "warning" for msg, level in logs)


def test_setup_tests_ui_batch_runner_exports_results(monkeypatch):
    menu_calls = {"tests": 0, "model": 0, "schema": 0, "mode": 0, "size": 0}
    batch_calls = []
    waits = {"count": 0}
    logs = []

    def interactive_menu(options, **kwargs):
        menu_id = kwargs.get("menu_id")
        if menu_id == "tests_manager_menu":
            menu_calls["tests"] += 1
            if menu_calls["tests"] == 1:
                return next(item for item in options if item.label == " Run Batch Runner")
            return None
        if menu_id == "batch_runner_model_selector":
            menu_calls["model"] += 1
            if menu_calls["model"] == 1:
                return next(item for item in options if item.label == "model-a")
            return next(item for item in options if item.label == "Cancel")
        if menu_id == "batch_runner_schema_selector":
            menu_calls["schema"] += 1
            return next(item for item in options if item.label == "PolypDetection")
        if menu_id == "batch_runner_reasoning_selector":
            menu_calls["mode"] += 1
            return next(item for item in options if item.label == "Con razonamiento")
        if menu_id == "batch_runner_size_selector":
            menu_calls["size"] += 1
            return next(item for item in options if item.label == "25 imágenes")
        raise AssertionError(f"Unexpected menu_id: {menu_id}")

    import src.scripts.batch_runner as batch_script

    monkeypatch.setattr(
        batch_script,
        "run_batch_job",
        lambda **kwargs: batch_calls.append(kwargs) or {
            "ok": 25,
            "invalid": 0,
            "fail": 0,
            "output_path": "data/processed/batch_results/demo.jsonl",
        },
    )

    ctx = {
        "Style": _DummyStyle,
        "MenuItem": _MenuItem,
        "print_banner": lambda: None,
        "log": lambda msg, level="info": logs.append((msg, level)),
        "run_cmd": lambda *_: None,
        "wait_for_any_key": lambda *args, **kwargs: waits.__setitem__("count", waits["count"] + 1),
        "interactive_menu": interactive_menu,
        "list_test_files": lambda: [],
        "get_installed_lms_models": lambda: ["model-a"],
        "clear_screen_ansi": lambda: None,
        "manage_models_menu_ui": lambda: None,
        "time": SimpleNamespace(sleep=lambda *_: None),
    }

    setup_tests_ui.run_tests_menu(*_make_kit_app(ctx))

    assert batch_calls[0]["model_id"] == "model-a"
    assert batch_calls[0]["schema_name"] == "PolypDetectionWithReasoning"
    assert batch_calls[0]["max_images"] == 25
    assert waits["count"] == 1
    assert any("Batch Runner completado" in msg for msg, _level in logs)


def test_setup_tests_ui_response_inspector_runs_from_menu(monkeypatch):
    menu_calls = {"tests": 0, "model": 0, "mode": 0}
    inspector_calls = []
    waits = {"count": 0}
    logs = []
    rendered_sections = []

    def interactive_menu(options, **kwargs):
        menu_id = kwargs.get("menu_id")
        if menu_id == "tests_manager_menu":
            menu_calls["tests"] += 1
            if menu_calls["tests"] == 1:
                return next(item for item in options if item.label == " Run Response Inspector (SDK Fields)")
            return None
        if menu_id == "response_inspector_model_selector":
            menu_calls["model"] += 1
            if menu_calls["model"] == 1:
                return next(item for item in options if item.label == "model-a")
            return next(item for item in options if item.label == "Cancel")
        if menu_id == "response_inspector_mode_selector":
            menu_calls["mode"] += 1
            return next(item for item in options if item.label == "Estructurada con reasoning")
        raise AssertionError(f"Unexpected menu_id: {menu_id}")

    import src.scripts.test_response as inspector_script

    monkeypatch.setattr(
        inspector_script,
        "run_inspection",
        lambda args: inspector_calls.append(args) or {
            "request": {"model_id": args.model, "image_path": "a.jpg", "schema_name": "GenericObjectDetectionWithReasoning"},
            "response": {"python_type": "PredictionResult", "structured": True, "public_attributes": {}, "text_extracted": "demo", "model_info": {}, "stats": {}, "parsed": {}},
        },
    )
    monkeypatch.setattr(inspector_script, "save_inspection_payload", lambda payload, output_path=None: f"data/processed/debug/{payload['request']['model_id']}.json")
    monkeypatch.setattr(inspector_script, "build_summary_sections", lambda payload: rendered_sections.append(payload) or [("Resumen", [("Model", payload["request"]["model_id"])])])

    ctx = {
        "Style": _DummyStyle,
        "MenuItem": _MenuItem,
        "print_banner": lambda: None,
        "log": lambda msg, level="info": logs.append((msg, level)),
        "run_cmd": lambda *_: None,
        "wait_for_any_key": lambda *args, **kwargs: waits.__setitem__("count", waits["count"] + 1),
        "interactive_menu": interactive_menu,
        "list_test_files": lambda: [],
        "get_installed_lms_models": lambda: ["model-a"],
        "clear_screen_ansi": lambda: None,
        "manage_models_menu_ui": lambda: None,
        "time": SimpleNamespace(sleep=lambda *_: None),
    }

    setup_tests_ui.run_tests_menu(*_make_kit_app(ctx))

    assert inspector_calls[0].model == "model-a"
    assert inspector_calls[0].structured is True
    assert inspector_calls[0].with_reasoning is True
    assert rendered_sections[0]["request"]["model_id"] == "model-a"
    assert waits["count"] == 1
    assert any("Response Inspector completado" in msg for msg, _level in logs)
