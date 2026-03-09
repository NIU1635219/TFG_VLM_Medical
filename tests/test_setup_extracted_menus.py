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
