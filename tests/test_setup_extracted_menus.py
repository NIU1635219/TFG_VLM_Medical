from types import SimpleNamespace
import sys
import types

import setup_env
from src.utils import setup_reinstall_ui, setup_tests_ui


class _DummyStyle:
    BOLD = ""
    ENDC = ""


def test_setup_env_wrappers_delegate_to_extracted_modules(monkeypatch):
    calls = {"tests": 0, "reinstall": 0}

    def fake_tests(*, ctx):
        calls["tests"] += 1
        assert "interactive_menu" in ctx
        assert "manage_models_menu_ui" in ctx

    def fake_reinstall(*, ctx):
        calls["reinstall"] += 1
        assert "fix_libs" in ctx
        assert "restart_program" in ctx

    monkeypatch.setattr(setup_env.setup_tests_ui, "run_tests_menu", fake_tests)
    monkeypatch.setattr(setup_env.setup_reinstall_ui, "reinstall_library_menu", fake_reinstall)

    setup_env.run_tests_menu()
    setup_env.reinstall_library_menu()

    assert calls["tests"] == 1
    assert calls["reinstall"] == 1


def test_setup_tests_ui_run_all_executes_pytest_command():
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
        "MenuItem": setup_env.MenuItem,
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

    setup_tests_ui.run_tests_menu(ctx=ctx)

    assert run_cmd_calls == ["uv run python -m pytest tests/"]
    assert waits["count"] == 1
    assert screens["count"] >= 1


def test_setup_reinstall_ui_core_selection_installs_and_restarts():
    fix_libs_calls = []
    fix_uv_calls = {"count": 0}
    restart_calls = {"count": 0}
    waits = {"count": 0}

    def fix_uv():
        fix_uv_calls["count"] += 1

    def interactive_menu(options, **kwargs):
        assert kwargs.get("menu_id") == "reinstall_menu"
        core_child = options[0].children[0]
        uv_option = options[1]
        return [core_child, uv_option]

    ctx = {
        "Style": _DummyStyle,
        "MenuItem": setup_env.MenuItem,
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

    setup_reinstall_ui.reinstall_library_menu(ctx=ctx)

    assert fix_libs_calls == [["lmstudio"]]
    assert fix_uv_calls["count"] == 1
    assert restart_calls["count"] == 1
    assert waits["count"] == 1


def test_setup_reinstall_ui_no_selection_does_nothing():
    calls = {"fix_libs": 0, "fix_uv": 0, "restart": 0, "wait": 0}

    ctx = {
        "Style": _DummyStyle,
        "MenuItem": setup_env.MenuItem,
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

    setup_reinstall_ui.reinstall_library_menu(ctx=ctx)

    assert calls == {"fix_libs": 0, "fix_uv": 0, "restart": 0, "wait": 0}


def test_setup_tests_ui_run_specific_cancel_returns_without_command():
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
        "MenuItem": setup_env.MenuItem,
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

    setup_tests_ui.run_tests_menu(ctx=ctx)

    assert run_cmd_calls == []
    assert menu_calls["specific"] == 1


def test_setup_tests_ui_run_specific_executes_selected_file():
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
        "MenuItem": setup_env.MenuItem,
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

    setup_tests_ui.run_tests_menu(ctx=ctx)

    assert run_cmd_calls == ["uv run python -m pytest tests/test_alpha.py"]
    assert waits["count"] == 1


def test_setup_tests_ui_smoke_without_models_warns_and_returns():
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
        "MenuItem": setup_env.MenuItem,
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

    setup_tests_ui.run_tests_menu(ctx=ctx)

    assert waits["count"] == 1
    assert any("No hay modelos disponibles" in msg for msg in warns)


def test_setup_reinstall_ui_uv_only_does_not_restart_or_install_core():
    calls = {"fix_libs": 0, "fix_uv": 0, "restart": 0, "wait": 0}

    def interactive_menu(options, **kwargs):
        assert kwargs.get("menu_id") == "reinstall_menu"
        uv_option = options[1]
        return [uv_option]

    ctx = {
        "Style": _DummyStyle,
        "MenuItem": setup_env.MenuItem,
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

    setup_reinstall_ui.reinstall_library_menu(ctx=ctx)

    assert calls["fix_uv"] == 1
    assert calls["fix_libs"] == 0
    assert calls["restart"] == 0
    assert calls["wait"] == 1


def test_setup_tests_ui_smoke_import_error_logs_and_recovers(monkeypatch):
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
        "MenuItem": setup_env.MenuItem,
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

    setup_tests_ui.run_tests_menu(ctx=ctx)

    assert any("Could not import smoke test script" in msg for msg in errors)
    assert any("Smoke test failed" in msg for msg in errors)
    assert waits["count"] >= 1


def test_setup_reinstall_ui_multiple_core_libs_grouped_once_and_restart():
    fix_libs_calls = []
    restart_calls = {"count": 0}

    def interactive_menu(options, **kwargs):
        assert kwargs.get("menu_id") == "reinstall_menu"
        core_children = options[0].children
        return [core_children[0], core_children[1]]

    ctx = {
        "Style": _DummyStyle,
        "MenuItem": setup_env.MenuItem,
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

    setup_reinstall_ui.reinstall_library_menu(ctx=ctx)

    assert fix_libs_calls == [["lmstudio", "pytest"]]
    assert restart_calls["count"] == 1


def test_setup_tests_ui_main_menu_back_exits_immediately():
    calls = {"count": 0}

    def interactive_menu(options, **kwargs):
        assert kwargs.get("menu_id") == "tests_manager_menu"
        calls["count"] += 1
        return "BACK"

    ctx = {
        "Style": _DummyStyle,
        "MenuItem": setup_env.MenuItem,
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

    setup_tests_ui.run_tests_menu(ctx=ctx)
    assert calls["count"] == 1


def test_setup_tests_ui_empty_list_choice_is_safe_and_exits_next_loop():
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
        "MenuItem": setup_env.MenuItem,
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

    setup_tests_ui.run_tests_menu(ctx=ctx)
    assert menu_calls["count"] == 1


def test_setup_tests_ui_non_action_choice_does_not_crash():
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
        "MenuItem": setup_env.MenuItem,
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

    setup_tests_ui.run_tests_menu(ctx=ctx)
    assert menu_calls["count"] == 2


def test_setup_tests_ui_run_specific_without_files_warns_and_returns():
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
        "MenuItem": setup_env.MenuItem,
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

    setup_tests_ui.run_tests_menu(ctx=ctx)
    assert any("No tests found in tests/ folder." in msg for msg in warns)
    assert sleeps["count"] == 1


def test_setup_tests_ui_smoke_model_selector_cancel_returns_cleanly():
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
        "MenuItem": setup_env.MenuItem,
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

    setup_tests_ui.run_tests_menu(ctx=ctx)
    assert menu_calls["smoke"] == 1
    assert waits["count"] == 0


def test_setup_tests_ui_smoke_empty_model_tag_then_cancel():
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
        "MenuItem": setup_env.MenuItem,
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

    setup_tests_ui.run_tests_menu(ctx=ctx)
    assert any("No model selected." in msg for msg in warns)
    assert waits["count"] == 1


def test_setup_reinstall_ui_empty_selection_list_behaves_like_cancel():
    calls = {"fix_libs": 0, "fix_uv": 0, "restart": 0, "wait": 0}

    ctx = {
        "Style": _DummyStyle,
        "MenuItem": setup_env.MenuItem,
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

    setup_reinstall_ui.reinstall_library_menu(ctx=ctx)
    assert calls == {"fix_libs": 0, "fix_uv": 0, "restart": 0, "wait": 0}
