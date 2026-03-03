import builtins

import setup_env
from src.utils import setup_ui_io, setup_menu_engine


def test_is_model_installed_does_not_mix_8b_and_latest():
    """Valida que `is_model_installed` distinga entre versiones/tags específicos (ej. 8b vs latest)."""
    installed = ["qwen3-vl:latest"]
    assert setup_env.lms_menu_helpers.is_model_installed("qwen3-vl:8b", installed) is False


def test_is_model_installed_accepts_latest_alias_without_tag():
    """Comprueba que si no se especifica tag, se asuma 'latest' o coincidencia parcial."""
    installed = ["qwen3-vl:latest"]
    assert setup_env.lms_menu_helpers.is_model_installed("qwen3-vl", installed) is True


def test_wait_for_any_key_windows_uses_getch(monkeypatch):
    """Prueba que en Windows se use `msvcrt.getch()` para esperar una tecla."""
    class FakeMsvcrt:
        @staticmethod
        def getch():
            return b"x"

    called = {"input": False}

    def fake_input(*args, **kwargs):
        called["input"] = True
        return ""

    import os as _os
    monkeypatch.setattr(builtins, "input", fake_input)

    setup_ui_io.wait_for_any_key(
        message="Press any key...",
        style=setup_env.Style,
        os_module=type("_OS", (), {"name": "nt"})(),
        msvcrt_module=FakeMsvcrt,
    )
    assert called["input"] is False


def test_ask_user_uses_arrow_selection(monkeypatch):
    """Verifica que `ask_user` maneje selección con flechas (simulada) correctamente."""
    # default = n => selected option should be "No"
    key_sequence = iter(["ENTER"])

    def fake_read_key():
        return next(key_sequence)

    result = setup_ui_io.ask_user(
        question="Confirm?",
        default="n",
        style=setup_env.Style,
        read_key_fn=fake_read_key,
        clear_screen_fn=lambda: None,
        info_text="",
    )
    assert result is False


def test_wait_for_any_key_fallback_non_windows(monkeypatch):
    """Asegura el fallback a `input()` en sistemas no Windows."""
    called = {"input": False}

    def fake_input(*args, **kwargs):
        called["input"] = True
        return ""

    monkeypatch.setattr(builtins, "input", fake_input)
    setup_ui_io.wait_for_any_key(
        message="Press any key...",
        style=setup_env.Style,
        os_module=type("_OS", (), {"name": "posix"})(),
        msvcrt_module=None,
    )
    assert called["input"] is True


def test_factory_reset_confirmation_defaults_to_no(monkeypatch):
    """Valida que la opción de Factory Reset tenga 'No' como valor predeterminado."""
    captured_default = {"value": ""}

    def fake_ask(question, default="y", info_text=""):
        if "delete .venv" in question:
            captured_default["value"] = default
            return False
        return False

    calls = {"count": 0}

    def fake_menu(options, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return next(opt for opt in options if getattr(opt, "label", "") == " Factory Reset")
        return None

    monkeypatch.setattr(setup_env._kit, "ask", fake_ask)
    monkeypatch.setattr(setup_env._kit, "menu", fake_menu)
    monkeypatch.setattr(setup_env._app, "print_banner", lambda: None)
    monkeypatch.setattr(setup_env._kit, "subtitle", lambda *_: None)
    monkeypatch.setattr(setup_env._kit, "clear", lambda: None)

    try:
        setup_env.show_menu()
    except SystemExit:
        pass

    assert captured_default["value"] == "n"


def test_interactive_menu_shows_selected_description(monkeypatch, capsys):
    """Comprueba que el menú interactivo muestre la descripción de la opción seleccionada."""
    keys = iter(["ENTER"])

    monkeypatch.setattr(setup_env._kit, "read_key", lambda: next(keys))
    monkeypatch.setattr(setup_env._kit, "clear", lambda: None)
    monkeypatch.setattr(setup_env.os, "system", lambda *_: 0)

    option = setup_menu_engine.MenuItem("Option A", description="Descripción de prueba")
    result = setup_env._kit.menu([option], header_func=lambda: None)
    output = capsys.readouterr().out

    assert result is option
    assert "Descripción: Descripción de prueba" in output


def test_interactive_menu_hides_description_when_missing(monkeypatch, capsys):
    """Verifica que no se muestre el campo 'Descripción' si la opción no la tiene."""
    keys = iter(["ENTER"])

    monkeypatch.setattr(setup_env._kit, "read_key", lambda: next(keys))
    monkeypatch.setattr(setup_env._kit, "clear", lambda: None)
    monkeypatch.setattr(setup_env.os, "system", lambda *_: 0)

    option = setup_menu_engine.MenuItem("Option A")
    setup_env._kit.menu([option], header_func=lambda: None)
    output = capsys.readouterr().out

    assert "Descripción:" not in output


def test_stop_lms_server_if_owned_only_stops_when_started(monkeypatch):
    """Asegura que `stop_lms_server_if_owned` solo detenga el servidor si esta sesión lo inició."""
    calls = {"stop": 0}

    def fake_stop_server():
        calls["stop"] += 1
        return True

    monkeypatch.setattr(setup_env.lms_models, "stop_server", fake_stop_server)
    monkeypatch.setattr(setup_env._kit, "log", lambda *args, **kwargs: None)

    setup_env.LMS_SERVER_STARTED_BY_THIS_SESSION = False
    setup_env.stop_lms_server_if_owned()
    assert calls["stop"] == 0

    setup_env.LMS_SERVER_STARTED_BY_THIS_SESSION = True
    setup_env.stop_lms_server_if_owned()
    assert calls["stop"] == 1


def test_main_wraps_show_menu_with_server_lifecycle(monkeypatch):
    """Prueba la integración del ciclo de vida del servidor (start -> menu -> stop) en `main`."""
    calls = {"start": 0, "show": 0, "stop": 0}

    monkeypatch.setattr(setup_env.os.path, "exists", lambda *_: True)
    monkeypatch.setattr(setup_env.os.path, "samefile", lambda *_: True)

    def fake_start():
        calls["start"] += 1
        return True

    def fake_stop():
        calls["stop"] += 1

    monkeypatch.setattr(setup_env._app, "ensure_lms_server_running", fake_start)
    monkeypatch.setattr(setup_env._app, "stop_lms_server_if_owned", fake_stop)

    import src.utils.setup_install_flow as _sif
    monkeypatch.setattr(_sif, "show_menu", lambda kit, app: calls.__setitem__("show", calls["show"] + 1))

    setup_env.main()

    assert calls["start"] == 1
    assert calls["show"] == 1
    assert calls["stop"] == 1


def test_run_diagnostics_ui_delegates_to_module(monkeypatch):
    """Asegura que setup_env delegue run_diagnostics_ui al módulo con kit y app."""
    captured = {"kit": None, "app": None}

    def fake_run_diagnostics_ui(kit, app):
        captured["kit"] = kit
        captured["app"] = app

    monkeypatch.setattr(setup_env.setup_diagnostics, "run_diagnostics_ui", fake_run_diagnostics_ui)

    setup_env.run_diagnostics_ui()

    assert captured["kit"] is setup_env._kit
    assert captured["app"] is setup_env._app
