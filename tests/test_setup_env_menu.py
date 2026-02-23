import builtins

import setup_env


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

    monkeypatch.setattr(setup_env.os, "name", "nt", raising=False)
    monkeypatch.setattr(setup_env, "msvcrt", FakeMsvcrt, raising=False)

    called = {"input": False}

    def fake_input(*args, **kwargs):
        called["input"] = True
        return ""

    monkeypatch.setattr(builtins, "input", fake_input)

    setup_env.wait_for_any_key("Press any key...")
    assert called["input"] is False


def test_ask_user_uses_arrow_selection(monkeypatch):
    """Verifica que `ask_user` maneje selección con flechas (simulada) correctamente."""
    # default = n => selected option should be "No"
    key_sequence = iter(["ENTER"])

    def fake_read_key():
        return next(key_sequence)

    monkeypatch.setattr(setup_env, "read_key", fake_read_key)
    monkeypatch.setattr(setup_env, "clear_screen_ansi", lambda: None)

    assert setup_env.ask_user("Confirm?", default="n") is False


def test_wait_for_any_key_fallback_non_windows(monkeypatch):
    """Asegura el fallback a `input()` en sistemas no Windows."""
    monkeypatch.setattr(setup_env.os, "name", "posix", raising=False)

    called = {"input": False}

    def fake_input(*args, **kwargs):
        called["input"] = True
        return ""

    monkeypatch.setattr(builtins, "input", fake_input)
    setup_env.wait_for_any_key("Press any key...")
    assert called["input"] is True


def test_factory_reset_confirmation_defaults_to_no(monkeypatch):
    """Valida que la opción de Factory Reset tenga 'No' como valor predeterminado."""
    captured_default = {"value": ""}

    def fake_ask_user(question, default="y"):
        if "delete .venv" in question:
            captured_default["value"] = default
            return False
        return False

    calls = {"count": 0}

    def fake_interactive_menu(options, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return next(opt for opt in options if getattr(opt, "label", "") == " Factory Reset")
        return None

    monkeypatch.setattr(setup_env, "ask_user", fake_ask_user)
    monkeypatch.setattr(setup_env, "print_banner", lambda: None)
    monkeypatch.setattr(setup_env, "clear_screen_ansi", lambda: None)
    monkeypatch.setattr(setup_env, "interactive_menu", fake_interactive_menu)

    try:
        setup_env.show_menu()
    except SystemExit:
        pass

    assert captured_default["value"] == "n"


def test_interactive_menu_shows_selected_description(monkeypatch, capsys):
    """Comprueba que el menú interactivo muestre la descripción de la opción seleccionada."""
    keys = iter(["ENTER"])

    monkeypatch.setattr(setup_env, "read_key", lambda: next(keys))
    monkeypatch.setattr(setup_env, "clear_screen_ansi", lambda: None)
    monkeypatch.setattr(setup_env.os, "system", lambda *_: 0)

    option = setup_env.MenuItem("Option A", description="Descripción de prueba")
    result = setup_env.interactive_menu([option], header_func=lambda: None)
    output = capsys.readouterr().out

    assert result is option
    assert "Descripción: Descripción de prueba" in output


def test_interactive_menu_hides_description_when_missing(monkeypatch, capsys):
    """Verifica que no se muestre el campo 'Descripción' si la opción no la tiene."""
    keys = iter(["ENTER"])

    monkeypatch.setattr(setup_env, "read_key", lambda: next(keys))
    monkeypatch.setattr(setup_env, "clear_screen_ansi", lambda: None)
    monkeypatch.setattr(setup_env.os, "system", lambda *_: 0)

    option = setup_env.MenuItem("Option A")
    setup_env.interactive_menu([option], header_func=lambda: None)
    output = capsys.readouterr().out

    assert "Descripción:" not in output


def test_stop_lms_server_if_owned_only_stops_when_started(monkeypatch):
    """Asegura que `stop_lms_server_if_owned` solo detenga el servidor si esta sesión lo inició."""
    calls = {"stop": 0}

    def fake_stop_server():
        calls["stop"] += 1
        return True

    monkeypatch.setattr(setup_env.lms_models, "stop_server", fake_stop_server)
    monkeypatch.setattr(setup_env, "log", lambda *args, **kwargs: None)

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

    def fake_show():
        calls["show"] += 1

    def fake_stop():
        calls["stop"] += 1

    monkeypatch.setattr(setup_env, "ensure_lms_server_running", fake_start)
    monkeypatch.setattr(setup_env, "show_menu", fake_show)
    monkeypatch.setattr(setup_env, "stop_lms_server_if_owned", fake_stop)

    setup_env.main()

    assert calls["start"] == 1
    assert calls["show"] == 1
    assert calls["stop"] == 1


def test_run_diagnostics_ui_passes_read_key_in_context(monkeypatch):
    """Asegura que setup_env pase read_key al contexto de setup_diagnostics.run_diagnostics_ui."""
    captured = {"ctx": None}

    def fake_run_diagnostics_ui(ctx):
        captured["ctx"] = ctx
        return None

    monkeypatch.setattr(setup_env.setup_diagnostics, "run_diagnostics_ui", fake_run_diagnostics_ui)

    setup_env.run_diagnostics_ui()

    assert captured["ctx"] is not None
    assert "read_key" in captured["ctx"]
    assert captured["ctx"]["read_key"] is setup_env.read_key
