import builtins

import setup_env


def test_is_model_installed_does_not_mix_8b_and_latest():
    installed = ["qwen3-vl:latest"]
    assert setup_env.is_model_installed("qwen3-vl:8b", installed) is False


def test_is_model_installed_accepts_latest_alias_without_tag():
    installed = ["qwen3-vl:latest"]
    assert setup_env.is_model_installed("qwen3-vl", installed) is True


def test_get_manageable_models_keeps_distinct_tags():
    installed = ["qwen3-vl:latest", "qwen3-vl:8b"]
    entries = setup_env.get_manageable_models(installed)
    tags = [entry["model_tag"] for entry in entries]

    assert "qwen3-vl:8b" in tags
    assert "qwen3-vl:latest" in tags


def test_wait_for_any_key_windows_uses_getch(monkeypatch):
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
    # default = n => selected option should be "No"
    key_sequence = iter(["ENTER"])

    def fake_read_key():
        return next(key_sequence)

    monkeypatch.setattr(setup_env, "read_key", fake_read_key)
    monkeypatch.setattr(setup_env, "clear_screen_ansi", lambda: None)

    assert setup_env.ask_user("Confirm?", default="n") is False


def test_wait_for_any_key_fallback_non_windows(monkeypatch):
    monkeypatch.setattr(setup_env.os, "name", "posix", raising=False)

    called = {"input": False}

    def fake_input(*args, **kwargs):
        called["input"] = True
        return ""

    monkeypatch.setattr(builtins, "input", fake_input)
    setup_env.wait_for_any_key("Press any key...")
    assert called["input"] is True


def test_factory_reset_confirmation_defaults_to_no(monkeypatch):
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
    keys = iter(["ENTER"])

    monkeypatch.setattr(setup_env, "read_key", lambda: next(keys))
    monkeypatch.setattr(setup_env, "clear_screen_ansi", lambda: None)
    monkeypatch.setattr(setup_env.os, "system", lambda *_: 0)

    option = setup_env.MenuItem("Option A")
    setup_env.interactive_menu([option], header_func=lambda: None)
    output = capsys.readouterr().out

    assert "Descripción:" not in output
