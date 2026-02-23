from types import SimpleNamespace
from unittest.mock import MagicMock
from src.utils.setup_models_ui import _FallbackIncrementalPanelRenderer, manage_models_menu_ui
import src.utils.setup_models_ui as setup_models_ui

def test_fallback_incremental_panel_renderer():
    mock_clear = MagicMock()
    mock_render_static = MagicMock()
    
    renderer = _FallbackIncrementalPanelRenderer(
        clear_screen_fn=mock_clear,
        render_static_fn=mock_render_static
    )
    
    # Initial render
    renderer.render(["line1", "line2"])
    assert mock_clear.called
    assert mock_render_static.called
    assert renderer.static_rendered is True
    
    # Reset
    renderer.reset()
    assert renderer.static_rendered is False
    
    # Render again
    renderer.render(["line3"])
    assert mock_clear.call_count == 2

def test_manage_models_menu_ui_exit():
    class DummyStyle:
        BOLD = ""
        ENDC = ""
        DIM = ""
        OKGREEN = ""
        OKCYAN = ""
        FAIL = ""

    class DummyMenuItem:
        def __init__(self, label, action=None, description=""):
            self.label = label
            self.action = action
            self.description = description
            self.selectable = True

    class DummyMenuStaticItem(DummyMenuItem):
        def __init__(self, label="", description=""):
            super().__init__(label, action=None, description=description)
            self.selectable = False

    class DummyMenuSeparator:
        def __init__(self, width=10):
            self.width = width
            self.selectable = False
            self.dynamic_label = lambda _is_selected: "─" * width

    lms_helpers = SimpleNamespace(
        detect_gpu_memory_bytes=lambda: 16 * 1024 * 1024 * 1024,
        detect_ram_memory_bytes=lambda _psutil: 32 * 1024 * 1024 * 1024,
        detect_local_model_quantization=lambda _model: "Q4_K_M",
    )

    lms_models = SimpleNamespace(
        list_local_llm_models=lambda: [],
        list_loaded_llm_model_keys=lambda: set(),
    )

    # Mock context
    ctx = {
        "log": MagicMock(),
        "clear_screen": MagicMock(),
        "interactive_menu": MagicMock(return_value=None),
        "get_installed_lms_models": MagicMock(return_value=[]),
        "read_key": MagicMock(),
        "run_cmd": MagicMock(),
        "check_lms": MagicMock(return_value=True),
        "get_server_status": MagicMock(return_value=(True, "Running")),
        "psutil": MagicMock(),
        "lms_menu_helpers": lms_helpers,
        "lms_models": lms_models,
        "Style": DummyStyle,
        "MenuItem": DummyMenuItem,
        "MenuStaticItem": DummyMenuStaticItem,
        "MenuSeparator": DummyMenuSeparator,
        "MODELS_REGISTRY": [],
        "MENU_CURSOR_MEMORY": {},
        "print_banner": MagicMock(),
        "clear_screen_ansi": MagicMock(),
        "input_with_esc": MagicMock(return_value=None),
        "wait_for_any_key": MagicMock(),
        "ask_user": MagicMock(return_value=False),
        "IncrementalPanelRenderer": MagicMock()
    }
    
    manage_models_menu_ui(ctx)
    
    assert ctx["interactive_menu"].called


def test_manage_models_table_separator_is_dynamic(monkeypatch):
    class DummyStyle:
        BOLD = ""
        ENDC = ""
        DIM = ""
        OKGREEN = ""
        OKCYAN = ""
        FAIL = ""

    class DummyMenuItem:
        def __init__(self, label, action=None, description=""):
            self.label = label
            self.action = action
            self.description = description
            self.selectable = True

    class DummyMenuStaticItem(DummyMenuItem):
        def __init__(self, label="", description=""):
            super().__init__(label, action=None, description=description)
            self.selectable = False

    class DummyMenuSeparator:
        def __init__(self, width=10):
            self.width = width
            self.selectable = False
            self.dynamic_label = lambda _is_selected: "─" * width

    lms_helpers = SimpleNamespace(
        detect_gpu_memory_bytes=lambda: 16 * 1024 * 1024 * 1024,
        detect_ram_memory_bytes=lambda _psutil: 32 * 1024 * 1024 * 1024,
        detect_local_model_quantization=lambda _model: "Q4_K_M",
    )

    lms_models = SimpleNamespace(
        list_local_llm_models=lambda: [{"model_key": "a/model-q4", "display_name": "A", "path": "p"}],
        list_loaded_llm_model_keys=lambda: set(),
    )

    width_state = {"value": 70}

    def fake_get_ui_width(*, shutil_module=None, max_width=86, min_width=60):
        _ = (shutil_module, max_width, min_width)
        return width_state["value"]

    monkeypatch.setattr(setup_models_ui, "get_ui_width", fake_get_ui_width)

    captured = {"options": None}

    def fake_interactive_menu(options, **kwargs):
        captured["options"] = options
        return None

    ctx = {
        "log": MagicMock(),
        "clear_screen": MagicMock(),
        "interactive_menu": fake_interactive_menu,
        "get_installed_lms_models": MagicMock(return_value=[]),
        "read_key": MagicMock(),
        "run_cmd": MagicMock(),
        "check_lms": MagicMock(return_value=True),
        "get_server_status": MagicMock(return_value=(True, "Running")),
        "psutil": MagicMock(),
        "lms_menu_helpers": lms_helpers,
        "lms_models": lms_models,
        "Style": DummyStyle,
        "MenuItem": DummyMenuItem,
        "MenuStaticItem": DummyMenuStaticItem,
        "MenuSeparator": DummyMenuSeparator,
        "MODELS_REGISTRY": [],
        "MENU_CURSOR_MEMORY": {},
        "print_banner": MagicMock(),
        "clear_screen_ansi": MagicMock(),
        "input_with_esc": MagicMock(return_value=None),
        "wait_for_any_key": MagicMock(),
        "ask_user": MagicMock(return_value=False),
        "IncrementalPanelRenderer": MagicMock(),
    }

    manage_models_menu_ui(ctx)

    assert captured["options"] is not None
    table_separator = captured["options"][1]
    first_separator = table_separator.dynamic_label(False)
    assert len(first_separator) == 70

    width_state["value"] = 82
    second_separator = table_separator.dynamic_label(False)
    assert len(second_separator) == 82
