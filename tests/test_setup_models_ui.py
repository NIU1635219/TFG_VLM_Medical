from types import SimpleNamespace
from unittest.mock import MagicMock
from src.utils.setup_ui_io import IncrementalPanelRenderer
from src.utils.setup_models_ui import manage_models_menu_ui
from src.utils.menu_kit import TableColumn, TableRow, _build_table_menu_items
import src.utils.setup_models_ui as setup_models_ui

def test_fallback_incremental_panel_renderer():
    mock_clear = MagicMock()
    mock_render_static = MagicMock()
    
    renderer = IncrementalPanelRenderer(
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

def _make_kit_app_for_models(lms_models_ns, lms_helpers, menu_fn=None):
    """Construye mocks (kit, app) para tests de setup_models_ui."""
    class DummyStyle:
        BOLD = ""; ENDC = ""; DIM = ""; OKGREEN = ""; OKCYAN = ""; FAIL = ""

    class DummyMenuItem:
        def __init__(self, label, action=None, description="", children=None):
            self.label = label
            self.action = action
            self.description = description
            self.selectable = True
            self.children = children or []

    kit = MagicMock()
    kit.style = DummyStyle
    kit.MenuItem = DummyMenuItem
    kit.TableColumn = TableColumn
    kit.TableRow = TableRow
    kit.IncrementalPanelRenderer = MagicMock()
    kit.menu = MagicMock(return_value=None) if menu_fn is None else menu_fn
    kit.log = MagicMock()
    kit.clear = MagicMock()
    kit.wait = MagicMock()
    kit.ask = MagicMock(return_value=False)
    kit.input = MagicMock(return_value=None)
    kit.read_key = MagicMock()
    kit.subtitle = MagicMock()
    kit.banner = MagicMock()
    kit.width = MagicMock(return_value=80)
    kit.divider = lambda w: "─" * w
    kit.cursor_memory = {}
    kit.os_module = MagicMock()
    kit.msvcrt_module = None

    def _fake_build_table_items(cols, rows, *, width=None):
        return _build_table_menu_items(
            cols, rows, style=DummyStyle, get_width_fn=kit.width if width is None else lambda: width
        )
    kit.build_table_items = _fake_build_table_items

    app = MagicMock()
    app.print_banner = MagicMock()
    app.lms_models = lms_models_ns
    app.lms_menu_helpers = lms_helpers
    app.psutil = MagicMock()
    app.MODELS_REGISTRY = {}
    app.get_installed_lms_models = MagicMock(return_value=[])

    return kit, app


def test_manage_models_menu_ui_exit():
    lms_helpers = SimpleNamespace(
        detect_gpu_memory_bytes=lambda: 16 * 1024 ** 3,
        detect_ram_memory_bytes=lambda _psutil: 32 * 1024 ** 3,
        detect_local_model_quantization=lambda _model: "Q4_K_M",
    )
    lms_models_ns = SimpleNamespace(
        list_installed_variants_flat=lambda: [],
        list_local_llm_models=lambda: [],
        list_loaded_llm_model_keys=lambda: set(),
    )


def test_manage_models_table_separator_is_dynamic():
    lms_helpers = SimpleNamespace(
        detect_gpu_memory_bytes=lambda: 16 * 1024 ** 3,
        detect_ram_memory_bytes=lambda _psutil: 32 * 1024 ** 3,
        detect_local_model_quantization=lambda _model: "Q4_K_M",
    )
    lms_models_ns = SimpleNamespace(
        list_installed_variants_flat=lambda: [{"model_key": "a/model-q4", "display_name": "A", "path": "p", "size_bytes": 0, "quantization": "Q4"}],
        list_local_llm_models=lambda: [{"model_key": "a/model-q4", "display_name": "A", "path": "p"}],
        list_loaded_llm_model_keys=lambda: set(),
    )

    width_state = {"value": 70}
    captured = {"options": None}

    def fake_menu(options, **kwargs):
        captured["options"] = options
        return None

    kit, app = _make_kit_app_for_models(lms_models_ns, lms_helpers, menu_fn=fake_menu)
    kit.width = lambda: width_state["value"]  # dinámico

    manage_models_menu_ui(kit, app)

    assert captured["options"] is not None
    table_separator = captured["options"][1]

    first_separator = table_separator.dynamic_label(False)
    assert len(first_separator) == 70

    width_state["value"] = 82
    second_separator = table_separator.dynamic_label(False)
    assert len(second_separator) == 82
