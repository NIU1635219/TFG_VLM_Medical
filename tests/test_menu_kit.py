"""Tests para src/utils/menu_kit.py — UIKit y AppContext."""

import pytest
from unittest.mock import MagicMock
from types import SimpleNamespace

from src.utils.menu_kit import UIKit, AppContext
from src.utils.menu_kit import TableColumn, TableRow, _compute_col_widths, _build_table_menu_items
from src.utils import setup_menu_engine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Style:
    BOLD = ENDC = DIM = OKCYAN = OKGREEN = WARNING = FAIL = HEADER = SELECTED = ""


def _make_kit(cursor_memory=None) -> UIKit:
    """Crea un UIKit mínimo con módulos falsos."""
    fake_shutil = SimpleNamespace(get_terminal_size=lambda: SimpleNamespace(columns=100, lines=30))
    fake_os = SimpleNamespace(name="posix", cpu_count=lambda: 4)
    fake_sys = SimpleNamespace(stdout=MagicMock())
    return UIKit(
        style=_Style,
        os_module=fake_os,
        sys_module=fake_sys,
        shutil_module=fake_shutil,
        msvcrt_module=None,
        cursor_memory=cursor_memory or {},
    )


def _make_app(**overrides) -> AppContext:
    """Crea un AppContext con valores mínimos válidos."""
    defaults = dict(
        REQUIRED_LIBS=["lib-a"],
        LIB_IMPORT_MAP={"lib-a": "lib_a"},
        MODELS_REGISTRY={"model-x": {"name": "https://example.com", "description": "test"}},
        lms_models=MagicMock(),
        lms_menu_helpers=MagicMock(),
        psutil=None,
        os_module=MagicMock(),
        sys_module=MagicMock(),
        subprocess_module=MagicMock(),
        time_module=MagicMock(),
        print_banner=lambda: None,
        fix_folders=lambda: None,
        fix_uv=lambda: None,
        fix_libs=lambda: None,
        check_lms=lambda: True,
        restart_program=lambda: None,
        get_installed_lms_models=lambda: [],
        list_test_files=lambda: [],
        ensure_lms_server_running=lambda: True,
        stop_lms_server_if_owned=lambda: None,
    )
    defaults.update(overrides)
    return AppContext(**defaults)


# ---------------------------------------------------------------------------
# UIKit construcción
# ---------------------------------------------------------------------------

class TestUIKitConstruction:
    def test_attributes_stored(self):
        kit = _make_kit()
        assert kit.style is _Style
        assert kit.msvcrt_module is None

    def test_cursor_memory_default(self):
        kit = _make_kit()
        assert isinstance(kit.cursor_memory, dict)

    def test_shared_cursor_memory(self):
        memory = {"test_key": 42}
        kit = _make_kit(cursor_memory=memory)
        assert kit.cursor_memory["test_key"] == 42


# ---------------------------------------------------------------------------
# UIKit.width
# ---------------------------------------------------------------------------

class TestUIKitWidth:
    def test_returns_integer(self):
        kit = _make_kit()
        assert isinstance(kit.width(), int)

    def test_respects_min_width(self):
        tiny_shutil = SimpleNamespace(
            get_terminal_size=lambda: SimpleNamespace(columns=10, lines=24)
        )
        kit = UIKit(
            style=_Style,
            os_module=MagicMock(),
            sys_module=MagicMock(),
            shutil_module=tiny_shutil,
            msvcrt_module=None,
            cursor_memory={},
        )
        assert kit.width(min_width=60) >= 60


# ---------------------------------------------------------------------------
# UIKit.wrap
# ---------------------------------------------------------------------------

class TestUIKitWrap:
    def test_short_text_single_line(self):
        kit = _make_kit()
        result = kit.wrap("hello", 80)
        assert result == ["hello"]

    def test_wraps_long_text(self):
        kit = _make_kit()
        result = kit.wrap("a " * 50, 20)
        assert len(result) > 1
        for line in result:
            assert len(line) <= 20


# ---------------------------------------------------------------------------
# UIKit.divider
# ---------------------------------------------------------------------------

class TestUIKitDivider:
    def test_returns_string_of_dashes(self):
        kit = _make_kit()
        d = kit.divider(40)
        assert "─" in d

    def test_default_width_from_terminal(self):
        kit = _make_kit()
        d = kit.divider()
        assert len(d) > 0


# ---------------------------------------------------------------------------
# UIKit.table — lógica de tabla integrada en el método
# ---------------------------------------------------------------------------


class TestUIKitTable:
    def test_produces_box_drawing_chars(self, capsys):
        kit = _make_kit()
        kit.table([("Python", "3.13", "OK")], width=80)
        out = capsys.readouterr().out
        assert "┌" in out
        assert "│" in out
        assert "└" in out

    def test_ok_status_appears(self, capsys):
        kit = _make_kit()
        kit.table([("comp", "value", "OK")], width=80)
        assert "OK" in capsys.readouterr().out

    def test_warn_status_appears(self, capsys):
        kit = _make_kit()
        kit.table([("comp", "degraded", "WARN")], width=80)
        assert "WARN" in capsys.readouterr().out

    def test_fail_status_appears(self, capsys):
        kit = _make_kit()
        kit.table([("comp", "missing", "FAIL")], width=80)
        assert "FAIL" in capsys.readouterr().out

    def test_all_statuses_in_one_table(self, capsys):
        kit = _make_kit()
        rows = [("A", "v", "OK"), ("B", "v", "WARN"), ("C", "v", "FAIL")]
        kit.table(rows, width=80)
        out = capsys.readouterr().out
        assert all(s in out for s in ["OK", "WARN", "FAIL"])

    def test_cv2_aliased_to_opencv(self, capsys):
        kit = _make_kit()
        kit.table([("cv2", "0.4.8", "OK")], width=80)
        assert "OpenCV" in capsys.readouterr().out

    def test_llama_cpp_aliased(self, capsys):
        kit = _make_kit()
        kit.table([("llama_cpp", "0.2.0", "OK")], width=80)
        assert "LlamaCPP" in capsys.readouterr().out

    def test_multiple_rows_all_printed(self, capsys):
        kit = _make_kit()
        rows = [("A", "1", "OK"), ("B", "2", "WARN"), ("C", "3", "FAIL")]
        kit.table(rows, width=80)
        out = capsys.readouterr().out
        assert all(c in out for c in ["A", "B", "C"])

    def test_empty_rows_no_crash(self, capsys):
        """tabla vacía: sólo cabecera, sin excepción."""
        kit = _make_kit()
        kit.table([], width=80)
        out = capsys.readouterr().out
        assert "┌" in out

    def test_narrow_width_does_not_crash(self, capsys):
        """Ancho mínimo (frontera): no debe lanzar error."""
        kit = _make_kit()
        kit.table([("comp", "val", "OK")], width=10)
        assert capsys.readouterr().out

    def test_width_32_boundary(self, capsys):
        """Presupuesto de contenido con width=32 (mínimo de max(32, ...))."""
        kit = _make_kit()
        kit.table([("comp", "val", "OK")], width=32)
        assert "OK" in capsys.readouterr().out

    def test_header_columns_present(self, capsys):
        kit = _make_kit()
        kit.table([("comp", "val", "OK")], width=80)
        out = capsys.readouterr().out
        assert "Component" in out
        assert "Status/Value" in out
        assert "Res" in out

    def test_long_component_name_no_crash(self, capsys):
        """Nombres muy largos no deben lanzar excepción."""
        kit = _make_kit()
        kit.table([("A" * 200, "v", "OK")], width=80)
        assert "OK" in capsys.readouterr().out

    def test_unknown_status_no_crash(self, capsys):
        """Estado desconocido cae en rama FAIL sin excepción."""
        kit = _make_kit()
        kit.table([("comp", "val", "UNKNOWN")], width=80)
        assert "UNKNOWN" in capsys.readouterr().out

    def test_width_none_uses_terminal_width(self, capsys):
        """Sin width explícito usa self.width() sin crash."""
        kit = _make_kit()
        kit.table([("comp", "val", "OK")])
        assert "OK" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# UIKit — re-exports de MenuItem
# ---------------------------------------------------------------------------

class TestUIKitMenuItemAccess:
    def test_kit_has_MenuItem(self):
        kit = _make_kit()
        assert kit.MenuItem is setup_menu_engine.MenuItem

    def test_kit_has_MenuSeparator(self):
        kit = _make_kit()
        assert kit.MenuSeparator is setup_menu_engine.MenuSeparator


# ---------------------------------------------------------------------------
# AppContext construcción
# ---------------------------------------------------------------------------

class TestAppContext:
    def test_required_fields_stored(self):
        app = _make_app()
        assert app.REQUIRED_LIBS == ["lib-a"]
        assert "lib-a" in app.LIB_IMPORT_MAP
        assert "model-x" in app.MODELS_REGISTRY

    def test_callable_fields_invocable(self):
        app = _make_app()
        # No deben lanzar
        app.print_banner()
        app.fix_folders()
        app.fix_uv()
        app.fix_libs()
        app.restart_program()
        app.stop_lms_server_if_owned()

    def test_menu_cursor_memory_defaults_to_empty_dict(self):
        app = _make_app()
        assert isinstance(app.MENU_CURSOR_MEMORY, dict)

    def test_custom_cursor_memory_stored(self):
        memory = {"cursor": 3}
        app = _make_app(MENU_CURSOR_MEMORY=memory)
        assert app.MENU_CURSOR_MEMORY["cursor"] == 3


# ---------------------------------------------------------------------------
# TableColumn / TableRow — dataclasses
# ---------------------------------------------------------------------------

class TestTableColumn:
    def test_defaults(self):
        col = TableColumn("MODEL")
        assert col.label == "MODEL"
        assert col.fixed_width is None
        assert col.width_ratio is None
        assert col.min_width == 8

    def test_fixed_width(self):
        col = TableColumn("QUANT", fixed_width=12)
        assert col.fixed_width == 12

    def test_width_ratio(self):
        col = TableColumn("DESC", width_ratio=0.5)
        assert col.width_ratio == 0.5


class TestTableRow:
    def test_defaults(self):
        row = TableRow(cells=["A", "B"])
        assert row.cells == ["A", "B"]
        assert row.action is None
        assert row.description == ""
        assert row.selected_cells is None
        assert row.cell_colors is None
        assert row.selected_cell_colors is None

    def test_with_action(self):
        fn = lambda: None
        row = TableRow(cells=["A"], action=fn)
        assert row.action is fn

    def test_with_colors(self):
        row = TableRow(cells=["A", "B"], cell_colors=["OKGREEN", None])
        assert row.cell_colors == ["OKGREEN", None]


# ---------------------------------------------------------------------------
# _compute_col_widths
# ---------------------------------------------------------------------------

class TestComputeColWidths:
    def test_all_fixed(self):
        cols = [TableColumn("A", fixed_width=10), TableColumn("B", fixed_width=20)]
        widths = _compute_col_widths(cols, 100)
        assert widths == [10, 20]

    def test_flex_shares_space(self):
        """Dos columnas sin ancho fijo deben repartirse el espacio restante."""
        cols = [TableColumn("A"), TableColumn("B")]
        widths = _compute_col_widths(cols, 50)
        assert len(widths) == 2
        assert all(w >= 8 for w in widths)  # min_width respetado

    def test_mixed_fixed_and_flex(self):
        cols = [
            TableColumn("FIXED", fixed_width=15),
            TableColumn("FLEX"),
        ]
        # separador 1 × " │ " (3 chars) → espacio flex = 50 - 15 - 3 = 32
        widths = _compute_col_widths(cols, 50)
        assert widths[0] == 15
        assert widths[1] >= 8

    def test_respects_min_width(self):
        """Con ancho total muy pequeño se aplica el min_width de cada columna."""
        cols = [TableColumn("A", min_width=10), TableColumn("B", min_width=10)]
        widths = _compute_col_widths(cols, 5)  # ancho ridículo
        assert widths[0] >= 10
        assert widths[1] >= 10

    def test_ratio_proportional(self):
        cols = [
            TableColumn("BIG", width_ratio=2.0),
            TableColumn("SMALL", width_ratio=1.0),
        ]
        widths = _compute_col_widths(cols, 93)  # 93 - 3 (sep) = 90 → big≈60, small≈30
        assert widths[0] > widths[1]


# ---------------------------------------------------------------------------
# _build_table_menu_items
# ---------------------------------------------------------------------------

class TestBuildTableMenuItems:
    def _items(self, rows=None, *, width=80):
        cols = [TableColumn("COL1", fixed_width=20), TableColumn("COL2", fixed_width=15)]
        if rows is None:
            rows = [TableRow(cells=["hello", "world"], action=lambda: None)]
        return _build_table_menu_items(cols, rows, style=_Style, get_width_fn=lambda: width)

    def test_returns_list(self):
        assert isinstance(self._items(), list)

    def test_structure_header_sep_row_sep(self):
        items = self._items()
        assert len(items) == 4  # header, top_sep, row, bot_sep

    def test_row_item_is_menu_item(self):
        items = self._items()
        assert isinstance(items[2], setup_menu_engine.MenuItem)

    def test_static_row_no_action(self):
        rows = [TableRow(cells=["A", "B"], action=None)]
        items = _build_table_menu_items(
            [TableColumn("C1", fixed_width=10), TableColumn("C2", fixed_width=10)],
            rows, style=_Style, get_width_fn=lambda: 40,
        )
        assert isinstance(items[2], setup_menu_engine.MenuStaticItem)

    def test_header_dynamic_label(self):
        items = self._items(width=80)
        label = items[0].dynamic_label(False)
        assert "COL1" in label
        assert "COL2" in label

    def test_separator_length_matches_width(self):
        items = self._items(width=70)
        sep_label = items[1].dynamic_label(False)
        # strip ANSI (en _Style todo es "") → "─" * 70
        assert len(sep_label) == 70

    def test_separator_resize(self):
        w = {"v": 60}
        cols = [TableColumn("A", fixed_width=20)]
        rows = [TableRow(cells=["x"], action=None)]
        items = _build_table_menu_items(cols, rows, style=_Style, get_width_fn=lambda: w["v"])
        assert len(items[1].dynamic_label(False)) == 60
        w["v"] = 90
        assert len(items[1].dynamic_label(False)) == 90

    def test_row_item_has_table_row_attr(self):
        rows = [TableRow(cells=["A", "B"], action=lambda: None)]
        items = self._items(rows=rows)
        assert items[2]._table_row is rows[0]

    def test_selected_cells_used_when_selected(self):
        rows = [TableRow(cells=["A", "B"], selected_cells=["SEL_A", "SEL_B"], action=lambda: None)]
        items = self._items(rows=rows)
        normal = items[2].dynamic_label(False)
        selected = items[2].dynamic_label(True)
        assert "SEL_A" in selected
        assert "A" in normal

    def test_multiple_rows(self):
        rows = [
            TableRow(cells=["R1", "v1"], action=lambda: None),
            TableRow(cells=["R2", "v2"], action=lambda: None),
        ]
        items = self._items(rows=rows)
        assert len(items) == 5  # header, sep, row1, row2, sep


# ---------------------------------------------------------------------------
# UIKit.build_table_items
# ---------------------------------------------------------------------------

class TestUIKitBuildTableItems:
    def test_returns_items_list(self):
        kit = _make_kit()
        cols = [TableColumn("A", fixed_width=20), TableColumn("B", fixed_width=15)]
        rows = [TableRow(cells=["hello", "world"], action=lambda: None)]
        items = kit.build_table_items(cols, rows)
        assert isinstance(items, list)
        assert len(items) >= 2  # al menos header + sep

    def test_class_attrs_reexport(self):
        kit = _make_kit()
        assert kit.TableColumn is TableColumn
        assert kit.TableRow is TableRow

    def test_custom_width(self):
        kit = _make_kit()
        cols = [TableColumn("A", fixed_width=10)]
        rows = [TableRow(cells=["x"], action=None)]
        items = kit.build_table_items(cols, rows, width=55)
        sep_label = items[1].dynamic_label(False)
        assert len(sep_label) == 55


# ---------------------------------------------------------------------------
# UIKit.table_menu
# ---------------------------------------------------------------------------

class TestUIKitTableMenu:
    def _kit_with_menu(self, return_item=None):
        kit = _make_kit()
        kit.menu = MagicMock(return_value=return_item)
        return kit

    def test_returns_none_when_menu_returns_none(self):
        kit = self._kit_with_menu(None)
        cols = [TableColumn("A", fixed_width=10)]
        rows = [TableRow(cells=["x"], action=lambda: None)]
        result = kit.table_menu(cols, rows)
        assert result is None

    def test_returns_table_row_when_item_selected(self):
        cols = [TableColumn("A", fixed_width=10)]
        row = TableRow(cells=["x"], action=lambda: None)
        # Crear ítem real con _table_row para simular selección
        items = _build_table_menu_items(cols, [row], style=_Style, get_width_fn=lambda: 40)
        selected_item = items[2]  # el ítem de la fila
        kit = self._kit_with_menu(selected_item)
        result = kit.table_menu(cols, [row])
        assert result is row

    def test_calls_menu_with_table_items(self):
        kit = self._kit_with_menu(None)
        cols = [TableColumn("A", fixed_width=10)]
        rows = [TableRow(cells=["x"], action=lambda: None)]
        kit.table_menu(cols, rows)
        assert kit.menu.called
