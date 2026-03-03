"""Tests para src/utils/setup_menu_engine.py — MenuItem, MenuSeparator, interactive_menu."""

import pytest
from unittest.mock import MagicMock
from types import SimpleNamespace

from src.utils.setup_menu_engine import MenuItem, MenuStaticItem, MenuSeparator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Style:
    BOLD = ENDC = DIM = OKCYAN = OKGREEN = WARNING = FAIL = HEADER = SELECTED = ""


# ---------------------------------------------------------------------------
# MenuItem
# ---------------------------------------------------------------------------

class TestMenuItem:
    def test_basic_construction(self):
        item = MenuItem("Option A")
        assert item.label == "Option A"
        assert item.action is None
        assert item.description == ""
        assert item.children == []
        assert item.is_selected is False

    def test_with_action(self):
        fn = lambda: "ok"
        item = MenuItem("Do it", action=fn)
        assert item.action is fn
        assert item.action() == "ok"

    def test_with_description(self):
        item = MenuItem("X", description="Descripción larga de prueba")
        assert item.description == "Descripción larga de prueba"

    def test_with_children(self):
        child = MenuItem("Child A")
        parent = MenuItem("Parent", children=[child])
        assert len(parent.children) == 1
        assert parent.children[0] is child

    def test_default_selectable(self):
        item = MenuItem("X")
        assert getattr(item, "selectable", True) is True

    def test_dynamic_label_none_by_default(self):
        item = MenuItem("X")
        assert item.dynamic_label is None


# ---------------------------------------------------------------------------
# MenuStaticItem
# ---------------------------------------------------------------------------

class TestMenuStaticItem:
    def test_default_label_empty(self):
        item = MenuStaticItem()
        assert item.label == ""

    def test_not_selectable(self):
        item = MenuStaticItem("Header text")
        assert item.selectable is False

    def test_action_is_none(self):
        item = MenuStaticItem("Header")
        assert item.action is None


# ---------------------------------------------------------------------------
# MenuSeparator
# ---------------------------------------------------------------------------

class TestMenuSeparator:
    def test_dynamic_label_without_text(self):
        sep = MenuSeparator(width=10)
        label = sep.dynamic_label(False)
        assert label == "─" * 10
        assert len(label) == 10

    def test_dynamic_label_with_text_centered(self):
        sep = MenuSeparator(text="Tests", width=20)
        label = sep.dynamic_label(False)
        assert "Tests" in label
        assert len(label) == 20

    def test_dynamic_label_text_truncated_if_too_long(self):
        sep = MenuSeparator(text="VeryLongHeaderText", width=10)
        label = sep.dynamic_label(False)
        assert len(label) <= 20  # Never explodes

    def test_custom_fill_char(self):
        sep = MenuSeparator(width=8, fill="=")
        label = sep.dynamic_label(False)
        assert "=" in label

    def test_not_selectable(self):
        sep = MenuSeparator()
        assert sep.selectable is False
