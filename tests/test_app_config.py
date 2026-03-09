"""Tests para src/utils/app_config.py — fuente única de constantes de configuración."""

import pytest
from src.utils.app_config import MODELS_REGISTRY, REQUIRED_LIBS, LIB_IMPORT_MAP


class TestModelsRegistry:
    def test_is_dict(self):
        assert isinstance(MODELS_REGISTRY, dict)

    def test_contains_expected_models(self):
        expected = {"minicpm_v_4_5_8b", "qwen3_5_9b", "internvl3_5_8b"}
        assert expected.issubset(set(MODELS_REGISTRY.keys()))

    def test_each_entry_has_name_and_description(self):
        for key, entry in MODELS_REGISTRY.items():
            assert "name" in entry, f"Falta 'name' en {key}"
            assert "description" in entry, f"Falta 'description' en {key}"
            assert entry["name"], f"'name' vacío en {key}"
            assert entry["description"], f"'description' vacío en {key}"

    def test_entries_are_huggingface_urls(self):
        for key, entry in MODELS_REGISTRY.items():
            assert entry["name"].startswith("https://"), f"URL inesperada en {key}: {entry['name']}"


class TestRequiredLibs:
    def test_is_list(self):
        assert isinstance(REQUIRED_LIBS, list)

    def test_not_empty(self):
        assert len(REQUIRED_LIBS) > 0

    def test_contains_core_dependencies(self):
        assert "lmstudio" in REQUIRED_LIBS
        assert "pydantic" in REQUIRED_LIBS
        assert "pillow" in REQUIRED_LIBS
        assert "pytest" in REQUIRED_LIBS

    def test_all_entries_are_strings(self):
        for lib in REQUIRED_LIBS:
            assert isinstance(lib, str), f"Entrada no-string: {lib!r}"

    def test_no_duplicates(self):
        assert len(REQUIRED_LIBS) == len(set(REQUIRED_LIBS))


class TestLibImportMap:
    def test_is_dict(self):
        assert isinstance(LIB_IMPORT_MAP, dict)

    def test_pillow_maps_to_pil(self):
        assert LIB_IMPORT_MAP.get("pillow") == "PIL"

    def test_opencv_maps_to_cv2(self):
        assert LIB_IMPORT_MAP.get("opencv-python") == "cv2"

    def test_all_values_are_nonempty_strings(self):
        for pkg, mod in LIB_IMPORT_MAP.items():
            assert isinstance(mod, str) and mod, f"Módulo vacío para '{pkg}'"
