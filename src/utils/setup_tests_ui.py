"""UI de tests y selección de modelos para setup_env."""

from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .menu_kit import AppContext, UIKit


# ---------------------------------------------------------------------------
# Utilidades de sistema puras
# ---------------------------------------------------------------------------


def list_test_files() -> list[str]:
    """Escanea la carpeta ``tests/`` y devuelve los archivos de prueba.

    Returns:
        list[str]: Lista ordenada de nombres de archivo que empiezan por
        ``test_`` y terminan en ``.py``.  Lista vacía si la carpeta no existe.
    """
    test_dir = "tests"
    if not os.path.exists(test_dir):
        return []
    files = [
        f for f in os.listdir(test_dir)
        if f.startswith("test_") and f.endswith(".py")
    ]
    return sorted(files)


# ---------------------------------------------------------------------------
# Menú de tests
# ---------------------------------------------------------------------------

def run_tests_menu(kit: "UIKit", app: "AppContext") -> None:
    """
    Despliega el submenú para ejecución de tests y gestión de pruebas.

    Ofrece opciones para:
    - Gestionar modelos (descarga/borrado).
    - Ejecutar todos los tests unitarios.
    - Ejecutar tests unitarios específicos.
    - Ejecutar 'Smoke Test' (prueba de inferencia end-to-end con modelo).
    - Ejecutar 'Schema Tester' (inferencia + validación por esquema Pydantic).

    Args:
        kit (UIKit): Interfaz de UI de terminal.
        app (AppContext): Contexto de dominio de la aplicación.
    """
    from . import setup_models_ui

    # ------------------------------------------------------------------
    # Helpers reutilizables
    # ------------------------------------------------------------------

    def _make_header(subtitle: str):
        """Devuelve una función de encabezado reutilizable."""
        def _hdr() -> None:
            app.print_banner()
            kit.subtitle(subtitle)
        return _hdr

    def _select_thinking_mode(
        model_tag: str,
        menu_id: str,
        subtitle: str,
    ) -> "bool | None | Literal['BACK']":
        """Muestra selector de modo thinking si el modelo lo soporta.

        Returns:
            True  → thinking ON
            False → thinking OFF
            None  → Auto (no se especifica al modelo)
            "BACK" → el usuario quiere volver al paso anterior
        """
        from src.inference.vlm_runner import is_thinking_capable
        if not is_thinking_capable(model_tag):
            return None

        items = [
            kit.MenuItem(
                "Thinking ON  (razonamiento extendido)",
                description="El modelo razona internamente antes de responder. Más lento pero más preciso.",
            ),
            kit.MenuItem(
                "Thinking OFF  (respuesta directa)",
                description="El modelo responde sin cadena de razonamiento. Más rápido.",
            ),
            kit.MenuItem(
                "Auto  (configuración por defecto del modelo)",
                description="No se especifica ningún parámetro; el modelo usa su comportamiento nativo.",
            ),
            kit.MenuItem(
                "← Volver al selector de modelo",
                lambda: None,
                description="Cancela y vuelve a elegir modelo.",
            ),
        ]
        sel = kit.menu(
            items,
            header_func=_make_header(subtitle),
            menu_id=menu_id,
            nav_hint_text="↑/↓ elegir modo · ENTER confirmar · ESC volver al selector de modelo",
        )
        if not sel or "Volver" in sel.label:
            return "BACK"
        if "ON" in sel.label.upper() and "THINKING" in sel.label.upper() and "OFF" not in sel.label.upper():
            return True
        if "OFF" in sel.label.upper():
            return False
        return None  # Auto

    def _select_model(menu_id: str, subtitle: str) -> str | None:
        """
        Muestra un selector de modelo LM Studio.

        Returns:
            Etiqueta del modelo seleccionado, o ``None`` si se cancela.
        """
        installed = app.get_installed_lms_models()
        if not installed:
            kit.log(
                "No hay modelos disponibles en LM Studio. "
                "Carga uno desde 'Manage/Pull LM Studio Models...'.",
                "warning",
            )
            kit.wait("Press any key to return to tests menu...")
            return None

        opts = [
            kit.MenuItem(tag, description="Usa este modelo para la inferencia.")
            for tag in installed
        ]
        opts.append(kit.MenuItem("Cancel", lambda: None, description="Vuelve al menú anterior."))

        sel = kit.menu(
            opts,
            header_func=_make_header(subtitle),
            menu_id=menu_id,
            nav_hint_text="↑/↓ elegir modelo · ENTER confirmar · ESC volver",
        )
        if not sel or sel.label.strip() == "Cancel":
            return None
        tag = sel.label.strip()
        if not tag:
            kit.log("No model selected.", "warning")
            kit.wait("Press any key to return to tests menu...")
            return None
        return tag

    # ------------------------------------------------------------------
    # Acciones de menú
    # ------------------------------------------------------------------

    def run_smoke_test_in_process(model_tag: str, *, enable_thinking: bool | None = None) -> int:
        """Ejecuta smoke test sin subprocess."""
        try:
            from src.scripts.test_inference import main as smoke_test_main
        except Exception as error:
            kit.log(f"Could not import smoke test script: {error}", "error")
            return 1
        try:
            return int(smoke_test_main(model_path=model_tag, enable_thinking=enable_thinking))
        except Exception as error:
            kit.log(f"Smoke test crashed: {error}", "error")
            return 1

    def run_all_unit_tests() -> None:
        """Ejecuta todos los tests unitarios."""
        kit.log("Running All Unit Tests...", "step")
        kit.run_cmd("uv run python -m pytest tests/")
        kit.wait()

    def run_specific_test() -> None:
        """Ejecuta un test específico."""
        while True:
            files = app.list_test_files()
            if not files:
                kit.log("No tests found in tests/ folder.", "warning")
                app.time_module.sleep(1)
                return

            test_opts = [
                kit.MenuItem(f, description="Ejecuta solo este archivo de tests con pytest.")
                for f in files
            ]
            test_opts.append(
                kit.MenuItem("Cancel", lambda: None, description="Vuelve al menú anterior sin ejecutar pruebas.")
            )

            selection = kit.menu(
                test_opts,
                header_func=_make_header("SELECT TEST FILE"),
                menu_id="run_specific_test_selector",
                nav_hint_text="↑/↓ navegar archivos · ENTER ejecutar test · ESC volver",
            )

            if not selection or selection.label.strip() == "Cancel":
                return

            fname = selection.label
            kit.clear()
            kit.log(f"Running {fname}...", "step")
            kit.run_cmd(f"uv run python -m pytest tests/{fname}")
            kit.wait("Finished. Press any key to return to test selector...")

    def run_schema_tester_wrapper() -> None:
        """Schema Tester: modelo → thinking → esquema → inferencia sobre 5 imágenes."""
        from src.inference.schemas import SCHEMA_REGISTRY
        from src.scripts.test_schema import (
            find_images,
            format_schema_info,
            format_schema_menu_description,
            run_batch,
        )

        while True:
            # ─ PASO 1: Seleccionar modelo ─────────────────────
            model_tag = _select_model(
                menu_id="schema_tester_model_selector",
                subtitle="SCHEMA TESTER · SELECT MODEL",
            )
            if model_tag is None:
                return

            # ─ PASO 2: Seleccionar modo thinking (si aplica) ────
            thinking_mode = _select_thinking_mode(
                model_tag,
                menu_id="schema_tester_thinking_selector",
                subtitle="SCHEMA TESTER · SELECT THINKING MODE",
            )
            if thinking_mode == "BACK":
                continue

            # ─ PASO 3: Seleccionar esquema ────────────────────
            schema_items = [
                kit.MenuItem(name, description=format_schema_menu_description(name, cls))
                for name, cls in SCHEMA_REGISTRY.items()
            ]
            schema_items.append(
                kit.MenuItem("Cancel", lambda: None, description="Vuelve a la selección de modelo.")
            )

            schema_sel = kit.menu(
                schema_items,
                header_func=_make_header("SCHEMA TESTER · SELECT SCHEMA"),
                menu_id="schema_tester_schema_selector",
                nav_hint_text="↑/↓ elegir esquema · ENTER confirmar · ESC volver",
                description_slot_rows=15,
            )
            if not schema_sel or schema_sel.label.strip() == "Cancel":
                continue
            schema_name = schema_sel.label
            schema_cls = SCHEMA_REGISTRY[schema_name]

            # ─ PASO 4: Inferencia automática ──────────────────────────
            images = find_images()
            if not images:
                kit.log("No se encontraron imágenes en los directorios del proyecto.", "warning")
                kit.wait("Press any key to return to model selector...")
                continue

            kit.clear()
            app.print_banner()
            kit.subtitle(f"SCHEMA TESTER · {schema_name}")
            print()
            _tw = max(60, shutil.get_terminal_size(fallback=(120, 30)).columns - 4)
            for line in format_schema_info(schema_name, schema_cls, text_width=_tw).splitlines():
                print(f"  {line}")
            print()
            kit.log(
                f"Modelo: {model_tag} · max 5 imágenes de {len(images)} disponibles...",
                "step",
            )
            try:
                ok, fail, invalid = run_batch(model_tag, schema_name, schema_cls, images, enable_thinking=thinking_mode)
                if fail > 0 or invalid > 0:
                    kit.log(
                        f"Schema Tester completado: {ok} válidas, "
                        f"{invalid} inválidas, {fail} errores.",
                        "warning",
                    )
                else:
                    kit.log(
                        f"Schema Tester completado: {ok}/{ok} inferencias válidas.",
                        "success",
                    )
            except Exception as error:
                kit.log(f"Schema Tester terminó con error: {error}", "error")
            kit.wait("Press any key to return to model selector...")

    def run_smoke_test_wrapper() -> None:
        """Lanza smoke test con selector de modelo (y thinking si procede)."""
        while True:
            model_tag = _select_model(
                menu_id="run_smoke_model_selector",
                subtitle="SELECT INFERENCE MODEL",
            )
            if model_tag is None:
                return

            thinking_mode = _select_thinking_mode(
                model_tag,
                menu_id="run_smoke_thinking_selector",
                subtitle="SELECT THINKING MODE",
            )
            if thinking_mode == "BACK":
                continue

            kit.clear()
            kit.log(f"Launching Inference with {model_tag}...", "step")
            result_code = run_smoke_test_in_process(model_tag, enable_thinking=thinking_mode)
            if result_code != 0:
                kit.log("Smoke test failed (Exit Code 1). Check output above.", "error")
            kit.wait("Press any key to return to model selector...")

    # ------------------------------------------------------------------
    # Menú principal de tests
    # ------------------------------------------------------------------

    options = [
        kit.MenuItem(
            " Manage/Pull LM Studio Models...",
            lambda: setup_models_ui.manage_models_menu_ui(kit, app),
            description="Descarga, carga o descarga de memoria modelos mediante LM Studio CLI.",
        ),
        kit.MenuItem(
            " Run All Unit Tests (pytest)",
            run_all_unit_tests,
            description="Ejecuta todos los tests dentro de la carpeta tests/.",
        ),
        kit.MenuItem(
            " Run Specific Test File...",
            run_specific_test,
            description="Abre un selector para ejecutar un único archivo de test.",
        ),
        kit.MenuItem(
            " Run Smoke Test (Inference Demo)",
            run_smoke_test_wrapper,
            description="Lanza una inferencia rápida para validar flujo modelo+imagen.",
        ),
        kit.MenuItem(
            " Run Schema Tester (VLM Interactive)",
            run_schema_tester_wrapper,
            description=(
                "Selecciona modelo y esquema Pydantic, ejecuta inferencia sobre 5 imágenes "
                "aleatorias y valida automáticamente que las respuestas cumplan el esquema."
            ),
        ),
        kit.MenuItem(
            " Return to Main Menu",
            lambda: "BACK",
            description="Vuelve al menú principal.",
        ),
    ]

    while True:
        choice = kit.menu(
            options,
            header_func=_make_header("TEST & MODEL MANAGER"),
            multi_select=False,
            menu_id="tests_manager_menu",
            nav_hint_text="↑/↓ navegar opciones · ENTER abrir/ejecutar · ESC volver al menú principal",
        )
        if choice == "BACK" or not choice:
            break

        if isinstance(choice, list):
            choice = choice[0] if len(choice) > 0 else None

        if choice and hasattr(choice, "action") and callable(choice.action):
            kit.clear()
            res = choice.action()
            if res == "BACK":
                break
