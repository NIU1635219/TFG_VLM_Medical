"""UI de tests y selección de modelos para setup_env."""

from __future__ import annotations

import shutil
from typing import Any


def run_tests_menu(*, ctx: dict[str, Any]) -> None:
    """
    Despliega el submenú para ejecución de tests y gestión de pruebas.

    Ofrece opciones para:
    - Gestionar modelos (descarga/borrado).
    - Ejecutar todos los tests unitarios.
    - Ejecutar tests unitarios específicos.
    - Ejecutar 'Smoke Test' (prueba de inferencia end-to-end con modelo).
    - Ejecutar 'Schema Tester' (inferencia + validación por esquema Pydantic).

    Args:
        ctx (dict): Contexto de aplicación.
    """
    Style = ctx["Style"]
    MenuItem = ctx["MenuItem"]
    print_banner = ctx["print_banner"]
    log = ctx["log"]
    run_cmd = ctx["run_cmd"]
    wait_for_any_key = ctx["wait_for_any_key"]
    interactive_menu = ctx["interactive_menu"]
    list_test_files = ctx["list_test_files"]
    get_installed_lms_models = ctx["get_installed_lms_models"]
    clear_screen_ansi = ctx["clear_screen_ansi"]
    time_module = ctx["time"]

    # ------------------------------------------------------------------
    # Helpers reutilizables (reducen duplicación entre smoke / schema /
    # specific-test / etc.)
    # ------------------------------------------------------------------

    def _make_header(subtitle: str):
        """Devuelve una función de encabezado reutilizable."""
        def _hdr() -> None:
            print_banner()
            print(f"{Style.BOLD} {subtitle} {Style.ENDC}")
        return _hdr

    def _select_model(menu_id: str, subtitle: str) -> str | None:
        """
        Muestra un selector de modelo LM Studio reutilizable.

        Returns:
            Etiqueta del modelo seleccionado, o None si se cancela / no hay modelos.
        """
        installed = get_installed_lms_models()
        if not installed:
            log(
                "No hay modelos disponibles en LM Studio. "
                "Carga uno desde 'Manage/Pull LM Studio Models...'.",
                "warning",
            )
            wait_for_any_key("Press any key to return to tests menu...")
            return None

        opts = [
            MenuItem(tag, description="Usa este modelo para la inferencia.")
            for tag in installed
        ]
        opts.append(MenuItem("Cancel", lambda: None, description="Vuelve al menú anterior."))

        sel = interactive_menu(
            opts,
            header_func=_make_header(subtitle),
            menu_id=menu_id,
            nav_hint_text="↑/↓ elegir modelo · ENTER confirmar · ESC volver",
        )
        if not sel or sel.label.strip() == "Cancel":
            return None
        tag = sel.label.strip()
        if not tag:
            log("No model selected.", "warning")
            wait_for_any_key("Press any key to return to tests menu...")
            return None
        return tag

    # ------------------------------------------------------------------
    # Acciones de menú
    # ------------------------------------------------------------------

    def run_smoke_test_in_process(model_tag: str) -> int:
        """
        Ejecuta smoke test sin subprocess para mantener navegación de menú estable.

        Args:
            model_tag (str): Etiqueta del modelo.

        Returns:
            int: Código de salida del smoke test.
        """
        try:
            from src.scripts.test_inference import main as smoke_test_main
        except Exception as error:
            log(f"Could not import smoke test script: {error}", "error")
            return 1
        try:
            return int(smoke_test_main(model_path=model_tag, interactive=False))
        except Exception as error:
            log(f"Smoke test crashed: {error}", "error")
            return 1

    def run_all_unit_tests() -> None:
        """Ejecuta todos los tests unitarios."""
        log("Running All Unit Tests...", "step")
        run_cmd("uv run python -m pytest tests/")
        wait_for_any_key()

    def run_specific_test() -> None:
        """Ejecuta un test específico."""
        while True:
            files = list_test_files()
            if not files:
                log("No tests found in tests/ folder.", "warning")
                time_module.sleep(1)
                return

            test_opts = [
                MenuItem(f, description="Ejecuta solo este archivo de tests con pytest.")
                for f in files
            ]
            test_opts.append(
                MenuItem("Cancel", lambda: None, description="Vuelve al menú anterior sin ejecutar pruebas.")
            )

            selection = interactive_menu(
                test_opts,
                header_func=_make_header("SELECT TEST FILE"),
                menu_id="run_specific_test_selector",
                nav_hint_text="↑/↓ navegar archivos · ENTER ejecutar test · ESC volver",
            )

            if not selection or selection.label.strip() == "Cancel":
                return

            fname = selection.label
            clear_screen_ansi()
            log(f"Running {fname}...", "step")
            run_cmd(f"uv run python -m pytest tests/{fname}")
            wait_for_any_key("Finished. Press any key to return to test selector...")

    def run_schema_tester_wrapper() -> None:
        """
        Schema Tester: seleccionar modelo → esquema → ejecutar inferencia
        sobre 5 imágenes aleatorias y validar contra el esquema.
        """
        from src.inference.schemas import SCHEMA_REGISTRY
        from src.scripts.interactive_schema_tester import (
            find_images,
            format_schema_info,
            format_schema_menu_description,
            run_batch,
        )

        while True:
            # ─ PASO 1: Seleccionar modelo ────────────────────────────
            model_tag = _select_model(
                menu_id="schema_tester_model_selector",
                subtitle="SCHEMA TESTER · SELECT MODEL",
            )
            if model_tag is None:
                return

            # ─ PASO 2: Seleccionar esquema ───────────────────────────
            schema_items = [
                MenuItem(name, description=format_schema_menu_description(name, cls))
                for name, cls in SCHEMA_REGISTRY.items()
            ]
            schema_items.append(
                MenuItem("Cancel", lambda: None, description="Vuelve a la selección de modelo.")
            )

            schema_sel = interactive_menu(
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

            # ─ PASO 3: Inferencia automática ─────────────────────────
            images = find_images()
            if not images:
                log("No se encontraron imágenes en los directorios del proyecto.", "warning")
                wait_for_any_key("Press any key to return to model selector...")
                continue

            clear_screen_ansi()
            print_banner()
            print(f"{Style.BOLD} SCHEMA TESTER · {schema_name} {Style.ENDC}\n")
            _tw = max(60, shutil.get_terminal_size(fallback=(120, 30)).columns - 4)
            for line in format_schema_info(schema_name, schema_cls, text_width=_tw).splitlines():
                print(f"  {line}")
            print()
            log(
                f"Modelo: {model_tag} · max 5 imágenes de {len(images)} disponibles...",
                "step",
            )
            try:
                ok, fail, invalid = run_batch(model_tag, schema_name, schema_cls, images)
                if fail > 0 or invalid > 0:
                    log(
                        f"Schema Tester completado: {ok} válidas, "
                        f"{invalid} inválidas, {fail} errores.",
                        "warning",
                    )
                else:
                    log(
                        f"Schema Tester completado: {ok}/{ok} inferencias válidas.",
                        "success",
                    )
            except Exception as error:
                log(f"Schema Tester terminó con error: {error}", "error")
            wait_for_any_key("Press any key to return to model selector...")

    def run_smoke_test_wrapper() -> None:
        """Lanza smoke test y mantiene retorno al selector de modelo."""
        while True:
            model_tag = _select_model(
                menu_id="run_smoke_model_selector",
                subtitle="SELECT INFERENCE MODEL",
            )
            if model_tag is None:
                return

            clear_screen_ansi()
            log(f"Launching Inference with {model_tag}...", "step")
            result_code = run_smoke_test_in_process(model_tag)
            if result_code != 0:
                log("Smoke test failed (Exit Code 1). Check output above.", "error")
            wait_for_any_key("Press any key to return to model selector...")

    # ------------------------------------------------------------------
    # Menú principal de tests
    # ------------------------------------------------------------------

    options = [
        MenuItem(
            " Manage/Pull LM Studio Models...",
            ctx["manage_models_menu_ui"],
            description="Descarga, carga o descarga de memoria modelos mediante LM Studio CLI.",
        ),
        MenuItem(
            " Run All Unit Tests (pytest)",
            run_all_unit_tests,
            description="Ejecuta todos los tests dentro de la carpeta tests/.",
        ),
        MenuItem(
            " Run Specific Test File...",
            run_specific_test,
            description="Abre un selector para ejecutar un único archivo de test.",
        ),
        MenuItem(
            " Run Smoke Test (Inference Demo)",
            run_smoke_test_wrapper,
            description="Lanza una inferencia rápida para validar flujo modelo+imagen.",
        ),
        MenuItem(
            " Run Schema Tester (VLM Interactive)",
            run_schema_tester_wrapper,
            description=(
                "Selecciona modelo y esquema Pydantic, ejecuta inferencia sobre 5 imágenes "
                "aleatorias y valida automáticamente que las respuestas cumplan el esquema."
            ),
        ),
        MenuItem(
            " Return to Main Menu",
            lambda: "BACK",
            description="Vuelve al menú principal.",
        ),
    ]

    while True:
        choice = interactive_menu(
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
            clear_screen_ansi()
            res = choice.action()
            if res == "BACK":
                break
