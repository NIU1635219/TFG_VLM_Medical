"""UI de tests y selección de modelos para setup_env."""

from __future__ import annotations

from typing import Any


def run_tests_menu(*, ctx: dict[str, Any]) -> None:
    """Menú para ejecutar tests (unitarios y smoke tests)."""
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

    def run_smoke_test_in_process(model_tag: str) -> int:
        """Ejecuta smoke test sin subprocess para mantener navegación de menú estable."""
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
        log("Running All Unit Tests...", "step")
        run_cmd("uv run python -m pytest tests/")
        wait_for_any_key()

    def run_specific_test() -> None:
        while True:
            files = list_test_files()
            if not files:
                log("No tests found in tests/ folder.", "warning")
                time_module.sleep(1)
                return

            test_opts = [MenuItem(f, description="Ejecuta solo este archivo de tests con pytest.") for f in files]
            test_opts.append(MenuItem("Cancel", lambda: None, description="Vuelve al menú anterior sin ejecutar pruebas."))

            def t_header() -> None:
                print_banner()
                print(f"{Style.BOLD} SELECT TEST FILE {Style.ENDC}")

            selection = interactive_menu(
                test_opts,
                header_func=t_header,
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

    def run_smoke_test_wrapper() -> None:
        """Lanza smoke test y mantiene retorno al selector de modelo (un nivel atrás)."""
        while True:
            model_opts = []
            installed_models = get_installed_lms_models()

            if not installed_models:
                log("No hay modelos disponibles en LM Studio. Carga uno desde 'Manage/Pull LM Studio Models...'.", "warning")
                wait_for_any_key("Press any key to return to tests menu...")
                return

            for model_tag in installed_models:
                model_opts.append(MenuItem(model_tag, description="Ejecuta el smoke test usando este modelo instalado."))

            model_opts.append(MenuItem("Cancel", lambda: None, description="Vuelve al menú anterior sin ejecutar smoke test."))

            def m_header() -> None:
                print_banner()
                print(f"{Style.BOLD} SELECT INFERENCE MODEL {Style.ENDC}")

            selection = interactive_menu(
                model_opts,
                header_func=m_header,
                menu_id="run_smoke_model_selector",
                nav_hint_text="↑/↓ elegir modelo · ENTER ejecutar smoke · ESC volver",
            )

            if not selection or selection.label.strip() == "Cancel":
                return

            model_tag = selection.label

            if not model_tag:
                log("No model selected.", "warning")
                wait_for_any_key("Press any key to return to model selector...")
                continue

            clear_screen_ansi()
            log(f"Launching Inference with {model_tag}...", "step")
            result_code = run_smoke_test_in_process(model_tag)
            if result_code != 0:
                log("Smoke test failed (Exit Code 1). Check output above.", "error")
            wait_for_any_key("Press any key to return to model selector...")

    def header() -> None:
        print_banner()
        print(f"{Style.BOLD} TEST & MODEL MANAGER {Style.ENDC}")

    options = [
        MenuItem(" Manage/Pull LM Studio Models...", ctx["manage_models_menu_ui"], description="Descarga, carga o descarga de memoria modelos mediante LM Studio CLI."),
        MenuItem(" Run All Unit Tests (pytest)", run_all_unit_tests, description="Ejecuta todos los tests dentro de la carpeta tests/."),
        MenuItem(" Run Specific Test File...", run_specific_test, description="Abre un selector para ejecutar un único archivo de test."),
        MenuItem(" Run Smoke Test (Inference Demo)", run_smoke_test_wrapper, description="Lanza una inferencia rápida para validar flujo modelo+imagen."),
        MenuItem(" Return to Main Menu", lambda: "BACK", description="Vuelve al menú principal."),
    ]

    while True:
        choice = interactive_menu(
            options,
            header_func=header,
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
