"""UI de tests y selección de modelos para setup_env."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .menu_kit import AppContext, UIKit


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
    from .tests_ui.batch import run_batch_runner_wrapper as run_batch_runner_screen
    from .tests_ui.manifest import (
        execution_schema_name as execution_schema_name_screen,
        linked_batch_output_path as linked_batch_output_path_screen,
        manifest_execution_snapshot as manifest_execution_snapshot_screen,
        prune_output_records_for_model as prune_output_records_for_model_screen,
        select_manifest_for_batch as select_manifest_for_batch_screen,
    )
    from .tests_ui.response_inspector import (
        run_response_inspector_wrapper as run_response_inspector_screen,
    )
    from .tests_ui.run_pytest import (
        run_all_unit_tests as run_all_unit_tests_screen,
        run_specific_test as run_specific_test_screen,
    )
    from .tests_ui.shared import (
        make_header,
        select_model,
        select_response_inspector_mode,
        select_schema_variant,
    )
    from .tests_ui.schema import run_schema_tester_wrapper as run_schema_tester_screen
    from .tests_ui.smoke import run_smoke_test_wrapper as run_smoke_test_screen
    from .tests_ui.telemetry import run_telemetry_probe_wrapper as run_telemetry_probe_screen

    make_header_fn = lambda subtitle: make_header(kit, app, subtitle)
    select_model_fn = lambda menu_id, subtitle: select_model(
        kit,
        app,
        menu_id=menu_id,
        subtitle=subtitle,
        make_header_fn=make_header_fn,
    )
    select_schema_variant_fn = lambda menu_prefix, subtitle_prefix: select_schema_variant(
        kit,
        menu_prefix=menu_prefix,
        subtitle_prefix=subtitle_prefix,
        make_header_fn=make_header_fn,
    )
    select_response_inspector_mode_fn = lambda: select_response_inspector_mode(
        kit,
        make_header_fn=make_header_fn,
    )

    def _select_manifest_for_batch() -> dict[str, Any] | None:
        """Selecciona un manifiesto autosuficiente o genera uno nuevo."""
        return select_manifest_for_batch_screen(
            kit,
            app,
            make_header=make_header_fn,
            select_schema_variant=select_schema_variant_fn,
        )

    # ------------------------------------------------------------------
    # Acciones de menú
    # ------------------------------------------------------------------

    def run_all_unit_tests() -> None:
        """Ejecuta todos los tests unitarios."""
        run_all_unit_tests_screen(kit)

    def run_specific_test() -> None:
        """Ejecuta un test específico."""
        run_specific_test_screen(kit, app, make_header=make_header_fn)

    def run_schema_tester_wrapper() -> None:
        """Schema Tester: modelo → schema base → modo reasoning → inferencia."""
        run_schema_tester_screen(
            kit,
            app,
            select_model=select_model_fn,
            select_schema_variant=select_schema_variant_fn,
        )

    def run_telemetry_probe_wrapper() -> None:
        """Ejecuta una prueba de telemetría TTFT/TPS con selección interactiva."""
        run_telemetry_probe_screen(
            kit,
            app,
            select_model=select_model_fn,
            select_schema_variant=select_schema_variant_fn,
        )

    def run_batch_runner_wrapper() -> None:
        """Ejecuta el batch runner usando manifiestos autosuficientes."""
        run_batch_runner_screen(
            kit,
            app,
            select_manifest_for_batch=_select_manifest_for_batch,
            execution_schema_name=execution_schema_name_screen,
            linked_batch_output_path=linked_batch_output_path_screen,
            manifest_execution_snapshot=manifest_execution_snapshot_screen,
            prune_output_records_for_model=prune_output_records_for_model_screen,
        )

    def run_response_inspector_wrapper() -> None:
        """Ejecuta el inspector de respuesta del SDK con defaults automáticos."""
        run_response_inspector_screen(
            kit,
            app,
            select_model=select_model_fn,
            select_response_inspector_mode=select_response_inspector_mode_fn,
        )

    def run_smoke_test_wrapper() -> None:
        """Lanza smoke test con selector de modelo."""
        run_smoke_test_screen(
            kit,
            app,
            select_model=select_model_fn,
        )

    # ------------------------------------------------------------------
    # Menú principal de tests
    # ------------------------------------------------------------------

    options =[
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
            " Run Telemetry Probe (TTFT/TPS)",
            run_telemetry_probe_wrapper,
            description=(
                "Mide métricas del SDK response.stats y recursos de /v1/models "
                "sobre una muestra de imágenes del proyecto."
            ),
        ),
        kit.MenuItem(
            " Run Batch Runner",
            run_batch_runner_wrapper,
            description=(
                "Ejecuta inferencia masiva con guardado incremental para análisis posterior "
                "sin romper la TUI."
            ),
        ),
        kit.MenuItem(
            " Run Response Inspector (SDK Fields)",
            run_response_inspector_wrapper,
            description=(
                "Lanza una petición real y resume solo los campos útiles que devuelve el SDK, "
                "con autodetección de imagen y prompt."
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
            header_func=make_header_fn("TEST & MODEL MANAGER"),
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