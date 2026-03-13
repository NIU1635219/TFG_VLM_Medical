from __future__ import annotations

import os
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from ..menu_kit import AppContext, UIKit


def list_test_files() -> list[str]:
    """List available pytest files under the tests folder.

    Returns:
        Sorted filenames matching `test_*.py`, or an empty list.
    """
    test_dir = "tests"
    if not os.path.exists(test_dir):
        return []
    files = [
        f for f in os.listdir(test_dir)
        if f.startswith("test_") and f.endswith(".py")
    ]
    return sorted(files)


def _coerce_test_files(value: object) -> list[str]:
    """Normalize dynamic menu sources into a stable list of filenames.

    Args:
        value: Raw value returned by app.list_test_files.

    Returns:
        Cleaned list of test filenames.
    """
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item]


def run_all_unit_tests(kit: "UIKit") -> None:
    """Run the full project unit-test suite.

    Args:
        kit: Terminal UI toolkit used to render logs and execute commands.
    """
    kit.log("Running All Unit Tests...", "step")
    kit.run_cmd("uv run python -m pytest tests/")
    kit.wait()


def run_specific_test(
    kit: "UIKit",
    app: "AppContext",
    *,
    make_header: Callable[[str], Callable[[], None]],
) -> None:
    """Run a single pytest file selected interactively.

    Args:
        kit: Terminal UI toolkit.
        app: Application context that provides test file discovery.
        make_header: Header factory used by the interactive selector.
    """
    while True:
        app_list_test_files = getattr(app, "list_test_files", None)
        files_obj: object = app_list_test_files() if callable(app_list_test_files) else list_test_files()
        files = _coerce_test_files(files_obj)
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
            header_func=make_header("SELECT TEST FILE"),
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
