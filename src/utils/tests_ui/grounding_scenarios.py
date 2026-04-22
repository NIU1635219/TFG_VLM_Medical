"""Selector de escenarios de grounding (stub) separado para mantener el menú principal limpio.

Este módulo contiene la UI de selección de escenarios A/B/C/D/E/F y está pensado
para recibir un `make_header_fn` desde el menú principal para conservar
consistencia en el renderizado de cabeceras.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

from ..setup_ui_io import ask_text
from .grounding_scenarios_helpers import (
    build_live_confusion_heatmap_lines as _build_live_confusion_heatmap_lines,
    empty_live_confusion_counts as _empty_live_confusion_counts,
    heatmap_cell_style as _heatmap_cell_style,
    heatmap_rgb_from_percentage as _heatmap_rgb_from_percentage,
    resolve_existing_dir as _resolve_existing_dir,
    resolve_existing_file as _resolve_existing_file,
    summarize_existing_scenario_records as _summarize_existing_scenario_records,
)
from .grounding_scenarios_live_runner import run_scenario_with_dashboards as _run_scenario_with_dashboards_impl

if TYPE_CHECKING:
    from ..menu_kit import AppContext, UIKit


_GROUNDING_SEED_CURSOR_KEY = "grounding_scenarios_seed"
_GROUNDING_LAST_OUTPUTS_CURSOR_KEY = "grounding_scenarios_last_outputs"


def _discover_last_scenario_output_from_meta(*, scenario_code: str) -> Path | None:
    """Descubre el último JSONL válido de un escenario usando cabecera __scenario_meta__."""
    scenario_dir = _resolve_existing_dir(
        Path("data/processed/scenario_results") / f"scenario_{scenario_code}"
    )
    if scenario_dir is None:
        return None

    candidates = sorted(
        scenario_dir.glob("run_*/results.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None

    try:
        from src.scripts.grounding_experiments.runner_core import (
            SCENARIO_META_KEY,
            load_jsonl_records,
        )
    except Exception:
        return candidates[0]

    for candidate in candidates:
        try:
            records = load_jsonl_records(candidate, include_system_records=True)
            if any(isinstance(item.get(SCENARIO_META_KEY), dict) for item in records):
                return candidate
        except Exception:
            continue
    return candidates[0]


def _is_scenario_run_incomplete(output_path: Path) -> bool:
    """Detecta si un run está incompleto por falta de summary o líneas sin rellenar."""
    try:
        from src.scripts.grounding_experiments.runner_core import (
            SCENARIO_META_KEY,
            SCENARIO_SUMMARY_KEY,
            has_unfilled_scenario_records,
            load_jsonl_records,
        )

        records = load_jsonl_records(output_path, include_system_records=True)
        has_meta = any(isinstance(item.get(SCENARIO_META_KEY), dict) for item in records)
        has_summary = any(isinstance(item.get(SCENARIO_SUMMARY_KEY), dict) for item in records)
        has_unfilled = has_unfilled_scenario_records(output_path=output_path)
        return bool(has_meta and ((not has_summary) or has_unfilled))
    except Exception:
        return False


def _load_scenario_meta(output_path: Path) -> dict[str, Any]:
    """Carga la cabecera __scenario_meta__ si existe en el JSONL de un run."""
    try:
        from src.scripts.grounding_experiments.runner_core import (
            SCENARIO_META_KEY,
            load_jsonl_records,
        )

        records = load_jsonl_records(output_path, include_system_records=True)
        for entry in records:
            raw_meta = entry.get(SCENARIO_META_KEY)
            if isinstance(raw_meta, dict):
                return {str(k): v for k, v in raw_meta.items()}
    except Exception:
        return {}
    return {}


def _select_grounding_sample_size(kit: "UIKit") -> int | None:
    """Reutiliza el selector numérico de batch para definir número de inferencias."""
    from .manifest import _select_manifest_sample_size_with_default

    selected = _select_manifest_sample_size_with_default(kit, initial_value=None)
    if selected == "BACK":
        return None
    if isinstance(selected, int) and selected > 0:
        return selected
    return None


def _select_grounding_seed(kit: "UIKit", *, initial_value: int = 42) -> int | None:
    """Solicita semilla común para muestreo reproducible en escenarios A/B/C/D/E/F."""

    def _validate_seed(raw_value: str) -> str | None:
        if not raw_value:
            return "Debes indicar una semilla entera (>= 0)."
        if not raw_value.isdigit():
            return "Entrada inválida. Usa solo dígitos."
        return None

    value = ask_text(
        kit=kit,
        title="GROUNDING SCENARIOS · RANDOM SEED",
        intro_lines=[
            (
                f"{kit.style.DIM}Usa la misma seed para que los escenarios A/B/C/D/E/F"
                f" evalúen el mismo subconjunto de imágenes.{kit.style.ENDC}"
            ),
        ],
        prompt_label="Seed:",
        help_line="[ENTER] confirmar · [Backspace] borrar · [ESC] cancelar",
        initial_value=str(max(0, int(initial_value))),
        allow_char_fn=lambda ch: ch.isdigit(),
        normalize_on_submit_fn=lambda text: text.strip(),
        validate_on_submit_fn=_validate_seed,
        force_full_on_update=True,
    )
    if value is None:
        return None
    return int(value)


def _select_grounding_run_mode(
    kit: "UIKit",
    *,
    scenario_code: str,
    last_output_path: Path | None,
) -> tuple[bool, Path | None] | None:
    """Permite elegir explícitamente entre reanudar el último run o iniciar uno nuevo."""
    if last_output_path is None or not last_output_path.exists() or not last_output_path.is_file():
        return False, None

    is_incomplete = _is_scenario_run_incomplete(last_output_path)
    if not is_incomplete:
        # Si el último run está completo, no se ofrece reanudar.
        return False, None

    resume_label = " Reanudar último run (recomendado)" if is_incomplete else " Reanudar último run"

    options = [
        kit.MenuItem(
            resume_label,
            lambda: "RESUME",
            description=f"Continuar desde {last_output_path}.",
        ),
        kit.MenuItem(
            " Nuevo run",
            lambda: "NEW",
            description="Crear una nueva carpeta de ejecución.",
        ),
        kit.MenuItem(
            " Back",
            lambda: "BACK",
            description="Cancelar ejecución del escenario.",
        ),
    ]

    selected = kit.menu(
        options,
        header_func=None,
        multi_select=False,
        menu_id=f"grounding_scenario_{str(scenario_code).lower()}_run_mode_selector",
        nav_hint_text="↑/↓ navegar · ENTER seleccionar · ESC volver",
    )

    if not selected or selected == "BACK":
        return None
    if isinstance(selected, list):
        selected = selected[0] if len(selected) > 0 else None
    if not selected:
        return None
    if hasattr(selected, "action") and callable(selected.action):
        choice = selected.action()
        if choice == "RESUME":
            return True, last_output_path
        if choice == "NEW":
            return False, None
    return None


def _run_scenario_with_dashboards(
    *,
    kit: "UIKit",
    app: "AppContext",
    scenario_code: str,
    run_main: Callable[..., int],
    model_tag: str,
    sample_size: int,
    seed: int,
    resume_mode: bool = False,
    resume_output_path: Path | None = None,
) -> int:
    """Ejecuta un escenario de grounding con dashboard en vivo y pantalla final."""
    return _run_scenario_with_dashboards_impl(
        kit=kit,
        app=app,
        scenario_code=scenario_code,
        run_main=run_main,
        model_tag=model_tag,
        sample_size=sample_size,
        seed=seed,
        outputs_cursor_key=_GROUNDING_LAST_OUTPUTS_CURSOR_KEY,
        resume_mode=resume_mode,
        resume_output_path=resume_output_path,
    )


def _run_scenario_a_with_dashboards(
    *,
    kit: "UIKit",
    app: "AppContext",
    model_tag: str,
    sample_size: int,
    seed: int,
    resume_mode: bool = False,
    resume_output_path: Path | None = None,
) -> int:
    """Ejecuta Scenario A con dashboard en vivo y pantalla final."""
    from src.scripts.grounding_experiments.run_scenario_A import main as run_scenario_a_main

    return _run_scenario_with_dashboards(
        kit=kit,
        app=app,
        scenario_code="A",
        run_main=run_scenario_a_main,
        model_tag=model_tag,
        sample_size=sample_size,
        seed=seed,
        resume_mode=resume_mode,
        resume_output_path=resume_output_path,
    )


def _run_scenario_b_with_dashboards(
    *,
    kit: "UIKit",
    app: "AppContext",
    model_tag: str,
    sample_size: int,
    seed: int,
    resume_mode: bool = False,
    resume_output_path: Path | None = None,
) -> int:
    """Ejecuta Scenario B con dashboard en vivo y pantalla final."""
    from src.scripts.grounding_experiments.run_scenario_B import main as run_scenario_b_main

    return _run_scenario_with_dashboards(
        kit=kit,
        app=app,
        scenario_code="B",
        run_main=run_scenario_b_main,
        model_tag=model_tag,
        sample_size=sample_size,
        seed=seed,
        resume_mode=resume_mode,
        resume_output_path=resume_output_path,
    )


def _run_scenario_c_with_dashboards(
    *,
    kit: "UIKit",
    app: "AppContext",
    model_tag: str,
    sample_size: int,
    seed: int,
    resume_mode: bool = False,
    resume_output_path: Path | None = None,
) -> int:
    """Ejecuta Scenario C con dashboard en vivo y pantalla final."""
    from src.scripts.grounding_experiments.run_scenario_C import main as run_scenario_c_main

    return _run_scenario_with_dashboards(
        kit=kit,
        app=app,
        scenario_code="C",
        run_main=run_scenario_c_main,
        model_tag=model_tag,
        sample_size=sample_size,
        seed=seed,
        resume_mode=resume_mode,
        resume_output_path=resume_output_path,
    )


def _run_scenario_d_with_dashboards(
    *,
    kit: "UIKit",
    app: "AppContext",
    model_tag: str,
    sample_size: int,
    seed: int,
    resume_mode: bool = False,
    resume_output_path: Path | None = None,
) -> int:
    """Ejecuta Scenario D con dashboard en vivo y pantalla final."""
    from src.scripts.grounding_experiments.run_scenario_D import main as run_scenario_d_main

    return _run_scenario_with_dashboards(
        kit=kit,
        app=app,
        scenario_code="D",
        run_main=run_scenario_d_main,
        model_tag=model_tag,
        sample_size=sample_size,
        seed=seed,
        resume_mode=resume_mode,
        resume_output_path=resume_output_path,
    )


def _run_scenario_e_with_dashboards(
    *,
    kit: "UIKit",
    app: "AppContext",
    model_tag: str,
    sample_size: int,
    seed: int,
    resume_mode: bool = False,
    resume_output_path: Path | None = None,
) -> int:
    """Ejecuta Scenario E con dashboard en vivo y pantalla final."""
    from src.scripts.grounding_experiments.run_scenario_E import main as run_scenario_e_main

    return _run_scenario_with_dashboards(
        kit=kit,
        app=app,
        scenario_code="E",
        run_main=run_scenario_e_main,
        model_tag=model_tag,
        sample_size=sample_size,
        seed=seed,
        resume_mode=resume_mode,
        resume_output_path=resume_output_path,
    )


def _run_scenario_f_with_dashboards(
    *,
    kit: "UIKit",
    app: "AppContext",
    model_tag: str,
    sample_size: int,
    seed: int,
    resume_mode: bool = False,
    resume_output_path: Path | None = None,
) -> int:
    """Ejecuta Scenario F con dashboard en vivo y pantalla final."""
    from src.scripts.grounding_experiments.run_scenario_F import main as run_scenario_f_main

    return _run_scenario_with_dashboards(
        kit=kit,
        app=app,
        scenario_code="F",
        run_main=run_scenario_f_main,
        model_tag=model_tag,
        sample_size=sample_size,
        seed=seed,
        resume_mode=resume_mode,
        resume_output_path=resume_output_path,
    )


def run_grounding_scenarios_selector_wrapper(
    kit: "UIKit",
    app: "AppContext",
    *,
    make_header_fn: Callable[[str], Any],
    select_model: Callable[[str, str], str | None],
) -> None:
    """Muestra selector stub de escenarios de grounding A/B/C/D/E/F.

    Este stub mantiene la UX y el texto tal como estaba en el menú original.
    Más adelante se podrá reemplazar la acción de cada opción por el runner
    correspondiente (A/B/C/D/E/F).
    """

    scenario_options = [
        kit.MenuItem(
            " Scenario A (Zero-Shot BBox)",
            lambda: "A",
            description="Escenario A seleccionado (stub).",
        ),
        kit.MenuItem(
            " Scenario B (BBox Asistido)",
            lambda: "B",
            description="Escenario B seleccionado (stub).",
        ),
        kit.MenuItem(
            " Scenario C (BBox Forzado visualmente)",
            lambda: "C",
            description="Escenario C seleccionado (stub).",
        ),
        kit.MenuItem(
            " Scenario D (Aislamiento Visual / Recorte ROI)",
            lambda: "D",
            description="Inferencia sobre recorte exacto del pólipo.",
        ),
        kit.MenuItem(
            " Scenario E (Techo de Calidad: Crop + Clase)",
            lambda: "E",
            description="Inferencia sobre recorte exacto con clase inyectada.",
        ),
        kit.MenuItem(
            " Scenario F (Reporte Clínico Asistido)",
            lambda: "F",
            description="Grounding + informe clínico explicativo con diagnóstico GT inyectado.",
        ),
        kit.MenuItem(
            " Back",
            lambda: "BACK",
            description="Volver al menú de tests.",
        ),
    ]

    selected = kit.menu(
        scenario_options,
        header_func=make_header_fn("GROUNDING SCENARIOS SELECTOR"),
        multi_select=False,
        menu_id="grounding_scenarios_selector",
        nav_hint_text="↑/↓ navegar escenarios · ENTER seleccionar · ESC volver",
    )
    if not selected or selected == "BACK":
        return

    if isinstance(selected, list):
        selected = selected[0] if len(selected) > 0 else None
    if not selected:
        return

    if hasattr(selected, "action") and callable(selected.action):
        scenario_code = str(selected.action())
        if scenario_code in {"A", "B", "C", "D", "E", "F"}:
            raw_outputs = kit.cursor_memory.get(_GROUNDING_LAST_OUTPUTS_CURSOR_KEY)
            known_outputs = cast(dict[str, str], raw_outputs) if isinstance(raw_outputs, dict) else {}
            last_output_path = None
            last_output_raw = known_outputs.get(scenario_code)
            if last_output_raw:
                last_output_path = _resolve_existing_file(str(last_output_raw))
            if last_output_path is None:
                last_output_path = _discover_last_scenario_output_from_meta(scenario_code=scenario_code)

            run_mode = _select_grounding_run_mode(
                kit,
                scenario_code=scenario_code,
                last_output_path=last_output_path,
            )
            if run_mode is None:
                return
            resume_mode, resume_output_path = run_mode

            previous_seed = int(kit.cursor_memory.get(_GROUNDING_SEED_CURSOR_KEY) or 42)
            selected_seed = previous_seed
            selected_sample_size: int | None = None
            model_tag: str | None = None

            if resume_mode and resume_output_path is not None:
                meta = _load_scenario_meta(resume_output_path)
                raw_model = str(meta.get("model_id") or "").strip()
                if raw_model:
                    model_tag = raw_model

                raw_seed = meta.get("seed")
                if isinstance(raw_seed, int) and raw_seed >= 0:
                    selected_seed = raw_seed

                raw_limit = meta.get("requested_limit")
                if isinstance(raw_limit, int) and raw_limit > 0:
                    selected_sample_size = raw_limit

            if model_tag is None:
                model_tag = select_model(
                    f"grounding_scenario_{str(scenario_code).lower()}_model_selector",
                    f"GROUNDING SCENARIO {scenario_code} · SELECT MODEL",
                )
                if model_tag is None:
                    return

            if selected_sample_size is None:
                selected_sample_size = _select_grounding_sample_size(kit)
                if selected_sample_size is None:
                    return

            if not (resume_mode and resume_output_path is not None and selected_seed >= 0):
                selected_seed_input = _select_grounding_seed(kit, initial_value=previous_seed)
                if selected_seed_input is None:
                    return
                selected_seed = selected_seed_input

            kit.cursor_memory[_GROUNDING_SEED_CURSOR_KEY] = selected_seed

            try:
                if scenario_code == "A":
                    exit_code = _run_scenario_a_with_dashboards(
                        kit=kit,
                        app=app,
                        model_tag=model_tag,
                        sample_size=selected_sample_size,
                        seed=selected_seed,
                        resume_mode=resume_mode,
                        resume_output_path=resume_output_path,
                    )
                elif scenario_code == "B":
                    exit_code = _run_scenario_b_with_dashboards(
                        kit=kit,
                        app=app,
                        model_tag=model_tag,
                        sample_size=selected_sample_size,
                        seed=selected_seed,
                        resume_mode=resume_mode,
                        resume_output_path=resume_output_path,
                    )
                elif scenario_code == "C":
                    exit_code = _run_scenario_c_with_dashboards(
                        kit=kit,
                        app=app,
                        model_tag=model_tag,
                        sample_size=selected_sample_size,
                        seed=selected_seed,
                        resume_mode=resume_mode,
                        resume_output_path=resume_output_path,
                    )
                elif scenario_code == "D":
                    exit_code = _run_scenario_d_with_dashboards(
                        kit=kit,
                        app=app,
                        model_tag=model_tag,
                        sample_size=selected_sample_size,
                        seed=selected_seed,
                        resume_mode=resume_mode,
                        resume_output_path=resume_output_path,
                    )
                elif scenario_code == "E":
                    exit_code = _run_scenario_e_with_dashboards(
                        kit=kit,
                        app=app,
                        model_tag=model_tag,
                        sample_size=selected_sample_size,
                        seed=selected_seed,
                        resume_mode=resume_mode,
                        resume_output_path=resume_output_path,
                    )
                elif scenario_code == "F":
                    exit_code = _run_scenario_f_with_dashboards(
                        kit=kit,
                        app=app,
                        model_tag=model_tag,
                        sample_size=selected_sample_size,
                        seed=selected_seed,
                        resume_mode=resume_mode,
                        resume_output_path=resume_output_path,
                    )
                else:
                    # Guardia defensiva ante códigos no previstos.
                    kit.log(f"Scenario no soportado: {scenario_code}", "warning")
                    return
                if exit_code == 0:
                    kit.log(f"Scenario {scenario_code} completed successfully.", "success")
                else:
                    kit.log(f"Scenario {scenario_code} finished with warnings/errors.", "warning")
            except Exception as error:
                kit.log(f"Scenario {scenario_code} failed: {error}", "error")
            return

