from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

from .test_dashboards_ui import (
    _build_named_value_lines,
    _render_final_sections_screen,
    _standard_final_intro,
)

if TYPE_CHECKING:
    from ..menu_kit import AppContext, UIKit


_MAX_PREVIEW_IMAGES_PER_VARIANT = 8


def run_ab_experiment_wrapper(
    kit: "UIKit",
    app: "AppContext",
) -> None:
    """Ejecuta el experimento A/B de prompting desde la TUI de tests."""
    from src.scripts.experiment_ab_text import (
        ABVariantPlan,
        ABResumeSummaryRow,
        build_resume_summary,
        build_execution_plans,
        build_parser,
        default_checkpoint_path,
        detect_results_file,
        main as run_ab_experiment_main,
    )

    def _build_preview_text(
        results_file: Path,
        plans: Sequence[ABVariantPlan],
        resume_rows: Sequence[ABResumeSummaryRow],
        checkpoint_file: Path,
        *,
        ui_width: int | None = None,
    ) -> str:
        """Construye resumen de pre-ejecucion para ask_user."""
        def _table_lines(headers: list[str], rows: list[list[str]], *, width: int) -> list[str]:
            """Renderiza una tabla usando el motor gráfico de menu_kit."""
            if not rows:
                return []

            # Heurística de distribución de columnas:
            # - Calcula la longitud máxima por columna (cabecera + contenido)
            # - Usa un mapa de pesos para cabeceras conocidas; en caso contrario
            #   usa la longitud máxima como proxy de necesidad de espacio.
            col_count = len(headers)
            max_lens: list[int] = [len(h) for h in headers]
            for row in rows:
                for i in range(min(col_count, len(row))):
                    l = len(str(row[i] or ""))
                    if l > max_lens[i]:
                        max_lens[i] = l

            weight_map: dict[str, int] = {
                "Imagen": 3,
                "Modelo": 3,
                "Variante": 2,
                "Schema": 2,
                "Estado": 1,
                "OK/TOTAL": 1,
                "Prediccion": 2,
                "GT": 1,
                "Valor": 3,
                "Campo": 2,
            }

            # Construye pesos combinando el mapa semántico y la longitud observada.
            weights: list[float] = []
            for i, h in enumerate(headers):
                semantic = weight_map.get(h, 0)
                # el componente de longitud toma más relevancia cuando es mucho mayor
                length_score = float(max_lens[i])
                base = float(semantic) if semantic > 0 else max(1.0, length_score / 8.0)
                # suavizamos para evitar extremos
                weights.append(max(1.0, base))

            total = sum(weights) or 1.0

            # Define un mínimo razonable para cada columna
            def _min_width_for(header: str, measured: int) -> int:
                if header in ("Imagen", "Modelo"):
                    return max(12, min(32, measured + 4))
                if header in ("Prediccion", "GT", "Estado"):
                    return max(8, min(14, measured + 2))
                return max(10, min(36, measured + 4))

            columns = [
                kit.TableColumn(label=header, width_ratio=(w / total), min_width=_min_width_for(header, max_lens[i]))
                for i, (header, w) in enumerate(zip(headers, weights))
            ]

            table_rows = [kit.TableRow(cells=[str(cell) for cell in row]) for row in rows]

            rendered = kit.table_menu(
                columns,
                table_rows,
                interactive=False,
                return_lines=True,
                width=width,
            )
            if isinstance(rendered, list):
                return [str(r) for r in rendered]
            return []

        width = kit.width() if ui_width is None else int(ui_width)

        lines: list[str] = []
        lines.append("Resumen previo del experimento A/B")
        lines.append("")
        lines.extend(
            _table_lines(
                ["Campo", "Valor"],
                [
                    ["JSONL origen", str(results_file)],
                    ["Checkpoint incremental", str(checkpoint_file)],
                    ["Variantes detectadas", str(len(plans))],
                    ["Max preview por variante", str(_MAX_PREVIEW_IMAGES_PER_VARIANT)],
                ],
                width=width,
            )
        )
        lines.append("")

        if resume_rows:
            lines.append("Estado de reanudacion")
            lines.extend(
                _table_lines(
                    ["Modelo", "Schema", "OK/TOTAL", "Pendientes", "Invalidos"],
                    [
                        [
                            row.variant.model_id,
                            row.variant.schema_name or "N/A",
                            f"{row.reusable_samples}/{row.total_samples}",
                            str(row.pending_samples),
                            str(row.invalid_samples),
                        ]
                        for row in resume_rows
                    ],
                    width=width,
                )
            )
            lines.append("")

        for idx, plan in enumerate(plans, start=1):
            variant = plan.variant
            lines.append(f"Variante {idx}")
            lines.extend(
                _table_lines(
                    ["Modelo", "Reasoning", "Schema", "Imagenes"],
                    [
                        [
                            variant.model_id,
                            str(variant.include_reasoning),
                            variant.schema_name or "N/A",
                            str(len(plan.samples)),
                        ]
                    ],
                    width=width,
                )
            )
            visible_samples = plan.samples[:_MAX_PREVIEW_IMAGES_PER_VARIANT]

            sample_rows = [
                [
                    sample.image_name,
                    sample.previous_status,
                    sample.predicted_cls,
                    sample.ground_truth_cls,
                ]
                for sample in visible_samples
            ]
            lines.extend(
                _table_lines(
                    ["Imagen", "Estado previo", "Prediccion", "GT"],
                    sample_rows,
                    width=width,
                )
            )
            lines.append("")

        return "\n".join(lines).strip()

    while True:
        final_screen_renderer: Callable[[int], None] | None = None

        parser = build_parser()
        defaults = Namespace(
            results_dir=parser.get_default("results_dir"),
            output_md=parser.get_default("output_md"),
            seed=parser.get_default("seed"),
            temperature=parser.get_default("temperature"),
            n_correct=parser.get_default("n_correct"),
            n_incorrect=parser.get_default("n_incorrect"),
        )

        kit.clear()
        app.print_banner()
        kit.subtitle("AB PROMPTING EXPERIMENT")
        kit.log(
            "Ejecutando comparativa zero-shot vs asistida con contexto autodetectado del JSONL...",
            "step",
        )

        try:
            results_file = detect_results_file(None, Path(str(defaults.results_dir)))
            plans = build_execution_plans(
                results_file=results_file,
                n_correct=int(defaults.n_correct),
                n_incorrect=int(defaults.n_incorrect),
                seed=int(defaults.seed),
                override_model=None,
            )
            checkpoint_file = default_checkpoint_path(Path(str(defaults.output_md)))
            resume_rows = build_resume_summary(plans=plans, checkpoint_file=checkpoint_file)

            total_samples = sum(int(row.total_samples) for row in resume_rows)
            reusable_samples = sum(int(row.reusable_samples) for row in resume_rows)
            pending_samples = sum(int(row.pending_samples) for row in resume_rows)
            invalid_samples = sum(int(row.invalid_samples) for row in resume_rows)

            has_previous_progress = reusable_samples > 0 or invalid_samples > 0
            is_checkpoint_complete = (
                total_samples > 0
                and reusable_samples == total_samples
                and pending_samples == 0
                and invalid_samples == 0
            )

            execution_mode = "resume"
            if is_checkpoint_complete:
                should_rerun = kit.ask(
                    "Se detecto una ejecucion completa. Quieres reejecutar todo desde cero?",
                    default="n",
                    info_text=lambda current_width=None: _build_preview_text(
                        results_file,
                        plans,
                        resume_rows,
                        checkpoint_file,
                        ui_width=(int(current_width) if current_width is not None else None),
                    ),
                )
                if not should_rerun:
                    kit.log("Ejecucion cancelada por el usuario.", "warning")
                    return
                execution_mode = "force"
            else:
                if has_previous_progress:
                    should_resume = kit.ask(
                        "Se detecto progreso previo del experimento A/B. Quieres reanudar?",
                        default="y",
                        info_text=lambda current_width=None: _build_preview_text(
                            results_file,
                            plans,
                            resume_rows,
                            checkpoint_file,
                            ui_width=(int(current_width) if current_width is not None else None),
                        ),
                    )
                    execution_mode = "resume" if should_resume else "force"

                should_run = kit.ask(
                    "Confirmar ejecucion del experimento A/B con las muestras detectadas?",
                    default="n",
                    info_text=lambda current_width=None: _build_preview_text(
                        results_file,
                        plans,
                        resume_rows,
                        checkpoint_file,
                        ui_width=(int(current_width) if current_width is not None else None),
                    ),
                )
                if not should_run:
                    kit.log("Ejecucion cancelada por el usuario.", "warning")
                    return

            exit_code = int(
                run_ab_experiment_main(
                    (
                        [
                            "--results-file",
                            str(results_file),
                        ]
                        + (["--force-recompute"] if execution_mode == "force" else [])
                    )
                )
            )
            output_path = Path(str(defaults.output_md))

            def _redraw_final_ab_screen(_ui_width: int) -> None:
                """Re-renderiza la pantalla final del experimento A/B tras resize."""
                sections = [
                    (
                        "Resumen",
                        _build_named_value_lines(
                            kit,
                            [
                                ("Modelo", "Autodetectado por variante desde JSONL"),
                                ("Resultados previos (auto)", str(defaults.results_dir)),
                                ("Variantes detectadas", len(plans)),
                                ("Aciertos AD objetivo", defaults.n_correct),
                                ("Fallos AD objetivo", defaults.n_incorrect),
                                ("Semilla", defaults.seed),
                                ("Temperatura", defaults.temperature),
                                ("Salida Markdown", str(output_path)),
                            ],
                            ui_width=_ui_width,
                        ),
                    ),
                    (
                        "Estado",
                        [
                            "  Experimento completado correctamente."
                            if exit_code == 0
                            else "  Experimento completado con incidencias. Revisa el log y el Markdown generado.",
                        ],
                    ),
                ]
                _render_final_sections_screen(
                    kit,
                    app,
                    subtitle="AB PROMPTING EXPERIMENT",
                    intro=_standard_final_intro(),
                    ui_width=_ui_width,
                    sections=sections,
                )

            _redraw_final_ab_screen(kit.width())
            final_screen_renderer = _redraw_final_ab_screen

            if exit_code == 0:
                kit.log(f"A/B experiment completado. Salida: {output_path}", "success")
            else:
                kit.log(
                    "A/B experiment terminó con incidencias. Revisa la salida mostrada arriba.",
                    "warning",
                )
        except Exception as error:
            kit.log(f"A/B experiment terminó con error: {error}", "error")
            kit.log(
                "Verifica que exista un JSONL batch con AD suficientes (aciertos y fallos) para construir el plan.",
                "warning",
            )
            kit.wait("Press any key to return to model selector...")
            return

        if final_screen_renderer is None:
            kit.wait("Press any key to return to model selector...")
        else:
            kit.render_and_wait_responsive(
                render_fn=final_screen_renderer,
                message="Press any key to return to model selector...",
                initial_render=False,
            )
