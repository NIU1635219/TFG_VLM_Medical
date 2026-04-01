"""Escenario C: atención forzada dibujando el bbox GT antes de inferencia.

Este script ejecuta un batch sobre el CSV de ground truth y, para cada imagen,
pinta el recuadro real en rojo antes de enviarla al VLM. Los resultados se
guardan de forma incremental en JSONL para análisis posterior.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

from src.inference.vlm_runner import VLMLoader

from .runner_core import (
    DEFAULT_GROUND_TRUTH_CSV,
    PolypDiagnosisAndGrounding,
    build_annotated_comparison_filename,
    build_class_lookup_from_m_split_csvs,
    build_markdown_records_from_scenario_jsonl,
    build_scenario_record,
    collect_processed_image_ids_from_jsonl,
    compute_iou_safe,
    default_scenario_output_path,
    draw_bbox_and_save_temp_image,
    emit_report_event,
    generate_single_detection_markdown_report,
    initialize_scenario_result_skeleton,
    load_ground_truth,
    normalize_bbox_for_metrics,
    normalize_image_stem,
    normalize_polyp_class,
    prepare_run_artifact_dirs,
    resolve_image_path_from_id,
    resolve_ground_truth_class_from_lookup,
    resolve_output_jsonl_path,
    safe_inference_with_optional_telemetry,
    select_ground_truth_rows,
    summarize_scenario_records_from_jsonl,
    upsert_scenario_execution_summary,
    upsert_scenario_meta_header,
    upsert_scenario_result_record,
    write_comparison_image,
)

SCENARIO_C_PROMPT = (
    "El recuadro rojo en la imagen señala la ubicación exacta de una lesión endoscópica. "
    "Centra tu atención exclusivamente en esa zona. Devuelve las coordenadas de este "
    "recuadro [0-1000] para confirmar que lo ves, describe detalladamente la morfología, "
    "bordes y patrón vascular del tejido dentro del recuadro, y finalmente emite un "
    "diagnóstico histológico a ciegas (AD, HP o ASS)."
)

DEFAULT_IMG_DIR = Path("data/processed/m_train/images")
Reporter = Callable[[str, dict[str, Any]], None]


def build_parser() -> argparse.ArgumentParser:
    """Construye el parser CLI para Escenario C."""
    parser = argparse.ArgumentParser(
        description="Run Scenario C (forced-attention grounding + diagnosis) over ground truth CSV."
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Nombre/ruta del modelo en LM Studio (ej. qwen3.5-9b@q8_0).",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=str(DEFAULT_GROUND_TRUTH_CSV),
        help="Ruta al CSV de ground truth.",
    )
    parser.add_argument(
        "--img-dir",
        type=str,
        default=str(DEFAULT_IMG_DIR),
        help="Directorio base de imágenes.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Archivo JSONL de salida. Si se omite, se crea en "
            "data/processed/scenario_results/scenario_C/run_*/results.jsonl"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Límite opcional de imágenes a procesar.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Semilla para muestreo reproducible de imágenes cuando se usa --limit. "
            "Usa la misma seed en escenarios A/B/C/D para evaluar el mismo subconjunto."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Desactiva tqdm para integración limpia en TUI.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reanuda ejecución reutilizando el JSONL de salida y saltando image_id ya procesados.",
    )
    return parser


def run(args: argparse.Namespace, reporter: Reporter | None = None) -> int:
    """Ejecuta el batch del Escenario C y guarda resultados incrementales en JSONL."""
    started_at = time.perf_counter()
    input_csv = str(args.input_csv)
    output_path = resolve_output_jsonl_path(
        raw_output=args.output,
        default_output=default_scenario_output_path(scenario_name="scenario_C"),
    )
    run_dir, annotated_dir = prepare_run_artifact_dirs(output_jsonl_path=output_path)
    forced_images_tmp_dir = run_dir / "_tmp_forced_images"
    img_dir = Path(str(args.img_dir))

    if not img_dir.exists() or not img_dir.is_dir():
        raise FileNotFoundError(f"Directorio de imágenes no encontrado: {img_dir}")

    dataframe = load_ground_truth(input_csv)
    class_lookup = build_class_lookup_from_m_split_csvs(img_dir=img_dir)
    if not class_lookup:
        raise RuntimeError(
            "No se encontraron etiquetas de clase en CSVs de split m_*; "
            "no se permite fallback."
        )
    seed_value = int(args.seed) if args.seed is not None else None
    if seed_value is not None and seed_value < 0:
        raise ValueError("--seed debe ser mayor o igual que 0")

    if args.limit is not None and int(args.limit) <= 0:
        raise ValueError("--limit debe ser mayor que 0 cuando se especifica")

    dataframe = select_ground_truth_rows(
        dataframe,
        limit=int(args.limit) if args.limit is not None else None,
        seed=seed_value,
    )

    upsert_scenario_meta_header(
        output_path=output_path,
        scenario_name="scenario_C",
        model_id=str(args.model),
        input_csv=input_csv,
        img_dir=str(img_dir),
        seed=seed_value,
        requested_limit=int(args.limit) if args.limit is not None else None,
        resume_mode=bool(args.resume),
    )

    rows_raw = dataframe.to_dict(orient="records")
    rows: list[dict[str, Any]] = [
        {str(key): value for key, value in row.items()}
        for row in rows_raw
    ]
    resumed_existing = 0
    if bool(args.resume):
        processed_ids = collect_processed_image_ids_from_jsonl(output_jsonl_path=output_path)
        if processed_ids:
            filtered_rows: list[dict[str, Any]] = []
            for row in rows:
                try:
                    row_image_id = normalize_image_stem(row.get("image_id"))
                except Exception:
                    filtered_rows.append(row)
                    continue
                if row_image_id in processed_ids:
                    resumed_existing += 1
                    continue
                filtered_rows.append(row)
            rows = filtered_rows

    total = len(rows)
    if total == 0:
        print("[INFO] No hay filas para procesar en el CSV.")
        emit_report_event(
            reporter,
            "run_complete",
            model=str(args.model),
            seed=seed_value,
            total=0,
            ok=0,
            fail=0,
            skip=0,
            resumed_existing=resumed_existing,
            output_path=str(output_path),
            avg_ttft_seconds=None,
            avg_tokens_per_second=None,
            avg_total_duration_seconds=None,
            elapsed_seconds=0.0,
        )
        return 0

    if not bool(args.resume):
        skeleton_records: list[dict[str, Any]] = []
        for row in rows:
            image_id_value = row.get("image_id")
            gt_bbox = [
                row.get("ymin"),
                row.get("xmin"),
                row.get("ymax"),
                row.get("xmax"),
            ]
            try:
                gt_cls_value = normalize_polyp_class(
                    resolve_ground_truth_class_from_lookup(
                        row=row,
                        class_lookup=class_lookup,
                    )
                )
            except Exception:
                gt_cls_value = None

            skeleton_records.append(
                build_scenario_record(
                    scenario_name="scenario_C",
                    model_id=str(args.model),
                    schema_name="PolypDiagnosisAndGrounding",
                    image_id=image_id_value,
                    image_path=img_dir / str(image_id_value or "unknown"),
                    status="pending",
                    duration_seconds=0.0,
                    ground_truth_bbox=gt_bbox,
                    ground_truth_cls=gt_cls_value,
                    payload=None,
                    telemetry_payload=None,
                    class_match=None,
                    iou_score=None,
                )
            )

        initialize_scenario_result_skeleton(
            output_path=output_path,
            skeleton_records=skeleton_records,
        )

    loader = VLMLoader(model_path=str(args.model))
    processed_ok = 0
    processed_fail = 0
    processed_skip = 0
    matched_class = 0
    mismatched_class = 0
    iou_values: list[float] = []
    ttft_values: list[float] = []
    tps_values: list[float] = []
    total_duration_values: list[float] = []

    try:
        loader.preload_model()
        emit_report_event(
            reporter,
            "run_start",
            model=str(args.model),
            total=total,
            seed=seed_value,
            resumed_existing=resumed_existing,
            output_path=str(output_path),
        )

        row_iterable = tqdm(
            rows,
            total=total,
            desc="Scenario C",
            unit="img",
            disable=bool(args.no_progress),
        )

        for index, row in enumerate(row_iterable, start=1):
            image_started_at = time.perf_counter()
            image_id_value = row.get("image_id")
            gt_bbox = [
                row.get("ymin"),
                row.get("xmin"),
                row.get("ymax"),
                row.get("xmax"),
            ]
            emit_report_event(
                reporter,
                "image_start",
                index=index,
                total=total,
                image_id=image_id_value,
            )
            try:
                image_path = resolve_image_path_from_id(image_id=row.get("image_id"), img_dir=img_dir)
            except Exception as error:
                processed_fail += 1
                processed_skip += 1
                print(f"[WARN] Saltando imagen por ruta inválida: {error}")
                upsert_scenario_result_record(
                    output_path=output_path,
                    result_dict=build_scenario_record(
                        scenario_name="scenario_C",
                        model_id=str(args.model),
                        schema_name="PolypDiagnosisAndGrounding",
                        image_id=image_id_value,
                        image_path=img_dir / str(image_id_value or "unknown"),
                        status="skip",
                        duration_seconds=max(0.0, time.perf_counter() - image_started_at),
                        ground_truth_bbox=gt_bbox,
                        ground_truth_cls=None,
                        payload=None,
                        telemetry_payload=None,
                        class_match=None,
                        iou_score=None,
                        error=error,
                    ),
                )
                emit_report_event(
                    reporter,
                    "image_skip",
                    index=index,
                    total=total,
                    image_id=image_id_value,
                    reason="image_path",
                    error=str(error),
                )
                continue

            gt_bbox_norm = normalize_bbox_for_metrics(gt_bbox)
            if gt_bbox_norm is None:
                processed_fail += 1
                processed_skip += 1
                error = ValueError("Bounding box GT inválido en CSV")
                upsert_scenario_result_record(
                    output_path=output_path,
                    result_dict=build_scenario_record(
                        scenario_name="scenario_C",
                        model_id=str(args.model),
                        schema_name="PolypDiagnosisAndGrounding",
                        image_id=image_id_value,
                        image_path=image_path,
                        status="skip",
                        duration_seconds=max(0.0, time.perf_counter() - image_started_at),
                        ground_truth_bbox=gt_bbox,
                        ground_truth_cls=None,
                        payload=None,
                        telemetry_payload=None,
                        class_match=None,
                        iou_score=None,
                        error=error,
                    ),
                )
                emit_report_event(
                    reporter,
                    "image_skip",
                    index=index,
                    total=total,
                    image_id=image_id_value,
                    image_path=str(image_path),
                    reason="invalid_gt_bbox",
                    error=str(error),
                )
                continue

            try:
                gt_cls = resolve_ground_truth_class_from_lookup(
                    row=row,
                    class_lookup=class_lookup,
                )
                gt_cls = normalize_polyp_class(gt_cls)
            except Exception as error:
                processed_fail += 1
                processed_skip += 1
                print(
                    "[WARN] Ground truth class no encontrada para "
                    f"image_id={row.get('image_id')}: {error}"
                )
                upsert_scenario_result_record(
                    output_path=output_path,
                    result_dict=build_scenario_record(
                        scenario_name="scenario_C",
                        model_id=str(args.model),
                        schema_name="PolypDiagnosisAndGrounding",
                        image_id=image_id_value,
                        image_path=image_path,
                        status="skip",
                        duration_seconds=max(0.0, time.perf_counter() - image_started_at),
                        ground_truth_bbox=gt_bbox,
                        ground_truth_cls=None,
                        payload=None,
                        telemetry_payload=None,
                        class_match=None,
                        iou_score=None,
                        error=error,
                    ),
                )
                emit_report_event(
                    reporter,
                    "image_skip",
                    index=index,
                    total=total,
                    image_id=image_id_value,
                    image_path=str(image_path),
                    reason="ground_truth_cls",
                    error=str(error),
                )
                continue

            forced_image_path: Path | None = None
            try:
                forced_image_path = draw_bbox_and_save_temp_image(
                    image_path=image_path,
                    bbox_norm=gt_bbox_norm,
                    temp_dir=forced_images_tmp_dir,
                )
            except Exception as error:
                processed_fail += 1
                print(
                    "[WARN] Falló la preparación OpenCV para "
                    f"image_id={row.get('image_id')} ({image_path.name}): {error}"
                )
                upsert_scenario_result_record(
                    output_path=output_path,
                    result_dict=build_scenario_record(
                        scenario_name="scenario_C",
                        model_id=str(args.model),
                        schema_name="PolypDiagnosisAndGrounding",
                        image_id=image_id_value,
                        image_path=image_path,
                        status="error",
                        duration_seconds=max(0.0, time.perf_counter() - image_started_at),
                        ground_truth_bbox=gt_bbox,
                        ground_truth_cls=gt_cls,
                        payload=None,
                        telemetry_payload=None,
                        class_match=None,
                        iou_score=None,
                        error=error,
                    ),
                )
                emit_report_event(
                    reporter,
                    "image_error",
                    index=index,
                    total=total,
                    image_id=image_id_value,
                    image_path=str(image_path),
                    error=str(error),
                )
                continue

            try:
                parsed_response, telemetry_payload = safe_inference_with_optional_telemetry(
                    loader=loader,
                    image_path=str(forced_image_path),
                    prompt=SCENARIO_C_PROMPT,
                    schema=PolypDiagnosisAndGrounding,
                )
            except Exception as error:
                processed_fail += 1
                print(
                    "[WARN] Inference falló para "
                    f"image_id={row.get('image_id')} ({image_path.name}): {error}"
                )
                upsert_scenario_result_record(
                    output_path=output_path,
                    result_dict=build_scenario_record(
                        scenario_name="scenario_C",
                        model_id=str(args.model),
                        schema_name="PolypDiagnosisAndGrounding",
                        image_id=image_id_value,
                        image_path=image_path,
                        status="error",
                        duration_seconds=max(0.0, time.perf_counter() - image_started_at),
                        ground_truth_bbox=gt_bbox,
                        ground_truth_cls=gt_cls,
                        payload=None,
                        telemetry_payload=None,
                        class_match=None,
                        iou_score=None,
                        error=error,
                    ),
                )
                emit_report_event(
                    reporter,
                    "image_error",
                    index=index,
                    total=total,
                    image_id=image_id_value,
                    image_path=str(image_path),
                    error=str(error),
                )
                continue
            finally:
                if forced_image_path is not None:
                    try:
                        forced_image_path.unlink(missing_ok=True)
                    except Exception:
                        pass

            parsed_payload = dict(parsed_response.model_dump())
            predicted_cls = normalize_polyp_class(
                parsed_payload.get("final_diagnosis_class")
                or ""
            )
            parsed_payload["final_diagnosis_class"] = predicted_cls
            pred_bbox = [
                parsed_payload.get("ymin"),
                parsed_payload.get("xmin"),
                parsed_payload.get("ymax"),
                parsed_payload.get("xmax"),
            ]
            iou_score = compute_iou_safe(gt_bbox=gt_bbox, pred_bbox=pred_bbox)
            if isinstance(iou_score, float):
                iou_values.append(iou_score)

            is_match = bool(predicted_cls and gt_cls and predicted_cls == gt_cls)
            if predicted_cls and gt_cls:
                if is_match:
                    matched_class += 1
                else:
                    mismatched_class += 1

            if telemetry_payload is not None:
                ttft_value = telemetry_payload.get("ttft_seconds")
                tps_value = telemetry_payload.get("tokens_per_second")
                total_duration_value = telemetry_payload.get("total_duration_seconds")
                if isinstance(ttft_value, (int, float)):
                    ttft_values.append(float(ttft_value))
                if isinstance(tps_value, (int, float)):
                    tps_values.append(float(tps_value))
                if isinstance(total_duration_value, (int, float)):
                    total_duration_values.append(float(total_duration_value))

            result_dict = build_scenario_record(
                scenario_name="scenario_C",
                model_id=str(args.model),
                schema_name="PolypDiagnosisAndGrounding",
                image_id=image_id_value,
                image_path=image_path,
                status="ok",
                duration_seconds=max(0.0, time.perf_counter() - image_started_at),
                ground_truth_bbox=gt_bbox,
                ground_truth_cls=gt_cls,
                payload=parsed_payload,
                telemetry_payload=telemetry_payload,
                class_match=is_match,
                iou_score=iou_score,
            )
            upsert_scenario_result_record(output_path=output_path, result_dict=result_dict)

            annotated_candidate = annotated_dir / build_annotated_comparison_filename(
                image_id=image_id_value,
                image_path=image_path,
            )
            _ = write_comparison_image(
                image_path=image_path,
                output_path=annotated_candidate,
                gt_bbox=gt_bbox,
                pred_bbox=pred_bbox,
                model_name=str(args.model),
                gt_label=str(gt_cls),
                pred_label=predicted_cls or "N/D",
                iou_score=iou_score,
            )

            processed_ok += 1
            emit_report_event(
                reporter,
                "image_ok",
                index=index,
                total=total,
                image_id=image_id_value,
                image_path=str(image_path),
                ground_truth_cls=gt_cls,
                predicted_cls=predicted_cls,
                class_match=is_match,
                iou_score=iou_score,
                telemetry=telemetry_payload,
            )

    finally:
        loader.unload_model()

    avg_ttft = (sum(ttft_values) / len(ttft_values)) if ttft_values else None
    avg_tps = (sum(tps_values) / len(tps_values)) if tps_values else None
    avg_total_duration = (sum(total_duration_values) / len(total_duration_values)) if total_duration_values else None
    avg_iou = (sum(iou_values) / len(iou_values)) if iou_values else None
    elapsed_seconds = max(0.0, time.perf_counter() - started_at)
    cumulative_summary = summarize_scenario_records_from_jsonl(output_path=output_path)
    report_records_full = build_markdown_records_from_scenario_jsonl(
        output_path=output_path,
        run_dir=run_dir,
    )

    markdown_path = generate_single_detection_markdown_report(
        run_dir=run_dir,
        report_title="Scenario C Report",
        scenario_name="scenario_C",
        model_id=str(args.model),
        jsonl_path=output_path,
        records=report_records_full,
    )

    emit_report_event(
        reporter,
        "run_complete",
        model=str(args.model),
        seed=seed_value,
        resumed_existing=resumed_existing,
        total=int(cumulative_summary.get("total") or 0),
        ok=int(cumulative_summary.get("ok") or 0),
        fail=int(cumulative_summary.get("fail") or 0),
        skip=int(cumulative_summary.get("skip") or 0),
        matched_class=int(cumulative_summary.get("matched_class") or 0),
        mismatched_class=int(cumulative_summary.get("mismatched_class") or 0),
        avg_iou=cumulative_summary.get("avg_iou"),
        output_path=str(output_path),
        markdown_path=str(markdown_path),
        run_dir=str(run_dir),
        avg_ttft_seconds=cumulative_summary.get("avg_ttft_seconds"),
        avg_tokens_per_second=cumulative_summary.get("avg_tokens_per_second"),
        avg_total_duration_seconds=cumulative_summary.get("avg_total_duration_seconds"),
        elapsed_seconds=elapsed_seconds,
    )

    upsert_scenario_execution_summary(
        output_path=output_path,
        summary_payload={
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scenario_name": "scenario_C",
            "model_id": str(args.model),
            "seed": seed_value,
            "requested_limit": int(args.limit) if args.limit is not None else None,
            "resumed_existing": resumed_existing,
            "total": int(cumulative_summary.get("total") or 0),
            "ok": int(cumulative_summary.get("ok") or 0),
            "fail": int(cumulative_summary.get("fail") or 0),
            "skip": int(cumulative_summary.get("skip") or 0),
            "matched_class": int(cumulative_summary.get("matched_class") or 0),
            "mismatched_class": int(cumulative_summary.get("mismatched_class") or 0),
            "avg_iou": cumulative_summary.get("avg_iou"),
            "avg_ttft_seconds": cumulative_summary.get("avg_ttft_seconds"),
            "avg_tokens_per_second": cumulative_summary.get("avg_tokens_per_second"),
            "avg_total_duration_seconds": cumulative_summary.get("avg_total_duration_seconds"),
            "elapsed_seconds": elapsed_seconds,
            "output_path": str(output_path),
            "markdown_path": str(markdown_path),
            "run_dir": str(run_dir),
            "run_total": total,
            "run_ok": processed_ok,
            "run_fail": processed_fail,
            "run_skip": processed_skip,
            "run_matched_class": matched_class,
            "run_mismatched_class": mismatched_class,
            "run_avg_iou": avg_iou,
            "run_avg_ttft_seconds": avg_ttft,
            "run_avg_tokens_per_second": avg_tps,
            "run_avg_total_duration_seconds": avg_total_duration,
        },
    )

    print(
        "[INFO] Scenario C finalizado. "
        f"ok={processed_ok} fail={processed_fail} output='{output_path}' md='{markdown_path}'"
    )
    return 0 if processed_ok > 0 else 1


def main(argv: list[str] | None = None, reporter: Reporter | None = None) -> int:
    """Punto de entrada CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args, reporter=reporter)


if __name__ == "__main__":
    sys.exit(main())
