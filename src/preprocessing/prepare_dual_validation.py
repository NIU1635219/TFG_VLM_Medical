from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import random
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2

from src.preprocessing.export_clinical_eval_data import (
    build_output_image_name,
    detect_project_root,
    draw_gt_bbox_only,
    extract_clinical_justification,
    load_image_bgr,
    normalize_class_label,
    read_jsonl_records,
    resolve_run_dir,
    select_top_k_normalized_by_class,
    to_bbox_int4,
    to_float_or_none,
)
from src.scripts.grounding_experiments.report_common import extract_predicted_class


@dataclass(frozen=True)
class DualValidationConfig:
    scenario_a_name: str = "scenario_A"
    scenario_f_name: str = "scenario_F"
    qwen_model_id: str = "qwen3_5-9b@q8_0"
    gemma_model_id: str = "google/gemma-4-e4b@q8_0"

    run_id_a: str | None = None
    run_id_f: str | None = None
    run_path_override_a: str | None = None
    run_path_override_f: str | None = None

    qwen_run_id_a: str | None = None
    qwen_run_id_f: str | None = None
    qwen_run_path_override_a: str | None = None
    qwen_run_path_override_f: str | None = None

    gemma_run_id_a: str | None = None
    gemma_run_id_f: str | None = None
    gemma_run_path_override_a: str | None = None
    gemma_run_path_override_f: str | None = None

    top_k: int = 50
    normalize_by_class: bool = True
    normalization_mode: str = "proportional"
    valid_classes: tuple[str, ...] = ("AD", "HP", "ASS")

    output_dist_dir: Path = Path("dist")
    source_app_dir: Path = Path("clinical_eval_app")
    app_a_name: str = "App_Autonoma"
    app_f_name: str = "App_Asistida"

    gt_color_bgr: tuple[int, int, int] = (0, 200, 0)
    gt_thickness: int = 3
    scale_max: int = 1000


@dataclass(frozen=True)
class DualValidationPaths:
    project_root: Path
    run_dir_qwen_a: Path
    run_dir_qwen_f: Path
    run_dir_gemma_a: Path
    run_dir_gemma_f: Path
    results_jsonl_qwen_a: Path
    results_jsonl_qwen_f: Path
    results_jsonl_gemma_a: Path
    results_jsonl_gemma_f: Path
    output_dist_dir: Path
    output_app_a_dir: Path
    output_app_f_dir: Path
    output_app_a_assets_dir: Path
    output_app_f_assets_dir: Path
    output_app_a_images_dir: Path
    output_app_f_images_dir: Path


@dataclass(frozen=True)
class DualValidationSelectionResult:
    candidates: list[dict[str, Any]]
    selected: list[dict[str, Any]]
    dist_candidates: dict[str, int]
    dist_selected: dict[str, int]
    skipped_by_reason: dict[str, int]


@dataclass(frozen=True)
class DualValidationExportResult:
    selected_image_ids: list[str]
    output_app_a_cases_json: Path
    output_app_f_cases_json: Path
    output_app_a_export_csv: Path
    output_app_f_export_csv: Path
    output_app_a_images_dir: Path
    output_app_f_images_dir: Path


def build_execution_id(*, suffix: str = "F", when: datetime | None = None) -> str:
    timestamp = (when or datetime.now()).strftime("%Y%m%d_%H%M%S")
    letter = str(suffix or "").strip().upper() or "F"
    return f"{timestamp}{letter}"


def normalize_image_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        numeric = float(text)
        if numeric.is_integer():
            return str(int(numeric))
    except Exception:
        return text
    return text


def build_hidden_model_id(model_id: Any, scenario_suffix: str) -> str:
    base = str(model_id or "").strip()
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", base).strip("-")
    if not cleaned:
        cleaned = "model"
    suffix = str(scenario_suffix or "").strip().upper() or "X"
    return f"{cleaned}-{suffix}"


def build_record_lookup(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for record in records:
        image_id = normalize_image_id(record.get("image_id"))
        if not image_id:
            continue
        lookup[image_id] = record
    return lookup


def extract_clinical_report(payload: dict[str, Any], record: dict[str, Any]) -> str:
    direct_candidates = [
        payload.get("clinical_report"),
        payload.get("clinical_justification"),
        payload.get("diagnostic_rationale"),
        record.get("clinical_report"),
        record.get("clinical_justification"),
        record.get("diagnostic_rationale"),
    ]
    for value in direct_candidates:
        text = str(value or "").strip()
        if text:
            return text

    parts: list[str] = []
    for key in (
        "lesion_morphology",
        "surface_vascular_pattern",
        "morphology_and_borders",
        "surface_and_vascular_pattern",
    ):
        text = str(payload.get(key) or "").strip()
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def resolve_paths(config: DualValidationConfig, start: Path | None = None) -> DualValidationPaths:
    project_root = detect_project_root(start=start)

    qwen_run_id_a = config.qwen_run_id_a or config.run_id_a
    qwen_run_id_f = config.qwen_run_id_f or config.run_id_f
    qwen_run_override_a = config.qwen_run_path_override_a or config.run_path_override_a
    qwen_run_override_f = config.qwen_run_path_override_f or config.run_path_override_f

    run_dir_qwen_a = resolve_run_dir(
        project_root=project_root,
        scenario_name=config.scenario_a_name,
        run_id=qwen_run_id_a,
        run_path_override=qwen_run_override_a,
    )
    run_dir_qwen_f = resolve_run_dir(
        project_root=project_root,
        scenario_name=config.scenario_f_name,
        run_id=qwen_run_id_f,
        run_path_override=qwen_run_override_f,
    )

    run_dir_gemma_a = resolve_run_dir(
        project_root=project_root,
        scenario_name=config.scenario_a_name,
        run_id=config.gemma_run_id_a,
        run_path_override=config.gemma_run_path_override_a,
    )
    run_dir_gemma_f = resolve_run_dir(
        project_root=project_root,
        scenario_name=config.scenario_f_name,
        run_id=config.gemma_run_id_f,
        run_path_override=config.gemma_run_path_override_f,
    )

    output_dist_dir = config.output_dist_dir if config.output_dist_dir.is_absolute() else project_root / config.output_dist_dir
    output_dist_dir = output_dist_dir.resolve()
    output_app_a_dir = output_dist_dir / config.app_a_name
    output_app_f_dir = output_dist_dir / config.app_f_name
    output_app_a_assets_dir = output_app_a_dir
    output_app_f_assets_dir = output_app_f_dir
    output_app_a_images_dir = output_app_a_dir / "images"
    output_app_f_images_dir = output_app_f_dir / "images"

    return DualValidationPaths(
        project_root=project_root,
        run_dir_qwen_a=run_dir_qwen_a,
        run_dir_qwen_f=run_dir_qwen_f,
        run_dir_gemma_a=run_dir_gemma_a,
        run_dir_gemma_f=run_dir_gemma_f,
        results_jsonl_qwen_a=run_dir_qwen_a / "results.jsonl",
        results_jsonl_qwen_f=run_dir_qwen_f / "results.jsonl",
        results_jsonl_gemma_a=run_dir_gemma_a / "results.jsonl",
        results_jsonl_gemma_f=run_dir_gemma_f / "results.jsonl",
        output_dist_dir=output_dist_dir,
        output_app_a_dir=output_app_a_dir,
        output_app_f_dir=output_app_f_dir,
        output_app_a_assets_dir=output_app_a_assets_dir,
        output_app_f_assets_dir=output_app_f_assets_dir,
        output_app_a_images_dir=output_app_a_images_dir,
        output_app_f_images_dir=output_app_f_images_dir,
    )


def _class_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for row in rows:
        key = str(row.get("ground_truth_cls_norm") or "UNKNOWN")
        distribution[key] = distribution.get(key, 0) + 1
    return distribution


def _resolve_source_image_path(project_root: Path, image_path_raw: Any) -> Path:
    image_path = Path(str(image_path_raw or "").strip())
    if not image_path.is_absolute():
        image_path = (project_root / image_path).resolve()
    return image_path


def _load_run_records(paths: DualValidationPaths) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        "qwen_a": build_record_lookup(read_jsonl_records(paths.results_jsonl_qwen_a)),
        "qwen_f": build_record_lookup(read_jsonl_records(paths.results_jsonl_qwen_f)),
        "gemma_a": build_record_lookup(read_jsonl_records(paths.results_jsonl_gemma_a)),
        "gemma_f": build_record_lookup(read_jsonl_records(paths.results_jsonl_gemma_f)),
    }


def _extract_record_payload(record: dict[str, Any]) -> dict[str, Any]:
    payload = record.get("payload")
    if isinstance(payload, dict):
        return cast(dict[str, Any], payload)
    return {}


def _build_model_case_payload(
    *,
    record: dict[str, Any],
    gt_class: str,
    scenario_name: str,
    model_id_fallback: str,
) -> dict[str, Any]:
    payload = _extract_record_payload(record)
    clinical_justification = extract_clinical_report(payload, record)
    if not clinical_justification:
        raise ValueError("clinical_justification vacia")

    predicted_class = normalize_class_label(extract_predicted_class(payload))
    if scenario_name == "scenario_F":
        predicted_class = gt_class
    elif not predicted_class:
        raise ValueError("final_diagnosis_class vacio")

    model_id = str(record.get("model_id") or model_id_fallback or "").strip()
    if not model_id:
        raise ValueError("model_id vacio")

    return {
        "ai_predicted_class": predicted_class,
        "clinical_justification": clinical_justification,
        "id_modelo_oculto": model_id,
        "source_record": record,
    }


def prepare_master_selection(config: DualValidationConfig, paths: DualValidationPaths) -> DualValidationSelectionResult:
    run_records = _load_run_records(paths)

    common_image_ids = set(run_records["qwen_a"].keys())
    for key in ("qwen_f", "gemma_a", "gemma_f"):
        common_image_ids &= set(run_records[key].keys())

    candidates: list[dict[str, Any]] = []
    skipped_by_reason: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped_by_reason[reason] = skipped_by_reason.get(reason, 0) + 1

    for image_id in sorted(common_image_ids):
        rec_qwen_a = run_records["qwen_a"].get(image_id)
        rec_qwen_f = run_records["qwen_f"].get(image_id)
        rec_gemma_a = run_records["gemma_a"].get(image_id)
        rec_gemma_f = run_records["gemma_f"].get(image_id)

        if rec_qwen_a is None or rec_qwen_f is None or rec_gemma_a is None or rec_gemma_f is None:
            skip("missing_in_one_model")
            continue

        records_by_model = {
            "qwen": {"A": rec_qwen_a, "F": rec_qwen_f, "model_id": str(rec_qwen_a.get("model_id") or "").strip()},
            "gemma": {"A": rec_gemma_a, "F": rec_gemma_f, "model_id": str(rec_gemma_a.get("model_id") or "").strip()},
        }

        gt_class = normalize_class_label(rec_qwen_f.get("ground_truth_cls"))
        if not gt_class:
            skip("missing_ground_truth_class")
            continue

        gt_bbox = to_bbox_int4(rec_qwen_f.get("ground_truth_bbox"))
        if gt_bbox is None:
            skip("invalid_bbox")
            continue

        image_path_qwen_f = _resolve_source_image_path(paths.project_root, rec_qwen_f.get("image_path"))
        image_path_gemma_f = _resolve_source_image_path(paths.project_root, rec_gemma_f.get("image_path"))
        if not image_path_qwen_f.exists() or not image_path_gemma_f.exists():
            skip("image_not_found")
            continue

        if image_path_qwen_f != image_path_gemma_f:
            skip("image_path_mismatch")
            continue

        source_image_path = image_path_qwen_f

        source_image_name = str(rec_qwen_f.get("image_name") or source_image_path.name)

        model_payloads: dict[str, dict[str, dict[str, Any]]] = {}
        quality_scores: list[float] = []

        for model_key, model_records in records_by_model.items():
            scenario_payloads: dict[str, dict[str, Any]] = {}
            for scenario_name in ("A", "F"):
                record = cast(dict[str, Any], model_records[scenario_name])
                if str(record.get("status", "")).lower() != "ok":
                    skip(f"{model_key.lower()}_{scenario_name.lower()}_status_not_ok")
                    break
                payload = _build_model_case_payload(
                    record=record,
                    gt_class=gt_class,
                    scenario_name=f"scenario_{scenario_name}",
                    model_id_fallback=model_records["model_id"],
                )
                scenario_payloads[scenario_name] = payload
            else:
                model_payloads[model_key] = scenario_payloads
                iou_score = to_float_or_none(model_records["F"].get("iou_score"))
                proximity_score = to_float_or_none(model_records["F"].get("proximity_score"))
                if iou_score is None and proximity_score is None:
                    skip(f"{model_key.lower()}_missing_quality_scores")
                    break
                quality_scores.append((iou_score or 0.0) + (proximity_score or 0.0))
                continue
            break
        else:
            candidate_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            candidates.append(
                {
                    "image_id": image_id,
                    "image_name": source_image_name,
                    "image_path": source_image_path,
                    "ground_truth_cls": gt_class,
                    "ground_truth_cls_norm": gt_class,
                    "quality_score": candidate_quality,
                    "iou_score": to_float_or_none(rec_qwen_f.get("iou_score")),
                    "proximity_score": to_float_or_none(rec_qwen_f.get("proximity_score")),
                    "ground_truth_bbox": gt_bbox,
                    "records_by_model": {
                        "qwen": model_payloads["qwen"],
                        "gemma": model_payloads["gemma"],
                    },
                }
            )
            continue

        continue

    selected = select_top_k_normalized_by_class(
        candidates,
        top_k=max(0, int(config.top_k)),
        normalize_by_class=bool(config.normalize_by_class),
        normalization_mode=str(config.normalization_mode),
        valid_classes=tuple(config.valid_classes),
    )

    if len(selected) != max(0, int(config.top_k)):
        raise RuntimeError(
            "No se pudieron seleccionar los casos requeridos desde el Escenario F con correspondencia en A. "
            f"Seleccionados={len(selected)}; requeridos={int(config.top_k)}."
        )

    print(f"Seleccionados {len(selected)} casos basados en el rendimiento del Escenario F")

    return DualValidationSelectionResult(
        candidates=candidates,
        selected=selected,
        dist_candidates=_class_distribution(candidates),
        dist_selected=_class_distribution(selected),
        skipped_by_reason=skipped_by_reason,
    )


def _reset_directory(target_dir: Path) -> None:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)


def _copy_static_assets(source_app_dir: Path, target_app_dir: Path) -> None:
    for file_name in ("index.html", "style.css", "app.js"):
        shutil.copy2(source_app_dir / file_name, target_app_dir / file_name)


def _build_case_payloads(
    *,
    config: DualValidationConfig,
    paths: DualValidationPaths,
    selection: DualValidationSelectionResult,
    bundle_scenario: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    output_app_dir = paths.output_app_a_dir if bundle_scenario == "A" else paths.output_app_f_dir
    output_images_dir = paths.output_app_a_images_dir if bundle_scenario == "A" else paths.output_app_f_images_dir

    output_images_dir.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, Any]] = []
    selected_ids: list[str] = []

    for row in selection.selected:
        image_id = str(row["image_id"])
        selected_ids.append(image_id)

        source_image_path: Path = row["image_path"]
        output_image_name = build_output_image_name(source_image_path, image_id)
        output_image_path = output_images_dir / output_image_name

        image_bgr = load_image_bgr(source_image_path)
        image_annotated = draw_gt_bbox_only(
            image_bgr=image_bgr,
            gt_bbox_norm=row["ground_truth_bbox"],
            scale_max=config.scale_max,
            gt_color_bgr=config.gt_color_bgr,
            gt_thickness=config.gt_thickness,
        )

        if not cv2.imwrite(str(output_image_path), image_annotated):
            raise RuntimeError(f"No se pudo guardar la imagen anotada: {output_image_path}")

        scenario_payloads = cast(dict[str, Any], row["records_by_model"])
        for model_key in ("qwen", "gemma"):
            model_scenario_payload = cast(dict[str, Any], scenario_payloads[model_key])
            payload = cast(dict[str, Any], model_scenario_payload[bundle_scenario])
            case = {
                "id_imagen": image_id,
                "image_path": output_image_path.resolve().as_uri(),
                "ground_truth_class": row["ground_truth_cls"],
                "ai_predicted_class": payload["ai_predicted_class"],
                "clinical_justification": payload["clinical_justification"],
                "id_modelo_oculto": payload["id_modelo_oculto"],
            }
            cases.append(case)

    random.seed(42)
    random.shuffle(cases)

    csv_rows = [
        {
            "id_imagen": str(case["id_imagen"]),
            "id_modelo_oculto": str(case["id_modelo_oculto"]),
            "veredicto": "",
            "comentario_medico": "",
        }
        for case in cases
    ]

    return cases, csv_rows, selected_ids


def resolve_web_paths(
    config: DualValidationConfig,
    *,
    execution_id: str,
    start: Path | None = None,
) -> DualValidationPaths:
    base_paths = resolve_paths(config=config, start=start)
    execution_token = str(execution_id or "").strip()
    if not execution_token:
        raise ValueError("execution_id no puede estar vacio")

    web_data_root = base_paths.output_dist_dir

    return DualValidationPaths(
        project_root=base_paths.project_root,
        run_dir_qwen_a=base_paths.run_dir_qwen_a,
        run_dir_qwen_f=base_paths.run_dir_qwen_f,
        run_dir_gemma_a=base_paths.run_dir_gemma_a,
        run_dir_gemma_f=base_paths.run_dir_gemma_f,
        results_jsonl_qwen_a=base_paths.results_jsonl_qwen_a,
        results_jsonl_qwen_f=base_paths.results_jsonl_qwen_f,
        results_jsonl_gemma_a=base_paths.results_jsonl_gemma_a,
        results_jsonl_gemma_f=base_paths.results_jsonl_gemma_f,
        output_dist_dir=web_data_root,
        output_app_a_dir=web_data_root / config.app_a_name / execution_token,
        output_app_f_dir=web_data_root / config.app_f_name / execution_token,
        output_app_a_assets_dir=web_data_root / config.app_a_name / execution_token,
        output_app_f_assets_dir=web_data_root / config.app_f_name / execution_token,
        output_app_a_images_dir=web_data_root / config.app_a_name / execution_token / "images",
        output_app_f_images_dir=web_data_root / config.app_f_name / execution_token / "images",
    )


def _export_dual_validation_to_paths(
    *,
    config: DualValidationConfig,
    paths: DualValidationPaths,
    selection: DualValidationSelectionResult,
) -> DualValidationExportResult:
    _reset_directory(paths.output_app_a_dir)
    _reset_directory(paths.output_app_f_dir)

    cases_a, csv_rows_a, selected_ids_a = _build_case_payloads(config=config, paths=paths, selection=selection, bundle_scenario="A")
    cases_f, csv_rows_f, selected_ids_f = _build_case_payloads(config=config, paths=paths, selection=selection, bundle_scenario="F")

    output_app_a_cases_json = paths.output_app_a_dir / "cases.json"
    output_app_f_cases_json = paths.output_app_f_dir / "cases.json"
    output_app_a_export_csv = paths.output_app_a_dir / "cases_export_debug.csv"
    output_app_f_export_csv = paths.output_app_f_dir / "cases_export_debug.csv"

    with output_app_a_cases_json.open("w", encoding="utf-8") as handle:
        json.dump(cases_a, handle, ensure_ascii=False, indent=2)
    with output_app_f_cases_json.open("w", encoding="utf-8") as handle:
        json.dump(cases_f, handle, ensure_ascii=False, indent=2)

    with output_app_a_export_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id_imagen", "id_modelo_oculto", "veredicto", "comentario_medico"])
        writer.writeheader()
        writer.writerows(csv_rows_a)

    with output_app_f_export_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id_imagen", "id_modelo_oculto", "veredicto", "comentario_medico"])
        writer.writeheader()
        writer.writerows(csv_rows_f)

    selected_ids = selected_ids_a if selected_ids_a else selected_ids_f
    return DualValidationExportResult(
        selected_image_ids=selected_ids,
        output_app_a_cases_json=output_app_a_cases_json,
        output_app_f_cases_json=output_app_f_cases_json,
        output_app_a_export_csv=output_app_a_export_csv,
        output_app_f_export_csv=output_app_f_export_csv,
        output_app_a_images_dir=paths.output_app_a_images_dir,
        output_app_f_images_dir=paths.output_app_f_images_dir,
    )


def export_dual_validation_web_data(
    *,
    config: DualValidationConfig,
    execution_id: str,
    start: Path | None = None,
) -> tuple[DualValidationPaths, DualValidationSelectionResult, DualValidationExportResult]:
    paths = resolve_web_paths(config, execution_id=execution_id, start=start)
    selection = prepare_master_selection(config, paths)
    export_result = _export_dual_validation_to_paths(config=config, paths=paths, selection=selection)
    return paths, selection, export_result


def export_dual_validation_bundle(
    *,
    config: DualValidationConfig,
    paths: DualValidationPaths,
    selection: DualValidationSelectionResult,
) -> DualValidationExportResult:
    return _export_dual_validation_to_paths(config=config, paths=paths, selection=selection)


def run_pipeline(config: DualValidationConfig) -> tuple[DualValidationPaths, DualValidationSelectionResult, DualValidationExportResult]:
    paths = resolve_paths(config)
    selection = prepare_master_selection(config, paths)
    export_result = export_dual_validation_bundle(config=config, paths=paths, selection=selection)
    return paths, selection, export_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Genera dos paquetes de validacion clinica desde los escenarios A y F, "
            "usando F como seleccion maestra y duplicando 50 imagenes por modelo."
        )
    )
    parser.add_argument("--scenario-a-name", default="scenario_A")
    parser.add_argument("--scenario-f-name", default="scenario_F")
    parser.add_argument("--qwen-model-id", default="qwen3_5-9b@q8_0")
    parser.add_argument("--gemma-model-id", default="google/gemma-4-e4b@q8_0")
    parser.add_argument("--run-id-a", default=None)
    parser.add_argument("--run-id-f", default=None)
    parser.add_argument("--run-path-override-a", default=None)
    parser.add_argument("--run-path-override-f", default=None)
    parser.add_argument("--qwen-run-id-a", default=None)
    parser.add_argument("--qwen-run-id-f", default=None)
    parser.add_argument("--qwen-run-path-override-a", default=None)
    parser.add_argument("--qwen-run-path-override-f", default=None)
    parser.add_argument("--gemma-run-id-a", default=None)
    parser.add_argument("--gemma-run-id-f", default=None)
    parser.add_argument("--gemma-run-path-override-a", default=None)
    parser.add_argument("--gemma-run-path-override-f", default=None)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--normalize-by-class", action="store_true", default=True)
    parser.add_argument("--no-normalize-by-class", action="store_false", dest="normalize_by_class")
    parser.add_argument("--normalization-mode", choices=["proportional", "uniform"], default="proportional")
    parser.add_argument("--valid-classes", nargs="+", default=["AD", "HP", "ASS"])
    parser.add_argument("--output-dist-dir", type=Path, default=Path("dist"))
    parser.add_argument("--source-app-dir", type=Path, default=Path("clinical_eval_app"))
    parser.add_argument("--app-a-name", default="App_Autonoma")
    parser.add_argument("--app-f-name", default="App_Asistida")
    parser.add_argument("--gt-thickness", type=int, default=3)
    parser.add_argument("--scale-max", type=int, default=1000)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    config = DualValidationConfig(
        scenario_a_name=str(args.scenario_a_name),
        scenario_f_name=str(args.scenario_f_name),
        qwen_model_id=str(args.qwen_model_id),
        gemma_model_id=str(args.gemma_model_id),
        run_id_a=args.run_id_a,
        run_id_f=args.run_id_f,
        run_path_override_a=args.run_path_override_a,
        run_path_override_f=args.run_path_override_f,
        qwen_run_id_a=args.qwen_run_id_a,
        qwen_run_id_f=args.qwen_run_id_f,
        qwen_run_path_override_a=args.qwen_run_path_override_a,
        qwen_run_path_override_f=args.qwen_run_path_override_f,
        gemma_run_id_a=args.gemma_run_id_a,
        gemma_run_id_f=args.gemma_run_id_f,
        gemma_run_path_override_a=args.gemma_run_path_override_a,
        gemma_run_path_override_f=args.gemma_run_path_override_f,
        top_k=int(args.top_k),
        normalize_by_class=bool(args.normalize_by_class),
        normalization_mode=str(args.normalization_mode),
        valid_classes=tuple(str(value).strip().upper() for value in args.valid_classes if str(value).strip()),
        output_dist_dir=Path(args.output_dist_dir),
        source_app_dir=Path(args.source_app_dir),
        app_a_name=str(args.app_a_name),
        app_f_name=str(args.app_f_name),
        gt_thickness=int(args.gt_thickness),
        scale_max=int(args.scale_max),
    )

    paths, selection, export_result = run_pipeline(config)

    print(f"PROJECT_ROOT: {paths.project_root}")
    print(f"RUN_DIR_QWEN_A:  {paths.run_dir_qwen_a}")
    print(f"RUN_DIR_QWEN_F:  {paths.run_dir_qwen_f}")
    print(f"RUN_DIR_GEMMA_A: {paths.run_dir_gemma_a}")
    print(f"RUN_DIR_GEMMA_F: {paths.run_dir_gemma_f}")
    print(f"JSONL_QWEN_A:    {paths.results_jsonl_qwen_a}")
    print(f"JSONL_QWEN_F:    {paths.results_jsonl_qwen_f}")
    print(f"JSONL_GEMMA_A:   {paths.results_jsonl_gemma_a}")
    print(f"JSONL_GEMMA_F:   {paths.results_jsonl_gemma_f}")
    print(f"DIST_DIR:     {paths.output_dist_dir}")
    print(f"App A cases:  {export_result.output_app_a_cases_json.resolve()}")
    print(f"App F cases:  {export_result.output_app_f_cases_json.resolve()}")
    print(f"App A images: {export_result.output_app_a_images_dir.resolve()}")
    print(f"App F images: {export_result.output_app_f_images_dir.resolve()}")
    print(f"Distribucion candidatos: {selection.dist_candidates}")
    print(f"Distribucion seleccion:  {selection.dist_selected}")
    if selection.skipped_by_reason:
        print(f"Descartes por motivo:    {selection.skipped_by_reason}")

    return 0


if __name__ == "__main__":
    sys.exit(main())