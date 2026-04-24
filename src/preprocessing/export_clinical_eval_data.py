from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from PIL import Image
except Exception:
    Image = None


@dataclass(frozen=True)
class ExportConfig:
    scenario_name: str = "scenario_F"
    run_id: str | None = None
    run_path_override: str | None = None

    top_k: int = 50
    only_class_match: bool = True
    require_status_ok: bool = True
    drop_empty_justification: bool = True

    normalize_by_class: bool = True
    normalization_mode: str = "proportional"
    valid_classes: tuple[str, ...] = ("AD", "HP", "ASS")

    export_mode: str = "replace"  # replace | scenario_id | append
    scenario_uid_override: str | None = None

    output_app_dir: Path = Path("clinical_eval_app")

    gt_color_bgr: tuple[int, int, int] = (0, 200, 0)
    gt_thickness: int = 3
    scale_max: int = 1000


@dataclass(frozen=True)
class ExportPaths:
    project_root: Path
    run_dir: Path
    results_jsonl: Path
    output_app_dir: Path
    output_data_dir: Path
    output_images_dir: Path
    output_cases_json: Path
    output_export_csv: Path


@dataclass(frozen=True)
class SelectionResult:
    prepared: list[dict[str, Any]]
    selected: list[dict[str, Any]]
    dist_prepared: dict[str, int]
    dist_selected: dict[str, int]
    skipped_by_reason: dict[str, int]


@dataclass(frozen=True)
class ExportResult:
    cases: list[dict[str, Any]]
    new_cases_count: int
    existing_cases_count: int
    mode: str
    scenario_uid: str
    output_cases_json: Path
    output_export_csv: Path
    output_images_dir: Path


def extract_scenario_letter(scenario_name: str) -> str:
    text = str(scenario_name or "").strip()
    if not text:
        return "X"
    suffix = text.split("_")[-1].strip()
    if len(suffix) == 1 and suffix.isalpha():
        return suffix.upper()
    for char in reversed(text):
        if char.isalpha():
            return char.upper()
    return "X"


def build_default_scenario_uid(*, scenario_name: str, run_dir_name: str) -> str:
    run_token = str(run_dir_name or "").strip()
    if run_token.startswith("run_"):
        run_token = run_token[4:]
    if not run_token:
        run_token = "run"
    return f"{run_token}{extract_scenario_letter(scenario_name)}"


def sanitize_folder_name(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    cleaned = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in text)
    compact = "_".join(part for part in cleaned.split("_") if part)
    return compact or fallback


def detect_project_root(start: Path | None = None) -> Path:
    base = (start or Path.cwd()).resolve()
    for candidate in [base, *base.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "data").exists():
            return candidate
    return base


def resolve_paths(config: ExportConfig, start: Path | None = None) -> ExportPaths:
    project_root = detect_project_root(start=start)
    run_dir = resolve_run_dir(
        project_root=project_root,
        scenario_name=config.scenario_name,
        run_id=config.run_id,
        run_path_override=config.run_path_override,
    )

    output_app_dir = (
        config.output_app_dir
        if config.output_app_dir.is_absolute()
        else (project_root / config.output_app_dir)
    ).resolve()
    output_data_dir = output_app_dir / "data"
    output_images_dir = output_data_dir / "images"
    output_cases_json = output_data_dir / "cases.json"
    output_export_csv = output_data_dir / "cases_export_debug.csv"

    return ExportPaths(
        project_root=project_root,
        run_dir=run_dir,
        results_jsonl=run_dir / "results.jsonl",
        output_app_dir=output_app_dir,
        output_data_dir=output_data_dir,
        output_images_dir=output_images_dir,
        output_cases_json=output_cases_json,
        output_export_csv=output_export_csv,
    )


def resolve_run_dir(
    *,
    project_root: Path,
    scenario_name: str,
    run_id: str | None,
    run_path_override: str | None,
) -> Path:
    if run_path_override:
        raw = Path(run_path_override)
        run_dir = raw if raw.is_absolute() else (project_root / raw)
        run_dir = run_dir.resolve()
        if not (run_dir / "results.jsonl").exists():
            raise FileNotFoundError(f"No existe results.jsonl en: {run_dir}")
        return run_dir

    scenario_dir = project_root / "data" / "processed" / "scenario_results" / scenario_name
    if not scenario_dir.exists():
        raise FileNotFoundError(f"Escenario no encontrado: {scenario_dir}")

    if run_id:
        run_id_text = str(run_id).strip()
        run_dir_name = run_id_text if run_id_text.startswith("run_") else f"run_{run_id_text}"
        run_dir = scenario_dir / run_dir_name
        if not (run_dir / "results.jsonl").exists():
            raise FileNotFoundError(
                f"No existe results.jsonl para RUN_ID={run_id}: {run_dir} "
                "(usa RUN_ID con o sin prefijo run_)"
            )
        return run_dir

    runs = sorted(
        [path for path in scenario_dir.glob("run_*") if (path / "results.jsonl").exists()],
        key=lambda path: path.name,
    )
    if not runs:
        raise FileNotFoundError(f"No hay runs con results.jsonl en: {scenario_dir}")
    return runs[-1]


def read_jsonl_records(jsonl_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON invalido en {jsonl_path}:{line_no} ({exc})") from exc
            if not isinstance(obj, dict) or "__scenario_meta__" in obj:
                continue
            records.append(obj)
    return records


def to_float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def to_bbox_int4(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    out: list[int] = []
    for item in value:
        try:
            out.append(int(round(float(item))))
        except Exception:
            return None
    return out


def normalize_class_label(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text.startswith("AD") or "ADENOMA" in text:
        return "AD"
    if text.startswith("HP") or "HIPER" in text:
        return "HP"
    if text.startswith("ASS") or "SERR" in text:
        return "ASS"
    return text


def extract_clinical_justification(payload: dict[str, Any], rec: dict[str, Any]) -> str:
    direct_candidates = [
        payload.get("clinical_justification"),
        payload.get("diagnostic_rationale"),
        payload.get("justification"),
        rec.get("clinical_justification"),
        rec.get("diagnostic_rationale"),
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


def _compute_target_counts(class_counts: dict[str, int], sample_size: int) -> dict[str, int]:
    if sample_size <= 0:
        raise ValueError("sample_size must be greater than zero")

    total_rows = sum(class_counts.values())
    if sample_size >= total_rows:
        return dict(class_counts)

    classes = list(class_counts.keys())
    target_counts = {class_name: 0 for class_name in classes}

    if sample_size >= len(classes):
        for class_name, count in class_counts.items():
            if count > 0:
                target_counts[class_name] = 1

    reserved = sum(target_counts.values())
    remaining = sample_size - reserved
    if remaining <= 0:
        return target_counts

    fractional_allocations: list[tuple[str, float]] = []
    for class_name, count in class_counts.items():
        available = count - target_counts[class_name]
        if available <= 0:
            continue

        exact = remaining * (count / total_rows)
        floor_value = min(int(np.floor(exact)), available)
        target_counts[class_name] += floor_value
        fractional_allocations.append((class_name, exact - floor_value))

    distributed = sum(target_counts.values())
    still_needed = sample_size - distributed
    if still_needed <= 0:
        return target_counts

    ranked = sorted(
        fractional_allocations,
        key=lambda item: (item[1], class_counts[item[0]]),
        reverse=True,
    )
    rank_index = 0
    while still_needed > 0 and ranked:
        class_name = ranked[rank_index % len(ranked)][0]
        if target_counts[class_name] < class_counts[class_name]:
            target_counts[class_name] += 1
            still_needed -= 1
        rank_index += 1

    return target_counts


def _compute_uniform_target_counts(class_counts: dict[str, int], sample_size: int) -> dict[str, int]:
    if sample_size <= 0:
        raise ValueError("sample_size must be greater than zero")

    classes = list(class_counts.keys())
    if not classes:
        return {}

    target = {cls: 0 for cls in classes}
    remaining = sample_size
    ordered = sorted(classes)

    while remaining > 0:
        progressed = False
        for cls in ordered:
            if remaining <= 0:
                break
            if target[cls] >= class_counts[cls]:
                continue
            target[cls] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            break

    return target


def select_top_k_normalized_by_class(
    candidates: list[dict[str, Any]],
    *,
    top_k: int,
    normalize_by_class: bool,
    normalization_mode: str,
    valid_classes: tuple[str, ...],
) -> list[dict[str, Any]]:
    ranked = sorted(candidates, key=lambda item: float(item.get("quality_score") or 0.0), reverse=True)
    if top_k <= 0:
        return []
    if not normalize_by_class or len(ranked) <= top_k:
        return ranked[:top_k]

    class_buckets: dict[str, list[dict[str, Any]]] = {cls: [] for cls in valid_classes}
    fallback_pool: list[dict[str, Any]] = []

    for item in ranked:
        cls = str(item.get("ground_truth_cls_norm") or "").strip().upper()
        if cls in class_buckets:
            class_buckets[cls].append(item)
        else:
            fallback_pool.append(item)

    available_counts = {cls: len(items) for cls, items in class_buckets.items() if items}
    if not available_counts:
        return ranked[:top_k]

    mode = str(normalization_mode or "proportional").strip().lower()
    if mode == "uniform":
        target_counts = _compute_uniform_target_counts(available_counts, top_k)
    else:
        target_counts = _compute_target_counts(available_counts, top_k)

    selected: list[dict[str, Any]] = []
    for cls in valid_classes:
        take = int(target_counts.get(cls, 0))
        if take <= 0:
            continue
        selected.extend(class_buckets[cls][:take])

    if len(selected) < top_k:
        leftovers: list[dict[str, Any]] = []
        for cls in valid_classes:
            taken = int(target_counts.get(cls, 0))
            leftovers.extend(class_buckets[cls][taken:])
        leftovers.extend(fallback_pool)
        leftovers.sort(key=lambda item: float(item.get("quality_score") or 0.0), reverse=True)
        selected.extend(leftovers[: top_k - len(selected)])

    selected.sort(key=lambda item: float(item.get("quality_score") or 0.0), reverse=True)
    return selected[:top_k]


def prepare_and_select_cases(config: ExportConfig, paths: ExportPaths) -> SelectionResult:
    records = read_jsonl_records(paths.results_jsonl)

    prepared: list[dict[str, Any]] = []
    skipped_by_reason: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped_by_reason[reason] = skipped_by_reason.get(reason, 0) + 1

    for rec in records:
        if config.require_status_ok and str(rec.get("status", "")).lower() != "ok":
            skip("status_not_ok")
            continue

        if config.only_class_match and not bool(rec.get("class_match")):
            skip("class_mismatch")
            continue

        gt_bbox = to_bbox_int4(rec.get("ground_truth_bbox"))
        if gt_bbox is None:
            skip("invalid_bbox")
            continue

        payload_raw = rec.get("payload")
        payload: dict[str, Any] = payload_raw if isinstance(payload_raw, dict) else {}
        justification = extract_clinical_justification(payload, rec)
        if config.drop_empty_justification and not justification:
            skip("empty_justification")
            continue

        image_path_raw = str(rec.get("image_path") or "").strip()
        if not image_path_raw:
            skip("missing_image_path")
            continue

        image_path = Path(image_path_raw)
        image_path = image_path if image_path.is_absolute() else (paths.project_root / image_path)
        image_path = image_path.resolve()
        if not image_path.exists():
            skip("image_not_found")
            continue

        iou = to_float_or_none(rec.get("iou_score"))
        prox = to_float_or_none(rec.get("proximity_score"))

        prepared.append(
            {
                "image_id": rec.get("image_id"),
                "image_name": str(rec.get("image_name") or image_path.name),
                "image_path": image_path,
                "ground_truth_cls": str(rec.get("ground_truth_cls") or "").strip(),
                "ground_truth_cls_norm": normalize_class_label(rec.get("ground_truth_cls")),
                "ground_truth_bbox": gt_bbox,
                "clinical_justification": justification,
                "id_modelo": str(rec.get("model_id") or "").strip(),
                "iou_score": iou,
                "proximity_score": prox,
                "quality_score": (iou or 0.0) + (prox or 0.0),
            }
        )

    selected = select_top_k_normalized_by_class(
        prepared,
        top_k=max(0, int(config.top_k)),
        normalize_by_class=bool(config.normalize_by_class),
        normalization_mode=str(config.normalization_mode),
        valid_classes=tuple(config.valid_classes),
    )

    dist_prepared = _class_distribution(prepared)
    dist_selected = _class_distribution(selected)

    return SelectionResult(
        prepared=prepared,
        selected=selected,
        dist_prepared=dist_prepared,
        dist_selected=dist_selected,
        skipped_by_reason=skipped_by_reason,
    )


def _class_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    dist: dict[str, int] = {}
    for row in rows:
        key = str(row.get("ground_truth_cls_norm") or "UNKNOWN")
        dist[key] = dist.get(key, 0) + 1
    return dist


def export_selected_cases(
    *,
    config: ExportConfig,
    paths: ExportPaths,
    selected: list[dict[str, Any]],
) -> ExportResult:
    mode = str(config.export_mode or "replace").strip().lower()
    valid_modes = {"replace", "scenario_id", "append"}
    if mode not in valid_modes:
        raise ValueError(f"EXPORT_MODE invalido: {config.export_mode}. Usa uno de {sorted(valid_modes)}")

    scenario_uid = str(config.scenario_uid_override or "").strip() or build_default_scenario_uid(
        scenario_name=config.scenario_name,
        run_dir_name=paths.run_dir.name,
    )

    # scenario_id trabaja en un paquete dedicado: data/<scenario_uid>/
    # replace/append trabajan en data/ raiz para mantener compatibilidad con la app.
    if mode == "scenario_id":
        scenario_token = sanitize_folder_name(scenario_uid, "scenario")
        output_data_dir = paths.output_data_dir / scenario_token
    else:
        output_data_dir = paths.output_data_dir

    output_images_dir = output_data_dir / "images"
    output_cases_json = output_data_dir / "cases.json"
    output_export_csv = output_data_dir / "cases_export_debug.csv"

    output_data_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_cases_json.parent.mkdir(parents=True, exist_ok=True)

    use_scenario_uid_in_case_id = mode in {"scenario_id", "append"}

    if mode == "replace":
        for image_file in output_images_dir.glob("*"):
            if image_file.is_file():
                image_file.unlink()

    existing_cases: list[dict[str, Any]] = []
    if mode == "append" and output_cases_json.exists():
        try:
            with output_cases_json.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, list):
                existing_cases = [item for item in loaded if isinstance(item, dict)]
        except Exception:
            existing_cases = []

    new_cases: list[dict[str, Any]] = []
    csv_rows_new: list[dict[str, Any]] = []

    for row in selected:
        src_img_path: Path = row["image_path"]
        gt_bbox_norm: list[int] = row["ground_truth_bbox"]

        base_image_id = str(row.get("image_id") or src_img_path.stem).strip() or src_img_path.stem
        model_id = str(row.get("id_modelo") or "").strip()
        case_id = build_case_id(
            base_image_id=base_image_id,
            model_id=model_id,
            scenario_uid_text=scenario_uid,
            with_scenario_uid=use_scenario_uid_in_case_id,
        )

        out_name = build_output_image_name(src_img_path, base_image_id)
        out_abs = output_images_dir / out_name
        if mode == "scenario_id":
            scenario_token = sanitize_folder_name(scenario_uid, "scenario")
            image_file_for_app = f"data/{scenario_token}/images/{out_name}"
        else:
            image_file_for_app = f"data/images/{out_name}"

        if not (mode == "append" and out_abs.exists()):
            image_bgr = load_image_bgr(src_img_path)
            image_annotated = draw_gt_bbox_only(
                image_bgr=image_bgr,
                gt_bbox_norm=gt_bbox_norm,
                scale_max=config.scale_max,
                gt_color_bgr=config.gt_color_bgr,
                gt_thickness=config.gt_thickness,
            )
            ok = cv2.imwrite(str(out_abs), image_annotated)
            if not ok:
                raise RuntimeError(f"No se pudo guardar: {out_abs}")

        case_obj = {
            "id_imagen": case_id,
            "image_path": image_file_for_app,
            "ground_truth_class": row["ground_truth_cls"],
            "id_modelo": model_id,
            "clinical_justification": row["clinical_justification"],
            "scenario_uid": scenario_uid,
            "source_image_id": base_image_id,
        }
        new_cases.append(case_obj)

        csv_rows_new.append(
            {
                "id_imagen": case_id,
                "id_modelo": model_id,
                "veredicto": "",
                "comentario_medico": "",
                "source_image_id": row["image_id"],
                "source_image_name": row["image_name"],
                "source_image_path": str(src_img_path),
                "exported_image_path": image_file_for_app,
                "true_class": row["ground_truth_cls"],
                "iou_score": row["iou_score"],
                "proximity_score": row["proximity_score"],
                "quality_score": row["quality_score"],
                "scenario_uid": scenario_uid,
            }
        )

    if mode == "append":
        new_ids = {str(item.get("id_imagen", "")) for item in new_cases}
        kept_existing = [item for item in existing_cases if str(item.get("id_imagen", "")) not in new_ids]
        cases = kept_existing + new_cases
    else:
        cases = new_cases

    with output_cases_json.open("w", encoding="utf-8") as handle:
        json.dump(cases, handle, ensure_ascii=False, indent=2)

    csv_rows_output: list[dict[str, Any]] = []
    new_row_by_id = {str(row.get("id_imagen", "")): row for row in csv_rows_new}
    for case in cases:
        case_id = str(case.get("id_imagen", ""))
        row = new_row_by_id.get(case_id)
        if row is not None:
            csv_rows_output.append(row)
            continue

        csv_rows_output.append(
            {
                "id_imagen": case_id,
                "id_modelo": str(case.get("id_modelo", "")),
                "veredicto": "",
                "comentario_medico": "",
                "source_image_id": str(case.get("source_image_id", "")),
                "source_image_name": "",
                "source_image_path": "",
                "exported_image_path": str(case.get("image_path", "")),
                "true_class": str(case.get("ground_truth_class", "")),
                "iou_score": "",
                "proximity_score": "",
                "quality_score": "",
                "scenario_uid": str(case.get("scenario_uid", "")),
            }
        )

    with output_export_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id_imagen",
                "id_modelo",
                "veredicto",
                "comentario_medico",
                "source_image_id",
                "source_image_name",
                "source_image_path",
                "exported_image_path",
                "true_class",
                "iou_score",
                "proximity_score",
                "quality_score",
                "scenario_uid",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows_output)

    return ExportResult(
        cases=cases,
        new_cases_count=len(new_cases),
        existing_cases_count=len(existing_cases),
        mode=mode,
        scenario_uid=scenario_uid,
        output_cases_json=output_cases_json,
        output_export_csv=output_export_csv,
        output_images_dir=output_images_dir,
    )


def safe_token(value: Any, fallback: str) -> str:
    text = str(value or "").strip().lower()
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in text)
    compact = "_".join(part for part in cleaned.split("_") if part)
    return compact or fallback


def build_case_id(*, base_image_id: str, model_id: str, scenario_uid_text: str, with_scenario_uid: bool) -> str:
    base_clean = safe_token(base_image_id, "img")
    if not with_scenario_uid:
        return base_clean
    scenario_token = safe_token(scenario_uid_text, "scenario")
    model_token = safe_token(model_id, "model")
    return f"{scenario_token}__{model_token}__{base_clean}"


def build_output_image_name(src_img_path: Path, base_image_id: str) -> str:
    image_token = safe_token(base_image_id, safe_token(src_img_path.stem, "img"))
    return f"img_{image_token}.jpg"


def load_image_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is not None:
        return image

    if Image is not None:
        pil = Image.open(path).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    raise RuntimeError(f"No se pudo cargar imagen: {path}")


def draw_gt_bbox_only(
    *,
    image_bgr: np.ndarray,
    gt_bbox_norm: list[int],
    scale_max: int,
    gt_color_bgr: tuple[int, int, int],
    gt_thickness: int,
) -> np.ndarray:
    out = image_bgr.copy()
    h, w = out.shape[:2]
    x1, y1, x2, y2 = denormalize_bbox_xyxy(gt_bbox_norm, width=w, height=h, scale_max=scale_max)
    cv2.rectangle(out, (x1, y1), (x2, y2), gt_color_bgr, gt_thickness)
    return out


def denormalize_bbox_xyxy(
    bbox_norm: list[int],
    *,
    width: int,
    height: int,
    scale_max: int,
) -> tuple[int, int, int, int]:
    xmin_n, ymin_n, xmax_n, ymax_n = bbox_norm

    xmin = int((xmin_n / scale_max) * width)
    ymin = int((ymin_n / scale_max) * height)
    xmax = int((xmax_n / scale_max) * width)
    ymax = int((ymax_n / scale_max) * height)

    xmin = max(0, min(xmin, width - 1))
    xmax = max(0, min(xmax, width - 1))
    ymin = max(0, min(ymin, height - 1))
    ymax = max(0, min(ymax, height - 1))

    return xmin, ymin, xmax, ymax


def run_pipeline(config: ExportConfig) -> tuple[ExportPaths, SelectionResult, ExportResult]:
    paths = resolve_paths(config)
    selection = prepare_and_select_cases(config=config, paths=paths)
    export_result = export_selected_cases(config=config, paths=paths, selected=selection.selected)
    return paths, selection, export_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Exporta casos para clinical_eval_app desde results.jsonl, con filtrado, "
            "ranking por iou+proximity, normalizacion por clase y modos replace/scenario_id/append."
        )
    )
    parser.add_argument("--scenario-name", default="scenario_F")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-path-override", default=None)

    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--only-class-match", action="store_true", default=True)
    parser.add_argument("--no-only-class-match", action="store_false", dest="only_class_match")
    parser.add_argument("--require-status-ok", action="store_true", default=True)
    parser.add_argument("--no-require-status-ok", action="store_false", dest="require_status_ok")
    parser.add_argument("--drop-empty-justification", action="store_true", default=True)
    parser.add_argument(
        "--no-drop-empty-justification",
        action="store_false",
        dest="drop_empty_justification",
    )

    parser.add_argument("--normalize-by-class", action="store_true", default=True)
    parser.add_argument("--no-normalize-by-class", action="store_false", dest="normalize_by_class")
    parser.add_argument("--normalization-mode", choices=["proportional", "uniform"], default="proportional")
    parser.add_argument("--valid-classes", nargs="+", default=["AD", "HP", "ASS"])

    parser.add_argument("--export-mode", choices=["replace", "scenario_id", "append"], default="replace")
    parser.add_argument("--scenario-uid-override", default=None)
    parser.add_argument("--output-app-dir", type=Path, default=Path("clinical_eval_app"))

    return parser


def main() -> int:
    args = build_parser().parse_args()

    config = ExportConfig(
        scenario_name=str(args.scenario_name),
        run_id=args.run_id,
        run_path_override=args.run_path_override,
        top_k=int(args.top_k),
        only_class_match=bool(args.only_class_match),
        require_status_ok=bool(args.require_status_ok),
        drop_empty_justification=bool(args.drop_empty_justification),
        normalize_by_class=bool(args.normalize_by_class),
        normalization_mode=str(args.normalization_mode),
        valid_classes=tuple(str(value).strip().upper() for value in args.valid_classes if str(value).strip()),
        export_mode=str(args.export_mode),
        scenario_uid_override=args.scenario_uid_override,
        output_app_dir=Path(args.output_app_dir),
    )

    paths, selection, export_result = run_pipeline(config)

    print(f"PROJECT_ROOT: {paths.project_root}")
    print(f"RUN_DIR:      {paths.run_dir}")
    print(f"JSONL:        {paths.results_jsonl}")
    print(f"APP_DIR:      {paths.output_app_dir}")
    print(f"DATA_DIR:     {paths.output_data_dir}")

    print(f"Registros candidatos: {len(selection.prepared)}")
    print(f"Seleccion final (TOP_K): {len(selection.selected)}")
    print(f"Distribucion candidatos: {selection.dist_prepared}")
    print(f"Distribucion seleccion:  {selection.dist_selected}")
    if selection.skipped_by_reason:
        print(f"Descartes por motivo:    {selection.skipped_by_reason}")

    print(f"Export completado. Cases: {len(export_result.cases)}")
    print(f"JSON: {export_result.output_cases_json.resolve()}")
    print(f"CSV:  {export_result.output_export_csv.resolve()}")
    print(f"IMG:  {export_result.output_images_dir.resolve()}")
    print(f"Modo export: {export_result.mode}")
    print(f"Scenario UID: {export_result.scenario_uid}")
    print(f"Casos nuevos en corrida: {export_result.new_cases_count}")
    if export_result.mode == "append":
        print(f"Casos existentes leidos: {export_result.existing_cases_count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
