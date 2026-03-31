"""Experimento A/B para comparar descripcion zero-shot vs asistida con Ground Truth."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.inference.schemas import PolypVisualAnalysis
from src.inference.vlm_runner import VLMLoader


VALID_CLASSES: tuple[str, ...] = ("AD", "HP", "ASS")


def _diagnosis_label_text(cls_label: str) -> str:
    """Devuelve etiqueta expandida de clase para prompts asistidos."""
    normalized = str(cls_label or "").strip().upper()
    names = {
        "AD": "Adenoma (AD)",
        "HP": "Polipo Hiperplasico (HP)",
        "ASS": "Adenoma Serrado Sesil (ASS)",
    }
    return names.get(normalized, f"{normalized} ({normalized})")


def build_prompt_a() -> str:
    """Prompt zero-shot neutral (sin inyeccion diagnostica)."""
    return (
        "Analiza esta imagen endoscopica y clasifica la lesion. "
        "Primero describe la morfologia y bordes; despues describe superficie, color y patron vascular; "
        "finalmente justifica clinicamente y cierra en AD, HP o ASS."
    )


def build_prompt_b(confirmed_cls: str) -> str:
    """Prompt asistido por diagnostico confirmado en biopsia."""
    diagnosis = _diagnosis_label_text(confirmed_cls)
    return (
        "El sistema de biopsia ha confirmado que la lesion en esta imagen es un "
        f"{diagnosis}. Sabiendo esto como un hecho irrefutable, completa el analisis visual "
        "y justifica por que las caracteristicas de la imagen coinciden con este diagnostico."
    )


@dataclass(frozen=True)
class ABInputSample:
    """Entrada seleccionada para el experimento A/B."""

    image_path: Path
    image_name: str
    ground_truth_cls: str
    predicted_cls: str
    previous_status: str


@dataclass(frozen=True)
class ModelVariant:
    """Describe una variante de inferencia detectada en JSONL previo."""

    model_id: str
    include_reasoning: bool | None
    schema_name: str | None


@dataclass(frozen=True)
class ABVariantPlan:
    """Plan de ejecucion para una variante concreta."""

    variant: ModelVariant
    samples: list[ABInputSample]


@dataclass(frozen=True)
class ABResult:
    """Resultado final por imagen con comparativa A/B."""

    sample: ABInputSample
    description_a: dict[str, str]
    description_b: dict[str, str]


@dataclass(frozen=True)
class ABVariantRunResult:
    """Resultados A/B para una variante concreta."""

    variant: ModelVariant
    items: list[ABResult]


@dataclass(frozen=True)
class ABResumeSummaryRow:
    """Resumen de reanudacion por variante."""

    variant: ModelVariant
    total_samples: int
    reusable_samples: int
    pending_samples: int
    invalid_samples: int


@dataclass(frozen=True)
class LegacyRunContext:
    """Compatibilidad con wrapper previo."""

    model_id: str
    schema_name: str | None
    include_reasoning: bool | None


def _normalize_class_label(value: Any) -> str:
    """Normaliza etiquetas de clase para comparaciones robustas."""
    text = str(value or "").strip().upper()
    if text.startswith("AD") or "ADENOMA" in text:
        return "AD"
    if text.startswith("HP") or "HIPER" in text:
        return "HP"
    if text.startswith("ASS") or "SERR" in text:
        return "ASS"
    if text.startswith("UNKNOWN"):
        return "UNKNOWN"
    return text


def _extract_final_diagnosis_class(record: dict[str, Any]) -> str:
    """Extrae clase final desde payload o campos planos de respaldo."""
    payload = record.get("payload")
    if isinstance(payload, dict):
        candidate = (
            payload.get("final_diagnosis_class")
            or payload.get("final_diagnosis")
            or payload.get("diagnosis")
            or payload.get("class")
            or payload.get("label")
        )
        normalized = _normalize_class_label(candidate)
        if normalized:
            return normalized

    for key in (
        "final_diagnosis_class",
        "final_diagnosis",
        "diagnosis",
        "class",
        "label",
    ):
        normalized = _normalize_class_label(record.get(key))
        if normalized:
            return normalized

    return ""


def _extract_ground_truth_class(record: dict[str, Any]) -> str:
    """Extrae clase real desde campos conocidos o metadata anidada."""
    for key in ("ground_truth_cls", "ground_truth", "gt_cls", "cls", "label"):
        normalized = _normalize_class_label(record.get(key))
        if normalized:
            return normalized

    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        for key in ("ground_truth_cls", "ground_truth", "gt_cls", "cls"):
            normalized = _normalize_class_label(metadata.get(key))
            if normalized:
                return normalized

    return ""


def _variant_sort_key(variant: ModelVariant) -> tuple[str, int, str]:
    """Clave estable para ordenar variantes detectadas."""
    reasoning_rank = 2
    if variant.include_reasoning is True:
        reasoning_rank = 1
    elif variant.include_reasoning is False:
        reasoning_rank = 0
    return (
        variant.model_id.lower(),
        reasoning_rank,
        str(variant.schema_name or "").lower(),
    )


def _reasoning_label(value: bool | None) -> str:
    """Etiqueta legible del modo de razonamiento."""
    if value is True:
        return "con razonamiento"
    if value is False:
        return "sin razonamiento"
    return "desconocido"


def _autodetect_results_file(results_dir: Path) -> Path:
    """Detecta el JSONL de batch mas reciente para PolypClassification."""
    fallback = sorted(
        results_dir.glob("batch_*.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not fallback:
        raise FileNotFoundError(
            f"No se encontraron archivos batch JSONL en: {results_dir}"
        )

    def _is_manifest_like(path: Path) -> bool:
        name = path.name.lower()
        return name.startswith("batch_manifest") or "_manifest_" in name

    preferred = [
        path
        for path in fallback
        if "polypclassification" in path.name.lower() and not _is_manifest_like(path)
    ]
    if not preferred:
        preferred = [path for path in fallback if not _is_manifest_like(path)]
    if not preferred:
        preferred = list(fallback)

    # Elige el archivo más reciente que contenga candidatos válidos AD/HP/ASS para A/B.
    for candidate in preferred:
        try:
            records = _iter_data_records(candidate)
            grouped = _build_variant_candidates(records, override_model=None)
            if grouped:
                return candidate
        except Exception:
            continue

    # Si ninguno contiene candidatos válidos, conserva el comportamiento estable.
    return preferred[0]


def detect_results_file(results_file: Path | None, results_dir: Path) -> Path:
    """Resuelve ruta de resultados previos con autodeteccion opcional."""
    return results_file if results_file is not None else _autodetect_results_file(results_dir)


def _display_path(path: Path) -> str:
    """Devuelve una ruta compacta para mostrar en consola."""
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return str(path)


def _iter_data_records(results_file: Path) -> list[dict[str, Any]]:
    """Carga un JSONL y devuelve solo registros de inferencia (sin metadatos)."""
    records: list[dict[str, Any]] = []
    with results_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            if any(key.startswith("__") and key.endswith("__") for key in record):
                continue
            records.append(record)
    return records


def _resolve_variant_from_record(record: dict[str, Any]) -> ModelVariant | None:
    """Resuelve datos de variante (modelo + reasoning + schema) desde un registro."""
    model_id = str(record.get("model_id") or "").strip()
    if not model_id:
        return None
    schema_name_raw = str(record.get("schema_name") or "").strip()
    schema_name = schema_name_raw if schema_name_raw else None
    include_reasoning: bool | None = None
    if "include_reasoning" in record:
        include_reasoning = bool(record.get("include_reasoning"))
    return ModelVariant(
        model_id=model_id,
        include_reasoning=include_reasoning,
        schema_name=schema_name,
    )


def _build_variant_candidates(
    records: list[dict[str, Any]],
    *,
    override_model: str | None,
) -> dict[ModelVariant, list[ABInputSample]]:
    """Agrupa candidatos AD/HP/ASS por variante detectada."""
    grouped: dict[ModelVariant, list[ABInputSample]] = {}

    for record in records:
        status = str(record.get("status") or "").strip().lower()
        if status != "ok":
            continue

        variant = _resolve_variant_from_record(record)
        if variant is None:
            continue

        if override_model and variant.model_id != override_model.strip():
            continue

        ground_truth = _extract_ground_truth_class(record)
        if ground_truth not in VALID_CLASSES:
            continue

        predicted = _extract_final_diagnosis_class(record)
        if not predicted:
            continue

        image_path_value = str(record.get("image_path") or "").strip()
        if not image_path_value:
            continue

        image_path = Path(image_path_value)
        if not image_path.exists():
            continue

        previous_status = "Acierto" if predicted == ground_truth else "Fallo"
        sample = ABInputSample(
            image_path=image_path,
            image_name=image_path.name,
            ground_truth_cls=ground_truth,
            predicted_cls=predicted,
            previous_status=previous_status,
        )
        grouped.setdefault(variant, []).append(sample)

    return grouped


def _select_samples(
    candidates: list[ABInputSample],
    *,
    n_correct: int,
    n_incorrect: int,
    seed: int,
) -> list[ABInputSample]:
    """Selecciona aciertos/fallos balanceados por clase usando el mínimo disponible."""
    unique_by_path: dict[str, ABInputSample] = {}
    for item in candidates:
        unique_by_path[str(item.image_path)] = item
    deduped = list(unique_by_path.values())

    grouped: dict[str, dict[str, list[ABInputSample]]] = {
        cls: {"Acierto": [], "Fallo": []} for cls in VALID_CLASSES
    }
    for item in deduped:
        if item.ground_truth_cls not in grouped:
            continue
        if item.previous_status not in ("Acierto", "Fallo"):
            continue
        grouped[item.ground_truth_cls][item.previous_status].append(item)

    for cls in VALID_CLASSES:
        grouped[cls]["Acierto"].sort(key=lambda item: str(item.image_path).lower())
        grouped[cls]["Fallo"].sort(key=lambda item: str(item.image_path).lower())

    rng = random.Random(seed)
    selected: list[ABInputSample] = []
    for cls in VALID_CLASSES:
        available_correct = len(grouped[cls]["Acierto"])
        available_incorrect = len(grouped[cls]["Fallo"])
        per_class_quota = min(available_correct, available_incorrect, n_correct, n_incorrect)
        if per_class_quota <= 0:
            continue
        selected.extend(rng.sample(grouped[cls]["Acierto"], per_class_quota))
        selected.extend(rng.sample(grouped[cls]["Fallo"], per_class_quota))

    if not selected:
        raise RuntimeError(
            "No hay suficientes pares balanceados (acierto/fallo) en AD/HP/ASS para el experimento A/B."
        )

    rng.shuffle(selected)
    return selected


def build_execution_plans(
    *,
    results_file: Path,
    n_correct: int,
    n_incorrect: int,
    seed: int,
    override_model: str | None = None,
) -> list[ABVariantPlan]:
    """Construye planes por variante detectada en JSONL (balanceado por clase)."""
    records = _iter_data_records(results_file)
    grouped = _build_variant_candidates(
        records,
        override_model=override_model,
    )

    if not grouped:
        model_hint = f" para el modelo '{override_model}'" if override_model else ""
        raise RuntimeError(
            "No se detectaron candidatos validos (AD/HP/ASS)"
            f"{model_hint} en {results_file}."
        )

    plans: list[ABVariantPlan] = []
    for variant in sorted(grouped.keys(), key=_variant_sort_key):
        selected = _select_samples(
            grouped[variant],
            n_correct=n_correct,
            n_incorrect=n_incorrect,
            seed=seed,
        )
        plans.append(ABVariantPlan(variant=variant, samples=selected))

    return plans


def _extract_previous_run_context(
    results_file: Path,
    *,
    override_model: str | None = None,
) -> LegacyRunContext:
    """Compatibilidad para wrappers existentes: usa la primera variante del plan."""
    plans = build_execution_plans(
        results_file=results_file,
        n_correct=1,
        n_incorrect=1,
        seed=0,
        override_model=override_model,
    )
    first = plans[0].variant
    return LegacyRunContext(
        model_id=first.model_id,
        schema_name=first.schema_name,
        include_reasoning=first.include_reasoning,
    )


def _safe_text(value: Any) -> str:
    """Sanitiza texto para celdas Markdown."""
    text = str(value or "").strip()
    if not text:
        return "-"
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("|", "\\|")
    text = text.replace("\n", "<br>")
    return text


def default_checkpoint_path(output_md: Path) -> Path:
    """Ruta del checkpoint incremental asociado al markdown final."""
    base = Path(output_md)
    return base.with_suffix(base.suffix + ".checkpoint.json")


def _variant_checkpoint_key(variant: ModelVariant) -> str:
    """Clave estable para guardar/leer checkpoints por variante."""
    reasoning_token = "unknown"
    if variant.include_reasoning is True:
        reasoning_token = "true"
    elif variant.include_reasoning is False:
        reasoning_token = "false"
    schema_token = str(variant.schema_name or "")
    return f"{variant.model_id}::{reasoning_token}::{schema_token}"


def _sample_checkpoint_key(sample: ABInputSample) -> str:
    """Clave estable por muestra para poder retomar sin duplicados."""
    try:
        return str(sample.image_path.resolve())
    except Exception:
        return str(sample.image_path)


def _empty_checkpoint_payload() -> dict[str, Any]:
    """Estructura base de checkpoint."""
    return {
        "version": 2,
        "variants": {},
    }


def _load_checkpoint(path: Path) -> dict[str, Any]:
    """Carga checkpoint tolerando archivos ausentes o corruptos."""
    if not path.exists():
        return _empty_checkpoint_payload()

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _empty_checkpoint_payload()

    if not isinstance(payload, dict):
        return _empty_checkpoint_payload()

    if int(payload.get("version") or 0) != 2:
        return _empty_checkpoint_payload()

    variants = payload.get("variants")
    if not isinstance(variants, dict):
        payload["variants"] = {}
    payload.setdefault("version", 2)
    return payload


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Guarda checkpoint en disco."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _is_description_complete(description: dict[str, str] | Any) -> bool:
    """Valida que una descripcion no este vacia ni en estado de error."""
    if not isinstance(description, dict):
        return False
    required = (
        "morphology_and_borders",
        "surface_and_vascular_pattern",
        "clinical_justification",
        "final_diagnosis",
    )
    for field in required:
        value = str(description.get(field) or "").strip()
        if not value:
            return False
        if value.upper().startswith("ERROR:"):
            return False
    return True


def _is_result_complete(result: ABResult) -> bool:
    """Valida que un resultado A/B es reutilizable para reanudacion."""
    return _is_description_complete(result.description_a) and _is_description_complete(result.description_b)


def _serialize_result(result: ABResult) -> dict[str, Any]:
    """Serializa ABResult para checkpoint."""
    return {
        "sample": {
            "image_path": str(result.sample.image_path),
            "image_name": result.sample.image_name,
            "ground_truth_cls": result.sample.ground_truth_cls,
            "predicted_cls": result.sample.predicted_cls,
            "previous_status": result.sample.previous_status,
        },
        "description_a": dict(result.description_a),
        "description_b": dict(result.description_b),
    }


def _deserialize_result(payload: dict[str, Any], expected_sample: ABInputSample) -> ABResult | None:
    """Reconstruye ABResult validando estructura y coherencia de muestra."""
    if not isinstance(payload, dict):
        return None
    sample_payload = payload.get("sample")
    if not isinstance(sample_payload, dict):
        return None

    path_value = str(sample_payload.get("image_path") or "").strip()
    if not path_value:
        return None

    try:
        sample_path = Path(path_value)
    except Exception:
        return None

    expected_key = _sample_checkpoint_key(expected_sample)
    candidate_key = _sample_checkpoint_key(
        ABInputSample(
            image_path=sample_path,
            image_name=str(sample_payload.get("image_name") or sample_path.name),
            ground_truth_cls=str(sample_payload.get("ground_truth_cls") or ""),
            predicted_cls=str(sample_payload.get("predicted_cls") or ""),
            previous_status=str(sample_payload.get("previous_status") or ""),
        )
    )
    if candidate_key != expected_key:
        return None

    description_a = payload.get("description_a")
    description_b = payload.get("description_b")
    if not isinstance(description_a, dict) or not isinstance(description_b, dict):
        return None

    return ABResult(
        sample=expected_sample,
        description_a={
            "morphology_and_borders": str(description_a.get("morphology_and_borders") or ""),
            "surface_and_vascular_pattern": str(description_a.get("surface_and_vascular_pattern") or ""),
            "clinical_justification": str(description_a.get("clinical_justification") or ""),
            "final_diagnosis": str(description_a.get("final_diagnosis") or ""),
        },
        description_b={
            "morphology_and_borders": str(description_b.get("morphology_and_borders") or ""),
            "surface_and_vascular_pattern": str(description_b.get("surface_and_vascular_pattern") or ""),
            "clinical_justification": str(description_b.get("clinical_justification") or ""),
            "final_diagnosis": str(description_b.get("final_diagnosis") or ""),
        },
    )


def _get_variant_bucket(checkpoint_payload: dict[str, Any], variant: ModelVariant) -> dict[str, Any]:
    """Obtiene/crea bucket de checkpoint para una variante."""
    variants = checkpoint_payload.setdefault("variants", {})
    if not isinstance(variants, dict):
        checkpoint_payload["variants"] = {}
        variants = checkpoint_payload["variants"]

    key = _variant_checkpoint_key(variant)
    bucket = variants.get(key)
    if not isinstance(bucket, dict):
        bucket = {}
        variants[key] = bucket

    bucket["variant"] = {
        "model_id": variant.model_id,
        "include_reasoning": variant.include_reasoning,
        "schema_name": variant.schema_name,
    }
    if not isinstance(bucket.get("items"), dict):
        bucket["items"] = {}
    return bucket


def _get_reusable_results(
    *,
    checkpoint_payload: dict[str, Any],
    variant: ModelVariant,
    samples: list[ABInputSample],
) -> tuple[dict[str, ABResult], int]:
    """Recupera resultados validos del checkpoint y limpia entradas corruptas."""
    bucket = _get_variant_bucket(checkpoint_payload, variant)
    items_payload = bucket.get("items")
    if not isinstance(items_payload, dict):
        bucket["items"] = {}
        return {}, 0

    reusable: dict[str, ABResult] = {}
    invalid_count = 0

    samples_by_key = {_sample_checkpoint_key(sample): sample for sample in samples}
    for sample_key, sample in samples_by_key.items():
        raw = items_payload.get(sample_key)
        if raw is None:
            continue

        result = _deserialize_result(raw, sample)
        if result is None or not _is_result_complete(result):
            invalid_count += 1
            items_payload.pop(sample_key, None)
            continue
        reusable[sample_key] = result

    # Limpieza de entradas huérfanas (muestras fuera del plan actual)
    for existing_key in list(items_payload.keys()):
        if existing_key not in samples_by_key:
            items_payload.pop(existing_key, None)

    return reusable, invalid_count


def build_resume_summary(
    *,
    plans: list[ABVariantPlan],
    checkpoint_file: Path,
) -> list[ABResumeSummaryRow]:
    """Construye resumen de progreso reutilizable por la TUI."""
    payload = _load_checkpoint(checkpoint_file)
    rows: list[ABResumeSummaryRow] = []
    for plan in plans:
        reusable, invalid_count = _get_reusable_results(
            checkpoint_payload=payload,
            variant=plan.variant,
            samples=plan.samples,
        )
        total = len(plan.samples)
        reusable_count = len(reusable)
        pending = max(0, total - reusable_count)
        rows.append(
            ABResumeSummaryRow(
                variant=plan.variant,
                total_samples=total,
                reusable_samples=reusable_count,
                pending_samples=pending,
                invalid_samples=invalid_count,
            )
        )
    return rows


def _run_single_inference(
    *,
    loader: VLMLoader,
    image_path: Path,
    prompt: str,
    temperature: float,
    seed: int | None,
) -> dict[str, str]:
    """Ejecuta una inferencia y devuelve dict serializable para markdown."""
    result = loader.inference(
        image_path=image_path,
        prompt=prompt,
        schema=PolypVisualAnalysis,
        temperature=temperature,
        seed=seed,
    )
    payload = result.model_dump()
    return {
        "morphology_and_borders": str(payload.get("morphology_and_borders") or ""),
        "surface_and_vascular_pattern": str(payload.get("surface_and_vascular_pattern") or ""),
        "clinical_justification": str(payload.get("clinical_justification") or ""),
        "final_diagnosis": str(payload.get("final_diagnosis") or ""),
    }


def _run_ab_inference(
    *,
    variant: ModelVariant,
    model_id: str,
    samples: list[ABInputSample],
    temperature: float,
    seed: int | None,
    verbose: bool,
    server_api_host: str | None,
    api_token: str | None,
    checkpoint_payload: dict[str, Any],
    checkpoint_path: Path,
    force_recompute: bool,
) -> list[ABResult]:
    """Ejecuta ambos prompts por muestra y asegura descarga de VRAM al finalizar."""
    results: list[ABResult] = []
    bucket = _get_variant_bucket(checkpoint_payload, variant)
    items_payload = bucket.get("items")
    if not isinstance(items_payload, dict):
        bucket["items"] = {}
        items_payload = bucket["items"]

    if force_recompute:
        reusable_by_key: dict[str, ABResult] = {}
        invalid_count = 0
        if items_payload:
            bucket["items"] = {}
            items_payload = bucket["items"]
            _save_checkpoint(checkpoint_path, checkpoint_payload)
    else:
        reusable_by_key, invalid_count = _get_reusable_results(
            checkpoint_payload=checkpoint_payload,
            variant=variant,
            samples=samples,
        )
        if invalid_count > 0:
            print(f"  - Checkpoint invalido detectado en {invalid_count} muestra(s); se recalcularan.")

    pending = sum(1 for sample in samples if _sample_checkpoint_key(sample) not in reusable_by_key)
    print(f"  - Reutilizables desde checkpoint: {len(reusable_by_key)}/{len(samples)} | pendientes: {pending}")

    if pending <= 0:
        ordered = [reusable_by_key[_sample_checkpoint_key(sample)] for sample in samples]
        return ordered

    loader = VLMLoader(
        model_path=model_id,
        verbose=verbose,
        server_api_host=server_api_host,
        api_token=api_token,
    )

    try:
        loader.load_model()
        for index, sample in enumerate(samples, start=1):
            sample_key = _sample_checkpoint_key(sample)
            cached = reusable_by_key.get(sample_key)
            if cached is not None:
                results.append(cached)
                print(
                    f"  [{index:>2}/{len(samples)}] {sample.image_name} "
                    f"| estado previo: {sample.previous_status} | origen: checkpoint"
                )
                continue

            print(
                f"  [{index:>2}/{len(samples)}] {sample.image_name} "
                f"| estado previo: {sample.previous_status} | origen: inferencia"
            )

            description_a: dict[str, str]
            description_b: dict[str, str]

            try:
                description_a = _run_single_inference(
                    loader=loader,
                    image_path=sample.image_path,
                    prompt=build_prompt_a(),
                    temperature=temperature,
                    seed=seed,
                )
            except Exception as error:
                description_a = {
                    "morphology_and_borders": "",
                    "surface_and_vascular_pattern": "",
                    "clinical_justification": f"ERROR: {error}",
                    "final_diagnosis": "",
                }

            try:
                description_b = _run_single_inference(
                    loader=loader,
                    image_path=sample.image_path,
                    prompt=build_prompt_b(sample.ground_truth_cls),
                    temperature=temperature,
                    seed=seed,
                )
            except Exception as error:
                description_b = {
                    "morphology_and_borders": "",
                    "surface_and_vascular_pattern": "",
                    "clinical_justification": f"ERROR: {error}",
                    "final_diagnosis": "",
                }

            result_item = ABResult(
                sample=sample,
                description_a=description_a,
                description_b=description_b,
            )
            results.append(result_item)
            items_payload[sample_key] = _serialize_result(result_item)
            _save_checkpoint(checkpoint_path, checkpoint_payload)
    finally:
        loader.unload_model()

    return results


def _render_markdown(
    *,
    runs: list[ABVariantRunResult],
    seed: int,
    temperature: float,
    results_file: Path,
) -> str:
    """Construye reporte markdown comparativo del experimento A/B."""
    lines: list[str] = []
    lines.append("# Results A/B Experiment")
    lines.append("")
    lines.append("## Configuracion")
    lines.append("")
    lines.append("| Campo | Valor |")
    lines.append("|---|---|")
    lines.append(f"| Registros de origen | `{results_file}` |")
    lines.append(f"| Semilla de seleccion | `{seed}` |")
    lines.append(f"| Temperatura | `{temperature}` |")
    lines.append(f"| Variantes evaluadas | `{len(runs)}` |")
    lines.append(f"| Total de imagenes AD evaluadas | `{sum(len(run.items) for run in runs)}` |")
    lines.append("")
    lines.append("## Resumen de Variantes")
    lines.append("")
    lines.append("| Modelo | Modo origen | Esquema origen | Casos evaluados |")
    lines.append("|---|---|---|---|")
    for run in runs:
        variant = run.variant
        lines.append(
            "| "
            f"{_safe_text(variant.model_id)} | "
            f"{_safe_text(_reasoning_label(variant.include_reasoning))} | "
            f"{_safe_text(variant.schema_name or 'N/A')} | "
            f"{len(run.items)} |"
        )
    lines.append("")
    lines.append("## Hipotesis")
    lines.append("")
    lines.append(
        "Comparar si la inyeccion del Ground Truth confirmado (Prompt B) cambia cualitativamente "
        "el analisis visual frente al enfoque zero-shot (Prompt A)."
    )
    lines.append("")

    for run in runs:
        variant = run.variant
        lines.append(f"## Variante: {variant.model_id}")
        lines.append("")
        lines.append(f"- Modo origen: **{_reasoning_label(variant.include_reasoning)}**")
        if variant.schema_name:
            lines.append(f"- Esquema origen: **{variant.schema_name}**")
        lines.append(f"- Casos evaluados: **{len(run.items)}**")
        lines.append("")

        for index, item in enumerate(run.items, start=1):
            sample = item.sample
            lines.append(f"### Caso {index}: {sample.image_name}")
            lines.append("")
            lines.append(f"- Estado previo: **{sample.previous_status}**")
            lines.append(f"- Ground Truth: **{sample.ground_truth_cls}**")
            lines.append(f"- Prediccion previa: **{sample.predicted_cls}**")
            lines.append(f"- Imagen: `{sample.image_path}`")
            lines.append("")
            lines.append("| Campo | Descripcion A (Zero-Shot) | Descripcion B (Asistido GT) |")
            lines.append("|---|---|---|")
            lines.append(
                f"| Morfologia y bordes | {_safe_text(item.description_a.get('morphology_and_borders'))} | "
                f"{_safe_text(item.description_b.get('morphology_and_borders'))} |"
            )
            lines.append(
                f"| Superficie y patron vascular | {_safe_text(item.description_a.get('surface_and_vascular_pattern'))} | "
                f"{_safe_text(item.description_b.get('surface_and_vascular_pattern'))} |"
            )
            lines.append(
                f"| Justificacion clinica | {_safe_text(item.description_a.get('clinical_justification'))} | "
                f"{_safe_text(item.description_b.get('clinical_justification'))} |"
            )
            lines.append(
                f"| Diagnostico final | {_safe_text(item.description_a.get('final_diagnosis'))} | "
                f"{_safe_text(item.description_b.get('final_diagnosis'))} |"
            )
            lines.append("")

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Crea parser de argumentos CLI para experimento A/B."""
    parser = argparse.ArgumentParser(
        description="Ejecuta experimento A/B: zero-shot vs prompt asistido por Ground Truth."
    )
    parser.add_argument(
        "--model",
        required=False,
        default=None,
        help="ID/tag del modelo en LM Studio (opcional, filtra variantes detectadas)",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=None,
        help="Ruta JSONL de resultados previos (si no se indica, se autodetecta)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("data/processed/batch_results"),
        help="Directorio para autodeteccion de JSONL previos",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("data/processed/ab_results/results_ab_experiment.md"),
        help="Ruta del Markdown de salida",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla reproducible")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Temperatura de inferencia para ambos prompts",
    )
    parser.add_argument(
        "--n-correct",
        type=int,
        default=5,
        help="Maximo de aciertos por clase; se ajusta automaticamente al minimo disponible",
    )
    parser.add_argument(
        "--n-incorrect",
        type=int,
        default=5,
        help="Maximo de fallos por clase; se ajusta automaticamente al minimo disponible",
    )
    parser.add_argument(
        "--server-api-host",
        default=None,
        help="Host opcional de API de LM Studio",
    )
    parser.add_argument(
        "--api-token",
        default=None,
        help="Token opcional de API de LM Studio",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Activa logs verbosos de VLMLoader",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignora checkpoint y recalcula todas las muestras del plan actual",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada CLI."""
    args = build_parser().parse_args(argv)
    output_path: Path = args.output_md
    checkpoint_path = default_checkpoint_path(output_path)

    results_file = detect_results_file(args.results_file, args.results_dir)
    print("\n=== Ejecucion experimento A/B ===")
    print(f"Resultados previos: {_display_path(results_file)}")
    print(f"Checkpoint incremental: {_display_path(checkpoint_path)}")

    plans = build_execution_plans(
        results_file=results_file,
        n_correct=args.n_correct,
        n_incorrect=args.n_incorrect,
        seed=args.seed,
        override_model=args.model,
    )

    print(f"Variantes a ejecutar: {len(plans)}")
    for variant_index, plan in enumerate(plans, start=1):
        print(
            f"  {variant_index}. model={plan.variant.model_id} "
            f"| reasoning={_reasoning_label(plan.variant.include_reasoning)} "
            f"| schema={plan.variant.schema_name or 'N/A'} "
            f"| muestras={len(plan.samples)}"
        )

    checkpoint_payload = _load_checkpoint(checkpoint_path)
    resume_rows = build_resume_summary(plans=plans, checkpoint_file=checkpoint_path)
    if resume_rows:
        print("Resumen de reanudacion:")
        for row in resume_rows:
            print(
                "  - "
                f"{row.variant.model_id} | "
                f"reutilizables={row.reusable_samples}/{row.total_samples} | "
                f"pendientes={row.pending_samples} | "
                f"invalidos={row.invalid_samples}"
            )

    runs: list[ABVariantRunResult] = []
    for variant_index, plan in enumerate(plans, start=1):
        variant = plan.variant
        print()
        print(
            f"[Variante {variant_index}/{len(plans)}] "
            f"model={variant.model_id} | "
            f"reasoning={_reasoning_label(variant.include_reasoning)} | "
            f"schema={variant.schema_name or 'N/A'}"
        )
        items = _run_ab_inference(
            variant=variant,
            model_id=variant.model_id,
            samples=plan.samples,
            temperature=args.temperature,
            seed=args.seed,
            verbose=args.verbose,
            server_api_host=args.server_api_host,
            api_token=args.api_token,
            checkpoint_payload=checkpoint_payload,
            checkpoint_path=checkpoint_path,
            force_recompute=bool(args.force_recompute),
        )
        runs.append(ABVariantRunResult(variant=variant, items=items))

    markdown = _render_markdown(
        runs=runs,
        seed=args.seed,
        temperature=args.temperature,
        results_file=results_file,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")

    print()
    print(f"Reporte generado: {_display_path(output_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
